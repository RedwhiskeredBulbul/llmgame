"""Real-Qwen targeted closed-loop pilot with LLM-specific diagnostics."""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from LLMreasoning.llmreason import LLMReasoningWorker
from LLMreasoning.vehicle_type_evaluation import EvaluationRecord, calculate_metrics
from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import LLM_VARIANTS, canonical_hash, load_config, missing_credentials, source_hashes
from experiments.ablation_v2.reporting import report
from experiments.ablation_v2.targeted_interaction import apply_case, targeted_cases


LABELS = ('Cautious', 'Normal', 'Aggressive')


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def probability_label(probabilities):
    if not isinstance(probabilities, list) or len(probabilities) != 3:
        return None
    return LABELS[max(range(3), key=lambda index: probabilities[index])]


def paired_analysis(rows, cfg):
    lookup = {(row['case_id'], row['variant']): row for row in rows}
    pairs = []
    for comparison in cfg['variants']:
        if comparison == 'full_model':
            continue
        for case in targeted_cases(cfg):
            full, other = lookup[(case['case_id'], 'full_model')], lookup[(case['case_id'], comparison)]
            pairs.append(dict(case_id=case['case_id'], template_index=case['template_index'],
                behaviour_index=case['behaviour_index'], comparison=comparison,
                full_status=full['status'], comparison_status=other['status'],
                success_difference=full['success']-other['success'],
                collision_difference=full['collision']-other['collision'],
                restricted_time_difference_s=full['restricted_task_time_s']-other['restricted_task_time_s'],
                full_probe_steps=full['probe_selected_steps'], comparison_probe_steps=other['probe_selected_steps']))
    return pairs


def llm_diagnostics(out, rows, cfg):
    records, key_records, response_errors = [], [], 0
    response_events = after_episode_events = total_tokens = 0
    seen = set()
    for result in rows:
        if result['variant'] not in LLM_VARIANTS:
            continue
        episode_dir = out/'episodes'/result['episode_id']
        initial = json.loads((episode_dir/'initial_state.json').read_text(encoding='utf-8'))['initial_state']
        truth = {int(vehicle['id']): vehicle.get('behaviour') for vehicle in initial if vehicle.get('behaviour') in LABELS}
        events = json.loads((episode_dir/'belief_events.json').read_text(encoding='utf-8'))
        for event in events:
            response_events += event.get('event') == 'response'
            after_episode_events += event.get('event') == 'response_after_episode'
            response_errors += bool(event.get('error'))
            usage = event.get('usage') or {}
            total_tokens += int(usage.get('total_tokens') or 0)
            if event.get('event') != 'response':
                continue
            for raw_id, probabilities in (event.get('probabilities') or {}).items():
                vehicle_id = int(raw_id)
                key = (result['episode_id'], vehicle_id)
                if key in seen or vehicle_id not in truth:
                    continue
                seen.add(key)
                values = list(probabilities) if isinstance(probabilities, list) else None
                record = EvaluationRecord(case_id=result['episode_id'], vehicle_id=vehicle_id,
                    true_label=truth[vehicle_id], predicted_label=probability_label(values),
                    p_cautious=values[0] if values else None, p_normal=values[1] if values else None,
                    p_aggressive=values[2] if values else None,
                    latency_s=(event.get('latency_ms') or 0.)/1000., error=event.get('error'))
                records.append(record)
                if vehicle_id == 1:
                    key_records.append(record)
    all_metrics = calculate_metrics(records)
    key_metrics = calculate_metrics(key_records)
    result = dict(model=cfg['llm']['model'], first_in_episode_vehicle_predictions=all_metrics,
        first_key_vehicle_predictions=key_metrics, response_events=response_events,
        responses_after_episode=after_episode_events, response_errors=response_errors,
        recorded_total_tokens=total_tokens,
        note='First accepted in-episode prediction per episode/vehicle; repeated temporal predictions are not treated as independent samples.')
    runner.dump(out/'llm_diagnostics.json', result)
    return result


def analyze(out, cfg):
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((out/'episodes').glob('*/result.json'))]
    by_variant = {}
    for variant in cfg['variants']:
        part = [row for row in rows if row['variant'] == variant]
        by_variant[variant] = dict(episodes=len(part), completed=sum(row['status'] == 'completed' for row in part),
            collisions=sum(row['status'] == 'collision' for row in part),
            route_end_failures=sum(row['status'] == 'offroad_or_route_end' for row in part),
            runtime_errors=sum(row['status'] == 'runtime_error' for row in part),
            llm_requests=sum(row['llm_requests'] for row in part), llm_updates=sum(row['llm_updates'] for row in part),
            mean_belief_fallback_fraction=float(np.mean([row['belief_fallback_fraction'] for row in part])),
            probe_steps=sum(row['probe_selected_steps'] for row in part),
            mean_restricted_time_s=float(np.mean([row['restricted_task_time_s'] for row in part])))
    pairs = paired_analysis(rows, cfg)
    runner.write_csv(out/'llm_targeted_pairs.csv', pairs)
    effects = {}
    for comparison in cfg['variants']:
        if comparison == 'full_model':
            continue
        part = [row for row in pairs if row['comparison'] == comparison]
        effects[comparison] = dict(full_better_success=sum(row['success_difference'] > 0 for row in part),
            full_worse_success=sum(row['success_difference'] < 0 for row in part),
            full_fewer_collisions=sum(row['collision_difference'] < 0 for row in part),
            full_more_collisions=sum(row['collision_difference'] > 0 for row in part),
            mean_time_difference_s=float(np.mean([row['restricted_time_difference_s'] for row in part])))
    initial = defaultdict(set)
    for row in rows:
        initial[row['case_id']].add(row['initial_hash'])
    diagnostics = llm_diagnostics(out, rows, cfg)
    llm_variants = [name for name in cfg['variants'] if name in LLM_VARIANTS]
    gates = dict(complete=len(rows) == len(cfg['seeds'])*len(cfg['variants']),
        paired=all(len(value) == 1 for value in initial.values()),
        no_runtime_errors=not any(row['status'] == 'runtime_error' for row in rows),
        llm_requested=all(by_variant[name]['llm_requests'] > 0 for name in llm_variants),
        llm_updated=all(by_variant[name]['llm_updates'] > 0 for name in llm_variants),
        response_errors_zero=diagnostics['response_errors'] == 0)
    result = dict(git_commit=git_commit(), protocol_hash=canonical_hash(cfg), by_variant=by_variant,
        paired_effects=effects, llm_diagnostics=diagnostics, gates=gates, all_gates_pass=all(gates.values()),
        algorithm_change=False, phase='LLM activation pilot; not confirmatory performance evidence')
    runner.dump(out/'llm_targeted_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ablation_v2_llm_targeted_pilot.json')
    parser.add_argument('--output-dir', default='simulation_results/ablation_v2/llm_targeted_pilot_001')
    args = parser.parse_args()
    cfg, out = load_config(args.config), Path(args.output_dir)
    blocked = missing_credentials(cfg)
    if blocked:
        raise RuntimeError('Missing DASHSCOPE_API_KEY for '+','.join(blocked))
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = dict(config=cfg, git_commit=git_commit(), source_hashes=hashes,
        code_hash=canonical_hash(hashes), protocol_hash=canonical_hash(cfg), algorithm_change=False,
        scene_change_only=True, llm_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1',
        data_source='fresh seeded fictional simulator states; no pre-existing dataset')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old['code_hash'] != manifest['code_hash'] or old['protocol_hash'] != manifest['protocol_hash']:
            raise RuntimeError('Changed code/protocol requires a new output directory')
    else:
        runner.dump(out/'manifest.json', manifest)
        for relative in hashes:
            destination = out/'source_snapshot'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(relative, destination)
    cases = list(targeted_cases(cfg))
    runner.dump(out/'cases.json', cases)
    warm_cfg = copy.deepcopy(cfg)
    warm_cfg['runtime']['max_steps'] = 1
    original = runner.make_env
    with patch.object(runner, 'make_env', lambda spec: apply_case(original(spec), spec)):
        warm = runner.run_episode(warm_cfg, cases[0], 'uniform_belief', out/'warmup', warmup=True)
    if warm['status'] == 'runtime_error':
        raise RuntimeError('Simulation warmup failed: '+str(warm['error']))
    probe = LLMReasoningWorker(model=cfg['llm']['model'], timeout_s=cfg['llm']['timeout_s'], max_retries=0)
    probe.get_response('Return only this fictional JSON probability example: {"0":[0.2,0.6,0.2]}', [0])
    runner.dump(out/'llm_connection_warmup.json', dict(latency_ms=1000.*(probe.last_latency_s or 0.),
        error=probe.last_error, usage=probe.last_usage, response_model=probe.last_response_model))
    if probe.last_error:
        raise RuntimeError('LLM connection warmup failed: '+probe.last_error)
    rows = []
    try:
        for case in cases:
            order = list(cfg['variants'])
            random.Random(case['seed']).shuffle(order)
            for variant in order:
                result_path = out/'episodes'/f"{case['case_id']}-{variant}"/'result.json'
                if result_path.exists() and cfg['runtime']['resume']:
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                else:
                    def make_env(spec, selected=copy.deepcopy(case)):
                        return apply_case(original(spec), selected)
                    with patch.object(runner, 'make_env', make_env):
                        result = runner.run_episode(cfg, case, variant, out, llm_client=probe._client)
                rows.append(result)
                runner.write_csv(out/'episodes.csv', rows)
                print(f"[{len(rows)}/{len(cases)*len(cfg['variants'])}] {case['case_id']} {variant}: "
                      f"{result['status']} requests={result['llm_requests']} updates={result['llm_updates']}", flush=True)
    finally:
        if probe._client:
            probe._client.close()
    runner.dump(out/'completion.json', dict(planned=len(cases)*len(cfg['variants']), finished=len(rows),
        runtime_errors=sum(row['status'] == 'runtime_error' for row in rows)))
    report(out)
    print(json.dumps(analyze(out, cfg), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
