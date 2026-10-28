"""Type-observable bounded merge: a slow lead bounds the target-lane gap.

The validated rear-key conflict is kept, but a slow vehicle is placed ahead in
the target lane.  The ego must merge into the gap between the key (behind) and
the lead (ahead); the key's type still controls whether the gap opens or
closes.  The observation window, prompt, controller and reward are unchanged.
This is a new scene module: the validated ``type_observable_merge`` code is
imported, not modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model.reasoning.worker import LLMReasoningWorker
from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import (LLM_VARIANTS, canonical_hash, load_config,
                                            missing_credentials, source_hashes)
from experiments.ablation_v2.llm_targeted_pilot import git_commit, llm_diagnostics
from experiments.ablation_v2.reporting import report
from experiments.ablation_v2.type_observable_merge import apply_scene as base_apply_scene
from experiments.ablation_v2.type_observable_merge import cases as base_cases


def install_target_lead(env, case):
    relative_x = case.get('target_lead_x_m')
    if relative_x is None:
        return env
    raw = env.unwrapped
    by_id = {vehicle.id: vehicle for vehicle in raw.road.vehicles}
    if 2 not in by_id:
        return env
    lead = by_id[2]
    lead.position[0] = float(raw.vehicle.position[0]) + float(relative_x)
    lead.position[1] = 0.
    lead.speed = float(case['target_lead_speed_ms'])
    lead.target_speed = float(case['target_lead_speed_ms'])
    lead.on_state_update()
    return env


def cases(cfg):
    spec = cfg['observable_bounded_merge']
    for case in base_cases(cfg):
        yield dict(case,
                   target_lead_x_m=float(spec['target_lead_x_m']),
                   target_lead_speed_ms=float(spec['target_lead_speed_ms']))


def apply_scene(env, case):
    return install_target_lead(base_apply_scene(env, case), case)


def paired_analysis(out, cfg, rows):
    lookup = {(row['case_id'], row['variant']): row for row in rows}
    pairs = []
    comparisons = [name for name in cfg['variants'] if name != 'full_model']
    for comparison in comparisons:
        for case in cases(cfg):
            full = lookup[(case['case_id'], 'full_model')]
            other = lookup[(case['case_id'], comparison)]
            pairs.append(dict(
                case_id=case['case_id'], behaviour_index=case['behaviour_index'],
                comparison=comparison,
                full_status=full['status'], comparison_status=other['status'],
                success_difference=full['success'] - other['success'],
                collision_difference=full['collision'] - other['collision'],
                restricted_time_difference_s=(full['restricted_task_time_s']
                                              - other['restricted_task_time_s']),
                clearance_difference_m=(full['min_clearance_m'] - other['min_clearance_m'])
                if full['min_clearance_m'] is not None and other['min_clearance_m'] is not None else None,
                full_probe_steps=full['probe_selected_steps'],
                comparison_probe_steps=other['probe_selected_steps']))
    if 'oracle_type' in cfg['variants'] and 'uniform_belief' in cfg['variants']:
        for case in cases(cfg):
            oracle = lookup[(case['case_id'], 'oracle_type')]
            uniform = lookup[(case['case_id'], 'uniform_belief')]
            pairs.append(dict(
                case_id=case['case_id'], behaviour_index=case['behaviour_index'],
                comparison='oracle_vs_uniform',
                full_status=oracle['status'], comparison_status=uniform['status'],
                success_difference=oracle['success'] - uniform['success'],
                collision_difference=oracle['collision'] - uniform['collision'],
                restricted_time_difference_s=(oracle['restricted_task_time_s']
                                              - uniform['restricted_task_time_s']),
                clearance_difference_m=(oracle['min_clearance_m'] - uniform['min_clearance_m'])
                if oracle['min_clearance_m'] is not None and uniform['min_clearance_m'] is not None else None,
                full_probe_steps=oracle['probe_selected_steps'],
                comparison_probe_steps=uniform['probe_selected_steps']))
    runner.write_csv(out / 'observable_bounded_merge_pairs.csv', pairs)
    effects = {}
    for comparison in sorted({row['comparison'] for row in pairs}):
        part = [row for row in pairs if row['comparison'] == comparison]
        effects[comparison] = dict(
            better_success=sum(row['success_difference'] > 0 for row in part),
            worse_success=sum(row['success_difference'] < 0 for row in part),
            fewer_collisions=sum(row['collision_difference'] < 0 for row in part),
            more_collisions=sum(row['collision_difference'] > 0 for row in part),
            mean_restricted_time_difference_s=float(np.mean(
                [row['restricted_time_difference_s'] for row in part])),
            mean_probe_step_difference=float(np.mean(
                [row['full_probe_steps'] - row['comparison_probe_steps'] for row in part])))
    return pairs, effects


def action_divergence(out, cfg, rows):
    result = {}
    for left, right in (('full_model', 'uniform_belief'), ('oracle_type', 'uniform_belief'),
                        ('full_model', 'oracle_type')):
        if left not in cfg['variants'] or right not in cfg['variants']:
            continue
        total = divergent = 0
        for case in cases(cfg):
            left_steps = out / 'episodes' / f"{case['case_id']}-{left}" / 'steps.csv'
            right_steps = out / 'episodes' / f"{case['case_id']}-{right}" / 'steps.csv'
            if not left_steps.exists() or not right_steps.exists():
                continue
            import csv
            with left_steps.open(encoding='utf-8-sig') as handle:
                left_rows = list(csv.DictReader(handle))
            with right_steps.open(encoding='utf-8-sig') as handle:
                right_rows = list(csv.DictReader(handle))
            for a, b in zip(left_rows, right_rows):
                total += 1
                if a['action_type'] != b['action_type'] or a['target_lane'] != b['target_lane']:
                    divergent += 1
        result[f'{left}_vs_{right}'] = dict(steps=total, divergent_steps=divergent,
                                            divergent_fraction=divergent / total if total else None)
    return result


def analyze(out, cfg):
    rows = [json.loads(path.read_text(encoding='utf-8'))
            for path in sorted((out / 'episodes').glob('*/result.json'))]
    by_variant = {}
    for variant in cfg['variants']:
        part = [row for row in rows if row['variant'] == variant]
        by_variant[variant] = dict(
            episodes=len(part),
            completed=sum(row['status'] == 'completed' for row in part),
            collisions=sum(row['status'] == 'collision' for row in part),
            timeouts=sum(row['status'] == 'timeout' for row in part),
            route_end_failures=sum(row['status'] == 'offroad_or_route_end' for row in part),
            runtime_errors=sum(row['status'] == 'runtime_error' for row in part),
            llm_requests=sum(row['llm_requests'] for row in part),
            llm_updates=sum(row['llm_updates'] for row in part),
            mean_belief_fallback_fraction=float(np.mean(
                [row['belief_fallback_fraction'] for row in part])),
            probe_steps=sum(row['probe_selected_steps'] for row in part),
            mean_restricted_time_s=float(np.mean(
                [row['restricted_task_time_s'] for row in part])),
            mean_min_clearance_m=float(np.mean(
                [row['min_clearance_m'] for row in part
                 if row['min_clearance_m'] is not None])))
    pairs, effects = paired_analysis(out, cfg, rows)
    initial = defaultdict(set)
    for row in rows:
        initial[row['case_id']].add(row['initial_hash'])
    diagnostics = llm_diagnostics(out, rows, cfg)
    diagnostics['prompt_version'] = cfg['llm'].get('prompt_version', 'legacy')
    divergence = action_divergence(out, cfg, rows)
    llm_variants = [name for name in cfg['variants'] if name in LLM_VARIANTS]
    gates = dict(
        complete=len(rows) == len(cfg['seeds']) * len(cfg['variants']),
        paired=all(len(value) == 1 for value in initial.values()),
        no_runtime_errors=not any(row['status'] == 'runtime_error' for row in rows),
        llm_requested=all(by_variant[name]['llm_requests'] > 0 for name in llm_variants),
        llm_updated=all(by_variant[name]['llm_updates'] > 0 for name in llm_variants),
        response_errors_zero=diagnostics['response_errors'] == 0)
    analysis = dict(
        git_commit=git_commit(), protocol_hash=canonical_hash(cfg),
        phase=cfg.get('phase', 'type-observable bounded merge'),
        by_variant=by_variant, paired_effects=effects, action_divergence=divergence,
        llm_diagnostics=diagnostics, gates=gates, all_gates_pass=all(gates.values()),
        algorithm_change=False, scene_change_only=True,
        interpretation=('The validated rear-key conflict is bounded ahead by a slow '
                        'target-lane lead; the ego merges into the gap between them. '
                        'Not confirmatory evidence.'))
    runner.dump(out / 'observable_bounded_merge_analysis.json', analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ablation_v2_type_observable_bounded_merge_lead45.json')
    parser.add_argument('--output-dir',
                        default='simulation_results/ablation_v2/type_observable_bounded_merge_lead45_001')
    args = parser.parse_args()
    cfg, out = load_config(args.config), Path(args.output_dir)
    blocked = missing_credentials(cfg)
    if blocked:
        raise RuntimeError('Missing DASHSCOPE_API_KEY for ' + ','.join(blocked))
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = dict(config=cfg, git_commit=git_commit(), source_hashes=hashes,
                    code_hash=canonical_hash(hashes), protocol_hash=canonical_hash(cfg),
                    algorithm_change=False, scene_change_only=True,
                    llm_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1',
                    data_source='fresh seeded fictional simulator states; no pre-existing dataset')
    if (out / 'manifest.json').exists():
        old = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        if old['code_hash'] != manifest['code_hash'] or old['protocol_hash'] != manifest['protocol_hash']:
            raise RuntimeError('Changed code/protocol requires a new output directory')
    else:
        runner.dump(out / 'manifest.json', manifest)
        for relative in hashes:
            destination = out / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(relative, destination)
    case_rows = list(cases(cfg))
    runner.dump(out / 'cases.json', case_rows)
    original = runner.make_env
    warm_cfg = copy.deepcopy(cfg)
    warm_cfg['runtime']['max_steps'] = 1
    with patch.object(runner, 'make_env', lambda spec: apply_scene(original(spec), case_rows[0])):
        warm = runner.run_episode(warm_cfg, case_rows[0], 'uniform_belief', out / 'warmup',
                                  warmup=True)
    if warm['status'] == 'runtime_error':
        raise RuntimeError('Simulation warmup failed: ' + str(warm['error']))
    llm_client = None
    if set(cfg['variants']) & LLM_VARIANTS:
        probe = LLMReasoningWorker(model=cfg['llm']['model'], timeout_s=cfg['llm']['timeout_s'],
                                   max_retries=0,
                                   prompt_version=cfg['llm'].get('prompt_version', 'legacy'))
        probe.get_response('Return only this fictional JSON probability example: '
                           '{"0":[0.2,0.6,0.2]}', [0])
        runner.dump(out / 'llm_connection_warmup.json', dict(
            latency_ms=1000. * (probe.last_latency_s or 0.), error=probe.last_error,
            usage=probe.last_usage, response_model=probe.last_response_model,
            prompt_version=cfg['llm'].get('prompt_version', 'legacy')))
        if probe.last_error:
            raise RuntimeError('LLM connection warmup failed: ' + probe.last_error)
        llm_client = probe._client
    rows = []
    try:
        for case in case_rows:
            order = list(cfg['variants'])
            random.Random(case['seed']).shuffle(order)
            for variant in order:
                result_path = out / 'episodes' / f"{case['case_id']}-{variant}" / 'result.json'
                if result_path.exists() and cfg['runtime']['resume']:
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                else:
                    def make_env(spec, selected=copy.deepcopy(case)):
                        return apply_scene(original(spec), selected)
                    with patch.object(runner, 'make_env', make_env):
                        result = runner.run_episode(cfg, case, variant, out,
                                                    llm_client=llm_client)
                rows.append(result)
                runner.write_csv(out / 'episodes.csv', rows)
                print(f"[{len(rows)}/{len(case_rows)*len(cfg['variants'])}] "
                      f"{case['case_id']} {variant}: {result['status']} "
                      f"requests={result['llm_requests']} updates={result['llm_updates']}",
                      flush=True)
    finally:
        if llm_client:
            llm_client.close()
    runner.dump(out / 'completion.json', dict(
        planned=len(case_rows) * len(cfg['variants']), finished=len(rows),
        runtime_errors=sum(row['status'] == 'runtime_error' for row in rows)))
    report(out)
    print(json.dumps(analyze(out, cfg), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
