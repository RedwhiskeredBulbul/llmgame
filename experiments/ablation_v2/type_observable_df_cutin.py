"""Type-belief cut-in experiment on the repository's CutinEnv environment.

The native ``CutinEnv`` populates one dense target lane with
``CautiousVehicle_df`` / ``NormalVehicle_df`` / ``AggressiveVehicle_df``
instances drawn at reset.  Every non-ego vehicle therefore already carries a
random type label.  The frozen v2 controller, however, has no mandatory lane
change in this open-road scene: keeping the empty ego lane is always cheaper.
This adapter keeps the native scene and its random labels, opens a local merge
window, and makes the cut-in mandatory for a fixed distance.  It does not add a
scripted yield/force response; the only type signal is the native behaviour
class.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest

import gymnasium as gym

import highway_env  # noqa: F401
from model.reasoning.worker import LLMReasoningWorker
from experiments.vehicle_types.evaluation import (
    BEHAVIOR_LABELS,
    EvaluationRecord,
    calculate_metrics,
)
from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import (LLM_VARIANTS, canonical_hash, load_config,
                                            missing_credentials, source_hashes)
from experiments.ablation_v2.llm_targeted_pilot import git_commit
from experiments.ablation_v2.reporting import report


def cases(cfg):
    spec = cfg['df_cutin']
    for seed in cfg['seeds']:
        yield dict(
            case_id=f'df-cutin-{seed}', scenario='df_2_2', seed=seed,
            difficulty=1, behaviour_index=0,
            required_target_lane=int(spec['required_target_lane']),
            required_progress_m=float(spec.get('required_progress_m', 0.0)),
            horizon_s=float(spec['horizon_s']),
            decision_open_x_m=float(spec['decision_open_x_m']),
            route_end_x_m=float(spec['route_end_x_m']),
            ego_speed_ms=(None if spec.get('ego_speed_ms') is None
                          else float(spec['ego_speed_ms'])),
            speed_limit_ms=(None if spec.get('speed_limit_ms') is None
                            else float(spec['speed_limit_ms'])),
            clear_radius_m=float(spec['clear_radius_m']),
            clear_ahead_m=(None if spec.get('clear_ahead_m') is None
                           else float(spec['clear_ahead_m'])),
            clear_behind_m=(None if spec.get('clear_behind_m') is None
                            else float(spec['clear_behind_m'])),
        )


def apply_scene(env, case):
    raw = env.unwrapped
    raw.kind = 'cutin'
    raw.required_target_lane = int(case['required_target_lane'])
    raw.required_progress_m = float(case['required_progress_m'])
    raw.decision_open_x = float(case['decision_open_x_m'])
    raw.route_end_x = float(case['route_end_x_m'])
    if case.get('ego_speed_ms') is not None:
        raw.vehicle.speed = float(case['ego_speed_ms'])
    raw.vehicle.on_state_update()
    if case.get('speed_limit_ms') is not None:
        raw.scene_speed_limit_ms = float(case['speed_limit_ms'])
    else:
        # Preserve the native seeded ego speed (30--40 km/h) and use it as the
        # controller speed limit; the target lane remains at 25 km/h.
        raw.scene_speed_limit_ms = float(raw.vehicle.speed)

    # Open a local cut-in window in the target lane.  The rest of the dense
    # stream and all of its random behaviour labels are left untouched.
    ego_x = float(raw.vehicle.position[0])
    radius = float(case['clear_radius_m'])
    ahead_limit = case.get('clear_ahead_m')
    behind_limit = case.get('clear_behind_m')
    for vehicle in list(raw.road.vehicles):
        if vehicle is raw.vehicle:
            continue
        if abs(float(vehicle.position[1])) >= 1.0:
            continue
        dx = float(vehicle.position[0]) - ego_x
        if ahead_limit is None and behind_limit is None:
            remove = abs(dx) < radius
        else:
            limit = ahead_limit if dx >= 0 else behind_limit
            remove = limit is not None and abs(dx) < float(limit)
        if remove:
            raw.road.vehicles.remove(vehicle)
    return env


def make_env(case):
    env = gym.make('CutinEnv')
    env.reset(seed=case['seed'])
    return apply_scene(env, case)


def mcnemar(left, right, lookup, case_rows, field='success'):
    b = c = 0
    for case in case_rows:
        a, o = lookup[(case['case_id'], left)], lookup[(case['case_id'], right)]
        if a[field] and not o[field]:
            b += 1
        elif o[field] and not a[field]:
            c += 1
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(better=b, worse=c, p=float(p))


def paired_analysis(rows, cfg, case_rows):
    lookup = {(row['case_id'], row['variant']): row for row in rows}
    comparisons = [name for name in cfg['variants'] if name != 'full_model']
    pairs = []
    for comparison in comparisons:
        for case in case_rows:
            full, other = lookup[(case['case_id'], 'full_model')], lookup[(case['case_id'], comparison)]
            pairs.append(dict(
                case_id=case['case_id'], comparison=comparison,
                full_status=full['status'], comparison_status=other['status'],
                success_difference=full['success'] - other['success'],
                collision_difference=full['collision'] - other['collision'],
                restricted_time_difference_s=full['restricted_task_time_s'] - other['restricted_task_time_s'],
                full_probe_steps=full['probe_selected_steps'],
                comparison_probe_steps=other['probe_selected_steps']))
    runner.write_csv(Path(cfg['_output_dir']) / 'df_cutin_pairs.csv', pairs)
    effects = {}
    for comparison in sorted({row['comparison'] for row in pairs}):
        part = [row for row in pairs if row['comparison'] == comparison]
        effects[comparison] = dict(
            better_success=sum(row['success_difference'] > 0 for row in part),
            worse_success=sum(row['success_difference'] < 0 for row in part),
            fewer_collisions=sum(row['collision_difference'] < 0 for row in part),
            more_collisions=sum(row['collision_difference'] > 0 for row in part),
            mean_restricted_time_difference_s=float(np.mean(
                [row['restricted_time_difference_s'] for row in part])))
    return pairs, effects


def action_divergence(out, cfg, case_rows):
    results = {}
    for left, right in (('full_model', 'uniform_belief'), ('full_model', 'oracle_type')):
        if left not in cfg['variants'] or right not in cfg['variants']:
            continue
        total = divergent = 0
        for case in case_rows:
            left_path = out / 'episodes' / f"{case['case_id']}-{left}" / 'steps.csv'
            right_path = out / 'episodes' / f"{case['case_id']}-{right}" / 'steps.csv'
            if not left_path.exists() or not right_path.exists():
                continue
            with left_path.open(encoding='utf-8-sig') as handle:
                left_rows = list(csv.DictReader(handle))
            with right_path.open(encoding='utf-8-sig') as handle:
                right_rows = list(csv.DictReader(handle))
            for a, b in zip(left_rows, right_rows):
                total += 1
                if a['action_type'] != b['action_type'] or a['target_lane'] != b['target_lane']:
                    divergent += 1
        results[f'{left}_vs_{right}'] = dict(steps=total, divergent_steps=divergent,
                                             divergent_fraction=divergent / total if total else None)
    return results


def binary_metrics(records):
    tp = sum(r.true_label == 'Aggressive' and r.predicted_label == 'Aggressive' for r in records)
    fn = sum(r.true_label == 'Aggressive' and r.predicted_label != 'Aggressive' for r in records)
    fp = sum(r.true_label != 'Aggressive' and r.predicted_label == 'Aggressive' for r in records)
    tn = sum(r.true_label != 'Aggressive' and r.predicted_label != 'Aggressive' for r in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_pos = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_n = tn / (tn + fn) if tn + fn else 0.0
    recall_n = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = 2 * precision_n * recall_n / (precision_n + recall_n) if precision_n + recall_n else 0.0
    return dict(accuracy=(tp + tn) / len(records) if records else 0.0,
                macro_f1=(f1_pos + f1_neg) / 2,
                confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn))


def diagnostics(out, cfg):
    records, by_truth = [], defaultdict(list)
    truth_counts = defaultdict(int)
    predicted_counts = defaultdict(int)
    for episode_dir in sorted((out / 'episodes').glob('*-full_model')):
        initial = json.loads((episode_dir / 'initial_state.json').read_text(encoding='utf-8'))['initial_state']
        truth = {int(v['id']): v.get('behaviour') for v in initial if v.get('behaviour') in BEHAVIOR_LABELS}
        for label in truth.values():
            truth_counts[label] += 1
        events = json.loads((episode_dir / 'belief_events.json').read_text(encoding='utf-8'))
        seen = set()
        for event in events:
            if event.get('event') != 'response':
                continue
            for raw_id, probabilities in (event.get('probabilities') or {}).items():
                vid = int(raw_id)
                if vid not in truth or (str(episode_dir), vid) in seen:
                    continue
                seen.add((str(episode_dir), vid))
                values = list(probabilities) if isinstance(probabilities, list) else None
                predicted = BEHAVIOR_LABELS[max(range(3), key=lambda i: values[i])] if values else None
                record = EvaluationRecord(
                    case_id=str(episode_dir), vehicle_id=vid, true_label=truth[vid],
                    predicted_label=predicted,
                    p_cautious=values[0] if values else None,
                    p_normal=values[1] if values else None,
                    p_aggressive=values[2] if values else None,
                    latency_s=(event.get('latency_ms') or 0.) / 1000.,
                    error=event.get('error'))
                records.append(record)
                by_truth[truth[vid]].append(record)
                predicted_counts[predicted] += 1
    metrics = calculate_metrics(records)
    return dict(
        samples=len(records), accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
        mean_brier_score=metrics['mean_brier_score'],
        mean_latency_ms=metrics['mean_latency_s'] * 1000.,
        confusion_matrix=metrics['confusion_matrix'],
        binary_force=binary_metrics(records),
        true_label_counts=dict(truth_counts), predicted_label_counts=dict(predicted_counts),
        by_true_label={label: dict(
            samples=len(part), accuracy=calculate_metrics(part)['accuracy'],
            macro_f1=calculate_metrics(part)['macro_f1']) for label, part in by_truth.items()})


def analyze(out, cfg, case_rows):
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
            runtime_errors=sum(row['status'] == 'runtime_error' for row in part),
            probe_steps=sum(row['probe_selected_steps'] for row in part),
            mean_speed_ms=float(np.mean([row['average_speed_ms'] for row in part])),
            mean_restricted_time_s=float(np.mean([row['restricted_task_time_s'] for row in part])),
            mean_min_clearance_m=float(np.mean([row['min_clearance_m'] for row in part
                                                if row['min_clearance_m'] is not None])))
    lookup = {(row['case_id'], row['variant']): row for row in rows}
    comparisons = {}
    for variant in cfg['variants']:
        if variant != 'full_model':
            comparisons[f'full_model_vs_{variant}'] = mcnemar('full_model', variant, lookup, case_rows)
    pairs, effects = paired_analysis(rows, cfg, case_rows)
    initial = defaultdict(set)
    for row in rows:
        initial[row['case_id']].add(row['initial_hash'])
    diag = diagnostics(out, cfg)
    divergence = action_divergence(out, cfg, case_rows)
    llm_variants = [name for name in cfg['variants'] if name in LLM_VARIANTS]
    gates = dict(
        complete=len(rows) == len(case_rows) * len(cfg['variants']),
        paired=all(len(value) == 1 for value in initial.values()),
        no_runtime_errors=not any(row['status'] == 'runtime_error' for row in rows),
        llm_requested=all(by_variant[name]['episodes'] > 0 for name in llm_variants),
        response_errors_zero=not any(
            event.get('error')
            for episode_dir in (out / 'episodes').glob('*-full_model')
            for event in json.loads((episode_dir / 'belief_events.json').read_text(encoding='utf-8'))
            if event.get('event') == 'response'))
    result = dict(
        git_commit=git_commit(), protocol_hash=canonical_hash({k: v for k, v in cfg.items() if not k.startswith('_')}),
        phase=cfg.get('phase', 'df_2_2 type cut-in'),
        by_variant=by_variant, paired_effects=effects, mcnemar=comparisons,
        action_divergence=divergence, llm_diagnostics=diag,
        gates=gates, all_gates_pass=all(gates.values()),
        algorithm_change=False, scene_change_only=True,
        interpretation=('CutinEnv native random-type traffic with a local merge window; '
                        'no scripted key-vehicle response. Scene feasibility adapter, not a '
                        'confirmatory benchmark.'))
    runner.dump(out / 'df_cutin_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ablation_v2_type_observable_df_cutin_pilot.json')
    parser.add_argument('--output-dir',
                        default='simulation_results/ablation_v2/type_observable_df_cutin_pilot_001')
    args = parser.parse_args()
    cfg, out = load_config(args.config), Path(args.output_dir)
    cfg['_output_dir'] = str(out)
    blocked = missing_credentials(cfg)
    if blocked:
        raise RuntimeError('Missing DASHSCOPE_API_KEY for ' + ','.join(blocked))
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = dict(config={k: v for k, v in cfg.items() if not k.startswith('_')},
                    git_commit=git_commit(), source_hashes=hashes,
                    code_hash=canonical_hash(hashes),
                    protocol_hash=canonical_hash({k: v for k, v in cfg.items() if not k.startswith('_')}),
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
    with patch.object(runner, 'make_env', lambda spec: make_env(spec)):
        warm = runner.run_episode(warm_cfg, case_rows[0], 'uniform_belief', out / 'warmup')
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
                    with patch.object(runner, 'make_env', lambda spec: make_env(spec)):
                        result = runner.run_episode(cfg, case, variant, out, llm_client=llm_client)
                rows.append(result)
                runner.write_csv(out / 'episodes.csv', rows)
                print(f"[{len(rows)}/{len(case_rows) * len(cfg['variants'])}] "
                      f"{case['case_id']} {variant}: {result['status']} "
                      f"steps={result['steps']} requests={result['llm_requests']} "
                      f"updates={result['llm_updates']}", flush=True)
    finally:
        if llm_client:
            llm_client.close()
    runner.dump(out / 'completion.json', dict(
        planned=len(case_rows) * len(cfg['variants']), finished=len(rows),
        runtime_errors=sum(row['status'] == 'runtime_error' for row in rows)))
    report(out)
    print(json.dumps(analyze(out, cfg, case_rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
