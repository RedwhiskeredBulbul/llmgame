"""Type-observable cut-in on DF_2_2 with a strategic rear key vehicle.

This is a scene-level iteration on the repository's native
``_2_2_TWOLaneChangeEnv``.  The native road, dense target-lane stream, and the
independent random behaviour labels of its background vehicles are retained.
The adapter only opens a local merge window and selects the nearest native
vehicle behind the ego as the *key* vehicle.  The key keeps its native
behaviour label, but its initial speed and its response while the ego crosses
are made type-consistent:

* Aggressive: approaches at 10.5 m/s and then accelerates to refuse the merge.
* Normal: approaches moderately and keeps native cooperative IDM behaviour.
* Cautious: hangs back and decelerates to yield.

When occlusion is enabled, half of the episodes add three constant-speed native
vehicles in the ego lane between the ego and the rear key.  At the first
synchronous observation these vehicles are closer than the key, so the prompt's
nearest-three rule does not include it even though the controller's nearest-five
set does.  This is an attention/observation-budget condition, assigned by seed
order independently of the key type.  A no-occlusion config sets
``df_cutin_key.occlusion`` to false and keeps the key in the first observation
for every seed.  The key response depends only on the ego's physical lateral
crossing and its native type, never on the controller variant, so every variant
faces the same environment.
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

import gymnasium as gym
import numpy as np
from scipy.stats import binomtest

import highway_env  # noqa: F401
from LLMreasoning.llmreason import LLMReasoningWorker
from LLMreasoning.vehicle_type_evaluation import BEHAVIOR_LABELS, calculate_metrics, EvaluationRecord
from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import (LLM_VARIANTS, canonical_hash, load_config,
                                            missing_credentials, source_hashes)
from experiments.ablation_v2.llm_targeted_pilot import git_commit
from experiments.ablation_v2.reporting import report

RESPONSE_DEFAULTS = dict(
    trigger_lateral_y_m=3.5,
    response_range_m=45.0,
    yield_deceleration_ms2=-2.0,
    force_acceleration_ms2=2.0,
    force_speed_cap_ms=15.0,
    force_min_gap_m=6.0,
    force_emergency_deceleration_ms2=-5.0,
)
TYPE_ORDER = ('Cautious', 'Normal', 'Aggressive')
TYPE_TO_INDEX = {label: index for index, label in enumerate(TYPE_ORDER)}


def _native_scene(seed):
    """Return the native rear key and the three nearest distractor vehicles."""
    env = gym.make('DF_2_2')
    try:
        env.reset(seed=int(seed))
        raw = env.unwrapped
        ego_x = float(raw.vehicle.position[0])
        lane0 = [v for v in raw.road.vehicles
                 if v.id != raw.vehicle.id and abs(float(v.position[1])) < 1.0]
        behind = [v for v in lane0 if float(v.position[0]) < ego_x]
        if not behind:
            raise RuntimeError(f'Seed {seed}: target lane needs a rear vehicle')
        key = max(behind, key=lambda v: float(v.position[0]))
        others = sorted((v for v in lane0 if v.id != key.id),
                        key=lambda v: abs(float(v.position[0]) - ego_x))
        if len(others) < 3:
            raise RuntimeError(f'Seed {seed}: target lane needs three distractors')
        return dict(key_id=int(key.id), key_type=str(key.BEHAVIOR_TYPE),
                    distractor_ids=[int(v.id) for v in others[:3]],
                    distractor_types=[str(v.BEHAVIOR_TYPE) for v in others[:3]])
    finally:
        env.close()


def cases(cfg):
    spec = cfg['df_cutin_key']
    base_speed = spec['key_speed_ms']
    target_speed = spec['key_target_speed_ms']
    responses = spec['key_response']
    for index, seed in enumerate(cfg['seeds']):
        native = _native_scene(seed)
        key_type = native['key_type']
        # When occlusion is enabled, alternate it by seed-list index,
        # independent of type and outcome.  In an occluded episode three
        # constant-speed native vehicles are placed in the ego lane between the
        # ego and the rear key.  They are closer than the key at the first
        # synchronous observation, so the nearest-three prompt does not include
        # the key.  The key remains in the controller's nearest-five set and in
        # the physical scene.  A no-occlusion config sets ``occlusion`` to false.
        occluded = bool(spec.get('occlusion', True)) and index % 2 == 0
        distractors = []
        if occluded:
            for vehicle_id, dx, speed in zip(native['distractor_ids'],
                                             spec['distractor_relative_x_m'],
                                             spec['distractor_speed_ms']):
                distractors.append(dict(id=int(vehicle_id), relative_x_m=float(dx),
                                        speed_ms=float(speed)))
        yield dict(
            case_id=f'df-cutin-key-{seed}', scenario='cutin', seed=int(seed),
            difficulty=1, behaviour_index=TYPE_TO_INDEX[key_type],
            required_target_lane=int(spec['required_target_lane']),
            required_progress_m=float(spec.get('required_progress_m', 0.0)),
            horizon_s=float(spec['horizon_s']),
            decision_open_x_m=float(spec['decision_open_x_m']),
            route_end_x_m=float(spec['route_end_x_m']),
            ego_speed_ms=float(spec['ego_speed_ms']),
            speed_limit_ms=float(spec['speed_limit_ms']),
            clear_ahead_m=float(spec['clear_ahead_m']),
            clear_behind_m=float(spec['clear_behind_m']),
            key_id=native['key_id'], key_type=key_type,
            key_relative_x_m=float(spec['key_relative_x_m']),
            key_initial_speed_ms=float(base_speed[key_type]),
            key_target_speed_ms=float(target_speed[key_type]),
            key_response=str(responses[key_type]),
            occluded=bool(occluded), distractors=distractors,
            response=dict(spec.get('response', {})),
        )


def install_key_response(env, mode, params=None):
    """Attach a type-consistent response to the key vehicle.

    ``mode`` is one of ``yield``, ``idm`` and ``force``.  The override is active
    only while the ego is physically crossing toward the key's lane, so a
    controller that keeps its lane never elicits a response.
    """
    if mode == 'idm':
        return
    if mode not in ('yield', 'force'):
        raise ValueError(f'Unknown key response: {mode}')
    raw = env.unwrapped
    key, ego = raw.key_vehicle, raw.vehicle
    settings = {**RESPONSE_DEFAULTS, **(params or {})}
    original = key.acceleration

    def acceleration(ego_vehicle=None, front_vehicle=None, rear_vehicle=None):
        gap = float(ego.position[0] - key.position[0])
        crossing = float(ego.position[1]) < float(settings['trigger_lateral_y_m'])
        if crossing and 0. < gap < float(settings['response_range_m']):
            if mode == 'yield' and key.speed > 4.:
                return float(settings['yield_deceleration_ms2'])
            if mode == 'force':
                if gap > float(settings['force_min_gap_m']):
                    return (float(settings['force_acceleration_ms2'])
                            if key.speed < float(settings['force_speed_cap_ms']) else 0.)
                return float(settings['force_emergency_deceleration_ms2'])
        return original(ego_vehicle=ego_vehicle, front_vehicle=front_vehicle,
                        rear_vehicle=rear_vehicle)

    key.acceleration = acceleration


def apply_scene(env, case):
    raw = env.unwrapped
    raw.kind = 'cutin'
    raw.required_target_lane = int(case['required_target_lane'])
    raw.required_progress_m = float(case['required_progress_m'])
    raw.decision_open_x = float(case['decision_open_x_m'])
    raw.route_end_x = float(case['route_end_x_m'])
    raw.vehicle.speed = float(case['ego_speed_ms'])
    raw.vehicle.on_state_update()
    raw.scene_speed_limit_ms = float(case['speed_limit_ms'])
    ego_x = float(raw.vehicle.position[0])

    by_id = {v.id: v for v in raw.road.vehicles}
    key = by_id[int(case['key_id'])]
    distractors = [by_id[int(item['id'])] for item in case.get('distractors', [])]
    keep = {id(raw.vehicle), id(key), *(id(v) for v in distractors)}
    # Keep only the key and the occlusion distractors in the local corridor.
    # All other native traffic and its random labels remain untouched outside
    # the corridor.
    for vehicle in list(raw.road.vehicles):
        if id(vehicle) in keep:
            continue
        if abs(float(vehicle.position[1])) >= 1.0:
            continue
        dx = float(vehicle.position[0]) - ego_x
        if -float(case['clear_behind_m']) <= dx <= float(case['clear_ahead_m']):
            raw.road.vehicles.remove(vehicle)

    key.position[0] = ego_x + float(case['key_relative_x_m'])
    key.position[1] = 0.
    key.speed = float(case['key_initial_speed_ms'])
    key.target_speed = float(case['key_target_speed_ms'])
    key.on_state_update()
    key.enable_lane_change = False
    key.target_lane_index = key.lane_index

    for vehicle, item in zip(distractors, case.get('distractors', [])):
        vehicle.position[0] = ego_x + float(item['relative_x_m'])
        vehicle.position[1] = 4.
        vehicle.speed = float(item['speed_ms'])
        vehicle.target_speed = float(item['speed_ms'])
        vehicle.on_state_update()
        # Constant-speed background vehicles: they create the observation
        # occlusion and must not brake, accelerate, or change lane away from
        # their slots.
        vehicle.enable_lane_change = False
        vehicle.target_lane_index = vehicle.lane_index
        vehicle.acceleration = lambda ego_vehicle=None, front_vehicle=None, rear_vehicle=None: 0.

    raw.key_vehicle = key
    install_key_response(env, case.get('key_response', 'idm'), case.get('response'))
    return env


def make_env(case):
    env = gym.make('DF_2_2')
    env.reset(seed=int(case['seed']))
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
    pairs = []
    for comparison in [name for name in cfg['variants'] if name != 'full_model']:
        for case in case_rows:
            full, other = lookup[(case['case_id'], 'full_model')], lookup[(case['case_id'], comparison)]
            pairs.append(dict(
                case_id=case['case_id'], key_type=case['key_type'],
                occluded=bool(case['occluded']), comparison=comparison,
                full_status=full['status'], comparison_status=other['status'],
                success_difference=full['success'] - other['success'],
                collision_difference=full['collision'] - other['collision'],
                restricted_time_difference_s=full['restricted_task_time_s'] - other['restricted_task_time_s'],
                full_probe_steps=full['probe_selected_steps'],
                comparison_probe_steps=other['probe_selected_steps']))
    runner.write_csv(Path(cfg['_output_dir']) / 'df_cutin_key_pairs.csv', pairs)
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


def action_divergence(out, cfg, case_rows):
    results = {}
    for left, right in (('full_model', 'uniform_belief'), ('full_model', 'oracle_type'),
                        ('oracle_type', 'uniform_belief')):
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
        results[f'{left}_vs_{right}'] = dict(
            steps=total, divergent_steps=divergent,
            divergent_fraction=divergent / total if total else None)
    return results


def _label(values):
    return BEHAVIOR_LABELS[max(range(3), key=lambda i: values[i])]


def _diagnostic_metrics(records, field):
    evaluation = []
    for record in records:
        values = record.get(field)
        if not values:
            continue
        evaluation.append(EvaluationRecord(
            case_id=record['case_id'], vehicle_id=0, true_label=record['key_type'],
            predicted_label=_label(values), p_cautious=values[0],
            p_normal=values[1], p_aggressive=values[2],
            latency_s=(record.get(field.replace('probabilities', 'latency_ms')) or 0.) / 1000.))
    metrics = calculate_metrics(evaluation)
    return dict(
        samples=len(evaluation),
        accuracy=metrics['accuracy'],
        macro_f1=metrics['macro_f1'],
        mean_brier_score=metrics['mean_brier_score'],
        mean_latency_ms=metrics['mean_latency_s'] * 1000.,
        confusion_matrix=metrics['confusion_matrix'])


def key_diagnostics(out, cfg, case_rows):
    records = []
    for case in case_rows:
        episode_dir = out / 'episodes' / f"{case['case_id']}-full_model"
        event_path = episode_dir / 'belief_events.json'
        if not event_path.exists():
            continue
        events = [event for event in json.loads(event_path.read_text(encoding='utf-8'))
                  if event.get('event') == 'response']
        first = events[0] if events else {}
        first_values = (first.get('probabilities') or {}).get(str(case['key_id']))
        later = next((event for event in events
                      if (event.get('probabilities') or {}).get(str(case['key_id'])) is not None),
                     {})
        later_values = (later.get('probabilities') or {}).get(str(case['key_id']))
        record = dict(
            case_id=case['case_id'], key_type=case['key_type'],
            occluded=bool(case['occluded']),
            first_observation_sim_time=first.get('observation_sim_time'),
            first_present=first_values is not None,
            first_predicted=(_label([float(x) for x in first_values])
                             if first_values is not None else None),
            first_probabilities=([float(x) for x in first_values]
                                 if first_values is not None else None),
            first_latency_ms=float(first.get('latency_ms') or 0.),
            later_predicted=(_label([float(x) for x in later_values])
                             if later_values is not None else None),
            later_probabilities=([float(x) for x in later_values]
                                 if later_values is not None else None),
            later_latency_ms=float(later.get('latency_ms') or 0.),
        )
        record['first_correct'] = (record['first_predicted'] == case['key_type'])
        record['later_correct'] = (record['later_predicted'] == case['key_type'])
        records.append(record)
    first = _diagnostic_metrics(records, 'first_probabilities')
    later = _diagnostic_metrics(records, 'later_probabilities')
    by_truth = {}
    for label in TYPE_ORDER:
        part = [record for record in records if record['key_type'] == label]
        observed = [record for record in part if record['first_present']]
        by_truth[label] = dict(
            samples=len(part), first_present=len(observed),
            first_missing=len(part) - len(observed),
            first_accuracy=(float(np.mean([record['first_correct'] for record in observed]))
                            if observed else None),
            first_predictions={name: sum(record['first_predicted'] == name for record in observed)
                               for name in BEHAVIOR_LABELS})
    return dict(
        samples=len(records),
        first_observation_key_present=sum(record['first_present'] for record in records),
        first_observation_key_missing=sum(not record['first_present'] for record in records),
        first_observation=first,
        later_prediction=later,
        by_true_label=by_truth,
        records=records)


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
            comparisons[f'full_model_vs_{variant}'] = mcnemar(
                'full_model', variant, lookup, case_rows)
    if 'oracle_type' in cfg['variants']:
        comparisons['oracle_type_vs_uniform_belief'] = mcnemar(
            'oracle_type', 'uniform_belief', lookup, case_rows)
    by_condition = {}
    for condition, flag in (('visible', False), ('occluded', True)):
        subset = [case for case in case_rows if bool(case['occluded']) == flag]
        if not subset:
            continue
        by_condition[condition] = dict(cases=len(subset), by_variant={}, mcnemar={})
        for variant in cfg['variants']:
            part = [row for row in rows if row['variant'] == variant
                    and row['case_id'] in {case['case_id'] for case in subset}]
            by_condition[condition]['by_variant'][variant] = dict(
                episodes=len(part),
                completed=sum(row['status'] == 'completed' for row in part),
                collisions=sum(row['status'] == 'collision' for row in part),
                timeouts=sum(row['status'] == 'timeout' for row in part))
        for variant in cfg['variants']:
            if variant != 'full_model':
                by_condition[condition]['mcnemar'][f'full_model_vs_{variant}'] = mcnemar(
                    'full_model', variant, lookup, subset)
    pairs, effects = paired_analysis(rows, cfg, case_rows)
    initial = defaultdict(set)
    for row in rows:
        initial[row['case_id']].add(row['initial_hash'])
    diag = key_diagnostics(out, cfg, case_rows)
    divergence = action_divergence(out, cfg, case_rows)
    by_key = {}
    for key_type in TYPE_ORDER:
        by_key[key_type] = {}
        for variant in cfg['variants']:
            part = [row for row in rows
                    if row['variant'] == variant and row['case_id'] in {
                        c['case_id'] for c in case_rows if c['key_type'] == key_type}]
            by_key[key_type][variant] = dict(
                episodes=len(part), completed=sum(row['status'] == 'completed' for row in part),
                collisions=sum(row['status'] == 'collision' for row in part),
                timeouts=sum(row['status'] == 'timeout' for row in part))
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
    occlusion_enabled = bool(cfg.get('df_cutin_key', {}).get('occlusion', True))
    if occlusion_enabled:
        interpretation = (
            'DF_2_2 native road, native random-type background, and a native rear key '
            'vehicle whose initial speed and crossing response are type-consistent. On '
            'alternating seeds three constant-speed native vehicles are closer than the '
            'key at the first synchronous prompt, so the nearest-three observation omits '
            'the key. The occlusion condition and the key response do not depend on the '
            'controller variant. This is a mechanism demonstration, not an unmodified '
            'native DF_2_2 benchmark; inference should be conditioned on the '
            'visible/occluded split.')
    else:
        interpretation = (
            'DF_2_2 native road, native random-type background, and a native rear key '
            'vehicle whose initial speed and crossing response are type-consistent. No '
            'occlusion condition is used: the key is available in the first synchronous '
            'observation for every seed. The key response is type-scripted and independent '
            'of the controller variant. The comparison is limited to the configured '
            'variants; scene-level changes are documented in the protocol.')
    result = dict(
        git_commit=git_commit(),
        protocol_hash=canonical_hash({k: v for k, v in cfg.items() if not k.startswith('_')}),
        phase=cfg.get('phase', 'df_2_2 type key cut-in'),
        occlusion_enabled=occlusion_enabled,
        by_variant=by_variant, by_key_type=by_key, by_condition=by_condition,
        paired_effects=effects, mcnemar=comparisons,
        action_divergence=divergence, key_diagnostics=diag,
        gates=gates, all_gates_pass=all(gates.values()),
        algorithm_change=False, scene_change_only=True,
        interpretation=interpretation)
    runner.dump(out / 'df_cutin_key_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='config/ablation_v2_type_observable_df_cutin_key_pilot.json')
    parser.add_argument('--output-dir',
                        default='simulation_results/ablation_v2/type_observable_df_cutin_key_pilot_001')
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
                        result = runner.run_episode(cfg, case, variant, out,
                                                    llm_client=llm_client)
                rows.append(result)
                runner.write_csv(out / 'episodes.csv', rows)
                print(f"[{len(rows)}/{len(case_rows) * len(cfg['variants'])}] "
                      f"{case['case_id']} {case['key_type']}/occluded={case['occluded']} "
                      f"{variant}: {result['status']} steps={result['steps']} "
                      f"probe={result['probe_selected_steps']} "
                      f"requests={result['llm_requests']} updates={result['llm_updates']}",
                      flush=True)
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
