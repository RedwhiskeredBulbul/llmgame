"""Type-observable low-speed merge: scenario and closed-loop protocol.

The key vehicle starts behind the ego in the target lane with a
behaviour-consistent cruising speed.  The ego keeps its ending lane until
``merge_open_x``, which gives the LLM a clean observation window before the
lane-change decision.  The key vehicle's type determines how quickly it closes
the gap; a Yield belief makes an early commit look safe, while a Force belief
should make the ego delay or probe.

Only the initial scene and the lane-change opening point are changed.  The
controller, reward, candidate set, vehicle models and success rules are shared
with the other v2 experiments.

The key vehicle additionally implements the strategic action the type belief
refers to: a Cautious key decelerates for a merging ego (yield), a Normal key
keeps plain IDM following, and an Aggressive key accelerates to refuse the
merge (force).  The response is keyed only on the ego's physical lateral
crossing and on the case's behaviour index, never on the controller variant,
so all variants face the same environment.  Without this response the stock
IDM followers always brake, which made the type belief redundant.
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

from LLMreasoning.llmreason import LLMReasoningWorker
from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import (LLM_VARIANTS, canonical_hash, load_config,
                                            missing_credentials, source_hashes)
from experiments.ablation_v2.llm_targeted_pilot import git_commit, llm_diagnostics
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


def install_merge_response(env, mode, params=None):
    """Attach the type-consistent response to the merge key vehicle.

    ``mode`` is one of ``yield``, ``idm`` and ``force``.  The override is
    active only while the ego is physically crossing toward the key's lane, so
    a controller that keeps its lane never elicits a response.  It is an
    instance-level override; the stock vehicle classes stay untouched.
    """
    if mode == 'idm':
        return
    if mode not in ('yield', 'force'):
        raise ValueError(f'Unknown merge response: {mode}')
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


def cases(cfg):
    spec = cfg['observable_merge']
    jitter = spec.get('jitter', {})
    target_speeds = spec.get('key_target_speed_ms', {'0': 8., '1': 10., '2': 12.})
    responses = spec.get('key_response', {'0': 'yield', '1': 'idm', '2': 'force'})
    speed_offsets = spec.get('key_initial_speed_offsets_ms', {})
    randomize_key = bool(spec.get('randomize_key_type', False))
    key_types = [index % 3 for index in range(len(cfg['seeds']))]
    if randomize_key:
        # Random permutation keeps the key-type counts balanced while removing
        # the fixed seed-to-type mapping.
        random.Random(int(spec.get('key_type_seed', 20261022))).shuffle(key_types)
    for index, seed in enumerate(cfg['seeds']):
        rng = np.random.default_rng(seed)
        behaviour = key_types[index] if randomize_key else index % 3
        background_types = None
        if spec.get('randomize_background_types'):
            # Independent seeded draw per background vehicle, independent of
            # the simulator RNG and of the key-type assignment.
            type_rng = np.random.default_rng(seed + int(spec.get('type_seed_offset', 20261022)))
            background_types = [int(type_rng.integers(0, 3)) for _ in range(2)]
        offsets = speed_offsets.get(str(behaviour)) or [0.]
        offset = float(offsets[(index // 3) % len(offsets)])
        yield dict(
            case_id=f'observable-merge-{seed}', scenario='merge', seed=seed,
            difficulty=1, behaviour_index=behaviour,
            background_types=background_types,
            ego_speed_ms=float(spec['ego_speed_ms'] + rng.uniform(
                -jitter.get('ego_speed_ms', 0.), jitter.get('ego_speed_ms', 0.))),
            key_relative_x_m=float(spec['key_relative_x_m'] + rng.uniform(
                -jitter.get('key_relative_x_m', 0.), jitter.get('key_relative_x_m', 0.))),
            key_initial_speed_ms=float(
                spec['key_initial_speed_ms'][str(behaviour)] + offset + rng.uniform(
                    -jitter.get('key_initial_speed_ms', 0.),
                    jitter.get('key_initial_speed_ms', 0.))),
            key_target_speed_ms=float(
                target_speeds[str(behaviour)] + rng.uniform(
                    -jitter.get('key_target_speed_ms', 0.),
                    jitter.get('key_target_speed_ms', 0.))),
            key_response=responses[str(behaviour)],
            response=dict(spec.get('response', {})),
            merge_open_x_m=float(spec['merge_open_x_m']),
            horizon_s=float(spec['horizon_s']))


def apply_scene(env, case):
    """Place the ego and key vehicle and enable the merge after the observation phase."""
    raw = env.unwrapped
    raw.merge_open_x = float(case['merge_open_x_m'])
    raw.vehicle.speed = float(case['ego_speed_ms'])
    raw.vehicle.on_state_update()
    key = raw.key_vehicle
    key.position[0] = raw.vehicle.position[0] + float(case['key_relative_x_m'])
    key.speed = float(case['key_initial_speed_ms'])
    if case.get('key_target_speed_ms') is not None:
        key.target_speed = float(case['key_target_speed_ms'])
    key.on_state_update()
    # Keep the same vehicle count but isolate the rear interaction: move the
    # other two cars well ahead so they do not dominate the merge decision.
    by_id = {vehicle.id: vehicle for vehicle in raw.road.vehicles}
    if 2 in by_id:
        by_id[2].position[0] = raw.vehicle.position[0] + 60.
        by_id[2].on_state_update()
    if 3 in by_id:
        by_id[3].position[0] = raw.vehicle.position[0] + 75.
        by_id[3].on_state_update()
    install_merge_response(env, case.get('key_response', 'idm'), case.get('response'))
    return env


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
    runner.write_csv(out / 'observable_merge_pairs.csv', pairs)
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
    """Count steps where two variants choose a different action or target lane."""
    results = {}
    for left, right in (('full_model', 'uniform_belief'), ('oracle_type', 'uniform_belief'),
                        ('full_model', 'oracle_type')):
        if left not in cfg['variants'] or right not in cfg['variants']:
            continue
        total = divergent = 0
        for case in cases(cfg):
            left_dir = out / 'episodes' / f"{case['case_id']}-{left}"
            right_dir = out / 'episodes' / f"{case['case_id']}-{right}"
            left_steps = left_dir / 'steps.csv'
            right_steps = right_dir / 'steps.csv'
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
        results[f'{left}_vs_{right}'] = dict(steps=total, divergent_steps=divergent,
                                             divergent_fraction=divergent / total if total else None)
    return results


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
    result = dict(
        git_commit=git_commit(), protocol_hash=canonical_hash(cfg),
        phase=cfg.get('phase', 'type-observable merge pilot'),
        by_variant=by_variant, paired_effects=effects, action_divergence=divergence,
        llm_diagnostics=diagnostics, gates=gates, all_gates_pass=all(gates.values()),
        algorithm_change=False, scene_change_only=True,
        interpretation=('Scenario opens the target lane only after the observation phase; '
                        'the key vehicle is behind in the target lane and its type controls '
                        'gap closure. Not confirmatory evidence.'))
    runner.dump(out / 'observable_merge_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ablation_v2_type_observable_merge_pilot.json')
    parser.add_argument('--output-dir',
                        default='simulation_results/ablation_v2/type_observable_merge_pilot_001')
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
