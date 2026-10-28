"""Type-observable work-zone merge on top of the DF highway environment.

``HighwayEnv`` (registered as ``HighwayEnv``) is the user's four-lane,
20 km cruise environment with the MPC trajectory action.  As a pure free
cruise it contains no key vehicle, no type-consistent response and no route
urgency, so the type-belief ablation has no channel through which it can
change a decision (see the offline cruise scans in the iteration-10 report).

This module keeps the DF environment's four-lane road, DF vehicle classes and
``Trajectory_for_mpc_curve`` action, but adds the mechanism that was validated
in the two- and three-lane merge scenes:

* the ego's right lane (lane 2) is closed at ``ending_lane_x_m`` (a work-zone
  lane closure), and lane 1 is required after the observation gate;
* a type-observable key vehicle runs behind the ego in lane 1, with the same
  yield / idm / force responses used by the validated merge protocol;
* other DF background traffic is cleared and replaced by two distant cruise
  vehicles so the prompt is focused on the key, matching the validated merge
  scene construction.

The controller is not modified: ``scene_data`` already reads
``decision_open_x``, ``required_target_lane`` and ``scene_speed_limit_ms``.
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

import gymnasium as gym
import highway_env  # noqa: F401  (registers HighwayEnv)
import numpy as np

from model.reasoning.worker import LLMReasoningWorker
from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import (LLM_VARIANTS, canonical_hash, load_config,
                                            missing_credentials, source_hashes)
from experiments.ablation_v2.llm_targeted_pilot import git_commit, llm_diagnostics
from experiments.ablation_v2.reporting import report
from experiments.ablation_v2.type_observable_multilane_merge import (
    install_multilane_response, speed_summary)
from highway_env.vehicle.behavior import (AggressiveVehicle_df, CautiousVehicle_df,
                                          NormalVehicle_df)

DF_TYPES = {0: CautiousVehicle_df, 1: NormalVehicle_df, 2: AggressiveVehicle_df}


def _nodes(raw):
    graph = raw.road.network.graph
    origin = next(iter(graph))
    destination = next(iter(graph[origin]))
    return origin, destination


def alias_road(raw):
    """Give the DF graph the ``'a' -> 'b'`` alias used by ``scene_data``.

    ``HighwayEnv`` builds its network with ``straight_road_network``'s
    default nodes ``'0' -> '1'``, while the ablation controller reads lane
    points from ``env.road.network.graph['a']['b']``.  The alias points at the
    same lane objects; no geometry is duplicated.
    """
    graph = raw.road.network.graph
    if 'a' not in graph:
        origin, destination = _nodes(raw)
        graph['a'] = {'b': graph[origin][destination]}


def _set_lane_length(lane, length):
    lane.end[0] = float(length)
    lane.length = float(length)
    lane.direction = (lane.end - lane.start) / lane.length


def _add_vehicle(raw, cls, vid, lane, x, speed, target_speed, target_lane):
    origin, destination = _nodes(raw)
    vehicle = cls(raw.road, [float(x), 4. * float(lane)], speed=float(speed))
    vehicle.id, vehicle.type = int(vid), 'car'
    vehicle.target_lane_index = (origin, destination, int(target_lane))
    vehicle.target_speed = float(target_speed)
    vehicle.enable_lane_change = False
    vehicle.on_state_update()
    raw.road.vehicles.append(vehicle)
    return vehicle


def cases(cfg):
    spec = cfg['observable_df_workzone']
    jitter = spec.get('jitter', {})
    target_speeds = spec.get('key_target_speed_ms', {'0': 8., '1': 10., '2': 12.})
    responses = spec.get('key_response', {'0': 'yield', '1': 'idm', '2': 'force'})
    speed_offsets = spec.get('key_initial_speed_offsets_ms', {})
    for index, seed in enumerate(cfg['seeds']):
        rng = np.random.default_rng(seed)
        behaviour = index % 3
        offsets = speed_offsets.get(str(behaviour)) or [0.]
        offset = float(offsets[(index // 3) % len(offsets)])
        yield dict(
            case_id=f'observable-df-workzone-{seed}', scenario='traffic', seed=seed,
            difficulty=1, behaviour_index=behaviour,
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
            key_ramp=copy.deepcopy(spec.get('key_ramp')),
            response=dict(spec.get('response', {})),
            decision_open_x_m=float(spec['decision_open_x_m']),
            scene_speed_limit_ms=float(spec['scene_speed_limit_ms']),
            ego_lane=int(spec['ego_lane']),
            key_lane=int(spec['key_lane']),
            required_target_lane=int(spec['required_target_lane']),
            ending_lane=int(spec['ending_lane']),
            ending_lane_x_m=float(spec['ending_lane_x_m']),
            required_progress_m=float(spec.get('required_progress_m', 100.)),
            road_length_m=float(spec.get('road_length_m', 800.)),
            background=copy.deepcopy(spec.get('background', [])),
            horizon_s=float(spec['horizon_s']))


def install_common_ramp(env, spec):
    """Delay the type-specific response behind a common speed ramp.

    All key vehicles follow the same speed profile until ``hold_until_s``.
    The first 20-frame observation window therefore cannot separate Yield,
    Normal and Force: the type-specific response only starts after the first
    belief has been formed.  This is a scene-level weak-identifiability
    manipulation; the controller and the LLM protocol are unchanged.
    """
    if not spec:
        return
    raw = env.unwrapped
    key = raw.key_vehicle
    original = key.acceleration
    target = float(spec['target_speed_ms'])
    acceleration = float(spec['acceleration_ms2'])
    duration = float(spec['duration_s'])
    hold_until = float(spec.get('hold_until_s', duration))
    key.ACC_MAX = max(float(key.ACC_MAX), acceleration)

    def ramp_acceleration(*args, **kwargs):
        now = float(raw.time)
        if now < duration and key.speed < target:
            return acceleration
        if now < hold_until:
            return 0.0
        return original(*args, **kwargs)

    key.acceleration = ramp_acceleration


def apply_scene(env, case):
    """Build the DF work-zone merge scene on a freshly reset DF environment."""
    raw = env.unwrapped
    alias_road(raw)

    # The DF road is 20 km long.  Shorten the lanes used by this scene so the
    # controller's lane-point cache stays small.  Lane 2 is the closed lane.
    ending_lane = int(case['ending_lane'])
    for index, lane in enumerate(raw.road.network.graph['a']['b']):
        if index == ending_lane:
            _set_lane_length(lane, float(case['ending_lane_x_m']))
        else:
            _set_lane_length(lane, float(case['road_length_m']))

    raw.kind = 'df_workzone'
    raw.decision_open_x = float(case['decision_open_x_m'])
    raw.required_target_lane = int(case['required_target_lane'])
    raw.scene_speed_limit_ms = float(case['scene_speed_limit_ms'])

    ego = raw.vehicle
    ego.position[:] = [20., 4. * float(case['ego_lane'])]
    ego.speed = float(case['ego_speed_ms'])
    ego.target_speed = float(case['scene_speed_limit_ms'])
    ego.id, ego.type = 888, 'AV'
    origin, destination = _nodes(raw)
    ego.target_lane_index = (origin, destination, int(case['ego_lane']))
    ego.on_state_update()
    raw.controlled_vehicles = [ego]
    raw.road.vehicles = [ego]

    key = _add_vehicle(raw, DF_TYPES[int(case['behaviour_index'])], 1,
                       int(case['key_lane']),
                       float(ego.position[0]) + float(case['key_relative_x_m']),
                       float(case['key_initial_speed_ms']),
                       float(case['key_target_speed_ms']), int(case['key_lane']))
    raw.key_vehicle = key

    for index, spec in enumerate(case.get('background', []), start=2):
        _add_vehicle(raw, NormalVehicle_df, index, int(spec['lane']),
                     float(ego.position[0]) + float(spec['relative_x_m']),
                     float(spec['speed_ms']), float(spec['speed_ms']),
                     int(spec['lane']))

    install_multilane_response(env, case.get('key_response', 'idm'), case.get('response'))
    install_common_ramp(env, case.get('key_ramp'))
    return env


def make_env(case):
    """Create and reset the DF environment for one case."""
    env = gym.make('HighwayEnv')
    env.reset(seed=case['seed'])
    return apply_scene(env, case)


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
                average_speed_difference_ms=(full['average_speed_ms']
                                             - other['average_speed_ms']),
                completion_time_difference_s=(full['completion_time_s']
                                              - other['completion_time_s'])
                if full['completion_time_s'] is not None and other['completion_time_s'] is not None else None,
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
                average_speed_difference_ms=(oracle['average_speed_ms']
                                             - uniform['average_speed_ms']),
                completion_time_difference_s=(oracle['completion_time_s']
                                              - uniform['completion_time_s'])
                if oracle['completion_time_s'] is not None and uniform['completion_time_s'] is not None else None,
                restricted_time_difference_s=(oracle['restricted_task_time_s']
                                              - uniform['restricted_task_time_s']),
                clearance_difference_m=(oracle['min_clearance_m']
                                        - uniform['min_clearance_m'])
                if oracle['min_clearance_m'] is not None and uniform['min_clearance_m'] is not None else None,
                full_probe_steps=oracle['probe_selected_steps'],
                comparison_probe_steps=uniform['probe_selected_steps']))
    runner.write_csv(out / 'observable_df_workzone_pairs.csv', pairs)
    effects = {}
    for comparison in sorted({row['comparison'] for row in pairs}):
        part = [row for row in pairs if row['comparison'] == comparison]
        effects[comparison] = dict(
            better_success=sum(row['success_difference'] > 0 for row in part),
            worse_success=sum(row['success_difference'] < 0 for row in part),
            fewer_collisions=sum(row['collision_difference'] < 0 for row in part),
            more_collisions=sum(row['collision_difference'] > 0 for row in part),
            mean_average_speed_difference_ms=float(np.mean(
                [row['average_speed_difference_ms'] for row in part])),
            mean_completion_time_difference_s=_mean(
                row['completion_time_difference_s'] for row in part),
            mean_restricted_time_difference_s=float(np.mean(
                [row['restricted_time_difference_s'] for row in part])),
            mean_probe_step_difference=float(np.mean(
                [row['full_probe_steps'] - row['comparison_probe_steps'] for row in part])))
    return pairs, effects


def _mean(values):
    part = [value for value in values if value is not None]
    return float(np.mean(part)) if part else None


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
            **speed_summary(part))
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
        phase=cfg.get('phase', 'type-observable DF work-zone merge'),
        by_variant=by_variant, paired_effects=effects, action_divergence=divergence,
        llm_diagnostics=diagnostics, gates=gates, all_gates_pass=all(gates.values()),
        algorithm_change=False, scene_change_only=True,
        interpretation=('HighwayEnv four-lane work-zone merge: the ego starts in the '
                        'closed right lane (2), lane 1 is required after the observation '
                        'gate, and the type-observable key vehicle is behind in lane 1. '
                        'The DF road, vehicle classes and MPC trajectory action are '
                        'kept; the work-zone lane closure and key vehicle are the scene '
                        'change.  Not confirmatory evidence.'))
    runner.dump(out / 'observable_df_workzone_analysis.json', analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/ablation_v2_type_observable_df_workzone_pilot.json')
    parser.add_argument('--output-dir',
                        default='simulation_results/ablation_v2/type_observable_df_workzone_pilot_001')
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
    warm_cfg = copy.deepcopy(cfg)
    warm_cfg['runtime']['max_steps'] = 1
    with patch.object(runner, 'make_env', make_env):
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
                    with patch.object(runner, 'make_env', make_env):
                        result = runner.run_episode(cfg, case, variant, out,
                                                    llm_client=llm_client)
                rows.append(result)
                runner.write_csv(out / 'episodes.csv', rows)
                print(f"[{len(rows)}/{len(case_rows)*len(cfg['variants'])}] "
                      f"{case['case_id']} {variant}: {result['status']} "
                      f"speed={result['average_speed_ms']:.3f} "
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
