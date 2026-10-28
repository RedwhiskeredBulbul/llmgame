"""Development scan of an Aggressive rear vehicle near a known decision boundary."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from experiments.ablation_v2.config import canonical_hash, load_config, source_hashes
from experiments.ablation_v2.controller import ExperimentGameSolver, scene_data
from experiments.ablation_v2.runner import dump, make_env, write_csv
from highway_env.vehicle.behavior import AggressiveVehicle_df


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def states(cfg):
    spec = cfg['boundary_scan']
    center, width = spec['center'], spec['half_width']
    for seed in cfg['seeds']:
        rng = np.random.default_rng(seed)
        yield dict(state_id=f'aggressive-boundary-{seed}', scenario='traffic', seed=seed,
            difficulty=1, behaviour_index=spec['background_behaviour_index'],
            ego_speed_ms=float(center['ego_speed_ms'] + rng.uniform(-width['ego_speed_ms'], width['ego_speed_ms'])),
            relative_x_m=float(center['rear_relative_x_m'] + rng.uniform(-width['rear_relative_x_m'], width['rear_relative_x_m'])),
            relative_speed_ms=float(center['rear_relative_speed_ms'] + rng.uniform(-width['rear_relative_speed_ms'], width['rear_relative_speed_ms'])),
            target_lane=spec['target_lane'])


def set_state(env, state, cfg):
    raw = env.unwrapped
    raw.vehicle.speed = state['ego_speed_ms']
    raw.vehicle.on_state_update()
    old = raw.key_vehicle
    key = AggressiveVehicle_df(raw.road,
        [raw.vehicle.position[0] + state['relative_x_m'], 4. * state['target_lane']],
        heading=0., speed=max(0., state['ego_speed_ms'] + state['relative_speed_ms']),
        target_lane_index=('a', 'b', state['target_lane']), enable_lane_change=False)
    key.id, key.type, key.enable_lane_change = old.id, old.type, False
    key.target_speed = 33.3
    raw.road.vehicles[raw.road.vehicles.index(old)] = key
    raw.key_vehicle = key
    clearance = cfg['boundary_scan']['background_clearance_m']
    for vehicle in raw.road.vehicles:
        if vehicle is raw.vehicle or vehicle is key:
            continue
        if int(vehicle.lane_index[2]) == state['target_lane'] and abs(vehicle.position[0] - key.position[0]) < clearance:
            vehicle.position[0] += 30.
            vehicle.on_state_update()
    return raw


def solve(cfg, data, profile):
    solver = ExperimentGameSolver(cfg['decision'])
    beliefs = {vid: [1/3, 1/3, 1/3] for vid in data.others_info['ids']}
    if 1 in beliefs:
        beliefs[1] = [0., 0., 1.] if profile == 'oracle_force' else [1/3, 1/3, 1/3]
    with open(os.devnull, 'w', encoding='utf-8') as null, contextlib.redirect_stdout(null):
        _, acceleration, target = solver.decide(data, beliefs)
    scores = {row['action_type']: row.get('score') for row in solver.branch_rows}
    return dict(profile=profile, action=solver.selected_action, target=target,
        acceleration=float(acceleration[0]), keep_score=scores.get('keep'),
        commit_score=scores.get('commit'), evaluated_branches=solver.evaluated_branches_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ablation_v2_aggressive_boundary_scan.json')
    parser.add_argument('--output-dir', default='simulation_results/ablation_v2/aggressive_boundary_scan_001')
    args = parser.parse_args()
    cfg, out = load_config(args.config), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = dict(git_commit=git_commit(), config=cfg, source_hashes=hashes,
        code_hash=canonical_hash(hashes), protocol_hash=canonical_hash(cfg), algorithm_change=False,
        scene_change_only=True, purpose='development localization; not performance evidence')
    dump(out/'manifest.json', manifest)
    for relative in hashes:
        destination = out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)
    state_rows, rows = list(states(cfg)), []
    dump(out/'states.json', state_rows)
    for index, state in enumerate(state_rows, 1):
        env = make_env(state)
        raw = set_state(env, state, cfg)
        data = scene_data(raw)
        data.near_target_idxes = [state['target_lane']]
        for profile in ('oracle_force', 'uniform'):
            rows.append(dict(state, **solve(cfg, data, profile)))
        env.close()
        write_csv(out/'scan.csv', rows)
        print(f'[{index}/{len(state_rows)}] {state["state_id"]}', flush=True)
    paired = []
    for state in state_rows:
        part = [row for row in rows if row['state_id'] == state['state_id']]
        by_profile = {row['profile']: row for row in part}
        oracle, uniform = by_profile['oracle_force'], by_profile['uniform']
        paired.append(dict(state_id=state['state_id'],
            action_differs=(oracle['action'], oracle['target']) != (uniform['action'], uniform['target']),
            oracle_action=oracle['action'], uniform_action=uniform['action'],
            acceleration_difference=oracle['acceleration']-uniform['acceleration'],
            oracle_commit_score=oracle['commit_score'], uniform_commit_score=uniform['commit_score']))
    write_csv(out/'paired.csv', paired)
    result = dict(states=len(state_rows), runtime_errors=0,
        action_difference_states=sum(row['action_differs'] for row in paired),
        acceleration_difference_over_005=sum(abs(row['acceleration_difference']) > .05 for row in paired),
        oracle_keep_uniform_commit=sum(row['oracle_action'] == 'keep' and row['uniform_action'] == 'commit' for row in paired),
        algorithm_change=False,
        interpretation='Development-only decision-boundary evidence; closed-loop held-out validation required.')
    dump(out/'summary.json', result)
    dump(out/'completion.json', dict(states=len(state_rows), solves=len(rows), runtime_errors=0))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
