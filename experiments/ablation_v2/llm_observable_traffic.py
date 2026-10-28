"""Real-Qwen pilot in type-observable forced-lane mixed traffic."""
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
from experiments.ablation_v2.config import LLM_VARIANTS, canonical_hash, load_config, missing_credentials, source_hashes
from experiments.ablation_v2.llm_targeted_pilot import git_commit, llm_diagnostics
from experiments.ablation_v2.reporting import report


def cases(cfg):
    spec = cfg['observable_traffic']
    for index, seed in enumerate(cfg['seeds']):
        rng = np.random.default_rng(seed)
        behaviour = index % 3
        yield dict(case_id=f'llm-traffic-{seed}', scenario='traffic', seed=seed, difficulty=1,
            behaviour_index=behaviour,
            ego_speed_ms=float(rng.uniform(*spec['ego_speed_range_ms'])),
            relative_x_m=float(rng.uniform(*spec['rear_relative_x_range_m'])),
            key_speed_ms=float(rng.uniform(*spec['key_speed_ranges_ms'][str(behaviour)])),
            required_target_lane=spec['required_target_lane'], horizon_s=spec['horizon_s'],
            required_progress_m=spec['required_progress_m'])


def set_scene(env, case, cfg):
    raw = env.unwrapped
    raw.vehicle.speed = case['ego_speed_ms']
    raw.vehicle.on_state_update()
    key = raw.key_vehicle
    key.position[:] = [raw.vehicle.position[0]+case['relative_x_m'], 4.*case['required_target_lane']]
    key.speed = case['key_speed_ms']
    key.target_lane_index = ('a', 'b', case['required_target_lane'])
    key.on_state_update()
    clearance = cfg['observable_traffic']['background_clearance_m']
    for vehicle in raw.road.vehicles:
        if vehicle is raw.vehicle or vehicle is key:
            continue
        if int(vehicle.lane_index[2]) == case['required_target_lane'] and abs(vehicle.position[0]-key.position[0]) < clearance:
            vehicle.position[0] += 30.
            vehicle.on_state_update()
    return env


def analyze(out, cfg):
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((out/'episodes').glob('*/result.json'))]
    by_variant = {}
    for variant in cfg['variants']:
        part = [row for row in rows if row['variant'] == variant]
        by_variant[variant] = dict(episodes=len(part), completed=sum(row['status'] == 'completed' for row in part),
            collisions=sum(row['status'] == 'collision' for row in part),
            timeouts=sum(row['status'] == 'timeout' for row in part),
            runtime_errors=sum(row['status'] == 'runtime_error' for row in part),
            llm_requests=sum(row['llm_requests'] for row in part), llm_updates=sum(row['llm_updates'] for row in part),
            mean_belief_fallback_fraction=float(np.mean([row['belief_fallback_fraction'] for row in part])),
            mean_restricted_time_s=float(np.mean([row['restricted_task_time_s'] for row in part])),
            mean_min_clearance_m=float(np.mean([row['min_clearance_m'] for row in part if row['min_clearance_m'] is not None])))
    lookup = {(row['case_id'], row['variant']): row for row in rows}
    pairs = []
    for comparison in cfg['variants']:
        if comparison == 'full_model':
            continue
        for case in cases(cfg):
            full, other = lookup[(case['case_id'], 'full_model')], lookup[(case['case_id'], comparison)]
            pairs.append(dict(case_id=case['case_id'], behaviour_index=case['behaviour_index'], comparison=comparison,
                full_status=full['status'], comparison_status=other['status'],
                success_difference=full['success']-other['success'],
                collision_difference=full['collision']-other['collision'],
                time_difference_s=full['restricted_task_time_s']-other['restricted_task_time_s'],
                clearance_difference_m=(full['min_clearance_m']-other['min_clearance_m'])
                    if full['min_clearance_m'] is not None and other['min_clearance_m'] is not None else None))
    runner.write_csv(out/'llm_traffic_pairs.csv', pairs)
    effects = {}
    for comparison in cfg['variants']:
        if comparison == 'full_model':
            continue
        part = [row for row in pairs if row['comparison'] == comparison]
        effects[comparison] = dict(full_better_success=sum(row['success_difference'] > 0 for row in part),
            full_worse_success=sum(row['success_difference'] < 0 for row in part),
            full_fewer_collisions=sum(row['collision_difference'] < 0 for row in part),
            full_more_collisions=sum(row['collision_difference'] > 0 for row in part),
            mean_time_difference_s=float(np.mean([row['time_difference_s'] for row in part])),
            mean_clearance_difference_m=float(np.mean([row['clearance_difference_m'] for row in part])))
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
        algorithm_change=False, phase='type-observable LLM traffic pilot; not confirmatory evidence')
    runner.dump(out/'llm_traffic_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ablation_v2_llm_observable_traffic_pilot.json')
    parser.add_argument('--output-dir', default='simulation_results/ablation_v2/llm_observable_traffic_pilot_001')
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
    case_rows = list(cases(cfg))
    runner.dump(out/'cases.json', case_rows)
    original_env, original_scene = runner.make_env, runner.scene_data
    warm_cfg = copy.deepcopy(cfg)
    warm_cfg['runtime']['max_steps'] = 1
    first = case_rows[0]
    with patch.object(runner, 'make_env', lambda spec: set_scene(original_env(spec), first, cfg)):
        warm = runner.run_episode(warm_cfg, first, 'uniform_belief', out/'warmup', warmup=True)
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
        for case in case_rows:
            order = list(cfg['variants'])
            random.Random(case['seed']).shuffle(order)
            for variant in order:
                result_path = out/'episodes'/f"{case['case_id']}-{variant}"/'result.json'
                if result_path.exists() and cfg['runtime']['resume']:
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                else:
                    def make_env(spec, selected=copy.deepcopy(case)):
                        return set_scene(original_env(spec), selected, cfg)
                    def forced_scene(raw, cache=None):
                        data = original_scene(raw, cache)
                        data.near_target_idxes = [case['required_target_lane']]
                        return data
                    with patch.object(runner, 'make_env', make_env), patch.object(runner, 'scene_data', forced_scene):
                        result = runner.run_episode(cfg, case, variant, out, llm_client=probe._client)
                rows.append(result)
                runner.write_csv(out/'episodes.csv', rows)
                print(f"[{len(rows)}/{len(case_rows)*len(cfg['variants'])}] {case['case_id']} {variant}: "
                      f"{result['status']} requests={result['llm_requests']} updates={result['llm_updates']}", flush=True)
    finally:
        if probe._client:
            probe._client.close()
    runner.dump(out/'completion.json', dict(planned=len(case_rows)*len(cfg['variants']), finished=len(rows),
        runtime_errors=sum(row['status'] == 'runtime_error' for row in rows)))
    report(out)
    print(json.dumps(analyze(out, cfg), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
