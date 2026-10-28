"""Held-out forced-lane mixed-traffic ablation; experiment setup only."""
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

from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import canonical_hash,load_config,source_hashes
from experiments.ablation_v2.reporting import report


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def cases(cfg):
    spec=cfg['forced_traffic']
    for index,seed in enumerate(cfg['seeds']):
        rng=np.random.default_rng(seed)
        yield dict(case_id=f'forced-traffic-{seed}',scenario='traffic',seed=seed,
            difficulty=spec['difficulty'],behaviour_index=index%3,
            ego_speed_ms=float(rng.uniform(*spec['ego_speed_range_ms'])),
            relative_x_m=float(rng.uniform(*spec['rear_relative_x_range_m'])),
            relative_speed_ms=float(rng.uniform(*spec['rear_relative_speed_range_ms'])),
            required_target_lane=spec['required_target_lane'],horizon_s=spec['horizon_s'],
            required_progress_m=spec['required_progress_m'])


def set_scene(env,case,cfg):
    raw=env.unwrapped
    raw.vehicle.speed=case['ego_speed_ms']
    raw.vehicle.on_state_update()
    key=raw.key_vehicle
    key.position[:]=[raw.vehicle.position[0]+case['relative_x_m'],4.*case['required_target_lane']]
    key.speed=max(0.,case['ego_speed_ms']+case['relative_speed_ms'])
    key.target_lane_index=('a','b',case['required_target_lane'])
    key.on_state_update()
    clearance=cfg['forced_traffic']['background_clearance_m']
    for vehicle in raw.road.vehicles:
        if vehicle is raw.vehicle or vehicle is key:
            continue
        if int(vehicle.lane_index[2])==case['required_target_lane'] and abs(vehicle.position[0]-key.position[0])<clearance:
            vehicle.position[0]+=30.
            vehicle.on_state_update()
    return env


def analyze(out,cfg):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'episodes').glob('*/result.json'))]
    by_variant={}
    for variant in cfg['variants']:
        part=[r for r in rows if r['variant']==variant]
        by_variant[variant]=dict(episodes=len(part),completed=sum(r['status']=='completed' for r in part),
            collisions=sum(r['status']=='collision' for r in part),timeouts=sum(r['status']=='timeout' for r in part),
            runtime_errors=sum(r['status']=='runtime_error' for r in part),
            mean_restricted_time_s=float(np.mean([r['restricted_task_time_s'] for r in part])),
            mean_min_clearance_m=float(np.mean([r['min_clearance_m'] for r in part if r['min_clearance_m'] is not None])))
    lookup={(r['case_id'],r['variant']):r for r in rows}
    pairs=[]
    for comparison in ('uniform_belief','oracle_without_interaction_response'):
        for case in cases(cfg):
            base=lookup[(case['case_id'],'oracle_type')]
            other=lookup[(case['case_id'],comparison)]
            pairs.append(dict(case_id=case['case_id'],behaviour_index=case['behaviour_index'],comparison=comparison,
                oracle_status=base['status'],comparison_status=other['status'],
                success_difference=base['success']-other['success'],collision_difference=base['collision']-other['collision'],
                time_difference_s=base['restricted_task_time_s']-other['restricted_task_time_s'],
                clearance_difference_m=(base['min_clearance_m']-other['min_clearance_m'])
                    if base['min_clearance_m'] is not None and other['min_clearance_m'] is not None else None))
    runner.write_csv(out/'forced_traffic_pairs.csv',pairs)
    effects={}
    for comparison in ('uniform_belief','oracle_without_interaction_response'):
        part=[r for r in pairs if r['comparison']==comparison]
        effects[comparison]=dict(oracle_better_success=sum(r['success_difference']>0 for r in part),
            oracle_worse_success=sum(r['success_difference']<0 for r in part),
            oracle_fewer_collisions=sum(r['collision_difference']<0 for r in part),
            oracle_more_collisions=sum(r['collision_difference']>0 for r in part),
            mean_time_difference_s=float(np.mean([r['time_difference_s'] for r in part])),
            mean_clearance_difference_m=float(np.mean([r['clearance_difference_m'] for r in part if r['clearance_difference_m'] is not None])))
    initial=defaultdict(set)
    for row in rows:initial[row['case_id']].add(row['initial_hash'])
    gates=dict(complete=len(rows)==len(cfg['seeds'])*len(cfg['variants']),
        paired=all(len(v)==1 for v in initial.values()),
        no_runtime_errors=not any(r['status']=='runtime_error' for r in rows),
        some_closed_loop_difference=any(v['oracle_better_success'] or v['oracle_worse_success'] or
            v['oracle_fewer_collisions'] or v['oracle_more_collisions'] for v in effects.values()))
    result=dict(git_commit=commit(),protocol_hash=canonical_hash(cfg),by_variant=by_variant,
                paired_effects=effects,gates=gates,all_gates_pass=all(gates.values()),algorithm_change=False)
    runner.dump(out/'forced_traffic_analysis.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/ablation_v2_forced_traffic_formal.json')
    p.add_argument('--output-dir',default='simulation_results/ablation_v2/forced_traffic_formal_001')
    args=p.parse_args()
    cfg=load_config(args.config)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    manifest=dict(config=cfg,git_commit=commit(),source_hashes=hashes,code_hash=canonical_hash(hashes),
        protocol_hash=canonical_hash(cfg),algorithm_change=False,scene_change_only=True,
        data_source='fresh seeded fictional simulator states; no pre-existing dataset')
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old['code_hash']!=manifest['code_hash'] or old['protocol_hash']!=manifest['protocol_hash']:
            raise RuntimeError('Changed code/protocol requires a new output directory')
    else:
        runner.dump(out/'manifest.json',manifest)
        for relative in hashes:
            dest=out/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(relative,dest)
    all_cases=list(cases(cfg));runner.dump(out/'cases.json',all_cases)
    original_env,original_scene=runner.make_env,runner.scene_data
    rows=[]
    for case in all_cases:
        order=list(cfg['variants']);random.Random(case['seed']).shuffle(order)
        for variant in order:
            result_path=out/'episodes'/f"{case['case_id']}-{variant}"/'result.json'
            if result_path.exists() and cfg['runtime']['resume']:
                result=json.loads(result_path.read_text(encoding='utf-8'))
            else:
                def make_env(spec,case=copy.deepcopy(case)):
                    return set_scene(original_env(spec),case,cfg)
                def forced_scene(raw,cache=None,case=case):
                    data=original_scene(raw,cache)
                    data.near_target_idxes=[case['required_target_lane']]
                    return data
                with patch.object(runner,'make_env',make_env),patch.object(runner,'scene_data',forced_scene):
                    result=runner.run_episode(cfg,case,variant,out)
            rows.append(result);runner.write_csv(out/'episodes.csv',rows)
            print(f"[{len(rows)}/{len(all_cases)*len(cfg['variants'])}] {case['case_id']} {variant}: {result['status']}",flush=True)
    runner.dump(out/'completion.json',dict(planned=len(all_cases)*len(cfg['variants']),finished=len(rows),
        runtime_errors=sum(r['status']=='runtime_error' for r in rows)))
    report(out)
    print(json.dumps(analyze(out,cfg),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
