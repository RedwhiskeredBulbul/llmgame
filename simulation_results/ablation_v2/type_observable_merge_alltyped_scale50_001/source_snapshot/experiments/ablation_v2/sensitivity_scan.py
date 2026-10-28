"""Development-only multi-scenario sensitivity scan; controller is unchanged."""
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


PROFILES={
    'yield':[1.,0.,0.],
    'normal':[0.,1.,0.],
    'force':[0.,0.,1.],
    'uniform':[1/3,1/3,1/3],
}
RANGES={
    'merge':dict(ego_speed=(3.5,9.),relative_x=(-15.,10.),relative_speed=(-2.,3.)),
    'cutin':dict(ego_speed=(15.,25.),relative_x=(5.,30.),relative_speed=(-8.,4.)),
    'traffic':dict(ego_speed=(18.,30.),relative_x=(8.,45.),relative_speed=(-10.,3.)),
}
FORCED_ROUTE_RANGES={
    'cutin':dict(ego_speed=(15.,25.),relative_x=(-10.,20.),relative_speed=(-5.,5.)),
    'traffic':dict(ego_speed=(18.,30.),relative_x=(-15.,20.),relative_speed=(-8.,5.)),
}


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def states(per_scenario=12,seed=1200,ranges=None):
    rows=[]
    ranges=ranges or RANGES
    for scenario_index,scenario in enumerate(ranges):
        rng=np.random.default_rng(seed+scenario_index)
        scenario_ranges=ranges[scenario]
        # Stratified one-dimensional bins with independently permuted axes.
        axes={}
        for key,(low,high) in scenario_ranges.items():
            values=low+(np.arange(per_scenario)+rng.random(per_scenario))/per_scenario*(high-low)
            rng.shuffle(values)
            axes[key]=values
        for index in range(per_scenario):
            rows.append(dict(state_id=f'{scenario}-{index:02d}',scenario=scenario,
                seed=seed+100*scenario_index+index,difficulty=index%3,behaviour_index=index%3,
                target_lane=1,ego_speed=float(axes['ego_speed'][index]),relative_x=float(axes['relative_x'][index]),
                relative_speed=float(axes['relative_speed'][index])))
    return rows


def set_state(raw,state):
    raw.vehicle.speed=state['ego_speed']
    raw.vehicle.on_state_update()
    raw.key_vehicle.position[0]=raw.vehicle.position[0]+state['relative_x']
    raw.key_vehicle.speed=max(0.,state['ego_speed']+state['relative_speed'])
    raw.key_vehicle.on_state_update()
    if state['scenario']=='cutin':
        raw.cutin_time=1.5+(state['seed']%16)/10.
    if state.get('forced_route') and state['scenario']=='traffic':
        # Put the designated typed vehicle in the required adjacent target lane.
        raw.key_vehicle.position[1]=4.
        raw.key_vehicle.target_lane_index=('a','b',1)
        raw.key_vehicle.on_state_update()
        # Retain mixed traffic but prevent an accidental initial overlap with it.
        for vehicle in raw.road.vehicles:
            if vehicle is not raw.key_vehicle and vehicle is not raw.vehicle and int(vehicle.lane_index[2])==1:
                if abs(vehicle.position[0]-raw.key_vehicle.position[0])<10.:
                    vehicle.position[0]+=25.
                    vehicle.on_state_update()
    if state['scenario']=='merge':
        by_id={v.id:v for v in raw.road.vehicles}
        by_id[2].position[0]=raw.vehicle.position[0]+32.
        by_id[2].on_state_update()
        by_id[3].position[0]=raw.vehicle.position[0]+38.
        by_id[3].on_state_update()


def solve(cfg,data,profile,response_mode='interactive'):
    solver=ExperimentGameSolver(cfg['decision'],response_mode=response_mode)
    beliefs={vid:list(PROFILES['uniform']) for vid in data.others_info['ids']}
    if 1 in beliefs:
        beliefs[1]=list(PROFILES[profile])
    with open(os.devnull,'w',encoding='utf-8') as null,contextlib.redirect_stdout(null):
        _,acc,target=solver.decide(data,beliefs)
    scores={row['action_type']:row.get('score') for row in solver.branch_rows}
    return dict(profile=profile,response_mode=response_mode,action=solver.selected_action,target=target,
        acceleration=float(acc[0]),keep_score=scores.get('keep'),probe_score=scores.get('probe'),
        commit_score=scores.get('commit'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config/ablation_v2_pilot.json')
    p.add_argument('--output-dir',default='simulation_results/ablation_v2/sensitivity_scan_001')
    p.add_argument('--states-per-scenario',type=int,default=12)
    p.add_argument('--forced-route',action='store_true')
    args=p.parse_args()
    cfg=load_config(args.config)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    scan_ranges=FORCED_ROUTE_RANGES if args.forced_route else RANGES
    manifest=dict(git_commit=commit(),config=cfg,scan_seed=1200,
        ranges=scan_ranges,states_per_scenario=args.states_per_scenario,forced_route=args.forced_route,
        source_hashes=hashes,code_hash=canonical_hash(hashes),
        purpose='development sensitivity scan; not performance evidence',algorithm_change=False)
    dump(out/'manifest.json',manifest)
    for relative in hashes:
        dest=out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(relative,dest)
    state_rows=states(args.states_per_scenario,ranges=scan_ranges)
    for state in state_rows:
        state['forced_route']=args.forced_route
    dump(out/'states.json',state_rows)
    rows=[]
    for state in state_rows:
        env=make_env(state)
        raw=env.unwrapped
        set_state(raw,state)
        data=scene_data(raw)
        if args.forced_route:
            data.near_target_idxes=[state['target_lane']]
        for profile in PROFILES:
            rows.append(dict(state,**solve(cfg,data,profile)))
        rows.append(dict(state,**solve(cfg,data,'normal',response_mode='constant_velocity')))
        write_csv(out/'scan.csv',rows)
        print(f"[{len(rows)//5}/{len(state_rows)}] {state['state_id']}",flush=True)
        env.close()
    grouped={state['state_id']:[r for r in rows if r['state_id']==state['state_id']] for state in state_rows}
    summary=[]
    for state_id,part in grouped.items():
        by={(r['profile'],r['response_mode']):r for r in part}
        intent=[by[(name,'interactive')] for name in PROFILES]
        signatures={(r['action'],r['target']) for r in intent}
        accelerations=[r['acceleration'] for r in intent]
        normal=by[('normal','interactive')]
        cv=by[('normal','constant_velocity')]
        summary.append(dict(state_id=state_id,scenario=part[0]['scenario'],
            belief_action_changes=len(signatures)>1,
            belief_acceleration_spread=float(np.ptp(accelerations)),
            response_action_changes=(normal['action'],normal['target'])!=(cv['action'],cv['target']),
            response_acceleration_difference=abs(normal['acceleration']-cv['acceleration'])))
    write_csv(out/'sensitivity_summary.csv',summary)
    result={scenario:dict(states=sum(r['scenario']==scenario for r in summary),
        belief_action_changes=sum(r['scenario']==scenario and r['belief_action_changes'] for r in summary),
        belief_acceleration_changes=sum(r['scenario']==scenario and r['belief_acceleration_spread']>0.05 for r in summary),
        response_action_changes=sum(r['scenario']==scenario and r['response_action_changes'] for r in summary),
        response_acceleration_changes=sum(r['scenario']==scenario and r['response_acceleration_difference']>0.05 for r in summary))
        for scenario in scan_ranges}
    dump(out/'summary.json',dict(by_scenario=result,
        belief_sensitive_states=[r for r in summary if r['belief_action_changes'] or r['belief_acceleration_spread']>0.05],
        response_sensitive_states=[r for r in summary if r['response_action_changes'] or r['response_acceleration_difference']>0.05],
        interpretation='Development routing only. Closed-loop held-out validation required.'))
    dump(out/'completion.json',dict(states=len(state_rows),solves=len(rows),runtime_errors=0))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
