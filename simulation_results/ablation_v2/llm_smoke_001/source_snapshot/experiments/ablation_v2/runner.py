from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np

from highway_env.envs.ablation_envs import ENV_IDS
from experiments.ablation_v2.belief_provider import BeliefProvider
from experiments.ablation_v2.config import canonical_hash, cases, missing_credentials, source_hashes
from experiments.ablation_v2.controller import ExperimentGameSolver, scene_data
from experiments.ablation_v2.metrics import safety, summarize_steps


def safe_json(value):
    if isinstance(value, dict):
        return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(safe_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_env(case):
    env = gym.make(ENV_IDS[case['scenario']], config=dict(kind=case['scenario'],
                   difficulty=case['difficulty'],behaviour_index=case['behaviour_index']))
    env.reset(seed=case['seed'])
    return env


def initial_state(env):
    return [dict(id=v.id, position=v.position.tolist(), heading=float(v.heading), speed=float(v.speed),
                 length=v.LENGTH,width=v.WIDTH,behaviour=getattr(v,'BEHAVIOR_TYPE','AV'),
                 target_speed=getattr(v,'target_speed',None)) for v in env.road.vehicles]


def build_action(trajectory, acceleration):
    formatted = np.column_stack((trajectory[:,0],trajectory[:,1],trajectory[:,3],trajectory[:,2],acceleration,
                    np.r_[0.,np.cumsum(np.linalg.norm(np.diff(trajectory[:,:2],axis=0),axis=1))]))
    # Ideal trajectory-point mode: steering is not integrated by this vehicle class.
    return {'trajectory':formatted,'action':np.column_stack((acceleration,np.zeros(len(acceleration)))).tolist()}


def run_episode(cfg, case, variant, output_dir, warmup=False):
    episode_id=f"{case['case_id']}-{variant}"
    start_wall=time.perf_counter()
    env=provider=None
    status='runtime_error'
    rows=[]
    completed_time=None
    error=None
    stable=0
    merged_x=None
    selected_probe=0
    solve_fallbacks=0
    horizon={'merge':40.,'cutin':20.,'traffic':50.}[case['scenario']]
    out=Path(output_dir)
    episode_dir=out/'episodes'/episode_id
    episode_dir.mkdir(parents=True,exist_ok=True)
    branch_path=episode_dir/'branch_metrics.jsonl.gz'
    initial_hash=None
    try:
        env=make_env(case)
        raw=env.unwrapped
        initial=initial_state(raw)
        initial_hash=canonical_hash(initial)
        dump(episode_dir/'initial_state.json',dict(case=case,initial_state=initial,initial_hash=initial_hash))
        response_mode='constant_velocity' if 'without_interaction_response' in variant else 'interactive'
        solver=ExperimentGameSolver(cfg['decision'], response_mode=response_mode,enable_probe='without_probe' not in variant)
        provider=BeliefProvider(variant,cfg['llm'],episode_id)
        lanes=None
        limit=cfg['runtime'].get('max_steps')
        steps=min(int(horizon/.1),int(limit)) if limit else int(horizon/.1)
        with gzip.open(branch_path,'wt',encoding='utf-8') as branch_stream, open(os.devnull,'w',encoding='utf-8') as null:
            for step in range(steps):
                started=time.perf_counter()
                data=scene_data(raw,lanes)
                lanes=data.lanes
                preprocess_ms=1000.*(time.perf_counter()-started)
                belief_start=time.perf_counter()
                beliefs,belief_info=provider.update(step,data,raw)
                belief_ms=1000.*(time.perf_counter()-belief_start)
                game_start=time.perf_counter()
                with contextlib.redirect_stdout(null):
                    traj,acc,target=solver.decide(data,beliefs)
                game_ms=1000.*(time.perf_counter()-game_start)
                action_start=time.perf_counter()
                action=build_action(traj,acc)
                action_ms=1000.*(time.perf_counter()-action_start)
                total_ms=1000.*(time.perf_counter()-started)
                before_speed=float(raw.vehicle.speed)
                before_t=raw.time
                _,_,terminated,truncated,_=env.step(action)
                state=safety(raw)
                collided=state['collision'] or (state['clearance_m'] is not None and state['clearance_m']<=1e-8)
                aa=(float(raw.vehicle.speed)-before_speed)/.1
                solve_fallbacks+=sum(r['status'].startswith('fallback') for r in solver.solve_diagnostics)
                selected_probe+=solver.selected_action=='probe'
                row=dict(episode_id=episode_id,step=step,sim_time_before=before_t,sim_time_after=raw.time,
                         x=float(raw.vehicle.position[0]),y=float(raw.vehicle.position[1]),heading=float(raw.vehicle.heading),
                         speed_before_ms=before_speed,speed_after_ms=float(raw.vehicle.speed),
                         a_command=float(acc[0]),a_executed=aa,action_type=solver.selected_action,target_lane=target,
                         preprocess_ms=preprocess_ms,belief_handling_ms=belief_ms,game_ms=game_ms,action_build_ms=action_ms,
                         decision_ms=belief_ms+game_ms+action_ms,control_total_ms=total_ms,
                         evaluated_branches=solver.evaluated_branches_count,**state,**belief_info)
                rows.append(row)
                branch_stream.write(json.dumps(safe_json(dict(step=step,beliefs=beliefs,branches=solver.branch_rows)),ensure_ascii=False,allow_nan=False)+'\n')
                if collided:
                    status='collision';break
                if not raw.vehicle.on_road or (raw.kind=='merge' and raw.vehicle.position[0]>raw.road_end and raw.vehicle.position[1]>2.):
                    status='offroad_or_route_end';break
                if raw.kind=='merge':
                    stable=stable+1 if abs(raw.vehicle.position[1])<.5 and abs(raw.vehicle.heading)<np.deg2rad(5) else 0
                    if stable>=10 and merged_x is None:
                        merged_x=float(raw.vehicle.position[0])
                    if merged_x is not None and stable>=10 and raw.vehicle.position[0]>=merged_x+20.:
                        status='completed';completed_time=raw.time;break
                elif raw.time>=horizon-1e-7:
                    enough_progress=raw.vehicle.position[0]-20. >= (100. if raw.kind=='cutin' else 250.)
                    status='completed' if enough_progress else 'timeout'
                    completed_time=raw.time if status=='completed' else None
                    break
                if truncated or terminated:
                    status='timeout';break
            else:
                status='smoke_horizon' if limit and limit<int(horizon/.1) else 'timeout'
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        dump(episode_dir/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        if provider:
            provider.close()
            dump(episode_dir/'belief_events.json',provider.events)
        if env:
            env.close()
    write_csv(episode_dir/'steps.csv',rows)
    result=dict(case,variant=variant,episode_id=episode_id,status=status,error=error,initial_hash=initial_hash,
                wall_duration_s=time.perf_counter()-start_wall,solver_fallback_count=solve_fallbacks,
                llm_requests=provider.request_count if provider else 0,
                llm_updates=sum(e['event']=='response' and bool(e.get('probabilities')) for e in provider.events) if provider else 0,
                belief_fallback_fraction=sum(r['source']=='uniform_fallback' for r in rows)/len(rows) if rows else 0.,
                **summarize_steps(rows,status,horizon,completed_time))
    dump(episode_dir/'result.json',result)
    return result


def execute(cfg, output_dir, phase='pilot', allow_llm=False):
    blocked=missing_credentials(cfg)
    if blocked:
        raise RuntimeError('Missing DASHSCOPE_API_KEY for '+','.join(blocked))
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    manifest=dict(config=cfg,source_hashes=hashes,code_hash=canonical_hash(hashes),protocol_hash=canonical_hash(cfg),
                  phase=phase,python=sys.version,platform=platform.platform(),machine=platform.processor(),
                  data_source='fresh fictional simulator trajectories; no existing datasets read',
                  llm_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1',
                  created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  limitations=['repaired controller differs from legacy September implementation',
                               'ideal trajectory-point execution; no steering dynamics validation'])
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old['code_hash']!=manifest['code_hash'] or old['protocol_hash']!=manifest['protocol_hash']:
            raise RuntimeError('Resume refuses changed code or protocol; use a new output directory')
    else:
        dump(out/'manifest.json',manifest)
        from experiments.ablation_v2.config import ROOT
        for relative in hashes:
            snapshot=out/'source_snapshot'/relative
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,snapshot)
    case_list=list(cases(cfg))
    dump(out/'cases.json',case_list)
    # Untimed warmup uses a distinct environment and never consumes test state.
    warm_cfg=copy_config(cfg)
    warm_cfg['runtime']['max_steps']=1
    warm=run_episode(warm_cfg,case_list[0],'uniform_belief',out/'warmup',warmup=True)
    if warm['status']=='runtime_error':
        raise RuntimeError('Warmup failed: '+str(warm['error']))
    results=[]
    for case in case_list:
        order=list(cfg['variants'])
        random.Random(case['seed']).shuffle(order)
        for variant in order:
            result_path=out/'episodes'/f"{case['case_id']}-{variant}"/'result.json'
            if result_path.exists() and cfg['runtime'].get('resume'):
                result=json.loads(result_path.read_text(encoding='utf-8'))
            else:
                result=run_episode(cfg,case,variant,out)
            results.append(result)
            write_csv(out/'episodes.csv',results)
            print(f"[{len(results)}/{len(case_list)*len(order)}] {case['case_id']} {variant}: {result['status']} steps={result['steps']} p95={result.get('control_p95_ms')} error={result['error'] or ''}",flush=True)
    dump(out/'completion.json',dict(planned=len(case_list)*len(cfg['variants']),finished=len(results),
         runtime_errors=sum(r['status']=='runtime_error' for r in results),smoke_only=bool(cfg['runtime'].get('max_steps'))))
    return results


def copy_config(config):
    return json.loads(json.dumps(config))
