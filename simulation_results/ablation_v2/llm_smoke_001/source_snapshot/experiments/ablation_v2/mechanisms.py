from __future__ import annotations

import contextlib
import copy
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.ablation_v2.belief_provider import BeliefProvider
from experiments.ablation_v2.config import canonical_hash,source_hashes
from experiments.ablation_v2.controller import ExperimentGameSolver,scene_data
from experiments.ablation_v2.runner import make_env,dump,write_csv


def run_mechanisms(cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    dump(out/'manifest.json',dict(config=cfg,source_hashes=hashes,code_hash=canonical_hash(hashes),
                                 data_source='fixed freshly generated simulator states; no API calls'))
    rows=[]
    diagnostics=[]
    probabilities=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1/3]*3,[.6,.3,.1],[.1,.3,.6]]
    conditions=[(-12.,1.5,6.),(-6.,2.,6.),(0.,2.,6.),(8.,-1.,6.),(-6.,1.,4.),(-12.,2.,8.)]
    order_errors=[]
    with open(os.devnull,'w',encoding='utf-8') as null:
        for index,(relative_x,dv,speed) in enumerate(conditions):
            env=make_env(dict(scenario='merge',seed=1000+index,difficulty=1,behaviour_index=index%3))
            raw=env.unwrapped
            raw.vehicle.speed=speed
            raw.key_vehicle.position[0]=raw.vehicle.position[0]+relative_x
            raw.key_vehicle.speed=speed+dv
            for belief_index,probs in enumerate(probabilities):
                solver=ExperimentGameSolver(cfg['decision'])
                data=scene_data(raw)
                beliefs={vid:list(probs) for vid in data.others_info['ids']}
                with contextlib.redirect_stdout(null):
                    traj,acc,target=solver.decide(data,beliefs)
                scores={r['action_type']:r.get('score') for r in solver.branch_rows}
                responses=[np.asarray(r['response_accelerations'])[1:] for r in solver.solve_diagnostics]
                type_spread=max((float(np.ptp(a,axis=1).max()) for a in responses if a.size),default=0.)
                rows.append(dict(state=index,belief_index=belief_index,probability=probs,selected=solver.selected_action,
                                 a=float(acc[0]),target=target,type_response_spread=type_spread,**scores))
                diagnostics.append(dict(state=index,belief_index=belief_index,branches=copy.deepcopy(solver.branch_rows)))
                write_csv(out/'belief_interventions.csv',rows)
                print(f"mechanism state={index} belief={belief_index} action={solver.selected_action} a={acc[0]:.4f}",flush=True)
                if belief_index==3:
                    reverse=ExperimentGameSolver(cfg['decision'])
                    with contextlib.redirect_stdout(null):
                        rtraj,racc,rtarget=reverse.decide(data,beliefs,reverse_candidates=True)
                    diff=float(np.max(np.abs(racc-acc)))
                    rscores={r['action_type']:r.get('score') for r in reverse.branch_rows}
                    score_diff=max((abs(scores[k]-rscores[k]) for k in scores if scores[k] is not None and rscores[k] is not None),default=0.)
                    order_errors.append(dict(state=index,action_error=diff,score_error=score_diff,same_target=target==rtarget))
            env.close()
    write_csv(out/'belief_interventions.csv',rows)
    dump(out/'branch_diagnostics.json',diagnostics)
    dump(out/'candidate_order_checks.json',order_errors)
    # Existing physical lane-overlap perception, with all controller labels absent.
    probe=[]
    for behaviour in range(3):
        for relative_x in (-12.,-6.,-2.):
            for y in (4.,2.8,1.8):
                env=make_env(dict(scenario='merge',seed=1000,difficulty=1,behaviour_index=behaviour))
                raw=env.unwrapped
                raw.vehicle.position[:]=[20.,y]
                raw.vehicle.on_state_update()
                raw.key_vehicle.position[0]=20.+relative_x
                raw.key_vehicle.speed=7.5
                raw.key_vehicle.on_state_update()
                raw.road.act()
                probe.append(dict(behaviour=behaviour,relative_x=relative_x,y=y,
                                  rear_acceleration=float(raw.key_vehicle.action['acceleration'])))
                env.close()
    write_csv(out/'probe_response.csv',probe)
    probe_changed=0
    for behaviour in range(3):
        for rx in (-12.,-6.,-2.):
            part=[r for r in probe if r['behaviour']==behaviour and r['relative_x']==rx]
            probe_changed+=abs(part[0]['rear_acceleration']-part[1]['rear_acceleration'])>1e-5
    latency=[]
    with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'mock-only-not-a-real-credential'}):
        for variant in ('full_model','sync_llm'):
            provider=BeliefProvider(variant,dict(cfg['llm'],history_len=1,interval_steps=1),'mock_latency')
            def mocked(history,sim,wall):
                time.sleep(.12)
                return dict(probabilities={1:[.2,.5,.3]},observation_sim_time=sim,observation_wall=wall,latency_ms=120.)
            provider._request=mocked
            data=SimpleNamespace(ego_state=np.zeros(6),others_info={'ids':[1]},curr_lane_idx=0)
            started=time.perf_counter()
            provider.update(0,data,SimpleNamespace(time=0.))
            latency.append(dict(variant=variant,control_block_ms=(time.perf_counter()-started)*1000.,mock_delay_ms=120.))
            provider.close()
    dump(out/'mock_latency.json',latency)
    changed_score_states=[]
    changed_action_states=[]
    for state in range(len(conditions)):
        subset=[r for r in rows if r['state']==state]
        scores=[r.get('commit') for r in subset if r.get('commit') is not None]
        if len(scores)>1 and np.ptp(scores)>1e-4:
            changed_score_states.append(state)
        if np.ptp([r['a'] for r in subset])>1e-4 or len({(r['selected'],r['target']) for r in subset})>1:
            changed_action_states.append(state)
    gates=dict(type_responses_differ=any(r['type_response_spread']>1e-5 for r in rows),
               candidate_scores_depend_on_belief=bool(changed_score_states),
               action_changes_observed=bool(changed_action_states),
               order_independent=all(r['action_error']<1e-4 and r['score_error']<1e-4 and r['same_target'] for r in order_errors),
               probe_physical_response=bool(probe_changed),
               async_nonblocking=latency[0]['control_block_ms']<60 and latency[1]['control_block_ms']>=110)
    summary=dict(gates=gates,changed_score_states=changed_score_states,changed_action_states=changed_action_states,
                 probe_response_changed_pairs=int(probe_changed),probe_pairs=9,
                 formal_gate_passed=all(gates.values()),
                 interpretation='Mechanism evidence only; not closed-loop or LLM performance evidence.')
    dump(out/'gates.json',summary)
    print(summary,flush=True)
    return summary
