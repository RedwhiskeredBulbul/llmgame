"""Audit complete runs and distinguish effect evidence from mere completion."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.ablation_v2.runner import dump


def audit(out):
    files=sorted((out/'episodes').glob('*/result.json'))
    results=[json.loads(p.read_text(encoding='utf-8')) for p in files]
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config']
    planned=len(cfg['scenarios'])*len(cfg['seeds'])*len(cfg['variants'])
    by_case=defaultdict(list)
    for r in results:
        by_case[r['case_id']].append(r)
    paired=all(len({r['initial_hash'] for r in rows})==1 for rows in by_case.values())
    errors=[r['episode_id'] for r in results if r['status']=='runtime_error']
    smoke=any(r['status']=='smoke_horizon' for r in results)
    full=[r for r in results if r['variant']=='full_model']
    actual_updates=sum(r.get('llm_updates',0) for r in full)
    merge_full=[r for r in full if r['scenario']=='merge']
    probe=sum(r['probe_selected_steps'] for r in merge_full)
    same_trajectory_pairs=[]
    fields=['x','y','speed_after_ms','a_executed']
    for case,rows in by_case.items():
        lookup={r['variant']:r for r in rows}
        if 'full_model' not in lookup:
            continue
        with (out/'episodes'/lookup['full_model']['episode_id']/'steps.csv').open(encoding='utf-8-sig') as f:
            base=list(csv.DictReader(f))
        for other in ('uniform_belief','hard_type','without_probe','oracle_type'):
            if other not in lookup:
                continue
            with (out/'episodes'/lookup[other]['episode_id']/'steps.csv').open(encoding='utf-8-sig') as f:
                compare=list(csv.DictReader(f))
            error=0.
            for a,b in zip(base,compare):
                error=max(error,max(abs(float(a[k])-float(b[k])) for k in fields))
            same_trajectory_pairs.append(dict(case_id=case,comparison=other,same_length=len(base)==len(compare),
                                               max_state_action_difference=error,
                                               identical=len(base)==len(compare) and error<1e-7))
    classification=[]
    label_index={'Cautious':0,'Normal':1,'Aggressive':2}
    for result in full:
        episode_dir=out/'episodes'/result['episode_id']
        initial=json.loads((episode_dir/'initial_state.json').read_text(encoding='utf-8'))['initial_state']
        truth={str(v['id']):label_index[v['behaviour']] for v in initial if v['behaviour'] in label_index}
        events=json.loads((episode_dir/'belief_events.json').read_text(encoding='utf-8'))
        for event in events:
            for vid,probs in (event.get('probabilities') or {}).items():
                if str(vid) in truth:
                    classification.append(dict(case_id=result['case_id'],vehicle_id=vid,truth=truth[str(vid)],
                                               predicted=int(np.argmax(probs)),returned_after_episode=event['event']=='response_after_episode'))
    summary=dict(planned=planned,recorded=len(results),all_planned_finished=len(results)==planned,
                 paired_initial_states=paired,runtime_error_episodes=errors,smoke_only=smoke,
                 full_llm_updates=actual_updates,merge_full_probe_steps=probe,
                 identical_trajectory_pairs=sum(r['identical'] for r in same_trajectory_pairs),
                 trajectory_pairs=len(same_trajectory_pairs),
                 label_predictions=len(classification),
                 descriptive_prediction_accuracy=float(np.mean([r['truth']==r['predicted'] for r in classification])) if classification else None,
                 classification_note='Smoothed predictions, correlated requests/vehicles, only completed Full episodes; diagnostic, not held-out classifier accuracy',
                 formal_readiness=dict(complete=len(results)==planned,paired=paired,no_runtime_errors=not errors,
                                      full_receives_predictions=actual_updates>0,probe_triggered_in_merge=probe>0),
                 trajectory_comparisons=same_trajectory_pairs)
    dump(out/'audit.json',summary)
    dump(out/'classification_diagnostic.json',classification)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run_dir')
    args=p.parse_args()
    result=audit(Path(args.run_dir))
    print(json.dumps({k:v for k,v in result.items() if k!='trajectory_comparisons'},indent=2))
