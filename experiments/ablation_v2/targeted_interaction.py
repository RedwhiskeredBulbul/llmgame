"""Held-out interaction-rich benchmark without changing the controller.

The six template states were fixed in mechanism_003.  This protocol keeps all
templates (including negative controls), adds seed-derived jitter, and crosses
them with balanced background behaviour labels.  It changes only initial scene
conditions; controller, beliefs, rewards, dynamics and success rules are reused.
"""
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
from experiments.ablation_v2.config import canonical_hash, load_config, source_hashes
from experiments.ablation_v2.reporting import report


def git_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def targeted_cases(cfg):
    spec=cfg['targeted_interaction']
    templates=spec['templates']
    jitter=spec['jitter']
    for index,seed in enumerate(cfg['seeds']):
        rng=np.random.default_rng(seed)
        template_index=index % len(templates)
        relative_x,dv,speed=templates[template_index]
        behaviour=index % 3
        yield dict(case_id=f'targeted-{seed}',scenario='merge',seed=seed,difficulty=1,
                   behaviour_index=behaviour,template_index=template_index,
                   relative_x_m=float(relative_x+rng.uniform(-jitter['relative_x_m'],jitter['relative_x_m'])),
                   relative_speed_ms=float(dv+rng.uniform(-jitter['relative_speed_ms'],jitter['relative_speed_ms'])),
                   ego_speed_ms=float(speed+rng.uniform(-jitter['ego_speed_ms'],jitter['ego_speed_ms'])))


def apply_case(env,case):
    raw=env.unwrapped
    raw.vehicle.speed=case['ego_speed_ms']
    raw.vehicle.on_state_update()
    raw.key_vehicle.position[0]=raw.vehicle.position[0]+case['relative_x_m']
    raw.key_vehicle.speed=max(0.,case['ego_speed_ms']+case['relative_speed_ms'])
    raw.key_vehicle.on_state_update()
    # Preserve the common traffic model; only move the existing lead vehicles
    # to fixed non-overlapping locations so the rear interaction is observable.
    by_id={v.id:v for v in raw.road.vehicles}
    if 2 in by_id:
        by_id[2].position[0]=raw.vehicle.position[0]+32.
        by_id[2].on_state_update()
    if 3 in by_id:
        by_id[3].position[0]=raw.vehicle.position[0]+38.
        by_id[3].on_state_update()
    return env


def analyze(out,cfg):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'episodes').glob('*/result.json'))]
    lookup={(r['case_id'],r['variant']):r for r in rows}
    comparisons=[]
    for comparison in ('uniform_belief','oracle_without_probe','oracle_without_interaction_response'):
        for case in targeted_cases(cfg):
            base=lookup[(case['case_id'],'oracle_type')]
            other=lookup[(case['case_id'],comparison)]
            comparisons.append(dict(case_id=case['case_id'],template_index=case['template_index'],
                behaviour_index=case['behaviour_index'],comparison=comparison,
                oracle_status=base['status'],comparison_status=other['status'],
                success_difference=base['success']-other['success'],
                collision_difference=base['collision']-other['collision'],
                restricted_time_difference_s=base['restricted_task_time_s']-other['restricted_task_time_s'],
                clearance_difference_m=(base['min_clearance_m']-other['min_clearance_m'])
                    if base['min_clearance_m'] is not None and other['min_clearance_m'] is not None else None,
                oracle_probe_steps=base['probe_selected_steps'],comparison_probe_steps=other['probe_selected_steps']))
    runner.write_csv(out/'targeted_paired_comparisons.csv',comparisons)
    by_variant={}
    for variant in cfg['variants']:
        selected=[r for r in rows if r['variant']==variant]
        by_variant[variant]=dict(episodes=len(selected),completed=sum(r['status']=='completed' for r in selected),
            collisions=sum(r['status']=='collision' for r in selected),
            route_end_failures=sum(r['status']=='offroad_or_route_end' for r in selected),
            runtime_errors=sum(r['status']=='runtime_error' for r in selected),
            probe_steps=sum(r['probe_selected_steps'] for r in selected),
            restricted_time_mean_s=float(np.mean([r['restricted_task_time_s'] for r in selected])))
    paired_by=defaultdict(list)
    for row in comparisons:
        paired_by[row['comparison']].append(row)
    effects={}
    for name,part in paired_by.items():
        values=np.asarray([r['restricted_time_difference_s'] for r in part])
        effects[name]=dict(oracle_better_success_pairs=sum(r['success_difference']>0 for r in part),
            oracle_worse_success_pairs=sum(r['success_difference']<0 for r in part),
            discordant_success_pairs=sum(r['success_difference']!=0 for r in part),
            mean_restricted_time_difference_s=float(values.mean()),
            oracle_fewer_collision_pairs=sum(r['collision_difference']<0 for r in part),
            oracle_more_collision_pairs=sum(r['collision_difference']>0 for r in part))
    initial_by_case=defaultdict(set)
    for row in rows:
        initial_by_case[row['case_id']].add(row['initial_hash'])
    gates=dict(complete=len(rows)==len(cfg['seeds'])*len(cfg['variants']),
        paired=all(len(v)==1 for v in initial_by_case.values()),
        no_runtime_errors=not any(r['status']=='runtime_error' for r in rows),
        probe_activated=by_variant['oracle_type']['probe_steps']>0,
        at_least_one_prespecified_ablation_differs=any(
            item['discordant_success_pairs']>0 for item in effects.values()))
    result=dict(protocol_hash=canonical_hash(cfg),git_commit=git_commit(),by_variant=by_variant,
                paired_effects=effects,gates=gates,all_gates_pass=all(gates.values()),
                interpretation='Held-out targeted scenario evidence; no controller or algorithm change.')
    runner.dump(out/'targeted_analysis.json',result)
    lines=['# 冻结近距离交互正式实验','',
        f"Git提交：`{result['git_commit']}`；协议哈希：`{result['protocol_hash']}`。",'',
        '本实验仅改变初始交通状态，不修改控制器、奖励、信念、车辆模型或成功条件。六个模板全部保留，并使用未参与开发的3000—3029种子添加预先规定的小扰动。','',
        '|变体|样本|完成|碰撞|道路终点失败|Probe周期|限制任务时间均值(s)|','|---|---:|---:|---:|---:|---:|---:|']
    for variant,value in by_variant.items():
        lines.append(f"|{variant}|{value['episodes']}|{value['completed']}|{value['collisions']}|{value['route_end_failures']}|{value['probe_steps']}|{value['restricted_time_mean_s']:.2f}|")
    lines+=['','## 配对效应','']
    for name,value in effects.items():
        lines.append(f"- Oracle vs {name}：成功改善/恶化={value['oracle_better_success_pairs']}/{value['oracle_worse_success_pairs']}；"
                     f"Oracle限制任务时间差={value['mean_restricted_time_difference_s']:.2f}s；"
                     f"少碰撞/多碰撞配对={value['oracle_fewer_collision_pairs']}/{value['oracle_more_collision_pairs']}。")
    lines+=['','## 放行检查','']+[f'- {key}: {value}' for key,value in gates.items()]
    lines+=['','该结果用于验证原算法在交互充分场景中的条件效应，不能替代自然交通分布中的总体性能结论。']
    (out/'targeted_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/ablation_v2_targeted_formal.json')
    p.add_argument('--output-dir',default='simulation_results/ablation_v2/targeted_formal_001')
    args=p.parse_args()
    cfg=load_config(args.config)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    manifest=dict(config=cfg,source_hashes=hashes,code_hash=canonical_hash(hashes),
        protocol_hash=canonical_hash(cfg),git_commit=git_commit(),phase='targeted_formal',
        data_source='fresh seeded fictional simulator states; no pre-existing dataset',
        algorithm_change=False,scene_change_only=True)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old['code_hash']!=manifest['code_hash'] or old['protocol_hash']!=manifest['protocol_hash']:
            raise RuntimeError('Changed code/protocol requires a new output directory')
    else:
        runner.dump(out/'manifest.json',manifest)
        for relative in hashes:
            dest=out/'source_snapshot'/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(relative,dest)
    cases=list(targeted_cases(cfg))
    runner.dump(out/'cases.json',cases)
    rows=[]
    original=runner.make_env
    for case in cases:
        order=list(cfg['variants'])
        random.Random(case['seed']).shuffle(order)
        for variant in order:
            path=out/'episodes'/f"{case['case_id']}-{variant}"/'result.json'
            if path.exists() and cfg['runtime']['resume']:
                result=json.loads(path.read_text(encoding='utf-8'))
            else:
                def make_env(spec,case=copy.deepcopy(case)):
                    return apply_case(original(spec),case)
                with patch.object(runner,'make_env',make_env):
                    result=runner.run_episode(cfg,case,variant,out)
            rows.append(result)
            runner.write_csv(out/'episodes.csv',rows)
            print(f"[{len(rows)}/{len(cases)*len(cfg['variants'])}] {case['case_id']} {variant}: "
                  f"{result['status']} probe={result['probe_selected_steps']}",flush=True)
    runner.dump(out/'completion.json',dict(planned=len(cases)*len(cfg['variants']),finished=len(rows),
        runtime_errors=sum(r['status']=='runtime_error' for r in rows)))
    report(out)
    print(json.dumps(analyze(out,cfg),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
