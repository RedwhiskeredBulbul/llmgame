from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.ablation_v2.runner import dump,write_csv


def report(out):
    result_files=sorted((out/'episodes').glob('*/result.json'))
    episodes=[json.loads(p.read_text(encoding='utf-8')) for p in result_files]
    if not episodes:
        raise ValueError('No episode results')
    summary=[]
    for scenario in sorted({r['scenario'] for r in episodes}):
        for variant in sorted({r['variant'] for r in episodes}):
            rows=[r for r in episodes if r['scenario']==scenario and r['variant']==variant]
            if not rows:
                continue
            row=dict(scenario=scenario,variant=variant,episodes=len(rows),
                     completed=sum(r['status']=='completed' for r in rows),
                     collisions=sum(r['status']=='collision' for r in rows),
                     timeouts=sum(r['status']=='timeout' for r in rows),
                     runtime_errors=sum(r['status']=='runtime_error' for r in rows),
                     smoke_episodes=sum(r['status']=='smoke_horizon' for r in rows),
                     success_rate=sum(r['success'] for r in rows)/len(rows))
            for field in ['restricted_task_time_s','average_speed_ms','min_clearance_m','jerk_rms','control_p95_ms','deadline_miss_rate','probe_selected_steps','belief_fallback_fraction','llm_updates','solver_fallback_count']:
                values=[r[field] for r in rows if r.get(field) is not None]
                row[field]=float(np.mean(values)) if values else None
            summary.append(row)
    write_csv(out/'scenario_summary.csv',summary)
    paired=[]
    baseline='full_model' if any(r['variant']=='full_model' for r in episodes) else 'oracle_type'
    lookup={(r['case_id'],r['variant']):r for r in episodes}
    for variant in sorted({r['variant'] for r in episodes}-{baseline}):
        for metric in ['success','collision','restricted_task_time_s','min_clearance_m','jerk_rms','control_p95_ms']:
            differences=[]
            for r in episodes:
                other=lookup.get((r['case_id'],variant))
                if r['variant']==baseline and other and r.get(metric) is not None and other.get(metric) is not None:
                    differences.append(float(r[metric])-float(other[metric]))
            if not differences:
                continue
            # Resample whole case pairs, not individual time steps.
            rng=np.random.default_rng(9017)
            values=np.asarray(differences)
            bootstrap=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
            paired.append(dict(baseline=baseline,comparison=variant,metric=metric,n_pairs=len(values),
                               mean_difference=float(values.mean()),ci_low=float(np.percentile(bootstrap,2.5)),
                               ci_high=float(np.percentile(bootstrap,97.5))))
    write_csv(out/'paired_comparisons.csv',paired)
    smoke=any(r['status']=='smoke_horizon' for r in episodes)
    lines=['# 消融实验运行报告','',f'已记录episode：{len(episodes)}。',
           f'当前数据性质：{"短程链路检查，不能作任务成功率结论" if smoke else "开发/诊断闭环结果；是否正式以manifest为准"}。','',
           '| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        latency=f"{r['control_p95_ms']:.1f}" if r['control_p95_ms'] is not None else 'NA'
        lines.append(f"|{r['scenario']}|{r['variant']}|{r['episodes']}|{r['completed']}|{r['collisions']}|{r['timeouts']}|{r['runtime_errors']}|{latency}|")
    lines += ['', '配对差值及探索性bootstrap区间见`paired_comparisons.csv`。开发数据不用于最终显著性结论。',
              '理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    names=sorted({r['variant'] for r in episodes})
    for ax,metric,title in [(axes[0],'control_p95_ms','Episode P95 control latency (ms)'),
                            (axes[1],'min_clearance_m','Minimum geometric clearance (m)')]:
        vals=[[r[metric] for r in episodes if r['variant']==name and r.get(metric) is not None] for name in names]
        ax.boxplot(vals,tick_labels=names)
        ax.set_title(title)
        ax.tick_params(axis='x',rotation=35,labelsize=7)
    fig.tight_layout()
    fig.savefig(out/'diagnostic_metrics.png',dpi=160)
    plt.close(fig)
    dump(out/'summary.json',dict(episodes=len(episodes),summary=summary,exploratory_pairs=paired))
