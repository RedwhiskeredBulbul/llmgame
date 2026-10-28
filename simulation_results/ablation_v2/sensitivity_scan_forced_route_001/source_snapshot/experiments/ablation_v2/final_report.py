"""Rebuild the human-readable evidence report from completed experiment logs."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.ablation_v2.audit import audit
from experiments.ablation_v2.reporting import report
from experiments.ablation_v2.runner import dump, write_csv


BASE=Path('simulation_results/ablation_v2')
NAMES=['pilot_stratified_001','interaction_diagnostic_001',
       'interaction_diagnostic_002','probe_fixed_merge_pilot_001']


def results(out):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'episodes').glob('*/result.json'))]


def table(headers,rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+[
        '|'+'|'.join(str(x) for x in row)+'|' for row in rows]


def main():
    runs={}
    for name in NAMES:
        out=BASE/name
        if not (out/'completion.json').exists():
            raise RuntimeError('Refusing to issue a final report for incomplete run: '+name)
        report(out)
        runs[name]=results(out)
    audits={name:audit(BASE/name) for name in ('pilot_stratified_001','probe_fixed_merge_pilot_001')}
    env={'python':platform.python_version(),'platform':platform.platform(),
         'packages':{p:importlib.metadata.version(p) for p in
                     ['numpy','gymnasium','casadi','scipy','matplotlib','openai','numba']}}
    out=BASE/'evidence_report_001'
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'environment.json',env)
    summaries={name:json.loads((BASE/name/'summary.json').read_text(encoding='utf-8'))['summary'] for name in NAMES}
    follow=[]
    for name in ('interaction_diagnostic_001','interaction_diagnostic_002'):
        for row in summaries[name]:
            follow.append(dict(run=name,**row))
    write_csv(out/'interaction_summary.csv',follow)
    event_summary=[]
    for name,episodes in runs.items():
        events=[]
        for ep in episodes:
            events+=json.loads((BASE/name/'episodes'/ep['episode_id']/'belief_events.json').read_text(encoding='utf-8'))
        lat=[e['latency_ms'] for e in events if e.get('latency_ms') is not None and not e.get('error')]
        usage=[e.get('usage') or {} for e in events]
        event_summary.append(dict(run=name,requests=sum(r['llm_requests'] for r in episodes),
            logged_responses=len(events),error_responses=sum(bool(e.get('error')) for e in events),
            late_after_episode=sum(e.get('event')=='response_after_episode' for e in events),
            total_tokens=sum(u.get('total_tokens',0) or 0 for u in usage),
            median_request_latency_ms=float(np.median(lat)) if lat else None,
            p95_request_latency_ms=float(np.percentile(lat,95)) if lat else None,
            summed_episode_wall_s=sum(r['wall_duration_s'] for r in episodes)))
    dump(out/'request_accounting.json',event_summary)
    paired=[]
    for name in ('interaction_diagnostic_001','interaction_diagnostic_002'):
        lookup={(r['case_id'],r['variant']):r for r in runs[name]}
        for r in runs[name]:
            if r['variant']!='oracle_type':
                continue
            for other in ('uniform_belief','oracle_without_probe'):
                q=lookup[r['case_id'],other]
                paired.append(dict(run=name,case_id=r['case_id'],comparison=other,
                    oracle_status=r['status'],other_status=q['status'],
                    success_difference=r['success']-q['success'],
                    restricted_time_difference=r['restricted_task_time_s']-q['restricted_task_time_s'],
                    oracle_probe_steps=r['probe_selected_steps']))
    write_csv(out/'interaction_pairs.csv',paired)
    mech=json.loads((BASE/'mechanism_004/gates.json').read_text(encoding='utf-8'))
    sweep=json.loads((BASE/'latency_sweep_001/summary.json').read_text(encoding='utf-8'))
    a=audits['pilot_stratified_001']
    fixed=audits['probe_fixed_merge_pilot_001']
    lines=['# llmgame 消融实验实施与论文有效性报告','',
        '## 1. 结论与完成范围','',
        '本报告根据实际保存的终态、逐步轨迹、API返回与候选评分生成。未使用PPO基线；未用合成结果代替真实API调用。',
        '已完成原始105回合开发预实验、修复前后各54回合近距离交互诊断，以及修复后35回合汇入复核，共248个完整闭环episode。机制测试、API测速和模拟延迟测试另计，不能相加为闭环样本数。',
        '**原方案的540回合独立正式测试尚未完成，也未冒充已完成。** 下述数据均为开发/机制证据；发现Probe几何实现问题后重新验证，不把两种源码版本混合为同一统计总体。',
        '目前最有依据的论文内容是：异步调度降低请求周期的阻塞、类型信念确实影响部分候选/动作，以及Probe在特定近距离状态下的条件性作用。尚无充分证据支持“LLM软概率普遍提高闭环安全/效率”或“整体方法全面优于所有消融组”。','',
        '## 2. 实施内容与方法边界','',
        '完成三类确定性种子场景、七组对照、独立候选状态、统一0.1秒执行、任务进度判据、碰撞/几何净间距、有效TTC、加速度/jerk、全周期耗时及请求审计。失败保留在分母，失败任务时间按场景时限惩罚。',
        '公共修复包括周车尺寸身份错位、概率检查、优化动作被上界覆盖、HV/AV响应顺序、TTCP正冲突位置判断和停止车辆不能倒退等。所有v2组共用；这些修复不是Full独享的优势。',
        'v2控制器使用同一CasADi目标的一维网格+有界局部细化，替换原IPOPT求解路径，不能声称复现原版IPOPT性能或保证全局最优。去除旧巨额战术加分，公共路线进度奖励2、切换惩罚0.1。',
        '最大3轮、加速度变化阈值0.05仅为工程停止条件，不是纳什均衡证明。配置100ms不构成硬实时保证：scalar_grid分支没有100ms强制中断，总周期包含多个候选及多轮求解。',
        '车辆采用理想轨迹点执行，不验证Stanley转角动力学；仿真在计算/API等待期间暂停。因此同步等待造成的墙钟代价不能转换成真实交通风险结论。','',
        '## 3. 原始三场景预实验（修复Probe之前）','',
        '配置：5个开发种子×3场景×7变体=105；种子1000—1004。每组每场景仅5个实例，难度与类型交叉不完整，不具备强分层或安全性证明能力。',
        'F=LLM软概率异步，U=均匀，H=硬类型，C=恒速响应，P=去Probe，S=同步，O=Oracle标签。Oracle只知道类型而非未来，且模型可能失配，不能保证是性能上界。','']
    lines+=table(['场景','变体','完成/样本','碰撞','时限失败','P95周期均值ms','100ms超期率均值'],[
        [r['scenario'],r['variant'],f"{r['completed']}/{r['episodes']}",r['collisions'],r['timeouts'],
         f"{r['control_p95_ms']:.1f}",f"{100*r['deadline_miss_rate']:.1f}%"] for r in summaries['pilot_stratified_001']])
    lines+=['',f"初始状态配对一致：{a['paired_initial_states']}；异常：{len(a['runtime_error_episodes'])}。F与U/H/P/O的{a['trajectory_pairs']}对轨迹中，{a['identical_trajectory_pairs']}对在x、y、速度及实际加速度的1e-7容差内一致。",
        'P95列是每个episode内部P95的跨episode均值，不是汇总全部step后的P95。配对bootstrap按跨场景共用seed聚类，仅5个seed簇；开发数据的p值和区间不应作为正式显著性。',
        '全部成功或轨迹相同是重要负结果，不证明模块等价，也不证明模块有效。常规汇入快速Commit、切入和流量场景的状态约束可能压过类型差异。','',
        '### LLM覆盖与回退','']
    lines+=table(['场景','F已取回更新均值','F均匀回退周期占比均值'],[
        [r['scenario'],f"{r['llm_updates']:.1f}",f"{100*r['belief_fallback_fraction']:.1f}%"]
        for r in summaries['pilot_stratified_001'] if r['variant']=='full_model'])
    lines+=['','已取回更新不等于每个车辆的预测均被动作使用；回退比例是周期source标记，可能低估个别车辆缺失。场景年龄阈值2秒，历史20步，间隔10步，EMA=0.7。',
        '低速汇入很快结束，而20步历史后才能发起首个请求；计算快于模拟时间时，返回可能过期。因此不能把F≈U直接归因于“LLM分类无用”。平滑后的相关重复预测只能作诊断，不能作为独立测试分类准确率。',
        '高密度交通中异步控制也可能超过100ms；异步解决请求阻塞，不等于解决博弈计算成本。多个开发进程存在时间重叠（包括本轮补诊断），这些墙钟数据应标为开发环境实测，不能当作隔离硬件的正式性能基准。','',
        '## 4. Probe几何缺陷、修复及复核','',
        '发现_get_target_path先将目标线移动到0.3车道宽，再由_get_sigmoid_path把相对横向距离乘0.3。车道中心从4m到0m时，旧版首次生成的Probe末端为3.64m，位移0.36m，而非目标2.8m/位移1.2m。重复规划也会改变其过渡过程。',
        '修复后v2只应用一次比例；新增测试在自车y=4.0/3.4/2.9m时均保持绝对目标y=2.8m且拼接连续。旧版默认控制路径保留，不静默改写历史实验。',
        '原mechanism_003手动将自车置于y=2.8m，虽说明物理响应存在，但没有验证实际路径生成器；不能据此证明原Probe实现正确。mechanism_004改用生成器实际目标，并通过几何回归。',
        f"修复后原5种子汇入复核35/35已结束；F选择Probe周期总数：{fixed['merge_full_probe_steps']}；异常：{len(fixed['runtime_error_episodes'])}。",'']
    lines+=table(['变体','完成/样本','碰撞','Probe周期均值'],[
        [r['variant'],f"{r['completed']}/{r['episodes']}",r['collisions'],f"{r['probe_selected_steps']:.1f}"]
        for r in summaries['probe_fixed_merge_pilot_001']])
    lines+=['','修复不会自动保证原场景需要Probe；未触发时只能作为负对照。不得通过降低安全阈值、读取变体标签让行，或删除失败实例制造优势。','',
        '## 5. 近距离交互闭环诊断（两版各54回合）','',
        '采用mechanism_003预先定义的6个状态，交叉3类后车×U/O/O去Probe。六组(后车相对纵向m,后车相对速度m/s,自车速度m/s)：(-12,1.5,6)、(-6,2,6)、(0,2,6)、(8,-1,6)、(-6,1,4)、(-12,2,8)。',
        '这些是针对性开发干预，不是随机抽取的18个独立交通分布样本。结果可说明条件性机制，不可替代30个正式种子的总体收益或真实LLM效果；两次运行不合并为108个独立样本。','']
    lines+=table(['版本','变体','完成/18','碰撞','道路/终点失败','Probe总周期'],[
        [name,r['variant'],f"{sum(q['status']=='completed' for q in episodes if q['variant']==r['variant'])}/18",
         sum(q['status']=='collision' for q in episodes if q['variant']==r['variant']),
         sum(q['status']=='offroad_or_route_end' for q in episodes if q['variant']==r['variant']),
         sum(q['probe_selected_steps'] for q in episodes if q['variant']==r['variant'])]
        for name in ('interaction_diagnostic_001','interaction_diagnostic_002')
        for episodes in [runs[name]] for r in summaries[name]])
    lines+=['','修复后O与O去Probe逐状态对照：','']
    rows=[r for r in paired if r['run']=='interaction_diagnostic_002' and r['comparison']=='oracle_without_probe']
    lines+=table(['状态/类型','O结果','O去Probe结果','O Probe周期','限制任务时间差O−去Probe(s)'],[
        [r['case_id'],r['oracle_status'],r['other_status'],r['oracle_probe_steps'],f"{r['restricted_time_difference']:.2f}"] for r in rows])
    gain=sum(r['success_difference']>0 for r in rows)
    loss=sum(r['success_difference']<0 for r in rows)
    lines+=['',f'修复后18对中，保留Probe使完成状态改善{gain}对、恶化{loss}对，其余完成状态相同。应同时报告代价与失败，不以单个有利状态证明普遍更安全。',
        '这组对照固定Oracle，仅删除Probe，较直接地检验候选作用；但仍受理想执行、确定性IDM、候选长时碰撞罚分及有限状态集限制。U/O差异则检验类型信息，不检验LLM是否准确。','',
        '## 6. 机制与调度证据','',
        f"修复后的36次信念干预中，候选评分变化状态：{mech['changed_score_states']}；最终动作变化状态：{mech['changed_action_states']}。候选反序评估一致，后车Keep/实际Probe物理响应变化{mech['probe_response_changed_pairs']}/{mech['probe_pairs']}对。",
        '只能写成“在预设状态下，类型条件响应和概率输入可传导到评分/动作”，不能写成“LLM正确理解所有车辆意图”或“求得真实纳什均衡”。',
        f"完整模拟延迟扫描：{sweep['trials']}次，0/0.5/1/2/4秒各3次×同步/异步；异步控制调用最大阻塞{sweep['max_async_control_block_ms']:.2f}ms。模拟结果只用于调度逻辑，不是API推理速度。",'']
    lines+=table(['注入延迟(s)','异步阻塞中位数ms','同步阻塞中位数ms'],[
        [delay,*[f"{np.median([r['control_block_ms'] for r in sweep['rows'] if r['delay_s']==delay and r['variant']==variant]):.2f}"
                  for variant in ('full_model','sync_llm')]] for delay in sweep['delays_s']])
    lines+=['','## 7. 千问模型、调用与复现','',
        '闭环固定qwen-flash-2025-07-28，关闭思考、温度0、JSON输出、超时10秒、不重试。密钥仅从运行环境读取，报告和源码不包含密钥。',
        '独立测速为现场生成的虚构轨迹：四模型初筛20次、两模型三车复测20次、快照检查3次。三车复测Flash中位数854.4ms，qwen3.5-flash为1276.9ms，均10/10有效；Flash含冷启动P95为2751.6ms。qwen-turbo初筛5次均429，不能比较其成功推理速度。',
        '选择依据是本机、本端点、本提示词成功请求的中位数，不是“平台所有模型的绝对最快”。价格未核实，不伪造账单；以下token仅覆盖四组闭环已记录返回，不含测速、warmup或其他早期开发运行。','']
    lines+=table(['数据集','请求数','记录返回','错误返回','回合结束后返回','token数','累计回合墙钟s'],[
        [r['run'],r['requests'],r['logged_responses'],r['error_responses'],r['late_after_episode'],r['total_tokens'],f"{r['summed_episode_wall_s']:.1f}"] for r in event_summary])
    lines+=['',f"实际环境：Python {env['python']}；"+'，'.join(f'{k} {v}' for k,v in env['packages'].items())+'。',
        '实际依赖与原requirements版本不完全一致；没有为凑版本强行重装。源码快照及SHA-256记录在各数据集manifest/source_snapshot中，原105回合必须使用其快照，而非修复后工作区。','',
        '## 8. 哪些实验可以用于论文','']
    lines+=table(['实验/主张','目前证据等级','可以写','不能写/补充条件'],[
        ['异步 vs 同步','机制充分；性能为开发测量','请求延迟不直接阻塞异步控制线程','不能推导真实风险下降；需隔离硬件复测及延迟注入闭环'],
        ['类型信念干预','机制证据','输入概率可改变部分候选评分与动作','不能推导LLM准确或总体安全收益'],
        ['Probe消融','条件性开发证据','列出近距离状态配对收益与失败','不能泛化到所有汇入；需独立随机种子和场景'],
        ['Full vs U','尚无普遍正向证据','如实报告负结果、覆盖和过期情况','不能称LLM模块已经验证有效'],
        ['软概率 vs 硬类型','不足','给出当前轨迹差异/无差异','独立API调用还引入响应差异，需固定同一输出对照'],
        ['交互预测 vs 恒速','性能和成本诊断','报告共同任务下计算成本与配对轨迹','恒速更快不是交互预测更有效的证据'],
        ['Oracle','归因诊断','说明类型价值与模型失配','不称必然性能上界或LLM能力'],
        ['Flash测速','工程选型附录','限定本任务本端点的实测选择','不称全球最快或分类能力提升']])
    lines+=['','建议当前论文表述：“类型条件博弈和部分横向探入在预设交互状态下可改变决策；异步推理实现了请求等待与控制线程的分离。现有开发闭环尚不足以确认LLM类型估计带来总体驾驶收益。”',
        '不建议当前写法：“所有模块都显著有效”“零碰撞证明安全”“LLM全面优于均匀信念”“异步消除了100ms超期”“Probe在所有场景均有效”。','',
        '## 9. 原方案阶段验收与未完成项','',
        'P0/P1：版本记录及公共修复完成；P2：三场景与任务判据完成，Probe几何补修复；P3：机制和近距离闭环诊断完成；P4：105回合预实验及35回合修复复核完成。',
        'P5/P6：未正式冻结、未执行540回合独立测试。原始自动放行未覆盖Probe几何正确性，现补充了回归；不得用旧版放行报告代替新版验收。',
        f"修复后汇入验收：{json.dumps(fixed['formal_readiness'],ensure_ascii=False)}。即使其他条件通过，原场景Probe未触发也无法评价Probe效益。",
        '进入正式测试前应冻结：已验证近距离场景的采样范围与独立随机扰动；每车有效信念覆盖与过期统计；同输出软/硬比较；完整控制周期预算与超期回退；源码/提示词/环境及任务指标。',
        '不得将3000—3029测试种子用于继续调参；不得因开发结果不显著而删组或只保留Probe有利状态。若缩减论文主张，应明确预注册“调度机制/条件性交互”而非继续宣称全部LLM模块提升。','',
        '## 10. 结果索引与复现入口','',
        '- 原始预实验：`simulation_results/ablation_v2/pilot_stratified_001/`。',
        '- 几何修复后复核：`simulation_results/ablation_v2/probe_fixed_merge_pilot_001/`。',
        '- 近距离两版：`interaction_diagnostic_001/`、`interaction_diagnostic_002/`，均在同一父目录。',
        '- 最终机制：`mechanism_004/`；模拟延迟：`latency_sweep_001/`。',
        '- 每轮`episodes/*/result.json`是终态，`steps.csv`是逐步状态，`belief_events.json`是API返回，`branch_metrics.jsonl.gz`是类型信念/候选求解。',
        '- 本目录`interaction_pairs.csv`提供18状态逐对证据；`request_accounting.json`提供调用统计；`environment.json`提供实际依赖。','',
        '重建报告：`python -X utf8 -m experiments.ablation_v2.final_report`。',
        '回归检查：`python -X utf8 -m unittest discover -s tests -v`。当前修复已通过21项测试，但单元测试不替代独立正式评测。','']
    text='\n'.join(lines)
    (out/'实验报告_论文可用性与有效性证据.md').write_text(text,encoding='utf-8')
    docs=Path('docs')
    (docs/'消融实验报告_论文可用性与有效性证据.md').write_text(text,encoding='utf-8')
    dump(out/'completion.json',dict(report_complete=True,closed_loop_episodes=sum(map(len,runs.values())),
         formal_540_complete=False,runs={k:len(v) for k,v in runs.items()}))
    print('Evidence report written. Formal 540 explicitly NOT completed.',flush=True)


if __name__=='__main__':
    main()
