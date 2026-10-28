"""Complete the pilot audit and conditionally run the preregistered main matrix.

Can supervise an already running pilot; never promotes a failed readiness gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.ablation_v2.audit import audit
from experiments.ablation_v2.config import ROOT,canonical_hash
from experiments.ablation_v2.runner import dump,execute
from experiments.ablation_v2.reporting import report


def publication_report(out, results, formal_out=None):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    summaries=json.loads((out/'summary.json').read_text(encoding='utf-8'))['summary']
    gates=results['formal_readiness']
    lines=['# llmgame 消融实验与论文可用性报告','',
           '本报告由实际运行数据生成。开发预实验与独立测试的身份由manifest决定，不将链路检查算作任务成功。','',
           f"预实验：计划{results['planned']}个episode，已记录{results['recorded']}个；初始状态配对一致：{results['paired_initial_states']}。",
           f"运行异常数：{len(results['runtime_error_episodes'])}；Full收到并在控制周期取回的概率更新总数：{results['full_llm_updates']}。",
           f"Full在汇入场景中选择Probe的周期总数：{results['merge_full_probe_steps']}。",'',
           '## 1. 代码和方法版本','',
           f"协议：`{manifest['config']['protocol_version']}`。源码哈希：`{manifest['code_hash']}`。",
           '本轮为修正后的v2控制器：保留优化输出、周车响应先于自车优化、修复TTCP无冲突判断、隔离候选状态、统一0.1秒轨迹推进。',
           '所有变体共享相同自车优化、物理约束和候选评分。共享恒加速度问题采用同一CasADi目标上的一维网格与有界细化，不能称为原版本IPOPT性能复现。',
           '取消旧版巨额动作战术加分；新增的公共路线进度奖励写入配置。理想轨迹点执行，尚未验证转角动力学。','',
           '## 2. 模型配置与测速','',
           f"闭环实验模型：`{manifest['config']['llm']['model']}`。关闭思考、JSON输出、温度0；密钥仅从运行环境读取。",
           '独立测速全部使用现场生成的虚构轨迹。四模型初筛20次、两模型三车复测20次、固定快照验证3次，共43次。',
           '三车复测：qwen-flash中位数854.4 ms，qwen3.5-flash中位数1276.9 ms，两者均10/10解析有效。qwen-flash的含冷启动P95为2751.6 ms，并非尾延迟最低的保证。',
           'qwen-turbo初筛5次均429限流；不能据此推断其推理速度。固定快照3/3调用成功。',
           '[官方Flash模型说明](https://help.aliyun.com/zh/model-studio/qwen-flash)。测速原始记录见相邻model_benchmark目录。','',
           '## 3. 完整时域预实验结果','',
           '|场景|变体|样本|完成|碰撞|超时|异常|P95总时延均值(ms)|LLM更新均值|Probe周期均值|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        latency=f"{row['control_p95_ms']:.1f}" if row['control_p95_ms'] is not None else 'NA'
        lines.append(f"|{row['scenario']}|{row['variant']}|{row['episodes']}|{row['completed']}|{row['collisions']}|{row['timeouts']}|{row['runtime_errors']}|{latency}|{row['llm_updates']:.1f}|{row['probe_selected_steps']:.1f}|")
    lines += ['', '## 4. 机制证据','',
              'mechanism_003进行了6个固定状态×6类信念的36次干预。四个状态的候选评分随信念变化，一个状态的最终动作发生变化；候选顺序检查通过。',
              'Keep/Probe物理响应对照的9个位置/类型组合均观察到后车加速度变化；背景车辆使用现有车道重叠感知与IDM，不读取控制器变体。',
              '120 ms模拟请求的调度测试证明异步调用不阻塞控制线程；这是调度测试，不是真实网络准确率或安全收益证明。','',
              '## 5. 论文可用性判定','',
              '|实验|现阶段可支持的结论|不能支持的结论|',
              '|---|---|---|',
              '|同步/异步|可以报告请求等待如何进入控制线程，以及当前设置下测到的时延分布|仿真等待时暂停，不能推断同步必然导致更多碰撞|',
              '|信念干预|可以作机制分析，证明信息到评分及部分动作的数据路径有效|不能替代跨场景LLM驾驶性能提升|',
              '|软概率/硬类型|需要配对轨迹差异和有效信念使用率；没有差异也应保留|不能仅凭one-hot和概率数值不同就声称软概率更安全|',
              '|交互响应/恒速预测|可同时报告预测模块计算代价和任务表现|恒速组更快或轨迹相同不证明交互模型更优|',
              '|Probe|只有实际触发并有任务差异，才能评估其闭环收益|Probe未触发的结果不能证明有效或无效|',
              '|Oracle|诊断类型信息是否值得进一步改进及模型失配|不能称作必然的性能上界|',
              '|Flash测速|可以作为模型选择依据，记录任务、样本和网络端点|不能称全球最快，也不是识别准确率评测|','',
              f"Full相对其他组的轨迹审计：{results['trajectory_pairs']}对中有{results['identical_trajectory_pairs']}对在1e-7容差内一致。逐对记录见audit.json。"]
    if results['descriptive_prediction_accuracy'] is not None:
        lines += [f"Full有效返回的平滑预测诊断：{results['label_predictions']}条，准确率{100.*results['descriptive_prediction_accuracy']:.2f}%。同车/同场景重复观测相关，未覆盖失败或漏车的完整分母，因此不能写成正式分类器准确率。"]
    lines += ['', '## 6. 正式实验放行检查','']
    for key,value in gates.items():
        lines.append(f'- `{key}`：{value}。')
    if formal_out:
        lines += ['', f'已执行540个独立正式episode，结果目录：`{formal_out}`。正式统计见该目录report.md、scenario_summary.csv和paired_comparisons.csv。',
                  '预实验参数和正式种子严格分开；不把开发集的置信区间当成最终检验结果。']
    else:
        lines += ['', '尚未启动540个正式episode：预实验放行条件未全部满足。正式完成状态明确为未完成，未用模拟数据补齐。',
                  '优先根据未通过的条目修正任务触发或调度问题，再升级协议；不能为了出现正向结果删除不利样本。']
    lines += ['', '## 7. 适用边界与复核','',
              '本轮预实验使用5个开发种子，每个场景仅5个实例；不足以支持高可靠安全性、显著性或泛化结论。',
              '更早的oracle_pilot_001、llm_smoke_001、llm_pilot_001用于开发诊断，部分运行时间重叠，不合并到本轮正式统计。',
              '仿真按计算完成后推进，LLM结果的新鲜度依赖墙钟计算速度；报告必须同时展示场景年龄、墙钟年龄及均匀回退占比。',
              '聚合指标来自episodes.csv；配对轨迹来自各episode的steps.csv；概率、响应、约束和评分来自belief_events及branch_metrics.jsonl.gz。',
              '如修改算法、场景、指标或模型，必须建立新协议和输出目录，不覆盖本轮数据。']
    (out/'实验报告_论文可用性分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot-dir',required=True)
    p.add_argument('--formal-dir',default='simulation_results/ablation_v2/formal_001')
    args=p.parse_args()
    out=Path(args.pilot_dir)
    # This supervisor only reads progress; the pilot owns all writes until completion.
    while not ((out/'completion.json').exists() and (out/'summary.json').exists()):
        count=len(list((out/'episodes').glob('*/result.json')))
        print(f'Waiting for pilot completion: {count} finished episodes',flush=True)
        time.sleep(30.)
    report(out)
    results=audit(out)
    dump(out/'readiness_decision.json',results['formal_readiness'])
    if not all(results['formal_readiness'].values()):
        publication_report(out,results)
        dump(out/'pipeline_status.json',dict(status='pilot_complete_formal_not_ready',formal_complete=False,
             reason='One or more prespecified readiness checks failed; see readiness_decision.json'))
        print('Pilot complete. Formal phase not ready; evidence-based report written.',flush=True)
        return
    cfg=json.loads((out/'manifest.json').read_text(encoding='utf-8'))['config']
    cfg['phase']='frozen'
    cfg['protocol_version']='ablation-v2-frozen-20261016'
    cfg['seeds']=list(range(3000,3030))
    cfg['variants']=['full_model','uniform_belief','hard_type','without_interaction_response','without_probe','sync_llm']
    cfg['mechanism_gate_hash']=canonical_hash(json.loads((ROOT/'simulation_results/ablation_v2/mechanism_003/gates.json').read_text(encoding='utf-8')))
    frozen=ROOT/'config/ablation_v2_frozen.json'
    if frozen.exists():
        raise RuntimeError('Refusing to overwrite an existing frozen protocol')
    dump(frozen,cfg)
    formal=Path(args.formal_dir)
    execute(cfg,formal,phase='formal')
    report(formal)
    audit(formal)
    publication_report(out,results,formal)
    dump(out/'pipeline_status.json',dict(status='formal_complete',formal_complete=True,formal_dir=str(formal)))


if __name__=='__main__':
    main()
