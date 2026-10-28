# 消融实验运行报告

已记录episode：21。
当前数据性质：开发/诊断闭环结果；是否正式以manifest为准。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|cutin|full_model|1|1|0|0|0|101.6|
|cutin|hard_type|1|1|0|0|0|101.2|
|cutin|oracle_type|1|1|0|0|0|102.1|
|cutin|sync_llm|1|1|0|0|0|877.1|
|cutin|uniform_belief|1|1|0|0|0|101.5|
|cutin|without_interaction_response|1|1|0|0|0|58.1|
|cutin|without_probe|1|1|0|0|0|106.8|
|merge|full_model|1|1|0|0|0|48.1|
|merge|hard_type|1|1|0|0|0|50.7|
|merge|oracle_type|1|1|0|0|0|47.0|
|merge|sync_llm|1|1|0|0|0|915.0|
|merge|uniform_belief|1|1|0|0|0|47.2|
|merge|without_interaction_response|1|1|0|0|0|24.9|
|merge|without_probe|1|1|0|0|0|48.0|
|traffic|full_model|1|1|0|0|0|222.5|
|traffic|hard_type|1|1|0|0|0|213.7|
|traffic|oracle_type|1|1|0|0|0|221.7|
|traffic|sync_llm|1|1|0|0|0|964.9|
|traffic|uniform_belief|1|1|0|0|0|224.7|
|traffic|without_interaction_response|1|1|0|0|0|118.2|
|traffic|without_probe|1|1|0|0|0|225.0|

配对差值、按seed聚类bootstrap区间和配对符号随机化检验见`paired_comparisons.csv`。三个主指标的多比较使用Holm校正；开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
