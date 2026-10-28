# 消融实验运行报告

已记录episode：105。
当前数据性质：开发/诊断闭环结果；是否正式以manifest为准。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|cutin|full_model|5|5|0|0|0|89.8|
|cutin|hard_type|5|5|0|0|0|90.6|
|cutin|oracle_type|5|5|0|0|0|87.8|
|cutin|sync_llm|5|5|0|0|0|970.1|
|cutin|uniform_belief|5|5|0|0|0|86.9|
|cutin|without_interaction_response|5|5|0|0|0|55.1|
|cutin|without_probe|5|5|0|0|0|89.0|
|merge|full_model|5|5|0|0|0|44.8|
|merge|hard_type|5|5|0|0|0|43.0|
|merge|oracle_type|5|5|0|0|0|43.1|
|merge|sync_llm|5|5|0|0|0|913.9|
|merge|uniform_belief|5|5|0|0|0|41.9|
|merge|without_interaction_response|5|5|0|0|0|24.7|
|merge|without_probe|5|5|0|0|0|42.7|
|traffic|full_model|5|5|0|0|0|177.2|
|traffic|hard_type|5|5|0|0|0|186.2|
|traffic|oracle_type|5|5|0|0|0|175.7|
|traffic|sync_llm|5|5|0|0|0|1053.6|
|traffic|uniform_belief|5|5|0|0|0|178.2|
|traffic|without_interaction_response|5|5|0|0|0|107.4|
|traffic|without_probe|5|5|0|0|0|187.5|

配对差值、按seed聚类bootstrap区间和配对符号随机化检验见`paired_comparisons.csv`。三个主指标的多比较使用Holm校正；开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
