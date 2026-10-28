# 消融实验运行报告

已记录episode：12。
当前数据性质：开发/诊断闭环结果；是否正式以manifest为准。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|cutin|oracle_type|1|1|0|0|0|97.4|
|cutin|oracle_without_interaction_response|1|1|0|0|0|53.0|
|cutin|oracle_without_probe|1|1|0|0|0|103.7|
|cutin|uniform_belief|1|1|0|0|0|98.4|
|merge|oracle_type|1|1|0|0|0|45.0|
|merge|oracle_without_interaction_response|1|1|0|0|0|24.3|
|merge|oracle_without_probe|1|1|0|0|0|44.8|
|merge|uniform_belief|1|1|0|0|0|44.3|
|traffic|oracle_type|1|1|0|0|0|226.9|
|traffic|oracle_without_interaction_response|1|1|0|0|0|115.9|
|traffic|oracle_without_probe|1|1|0|0|0|225.8|
|traffic|uniform_belief|1|1|0|0|0|225.6|

配对差值及探索性bootstrap区间见`paired_comparisons.csv`。开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
