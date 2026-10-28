# 消融实验运行报告

已记录episode：21。
当前数据性质：短程链路检查，不能作任务成功率结论。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|cutin|full_model|1|0|0|0|0|105.9|
|cutin|hard_type|1|0|0|0|0|99.7|
|cutin|oracle_type|1|0|0|0|0|100.4|
|cutin|sync_llm|1|0|0|0|0|501.5|
|cutin|uniform_belief|1|0|0|0|0|101.3|
|cutin|without_interaction_response|1|0|0|0|0|57.3|
|cutin|without_probe|1|0|0|0|0|100.8|
|merge|full_model|1|0|0|0|0|47.6|
|merge|hard_type|1|0|0|0|0|83.2|
|merge|oracle_type|1|0|0|0|0|48.0|
|merge|sync_llm|1|0|0|0|0|501.5|
|merge|uniform_belief|1|0|0|0|0|47.6|
|merge|without_interaction_response|1|0|0|0|0|25.3|
|merge|without_probe|1|0|0|0|0|48.6|
|traffic|full_model|1|0|0|0|0|52.7|
|traffic|hard_type|1|0|0|0|0|54.0|
|traffic|oracle_type|1|0|0|0|0|48.1|
|traffic|sync_llm|1|0|0|0|0|588.8|
|traffic|uniform_belief|1|0|0|0|0|50.5|
|traffic|without_interaction_response|1|0|0|0|0|30.7|
|traffic|without_probe|1|0|0|0|0|55.3|

配对差值及探索性bootstrap区间见`paired_comparisons.csv`。开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
