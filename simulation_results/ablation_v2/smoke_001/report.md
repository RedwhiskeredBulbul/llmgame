# 消融实验运行报告

已记录episode：12。
当前数据性质：短程链路检查，不能作任务成功率结论。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|cutin|oracle_type|1|0|0|0|0|993.6|
|cutin|oracle_without_interaction_response|1|0|0|0|0|474.5|
|cutin|oracle_without_probe|1|0|0|0|0|1040.7|
|cutin|uniform_belief|1|0|0|0|0|1067.0|
|merge|oracle_type|1|0|0|0|0|1297.9|
|merge|oracle_without_interaction_response|1|0|0|0|0|551.7|
|merge|oracle_without_probe|1|0|0|0|0|809.9|
|merge|uniform_belief|1|0|0|0|0|1360.9|
|traffic|oracle_type|1|0|0|0|0|1306.4|
|traffic|oracle_without_interaction_response|1|0|0|0|0|584.8|
|traffic|oracle_without_probe|1|0|0|0|0|1332.8|
|traffic|uniform_belief|1|0|0|0|0|1568.7|

配对差值及探索性bootstrap区间见`paired_comparisons.csv`。开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
