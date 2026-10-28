# 消融实验运行报告

已记录episode：30。
当前数据性质：开发/诊断闭环结果；是否正式以manifest为准。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|merge|full_model|6|6|0|0|0|89.9|
|merge|hard_type|6|6|0|0|0|87.5|
|merge|oracle_type|6|6|0|0|0|100.4|
|merge|uniform_belief|6|6|0|0|0|90.8|
|merge|without_probe|6|5|1|0|0|65.3|

配对差值、按seed聚类bootstrap区间和配对符号随机化检验见`paired_comparisons.csv`。三个主指标的多比较使用Holm校正；开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
