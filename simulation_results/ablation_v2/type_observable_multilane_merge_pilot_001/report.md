# 消融实验运行报告

已记录episode：36。
当前数据性质：开发/诊断闭环结果；是否正式以manifest为准。

| 场景 | 变体 | 样本 | 完成 | 碰撞 | 超时 | 异常 | P95总时延均值(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
|traffic|full_model|9|9|0|0|0|50.6|
|traffic|oracle_type|9|9|0|0|0|49.0|
|traffic|uniform_belief|9|6|3|0|0|29.5|
|traffic|without_probe|9|6|3|0|0|31.7|

配对差值、按seed聚类bootstrap区间和配对符号随机化检验见`paired_comparisons.csv`。三个主指标的多比较使用Holm校正；开发数据不用于最终显著性结论。
理想轨迹执行、仿真暂停计时，不代表车辆动力学跟踪或延迟造成的真实交通风险。
