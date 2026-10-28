# 冻结近距离交互正式实验

Git提交：`e6e3e68f436106eefc445b102d2d97a1ff429f9f`；协议哈希：`d9d759ccdecd2d15b4971aa74f3c0f8fa9864b5539b483f28cfd68e5e6b4b035`。

本实验仅改变初始交通状态，不修改控制器、奖励、信念、车辆模型或成功条件。六个模板全部保留，并使用未参与开发的3000—3029种子添加预先规定的小扰动。

|变体|样本|完成|碰撞|道路终点失败|Probe周期|限制任务时间均值(s)|
|---|---:|---:|---:|---:|---:|---:|
|oracle_type|30|30|0|0|868|8.59|
|uniform_belief|30|30|0|0|826|8.55|
|oracle_without_probe|30|24|6|0|0|15.44|
|oracle_without_interaction_response|30|30|0|0|898|8.63|

## 配对效应

- Oracle vs uniform_belief：成功改善/恶化=0/0；Oracle限制任务时间差=0.04s；少碰撞/多碰撞配对=0/0。
- Oracle vs oracle_without_probe：成功改善/恶化=6/0；Oracle限制任务时间差=-6.85s；少碰撞/多碰撞配对=6/0。
- Oracle vs oracle_without_interaction_response：成功改善/恶化=0/0；Oracle限制任务时间差=-0.04s；少碰撞/多碰撞配对=0/0。

## 放行检查

- complete: True
- paired: True
- no_runtime_errors: True
- probe_activated: True
- at_least_one_prespecified_ablation_differs: True

该结果用于验证原算法在交互充分场景中的条件效应，不能替代自然交通分布中的总体性能结论。
