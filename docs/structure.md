# 代码结构与阅读顺序

## 主流程

1. `main.py` 解析运行参数，调用运行器并保存结果。
2. `simulation/runner.py` 组织闭环，不实现模型数学计算。
3. `model/perception/env_adapter.py` 转换状态，维护四环境道路映射与参考线。
4. `model/perception/filter.py` 筛选交互车辆并设置限速。
5. `model/reasoning/worker.py` 异步推断车辆行为类型，输出类型概率。
6. `model/game/solver.py` 结合信念、风格、安全约束及收益函数，求解自车轨迹。风格参数在同目录 `style_params.py`。
7. `simulation/actions.py` 转换轨迹格式，调用 `model/control/stanley.py` 计算转角。
8. `highway_env` 执行动作，`simulation/metrics.py` 计算运行指标。

`model/geometry/collision.py` 提供碰撞检测，`model/settings.py` 只保留当前模型引用的全局参数。

## 目录边界

| 目录 | 职责 |
| --- | --- |
| `model/` | 仿真和实验共享的核心算法，不依赖实验包 |
| `simulation/` | 单次仿真、动作适配与输出指标 |
| `highway_env/` | 环境、道路、车辆、动作、观察与渲染 |
| `experiments/ablation.py` | 通用消融协议 |
| `experiments/ablation_v2/` | 专项实验、报告生成和独立合成场景 |
| `experiments/vehicle_types/` | 类型采集、评估与原型拟合 |
| `configs/` | 实验配置 |
| `experiment_results/`、`simulation_results/` | 实验材料，不参与主模型导入 |

在 `llmgame_curated` 内使用 `python -m experiments.<模块>` 运行实验，避免直接运行深层脚本造成包导入失败。

## 删除范围

删除依据是主入口调用链、传递依赖和保留实验的引用，没有单凭短程覆盖率删除博弈或安全制动分支。

- 删除整个未使用的 `Optimizer/`，包括 MPC、iLQR 优化器。
- 删除 `P_planner.py`、`mpc_planner.py`、`coor_global_frenet.py`。
- 跟踪控制器仅保留初始化、Stanley 控制和最近路径点计算，删除未调用的规划、绕行和曲线生成方法。
- 删除 ROS 消息收发、数据处理、标记生成、桥接和旧通用工具模块。
- 删除筛选模块中未使用的 MPC 筛选、交通灯处理和插值函数。
- 删除两个旧 MPC 仿真车辆类和仅服务旧流程的三种动作入口，消除对旧规划器的隐式导入。
- 删除未引用的优化参数、废弃导入、固定机器路径、注释掉的替代实现及入口中的无效计时和重复代码。

博弈求解器的方法经调用关系检查均有用途，保留数学逻辑。LLM 提示词版本被现有实验使用，予以保留。仿真库的公共道路、车辆和渲染设施保留，避免破坏动态配置及基础接口。

## 主要路径迁移

| 原路径 | 新路径 |
| --- | --- |
| `Game/Game_solver.py` | `model/game/solver.py` |
| `Game/style_params.py` | `model/game/style_params.py` |
| `LLMreasoning/llmreason.py` | `model/reasoning/worker.py` |
| `Planner/bezier_stanley_planner.py` | `model/control/stanley.py` |
| `utils/Filter.py` | `model/perception/filter.py` |
| `utils/_from_env.py` | `model/perception/env_adapter.py` |
| `utils/pathcollision_detect.py` | `model/geometry/collision.py` |
| `config/params.py` | `model/settings.py` |
| `config/*.json` | `configs/*.json` |
| 根目录类型评估脚本 | `experiments/vehicle_types/` |
| `ablation_study.py` | `experiments/ablation.py` |

导入、默认配置路径、测试和实验源码哈希路径同步更新。旧 Python 路径不保留兼容空壳。
