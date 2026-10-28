# 验证记录

整理日期：2026-10-28。

- `python -m pytest tests -q`：36 项通过，覆盖博弈、类型评估、提示词、四环境注册、种子复现、道路映射、动作执行、渲染和完整离线模型流程。
- 整理前后分别使用 seed 13，在四个场景运行 3 步。自车和背景车辆的记录数、ID、位置、速度、航向和加速度匹配，数值比较使用 `rtol=1e-9`、`atol=1e-9`。详情见 `structure_verification.json`。
- `main.py --help` 与六个实验入口的 `--help` 均正常退出。
- 模型与运行器的未定义名称检查通过，代码中已消除旧模块导入。
- 原归档结果、PPT 与视频保留。此前校验的 8 个嵌入 MP4、2 个实车 MP4 和 PPT 的 SHA-256 记录见 `../experiment_results/media_integrity.json`。

| 场景 | 整理前后匹配的状态记录数 |
| --- | --- |
| CutinEnv | 129 |
| HighwayEnv | 153 |
| AdversarialMergeEnv | 21 |
| MergeEnv | 15 |

本次测试使用现有 Python 3.13.9 环境。安装说明建议 Python 3.11，以配合原项目的 NumPy 版本约束。测试环境的 NumPy 为 2.3.5、Gymnasium 为 1.3.0、Pygame 为 2.6.1、CasADi 为 3.7.2、Numba 为 0.62.1、SciPy 为 1.16.3。

验证范围是离线流程、短程数值一致性和现有回归测试，不代表所有长程工况均已验证。没有调用在线 LLM、重跑正式统计实验、部署实车或重新录制视频。MoviePy 录制功能仍需安装可选依赖后验证。
