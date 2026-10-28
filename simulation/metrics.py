"""Episode metrics used by the simulation entry point."""

import pandas as pd


def calculate_episode_metrics(all_records, ego_id=888, lane_width=4.0):
    """
    计算单个 Episode 的核心评价指标
    :param all_records: main.py 中记录的字典列表
    :param ego_id: 自车在记录中的 id 标识（请根据你 main.py 里的实际设定制）
    :param lane_width: 车道宽度，用于判断是否在同一车道（默认 4.0 米）
    :return: 包含各项指标的字典
    """
    df = pd.DataFrame(all_records)

    if df.empty:
        return None

    # 分离自车数据和其他车辆数据
    ego_df = df[df["id"] == ego_id].copy()
    other_df = df[df["id"] != ego_id].copy()

    if ego_df.empty:
        print("警告: 未能在记录中找到自车数据！请检查 ego_id。")
        return None

    # ==========================================
    # 1. 平均速度 (Average Speed)
    # ==========================================
    # 假设记录的速度单位为 m/s，直接计算均值
    avg_speed = ego_df["v"].mean()

    # ==========================================
    # 2. 加速度变异性 (Acceleration Variability)
    # ==========================================
    # 计算整个过程中加速度的方差 (Variance)
    # 也可以使用标准差 (std)，论文中通常指 Variance
    accel_variability = ego_df["a"].var()

    # ==========================================
    # 3. 最小碰撞时间 (Min TTC)
    # ==========================================
    min_ttc = float("inf")

    # 按时间步遍历，计算每一步的 TTC
    unique_times = ego_df["time"].unique()

    for t in unique_times:
        # 获取当前时刻自车状态
        ego_state = ego_df[ego_df["time"] == t].iloc[0]
        ego_x = ego_state["x"]
        ego_y = ego_state["y"]
        ego_v = ego_state["v"]

        # 获取当前时刻其他所有车辆状态
        others_at_t = other_df[other_df["time"] == t]

        for _, other in others_at_t.iterrows():
            other_x = other["x"]
            other_y = other["y"]
            other_v = other["v"]

            # 判断是否在“同一车道”或“碰撞冲突域”内
            # 简单启发式：横向距离小于半个车道宽，且前车在自车前方
            if abs(other_y - ego_y) < (lane_width / 2.0) and other_x > ego_x:

                # 计算相对距离和相对速度
                dx = other_x - ego_x - 5.0  # 减去车身长度(约5米)作为安全缓冲
                dv = ego_v - other_v

                # 只有当自车速度大于前车速度（正在靠近）时，TTC 才有意义
                if dv > 0.1 and dx > 0:
                    ttc = dx / dv
                    if ttc < min_ttc:
                        min_ttc = ttc

    # 如果全程没有冲突车辆，给一个安全的上限值（例如 50 秒）
    if min_ttc == float("inf"):
        min_ttc = 50.0

    metrics = {
        "Average_Speed_ms": avg_speed,
        "Accel_Variability": accel_variability,
        "Min_TTC_s": min_ttc,
    }

    return metrics
