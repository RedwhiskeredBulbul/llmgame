import numpy as np

from model.settings import *

ros_log = False

if ros_log:
    import rospy
else:
    pass

direction_dict = {0: "直行", 1: "左拐", 2: "右拐"}

flash_state_dict = {
    0: "常红",
    1: "常黄",
    2: "常绿",
    3: "常黑",
    4: "绿闪",
    5: "默认",
    6: "未知",
}


class InteractionFilter:
    def __init__(self):
        self.new_others_info = None
        self.other_style_dict = {}  # 他车风格辨识结果存储
        pass

    def init_plan_traj_by_path(self, ego_state, plan_path, dt=0.1):
        if sum(ego_state) == 0 or len(plan_path) == 0:
            self.plan_traj = np.array([[0, 0], [0, 0]], dtype=np.float64)
            return

        n = 80  # 8 seconds at 10 Hz
        x, y, h, v = ego_state[:4]
        v = v + 6  # 防止速度过低路线无法向前延长
        plan_path = np.array(plan_path)  # ensure numpy array

        # Step 1: 计算路径的累计距离
        diffs = np.diff(plan_path, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate([[0], np.cumsum(seg_lengths)])  # shape: (M,)

        # Step 2: 找到最近点及其累计距离 s0
        distances_to_ego = np.linalg.norm(plan_path - np.array([x, y]), axis=1)
        start_idx = np.argmin(distances_to_ego)
        s0 = cum_dists[start_idx]

        # Step 3: 构建目标累计距离序列（从 s0 开始，匀速前进）
        if v < 1e-3:  # 静止或极低速：保持在最近点
            traj = np.tile(plan_path[start_idx], (n, 1))
        else:
            timesteps = np.arange(1, n + 1) * dt  # [0.1, 0.2, ..., 8.0]
            s_targets = s0 + v * timesteps  # shape: (n,)

            # 路径最大距离
            s_max = cum_dists[-1]

            traj = np.zeros((n, 2))
            for i, s_target in enumerate(s_targets):
                if s_target >= s_max:
                    traj[i] = plan_path[-1]
                else:
                    # 找到 s_target 所在的线段 [j, j+1]
                    j = np.searchsorted(cum_dists, s_target, side="right") - 1
                    j = np.clip(j, 0, len(plan_path) - 2)

                    # 线性插值
                    ds = s_target - cum_dists[j]
                    segment_length = cum_dists[j + 1] - cum_dists[j]
                    if segment_length < 1e-6:
                        ratio = 0.0
                    else:
                        ratio = ds / segment_length
                    traj[i] = plan_path[j] + ratio * (plan_path[j + 1] - plan_path[j])

        self.plan_traj = traj

    def get_close_vehicles_game(
        self, others_info, dist_th=10.0, dt_ego=0.1, dt_others=0.5, max_vehicles=5
    ):
        """
        从他车预测轨迹中筛选出与自车在 8 秒内距离小于 dist_th 的车辆，最多返回 max_vehicles 辆。

        筛选逻辑如下：
        - 时间范围限定为 t ∈ (1, 8] 秒；
        - 将他车轨迹（时间步长 dt_others）与自车规划轨迹（时间步长 dt_ego）按时间对齐；
        - 在每个有效时间点计算他车与自车之间的欧氏距离；
        - 若某时刻距离 < dist_th，则该车被视为候选；
        - 车辆按“首次满足距离条件的时间”排序（越早优先），同一时刻内按距离由近到远排序；
        - 每辆车仅添加一次，最多返回 max_vehicles 辆。

        参数：
            others_info (dict): 包含他车信息，键包括：
                - 'states': (N, 5) 当前状态 [x, y, heading, v, a]
                - 'pred':   (N, T_pred, 5) 未来轨迹预测
                - 'ids':    (N,) 车辆唯一ID
                - 'types':  (N,) 车辆类型（如 'car', 'truck', 'cone' 等）
            dist_th (float): 距离阈值（米），默认 10.0
            dt_ego (float): 自车轨迹时间步长（秒），默认 0.1
            dt_others (float): 他车预测轨迹时间步长（秒），默认 0.5
            max_vehicles (int): 最多返回的车辆数，默认 5

        返回：
            dict or None: 筛选后的他车信息字典（结构同输入），若无符合条件车辆则各字段为 None。
        """

        other_states = others_info["states"]  # (N, 5)
        others_pred = others_info["pred"]  # (N, T_pred, 5)
        other_ids = others_info["ids"]
        other_types = others_info["types"]

        N = len(other_ids)
        if N == 0 or max_vehicles == 0:
            return {"states": None, "pred": None, "ids": None, "types": None}

        filtered_idxs = []
        t_others = np.arange(others_pred.shape[1]) * dt_others  # (T_pred,)
        valid_mask = (t_others > 1) & (t_others <= 8)  # 从第1秒开始，防止后面车被优先筛
        if not np.any(valid_mask):
            return {"states": None, "pred": None, "ids": None, "types": None}

        valid_t = t_others[valid_mask]  # (T_valid,)
        valid_others = others_pred[:, valid_mask, :]  # (N, T_valid, 5)

        # 映射到自车轨迹的时间索引（注意：self.plan_traj[0] 对应 t=0）
        ego_indices = (valid_t / dt_ego).astype(
            int
        )  # 注意：t=0.1 对应 index 1，所以直接除即可（无需 -1）
        ego_indices = np.clip(ego_indices, 0, len(self.plan_traj) - 1)
        ego_xy = (
            self.plan_traj[ego_indices, :2]
            if self.plan_traj.shape[1] > 2
            else self.plan_traj[ego_indices]
        )  # (T_valid, 2)
        others_xy = valid_others[..., :2]  # (N, T_valid, 2)

        dists = np.linalg.norm(others_xy - ego_xy[None, :, :], axis=2)
        selected = set()
        # 按时间顺序遍历
        for t_step in range(dists.shape[1]):
            close_veh_indices = np.where(dists[:, t_step] < dist_th)[0]
            if len(close_veh_indices) == 0:
                continue

            sorted_indices = close_veh_indices[
                np.argsort(dists[close_veh_indices, t_step])
            ]
            for veh_idx in sorted_indices:
                if veh_idx not in selected:
                    selected.add(veh_idx)
                    filtered_idxs.append(int(veh_idx))
                    if len(filtered_idxs) >= max_vehicles:
                        break
            if len(filtered_idxs) >= max_vehicles:
                break

        if not filtered_idxs:
            return {"states": None, "pred": None, "ids": None, "types": None}

        new_others_info = {
            "states": other_states[filtered_idxs],
            "pred": others_pred[filtered_idxs],
            "ids": [other_ids[i] for i in filtered_idxs],
            "types": [other_types[i] for i in filtered_idxs],
        }

        return new_others_info

    def get_close_vehicles_game_lane(
        self, data_process, limit_speed, max_vehicles_game=5, max_vehicles_mpc=10
    ):
        """
        基于车道拓扑结构初步选取关键他车，并结合轨迹距离补足至最多 5 辆。

        筛选流程：
        1. **车道级关键车辆选取**：
            - 在自车当前车道上，选取位于自车前方（沿车道方向）且横向距离 ≤2.5m 的最近前车；
            - 在相邻车道（最多两条）上，分别选取位于自车纵向位置前后、横向距离 ≤2.5m 的最近车辆（前/后车）；
            - 使用 point_to_line / points_to_line 计算车辆在车道上的投影索引以判断前后关系。

        2. **轨迹级补全**：
            - 若已选车辆数不足 5 辆，则调用 self.get_close_vehicles_game 补充剩余名额；
            - 补充时使用相同的距离阈值和时间窗口（1~5 秒），但不重复已选车辆。

        参数：
            data_process: 数据容器，需包含：
                - others_info: 他车信息（含 states, pred, ids, types）
                - ego_state: 自车状态 [x, y, ...]
                - lanes: 车道中心线列表（每个元素为点序列）
                - curr_lane_idx: 自车当前车道索引
                - near_target_idxes: 当前及相邻车道索引列表

        返回：
            dict or None: 筛选后的他车信息字典（结构同 others_info），若无有效车辆则各字段为 None。
        """
        is_detour = False
        filtered_idxs = []

        force_change_id_list = [None, None]

        other_states = data_process.others_info["states"]  # (N, 5)
        others_pred = data_process.others_info["pred"]  # (N, T_pred, 5)
        other_ids = data_process.others_info["ids"]
        other_types = data_process.others_info["types"]
        if others_pred is None:
            return {
                "states": None,
                "pred": None,
                "ids": None,
                "types": None,
                "force_change_id": force_change_id_list,
                "laneids": None,
                "frenet_relative_coords": [],
            }
        other_xy = other_states[:, :2]

        ego_xy = data_process.ego_state[:2]
        ego_speed = data_process.ego_state[3]

        # cur_lane_xy = np.vstack([np.linspace(a, b, k+2)[:-1] for a, b in zip(cur_lane_xy[:-1], cur_lane_xy[1:])] + [cur_lane_xy[-1]])
        # =========================================================
        # 优化核心：全向量化提取与加密函数 (避免 Python 原生循环)
        # =========================================================
        def fast_extract_and_dense(lane_points_obj, ego_xy, k=3):
            total_pts = len(lane_points_obj)
            if total_pts > 400:
                # 1. 稀疏定位：只抽取每隔 20 个点的坐标（极其轻量）
                stride = 20
                sparse_xy = np.array(
                    [
                        [lane_points_obj[i].x, lane_points_obj[i].y]
                        for i in range(0, total_pts, stride)
                    ]
                )
                coarse_idx = (
                    np.argmin(np.linalg.norm(sparse_xy - ego_xy, axis=1)) * stride
                )

                # 2. 物理截断：切片提取自车前后的一小段对象
                slice_start = max(0, coarse_idx - 50)
                slice_end = min(total_pts, coarse_idx + 250)
                lane_points_obj = lane_points_obj[slice_start:slice_end]

            # 3. 此时只对这几百个对象提取坐标，耗时 < 0.5ms
            lane_xy = np.array([[p.x, p.y] for p in lane_points_obj])

            # 4. 全向量化矩阵加密 (干掉原有的 np.vstack + for 循环)，耗时 < 0.2ms
            if len(lane_xy) > 1 and k > 0:
                starts = lane_xy[:-1]
                ends = lane_xy[1:]
                # 构造插值比例矩阵 t: [0, 0.25, 0.5, 0.75] (当 k=3 时)
                t = np.linspace(0, 1, k + 2)[:-1]
                # 三维张量广播运算，瞬间完成所有线段的内部点插值
                interp = (
                    starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]
                )
                lane_xy = np.vstack((interp.reshape(-1, 2), lane_xy[-1:]))

            return lane_xy

        # =========================================================

        # 完成主车道的提取、截断与加密
        cur_lane_xy = fast_extract_and_dense(
            data_process.lanes[data_process.curr_lane_idx], ego_xy, k=3
        )

        other_lanes_idxs = data_process.near_target_idxes.copy()  # 包含自己车道最多三条

        other_lanes_idxs = data_process.near_target_idxes.copy()  # 包含自己车道最多三条
        if data_process.curr_lane_idx in other_lanes_idxs:
            other_lanes_idxs.remove(data_process.curr_lane_idx)
        if other_lanes_idxs == [None]:
            other_lanes_idxs = []
        if other_lanes_idxs == []:
            if data_process.curr_lane_idx - 1 in data_process.lanes:
                other_lanes_idxs.append(data_process.curr_lane_idx - 1)
            if data_process.curr_lane_idx + 1 in data_process.lanes:
                other_lanes_idxs.append(data_process.curr_lane_idx + 1)
            if ros_log:
                rospy.loginfo(
                    "[筛选流程]仅有当前车道，curr_lane_idx+-1若存在则作为相邻车道"
                )
            else:
                print("[筛选流程]仅有当前车道，curr_lane_idx+-1若存在则作为相邻车道")

        ego_idx, ego_min_dist = point_to_line(ego_xy, cur_lane_xy)

        # 1. 添加自车道前车（以及后车，如果需要的话）
        # Step 1: 使用 5 点法判断哪些车靠近当前车道（更鲁棒）
        valid_mask, vehicle_min_distances = vehicle_points_to_lane(
            others_pred[:, 0, :6], cur_lane_xy, max_distance=1.2
        )

        # Step 2: 对所有车辆计算中心点在当前车道上的投影索引
        all_proj_indices, _ = points_to_line(other_xy, cur_lane_xy)  # (n,)

        # Step 3: 只保留有效车辆的投影索引
        valid_indices = np.where(valid_mask)[0]  # 原始车辆索引，如 [3, 7, 12]
        valid_proj_indices = all_proj_indices[valid_mask]  # 对应的投影索引

        laneids = []
        # Step 4: 前车
        front_mask = valid_proj_indices >= ego_idx
        if np.any(front_mask):
            front_proj_vals = valid_proj_indices[front_mask]
            front_original_idxs = valid_indices[front_mask]

            # 找最靠近 ego 的前车（即投影索引最小的那个）
            closest_front_idx = np.argmin(front_proj_vals)
            selected_front = front_original_idxs[closest_front_idx]

            if selected_front not in filtered_idxs:
                filtered_idxs.append(int(selected_front))
                laneids.append(data_process.curr_lane_idx)
                if ros_log:
                    rospy.loginfo(
                        f"[筛选流程]按车道筛选：自车车道{data_process.curr_lane_idx}的前车ID为{other_ids[selected_front]}"
                    )
                else:
                    print(
                        f"[筛选流程]按车道筛选：自车车道{data_process.curr_lane_idx}的前车ID为{other_ids[selected_front]}"
                    )

                # 启用绕行条件:
                # 1.速度小于1m/s;
                # 2.自车往前的中心线五米内的最近车，欧式距离max(50m,ego_speed*3)内；
                # 3.车框四个点距离中心线的最近点小于1.0m，最远点大于1.0m；（中心线点加密）
                # 4.博弈不换道(main_play中体现)

                # 获取对应车辆的状态信息
                x, y, heading, speed, length, width = others_pred[selected_front, 0, :]
                ego_x, ego_y = ego_xy  # 自车坐标
                distance_to_ego = np.sqrt((x - ego_x) ** 2 + (y - ego_y) ** 2)
                if speed < 1 and distance_to_ego < max(
                    80, data_process.ego_state[3] * 3
                ):
                    half_length = length / 2
                    half_width = width / 2
                    # 车辆中心 (x, y) 和朝向 heading
                    corners = np.array(
                        [
                            [
                                x
                                + half_length * np.cos(heading)
                                - half_width * np.sin(heading),
                                y
                                + half_length * np.sin(heading)
                                + half_width * np.cos(heading),
                            ],
                            [
                                x
                                + half_length * np.cos(heading)
                                + half_width * np.sin(heading),
                                y
                                + half_length * np.sin(heading)
                                - half_width * np.cos(heading),
                            ],
                            [
                                x
                                - half_length * np.cos(heading)
                                - half_width * np.sin(heading),
                                y
                                - half_length * np.sin(heading)
                                + half_width * np.cos(heading),
                            ],
                            [
                                x
                                - half_length * np.cos(heading)
                                + half_width * np.sin(heading),
                                y
                                - half_length * np.sin(heading)
                                - half_width * np.cos(heading),
                            ],
                        ]
                    )
                    _, corners_ego_min_dists = points_to_line(corners, cur_lane_xy)
                    # 判断条件
                    if ros_log:
                        rospy.loginfo(
                            f"[绕行判断]ID:{other_ids[selected_front]},距离最小和最大为:{min(corners_ego_min_dists),max(corners_ego_min_dists)}"
                        )
                    else:
                        print(
                            f"[绕行判断]ID:{other_ids[selected_front]},距离最小和最大为:{min(corners_ego_min_dists),max(corners_ego_min_dists)}"
                        )

                    if (
                        min(corners_ego_min_dists) < 1.0
                        and max(corners_ego_min_dists) > 1.0
                    ):
                        is_detour = True
                        if ros_log:
                            rospy.loginfo(
                                f"[绕行判断]需要绕行的ID为:{other_ids[selected_front]}, 判断博弈是否换道:"
                            )
                        else:
                            print(
                                f"[绕行判断]需要绕行的ID为:{other_ids[selected_front]}, 判断博弈是否换道:"
                            )

        # 2. 添加左右车道的前后车
        for other_lanes_idx in other_lanes_idxs:

            # 使用同样的内联函数极速处理相邻车道
            other_lane_xy = fast_extract_and_dense(
                data_process.lanes[other_lanes_idx], ego_xy, k=3
            )
            ego_other_idx, _ = point_to_line(ego_xy, other_lane_xy)

            # 仿真出现的问题:后方车道只有50m,但是车从50m外就开始出现,因此直线延长到至少80m
            other_lane_xy = extend_forward(
                other_lane_xy, ego_other_idx, min_length=80.0, spacing=2.0
            )
            ego_other_idx, _ = point_to_line(ego_xy, other_lane_xy)
            # 获取有效车辆（满足距离阈值）
            # Step 1: 使用 5 点法判断哪些车靠近车道（更鲁棒）
            valid_mask, vehicle_min_distances = vehicle_points_to_lane(
                others_pred[:, 0, :6], other_lane_xy, max_distance=1.2
            )

            # Step 2: 对所有车辆（或至少有效车辆）计算中心点在车道上的投影索引
            # 注意：即使车不在车道附近，也可以算投影，但我们只关心 valid_mask 为 True 的
            all_proj_indices, _ = points_to_line(other_xy, other_lane_xy)  # (n,)

            # Step 3: 只保留有效车辆的投影索引
            valid_indices = np.where(valid_mask)[0]  # 原始索引，如 [3, 7, 12]
            valid_proj_indices = all_proj_indices[valid_mask]  # 对应的投影索引

            # Step 4: 前车（投影索引 >= ego_other_idx + 3）
            threshold = ego_other_idx + 3
            front_mask = valid_proj_indices >= threshold
            if np.any(front_mask):
                # 在前车中，找投影索引最小的那个（最靠近 ego 的前方车）
                front_proj_vals = valid_proj_indices[front_mask]
                front_original_idxs = valid_indices[front_mask]

                # 找投影索引最小的（即最靠近 ego 的前车）
                closest_front_idx = np.argmin(front_proj_vals)
                selected_front = front_original_idxs[closest_front_idx]

                if [other_lanes_idx] == data_process.near_target_idxes.copy():
                    force_change_id_list[0] = other_ids[selected_front]

                if selected_front not in filtered_idxs:
                    filtered_idxs.append(int(selected_front))
                    laneids.append(other_lanes_idx)
                    if ros_log:
                        rospy.loginfo(
                            f"[筛选流程]按车道筛选：相邻车道{other_lanes_idx}的前车ID为{other_ids[selected_front]}"
                        )
                    else:
                        print(
                            f"[筛选流程]按车道筛选：相邻车道{other_lanes_idx}的前车ID为{other_ids[selected_front]}"
                        )

            # Step 5: 后车（投影索引 < threshold）
            rear_mask = valid_proj_indices < threshold
            if np.any(rear_mask):
                # 在后车中，找投影索引最大的那个（最靠近 ego 的后方车）
                rear_proj_vals = valid_proj_indices[rear_mask]
                rear_original_idxs = valid_indices[rear_mask]

                closest_rear_idx = np.argmax(rear_proj_vals)  # 最大索引 = 最靠前的后车
                selected_rear = rear_original_idxs[closest_rear_idx]

                if [other_lanes_idx] == data_process.near_target_idxes.copy():
                    force_change_id_list[1] = other_ids[selected_rear]

                if selected_rear not in filtered_idxs:
                    filtered_idxs.append(int(selected_rear))
                    laneids.append(other_lanes_idx)
                    if ros_log:
                        rospy.loginfo(
                            f"[筛选流程]按车道筛选：相邻车道{other_lanes_idx}的后车ID为{other_ids[selected_rear]}"
                        )
                    else:
                        print(
                            f"[筛选流程]按车道筛选：相邻车道{other_lanes_idx}的后车ID为{other_ids[selected_rear]}"
                        )

        self.init_plan_traj_by_path(data_process.ego_state, cur_lane_xy)
        new_others_info_subset = self.get_close_vehicles_game(
            data_process.others_info, max_vehicles=max_vehicles_mpc - len(filtered_idxs)
        )
        if new_others_info_subset["ids"] != None:
            for id_ in new_others_info_subset["ids"]:
                idx_ = other_ids.index(id_)
                # 距离路线10m外且速度小于0.5m/s的静止其他类型障碍物不考虑
                if idx_ not in filtered_idxs:
                    if not (
                        vehicle_min_distances[idx_] > 5
                        and other_states[idx_][3] < 0.5
                        and other_types[idx_] == "obs"
                    ):
                        filtered_idxs.append(int(idx_))
                        laneids.append(None)
                        if ros_log:
                            rospy.loginfo(
                                f"[筛选流程]按时空距离筛选增加：{other_ids[idx_]}"
                            )
                        else:
                            print(f"[筛选流程]按时空距离筛选增加：{other_ids[idx_]}")

        self.other_style_dict = self.get_style(
            ego_speed,
            limit_speed,
            filtered_idxs[:max_vehicles_game],
            other_states,
            others_pred,
            other_ids,
            other_types,
        )

        if len(filtered_idxs) == 0:
            return {
                "states": None,
                "pred": None,
                "ids": None,
                "types": None,
                "force_change_id": force_change_id_list,
                "laneids": None,
            }

        if len(filtered_idxs) == 0:
            return {
                "states": None,
                "pred": None,
                "ids": None,
                "types": None,
                "force_change_id": force_change_id_list,
                "laneids": None,
            }

        # ==========================================================
        # 【新增逻辑】：仅对筛选出的目标车辆进行精确 Frenet 投影
        # ==========================================================
        game_idxs = filtered_idxs[:max_vehicles_game]

        # 1. 提取自车坐标，并计算精确 (s, l)
        ego_s_array, ego_l_array = points_to_exact_frenet(
            np.array([ego_xy]), cur_lane_xy
        )
        ego_s_exact, ego_l_exact = ego_s_array[0], ego_l_array[0]

        # 2. 提取筛选出的他车坐标，并批量计算精确 (s, l)
        filtered_other_xy = other_states[game_idxs, :2]
        others_s_exact, others_l_exact = points_to_exact_frenet(
            filtered_other_xy, cur_lane_xy
        )

        # 3. 计算相对距离并存入列表
        frenet_coords_list = []
        for i in range(len(game_idxs)):
            ds = others_s_exact[i] - ego_s_exact
            dl = others_l_exact[i] - ego_l_exact
            frenet_coords_list.append((ds, dl))
        # ==========================================================

        new_others_info = {
            "states": other_states[filtered_idxs[:max_vehicles_game]],
            "pred": others_pred[filtered_idxs[:max_vehicles_game]],
            "ids": [other_ids[i] for i in filtered_idxs[:max_vehicles_game]],
            "types": [other_types[i] for i in filtered_idxs[:max_vehicles_game]],
            "pred_for_mpc": others_pred[filtered_idxs],
            "ids_for_mpc": [other_ids[i] for i in filtered_idxs],
            "is_detour": is_detour,
            "force_change_id": force_change_id_list,
            "laneids": laneids,
            "frenet_relative_coords": frenet_coords_list,  # 【新增】直接透传算好的相对坐标
        }

        if ros_log:
            rospy.loginfo(
                f"最终博弈筛选ID为:{new_others_info['ids']},\n最终mpc筛选ID为:{[other_ids[i] for i in filtered_idxs[:max_vehicles_mpc]]}"
            )
        else:
            print(
                f"最终博弈筛选ID为:{new_others_info['ids']},\n最终mpc筛选ID为:{[other_ids[i] for i in filtered_idxs[:max_vehicles_mpc]]}"
            )
        return new_others_info

    def get_style(
        self,
        ego_speed,
        limit_speed,
        filtered_idxs,
        other_states,
        others_pred,
        other_ids,
        other_types,
    ):
        if ros_log:
            rospy.loginfo(
                f"[驾驶风格识别]ego_speed:{ego_speed*3.6:.1f},limit_speed:{limit_speed*3.6:.1f}"
            )
        else:
            print(
                f"[驾驶风格识别]ego_speed:{ego_speed:.1f},limit_speed:{limit_speed*3.6:.1f}"
            )

        other_style_dict = {}
        for idx in filtered_idxs:
            obj_id = other_ids[idx]
            # 非交通参与者为不进行风格辨识
            if other_types[idx] == "obs":
                other_style_dict[obj_id] = "None"
                continue

            v = other_states[idx, 3] * 3.6
            a = other_states[idx, 4]
            length = others_pred[idx, 0, 4]
            value = v + a * 3 + min(max(0, length - 4.5), 3) * 5
            obj_id = other_ids[idx]
            if ros_log:
                rospy.loginfo(
                    f"[驾驶风格识别]obj_id:{obj_id},v:{v:.1f},a:{a:.1f},length:{length:.1f},value:{value:.1f}"
                )
            else:
                print(
                    f"[驾驶风格识别]obj_id:{obj_id},v:{v:.1f},a:{a:.1f},length:{length:.1f},value:{value:.1f}"
                )
            if v > limit_speed * 3.6:
                other_style_dict[obj_id] = "Aggressive"
                continue
            elif v < 1:
                other_style_dict[obj_id] = "Normal"
                continue

            if value > ego_speed * 3.6 + 12:
                other_style_dict[obj_id] = "Aggressive"
            elif value < ego_speed * 3.6:
                other_style_dict[obj_id] = "Conservative"
            else:
                other_style_dict[obj_id] = "Normal"

        return other_style_dict


def extend_forward(points, ego_other_idx, min_length=80.0, spacing=2.0):
    points = np.asarray(points, dtype=float)
    total_len = np.linalg.norm(np.diff(points[:ego_other_idx], axis=0), axis=1).sum()
    if total_len >= min_length:
        return points.copy()
    need = min_length - total_len
    # Smoothed/stitched reference lines may start with duplicate points.
    # Use the first non-degenerate segment so NaNs cannot enter predictions.
    directions = np.diff(points, axis=0)
    norms = np.linalg.norm(directions, axis=1)
    valid = np.flatnonzero(norms > 1e-9)
    if valid.size == 0:
        return points.copy()
    first = int(valid[0])
    unit = directions[first] / norms[first]
    # 向前延长 = 反方向（往更远处）
    n_steps = int(np.ceil(need / spacing))
    dists = np.linspace(spacing, need, n_steps)
    new_pts = points[0] - np.outer(dists, unit)  # 注意是减号！
    return np.vstack([new_pts[::-1], points])


def points_to_exact_frenet(points, polyline):
    """
    全量向量化：精确计算多个点到折线的 Frenet 坐标 (s, l)
    """
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if len(points) == 0:
        return np.array([]), np.array([])

    N = points.shape[0]

    starts = polyline[:-1]
    ends = polyline[1:]
    line_vecs = ends - starts

    line_lens_sq = np.einsum("ij,ij->i", line_vecs, line_vecs)
    line_lens = np.sqrt(line_lens_sq)
    cum_s = np.concatenate([[0.0], np.cumsum(line_lens)])

    point_vecs = points[:, np.newaxis, :] - starts

    dots = np.einsum("nic,ic->ni", point_vecs, line_vecs)
    t = np.divide(dots, line_lens_sq, out=np.zeros_like(dots), where=line_lens_sq != 0)
    t = np.clip(t, 0.0, 1.0)

    closest_points = starts + t[:, :, np.newaxis] * line_vecs
    distances = np.linalg.norm(points[:, np.newaxis, :] - closest_points, axis=2)

    min_seg_indices = np.argmin(distances, axis=1)

    best_t = t[np.arange(N), min_seg_indices]
    exact_s = cum_s[min_seg_indices] + best_t * line_lens[min_seg_indices]

    best_point_vecs = point_vecs[np.arange(N), min_seg_indices, :]
    best_line_vecs = line_vecs[min_seg_indices, :]
    cross_prod = (
        best_line_vecs[:, 0] * best_point_vecs[:, 1]
        - best_line_vecs[:, 1] * best_point_vecs[:, 0]
    )

    exact_l_abs = distances[np.arange(N), min_seg_indices]
    exact_l = np.sign(cross_prod) * exact_l_abs  # 左正右负

    return exact_s, exact_l


def set_limit_speed(
    data_process, gamesolver, MAX_ACCEL=2.5, MAX_DECEL=1.5, DT=0.1
):  # DT=0.0792
    set_speed_lane = data_process.lanes[data_process.curr_lane_idx][
        data_process.curr_lane_point_idx + 5 : data_process.curr_lane_point_idx + 100
    ]
    # 计算原始限速（道路+曲率）
    speed_list = [j.speed for j in set_speed_lane]
    # 如果正在变道就同时考虑目标车道限速
    if (
        gamesolver.changing == True
        and gamesolver.current_target_idx_point is not None
        and gamesolver.current_target_idx in data_process.lanes
    ):
        target_lane = data_process.lanes[gamesolver.current_target_idx]
        target_lane_points = np.array([[i.x, i.y] for i in target_lane])
        target_lane_speed_list = [i.speed for i in target_lane]
        target_idx, target_min_dist = point_to_line(
            gamesolver.current_target_idx_point, target_lane_points
        )
        if target_min_dist < 3:
            speed_list.extend(target_lane_speed_list[target_idx : target_idx + 10])
    if len(speed_list) == 0:
        raw_min_speed = 12
    else:
        raw_min_speed = min(speed_list)
    new_min_speed = raw_min_speed

    # 上一时刻已设置的限速
    last_limit = gamesolver.limits[0]["MAX_SPEED"]
    current_speed = data_process.ego_state[3]
    # 根据目标与当前值，只在限速减速过多时候降低限速
    if raw_min_speed < last_limit - 0.5:
        if current_speed > raw_min_speed:
            # 最大限速需要降速，直接从当前速度开始限，但为了防止自车反常加速导致限速也会跟着反常增加,因此需要根据上一时刻进行判断
            max_delta_v = MAX_DECEL * DT
            new_min_speed = min(
                max(current_speed - max_delta_v, raw_min_speed),
                last_limit - max_delta_v,
            )
    # # 防止出掉头加速过猛,限速涨幅进行约束
    # elif raw_min_speed > last_limit + 0.5:
    #     if current_speed < raw_min_speed:
    #         max_delta_v = MAX_ACCEL * DT
    #         new_min_speed = min(raw_min_speed, last_limit + max_delta_v)

    gamesolver.limits[0]["MAX_SPEED"] = new_min_speed
    if ros_log:
        rospy.loginfo(
            f"自车速度:{round(current_speed,1)}m/s，计算限速:{round(raw_min_speed,1)}m/s，实际执行限速:{round(new_min_speed,1)}m/s"
        )
    else:
        print(
            f"自车速度:{round(current_speed,1)}m/s，计算限速:{round(raw_min_speed,1)}m/s，实际执行限速:{round(new_min_speed,1)}m/s"
        )
        print(
            f"speed_plot {round(current_speed,1)} {round(raw_min_speed,1)} {round(new_min_speed,1)}"
        )
    return new_min_speed


def point_to_line(point, line_points):
    """
    找到输入点(2,)到线段点集(n,2)的最近距离及对应点的索引

    参数:
    point -- 输入点，形状为(2,)的numpy数组
    line_points -- 线段点集，形状为(n,2)的numpy数组

    返回:
    min_distance -- 最近距离
    min_index -- 最近点的索引
    """
    # 确保输入是numpy数组
    point = np.asarray(point)
    line_points = np.asarray(line_points)

    # 计算每个线段点到输入点的距离
    distances = np.linalg.norm(line_points - point, axis=1)

    # 找到最小距离和对应的索引
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    return min_index, min_distance


def points_to_line(points, line_points):
    """
    计算 points (n1,2) 到 line_points (n2,2) 的最近距离和索引（完全向量化）

    参数:
        points: (n1, 2) 的点集
        line_points: (n2, 2) 的线段点集

    返回:
        min_distances: (n1,) 最近距离数组
        min_indices: (n1,) 最近点的索引数组
    """
    distances = np.linalg.norm(points[:, np.newaxis, :] - line_points, axis=2)

    min_indices = np.argmin(distances, axis=1)

    min_distances = np.min(distances, axis=1)

    return min_indices, min_distances


def vehicle_points_to_lane(others_states, lane_xy, max_distance=1.0):
    """
    计算多辆车到车道的最近距离（基于车辆5个关键点）

    参数:
        others_states: (n, 6) 车辆状态数组 [x, y, heading, speed, length, width]
        lane_xy: (m, 2) 车道参考线点集
        max_distance: 最大匹配距离阈值

    返回:
        valid_mask: (n,) 布尔数组
        min_distances: (n,) 每辆车的最小距离
    """
    if others_states.size == 0:
        return np.array([], dtype=bool), np.array([])

    x = others_states[:, 0]
    y = others_states[:, 1]
    heading = others_states[:, 2]
    length = others_states[:, 4]
    width = others_states[:, 5]

    half_length = length / 2  # (n,)
    half_width = width / 2  # (n,)

    cos_h = np.cos(heading)  # (n,)
    sin_h = np.sin(heading)  # (n,)

    hl = half_length[:, None]
    hw = half_width[:, None]
    ch = cos_h[:, None]
    sh = sin_h[:, None]

    # x 偏移 (n, 4)
    dx = np.concatenate(
        [hl * ch - hw * sh, hl * ch + hw * sh, -hl * ch + hw * sh, -hl * ch - hw * sh],
        axis=1,
    )

    # y 偏移 (n, 4)
    dy = np.concatenate(
        [hl * sh + hw * ch, hl * sh - hw * ch, -hl * sh - hw * ch, -hl * sh + hw * ch],
        axis=1,
    )

    offsets = np.stack([dx, dy], axis=2)

    centers = np.stack([x, y], axis=1)

    corners = centers[:, np.newaxis, :] + offsets

    # 5个点：4角 + 中心
    all_points = np.concatenate(
        [centers[:, np.newaxis, :], corners], axis=1
    )  # (n, 5, 2)

    diff = (
        all_points[:, :, np.newaxis, :] - lane_xy[np.newaxis, np.newaxis, :, :]
    )  # (n,5,m,2)
    distances = np.linalg.norm(diff, axis=3)  # (n, 5, m)

    min_per_point = np.min(
        distances, axis=2
    )  # (n, 5)，五个点到车道中心线的距离，第一个是中心点
    vehicle_min_distances = np.min(min_per_point, axis=1)  # (n,)

    valid_mask = vehicle_min_distances <= max_distance

    return valid_mask, vehicle_min_distances
