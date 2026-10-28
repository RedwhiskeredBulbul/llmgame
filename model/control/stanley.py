import numpy as np

from model.settings import SIM_DT, DELTA_MAX, WHEEL_BASE


class StanleyController:
    def __init__(self):
        self.L = WHEEL_BASE
        self.dt = SIM_DT
        self.ay_lim = 3.2
        self.jy_lim = 10
        self.delta_max = DELTA_MAX
        self.route = []
        self.now_act1 = 0.0
        self.k_base = 0.7
        self.k_yaw = 3

    def act_stanley(self, ego_state, plan_idx=0):

        result_r = self.stanley(
            ego_state[0], ego_state[1], yaw=ego_state[3], v=ego_state[2]
        )

        delta_lim = np.arctan(self.ay_lim * self.L / (ego_state[2] ** 2 + 1e-6))
        delta_lim = min(delta_lim, self.delta_max)

        if result_r > delta_lim:
            result_r = delta_lim
        elif result_r < -delta_lim:
            result_r = -delta_lim

        d_delta_lim = (
            self.jy_lim
            * self.L
            * np.cos(self.now_act1) ** 2
            / (ego_state[2] ** 2 + 1e-6)
            * self.dt
        )
        if plan_idx == 0:
            d_delta_lim = max(d_delta_lim, 0.01)  # 第一个点的前轮转角存在传感器误差
        else:
            d_delta_lim = max(d_delta_lim, 0.005)

        if result_r - self.now_act1 > d_delta_lim:
            result_r = self.now_act1 + d_delta_lim
        elif result_r - self.now_act1 < -d_delta_lim:
            result_r = self.now_act1 - d_delta_lim

        self.now_act1 = result_r

        return result_r

    def stanley(self, x_r, y_r, yaw, v):

        # 前轮坐标
        x_f = x_r + self.L * np.cos(yaw)
        y_f = y_r + self.L * np.sin(yaw)

        # 最近索引
        idx, e = self.find_nearest_point(self.route, np.array([x_f, y_f]))

        yaw_ref = np.arctan2(
            self.route[idx, 1] - self.route[idx - 1, 1],
            self.route[idx, 0] - self.route[idx - 1, 0],
        )

        side = np.sign(
            (self.route[idx, 0] - x_f) * np.sin(yaw_ref)
            - (self.route[idx, 1] - y_f) * np.cos(yaw_ref)
        )
        e = e * side

        max_ey = 1.0
        e = np.clip(e, -max_ey, max_ey)  # 对横向误差进行限幅 0.8

        # 航向误差
        delta_yaw = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi

        delta = self.k_yaw * delta_yaw + np.arctan2(-self.k_base * e, max(v, 0.1))

        return delta

    def find_nearest_point(self, points, new_point):
        """
        找到与新坐标最近的点的索引，并计算到参考路径（折线）的精确最短距离。

        参数:
        points (numpy.ndarray): 一个形状为 (n, 2) 的数组，表示 n 个点的坐标。
        new_point (list or numpy.ndarray): 一个包含两个元素的列表或数组，表示新坐标的 [x, y]。

        返回:
        int: 距离最近的点的索引。
        float: 到路径的最短距离。
        """
        # 将新坐标转换为 numpy 数组
        new_point = np.array(new_point)

        # 计算每个点到新坐标的欧几里得距离
        distances = np.linalg.norm(points - new_point, axis=1)

        # 找到距离最近的点的索引
        nearest_index = int(np.argmin(distances))

        # 如果路径点少于2个，直接返回点到点距离
        if len(points) < 2:
            return nearest_index, distances[nearest_index]

        # 依次检查最近点所在的“前一节”和“后一节”线段
        def seg_dist(p1, p2):
            """点到线段(p1,p2)的最短距离"""
            line_vec = p2 - p1
            line_len2 = np.dot(line_vec, line_vec)
            if line_len2 == 0:  # 零长度线段
                return np.linalg.norm(new_point - p1)
            point_vec = new_point - p1
            t = np.clip(np.dot(point_vec, line_vec) / line_len2, 0.0, 1.0)
            closest = p1 + t * line_vec
            return np.linalg.norm(new_point - closest)

        cand = [distances[nearest_index]]  # 至少保留“点到点”距离
        if nearest_index > 0:  # 前一节线段
            cand.append(seg_dist(points[nearest_index - 1], points[nearest_index]))
        if nearest_index < len(points) - 1:  # 后一节线段
            cand.append(seg_dist(points[nearest_index], points[nearest_index + 1]))

        min_dis = min(cand)

        return nearest_index, min_dis
