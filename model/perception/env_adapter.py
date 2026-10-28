import numpy as np
import math
from dataclasses import dataclass
from highway_env.road.lane import StraightLane


class EnvironmentAdapter:

    def __init__(self):
        self.ego_states = np.array([0, 0, 0, 0, 0, 0])
        self.others_info = {
            "states": np.zeros((0, 6)),
            "pred": np.zeros((0, 0, 3)),
            "ids": [],
            "types": [],
        }
        self.target_idxes = []
        self.near_target_idxes = []
        self.near_permit_idxes = []
        self.lanes = {}
        self.curr_lane_idx = None
        self.auto_state = 1

    def update_data(self, env_name, env):
        ego_state = np.array(
            [
                env.unwrapped.controlled_vehicles[0].position[0],
                env.unwrapped.controlled_vehicles[0].position[1],
                env.unwrapped.controlled_vehicles[0].heading,
                env.unwrapped.controlled_vehicles[0].speed,
                env.unwrapped.controlled_vehicles[0].action["acceleration"],
                env.unwrapped.controlled_vehicles[0].action["steering"],
            ]
        )
        other_states, others_pred, other_ids, other_types = get_hv_preds(
            env.unwrapped.road.vehicles, env.unwrapped.controlled_vehicles[0].id
        )
        others_info = {
            "states": other_states,
            "pred": others_pred,
            "ids": other_ids,
            "types": other_types,
        }
        raw_lane_index = env.unwrapped.controlled_vehicles[0].lane_index
        curr_lane_idx = lane_path_map_dict[env_name][raw_lane_index]
        target_idxes = lane_id_targetlane_dict[env_name][curr_lane_idx]
        lanes = lane_id_refline_dict[env_name]
        self.ego_state = ego_state
        self.others_info = others_info
        self.lanes = lanes
        self.target_idxes = target_idxes
        self.near_target_idxes = target_idxes
        self.near_permit_idxes = target_idxes
        self.curr_lane_idx = curr_lane_idx
        lane_points = self.lanes[self.curr_lane_idx]
        lane_xy = np.array([[p.x, p.y] for p in lane_points])
        dists = np.hypot(
            lane_xy[:, 0] - self.ego_state[0], lane_xy[:, 1] - self.ego_state[1]
        )
        nearest_idx = np.argmin(dists)
        self.curr_lane_point_idx = nearest_idx

    def mpc_bound(self, target_index):
        curr_lane = self.lanes[self.curr_lane_idx]
        cur_lane_centers = []
        cur_lane_d = []
        for point in curr_lane:
            cur_lane_centers.append([point.x, point.y])
            cur_lane_d.append([point.x, point.to_left, point.to_right])
        cur_lane_centers = np.array(cur_lane_centers)
        cur_lane_d = np.array(cur_lane_d)
        target_lane = self.lanes[target_index]
        target_lane_centers = []
        target_lane_d = []
        for point in target_lane:
            target_lane_centers.append([point.x, point.y])
            target_lane_d.append([point.x, point.to_left, point.to_right])
        target_lane_centers = np.array(target_lane_centers)
        target_lane_d = np.array(target_lane_d)
        return (cur_lane_centers, cur_lane_d, target_lane_centers, target_lane_d)


@dataclass
class Point:
    x: float
    y: float
    heading: float
    speed: float = 0.0
    curvature: float = 0.0
    to_left: float = 2.0
    to_right: float = 2.0
    point_type: int = 0
    left_traverse_flag: int = 0
    right_traverse_flag: int = 0


def get_hv_preds(other_vehicles, av_id, N=16, dt=0.5, n_int=6):
    other_states = []
    hv_ids = []
    other_types = []
    observed_vehicles = [vehicle for vehicle in other_vehicles if vehicle.id != av_id]
    for vehicle in observed_vehicles:
        other_states.append(
            [
                vehicle.position[0],
                vehicle.position[1],
                vehicle.heading,
                vehicle.speed,
                vehicle.action["acceleration"],
                0,
            ]
        )
        hv_ids.append(vehicle.id)
        other_types.append(vehicle.type)
    other_states = np.array(other_states).reshape(-1, n_int)
    other = np.asarray(other_states) if other_states is not None else np.zeros((0, 6))
    if other.ndim == 1 and other.size > 0:
        other = other.reshape(1, -1)
    n_other = other.shape[0] if other.size > 0 else 0
    other_preds = np.zeros((n_other, N + 1, 6), dtype=float)
    for j in range(n_other):
        ox, oy, otheta, ov = (
            float(other[j, 0]),
            float(other[j, 1]),
            float(other[j, 2]),
            float(other[j, 3]),
        )
        for k in range(N + 1):
            tpred = k * dt
            other_preds[j, k, 0] = ox + ov * np.cos(otheta) * tpred
            other_preds[j, k, 1] = oy + ov * np.sin(otheta) * tpred
            other_preds[j, k, 2] = otheta
            other_preds[j, k, 3] = ov
            other_preds[j, k, 4] = observed_vehicles[j].LENGTH
            other_preds[j, k, 5] = observed_vehicles[j].WIDTH
    return (other_states, other_preds, hv_ids, other_types)


def build_refline_from_endpoints(
    start, end, start_to_lr=(2.0, 2.0), end_to_lr=(2.0, 2.0), step=1.0, speed=120 / 3.6
):
    """
    输入起点和终点坐标，沿直线等距采样生成车道中心线参考线（Point格式）。
    Args:
        start: (x0, y0)
        end: (x1, y1)
        to_left: 左侧到车道中心线的距离 (m)，保存到 Point.to_left
        to_right: 右侧到车道中心线的距离 (m)，保存到 Point.to_right
        step: 采样步长 (m)
        speed: 参考线上的默认速度 (m/s)
    Returns:
        list[Point]: 从起点到终点的采样点序列
    """
    sx, sy = (float(start[0]), float(start[1]))
    ex, ey = (float(end[0]), float(end[1]))
    dx, dy = (ex - sx, ey - sy)
    d2l = end_to_lr[0] - start_to_lr[0]
    d2r = end_to_lr[1] - start_to_lr[1]
    dist = math.hypot(dx, dy)
    if dist <= 0.0:
        heading = 0.0
        return [
            Point(
                x=sx,
                y=sy,
                heading=heading,
                speed=float(speed),
                curvature=0.0,
                to_left=float(start_to_lr[0]),
                to_right=float(start_to_lr[1]),
            )
        ]
    intervals = max(1, int(math.ceil(dist / float(step))))
    heading = math.atan2(dy, dx)
    pts = []
    for i in range(intervals + 1):
        t = i / intervals
        x = sx + t * dx
        y = sy + t * dy
        to_left = start_to_lr[0] + t * d2l
        to_right = start_to_lr[1] + t * d2r
        pts.append(
            Point(
                x=float(x),
                y=float(y),
                heading=float(heading),
                speed=float(speed),
                curvature=0.0,
                to_left=float(to_left),
                to_right=float(to_right),
            )
        )
    return pts


def smooth_refline(points, window=5, iterations=1):
    """
    平滑 Point 列表（输入/输出与 build_refline_from_endpoints 保持一致）。
    平滑的属性：x, y, heading (角度), speed, curvature, to_left, to_right。
    参数:
        points: list[Point]
        window: int, 移动平均窗口大小（奇数更好）
        iterations: int, 重复平滑次数
    返回:
        list[Point]: 平滑后的点列表
    """
    if points is None:
        return []
    pts = list(points)
    n = len(pts)
    if n == 0:
        return []
    if window < 1:
        window = 1
    if n <= 2 or window == 1:
        return [
            Point(
                x=p.x,
                y=p.y,
                heading=p.heading,
                speed=p.speed,
                curvature=p.curvature,
                to_left=p.to_left,
                to_right=p.to_right,
            )
            for p in pts
        ]
    x = np.asarray([p.x for p in pts], dtype=float)
    y = np.asarray([p.y for p in pts], dtype=float)
    h_rad = np.asarray([p.heading for p in pts], dtype=float)
    s = np.asarray([p.speed for p in pts], dtype=float)
    k = np.asarray([p.curvature for p in pts], dtype=float)
    tl = np.asarray([p.to_left for p in pts], dtype=float)
    tr = np.asarray([p.to_right for p in pts], dtype=float)

    def moving_average(arr, w):
        if w <= 1:
            return arr.copy()
        pad = w // 2
        arr_pad = np.pad(arr, pad_width=pad, mode="edge")
        kernel = np.ones(w, dtype=float) / float(w)
        sm = np.convolve(arr_pad, kernel, mode="valid")
        return sm

    for _ in range(max(1, int(iterations))):
        x = moving_average(x, window)
        y = moving_average(y, window)
        h_rad = np.unwrap(h_rad)
        h_rad = moving_average(h_rad, window)
        s = moving_average(s, window)
        k = moving_average(k, window)
        tl = moving_average(tl, window)
        tr = moving_average(tr, window)
    smoothed = []
    for i in range(n):
        orig = pts[i]
        smoothed.append(
            Point(
                x=float(x[i]),
                y=float(y[i]),
                heading=float(h_rad[i]),
                speed=float(s[i]),
                curvature=float(k[i]),
                to_left=float(tl[i]),
                to_right=float(tr[i]),
            )
        )
    return smoothed


cutin_lane_ends = {
    ("a", "b", 0): [(0, 0), (10000.0, 0)],
    ("a", "b", 1): [(0, 4), (10000.0, 4)],
}
cutin_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0],
        end=ends[1],
        start_to_lr=(
            2.0
            + list(cutin_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH
            * (
                len(list(cutin_lane_ends.keys()))
                - list(cutin_lane_ends.keys()).index(lane_id)
            )
            - 2.0,
        ),
        end_to_lr=(
            2.0
            + list(cutin_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH
            * (
                len(list(cutin_lane_ends.keys()))
                - list(cutin_lane_ends.keys()).index(lane_id)
            )
            - 2.0,
        ),
        step=1.0,
        speed=120 / 3.6,
    )
    for lane_id, ends in cutin_lane_ends.items()
}
cutin_lane_path = {
    0: cutin_lane_refline["a", "b", 0],
    1: cutin_lane_refline["a", "b", 1],
}
cutin_lane_path_map = {("a", "b", 0): 0, ("a", "b", 1): 1}
cutin_target_lanes = {0: [0], 1: [0, 1]}
highway_lane_ends = {
    ("0", "1", 0): [(0, 0), (10000.0, 0)],
    ("0", "1", 1): [(0, 4), (10000.0, 4)],
    ("0", "1", 2): [(0, 8), (10000.0, 8)],
    ("0", "1", 3): [(0, 12), (10000.0, 12)],
}
highway_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0],
        end=ends[1],
        start_to_lr=(
            2.0
            + list(highway_lane_ends.keys()).index(lane_id)
            * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH
            * (
                len(list(highway_lane_ends.keys()))
                - list(highway_lane_ends.keys()).index(lane_id)
            )
            - 2.0,
        ),
        end_to_lr=(
            2.0
            + list(highway_lane_ends.keys()).index(lane_id)
            * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH
            * (
                len(list(highway_lane_ends.keys()))
                - list(highway_lane_ends.keys()).index(lane_id)
            )
            - 2.0,
        ),
        step=1.0,
        speed=120 / 3.6,
    )
    for lane_id, ends in highway_lane_ends.items()
}
highway_lane_path = {
    0: highway_lane_refline["0", "1", 0],
    1: highway_lane_refline["0", "1", 1],
    2: highway_lane_refline["0", "1", 2],
    3: highway_lane_refline["0", "1", 3],
}
highway_lane_path_map = {
    ("0", "1", 0): 0,
    ("0", "1", 1): 1,
    ("0", "1", 2): 2,
    ("0", "1", 3): 3,
}
highway_target_lanes = {0: [0, 1], 1: [0, 1, 2], 2: [1, 2, 3], 3: [2, 3]}
merge_ends = {
    ("a", "b", 0): [(0, 0), (150, 0)],
    ("a", "b", 1): [(0, 4), (150, 4)],
    ("b", "c", 0): [(150, 0), (310, 0)],
    ("b", "c", 1): [(150, 4), (310, 4)],
    ("c", "d", 0): [(310, 0), (450, 0)],
    ("c", "d", 1): [(310, 4), (450, 4)],
    ("j", "k", 0): [(0, 8), (150, 8)],
    ("k", "b", 0): [(150, 8), (310, 4)],
}
merge_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], end=ends[1], step=1.0, speed=80 / 3.6
    )
    for lane_id, ends in merge_ends.items()
}
merge_lane_path = {
    0: merge_lane_refline["a", "b", 0]
    + merge_lane_refline["b", "c", 0]
    + merge_lane_refline["c", "d", 0],
    1: merge_lane_refline["a", "b", 1]
    + merge_lane_refline["b", "c", 1]
    + merge_lane_refline["c", "d", 1],
    "ramp": merge_lane_refline["j", "k", 0]
    + merge_lane_refline["k", "b", 0]
    + merge_lane_refline["c", "d", 1],
}
merge_lane_path["ramp"] = smooth_refline(merge_lane_path["ramp"])
merge_lane_path_map = {
    ("a", "b", 0): 0,
    ("a", "b", 1): 1,
    ("b", "c", 0): 0,
    ("b", "c", 1): 1,
    ("c", "d", 0): 0,
    ("c", "d", 1): 1,
    ("j", "k", 0): "ramp",
    ("k", "b", 0): "ramp",
    ("b", "c", 2): "ramp",
}
merge_target_lanes = {0: [0, 1], 1: [0, 1], "ramp": [1, "ramp"]}
width = 4.0
adversarial_lane_ends = {
    ("a", "b", 0): [(0, 0), (300, 0)],
    ("b", "c", 0): [(300, 0), (380, 0)],
    ("c", "d", 0): [(380, 0), (1000, 0)],
    ("a", "b", 1): [(0, width), (300, width)],
    ("b", "c", 1): [(300, width), (380, width)],
    ("c", "d", 1): [(380, width), (1000, width)],
    ("a", "b", 2): [(0, 2 * width), (300, 2 * width)],
    ("b", "c", 2): [(300, 2 * width), (380, 2 * width)],
    ("c", "d", 2): [(380, 2 * width), (1000, 2 * width)],
    ("a", "b", 3): [(0, 3 * width), (300, 3 * width)],
    ("b", "c", 3): [(300, 3 * width), (380, 3 * width)],
    ("c", "d", 3): [(380, 3 * width), (1000, 3 * width)],
    ("a", "b", 4): [(0, 4 * width), (300, 4 * width)],
    ("b", "c", 4): [(300, 4 * width), (380, 3 * width)],
}
adversarial_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=pts[0], end=pts[1], step=1.0, speed=80 / 3.6
    )
    for lane_id, pts in adversarial_lane_ends.items()
}
adversarial_lane_path = {
    0: adversarial_lane_refline["a", "b", 0]
    + adversarial_lane_refline["b", "c", 0]
    + adversarial_lane_refline["c", "d", 0],
    1: adversarial_lane_refline["a", "b", 1]
    + adversarial_lane_refline["b", "c", 1]
    + adversarial_lane_refline["c", "d", 1],
    2: adversarial_lane_refline["a", "b", 2]
    + adversarial_lane_refline["b", "c", 2]
    + adversarial_lane_refline["c", "d", 2],
    3: adversarial_lane_refline["a", "b", 3]
    + adversarial_lane_refline["b", "c", 3]
    + adversarial_lane_refline["c", "d", 3],
    "ramp": adversarial_lane_refline["a", "b", 4]
    + adversarial_lane_refline["b", "c", 4]
    + adversarial_lane_refline["c", "d", 3],
}
adversarial_lane_path["ramp"] = smooth_refline(adversarial_lane_path["ramp"])
adversarial_lane_path_map = {
    ("a", "b", 0): 0,
    ("b", "c", 0): 0,
    ("c", "d", 0): 0,
    ("a", "b", 1): 1,
    ("b", "c", 1): 1,
    ("c", "d", 1): 1,
    ("a", "b", 2): 2,
    ("b", "c", 2): 2,
    ("c", "d", 2): 2,
    ("a", "b", 3): 3,
    ("b", "c", 3): 3,
    ("c", "d", 3): 3,
    ("a", "b", 4): "ramp",
    ("b", "c", 4): "ramp",
}
adversarial_target_lanes = {
    0: [0, 1],
    1: [0, 1, 2],
    2: [1, 2, 3],
    3: [2, 3],
    "ramp": [3],
}

lane_path_map_dict = {
    "CutinEnv": cutin_lane_path_map,
    "HighwayEnv": highway_lane_path_map,
    "AdversarialMergeEnv": adversarial_lane_path_map,
    "MergeEnv": merge_lane_path_map,
}


lane_id_refline_dict = {
    "CutinEnv": cutin_lane_path,
    "HighwayEnv": highway_lane_path,
    "AdversarialMergeEnv": adversarial_lane_path,
    "MergeEnv": merge_lane_path,
}


lane_id_targetlane_dict = {
    "CutinEnv": cutin_target_lanes,
    "HighwayEnv": highway_target_lanes,
    "AdversarialMergeEnv": adversarial_target_lanes,
    "MergeEnv": merge_target_lanes,
}
