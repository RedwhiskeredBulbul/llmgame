import numpy as np
import math
from dataclasses import dataclass
import sys
sys.path.append('e:/项目/东风/第三周/DF_highway_v1.1')
from highway_env.road.lane import CircularLane, SineLane, StraightLane

# TODO：轨迹规划
# 预留停止线信息，预留限速和接口，预留不等间隔时间步长规划功能

# 沿着y轴正方向依次排序: <guidance.lanes>

from utils._from_env import *
class DATA_PROCESS_HW():
    def __init__(self):
        self.ego_states = np.array([0,0,0,0,0,0])  
        self.others_info = {
            'states': np.zeros((0,6)),
            'pred': np.zeros((0,0,3)),
            'ids': [],
            'types': []}
        self.target_idxes = []
        self.near_target_idxes = []
        self.near_permit_idxes = []
        self.lanes = {}
        self.curr_lane_idx = None
        self.auto_state = 1 
    
    def update_data(self, env_name, env):
        ego_state = np.array([env.unwrapped.controlled_vehicles[0].position[0], env.unwrapped.controlled_vehicles[0].position[1], env.unwrapped.controlled_vehicles[0].heading, env.unwrapped.controlled_vehicles[0].speed, env.unwrapped.controlled_vehicles[0].action["acceleration"], env.unwrapped.controlled_vehicles[0].action["steering"]])  # np.ndarray [x, y, heading, vx, vy, a]
        other_states, others_pred, other_ids, other_types = get_hv_preds(env.unwrapped.road.vehicles, env.unwrapped.controlled_vehicles[0].id)   # list of np.ndarray [K, state_dim] 其他车辆预测轨迹
        others_info = {
            'states': other_states, 
            'pred': others_pred,
            'ids': other_ids,   
            'types': other_types
        }

        # curr_lane_idx = lane_path_map_dict[env_name][env.unwrapped.controlled_vehicles[0].lane_index]   
        raw_lane_index = env.unwrapped.controlled_vehicles[0].lane_index
    
        # 针对环岛场景做特殊处理，防止 KeyError
        if env_name == 'huandao-v0':
            # 如果当前车道索引不在字典里，且看起来像是环岛内部节点（包含下划线或特定前缀）
            if raw_lane_index not in lane_path_map_dict[env_name]:
                # 强制指向路径 0，因为环岛内所有碎段在逻辑上都属于主环线
                curr_lane_idx = 0 
            else:
                curr_lane_idx = lane_path_map_dict[env_name][raw_lane_index]
        else:
            # 其他场景保持原样
            curr_lane_idx = lane_path_map_dict[env_name][raw_lane_index]    # 当前车道索引 -> 当前path索引
        target_idxes = lane_id_targetlane_dict[env_name][curr_lane_idx]   # List[int] 待评估车道索引
        lanes = lane_id_refline_dict[env_name]     # dict {lane_idx: np.ndarray(N,2)} 所有车道中心线
        
        self.ego_state = ego_state
        self.others_info = others_info
        self.lanes = lanes
        
        self.target_idxes = target_idxes
        
        self.near_target_idxes = target_idxes

        # 测试加塞用
        # self.near_target_idxes = [target_idxes[0]]

        self.near_permit_idxes = target_idxes
        self.curr_lane_idx = curr_lane_idx
        # 根据自车位置匹配当前车道上的路径点
        lane_points = self.lanes[self.curr_lane_idx]
        lane_xy = np.array([[p.x, p.y] for p in lane_points])
        dists = np.hypot(lane_xy[:,0] - self.ego_state[0], lane_xy[:,1] - self.ego_state[1])
        nearest_idx = np.argmin(dists)
        self.curr_lane_point_idx = nearest_idx
    
    

    def mpc_bound(self, target_index):
        curr_lane = self.lanes[self.curr_lane_idx]
        
        cur_lane_centers = []
        cur_lane_d = []
        for point in curr_lane:
            cur_lane_centers.append([point.x, point.y])
            cur_lane_d.append([point.x, point.to_left,point.to_right]) 
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
        
        return cur_lane_centers, cur_lane_d, target_lane_centers, target_lane_d


@dataclass
class Point:
    x: float
    y: float
    heading: float
    speed: float = 0.0
    curvature: float = 0.0
    to_left: float = 2.0
    to_right: float = 2.0
    point_type: int = 0  #点属性： 0-普通点,2-左拐、左掉头(持续整个左拐、左掉头),3-右拐、右掉头(持续整个右拐、右掉头), 4-减速带, 5-人行横道, 6-停止线
    left_traverse_flag: int = 0  # 左侧可通行标志：0-未知的跨越属性, 1-虚线, 2-实线, 3-物理边界
    right_traverse_flag: int = 0  # 右侧可通行标：0-未知的跨越属性, 1-虚线, 2-实线, 3-物理边界

'设计一个简单的HV轨迹预测函数，以8s-16个轨迹点的形式输出'
def get_hv_preds(
        other_vehicles, 
        av_id,
        N = 16,
        dt = 0.5, 
        n_int = 6
        ):
    other_states = []
    hv_ids = []
    other_types = []
    observed_vehicles = [vehicle for vehicle in other_vehicles if vehicle.id != av_id]
    for vehicle in observed_vehicles:
        other_states.append([vehicle.position[0], vehicle.position[1], vehicle.heading, vehicle.speed, vehicle.action["acceleration"], 0])
        hv_ids.append(vehicle.id)
        other_types.append(vehicle.type)
    other_states = np.array(other_states).reshape(-1, n_int)
    
    # 处理 other_states: 预测到 discrete time steps (N steps)
    other = np.asarray(other_states) if other_states is not None else np.zeros((0,6))
    if other.ndim == 1 and other.size > 0:
        other = other.reshape(1, -1)

    n_other = other.shape[0] if other.size>0 else 0
    # precompute other vehicles predicted positions at t=1..N (relative to current time)
    other_preds = np.zeros((n_other, N+1, 6), dtype=float)  # include t=0 if needed
    for j in range(n_other):
        ox, oy, otheta, ov = float(other[j,0]), float(other[j,1]), float(other[j,2]), float(other[j,3])
        
        for k in range(N+1):
            tpred = k * dt
            other_preds[j, k, 0] = ox + ov * np.cos(otheta) * tpred
            other_preds[j, k, 1] = oy + ov * np.sin(otheta) * tpred
            other_preds[j, k, 2] = otheta
            other_preds[j, k, 3] = ov
            other_preds[j, k, 4] = observed_vehicles[j].LENGTH
            other_preds[j, k, 5] = observed_vehicles[j].WIDTH
    return other_states, other_preds, hv_ids, other_types # (n_other, N+1, 2)


'提取地图信息：不同场景的车道线信息、车辆之间的交互关系信息'
# 输入起、终点、左右边界到车道中心线的距离, 输出两点之间连线构成的车道参考线（Point类的格式）
def build_refline_from_endpoints(start, end, start_to_lr=(2.0, 2.0), end_to_lr=(2.0, 2.0), step=1.0, speed=120/3.6):
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
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    d2l = end_to_lr[0] - start_to_lr[0]
    d2r = end_to_lr[1] - start_to_lr[1]
    
    dist = math.hypot(dx, dy)
    if dist <= 0.0:
        heading = 0.0
        return [Point(x=sx, y=sy, heading=heading, speed=float(speed), curvature=0.0,
                      to_left=float(start_to_lr[0]), to_right=float(start_to_lr[1]))]
    
    intervals = max(1, int(math.ceil(dist / float(step))))
    heading =math.atan2(dy, dx)
    pts = []
    for i in range(intervals + 1):
        t = i / intervals
        x = sx + t * dx
        y = sy + t * dy
        to_left = start_to_lr[0] + t * d2l
        to_right = start_to_lr[1] + t * d2r
        pts.append(Point(x=float(x),
                         y=float(y),
                         heading=float(heading),
                         speed=float(speed),
                         curvature=0.0,
                         to_left=float(to_left),
                         to_right=float(to_right)))
    return pts

# 轨迹平滑
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
        # 返回浅复制（保持类型一致）
        return [Point(x=p.x, y=p.y, heading=p.heading, speed=p.speed,
                      curvature=p.curvature, to_left=p.to_left, to_right=p.to_right)
                for p in pts]

    # 提取属性数组
    x = np.asarray([p.x for p in pts], dtype=float)
    y = np.asarray([p.y for p in pts], dtype=float)
    # heading 使用弧度并 unwrap 以避免 +/-180 度跳变
    h_rad = np.asarray([p.heading for p in pts], dtype=float)
    s = np.asarray([p.speed for p in pts], dtype=float)
    k = np.asarray([p.curvature for p in pts], dtype=float)
    tl = np.asarray([p.to_left for p in pts], dtype=float)
    tr = np.asarray([p.to_right for p in pts], dtype=float)

    def moving_average(arr, w):
        if w <= 1:
            return arr.copy()
        pad = w // 2
        # 边界使用 edge 填充以避免靠近端点的值被拉偏
        arr_pad = np.pad(arr, pad_width=pad, mode='edge')
        kernel = np.ones(w, dtype=float) / float(w)
        sm = np.convolve(arr_pad, kernel, mode='valid')
        # 若 w 为偶数，convolve 的 valid 长度仍与原数组一致，因为 pad 为 floor(w/2)
        return sm

    # 多次迭代平滑
    for _ in range(max(1, int(iterations))):
        x = moving_average(x, window)
        y = moving_average(y, window)
        h_rad = np.unwrap(h_rad)  # 保证连续性
        h_rad = moving_average(h_rad, window)
        s = moving_average(s, window)
        k = moving_average(k, window)
        tl = moving_average(tl, window)
        tr = moving_average(tr, window)

    # 构造新的 Point 列表（保留原点的其它属性如果需要可以扩展）
    smoothed = []
    for i in range(n):
        orig = pts[i]
        smoothed.append(Point(
            x=float(x[i]),
            y=float(y[i]),
            heading=float(h_rad[i]),
            speed=float(s[i]),
            curvature=float(k[i]),
            to_left=float(tl[i]),
            to_right=float(tr[i])
        ))

    return smoothed

## 1. 变道博弈
# 1-1 变道时后方快速来车
_1_1_lane_ends = {
    ('a', 'b', 0): [(0, 0), (1e4, 0)],
    ('a', 'b', 1): [(0, 4), (1e4, 4)],
}
# 车道中心线
_1_1_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1], 
        start_to_lr= (
            2.0 + list(_1_1_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(_1_1_lane_ends.keys())) - list(_1_1_lane_ends.keys()).index(lane_id)) - 2.0
        ), 
        end_to_lr= (
            2.0 + list(_1_1_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(_1_1_lane_ends.keys())) - list(_1_1_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        step=1.0, 
        speed=120/3.6
        )
    for lane_id, ends in _1_1_lane_ends.items()
}
# 车道连接构成路径点（TODO：是否需要平滑）
_1_1_lane_path = {
    0: _1_1_lane_refline[('a', 'b', 0)],
    1: _1_1_lane_refline[('a', 'b', 1)],
}
# 车道和路径的对应关系
_1_1_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
}
# target_lanes对应关系
# _1_1_target_lanes = {
#     0: [0, 1],
#     1: [0, 1],
# }
_1_1_target_lanes = {
    0: [0, 1],
    1: [0, 1],
}
# 1-2 快速变道超车
_1_2_lane_ends = {
    ('a', 'b', 0): [(0, 0), (1e4, 0)],
    ('a', 'b', 1): [(0, 4), (1e4, 4)],
    ('a', 'b', 2): [(0, 8), (1e4, 8)],
}
# 车道中心线
_1_2_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1], 
        start_to_lr= (
            2.0 + list(_1_2_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(_1_2_lane_ends.keys())) - list(_1_2_lane_ends.keys()).index(lane_id)) - 2.0
        ), 
        end_to_lr= (
            2.0 + list(_1_2_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(_1_2_lane_ends.keys())) - list(_1_2_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        step=1.0, 
        speed=120/3.6
        )
    for lane_id, ends in _1_2_lane_ends.items()
}
# 车道连接构成路径点（TODO：是否需要平滑）
_1_2_lane_path = {
    0: _1_2_lane_refline[('a', 'b', 0)],
    1: _1_2_lane_refline[('a', 'b', 1)],
    "path2": _1_2_lane_refline[('a', 'b', 2)],
}
# 车道和路径的对应关系
_1_2_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
    ('a', 'b', 2): "path2",
}
# target_lanes对应关系
_1_2_target_lanes = {
    0: [0, 1],
    1: [0, 1, "path2"],
    "path2": [1, "path2"],
}


## 2. 绕行博弈
# 2-1 车道内绕行
_2_1_lane_ends = {
    ('a', 'b', 0): [(0, 0), (1e4, 0)],
    ('a', 'b', 1): [(0, 5), (1e4, 5)],
}
# 车道中心线
_2_1_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1],
        start_to_lr= (
            2.5 + list(_2_1_lane_ends.keys()).index(lane_id) * 5,
            5 * (len(list(_2_1_lane_ends.keys())) - list(_2_1_lane_ends.keys()).index(lane_id)) - 2.5
        ),
        end_to_lr= (
            2.5 + list(_2_1_lane_ends.keys()).index(lane_id) * 5,
            5 * (len(list(_2_1_lane_ends.keys())) - list(_2_1_lane_ends.keys()).index(lane_id)) - 2.5
        ),
        step=1.0, 
        speed=120/3.6
        )
    for lane_id, ends in _2_1_lane_ends.items()
}
# 车道连接构成路径点（TODO：是否需要平滑）
_2_1_lane_path = {
    0: _2_1_lane_refline[('a', 'b', 0)],
    1: _2_1_lane_refline[('a', 'b', 1)],
}
# 车道和路径的对应关系
_2_1_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
}
# target_lanes对应关系
_2_1_target_lanes = {
    0: [0],
    1: [1],
}

# 2-2 借道超车绕行
_2_2_lane_ends = {
    ('a', 'b', 0): [(0, 0), (1e4, 0)],
    ('a', 'b', 1): [(0, 4), (1e4, 4)],
}
# 车道中心线
_2_2_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1], 
        start_to_lr= (
            2.0 + list(_2_2_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(_2_2_lane_ends.keys())) - list(_2_2_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        end_to_lr= (
            2.0 + list(_2_2_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(_2_2_lane_ends.keys())) - list(_2_2_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        step=1.0, 
        speed=120/3.6
        )
    for lane_id, ends in _2_2_lane_ends.items()
}
# 车道连接构成路径点（TODO：是否需要平滑）
_2_2_lane_path = {
    0: _2_2_lane_refline[('a', 'b', 0)],
    1: _2_2_lane_refline[('a', 'b', 1)],
}
# 车道和路径的对应关系
_2_2_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
}
# target_lanes对应关系
_2_2_target_lanes = {
    0: [0],
    1: [0, 1],
}


# 3. 自车合入博弈
# 3-1 匝道合入
_3_1_lane_ends = {
    ('a', 'b', 0): [(0, 0), (500, 0)],
    ('a', 'b', 1): [(0, 4), (500, 4)],
    ('b', 'c', 0): [(500, 0), (500+100, 0)],
    ('b', 'c', 1): [(500, 4), (500+100, 0)],
    ('c', 'd', 0): [(500+100, 0), (500+100+500, 0)],
}
# 车道中心线
_3_1_lane_refline = {
    ('a', 'b', 0): build_refline_from_endpoints(
                    start=_3_1_lane_ends[('a', 'b', 0)][0],
                    end=_3_1_lane_ends[('a', 'b', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 6),
                    step=1.0, 
                    speed=80/3.6
                ),
    ('a', 'b', 1): build_refline_from_endpoints(
                    start=_3_1_lane_ends[('a', 'b', 1)][0],
                    end=_3_1_lane_ends[('a', 'b', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (6, 2),
                    step=1.0, 
                    speed=80/3.6
                ),
    ('b', 'c', 0): build_refline_from_endpoints(
                    start=_3_1_lane_ends[('b', 'c', 0)][0],
                    end=_3_1_lane_ends[('b', 'c', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=80/3.6
                ),
    ('b', 'c', 1): build_refline_from_endpoints(
                    start=_3_1_lane_ends[('b', 'c', 1)][0],
                    end=_3_1_lane_ends[('b', 'c', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=80/3.6
                ),
    ('c', 'd', 0): build_refline_from_endpoints(
                    start=_3_1_lane_ends[('c', 'd', 0)][0],
                    end=_3_1_lane_ends[('c', 'd', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=80/3.6
                ),
}
# 车道连接构成路径点（TODO：是否需要平滑）
_3_1_lane_path = {
    0: _3_1_lane_refline[('a', 'b', 0)] + _3_1_lane_refline[('b', 'c', 0)] + _3_1_lane_refline[('c', 'd', 0)],
    1: _3_1_lane_refline[('a', 'b', 1)] #+ _3_1_lane_refline[('b', 'c', 1)] + _3_1_lane_refline[('c', 'd', 0)],
}
    # 轨迹平滑
_3_1_lane_path[1] = smooth_refline(_3_1_lane_path[1])
# 车道和路径的对应关系
_3_1_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
    ('b', 'c', 0): 0,
    ('b', 'c', 1): 1,
    ('c', 'd', 0): 0,
}
# target_lanes对应关系
_3_1_target_lanes = {
    0: [0],
    1: [0],
}

# 3-2 车道合并
part = 0.72
ramp_y = 6.5 + StraightLane.DEFAULT_WIDTH * 2
_3_2_lane_ends = {
    ('a', 'b', 0): [(200, 0), (200 + part * 380, 0)],
    ('b', 'c', 0): [(200 + part * 380, 0), (200 + 380, 0)],
    ('c', 'd', 0): [(200 + 380, 0), (200 + 380 + 500, 0)],
    ('j', 'k', 0): [(200, ramp_y), (200 + part * 380, ramp_y - part * ramp_y)],
    ('k', 'c', 0): [(200 + part * 380, ramp_y - part * ramp_y), (200 + 380, 0)],
}
# 车道中心线
_3_2_lane_refline = {
    ('a', 'b', 0): build_refline_from_endpoints(
                    start=_3_2_lane_ends[('a', 'b', 0)][0],
                    end=_3_2_lane_ends[('a', 'b', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'c', 0): build_refline_from_endpoints(
                    start=_3_2_lane_ends[('b', 'c', 0)][0],
                    end=_3_2_lane_ends[('b', 'c', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('c', 'd', 0): build_refline_from_endpoints(
                    start=_3_2_lane_ends[('c', 'd', 0)][0],
                    end=_3_2_lane_ends[('c', 'd', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('j', 'k', 0): build_refline_from_endpoints(
                    start=_3_2_lane_ends[('j', 'k', 0)][0],
                    end=_3_2_lane_ends[('j', 'k', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('k', 'c', 0): build_refline_from_endpoints(
                    start=_3_2_lane_ends[('k', 'c', 0)][0],
                    end=_3_2_lane_ends[('k', 'c', 0)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6   
                ),
}
# 车道连接构成路径点（TODO：是否需要平滑）
_3_2_lane_path = {
    0: _3_2_lane_refline[('a', 'b', 0)] + _3_2_lane_refline[('b', 'c', 0)] + _3_2_lane_refline[('c', 'd', 0)],
    1: _3_2_lane_refline[('j', 'k', 0)] + _3_2_lane_refline[('k', 'c', 0)] + _3_2_lane_refline[('c', 'd', 0)],
}
_3_2_lane_path[1] = smooth_refline(_3_2_lane_path[1])
# 车道和路径的对应关系
_3_2_lane_path_map = {
    ('a', 'b', 0): 0,
    ('b', 'c', 0): 0,
    ('c', 'd', 0): 0,
    ('j', 'k', 0): 1,
    ('k', 'c', 0): 1,
}
# target_lanes对应关系
_3_2_target_lanes = {
    0: [0],
    1: [1],
}


# 4. 周车合入博弈
# 4-1 匝道合入
_4_1_lane_ends = {
    ('a', 'b', 0): [(0, 0), (200, 0)],
    ('a', 'b', 1): [(0, 4), (200, 4)],
    ('b', 'c', 0): [(200, 0), (280, 0)],
    ('b', 'c', 1): [(200, 4), (280, 0)],
    ('c', 'd', 0): [(280, 0), (780, 0)],
}
# 车道中心线
_4_1_lane_refline = {
    ('a', 'b', 0): build_refline_from_endpoints(
                    start=_4_1_lane_ends[('a', 'b', 0)][0],
                    end=_4_1_lane_ends[('a', 'b', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 6),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('a', 'b', 1): build_refline_from_endpoints(
                    start=_4_1_lane_ends[('a', 'b', 1)][0],
                    end=_4_1_lane_ends[('a', 'b', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (6, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'c', 0): build_refline_from_endpoints(
                    start=_4_1_lane_ends[('b', 'c', 0)][0],
                    end=_4_1_lane_ends[('b', 'c', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'c', 1): build_refline_from_endpoints(
                    start=_4_1_lane_ends[('b', 'c', 1)][0],
                    end=_4_1_lane_ends[('b', 'c', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('c', 'd', 0): build_refline_from_endpoints(
                    start=_4_1_lane_ends[('c', 'd', 0)][0],
                    end=_4_1_lane_ends[('c', 'd', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
}
# 车道连接构成路径点（TODO：是否需要平滑）
_4_1_lane_path = {
    0: _4_1_lane_refline[('a', 'b', 0)] + _4_1_lane_refline[('b', 'c', 0)] + _4_1_lane_refline[('c', 'd', 0)],
    1: _4_1_lane_refline[('a', 'b', 1)] + _4_1_lane_refline[('b', 'c', 1)] + _4_1_lane_refline[('c', 'd', 0)],
}
_4_1_lane_path[1] = smooth_refline(_4_1_lane_path[1])
# 车道和路径的对应关系
_4_1_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
    ('b', 'c', 0): 0,
    ('b', 'c', 1): 1,
    ('c', 'd', 0): 0,
}
# target_lanes对应关系
_4_1_target_lanes = {
    0: [0],
    1: [1],
}

# 4-2 车道合并
part = 0.72
ramp_y = 6.5 + StraightLane.DEFAULT_WIDTH * 2
_4_2_lane_ends = {
    ('a', 'b', 0): [(200, 0), (200 + part * 180, 0)],
    ('b', 'c', 0): [(200 + part * 180, 0), (200 + 180, 0)],
    ('c', 'd', 0): [(200 + 180, 0), (200 + 180 + 500, 0)],
    ('j', 'k', 0): [(200, ramp_y), (200 + part * 180, ramp_y - part * ramp_y)],
    ('k', 'c', 0): [(200 + part * 180, ramp_y - part * ramp_y), (200 + 180, 0)],
}
# 车道中心线
_4_2_lane_refline = {
    ('a', 'b', 0): build_refline_from_endpoints(
                    start=_4_2_lane_ends[('a', 'b', 0)][0],
                    end=_4_2_lane_ends[('a', 'b', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'c', 0): build_refline_from_endpoints(
                    start=_4_2_lane_ends[('b', 'c', 0)][0],
                    end=_4_2_lane_ends[('b', 'c', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('c', 'd', 0): build_refline_from_endpoints(
                    start=_4_2_lane_ends[('c', 'd', 0)][0],
                    end=_4_2_lane_ends[('c', 'd', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('j', 'k', 0): build_refline_from_endpoints(
                    start=_4_2_lane_ends[('j', 'k', 0)][0],
                    end=_4_2_lane_ends[('j', 'k', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('k', 'c', 0): build_refline_from_endpoints(
                    start=_4_2_lane_ends[('k', 'c', 0)][0],
                    end=_4_2_lane_ends[('k', 'c', 0)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6   
                ),
}
# 车道连接构成路径点（TODO：是否需要平滑）
_4_2_lane_path = {
    0: _4_2_lane_refline[('a', 'b', 0)] + _4_2_lane_refline[('b', 'c', 0)] + _4_2_lane_refline[('c', 'd', 0)],
    1: _4_2_lane_refline[('j', 'k', 0)] + _4_2_lane_refline[('k', 'c', 0)] + _4_2_lane_refline[('c', 'd', 0)],
}
_4_2_lane_path[1] = smooth_refline(_4_2_lane_path[1])
# 车道和路径的对应关系
_4_2_lane_path_map = {
    ('a', 'b', 0): 0,
    ('b', 'c', 0): 0,
    ('c', 'd', 0): 0,
    ('j', 'k', 0): 1,
    ('k', 'c', 0): 1,
}
# target_lanes对应关系
_4_2_target_lanes = {
    0: [0],
    1: [1],
}


# DF_5_1 临时 出口博弈
part = 0.72
ramp_y = 6.5 + StraightLane.DEFAULT_WIDTH * 3
ends = [200, 180, 500]
_5_1_lane_ends = {
    ('a', 'b', 0): [(0, 0), (200, 0)],
    ('a', 'b', 1): [(0, 4), (200, 4)],
    ('b', 'c', 0): [(200, 0), (200 + (1-part) * 180, 0)],
    ('b', 'c', 1): [(200, 4), (200 + (1-part) * 180, 4)],
    ('c', 'd', 0): [(200 + (1-part) * 180, 0), (200+180+500, 0)],
    ('c', 'd', 1): [(200 + (1-part) * 180, 4), (200+180+500, 4)],
    ('b', 'j', 0): [(sum(ends[:1]), 4), (sum(ends[:1])+ (1-part) * (sum(ends[:2])-sum(ends[:1])), ramp_y - part * ramp_y)],
    ('j', 'k', 0): [(200 +(1-part) * 180, ramp_y - part * ramp_y), (200 + 180, ramp_y)],
}
# 车道中心线
_5_1_lane_refline = {
    ('a', 'b', 0): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('a', 'b', 0)][0],
                    end=_5_1_lane_ends[('a', 'b', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 6),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('a', 'b', 1): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('a', 'b', 1)][0],
                    end=_5_1_lane_ends[('a', 'b', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (6, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'c', 0): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('b', 'c', 0)][0],
                    end=_5_1_lane_ends[('b', 'c', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 6),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'c', 1): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('b', 'c', 1)][0],
                    end=_5_1_lane_ends[('b', 'c', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (6, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('c', 'd', 0): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('c', 'd', 0)][0],
                    end=_5_1_lane_ends[('c', 'd', 0)][1],
                    start_to_lr= (2, 6),
                    end_to_lr= (2, 6),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('c', 'd', 1): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('c', 'd', 1)][0],
                    end=_5_1_lane_ends[('c', 'd', 1)][1],
                    start_to_lr= (6, 2),
                    end_to_lr= (6, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('b', 'j', 0): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('b', 'j', 0)][0],
                    end=_5_1_lane_ends[('b', 'j', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
    ('j', 'k', 0): build_refline_from_endpoints(
                    start=_5_1_lane_ends[('j', 'k', 0)][0],
                    end=_5_1_lane_ends[('j', 'k', 0)][1],
                    start_to_lr= (2, 2),
                    end_to_lr= (2, 2),
                    step=1.0, 
                    speed=120/3.6
                ),
}

# 车道连接构成路径点（TODO：是否需要平滑）
_5_1_lane_path = {
    0: _5_1_lane_refline[('a', 'b', 0)] + _5_1_lane_refline[('b', 'c', 0)] + _5_1_lane_refline[('c', 'd', 0)],
    "path2": _5_1_lane_refline[('a', 'b', 1)] + _5_1_lane_refline[('b', 'j', 0)] + _5_1_lane_refline[('j', 'k', 0)],
}
_5_1_lane_path["path2"] = smooth_refline(_5_1_lane_path["path2"])
# 车道和路径的对应关系
_5_1_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): "path2",
    ('b', 'c', 0): 0,
    ('b', 'c', 1): "path2",
    ('c', 'd', 0): 0,
    ('c', 'd', 1): 1,
    ('b', 'j', 0): "path2",
    ('j', 'k', 0): "path2",
}
# target_lanes对应关系
_5_1_target_lanes = {
    0: ["path2"],
    "path2": ["path2"],
}


# hd 高密度车流
hd_lane_ends = {
    ('a', 'b', 0): [(0, 0), (1e4, 0)],
    ('a', 'b', 1): [(0, 4), (1e3, 4)],
}
# 车道中心线
hd_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1], 
        start_to_lr= (
            2.0 + list(hd_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(hd_lane_ends.keys())) - list(hd_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        end_to_lr= (
            2.0 + list(hd_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(hd_lane_ends.keys())) - list(hd_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        step=1.0, 
        speed=120/3.6
        )
    for lane_id, ends in hd_lane_ends.items()
}
# 车道连接构成路径点（TODO：是否需要平滑）
hd_lane_path = {
    0: hd_lane_refline[('a', 'b', 0)],
    1: hd_lane_refline[('a', 'b', 1)],
}
# 车道和路径的对应关系
hd_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
}
# target_lanes对应关系
hd_target_lanes = {
    0: [0],
    1: [0],
}

# --- 环岛场景 huandao-v0 (西进东出：平行直道 + 分叉引道版) ---
_hd_radius = 20.0
_hd_alpha = 10.0   # 增大 alpha 以体现“分开”的视觉效果
_hd_speed = 30 / 3.6
_hd_N = 5          
_hd_split_dist = 25.0 # 分叉点距离圆心的距离

def _build_arc(a1_deg, a2_deg):
    """适配 Y 轴向下坐标系的圆弧生成"""
    a1, a2 = np.deg2rad(a1_deg), np.deg2rad(a2_deg)
    
    # 1. 计算这段圆弧的实际物理长度 (弧长 = 半径 * 弧度差)
    arc_length = _hd_radius * abs(a1 - a2)
    
    # 2. 根据步长（1.0m）计算需要的点数，至少保证 2 个点
    num_points = max(2, int(arc_length / 1.0)) 
    
    # 3. 进行等距离采样
    angles = np.linspace(a1, a2, num_points)
    
    return [Point(x=_hd_radius*np.cos(a), y=_hd_radius*np.sin(a), 
                  heading=a - np.pi/2, speed=_hd_speed) for a in angles]

def _get_p(angle_deg):
    """获取圆周节点物理坐标"""
    return [_hd_radius*np.cos(np.deg2rad(angle_deg)), _hd_radius*np.sin(np.deg2rad(angle_deg))]

# --- 第一步：计算关键坐标点 ---
p_we = _get_p(180 - _hd_alpha)  # 西入口节点 (~ -18, 8)
p_wx = _get_p(180 + _hd_alpha)  # 西出口节点 (~ -18, -8)
p_ee = _get_p(-_hd_alpha)       # 东入口节点 (~ 18, -8)
p_ex = _get_p(_hd_alpha)        # 东出口节点 (~ 18, 8)

# 分叉点坐标 (保持 4m 间距)
w_split_in = [-_hd_split_dist, 2.0]
w_split_out = [-_hd_split_dist, -2.0]
e_split_in = [_hd_split_dist, -2.0]
e_split_out = [_hd_split_dist, 2.0]

# --- 第二步：组装所有参考线段 ---
_huandao_refline = {}

# 1. 西侧路径：远端平行段 -> 分叉过渡段 -> 环岛
# 入口
_huandao_refline[("w_para_in", "w_split", 0)] = build_refline_from_endpoints([-150, 2.0], w_split_in, speed=_hd_speed)
_huandao_refline[("w_split", "we", 0)] = build_refline_from_endpoints(w_split_in, p_we, speed=_hd_speed)
# 出口
_huandao_refline[("wx", "w_merge", 0)] = build_refline_from_endpoints(p_wx, w_split_out, speed=_hd_speed)
_huandao_refline[("w_merge", "w_para_out", 0)] = build_refline_from_endpoints(w_split_out, [-150, -2.0], speed=_hd_speed)

# 2. 东侧路径：环岛 -> 分叉过渡段 -> 远端平行段
# 入口
_huandao_refline[("e_para_in", "e_split", 0)] = build_refline_from_endpoints([150, -2.0], e_split_in, speed=_hd_speed)
_huandao_refline[("e_split", "ee", 0)] = build_refline_from_endpoints(e_split_in, p_ee, speed=_hd_speed)
# 出口
_huandao_refline[("ex", "e_merge", 0)] = build_refline_from_endpoints(p_ex, e_split_out, speed=_hd_speed)
_huandao_refline[("e_merge", "e_para_out", 0)] = build_refline_from_endpoints(e_split_out, [150, 2.0], speed=_hd_speed)

# 3. 环岛内部碎段 (东南和西南象限)
quadrants = {
    "SE": (90 - _hd_alpha, _hd_alpha, "se", "ex"),
    "SW": (180 - _hd_alpha, 90 + _hd_alpha, "we", "sx")
}
for q_name, (start_ang, end_ang, start_node, end_node) in quadrants.items():
    delta = (end_ang - start_ang) / _hd_N
    for i in range(_hd_N):
        n1 = f"{start_node}_{i}" if i > 0 else start_node
        n2 = f"{start_node}_{i+1}" if i < _hd_N - 1 else end_node
        _huandao_refline[(n1, n2, 0)] = _build_arc(start_ang + i * delta, start_ang + (i + 1) * delta)

# 4. 环岛内短连接
_huandao_refline[("sx", "se", 0)] = _build_arc(90 + _hd_alpha, 90 - _hd_alpha)

# --- 第三步：拼接“西进东出”主路径 0 ---
main_path_points = []
main_path_points.extend(_huandao_refline[("w_para_in", "w_split", 0)]) # 平行段
main_path_points.extend(_huandao_refline[("w_split", "we", 0)])        # 分叉段
# 西南象限
for i in range(_hd_N):
    n1 = f"we_{i}" if i > 0 else "we"; n2 = f"we_{i+1}" if i < _hd_N - 1 else "sx"
    main_path_points.extend(_huandao_refline[(n1, n2, 0)])
main_path_points.extend(_huandao_refline[("sx", "se", 0)])            # 南部连接
# 东南象限
for i in range(_hd_N):
    n1 = f"se_{i}" if i > 0 else "se"; n2 = f"se_{i+1}" if i < _hd_N - 1 else "ex"
    main_path_points.extend(_huandao_refline[(n1, n2, 0)])
main_path_points.extend(_huandao_refline[("ex", "e_merge", 0)])       # 合并段
main_path_points.extend(_huandao_refline[("e_merge", "e_para_out", 0)]) # 平行段

# --- 汇总输出 ---
_huandao_lane_path = { 0: main_path_points }
_huandao_lane_path[0] = smooth_refline(_huandao_lane_path[0])
_huandao_lane_path_map = {k: 0 for k in _huandao_refline.keys()}
_huandao_target_lanes = {0: [0]}

# ==========================================
# DF_Highway (标准 4 车道 HighwayEnv)
# ==========================================
# 注意：原生 HighwayEnv 的节点名称是 "0" 和 "1"，而不是 "a" 和 "b"
df_hw_lane_ends = {
    ('0', '1', 0): [(0, 0), (1e4, 0)],
    ('0', '1', 1): [(0, 4), (1e4, 4)],
    ('0', '1', 2): [(0, 8), (1e4, 8)],
    ('0', '1', 3): [(0, 12), (1e4, 12)],
}

# 车道中心线 (这段代码根据上面的 ends 自动生成，保持不变)
df_hw_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1], 
        start_to_lr= (
            2.0 + list(df_hw_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(df_hw_lane_ends.keys())) - list(df_hw_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        end_to_lr= (
            2.0 + list(df_hw_lane_ends.keys()).index(lane_id) * StraightLane.DEFAULT_WIDTH,
            StraightLane.DEFAULT_WIDTH * (len(list(df_hw_lane_ends.keys())) - list(df_hw_lane_ends.keys()).index(lane_id)) - 2.0
        ),
        step=1.0, 
        speed=120/3.6
        )
    for lane_id, ends in df_hw_lane_ends.items()
}

# 车道连接构成路径点 (注意这里也改成了 '0', '1')
df_hw_lane_path = {
    0: df_hw_lane_refline[('0', '1', 0)],
    1: df_hw_lane_refline[('0', '1', 1)],
    2: df_hw_lane_refline[('0', '1', 2)],
    3: df_hw_lane_refline[('0', '1', 3)],
}

# 车道和路径的对应关系 (重点修改这里！)
df_hw_lane_path_map = {
    ('0', '1', 0): 0,
    ('0', '1', 1): 1,
    ('0', '1', 2): 2,
    ('0', '1', 3): 3,
}

# target_lanes 对应关系 (允许左右相邻车道进行规划，保持不变)
df_hw_target_lanes = {
    0: [0, 1],
    1: [0, 1, 2],
    2: [1, 2, 3],
    3: [2, 3],
}

# ==========================================
# DF_Merge (标准 MergeEnv 拓扑)
# ==========================================
# 完整定义主路2条车道 + 1条匝道
df_merge_ends = {
    ('a', 'b', 0): [(0, 0), (150, 0)],     # 主路左车道
    ('a', 'b', 1): [(0, 4), (150, 4)],     # 主路右车道
    ('b', 'c', 0): [(150, 0), (310, 0)],   # 汇入区左车道
    ('b', 'c', 1): [(150, 4), (310, 4)],   # 汇入区右车道
    ('c', 'd', 0): [(310, 0), (450, 0)],   # 汇入后左车道
    ('c', 'd', 1): [(310, 4), (450, 4)],   # 汇入后右车道
    ('j', 'k', 0): [(0, 8), (150, 8)],     # 匝道平行段
    ('k', 'b', 0): [(150, 8), (310, 4)],   # 匝道汇入段 (接入主路右车道)
}

# 批量生成所有的车道中心线 Point 对象
df_merge_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=ends[0], 
        end=ends[1], 
        step=1.0, 
        speed=80/3.6
    )
    for lane_id, ends in df_merge_ends.items()
}

# 拼接出三条完整的逻辑路径 (0:主路左, 1:主路右, "ramp":匝道)
df_merge_lane_path = {
    0: df_merge_lane_refline[('a', 'b', 0)] + df_merge_lane_refline[('b', 'c', 0)] + df_merge_lane_refline[('c', 'd', 0)],
    1: df_merge_lane_refline[('a', 'b', 1)] + df_merge_lane_refline[('b', 'c', 1)] + df_merge_lane_refline[('c', 'd', 1)],
    "ramp": df_merge_lane_refline[('j', 'k', 0)] + df_merge_lane_refline[('k', 'b', 0)] + df_merge_lane_refline[('c', 'd', 1)]
}

# 单独平滑匝道的转弯处
df_merge_lane_path["ramp"] = smooth_refline(df_merge_lane_path["ramp"])

# 映射表：把所有物理短节映射到长路径上
df_merge_lane_path_map = {
    ('a', 'b', 0): 0,
    ('a', 'b', 1): 1,
    ('b', 'c', 0): 0,
    ('b', 'c', 1): 1,
    ('c', 'd', 0): 0,
    ('c', 'd', 1): 1,
    ('j', 'k', 0): "ramp",
    ('k', 'b', 0): "ramp",
}

# 变道博弈的目标集合
df_merge_target_lanes = {
    0: [0, 1],
    1: [0, 1],
    "ramp": [1, "ramp"], # 匝道车可以尝试从ramp汇入主路的右侧车道(1)
}

width = 4.0
ends = [200, 180, 500]
ramp_y = 4 * width + 2

_adv_lane_ends = {
    # 主路第 0 车道
    ('a', 'b', 0): [(0, 0), (300, 0)],
    ('b', 'c', 0): [(300, 0), (380, 0)],
    ('c', 'd', 0): [(380, 0), (1000, 0)],
    # 主路第 1 车道
    ('a', 'b', 1): [(0, width), (300, width)],
    ('b', 'c', 1): [(300, width), (380, width)],
    ('c', 'd', 1): [(380, width), (1000, width)],
    # 主路第 2 车道
    ('a', 'b', 2): [(0, 2*width), (300, 2*width)],
    ('b', 'c', 2): [(300, 2*width), (380, 2*width)],
    ('c', 'd', 2): [(380, 2*width), (1000, 2*width)],
    # 主路第 3 车道 (激进车所在车道)
    ('a', 'b', 3): [(0, 3*width), (300, 3*width)],
    ('b', 'c', 3): [(300, 3*width), (380, 3*width)],
    ('c', 'd', 3): [(380, 3*width), (1000, 3*width)],
    # 匝道 (自车所在车道，并行加速后收窄)
    ('a', 'b', 4): [(0, 4*width), (300, 4*width)],
    ('b', 'c', 4): [(300, 4*width), (380, 3*width)],
}

# 批量生成骨架线
_adv_lane_refline = {
    lane_id: build_refline_from_endpoints(
        start=pts[0], end=pts[1], step=1.0, speed=80/3.6
    ) for lane_id, pts in _adv_lane_ends.items()
}

# 将碎段拼接成长路径 (0, 1, 2, 3 为主路，"ramp" 为匝道)
_adv_lane_path = {
    0: _adv_lane_refline[('a', 'b', 0)] + _adv_lane_refline[('b', 'c', 0)] + _adv_lane_refline[('c', 'd', 0)],
    1: _adv_lane_refline[('a', 'b', 1)] + _adv_lane_refline[('b', 'c', 1)] + _adv_lane_refline[('c', 'd', 1)],
    2: _adv_lane_refline[('a', 'b', 2)] + _adv_lane_refline[('b', 'c', 2)] + _adv_lane_refline[('c', 'd', 2)],
    3: _adv_lane_refline[('a', 'b', 3)] + _adv_lane_refline[('b', 'c', 3)] + _adv_lane_refline[('c', 'd', 3)],
    "ramp": _adv_lane_refline[('a', 'b', 4)] + _adv_lane_refline[('b', 'c', 4)] + _adv_lane_refline[('c', 'd', 3)],
}
_adv_lane_path["ramp"] = smooth_refline(_adv_lane_path["ramp"])

# 将物理网格映射到逻辑路径
_adv_lane_path_map = {
    ('a', 'b', 0): 0, ('b', 'c', 0): 0, ('c', 'd', 0): 0,
    ('a', 'b', 1): 1, ('b', 'c', 1): 1, ('c', 'd', 1): 1,
    ('a', 'b', 2): 2, ('b', 'c', 2): 2, ('c', 'd', 2): 2,
    ('a', 'b', 3): 3, ('b', 'c', 3): 3, ('c', 'd', 3): 3,
    ('a', 'b', 4): "ramp", ('b', 'c', 4): "ramp"
}

# 规划器目标车道可用范围 (允许匝道车看准时机压线去 lane 3)
_adv_target_lanes = {
    0: [0, 1],
    1: [0, 1, 2],
    2: [1, 2, 3],
    3: [2, 3],
    "ramp": [3], 
}

# Curve for DF_sim
# 车道中心线
import pandas as pd

def read_lane_centerline_from_csv(csv_path):
    """
    读取csv文件，生成车道中心线Point列表
    csv格式：x, y, speed, curvature, to_left, to_right
    heading自动根据相邻点计算（切线方向，弧度）
    """
    df = pd.read_csv(csv_path)
    points = []
    xs = df['x'].values
    ys = df['y'].values
    n = len(df)
    # 计算heading（切线方向，弧度），采用中心差分法，保证切线方向准确
    headings = np.zeros(n)
    for i in range(n):
        if n == 1:
            headings[i] = 0.0
        elif i == 0:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            headings[i] = math.atan2(dy, dx)
        elif i == n - 1:
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
            headings[i] = math.atan2(dy, dx)
        else:
            dx = xs[i + 1] - xs[i - 1]
            dy = ys[i + 1] - ys[i - 1]
            # 用三点法计算切线方向，避免曲线段误差
            headings[i] = math.atan2(dy, dx)
    # 保证heading连续（消除跳变）
    headings = np.unwrap(headings)
    for i, row in df.iterrows():
        points.append(Point(
            x=float(row['x']),
            y=float(row['y']),
            heading=float(headings[i]),
            speed=float(row.get('speed', 0.0)),
            curvature=float(row.get('curvature', 0.0)),
            to_left=float(row.get('to_left', 2.0)),
            to_right=float(row.get('to_right', 14.0))
        ))
    return points

# 车道连接构成路径点（TODO：是否需要平滑）
# 获取highway中的车道
_sim_1_lane_path = {
    0: read_lane_centerline_from_csv(csv_path=r'.\highway_env\envs\_DFsim_curveroad_1.csv'),
}
# 车道和路径的对应关系
_sim_1_lane_path_map = {
    ('a', 'b', 0): 0,
    ('b', 'c', 0): 0,
    ('c', 'd', 0): 0,
    ('d', 'e', 0): 0,
}
# target_lanes对应关系
_sim_1_target_lanes = {
    0: [0],
}


# 获取highway中的车道
_sim_2_lane_path = {
    0: read_lane_centerline_from_csv(csv_path=r'.\highway_env\envs\_DFsim_curveroad_2.csv'),
}
# 车道和路径的对应关系
_sim_2_lane_path_map = {
    ('a', 'b', 0): 0,
    ('b', 'c', 0): 0,
    ('c', 'd', 0): 0,
    ('d', 'e', 0): 0,
}
# target_lanes对应关系
_sim_2_target_lanes = {
    0: [0],
}


'汇总的车辆与路径对应关系'
lane_path_map_dict = {
    # 1-1
    'DF_1_1': _1_1_lane_path_map,
    # 1-2
    'DF_1_2': _1_2_lane_path_map,
    # 2-1
    'DF_2_1': _2_1_lane_path_map,
    # 2-2
    'DF_2_2': _2_2_lane_path_map,
    # 3-1
    'DF_3_1': _3_1_lane_path_map,
    # 3-2
    'DF_3_2': _3_2_lane_path_map,
    # 4-1
    'DF_4_1': _4_1_lane_path_map,
    # 4-2
    'DF_4_2': _4_2_lane_path_map,

    # 5-1
    'DF_5_1': _5_1_lane_path_map,

    # sim1
    'DFSim_1': _sim_1_lane_path_map,
    # sim2
    'DFSim_2': _sim_2_lane_path_map,

    # hd
    'DF_hd': hd_lane_path_map,
    'DF_CutIn': _1_2_lane_path_map,
    'huandao-v0': _huandao_lane_path_map,
    'DF_Cruise': _1_2_lane_path_map,
    'Adversarial_merge': _adv_lane_path_map,
    'DF_Highway': df_hw_lane_path_map,
    'DF_Merge': df_merge_lane_path_map,
}

lane_id_refline_dict = {
    # 1-1
    'DF_1_1': {
        path_id: _1_1_lane_path.get(path_id, "") for path_id in list(_1_1_lane_path.keys())
    },
    # 1-2
    'DF_1_2': {
        path_id: _1_2_lane_path.get(path_id, "") for path_id in list(_1_2_lane_path.keys())
    },
    # 2-1
    'DF_2_1': {
        path_id: _2_1_lane_path.get(path_id, "") for path_id in list(_2_1_lane_path.keys())
    },
    # 2-2
    'DF_2_2': {
        path_id: _2_2_lane_path.get(path_id, "") for path_id in list(_2_2_lane_path.keys())
    },
    # 3-1
    'DF_3_1': {
        path_id: _3_1_lane_path.get(path_id, "") for path_id in list(_3_1_lane_path.keys())
    },
    # 3-2
    'DF_3_2': {
        path_id: _3_2_lane_path.get(path_id, "") for path_id in list(_3_2_lane_path.keys())
    },
    # 4-1
    'DF_4_1': {
        path_id: _4_1_lane_path.get(path_id, "") for path_id in list(_4_1_lane_path.keys())
    },
    # 4-2
    'DF_4_2': {
        path_id: _4_2_lane_path.get(path_id, "") for path_id in list(_4_2_lane_path.keys())
    },

    # 5-1
    'DF_5_1': {
        path_id: _5_1_lane_path.get(path_id, "") for path_id in list(_5_1_lane_path.keys())
    },

    # sim1
    'DFSim_1': {
        path_id: _sim_1_lane_path.get(path_id, "") for path_id in list(_sim_1_lane_path.keys())
    },

    # sim2
    'DFSim_2': {
        path_id: _sim_2_lane_path.get(path_id, "") for path_id in list(_sim_2_lane_path.keys())
    },

    # hd
    'DF_hd': {
        path_id: hd_lane_path.get(path_id, "") for path_id in list(hd_lane_path.keys())
    },
    'DF_CutIn': {
        path_id: _1_2_lane_path.get(path_id, "") for path_id in list(_1_2_lane_path.keys())
    },
    'huandao-v0': {                       
        path_id: _huandao_lane_path.get(path_id, "") for path_id in list(_huandao_lane_path.keys())
    },

    'DF_Cruise': {
        path_id: _1_2_lane_path.get(path_id, "") for path_id in list(_1_2_lane_path.keys())
    },

    'Adversarial_merge': {  
        path_id: _adv_lane_path.get(path_id, "") for path_id in list(_adv_lane_path.keys())
    },

    'DF_Highway': {
        path_id: df_hw_lane_path.get(path_id, "") for path_id in list(df_hw_lane_path.keys())
    },

    'DF_Merge': {
        path_id: df_merge_lane_path.get(path_id, "") for path_id in list(df_merge_lane_path.keys())
    },
}

lane_id_targetlane_dict = {
    # 1-1
    'DF_1_1': _1_1_target_lanes,
    # 1-2
    'DF_1_2': _1_2_target_lanes,
    # 2-1
    'DF_2_1': _2_1_target_lanes,
    # 2-2
    'DF_2_2': _2_2_target_lanes,
    # 3-1
    'DF_3_1': _3_1_target_lanes,
    # 3-2
    'DF_3_2': _3_2_target_lanes,
    # 4-1
    'DF_4_1': _4_1_target_lanes,
    # 4-2
    'DF_4_2': _4_2_target_lanes,
    # 5-1
    'DF_5_1': _5_1_target_lanes,

    # sim1
    'DFSim_1': _sim_1_target_lanes,
    # sim2
    'DFSim_2': _sim_2_target_lanes,

    # hd
    'DF_hd': hd_target_lanes,
    'DF_CutIn': _1_2_target_lanes,
    'huandao-v0': _huandao_target_lanes,
    'DF_Cruise': _1_2_target_lanes,
    'Adversarial_merge': _adv_target_lanes,
    'DF_Highway': df_hw_target_lanes,
    'DF_Merge': df_merge_target_lanes,
}

# 为所有已定义的场景生成图（按命名约定）
if __name__ == '__main__':
    from matplotlib import cm
    import matplotlib.pyplot as plt

    def _extract_xy(points):
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return xs, ys

    def plot_scenario(name, lane_refline, lane_path, lane_path_map=None, target_lanes=None, figsize=(8, 6), save_path=None):
        """
        绘制单个场景：
        - lane_refline: dict(lane_id -> list[Point])  -> 用虚线不同颜色绘制
        - lane_path: dict(path_name -> list[Point])   -> 用实线绘制
        - lane_path_map / target_lanes 可用于图例注释（可选）
        """
        if lane_refline is None or lane_path is None:
            return

        fig, ax = plt.subplots(figsize=figsize)
        # 颜色映射，保证不同车道颜色不重复
        lane_ids = list(lane_refline.keys())
        n_colors = max(3, len(lane_ids))
        colors = [cm.tab10.colors[i % 10] for i in range(n_colors)]

        # 绘制 path（实线，较粗）
        path_names = list(lane_path.keys())
        # pick colors for paths (reuse color map)
        for j, path_name in enumerate(path_names):
            print(f"Drawing path {path_name}")
            pts = lane_path[path_name]
            if not pts:
                continue
            xs, ys = _extract_xy(pts)
            # path color: reuse color cycle but offset to be distinct
            color = colors[j % len(colors)]
            ax.plot(xs, ys, linestyle='-', color=color, linewidth=4.5, alpha=0.3, label=f"path {path_name}")
        # 绘制车道中心线（虚线）
        for i, lane_id in enumerate(lane_ids):
            pts = lane_refline[lane_id]
            if not pts:
                continue
            xs, ys = _extract_xy(pts)
            label = f"lane {lane_id}" if lane_path_map is None else f"lane {lane_id} -> {lane_path_map.get(lane_id, '')}"
            ax.plot(xs, ys, linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9, label=label)

        # 可选：高亮 target lanes（在图例中展示）
        # if target_lanes:
        #     txt = "\n".join([f"{k}: {v}" for k, v in target_lanes.items()])
        #     ax.text(0.01, 0.99, txt, transform=ax.transAxes, va='top', ha='left',
        #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)

        ax.set_title(f"Scenario: {name}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis('equal')
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
        # 合并图例，避免重复项
        handles, labels = ax.get_legend_handles_labels()
        # keep unique label entries preserving order
        seen = set()
        unique = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique.append(h)
                unique_labels.append(l)
        ax.legend(unique, unique_labels, fontsize=8, loc='best')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    try:
        plot_scenario("1-1_lane", _1_1_lane_refline, _1_1_lane_path, lane_path_map=_1_1_lane_path_map, target_lanes=_1_1_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("1-2_lane", _1_2_lane_refline, _1_2_lane_path, lane_path_map=_1_2_lane_path_map, target_lanes=_1_2_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("2-1_lane", _2_1_lane_refline, _2_1_lane_path, lane_path_map=_2_1_lane_path_map, target_lanes=_2_1_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("2-2_lane", _2_2_lane_refline, _2_2_lane_path, lane_path_map=_2_2_lane_path_map, target_lanes=_2_2_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("3-1_lane", _3_1_lane_refline, _3_1_lane_path, lane_path_map=_3_1_lane_path_map, target_lanes=_3_1_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("3-2_lane", _3_2_lane_refline, _3_2_lane_path, lane_path_map=_3_2_lane_path_map, target_lanes=_3_2_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("4-1_lane", _4_1_lane_refline, _4_1_lane_path, lane_path_map=_4_1_lane_path_map, target_lanes=_4_1_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("4-2_lane", _4_2_lane_refline, _4_2_lane_path, lane_path_map=_4_2_lane_path_map, target_lanes=_4_2_target_lanes)
    except NameError:
        pass

    try:
        plot_scenario("5-1_lane", _5_1_lane_refline, _5_1_lane_path, lane_path_map=_5_1_lane_path_map, target_lanes=_5_1_target_lanes)
    except NameError:
        pass
