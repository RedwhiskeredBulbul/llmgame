from __future__ import annotations

import random
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle

Observation = np.ndarray


class CutinEnv(AbstractEnv):
    """双车道高密度车流加塞场景。"""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {"type": "Trajectory_for_mpc_curve"},
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },
                "lanes_count": 3,
                "duration": 900,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "show_trajectorie_points": True,
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        cfg.update({"lanes_count": 2})
        return cfg

    def reset(self, seed=None, **kwargs):
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 【修复】：必须把 seed 原封不动地喂给父类！
        return super().reset(seed=seed, **kwargs)

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 999)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reward(self, action) -> float:  # 删掉类型注解
        print(f"正在计算奖励... 当前 action 类型: {type(action)}, 值: {action}")

        if action is None:
            return 0.0

        reward = 0.0

        if self.vehicle.crashed:
            return -10.0

        target_speed = getattr(self.vehicle, "target_speed", 25.0)
        speed_reward = utils.lmap(self.vehicle.speed, [10.0, target_speed], [0.0, 1.0])
        reward += 1.0 * np.clip(speed_reward, 0.0, 1.0)

        # 兼容 list, tuple, np.ndarray 等各种类型
        try:
            reward -= 0.1 * (float(action[0]) ** 2)
            reward -= 0.5 * (float(action[1]) ** 2)
        except (IndexError, TypeError):
            pass  # 如果动作不是包含两个元素的列表，跳过惩罚，避免报错崩溃

        reward -= 0.05 * abs(self.vehicle.heading)

        return float(reward)

    def _create_road(self) -> None:
        net = RoadNetwork()
        length = 1e4
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        width = StraightLane.DEFAULT_WIDTH
        line_types = [[c, s], [n, s], [n, c]]
        for i in range(self.config["lanes_count"]):
            if i == 0:
                line_type = line_types[0]
            elif i == self.config["lanes_count"] - 1:
                line_type = line_types[-1]
            else:
                line_type = line_types[1]
            # line_type = line_types[min(i, len(line_types) - 1)]
            net.add_lane(
                "a",
                "b",
                StraightLane([0, i * width], [length, i * width], line_types=line_type),
            )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self) -> None:
        road = self.road

        # --- 1. 生成主车 (AV) - 保持不变 ---
        self.controlled_vehicles = []
        ego_lane_index = 1
        ego_vehicle_speed = np.random.randint(30, 40) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("a", "b", ego_lane_index)).position(200, 0),
            speed=ego_vehicle_speed,
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 车辆类型池
        aggressive_type = utils.class_from_path(
            self.config["other_vehicles_types"]["Aggressive"]
        )
        normal_type = utils.class_from_path(
            self.config["other_vehicles_types"]["Normal"]
        )
        cautious_type = utils.class_from_path(
            self.config["other_vehicles_types"]["Cautious"]
        )
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # --- 2. 生成连续密集的交通流 ---
        id_sv_start = 0

        # [控制密度的关键参数]
        start_x = 50  # 车流起始位置 (主车在 200)
        end_x = 600  # 车流结束位置 (根据你的道路长度调整)
        base_spacing = (
            5  # 车辆基础间距(米) - 越小越密集 (建议不要小于 20，否则容易初始碰撞)
        )
        spawn_probability = 1.0  # 该位置生成车辆的概率 (0.0 到 1.0)，增加车流的随机性

        for lane_index in range(self.config["lanes_count"]):
            current_x = start_x
            if lane_index == ego_lane_index:
                continue
            while current_x < end_x:
                # [安全保护]：避免在主车 (x=200) 周围生成背景车，防止一开局就碰撞
                if lane_index == ego_lane_index and 170 < current_x < 240:
                    current_x += base_spacing
                    continue

                # 随机决定是否在这个位置放置车辆
                if np.random.rand() < spawn_probability:
                    vehicle_type = np.random.choice(sv_vehicle_types)
                    lane = road.network.get_lane(("a", "b", lane_index))

                    # 引入位置和速度的随机扰动，让车流看起来更真实
                    pos_noise = np.random.uniform(-5, 5)
                    speed_noise = np.random.uniform(-10, 5)  # 背景车速度比主车慢一点

                    sv = vehicle_type(road, lane.position(current_x, 0), speed=25 / 3.6)
                    sv.id = id_sv_start
                    sv.type = "car"
                    sv.target_speed = 25 / 3.6  # 设置随机目标速度

                    road.vehicles.append(sv)
                    id_sv_start += 1

                # 推进到下一个空位
                current_x += base_spacing + np.random.uniform(0, 15)
        road = self.road
