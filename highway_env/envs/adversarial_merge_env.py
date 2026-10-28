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


class AdversarialMergeEnv(AbstractEnv):
    """主路对抗车辆堵截自车的匝道汇入场景。"""

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
                "duration": 900,
                "two_lane_length": 200,
                "merge_length": 180,
                "single_lane_length": 500,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "show_trajectorie_points": True,
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        cfg.update({"duration": 500, "lanes_count": 4})
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(seed=seed, **kwargs)

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 999)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reward(self, action: int) -> float:
        return 0.0

    def _create_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        width = StraightLane.DEFAULT_WIDTH  # 默认 4.0

        # --- 第一段 (0 ~ 200m)：自由汇入区 ---
        # 4条主车道 (0, 1, 2, 3)
        for i in range(self.config["lanes_count"]):
            # 第4车道(i=3)的右侧是虚线(s)，允许从匝道自由跨越
            line_types = [c, s] if i == 0 else [n, s]
            net.add_lane(
                "a",
                "b",
                StraightLane([0, i * width], [300, i * width], line_types=line_types),
            )

        # 加速匝道 (作为第5条车道，i=4)
        # 左侧无标线(依赖主路划的虚线)，右侧为道路物理边缘(实线c)
        net.add_lane(
            "a", "b", StraightLane([0, 4 * width], [300, 4 * width], line_types=[n, c])
        )

        # --- 第二段 (200 ~ 250m)：匝道收窄关闭区 ---
        for i in range(self.config["lanes_count"]):
            # 第4车道(i=3)右侧恢复为实线(c)
            line_types = [c, s] if i == 0 else ([n, c] if i == 3 else [n, s])
            net.add_lane(
                "b",
                "c",
                StraightLane([300, i * width], [380, i * width], line_types=line_types),
            )

        # 匝道在此段收窄并标记为不可行驶 (forbidden=True)
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [300, 4 * width], [380, 3 * width], line_types=[n, c], forbidden=True
            ),
        )

        # --- 第三段 (250 ~ 1000m)：常规 4 车道直行 ---
        for i in range(self.config["lanes_count"]):
            line_types = [c, s] if i == 0 else ([n, c] if i == 3 else [n, s])
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [380, i * width], [1000, i * width], line_types=line_types
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", True),
        )

    def _create_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        # --- 1. 放置自车 (位于匝道: node "a"->"b", lane_index 4) ---
        ego_position = 10  # 稍微往前放一点，留出 190m 汇入空间
        ego_vehicle_speed = 60 / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("a", "b", 4)).position(ego_position, 0),
            speed=ego_vehicle_speed,
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = 25  # 自车期望加速汇入
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 准备车型池
        aggressive_type = utils.class_from_path(
            self.config["other_vehicles_types"]["Aggressive"]
        )
        normal_type = utils.class_from_path(
            self.config["other_vehicles_types"]["Normal"]
        )
        id_start = 2
        bg_speed = 60 / 3.6

        # --- 2. 内侧三车道 (lane 0, 1, 2) 添加 60km/h 直行且不变道的车辆 ---
        for lane_idx in [0, 1]:
            # 每条车道错开布置 1~2 辆
            pos_1 = ego_position + self.np_random.uniform(10, 30)
            bg_veh_1 = normal_type(
                road,
                road.network.get_lane(("a", "b", lane_idx)).position(pos_1, 0),
                speed=bg_speed,
            )
            bg_veh_1.id = id_start
            bg_veh_1.type = "car"
            bg_veh_1.target_speed = bg_speed
            bg_veh_1.enable_lane_change = False
            road.vehicles.append(bg_veh_1)
            id_start += 1

            pos_2 = pos_1 + self.np_random.uniform(40, 60)
            bg_veh_2 = normal_type(
                road,
                road.network.get_lane(("a", "b", lane_idx)).position(pos_2, 0),
                speed=bg_speed,
            )
            bg_veh_2.id = id_start
            bg_veh_2.type = "car"
            bg_veh_2.target_speed = bg_speed
            bg_veh_2.enable_lane_change = False
            road.vehicles.append(bg_veh_2)
            id_start += 1

        # --- 3. 放置激进车 (主路最外侧车道 lane 3) ---
        blocker_pos = ego_position - 5  # 放在自车左侧稍靠后的盲区位置
        blocker_speed = ego_vehicle_speed

        self.blocker_vehicle = aggressive_type(
            road,
            road.network.get_lane(("a", "b", 3)).position(blocker_pos, 0),
            speed=blocker_speed,
        )
        self.blocker_vehicle.id = 1
        self.blocker_vehicle.type = "car"
        self.blocker_vehicle.target_speed = blocker_speed
        self.blocker_vehicle.enable_lane_change = False  # 死守主路外侧
        self.blocker_vehicle.color = (200, 0, 0)
        road.vehicles.append(self.blocker_vehicle)

        # --- 4. 激进车后方增加一辆正常跟车，速度 60km/h ---
        follower_pos = blocker_pos - 35
        follower_veh = normal_type(
            road,
            road.network.get_lane(("a", "b", 3)).position(follower_pos, 0),
            speed=bg_speed,
        )
        follower_veh.id = id_start
        follower_veh.type = "car"
        follower_veh.target_speed = bg_speed
        follower_veh.enable_lane_change = False
        road.vehicles.append(follower_veh)

    def step(self, action):
        """
        精准拦截逻辑：保持原代码逻辑不变
        """
        if (
            hasattr(self, "blocker_vehicle")
            and self.blocker_vehicle in self.road.vehicles
        ):
            ego = self.vehicle
            blocker = self.blocker_vehicle
            L = ego.LENGTH

            ego_head = ego.position[0] + L / 2
            blocker_head = blocker.position[0] + blocker.LENGTH / 2
            head_to_head_dist = blocker_head - ego_head

            if head_to_head_dist > 2 * L:
                blocker.target_speed = blocker.speed
            else:
                target_pos = ego_head - 0.5 * L
                error = target_pos - blocker_head
                speed_error = ego.speed - blocker.speed

                kp = 0.03
                kd = 2.0

                desired_speed = ego.speed
                blocker.target_speed = np.clip(desired_speed, 10.0, 70.0 / 3.6)

        obs, original_reward, terminated, truncated, info = super().step(action)
        return obs, original_reward, terminated, truncated, info
