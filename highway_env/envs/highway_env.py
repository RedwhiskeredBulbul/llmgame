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


class HighwayEnv(AbstractEnv):
    """四车道高速巡航，使用项目轨迹动作接口。"""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": 0,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        cfg.update(
            {
                "action": {"type": "Trajectory_for_mpc_curve"},
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "lanes_count": 4,
                "duration": 900,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "show_trajectories": True,
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        return cfg

    def _initialize_native_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        重写原生的路网生成逻辑，将道路长度拉长，防止车跑出地图
        """
        from highway_env.road.road import Road, RoadNetwork

        # 强制将道路物理长度设为 20000 米 (20公里)
        road_length = 20000

        # 生成足够长的直道网络
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"], length=road_length, speed_limit=30
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        # 1. 先调用原生 HighwayEnv 的车辆生成逻辑（它会生成主车和一堆 IDM 背景车）
        self._initialize_native_vehicles()

        # 2. 为主车手动打上 id 和 type 标签
        for i, vehicle in enumerate(self.controlled_vehicles):
            vehicle.id = 888 + i
            vehicle.type = "AV"
            if not hasattr(vehicle, "target_speed"):
                vehicle.target_speed = 26.0

        # 3. 为所有背景车手动打上 id 和 type 标签
        id_sv_start = 1
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.id = id_sv_start
                vehicle.type = "car"
                id_sv_start += 1

    def _reward(self, action) -> float:
        """
        覆盖原生 reward，使用与 _1_2_HwLaneChangeEnv 一致的连续动作惩罚奖励函数
        """
        if action is None:
            return 0.0

        reward = 0.0
        if self.vehicle.crashed:
            return -10.0

        target_speed = getattr(self.vehicle, "target_speed", 25.0)
        speed_reward = utils.lmap(self.vehicle.speed, [10.0, target_speed], [0.0, 1.0])
        reward += 1.0 * np.clip(speed_reward, 0.0, 1.0)

        # 惩罚过大的加速度和转向
        try:
            reward -= 0.1 * (float(action[0]) ** 2)
            reward -= 0.5 * (float(action[1]) ** 2)
        except (IndexError, TypeError):
            pass

        reward -= 0.05 * abs(self.vehicle.heading)
        return float(reward)

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def reset(self, seed=None, **kwargs):
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return super().reset(seed=seed, **kwargs)
