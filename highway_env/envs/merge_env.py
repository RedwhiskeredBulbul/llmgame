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


class MergeEnv(AbstractEnv):
    """标准匝道汇入，使用项目轨迹动作接口。"""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
            }
        )
        cfg.update(
            {
                "action": {"type": "Trajectory_for_mpc_curve"},
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "duration": 500,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "show_trajectories": True,
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        return cfg

    def _initialize_native_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
            speed += self.np_random.uniform(-1.0, 1.0)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110.0, 0.0), speed=20.0
        )
        merging_v.target_speed = 30.0
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle

    def _reward(self, action) -> float:
        # 使用你项目中统一的连续动作奖励函数
        if action is None:
            return 0.0
        reward = 0.0
        if self.vehicle.crashed:
            return -10.0

        target_speed = getattr(self.vehicle, "target_speed", 25.0)
        speed_reward = utils.lmap(self.vehicle.speed, [10.0, target_speed], [0.0, 1.0])
        reward += 1.0 * np.clip(speed_reward, 0.0, 1.0)

        try:
            reward -= 0.1 * (float(action[0]) ** 2)
            reward -= 0.5 * (float(action[1]) ** 2)
        except:
            pass

        reward -= 0.05 * abs(self.vehicle.heading)
        return float(reward)

    def _rewards(self, action) -> dict:
        """
        覆盖原生 MergeEnv 的 _rewards 方法，
        防止连续动作数组在判断 action in [0, 2] 时触发 NumPy ValueError。
        """
        # 返回一个安全的字典给 info 记录使用即可，实际强化学习用的奖励由 _reward 提供
        return {
            "collision_reward": float(self.vehicle.crashed),
            "speed_reward": float(
                self.vehicle.speed / getattr(self.vehicle, "target_speed", 25.0)
            ),
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        # 调用父类生成逻辑（主车默认在匝道，背景车在主路）
        self._initialize_native_vehicles()

        # 为主车赋予 ID 和类型
        for i, vehicle in enumerate(self.controlled_vehicles):
            vehicle.id = 888 + i
            vehicle.type = "AV"
            if not hasattr(vehicle, "target_speed"):
                vehicle.target_speed = 25.0

        # 为所有背景车赋予 ID 和类型
        id_sv_start = 1
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.id = id_sv_start
                vehicle.type = "car"
                id_sv_start += 1

    def reset(self, seed=None, **kwargs):
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return super().reset(seed=seed, **kwargs)
