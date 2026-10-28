"""Seeded ablation environments with no access to controller variant names.

Background traffic uses the repository's existing IDM/MOBIL behaviour models.
No explicit 'if probe then yield' behaviour is added.
"""
from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register, registry

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.behavior import AggressiveVehicle_df, NormalVehicle_df, CautiousVehicle_df


LABELS = ("Cautious", "Normal", "Aggressive")
TYPES = (CautiousVehicle_df, NormalVehicle_df, AggressiveVehicle_df)
ENV_IDS = {
    "merge": "Ablation_LowSpeedMerge-v0",
    "cutin": "Ablation_CutInConflict-v0",
    "traffic": "Ablation_MixedTraffic-v0",
}


class AblationEnv(AbstractEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(action={"type": "Trajectory_for_mpc_curve"},
                   simulation_frequency=10, policy_frequency=10,
                   show_trajectories=False, kind="merge", case_seed=1000,
                   difficulty=1, behaviour_index=0, duration=40)
        return cfg

    def _reset(self):
        self.kind = self.config["kind"]
        difficulty = int(self.config["difficulty"])
        self.horizon = {"merge": 40., "cutin": 20., "traffic": 50.}[self.kind]
        self.road_end = 100. + 10. * difficulty if self.kind == "merge" else 2000.
        self.cutin_time = float(self.np_random.uniform(3., 6.))
        self.cutin_triggered = False
        net = RoadNetwork()
        lanes = 3 if self.kind == "traffic" else 2
        for lane in range(lanes):
            length = self.road_end if self.kind == "merge" and lane == 1 else 2000.
            net.add_lane("a", "b", StraightLane([0., 4. * lane], [length, 4. * lane],
                         line_types=[LineType.STRIPED, LineType.STRIPED], speed_limit=33.3))
        self.road = Road(network=net, np_random=self.np_random, record_history=False)
        ego_lane = 1 if self.kind == "merge" else 0
        ego_speed = (4., 6., 8.)[difficulty] if self.kind == "merge" else (20. if self.kind == "cutin" else 25.)
        ego = self.action_type.vehicle_class(self.road, [20., 4. * ego_lane], speed=ego_speed)
        ego.id, ego.type = 888, "AV"
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)
        self.key_type = int(self.config["behaviour_index"])
        background_types = self.config.get("background_types")
        if background_types is not None:
            background_types = [int(index) for index in background_types]
            if len(background_types) != 2 or any(index not in (0, 1, 2) for index in background_types):
                raise ValueError("background_types must contain two type indices in {0, 1, 2}")

        def add(vid, lane, xpos, speed, type_index):
            vehicle = TYPES[type_index](self.road, [float(xpos), lane * 4.], speed=float(speed))
            vehicle.id, vehicle.type = vid, "car"
            vehicle.enable_lane_change = False
            vehicle.target_speed = (8., 10., 12.)[type_index] if self.kind == "merge" else (25., 30.6, 33.3)[type_index]
            # Independent deterministic traffic: no mutable global RNG consumption.
            self.road.vehicles.append(vehicle)
            return vehicle

        if self.kind == "merge":
            gap = (20., 13., 8.)[difficulty] + float(self.np_random.uniform(-1., 1.))
            # The default keeps both background cars Normal for the frozen
            # key-only protocols.  An all-typed protocol supplies two explicit
            # background indices so every surrounding vehicle is typed.
            background = background_types if background_types is not None else [1, 1]
            self.key_vehicle = add(1, 0, 20. - 5. - gap, ego_speed + 1.5, self.key_type)
            add(2, 0, 20. + 5. + (28., 22., 16.)[difficulty], ego_speed + .5, background[0])
            add(3, 1, 55., ego_speed - 1., background[1])
        elif self.kind == "cutin":
            self.key_vehicle = add(1, 1, 20. + (28., 20., 13.)[difficulty], 18., self.key_type)
            add(2, 0, 90., 18., 1)
            add(3, 1, -15., 23., 2)
        else:
            self.key_vehicle = add(1, 0, 55., 22., self.key_type)
            spacing = (75., 50., 30.)[difficulty]
            vid = 2
            for lane in range(3):
                for idx in range(3):
                    xpos = (55. + spacing * idx + self.np_random.uniform(-3., 3.))
                    if lane == 0 and idx == 0:
                        continue
                    add(vid, lane, xpos, float(self.np_random.uniform(21., 28.)), (vid + self.key_type) % 3)
                    vid += 1

    def step(self, action):
        if self.kind == "cutin" and not self.cutin_triggered and self.time >= self.cutin_time:
            self.key_vehicle.target_lane_index = ("a", "b", 0)
            self.cutin_triggered = True
        return super().step(action)

    def _is_terminated(self):
        return bool(self.vehicle.crashed)

    def _is_truncated(self):
        return bool(self.time >= self.horizon)

    def _reward(self, action):
        return 0.


for kind, env_id in ENV_IDS.items():
    if env_id not in registry:
        register(id=env_id, entry_point="highway_env.envs.ablation_envs:AblationEnv",
                 kwargs={"config": {"kind": kind}})
