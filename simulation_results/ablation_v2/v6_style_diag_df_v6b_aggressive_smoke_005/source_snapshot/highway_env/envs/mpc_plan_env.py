from __future__ import annotations
import numpy as np
import random
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.vehicle.objects import Obstacle

Observation = np.ndarray

# 用于东风场景随机初始化测试
class _1_2_HwLaneChangeEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },
                
                "lanes_count": 3,
                
                "duration": 900,
                
                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,
                
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
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
    
    def _reward(self, action) -> float: # 删掉类型注解
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
            pass # 如果动作不是包含两个元素的列表，跳过惩罚，避免报错崩溃

        reward -= 0.05 * abs(self.vehicle.heading)

        return float(reward)

    # def _reward(self, action: np.ndarray) -> float:
    #     """
    #     为强化学习设计的连续动作奖励函数
    #     """
    #     # 如果动作是 None 或者不是数组（例如初始化时），给个默认 0 奖励
    #     if action is None or not isinstance(action, np.ndarray):
    #         return 0.0
            
    #     reward = 0.0

    #     # --- 1. 碰撞惩罚 (最高优先级) ---
    #     if self.vehicle.crashed:
    #         # 一旦碰撞，给予严厉惩罚
    #         return -10.0  

    #     # --- 2. 速度效率奖励 ---
    #     # 假设环境初始设定时，自车的 target_speed 是期望的高速巡航速度
    #     target_speed = getattr(self.vehicle, "target_speed", 25.0) 
        
    #     # 使用 utils.lmap 将速度线性映射。
    #     # 假设车速在 10m/s 到 target_speed 之间，获得 0 到 1 的奖励
    #     # 如果超过 target_speed 则不再增加奖励（甚至可以稍微惩罚超速）
    #     speed_reward = utils.lmap(self.vehicle.speed, [10.0, target_speed], [0.0, 1.0])
    #     speed_reward = np.clip(speed_reward, 0.0, 1.0)
        
    #     # 赋予速度奖励一个权重，例如 1.0
    #     reward += 1.0 * speed_reward

    #     # --- 3. 舒适性与平顺性惩罚 (针对连续动作) ---
    #     # 假设 action[0] 是加速度 (-1~1)，action[1] 是转向角 (-1~1)
    #     # 我们希望惩罚剧烈的方向盘打角，保持车辆在车道内稳定行驶
    #     if len(action) >= 2:
    #         steering = action[1]
    #         steering_penalty = -0.1 * (steering ** 2)  # 转向角越大，惩罚呈平方级增加
    #         reward += steering_penalty
            
    #     # --- 4. 车道偏离惩罚 (鼓励行驶在车道中心线上) ---
    #     # self.vehicle.heading 偏离 0 度（假设直道朝向为0）的惩罚
    #     heading_penalty = -0.05 * abs(self.vehicle.heading)
    #     reward += heading_penalty

    #     return float(reward)
    
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
                'a',
                'b',
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
        self.controlled_vehicles = []
        ego_lane_index = np.random.randint(0, self.config['lanes_count'] - 1)
        ego_vehicle_speed = np.random.randint(40, 60) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", 1)
            ).position(20, 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 100
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # slow vehicle
        slow_vehicle_type = np.random.choice(sv_vehicle_types)
        slow_lane = road.network.get_lane(('a', 'b', ego_lane_index))
        slow_lane_position = slow_lane.position(200 + safety_distance + np.random.randint(-15, 15), 0)
        slow_vehicle = slow_vehicle_type(
                    road,
                    slow_lane_position,
                    speed=0 # ego_vehicle_speed + np.random.randint(-6, -4),
                )
        slow_vehicle.id = 0
        slow_vehicle.type = 'car'
        slow_vehicle.target_speed = 0
        road.vehicles.append(slow_vehicle)

        positions_x = [200 - safety_distance, 200, 200 + safety_distance]
        lane_indexes = list(range(self.config['lanes_count']))
        possible_informations = []
        for position in positions_x:
            for lane_index in lane_indexes:
                if (position, lane_index) != (200, ego_lane_index) and (position, lane_index) != (200 + safety_distance, ego_lane_index):
                    speed = ego_vehicle_speed + np.random.randint(-7, 8)
                    possible_informations.append((position, lane_index, speed))
        num_env_vehicles = np.random.randint(1, 3)
        information_to_generate = random.sample(possible_informations, num_env_vehicles)

        id_sv_start = 1
        for position, lane_index, speed in information_to_generate:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('a', 'b', 1))
            lane_position = lane.position(position + np.random.randint(-15, 15), 0)
            sv = vehicle_type(
                        road,
                        lane_position,
                        speed=0,
                    )
            sv.id = id_sv_start
            sv.type = 'car'

            sv.target_speed = 0
            id_sv_start += 1
            road.vehicles.append(sv)

class _2_1_InsideLaneChangeEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },

                "lanes_count": 2,

                "duration": 900,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 1800,
                "screen_height": 900,

            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)
    
    def _reset(self) -> None:
        # ego_lane_index = np.random.randint(0, self.config['lanes_count'] - 1)
        ego_lane_index = 1
        self._create_road(ego_lane_index)
        self._create_vehicles(ego_lane_index)

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 999)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reward(self, action: int) -> float:
        return 0.0

    def _create_road(self, ego_lane_index) -> None:
        net = RoadNetwork()
        length = 1e4
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        width = StraightLane.DEFAULT_WIDTH + 1
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
                'a',
                'b',
                StraightLane([0, i * width], [length, i * width],
                             width = width,
                             line_types=line_type),
                        )
        offset_value = width / 2 
        net.add_lane(
            'c', 'd',
            StraightLane([0, offset_value + ego_lane_index * width],
                         [length, offset_value + ego_lane_index * width],
                         width = width,
                         line_types=[n,n]),
        )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self, ego_lane_index) -> None:
        road = self.road

        self.controlled_vehicles = []

        ego_vehicle_speed = np.random.randint(40, 121) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", ego_lane_index)
            ).position(200 + random.randint(-40, 40), 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 150
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # slow vehicle
        slow_vehicle_type = np.random.choice(sv_vehicle_types)
        slow_lane = road.network.get_lane(('c', 'd', 0))
        slow_lane_position = slow_lane.position(200 + safety_distance + np.random.randint(-15, 15), -1)  # 0
        slow_vehicle = slow_vehicle_type(
                    road,
                    slow_lane_position,
                    speed=0,# ego_vehicle_speed + np.random.randint(-6, -4),
                    enable_lane_change=False
                )
        slow_vehicle.id = 0
        slow_vehicle.type = 'car'
        slow_vehicle.target_speed = 0 #40/3.6
        road.vehicles.append(slow_vehicle)

        positions_x = [200 - safety_distance, 200, 200 + safety_distance]
        lane_indexes = list(range(self.config['lanes_count']))
        possible_informations = []
        for position in positions_x:
            for lane_index in lane_indexes:
                if (position, lane_index) != (200, ego_lane_index) and (position, lane_index) != (200 + safety_distance, ego_lane_index):
                    speed = ego_vehicle_speed + np.random.randint(-7, 8)
                    possible_informations.append((position, lane_index, speed))
        num_env_vehicles = np.random.randint(1, 3)
        information_to_generate = random.sample(possible_informations, num_env_vehicles)

        id_sv_start = 1
        for position, lane_index, speed in information_to_generate:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('a', 'b', lane_index))
            lane_position = lane.position(position + np.random.randint(-15, 15), 0)
            sv = vehicle_type(
                        road,
                        lane_position,
                        speed=speed,
                    )
            sv.id = id_sv_start
            sv.type = 'car'
            id_sv_start += 1
            road.vehicles.append(sv)

class _2_2_TWOLaneChangeEnv(_1_2_HwLaneChangeEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"lanes_count": 2})
        return cfg

    def _create_vehicles(self) -> None:
        road = self.road

        # --- 1. 生成主车 (AV) - 保持不变 ---
        self.controlled_vehicles = []
        ego_lane_index = 1
        ego_vehicle_speed = np.random.randint(30, 40) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", ego_lane_index)
            ).position(200, 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 车辆类型池
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # --- 2. 生成连续密集的交通流 ---
        id_sv_start = 0
        
        # [控制密度的关键参数]
        start_x = 50           # 车流起始位置 (主车在 200)
        end_x = 600            # 车流结束位置 (根据你的道路长度调整)
        base_spacing = 5      # 车辆基础间距(米) - 越小越密集 (建议不要小于 20，否则容易初始碰撞)
        spawn_probability = 1.0 # 该位置生成车辆的概率 (0.0 到 1.0)，增加车流的随机性
        
        for lane_index in range(self.config['lanes_count']):
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
                    lane = road.network.get_lane(('a', 'b', lane_index))
                    
                    # 引入位置和速度的随机扰动，让车流看起来更真实
                    pos_noise = np.random.uniform(-5, 5)
                    speed_noise = np.random.uniform(-10, 5) # 背景车速度比主车慢一点
                    
                    sv = vehicle_type(
                        road,
                        lane.position(current_x, 0),
                        speed = 25 / 3.6
                    )
                    sv.id = id_sv_start
                    sv.type = 'car'
                    sv.target_speed = 25/3.6 # 设置随机目标速度
                    
                    road.vehicles.append(sv)
                    id_sv_start += 1
                
                # 推进到下一个空位
                current_x += base_spacing + np.random.uniform(0, 15)
        road = self.road

        # self.controlled_vehicles = []
        # ego_lane_index = 1
        # ego_vehicle_speed = np.random.randint(40, 121) / 3.6
        # ego_vehicle = self.action_type.vehicle_class(
        #     road,
        #     road.network.get_lane(
        #         ("a", "b", ego_lane_index)
        #     ).position(200, 0),
        #     speed=ego_vehicle_speed
        # )
        # ego_vehicle.id = 888
        # ego_vehicle.type = "AV"
        # road.vehicles.append(ego_vehicle)
        # self.controlled_vehicles.append(ego_vehicle)
        # self.vehicle = ego_vehicle

        # safety_distance = 50
        # aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        # normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        # cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        # sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # # slow vehicle
        # slow_vehicle_type = np.random.choice(sv_vehicle_types)
        # slow_lane = road.network.get_lane(('a', 'b', ego_lane_index))
        # slow_lane_position = slow_lane.position(200 + safety_distance + np.random.randint(-15, 15), 0)
        # slow_vehicle = slow_vehicle_type(
        #             road,
        #             slow_lane_position,
        #             speed=ego_vehicle_speed + np.random.randint(-6, -4),
        #         )
        # slow_vehicle.id = 0
        # slow_vehicle.type = 'car'
        # slow_vehicle.target_speed = random.randint(40, 60)/3.6
        # road.vehicles.append(slow_vehicle)

        # # left_slow_vehicle
        # slow_vehicle_type = np.random.choice(sv_vehicle_types)
        # slow_lane = road.network.get_lane(('a', 'b', 0))
        # slow_lane_position = slow_lane.position(200 + + safety_distance * 2 +np.random.randint(-10, 20), 0)
        # slow_vehicle = slow_vehicle_type(
        #     road,
        #     slow_lane_position,
        #     speed=ego_vehicle_speed + np.random.randint(-7, -5),
        # )
        # slow_vehicle.id = 1
        # slow_vehicle.type = 'car'
        # slow_vehicle.target_speed = random.randint(40, 50)/3.6
        # road.vehicles.append(slow_vehicle)

        # positions_x = [200 - 1 * safety_distance, 200 + 2 * safety_distance]
        # lane_indexes = list(range(self.config['lanes_count']))
        # possible_informations = []
        # for position in positions_x:
        #     for lane_index in lane_indexes:
        #         if (position, lane_index) != (200, ego_lane_index) and (position, lane_index) != (200 + safety_distance, ego_lane_index):
        #             speed = ego_vehicle_speed + np.random.randint(-7, 8)
        #             possible_informations.append((position, lane_index, speed))
        # num_env_vehicles = np.random.randint(0, 2)
        # information_to_generate = random.sample(possible_informations, num_env_vehicles)

        # id_sv_start = 2
        # for position, lane_index, speed in information_to_generate:
        #     vehicle_type = np.random.choice(sv_vehicle_types)
        #     lane = road.network.get_lane(('a', 'b', lane_index))
        #     lane_position = lane.position(position + np.random.randint(-15, 15), 0)
        #     sv = vehicle_type(
        #                 road,
        #                 lane_position,
        #                 speed=speed,
        #             )
        #     sv.id = id_sv_start
        #     sv.type = 'car'
        #     id_sv_start += 1
        #     road.vehicles.append(sv)

# 由于存在继承关系后置
class _1_1_TWOLaneChangeObstacleEnv(_2_2_TWOLaneChangeEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"lanes_count": 2,
                    "other_vehicles_types": {
                        "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                        "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                        "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                        "Obstacle": "highway_env.vehicle.behavior.ObstacleVehicle_df",
                        },
                })
        return cfg
    
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
                'a',
                'b',
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

        self.controlled_vehicles = []
        ego_lane_index = 1
        ego_vehicle_speed = np.random.randint(120, 121) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", ego_lane_index)
            ).position(200, 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed + np.random.randint(0, 1)/3.6
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 50
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # # left_slow_vehicle
        # slow_vehicle_type = np.random.choice(sv_vehicle_types)
        # slow_lane = road.network.get_lane(('a', 'b', 0))
        # slow_lane_position = slow_lane.position(200 + np.random.randint(-40, -30), 0)
        # slow_vehicle = slow_vehicle_type(
        #     road,
        #     slow_lane_position,
        #     speed=ego_vehicle_speed + np.random.randint(-10, 20)/3.6,
        # )
        # slow_vehicle.id = 0
        # slow_vehicle.type = "car"
        # slow_vehicle.target_speed = np.random.randint(80, 120)/3.6
        # road.vehicles.append(slow_vehicle)

        # positions_x = [200 - 1 * safety_distance, 200 + 2 * safety_distance]
        # lane_indexes = list(range(self.config['lanes_count']))
        # possible_informations = []
        # for position in positions_x:
        #     for lane_index in lane_indexes:
        #         if (position, lane_index) != (200, ego_lane_index) and (position, lane_index) != (
        #                 200 + safety_distance, ego_lane_index):
        #             speed = ego_vehicle_speed + np.random.randint(-7, 8)
        #             possible_informations.append((position, lane_index, speed))
        # num_env_vehicles = np.random.randint(0, 2)
        # information_to_generate = random.sample(possible_informations, num_env_vehicles)

        # id_sv_start = 1
        # for position, lane_index, speed in information_to_generate:
        #     vehicle_type = np.random.choice(sv_vehicle_types)
        #     lane = road.network.get_lane(('a', 'b', lane_index))
        #     lane_position = lane.position(position + np.random.randint(-15, 15), 0)
        #     sv = vehicle_type(
        #         road,
        #         lane_position,
        #         speed=speed,
        #     )
        #     sv.id = id_sv_start
        #     sv.type = "car"
        #     id_sv_start += 1
        #     road.vehicles.append(sv)

        # 添加静止车辆作为障碍物
        obs_type = utils.class_from_path(self.config["other_vehicles_types"]["Obstacle"])
        obs_pos = 200 + 50 + 2* ego_vehicle_speed + random.randint(-10, 10)
        obstacle_vehicle = obs_type(
            road,
            road.network.get_lane(
                ("a", "b", ego_lane_index)
            ).position(obs_pos, 0),
            speed=0,
        )
        obstacle_vehicle.id = 1000
        obstacle_vehicle.type = "obs"
        obstacle_vehicle.color = (255, 0, 0)
        road.vehicles.append(obstacle_vehicle)

        # 添加静止车辆作为障碍物
        obs_type = utils.class_from_path(self.config["other_vehicles_types"]["Obstacle"])
        obstacle_vehicle = obs_type(
            road,
            road.network.get_lane(
                ("a", "b", 0)
            ).position(obs_pos, 0),
            speed=0,
            heading = 90
        )
        obstacle_vehicle.id = 2000
        obstacle_vehicle.type = "obs"
        obstacle_vehicle.color = (255, 0, 0)
        road.vehicles.append(obstacle_vehicle)

class _3_1_EgoRampMergeEnvFIRST(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },

                "duration": 900,

                "two_lane_length":200,
                "merge_length":180,
                "single_lane_length":500,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 1800,
                "screen_height": 900,

            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)
    
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
        ends = [self.config['two_lane_length'], self.config['merge_length'], self.config['single_lane_length']]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type_main = [[c, s], [n, c]]
        for i in range(2):
            net.add_lane(
                "a", "b",
                StraightLane([0, y[i]], [ends[0], y[i]], line_types=line_type_main[i])
            )
        net.add_lane(
            "b","c",
            StraightLane([ends[0], y[0]], [ends[0]+ends[1], y[0]], line_types=line_type_main[0])
        )
        net.add_lane(
            "b","c",
            StraightLane([ends[0], y[1]], [ends[0]+ends[1], y[0]], line_types=line_type_main[1])
        )
        
        net.add_lane(
            "c", "d",
            StraightLane([ends[0]+ends[1], y[0]], [ends[0]+ends[1]+ends[2], y[0]], line_types=[c,c])
        )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        ego_position = 130 + np.random.randint(-20, 20)
        ego_vehicle_speed = np.random.randint(40, 80) / 3.6  # 40~121 km/h
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("a", "b", 1)).position(ego_position, 0),
            speed=ego_vehicle_speed
        )

        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed + np.random.randint(10, 20)/3.6
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 50
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # generate slow vehicle ahead
        slow_vehicle_type = np.random.choice(sv_vehicle_types)
        slow_lane = road.network.get_lane(('a', 'b', 1))
        slow_lane_position = slow_lane.position(ego_position + safety_distance + np.random.randint(-15, 15), 0)
        slow_vehicle_speed = ego_vehicle_speed + np.random.randint(-15, -10)/3.6
        slow_vehicle = slow_vehicle_type(
            road,
            slow_lane_position,
            speed=slow_vehicle_speed,
        )
        slow_vehicle.id = 0
        slow_vehicle.type = 'car'
        slow_vehicle.target_speed = slow_vehicle_speed
        road.vehicles.append(slow_vehicle)

        # generate vehicles on a, b, 0
        positions_x = [ego_position - safety_distance, ego_position, ego_position + safety_distance/2, slow_lane_position + safety_distance/2]
        # 插值生成更多位置
        # positions_x = np.linspace(positions_x[0], positions_x[-1], num=int((positions_x[-1]-positions_x[0])/20)+1)

        num_env_vehicles = np.random.randint(3, 4)
        used_positions_x = random.sample(list(positions_x), num_env_vehicles)

        id_sv_start = 1
        for positions_x in used_positions_x:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('a', 'b', 0))
            lane_position = lane.position(positions_x + np.random.randint(-15, 15), 0)
            speed = ego_vehicle_speed + np.random.randint(-15, 15)/3.6
            sv = vehicle_type(
                        road,
                        lane_position,
                        speed=speed,
                    )
            sv.id = id_sv_start
            sv.type = "car"
            sv.target_speed = speed
            id_sv_start += 1
            road.vehicles.append(sv)

class _3_2_EgoRampMergeEnvSecond(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },

                "duration": 900,

                "two_lane_length":200,
                "merge_length":380,
                "single_lane_length":500,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 1800,
                "screen_height": 900,

            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)
    
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
        ends = [self.config['two_lane_length'], self.config['merge_length'], self.config['single_lane_length']]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        part = 0.72
        net.add_lane(
            "a", "b",
            StraightLane([sum(ends[:1]), 0], [sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), 0], line_types=[c,c]),
        )
        net.add_lane(
            "b", "c",
            StraightLane([sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), 0], [sum(ends[:2]), 0], line_types=[c, s]),
        )
        net.add_lane(
            "c", "d",
            StraightLane([sum(ends[:2]), 0], [sum(ends), 0], line_types=[c,c]),
        )

        ramp_y = 6.5 + StraightLane.DEFAULT_WIDTH * 2
        # 起终点位置？
        
        ljk = StraightLane(
            [sum(ends[:1]), ramp_y],
            [sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), ramp_y - part * ramp_y],
            line_types=[c, c],
            forbidden=True,
        )
        lkc = StraightLane(
            [sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), ramp_y - part * ramp_y],
            [sum(ends[:2]), 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "c", lkc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        ego_position = 20 + np.random.randint(0, 10)
        ego_vehicle_speed = np.random.randint(40, 121) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("j", "k", 0)).position(ego_position, 0),
            speed=ego_vehicle_speed,
            heading = road.network.get_lane(("j", "k", 0)).heading_at(ego_position)
        )

        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed + np.random.randint(10, 20)/3.6
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 30 + ego_vehicle_speed + 1.5
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # generate slow vehicle ahead
        slow_vehicle_type = np.random.choice(sv_vehicle_types)
        slow_lane = road.network.get_lane(('j', 'k', 0))
        slow_lane_position = slow_lane.position(ego_position + safety_distance + np.random.randint(0, 15), 0)
        slow_lane_speed = ego_vehicle_speed + np.random.randint(-6, 6)/3.6
        slow_vehicle = slow_vehicle_type(
            road,
            slow_lane_position,
            speed = slow_lane_speed,
            # heading = -np.arctan(14.5/80)
            heading=road.network.get_lane(("j", "k", 0)).heading_at(slow_lane_position[0])
        )
        slow_vehicle.id = 0
        slow_vehicle.type = 'car'
        slow_vehicle.target_speed = slow_lane_speed
        road.vehicles.append(slow_vehicle)

        # generate vehicles on a, b, 0
        positions_x = [ego_position - safety_distance + np.random.randint(0, 10), ego_position]
        id_sv_start = 1
        vehicle_type = np.random.choice(sv_vehicle_types)
        lane = road.network.get_lane(('a', 'b', 0))
        lane_position = lane.position(positions_x[np.random.randint(0, 2)] + np.random.randint(-5, 5), 0)
        speed = ego_vehicle_speed + np.random.randint(-10, 10)/3.6
        sv = vehicle_type(
                    road,
                    lane_position,
                    speed=speed,
                )
        sv.id = id_sv_start
        sv.type = "car"
        sv.target_speed = speed
        road.vehicles.append(sv)

class _4_1_EgoMainMergeEnvFIRST(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },

                "duration": 900,

                "two_lane_length":200,
                "merge_length":80,
                "single_lane_length":500,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 1800,
                "screen_height": 900,

            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)
    
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
        ends = [self.config['two_lane_length'], self.config['merge_length'], self.config['single_lane_length']]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type_main = [[c, s], [n, c]]
        for i in range(2):
            net.add_lane(
                "a", "b",
                StraightLane([0, y[i]], [ends[0], y[i]], line_types=line_type_main[i])
            )
        net.add_lane(
            "b","c",
            StraightLane([ends[0], y[0]], [ends[0]+ends[1], y[0]], line_types=line_type_main[0])
        )
        net.add_lane(
            "b","c",
            StraightLane([ends[0], y[1]], [ends[0]+ends[1], y[0]], line_types=line_type_main[1])
        )
        net.add_lane(
            "c", "d",
            StraightLane([ends[0]+ends[1], y[0]], [ends[0]+ends[1]+ends[2], y[0]], line_types=[c,c])
        )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        ego_position = 130 + np.random.randint(-20, 20)
        ego_vehicle_speed = np.random.randint(40, 121) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("a", "b", 0)).position(ego_position, 0),
            speed=ego_vehicle_speed
        )

        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed + np.random.randint(10, 20)/3.6
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 50
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # generate slow vehicle ahead
        slow_vehicle_type = np.random.choice(sv_vehicle_types)
        slow_lane = road.network.get_lane(('a', 'b', 0))
        slow_lane_position = slow_lane.position(ego_position + safety_distance + np.random.randint(-15, 15), 0)
        slow_vehicle_speed = ego_vehicle_speed + np.random.randint(-15, -10)/3.6
        slow_vehicle = slow_vehicle_type(
            road,
            slow_lane_position,
            speed=slow_vehicle_speed,
        )
        slow_vehicle.id = 0
        slow_vehicle.type = 'car'
        slow_vehicle.target_speed = slow_vehicle_speed
        road.vehicles.append(slow_vehicle)

        # generate vehicles on a, b, 0
        positions_x = [ego_position - safety_distance, ego_position, ego_position + safety_distance/2, slow_lane_position + safety_distance/2]
        num_env_vehicles = np.random.randint(1, 3)
        used_positions_x = random.sample(positions_x, num_env_vehicles)

        id_sv_start = 1
        for positions_x in used_positions_x:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('a', 'b', 1))
            lane_position = lane.position(positions_x + np.random.randint(-15, 15), 0)
            speed = ego_vehicle_speed + np.random.randint(-15, 15)/3.6
            sv = vehicle_type(
                        road,
                        lane_position,
                        speed=speed,
                    )
            sv.id = id_sv_start
            sv.type = "car"
            sv.target_speed = speed
            id_sv_start += 1
            road.vehicles.append(sv)

class _4_2_EgoMainMergeEnvSecond(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },

                "duration": 900,

                "two_lane_length":200,
                "merge_length":180,
                "single_lane_length":500,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 1800,
                "screen_height": 900,

            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)
    
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
        ends = [self.config['two_lane_length'], self.config['merge_length'], self.config['single_lane_length']]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        part = 0.72
        net.add_lane(
            "a", "b",
            StraightLane([sum(ends[:1]), 0], [sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), 0], line_types=[c,c]),
        )
        net.add_lane(
            "b", "c",
            StraightLane([sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), 0], [sum(ends[:2]), 0], line_types=[c, s]),
        )
        net.add_lane(
            "c", "d",
            StraightLane([sum(ends[:2]), 0], [sum(ends), 0], line_types=[c,c]),
        )

        ramp_y = 6.5 + StraightLane.DEFAULT_WIDTH * 2

        ljk = StraightLane(
            [sum(ends[:1]), ramp_y],
            [sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), ramp_y - part * ramp_y],
            line_types=[c, c],
            forbidden=True,
        )
        lkc = StraightLane(
            [sum(ends[:1])+ part * (sum(ends[:2])-sum(ends[:1])), ramp_y - part * ramp_y],
            [sum(ends[:2]), 0],
            line_types=[n, c],
            forbidden=True,
        )

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "c", lkc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        ego_position = 20 + np.random.randint(0, 10)
        ego_vehicle_speed = np.random.randint(40, 121) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("a", "b", 0)).position(ego_position, 0),
            speed=ego_vehicle_speed,
            heading = road.network.get_lane(("a", "b", 0)).heading_at(ego_position)
        )

        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed + np.random.randint(10, 20)/3.6
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 30 + ego_vehicle_speed + 1.5
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # generate vehicles on a, b, 0
        positions_xs = [ego_position - safety_distance + np.random.randint(-10, 10), ego_position + np.random.randint(-10, 50)]
        id_sv_start = 1
        for positions_x in positions_xs:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('j', 'k', 0))
            lane_position = lane.position(positions_x + np.random.randint(-5, 5), 0)
            speed = ego_vehicle_speed + np.random.randint(-10, 10)/3.6
            sv = vehicle_type(
                        road,
                        lane_position,
                        speed=speed,
                    )
            sv.id = id_sv_start
            id_sv_start += 1
            sv.type = "car"
            sv.target_speed = speed
            road.vehicles.append(sv)

# 适配东风仿真环境的弯道场景
from highway_env.road.lane import CircularLane, LineType, StraightLane
import math

# 弯道场景1：曲线匝道冲出路外
class _DFsim_curveroad_1(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },


                "duration": 900,

                "scaling": 5.5,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 900,
                "screen_height": 900,

            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        return 0

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        net.add_lane(
                'a', 'b',
                StraightLane(
                    [0.0491, -5.6687], [-0.8743, 81.4898], line_types=[c,c]
                )
            )


        start_point, end_point = self.arc_endpoints(
            21.33,86.97,
            22.50,
            -3.15,
            -5.17
        )

        net.add_lane(
            'b', 'c',
            StraightLane(
                [-0.8743, 81.4898], start_point, line_types=[c, c]
            )
        )

        net.add_lane(
            "c",
            "d",
            CircularLane(
                [21.33,86.97],
                22.50,
                -3.15,
                -5.17,
                width=4,
                clockwise=False,
                line_types=[c,c],
            ),
        )

        start_point_2, end_point_2 = self.arc_endpoints(
            141.13076180780945, -476.0190571205332,
            587.3856748264097,
            1.7102453489051335,
            1.5423386247113628,
        )

        net.add_lane(
            'c', 'd',
            StraightLane(
                end_point, start_point_2, line_types=[c, c]
            )
        )

        net.add_lane(
            'd', 'e',
            CircularLane(
                [141.13076180780945, -476.0190571205332],
                587.3856748264097,
                1.7102453489051335,
                1.5423386247113628,
                width=4,
                clockwise=False,
                line_types=[c, c],
            )
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        ego_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(10.0, 1.0),
            speed=8.0,
            heading=ego_lane.heading_at(140.0),
        )
        ego_vehicle.id = 888
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def arc_endpoints(
            self,
            cx: float,
            cy: float,
            radius: float,
            start_rad: float,
            end_rad: float
    ):
        x_start = cx + radius * math.cos(start_rad)
        y_start = cy + radius * math.sin(start_rad)

        x_end = cx + radius * math.cos(end_rad)
        y_end = cy + radius * math.sin(end_rad)

        return [x_start, y_start], [x_end, y_end]

# 弯道场景2：大弧度弯道
class _DFsim_curveroad_2(_DFsim_curveroad_1): 
    # 大弧度弯道
    def _make_road(self) -> None:
        center = [0, 0]
        radius_1 = 400
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line_types = [[s,c], [s,n], [s,n], [c,n]]
        width = 4
        for i in range(4):
            net.add_lane(
                'a', 'b',
                StraightLane(
                    [- (radius_1 + i*width), 100],
                    [-(radius_1 + i*width), 0],
                    line_types=line_types[i],
                )
            )
            net.add_lane(
                'b', 'c',
                CircularLane(
                    center,
                    radius_1 + i*width,
                    np.deg2rad(180),
                    np.deg2rad(180+45),
                    clockwise=True,
                    line_types=line_types[i],
                ),
            )
            # net.add_lane(
            #     'c', 'd',
            #     StraightLane(
            #         [-200*np.sqrt(2)-i*width/np.sqrt(2), -200*np.sqrt(2)-i*width/np.sqrt(2)],
            #         [-200*np.sqrt(2)-i*width/np.sqrt(2)+50*np.sqrt(2),
            #          -200*np.sqrt(2)-i*width/np.sqrt(2)-50*np.sqrt(2)],
            #         line_types=line_types[i],
            #     )
            # )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        ego_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(10.0, 0.0),
            speed=8.0,
            heading=ego_lane.heading_at(140.0),
        )
        ego_vehicle.id = 888
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle


class _DFsimInsideLaneChangeEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },

                "lanes_count": 2,

                "duration": 900,

                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,

                "screen_width": 1800,
                "screen_height": 900,

            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)

    def _reset(self) -> None:
        # ego_lane_index = np.random.randint(0, self.config['lanes_count'] - 1)
        ego_lane_index = 1
        self._create_road(ego_lane_index)
        self._create_vehicles(ego_lane_index)

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 999)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reward(self, action: int) -> float:
        return 0.0

    def _create_road(self, ego_lane_index) -> None:
        net = RoadNetwork()
        length = 1e4
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        width = StraightLane.DEFAULT_WIDTH + 1
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
                'a',
                'b',
                StraightLane([0, i * width], [length, i * width],
                             width=width,
                             line_types=line_type),
            )
        offset_value = width / 2
        net.add_lane(
            'c', 'd',
            StraightLane([0, offset_value + ego_lane_index * width],
                         [length, offset_value + ego_lane_index * width],
                         width=width,
                         line_types=[n, n]),
        )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _create_vehicles(self, ego_lane_index) -> None:
        road = self.road

        self.controlled_vehicles = []

        ego_vehicle_speed = np.random.randint(40, 121) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", ego_lane_index)
            ).position(200 + random.randint(-40, 40), 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 150
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # slow vehicle
        slow_vehicle_type = np.random.choice(sv_vehicle_types)
        slow_lane = road.network.get_lane(('c', 'd', 0))
        slow_lane_position = slow_lane.position(200 + safety_distance + np.random.randint(-15, 15), -1)  # 0
        slow_vehicle = slow_vehicle_type(
            road,
            slow_lane_position,
            speed=0,  # ego_vehicle_speed + np.random.randint(-6, -4),
            enable_lane_change=False
        )
        slow_vehicle.id = 0
        slow_vehicle.type = 'car'
        slow_vehicle.target_speed = 0  # 40/3.6
        road.vehicles.append(slow_vehicle)

        positions_x = [200 - safety_distance, 200, 200 + safety_distance]
        lane_indexes = list(range(self.config['lanes_count']))
        possible_informations = []
        for position in positions_x:
            for lane_index in lane_indexes:
                if (position, lane_index) != (200, ego_lane_index) and (position, lane_index) != (
                200 + safety_distance, ego_lane_index):
                    speed = ego_vehicle_speed + np.random.randint(-7, 8)
                    possible_informations.append((position, lane_index, speed))
        num_env_vehicles = np.random.randint(1, 3)
        information_to_generate = random.sample(possible_informations, num_env_vehicles)

        id_sv_start = 1
        for position, lane_index, speed in information_to_generate:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('a', 'b', lane_index))
            lane_position = lane.position(position + np.random.randint(-15, 15), 0)
            sv = vehicle_type(
                road,
                lane_position,
                speed=speed,
            )
            sv.id = id_sv_start
            sv.type = 'car'
            id_sv_start += 1
            road.vehicles.append(sv)


# 用于东风场景随机初始化测试
class HighDensityEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },
                
                "lanes_count": 2,
                
                "duration": 900,
                
                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,
                
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        return cfg
    
    def reset(self, seed=None, **kwargs):
        # If a seed is provided either positionally or in kwargs, use it to seed Python and NumPy
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Defer to the base class reset to perform the usual environment reset and return its result
        return super().reset(**kwargs)
    
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
                'a',
                'b',
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
        self.controlled_vehicles = []
        ego_lane_index = np.random.randint(0, self.config['lanes_count'] - 1)
        ego_vehicle_speed = np.random.randint(40, 60) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", 1)
            ).position(850, 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = ego_vehicle_speed
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        safety_distance = 100
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # 间隔30m一辆车
        positions_x = list(range(0, 1000, 30))
        lane_indexes = list(range(self.config['lanes_count']))
        possible_informations = []
        for position in positions_x:
                speed = ego_vehicle_speed + 20/3.6
                possible_informations.append((position, speed))

        id_sv_start = 0
        for position, speed in possible_informations:
            vehicle_type = np.random.choice(sv_vehicle_types)
            lane = road.network.get_lane(('a', 'b', 0))
            lane_position = lane.position(position, 0)
            sv = vehicle_type(
                        road,
                        lane_position,
                        speed=speed,
                    )
            sv.id = id_sv_start
            sv.type = 'car'

            sv.target_speed = ego_vehicle_speed + 20/3.6
            sv.enable_lane_change = False
            id_sv_start += 1
            road.vehicles.append(sv)

class CutInEnv(_1_2_HwLaneChangeEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    # "type": "UnionAction",
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.CutInVehicle_aggressive",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },
                
                "lanes_count": 2,
                
                "duration": 900,
                
                "simulation_frequency": 10,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "show_trajectorie_points": True,
                
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        return cfg
    def _create_vehicles(self) -> None:
        road = self.road

        # --- 1. 生成主车 (AV) - 保持不变 ---
        self.controlled_vehicles = []
        ego_lane_index = 1
        ego_vehicle_speed = np.random.randint(30, 40) / 3.6
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(
                ("a", "b", ego_lane_index)
            ).position(200, 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 车辆类型池
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # --- 2. 生成连续密集的交通流 ---
        id_sv_start = 0
        
        # [控制密度的关键参数]
        start_x = 50           # 车流起始位置 (主车在 200)
        end_x = 600            # 车流结束位置 (根据你的道路长度调整)
        base_spacing = 7      # 车辆基础间距(米) - 越小越密集 (建议不要小于 20，否则容易初始碰撞)
        spawn_probability = 1.0 # 该位置生成车辆的概率 (0.0 到 1.0)，增加车流的随机性
        
        for lane_index in range(self.config['lanes_count']):
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
                    lane = road.network.get_lane(('a', 'b', lane_index))
                    
                    # 引入位置和速度的随机扰动，让车流看起来更真实
                    pos_noise = np.random.uniform(-5, 5)
                    speed_noise = np.random.uniform(-10, 5) # 背景车速度比主车慢一点
                    
                    sv = vehicle_type(
                        road,
                        lane.position(current_x, 0),
                        speed = 25 / 3.6
                    )
                    sv.id = id_sv_start
                    sv.type = 'car'
                    sv.target_speed = 25/3.6 # 设置随机目标速度
                    
                    road.vehicles.append(sv)
                    id_sv_start += 1
                
                # 推进到下一个空位
                current_x += base_spacing + np.random.uniform(0, 15)
        road = self.road

class _DFsimRoundaboutEnv(AbstractEnv):
    """
    标准四出口环岛博弈场景 (适配 MPC 轨迹跟踪)：
    - 包含 8 段组成的核心环岛（解决菱形渲染问题）。
    - 包含东、南、西、北 4 个方向的直线出入口。
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "Trajectory_for_mpc_curve",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "other_vehicles_types": {
                    "Aggressive": "highway_env.vehicle.behavior.AggressiveVehicle_df",
                    "Normal": "highway_env.vehicle.behavior.NormalVehicle_df",
                    "Cautious": "highway_env.vehicle.behavior.CautiousVehicle_df",
                },
                "duration": 900,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "show_trajectorie_points": True,
                "screen_width": 1000,
                "screen_height": 1000,
                "centering_position": [0.5, 0.5], # 居中显示
            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return super().reset(**kwargs)
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reward(self, action: int) -> float:
        return 0.0

    def _create_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        
        center = [0, 0]
        radius = 20    # 环岛半径
        alpha = 10     # 增大 alpha 以实现更明显的“分叉”效果
        N = 5          # 环岛碎段数
        access_total = 100 # 总引道长度
        split_dist = 25.0  # 分叉点距离圆心的距离 (在此处由平行转为分叉)
        lane_width = 4.0   # 标准车道宽度
        
        # 1. 构建核心环岛 (任意 N 段)
        quadrants = [
            (90 - alpha, alpha, "se", "ex"),         # 东南
            (-alpha, -90 + alpha, "ee", "nx"),       # 东北
            (-90 - alpha, -180 + alpha, "ne", "wx"), # 西北
            (180 - alpha, 90 + alpha, "we", "sx")    # 西南
        ]
        
        segments = []
        for start_ang, end_ang, start_node, end_node in quadrants:
            delta = (end_ang - start_ang) / N
            for i in range(N):
                n1 = f"{start_node}_{i}" if i > 0 else start_node
                n2 = f"{start_node}_{i+1}" if i < N - 1 else end_node
                a1, a2 = start_ang + i * delta, start_ang + (i + 1) * delta
                segments.append((n1, n2, a1, a2))
        
        segments.extend([
            ("ex", "ee", alpha, -alpha),
            ("nx", "ne", -90 + alpha, -90 - alpha),
            ("wx", "we", -180 + alpha, -180 - alpha),
            ("sx", "se", 90 + alpha, 90 - alpha)
        ])
        
        for n1, n2, a1, a2 in segments:
            net.add_lane(n1, n2, CircularLane(center, radius, np.deg2rad(a1), np.deg2rad(a2), 
                                            clockwise=False, line_types=[c, c]))

        # 2. 构建 4 个方向的出入口 (平行 -> 分叉逻辑)
        def get_pos(angle_deg):
            rad = np.deg2rad(angle_deg)
            return [radius * np.cos(rad), radius * np.sin(rad)]

        # --- 西侧 (West, 180°): 两根平行线在 x=-60 处分叉 ---
        p_we = get_pos(180 - alpha) # 入口节点 (~ -18, 8)
        p_wx = get_pos(180 + alpha) # 出口节点 (~ -18, -8)
        
        # 平行段 (间距 4m, y=2 和 y=-2)
        net.add_lane("w_para_in", "w_split", StraightLane([-150, 2], [-split_dist, 2], line_types=[c, c]))
        net.add_lane("w_merge", "w_para_out", StraightLane([-split_dist, -2], [-150, -2], line_types=[n, c]))
        
        # 分叉过渡段 (接入环岛节点)
        net.add_lane("w_split", "we", StraightLane([-split_dist, 2], p_we, line_types=[c, c]))
        net.add_lane("wx", "w_merge", StraightLane(p_wx, [-split_dist, -2], line_types=[c, c]))

        # --- 东侧 (East, 0°): 两根平行线在 x=60 处分叉 ---
        p_ee = get_pos(-alpha)
        p_ex = get_pos(alpha)
        
        # 平行段 (y=-2 入, y=2 出)
        net.add_lane("e_para_in", "e_split", StraightLane([150, -2], [split_dist, -2], line_types=[c, c]))
        net.add_lane("e_merge", "e_para_out", StraightLane([split_dist, 2], [150, 2], line_types=[n, c]))
        
        # 分叉过渡段
        net.add_lane("e_split", "ee", StraightLane([split_dist, -2], p_ee, line_types=[c, c]))
        net.add_lane("ex", "e_merge", StraightLane(p_ex, [split_dist, 2], line_types=[c, c]))

        # --- 南侧 (South, 90°): 两根平行线在 y=60 处分叉 ---
        p_se = get_pos(90 - alpha)
        p_sx = get_pos(90 + alpha)
        
        # 平行段 (x=2 入, x=-2 出)
        net.add_lane("s_para_in", "s_split", StraightLane([2, 150], [2, split_dist], line_types=[c, c]))
        net.add_lane("s_merge", "s_para_out", StraightLane([-2, split_dist], [-2, 150], line_types=[n, c]))
        
        # 分叉过渡段
        net.add_lane("s_split", "se", StraightLane([2, split_dist], p_se, line_types=[c, c]))
        net.add_lane("sx", "s_merge", StraightLane(p_sx, [-2, split_dist], line_types=[c, c]))

        # --- 北侧 (North, -90°): 两根平行线在 y=-60 处分叉 ---
        p_ne = get_pos(-90 - alpha)
        p_nx = get_pos(-90 + alpha)
        
        # 平行段 (x=-2 入, x=2 出)
        net.add_lane("n_para_in", "n_split", StraightLane([-2, -150], [-2, -split_dist], line_types=[c, c]))
        net.add_lane("n_merge", "n_para_out", StraightLane([2, -split_dist], [2, -150], line_types=[n, c]))
        
        # 分叉过渡段
        net.add_lane("n_split", "ne", StraightLane([-2, -split_dist], p_ne, line_types=[c, c]))
        net.add_lane("nx", "n_merge", StraightLane(p_nx, [2, -split_dist], line_types=[c, c]))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        road = self.road
        self.controlled_vehicles = []

        # --- 自车 (从南入口进，去北出口) ---
        ego_vehicle_speed = 25 / 3.6 
        ego_lane = road.network.get_lane(("w_para_in", "w_split", 0))
        ego_vehicle = self.action_type.vehicle_class(
            road,
            ego_lane.position(50, 0), # 距离切入点 50m
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = 30 / 3.6
        
        try:
            ego_vehicle.plan_route_to("e_out") # 规划至北侧出口
        except AttributeError:
            pass
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # --- 添加一辆背景车辆做交互 (西入口进，东出口出) ---
        sv_lane = road.network.get_lane(("n_para_in", "n_split", 0))
        sv_1 = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])(
            road,
            sv_lane.position(35, 0),
            speed=30 / 3.6
        )
        sv_1.id = 1
        sv_1.type = 'car'
        sv_1.target_speed = 30 / 3.6
        try:
            sv_1.plan_route_to("s_para_out")
        except AttributeError:
            pass
        road.vehicles.append(sv_1)
        bv_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])

class HwCruiseEnv(_1_2_HwLaneChangeEnv):
    """
    基于多智能体博弈与LLM推理的高速巡航场景
    特点：动态车流、多车道分布、包含不同隐藏类型(Aggressive, Normal, Cautious)的NPC车辆
    """
    
    @classmethod
    def default_config(cls) -> dict:
        # 继承父类配置
        cfg = super().default_config()
        # 针对巡航场景进行微调
        cfg.update(
            {
                "duration": 1200,          # 巡航场景通常需要更长的仿真时间来观察长期博弈
                "vehicles_count": 15,      # 设定场景中NPC车辆的目标生成数量
                "ego_spacing": 2,          # 自车与初始前车的距离比例
            }
        )
        return cfg
    
    def _is_terminated(self) -> bool:
        # 将终点线延长到 10000 米，或者只要不撞就一直跑，直到 _is_truncated 触发时间上限
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 9999)
    
    def _create_vehicles(self) -> None:
        """
        重写车辆生成逻辑：
        不再生成静止的障碍车，而是生成具有不同速度、不同风格的动态车流。
        """
        road = self.road
        self.controlled_vehicles = []
        
        # ---------------- 1. 初始化自车 (Ego Vehicle) ----------------
        # 倾向于生成在中间或右侧车道，以便有向左超车的空间
        ego_lane_index = np.random.randint(0, self.config['lanes_count'])
        # 设定自车期望速度：22~28 m/s (约 80~100 km/h)
        ego_vehicle_speed = 22
        
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("a", "b", ego_lane_index)).position(20, 0),
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = 28 # 允许一定的提速空间
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # ---------------- 2. 获取隐藏类型配置 ----------------
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        cautious_type = utils.class_from_path(self.config["other_vehicles_types"]["Cautious"])
        # 在巡航场景中，可以根据需要调整三种风格车辆的生成比例 (目前是等概率)
        sv_vehicle_types = [aggressive_type, normal_type, cautious_type]

        # ---------------- 3. 生成动态车流 (Traffic Stream) ----------------
        vehicles_count = self.config.get("vehicles_count", 15)
        lanes_count = self.config["lanes_count"]
        id_sv_start = 1

        for _ in range(vehicles_count):
            # 随机分配车道
            lane_index = self.np_random.integers(0, lanes_count)
            # 在自车前方 30m 到 400m 范围内随机分布
            position_x = self.np_random.uniform(30, 400)
            
            # 随机分配初始速度：有的车比自车慢，有的比自车快
            # 速度范围大约在 18~32 m/s (约 65~115 km/h)
            speed = ego_vehicle_speed + self.np_random.uniform(-3, 3)
            
            # 随机选择该车辆的底层行为逻辑(激进、正常、保守)
            vehicle_type = self.np_random.choice(sv_vehicle_types)
            
            lane = road.network.get_lane(('a', 'b', lane_index))
            # 增加少量的横向随机偏移，模拟真实情况
            lane_position = lane.position(position_x + self.np_random.uniform(-10, 10), 0)

            # ---------------- 4. 碰撞检测与车辆实例化 ----------------
            # 简单检查新生成的车辆是否与已有车辆发生重叠，保证初始环境的安全
            is_clear = True
            for v in road.vehicles:
                if np.linalg.norm(np.array(lane_position) - np.array(v.position)) < 20:
                    is_clear = False
                    break
            
            if is_clear:
                sv = vehicle_type(
                    road,
                    lane_position,
                    speed=speed,
                )
                sv.id = id_sv_start
                sv.type = 'car'
                
                # 设置其目标巡航速度，通常与初始速度相近，但也允许微小浮动
                # Keep the class-specific desired speed (90/110/120 km/h).
                # Overwriting it with the common initial speed made Cautious and
                # Normal trajectories statistically indistinguishable to the LLM.
                sv.target_speed += self.np_random.uniform(-2, 2)
                
                id_sv_start += 1
                road.vehicles.append(sv)
    
    def _reward(self, action) -> float: # 删掉类型注解
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
            pass # 如果动作不是包含两个元素的列表，跳过惩罚，避免报错崩溃

        reward -= 0.05 * abs(self.vehicle.heading)

        return float(reward)

class AdversarialMergeEnv(_3_1_EgoRampMergeEnvFIRST):
    """
    对抗性汇入场景 (Adversarial Merge)
    特点：自车前方无车，主路上有一辆激进车。激进车会实时监控自车状态，
    故意同步加速以封锁自车的汇入空间，逼迫自车减速或发生博弈。
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        # 针对该场景可适当调整参数
        cfg.update({
            "duration": 500, # 汇入对抗通常在较短时间内分出胜负
        })
        return cfg

    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "duration": 500, # 汇入对抗通常在较短时间内分出胜负
            "lanes_count": 4, # 更新为4车道
        })
        return cfg

    def _create_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        width = StraightLane.DEFAULT_WIDTH  # 默认 4.0

        # --- 第一段 (0 ~ 200m)：自由汇入区 ---
        # 4条主车道 (0, 1, 2, 3)
        for i in range(self.config["lanes_count"]):
            # 第4车道(i=3)的右侧是虚线(s)，允许从匝道自由跨越
            line_types = [c, s] if i == 0 else [n, s] 
            net.add_lane("a", "b", StraightLane([0, i * width], [300, i * width], line_types=line_types))
        
        # 加速匝道 (作为第5条车道，i=4)
        # 左侧无标线(依赖主路划的虚线)，右侧为道路物理边缘(实线c)
        net.add_lane("a", "b", StraightLane([0, 4 * width], [300, 4 * width], line_types=[n, c]))

        # --- 第二段 (200 ~ 250m)：匝道收窄关闭区 ---
        for i in range(self.config["lanes_count"]):
            # 第4车道(i=3)右侧恢复为实线(c)
            line_types = [c, s] if i == 0 else ([n, c] if i == 3 else [n, s])
            net.add_lane("b", "c", StraightLane([300, i * width], [380, i * width], line_types=line_types))
        
        # 匝道在此段收窄并标记为不可行驶 (forbidden=True)
        net.add_lane("b", "c", StraightLane([300, 4 * width], [380, 3 * width], line_types=[n, c], forbidden=True))

        # --- 第三段 (250 ~ 1000m)：常规 4 车道直行 ---
        for i in range(self.config["lanes_count"]):
            line_types = [c, s] if i == 0 else ([n, c] if i == 3 else [n, s])
            net.add_lane("c", "d", StraightLane([380, i * width], [1000, i * width], line_types=line_types))

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", True)
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
            speed=ego_vehicle_speed
        )
        ego_vehicle.id = 888
        ego_vehicle.type = "AV"
        ego_vehicle.target_speed = 25 # 自车期望加速汇入
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 准备车型池
        aggressive_type = utils.class_from_path(self.config["other_vehicles_types"]["Aggressive"])
        normal_type = utils.class_from_path(self.config["other_vehicles_types"]["Normal"])
        id_start = 2
        bg_speed = 60 / 3.6 

        # --- 2. 内侧三车道 (lane 0, 1, 2) 添加 60km/h 直行且不变道的车辆 ---
        for lane_idx in [0,1]:
            # 每条车道错开布置 1~2 辆
            pos_1 = ego_position + self.np_random.uniform(10, 30)
            bg_veh_1 = normal_type(road, road.network.get_lane(("a", "b", lane_idx)).position(pos_1, 0), speed=bg_speed)
            bg_veh_1.id = id_start
            bg_veh_1.type = 'car'
            bg_veh_1.target_speed = bg_speed
            bg_veh_1.enable_lane_change = False
            road.vehicles.append(bg_veh_1)
            id_start += 1
            
            pos_2 = pos_1 + self.np_random.uniform(40, 60)
            bg_veh_2 = normal_type(road, road.network.get_lane(("a", "b", lane_idx)).position(pos_2, 0), speed=bg_speed)
            bg_veh_2.id = id_start
            bg_veh_2.type = 'car'
            bg_veh_2.target_speed = bg_speed
            bg_veh_2.enable_lane_change = False
            road.vehicles.append(bg_veh_2)
            id_start += 1

        # --- 3. 放置激进车 (主路最外侧车道 lane 3) ---
        blocker_pos = ego_position - 5 # 放在自车左侧稍靠后的盲区位置
        blocker_speed = ego_vehicle_speed 
        
        self.blocker_vehicle = aggressive_type(
            road,
            road.network.get_lane(("a", "b", 3)).position(blocker_pos, 0),
            speed=blocker_speed
        )
        self.blocker_vehicle.id = 1
        self.blocker_vehicle.type = 'car'
        self.blocker_vehicle.target_speed = blocker_speed
        self.blocker_vehicle.enable_lane_change = False # 死守主路外侧
        self.blocker_vehicle.color = (200, 0, 0) 
        road.vehicles.append(self.blocker_vehicle)
        
        # --- 4. 激进车后方增加一辆正常跟车，速度 60km/h ---
        follower_pos = blocker_pos - 35
        follower_veh = normal_type(
            road,
            road.network.get_lane(("a", "b", 3)).position(follower_pos, 0),
            speed=bg_speed
        )
        follower_veh.id = id_start
        follower_veh.type = 'car'
        follower_veh.target_speed = bg_speed
        follower_veh.enable_lane_change = False 
        road.vehicles.append(follower_veh)

    def step(self, action):
        """
        精准拦截逻辑：保持原代码逻辑不变
        """
        if hasattr(self, 'blocker_vehicle') and self.blocker_vehicle in self.road.vehicles:
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
                blocker.target_speed = np.clip(desired_speed, 10.0, 70.0/3.6)

        obs, original_reward, terminated, truncated, info = super().step(action)
        return obs, original_reward, terminated, truncated, info

class DF_HighwayEnv(HighwayEnv):
    """
    与原生 HighwayEnv 完全一致的场景，但适配了东风项目的 MPC 轨迹规划控制
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "Trajectory_for_mpc_curve", # 使用 MPC 动作空间
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "lanes_count": 4, # 原生 HighwayEnv 默认为 4 车道
                "duration": 900,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "show_trajectories": True, # 注意：你之前的代码有些地方写成了 show_trajectorie_points，这里按照 HighwayEnv 标准来
                "screen_width": 1800,
                "screen_height": 900,
            }
        )
        return cfg

    def reset(self, seed=None, **kwargs):
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return super().reset(seed=seed, **kwargs)

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
    
    def _create_road(self) -> None:
        """
        重写原生的路网生成逻辑，将道路长度拉长，防止车跑出地图
        """
        from highway_env.road.road import Road, RoadNetwork
        
        # 强制将道路物理长度设为 20000 米 (20公里)
        road_length = 20000  
        
        # 生成足够长的直道网络
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"], 
            length=road_length, 
            speed_limit=30
        )
        
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:
        # 1. 先调用原生 HighwayEnv 的车辆生成逻辑（它会生成主车和一堆 IDM 背景车）
        super()._create_vehicles()
        
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

class DF_MergeEnv(MergeEnv):
    """
    与原生 MergeEnv 完全一致的汇入场景，适配东风 MPC 框架
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "Trajectory_for_mpc_curve", # 适配 MPC
                },
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

    def reset(self, seed=None, **kwargs):
        if seed is None:
            seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return super().reset(seed=seed, **kwargs)

    def _make_vehicles(self) -> None:
        # 调用父类生成逻辑（主车默认在匝道，背景车在主路）
        super()._make_vehicles()
        
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

    def _reward(self, action) -> float:
        # 使用你项目中统一的连续动作奖励函数
        if action is None: return 0.0
        reward = 0.0
        if self.vehicle.crashed: return -10.0  
        
        target_speed = getattr(self.vehicle, "target_speed", 25.0) 
        speed_reward = utils.lmap(self.vehicle.speed, [10.0, target_speed], [0.0, 1.0])
        reward += 1.0 * np.clip(speed_reward, 0.0, 1.0)

        try:
            reward -= 0.1 * (float(action[0]) ** 2)
            reward -= 0.5 * (float(action[1]) ** 2)
        except: pass 
        
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
            "speed_reward": float(self.vehicle.speed / getattr(self.vehicle, "target_speed", 25.0))
        }
