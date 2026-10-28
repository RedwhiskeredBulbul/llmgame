import array
import copy, time
from tkinter import N
from typing import List, Optional, Tuple, Union
import numpy as np
from numba.core.callconv import error_models

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle
from Planner.mpc_planner import MPCPlanner


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [
                (path[i], path[i + 1], None) for i in range(len(path) - 1)
            ]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(
                self.target_lane_index,
                route=self.route,
                position=self.position,
                np_random=self.road.np_random,
            )

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [
            self.route[0 : index + 1]
            + [(self.route[index][1], destination, self.route[index][2])]
            for destination in next_destinations_from
        ]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.integers(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [
            self.road.network.position_heading_along_route(
                route, coordinates[0] + self.speed * t, 0, self.lane_index
            )
            for t in times
        ]
        return tuple(zip(*pos_heads))


class MDPVehicle(ControlledVehicle):
    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
    ) -> None:
        """
        Initializes an MDPVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        """
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(
            np.clip(self.speed_index, 0, self.target_speeds.size - 1)
        )
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


class MPCPlanningVehicle(Vehicle):
    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    # TODO
    MAX_ACCELERATION = 3  # [m/s2]
    MAX_DECELERATION = -5  # [m/s2]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0, target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        self.mpc_planner = MPCPlanner()  # 创建MPC规划器实例
        self.pred_trajs = None
        self.corridor_space = None

    def act(self, action: Union[dict, str] = None) -> None:
        '通过外接规划器进行轨迹规划，计算期望加速度和前轮转角'
        n_int = 6
        ego_state = np.array([self.position[0], self.position[1], self.heading, self.speed, self.action["acceleration"], self.action["steering"]])
        other_states = []
        for vehicle in self.road.vehicles:
            if vehicle != self:
                other_states.append([vehicle.position[0], vehicle.position[1], vehicle.heading, vehicle.speed, vehicle.action["acceleration"], 0])
        other_states = np.array(other_states).reshape(-1, n_int)

        lane_center_y = [4, 8]  # 车道中心线 [current_lane, target_lane]
        lane_wid = 4

        # target_speed = self.road.vehicles[1].speed # 目标车速是换道前车车速
        target_speed = 120 / 3.6  # 目标车速120km/h

        t1 = time.time()
        acc_list, delta_list = self.mpc_planner.act(ego_state, other_states, lane_center_y, lane_wid, target_speed)
        t2 = time.time()
        print("MPC planning time：", (t2 - t1)*1000, "ms")
        
        self.action["acceleration"] = float(acc_list[0])
        self.action["steering"] = float(delta_list[0])

        self.pred_trajs = self.mpc_planner.predict_trajectory(ego_state, acc_list, delta_list, dt=self.mpc_planner.dt)
        # self.pred_trajs = self.predict_trajectory(acc_list, delta_list, dt=self.mpc_planner.dt)

        self.corridor_space = self.mpc_planner.predict_corridor()
        return

    def step(self, dt: float) -> None:
        '更新车辆状态'
        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))  
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        self.on_state_update()

    # def predict_trajectory(self,
    #                        pred_acc: np.ndarray,
    #                        pred_delta: np.ndarray,
    #                        dt: float,
    #                        ) -> np.array:
    #     '将MPC的输出序列转化为未来n步的轨迹'
    #     states = np.zeros((pred_acc.shape[0], 4))  # x, y, heading, speed
    #     position = copy.deepcopy(self.position)
    #     heading = copy.deepcopy(self.heading)
    #     speed = copy.deepcopy(self.speed)
    #     L = self.LENGTH
    #     for i in range(pred_acc.shape[0]):
    #         # 先根据kinematic bicycle model计算下一个状态
    #         beta = np.arctan(1 / 2 * np.tan(pred_delta[i]))
    #         v = speed * np.array([np.cos(heading + beta), np.sin(heading + beta)])
    #         position += v * dt
    #         heading += speed * np.sin(beta) / (L / 2) * dt
    #         speed += pred_acc[i] * dt

    #         states[i, :] = np.array([position[0], position[1], heading, speed])

    #     return states


class MPCPlanningVehicle_GameOnly(MPCPlanningVehicle):
    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    # TODO: 
    MAX_ACCELERATION = 3  # [m/s2]
    MAX_DECELERATION = -5  # [m/s2]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0, target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None):
        super().__init__(road, position, heading, speed)
        self.kp_heading = 0.5
        self.kp_speed = 1.0
        self.last_action = np.zeros(2)
        self.dt = 0.1
        self.id = 'AV'
        self.mpc_planner = MPCPlanner()  # 创建MPC规划器实例

    def act(self, info = None) -> None:
        '通过外接规划器进行轨迹规划，计算期望加速度和前轮转角'
        if info is None:
            return  # no valid trajectory

        # target = trajectory[0]
        # PREVIEW_TIME = 0.5        # 预瞄时间 (s)

        # # === 1. 获取参考点 ===
        # # 使用预瞄点代替第一个轨迹点，减少频繁方向切换
        # preview_idx = min(int(PREVIEW_TIME / self.dt), len(trajectory) - 1)
        # target = trajectory[preview_idx]
        # target_x, target_y, _, target_v = target[:4]

        # # === 1. Heading control ===
        # dx = target_x - self.position[0]
        # dy = target_y - self.position[1]
        # desired_yaw = np.arctan2(dy, dx)
        # heading_error = np.arctan2(
        #     np.sin(desired_yaw - self.heading),
        #     np.cos(desired_yaw - self.heading)
        # )
        # steer = self.kp_heading * heading_error
        # steer = np.clip(steer, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # # === 2. Speed control ===
        # speed_error = target_v - self.speed
        # accel = self.kp_speed * speed_error
        # accel = np.clip(accel, self.MAX_DECELERATION, self.MAX_ACCELERATION)

        'Use MPC planner to get action'
        n_int = 6
        ego_state = np.array([self.position[0], self.position[1], self.heading, self.speed, self.action["acceleration"], self.action["steering"]])
        other_states = []
        for vehicle in self.road.vehicles:
            if vehicle != self:
                other_states.append([vehicle.position[0], vehicle.position[1], vehicle.heading, vehicle.speed, vehicle.action["acceleration"], 0])
        other_states = np.array(other_states).reshape(-1, n_int)

        'obstacle prediction'
        other = np.asarray(other_states) if other_states is not None else None
        if other.ndim == 1 and other.size > 0:
            other = other.reshape(1, -1)
        n_other = other.shape[0] if other.size>0 else 0
        # precompute other vehicles predicted positions at t=1..N (relative to current time)
        other_preds = np.zeros((self.mpc_planner.N+1, n_other, 2), dtype=float)  # include t=0 if needed [X, Y]
        for j in range(n_other):
            ox, oy, otheta, ov = float(other[j,0]), float(other[j,1]), float(other[j,2]), float(other[j,3])
            for k in range(self.mpc_planner.N+1):
                tpred = k * self.mpc_planner.dt
                other_preds[k, j, 0] = ox + ov * np.cos(otheta) * tpred
                other_preds[k, j, 1] = oy + ov * np.sin(otheta) * tpred

        lane_wid = 4
        target_speed = info['ego_traj'][1][3]  # 目标车速为轨迹第二个点的速度
        lane_center_y = [ego_state[1], info['ref_line'][1].y]  # 车道中心线 y 坐标
        'Planner act'
        t1 = time.time()
        acc_list, delta_list = self.mpc_planner.act(ego_state, other_states, other_preds, lane_center_y, lane_wid, target_speed)
        t2 = time.time()
        print("MPC planning time：", (t2 - t1)*1000, "ms")

        self.action["acceleration"] = acc_list[0]
        self.action["steering"] = delta_list[0]
        print("steer:", self.action["steering"], "accel:", self.action["acceleration"])
        # TODO: predict trajectory
        self.pred_trajs = info['ego_traj'][:, :4]
        return
    
    # TODO: 更新车辆的规划轨迹，用于可视化
    def step(self, dt: float) -> None:
        '更新车辆状态'
        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))  
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        self.on_state_update()


class ExecuteVehicle(Vehicle):
    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    # TODO
    MAX_ACCELERATION = 3  # [m/s2]
    MAX_DECELERATION = -5  # [m/s2]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0, target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None):
        super().__init__(road, position, heading, speed)
        self.pred_trajs = None
        self.game_traj = None
        self.opt_trajs = None

    def act(self, action: list = None) -> None:
        'Execute given action'
        if action is None:
            return
        self.action["acceleration"] = float(action[0])
        self.action["steering"] = float(action[1])
        return

    def step(self, dt: float) -> None:
        'Update vehicle state'
        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))  
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        self.on_state_update()


class TrajectoryPlanVehicleForMPCCurve(ExecuteVehicle):

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0, target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None):
        super().__init__(road, position, heading, speed)
        self.pred_trajs = None
        self.game_traj = None
        self.opt_trajs = None
        self.mpc_plan_trajectory = None

    def act(self, action = None) -> None:
        '''  output trajectory:  [x y v heading a s]  '''

        if action is None:
            return

        execute_action = np.array(action['action'])
        trajectory = action['trajectory']

        if execute_action.shape != ():
            self.action["acceleration"] = execute_action[0, 0]
            self.action["steering"] = execute_action[0, 1]
        else:
            self.action["acceleration"] = trajectory[1, 4]
            self.action["steering"] = 0

        self.mpc_plan_trajectory = trajectory
        return

    def step(self, dt: float) -> None:
        'Update vehicle state'
        if self.mpc_plan_trajectory is None:
            return

        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None

        # [x y v heading a s]
        self.heading = self.mpc_plan_trajectory[1, 3]
        self.speed = self.mpc_plan_trajectory[1, 2]
        self.position[0] = self.mpc_plan_trajectory[1, 0]
        self.position[1] = self.mpc_plan_trajectory[1, 1]

        self.on_state_update()


