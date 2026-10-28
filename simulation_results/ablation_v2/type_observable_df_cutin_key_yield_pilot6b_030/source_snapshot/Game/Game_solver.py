# 创建博弈求解的类
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math
import sys
sys.path.append(r'.\Game')
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
from scipy.interpolate import interp1d
from numba import njit, prange
import casadi as ca
import copy
from scipy.optimize import minimize_scalar

from utils.pathcollision_detect import path_collision_detect, rect_intersects_any_ellipse, get_rectangle_vertices
from Game.style_params import STYLE_PARAMS
try:
    import rospy
except:
    pass

class GameSolver:
    def __init__(self, AV_style="Normal", roslog=True, decision_config=None):
        self.decision_config = dict(decision_config or {})
        # Physical safety buffer added to the predicted-obstacle ellipses in
        # RULE4.  Zero keeps the legacy geometry exactly; a positive value
        # makes the hard collision check require a real gap instead of the
        # ellipse's permissive 0.1-0.3 m squeeze.  Applies to every vehicle
        # type and every lane-change candidate equally.
        self.collision_margin_m = 0.
        # When enabled, the Force intent's predicted trajectory ignores the
        # IDM follow cap while the AV is cutting in, so the aggressive branch
        # is predicted to keep going rather than brake for the ego.  The ego
        # still decides through the same game/reward machinery; no
        # type-specific ego rule is introduced.
        self.force_ignore_idm_on_av_lc = False
        self.solve_diagnostics = []
        self.roslog = roslog
        if self.roslog:
            import rospy
        
        # 定义类型和ID的映射
        self.type_mapping = {
            'AV': 0,  # 自动驾驶车辆
            'car': 1,   # 人类驾驶车辆
            'obs': 2,  # 障碍物
            'water_barrier':2, # 水马
            'cone':2, # 锥桶
            "noneVeh":2, # 非机动车
            'truck':2, # 卡车
            "other":2, # 其他类型视为障碍物
            'otherVeh':2, # 其他车辆
            'specialVeh':2, # 特殊车辆
            'car_pred': 1,   # 人类驾驶车辆
            'truck_pred':1, # 卡车
            'specialVeh_pred':1, # 特殊车辆
            'noneVeh_pred':1, # 非机动车
            'otherVeh_pred':1, # 其他车辆
            'water_barrier_pred':1, # 水马
            'cone_pred':1, # 锥桶
        }
        self.id_mapping = {
            'AV': 888,  # 自动驾驶车辆
        }
        self.limits = {
            self.type_mapping['AV']:{
                'MAX_SPEED': 60/3.6, # m/s
                'MAX_ACCELERATION': 1,  # [m/s2]
                'MAX_DECELERATION': -2,  # [m/s2]'
            },
            
            ## 以下是对周围交通参与者的预测参数
            self.type_mapping['car']:{
                'MAX_SPEED': 120/3.6, # m/s
                'MAX_ACCELERATION': 1,  # [m/s2]
                'MAX_DECELERATION': -2,  # [m/s2]'
            },
            # 'NoneVeh':None, # TODO：非机动车的参数待补充
            self.type_mapping['obs']:{
                'MAX_SPEED': 0.0, # m/s
                'MAX_ACCELERATION': 0.0,  # [m/s2]
                'MAX_DECELERATION': 0.0,  # [m/s2]'
            },
        }
        # 
        self.max_long_jerk = 4.0
        self.ego_length = 5.0 
        self.ego_width = 2.0
        self.if_use_aeb = False # 暂时不启用AEB
        self.aeb_max_decel = -5.0  # 紧急制动最大减速度
        
        self.max_iters = 1 # 最大博弈迭代次数
        self.max_solve_time = 100/1000 # 最大求解时间(s)
        self.dt = 0.1 # 实际执行步长
        self.t_list = np.array([0.1, 0.5, 8.0]) # TODO:需要设计满足8s的需求，为简化计算的实际规划步长
        self.t_output_av = np.arange(0.0792, 8.0792, 0.0792) # 输出的规划时间
        self.t_output_other = np.arange(0.5, 8.5, 0.5) # 输出的对对象预测的时间
        
        # 需要存储的全局变量：是否允许换道 TODO：需要确保在实车中如何设计实现
        self.Changing_Permitted = True # False: 不允许换道，True: 允许换道，需要根据车道线、交叉口环岛场景设置规则
        self.changing = False # 换道标志，True: 正在换道，False: 非换道
        self.stopchanging = False # 换道中止标志，True: 正在换道中止，False: 非换道中止
        self.current_path = None # np.array([[i, 0, 0, 0] for i in range(1000)])# None # 当前路径：正在换道时的换道路径、非换道时的当前路径
        self.current_target_idx = None # 换道时目标索引
        self.current_target_idx_point = None
        self.changing_finish_threshold = 0.5
        self.smooth_long_acc = False # 是否对纵向加速度进行平滑
        self.smooth_a = True # 是否对加速度进行平滑（固定jerk值）
        self.smooth_cal_a = True # 是否对计算的加速度进行平滑
        self.memory_cal_a = 0.0 # 上一时刻计算的加速度，用于平滑
        
        self.L_min = 5.0  # 换道最短距离
        self.L_max = 200.0  # 换道最长距离
        self.change_t = 3.0
        self.L_delta = 3.3 # 换道横向距离
        self.path_foresee_num = 1500  # 换道路径预见点数
        self.s0 = 2.5 # 静止跟车间距
        self.idm_T = 1.0 # IDM安全时间头距
        self.dis2end_threshold = 20.0 # 车辆需要在终点前10m停车
        self.safechange_threshold = 10.0 # 与换道车道后车的静止安全距离，实际计算时加入了thw*delta_v的考虑
        self.confcal_threshold = 0.8 # 碰撞检测阈值，与绕行是否停车直接相关
        self.max_lateral_dist = 0.8 # 最大横向距离，用于IDM判断前车
        self.force_flag = False # 强制换道标志位，用于下匝道等
        self.change_memory = [False, False] # 换道记忆模块
        self.stopchange_memory_1 = [False, False] # 换道中止记忆模块1：危险中止
        self.stopchange_memory_2 = [False, False] # 换道中止记忆模块2：不优中止
        self.waiting = False # 等待换道标志位

        # 信念表
        self.external_probs = None

        # 博弈相关：是否将AV的动作作为其他车的预测输入
        self.AV_LEADER = True 
        self.SHARED_ACC = True # 每个规划时刻是否共享加速度变量
        self.LONG_REWARD = False # 是否使用长期奖励
        self.n_time = len(self.t_list) if self.LONG_REWARD else 1
        self.other_discrete_precision = 1.0 # 其他车辆的离散求解精度
        
        # 收益相关
        self.TTC_max = 5.0 # 调大可能会提前换道
        self.TTCP_max = 5.0
        self.decay_rate = 0.8 # 衰减率
        self.eps = 1e-6 # 防止除零错误
        self.converged_eps = 0.1 # 收敛阈值
        self.AV_reward_diff2max_threshold = 1e-4  # 当前车道可以到最大收益的差值阈值
        self.path_diff_reward_threshold = 0.1  # 1e-4 # 路径差异奖励阈值
        self.path_diff_action_threshold = 0.4  # 路径差异动作阈值
        
        # AV的收益权重
        # self.av_weights = (0.1, 1.0, 0.000) 
        self.av_weights = (0.1, 5.0, 0.000) 
        
        # other的收益权重
        self.other_init_weight = (1.0, 1.0, 0.000) # self.av_weights
        self.other_init_belief = (0.3, 0.4, 0.3)
        self.weight_dict = {self.id_mapping['AV']: self.av_weights}
        self.weight_keys = np.array(list(self.weight_dict.keys()), dtype=float)
        self.weight_values = np.array(list(self.weight_dict.values()), dtype=float)
        self.style_reward_dict = {} # id: (safety_reward, efficiency_reward, comfort_reward)
        self.old_style_reward_dict = {} # 用于储存上一时刻的奖励
        self.other_style_dict = {} # 他车风格辨识结果存储
        self.style_map = {"Aggressive":1, "Conservative":-1, "Normal":0}
        # ==========================================================
        # 【新增】第一阶段：贝叶斯博弈意图空间与参数配置
        # ==========================================================
        # 1. 定义离散的意图类型 (K=3)
        self.intent_types = ['Yield', 'Normal', 'Force']
        self.K_intents = len(self.intent_types)
        
        # 2. 定义每种意图对应的物理边界与收益偏好
        # weights 顺序为: [安全权重, 效率权重, 舒适权重]
        self.intent_profiles = {
            'Yield': {  # 保守型：纵向跟车远，横向极易让道
                'weights': [100.0, 0.5, 0.0], 
                'idm_T': 2.0,   
                's0': 6.0,
                # --- 新增的横向语义参数 ---
                'yield_to_lc': True,       # 是否愿意对换道车让行
                'lat_safe_margin': 3.0,    # 横向博弈时的安全距离余量极大
                'lc_efficiency_decay': 0.2 # 别人换道时，不在乎自己掉速
            },
            'Normal': { # 普通型：纵向均衡，横向看情况
                'weights': [2.0, 1.0, 0.0],  
                'idm_T': 1.2,   
                's0': 4.0,
                # --- 新增的横向语义参数 ---
                'yield_to_lc': False,      # 默认不主动让，但逼近了会减速
                'lat_safe_margin': 1.0, 
                'lc_efficiency_decay': 0.8
            },
            'Force': {  # 激进型：纵向贴屁股，横向死活不让
                'weights': [0.5, 100.0, 0.0],  
                'idm_T': 0.3,   
                's0': 1.0,       
                # --- 新增的横向语义参数 ---
                'yield_to_lc': False,      # 绝不让行
                'lat_safe_margin': 0.1,    # 敢于贴身防守（允许极小的 TTC）
                'lc_efficiency_decay': 2.0 # 别人一换道，自己反而加速（效率权重翻倍）
            }
        }
                
        # 3. 初始化信念分布字典 (Belief Distribution)
        # 替代原有的 weight_dict，用于存储每个交互对象的意图概率
        # 格式: { vehicle_id: [P(Yield), P(Normal), P(Force)] }
        self.belief_dict = {}
        
        # TODO：IMPORTANT: 主车风格可调节参数
        self.AV_style = AV_style
        style_params = STYLE_PARAMS[AV_style]
        self.style_params = style_params
        
        # 换道长度
        self.L_min = style_params['L_min']
        self.change_t = style_params['change_t']
        # 换道动作diff阈值
        self.path_diff_action_threshold = style_params['path_diff_action_threshold']
        # 换道危险判断条件
        self.safechange_threshold = style_params['safechange_threshold'] # 静止距离
        self.k_force_lead_TTC = style_params['k_force_lead_TTC'] # 换道时与前车的TTC
        self.k_force_follow_TTC = style_params['k_force_follow_TTC'] # 换道时与后车的TTC
        self.k_free_lead_TTC = style_params['k_free_lead_TTC'] # 换道时与前车的TTC
        self.k_free_follow_TTC = style_params['k_free_follow_TTC'] # 换道时与后车的TTC
        self.k_free_TTC = style_params['k_free_TTC'] # 换道时与前后车的TTC统一系数
        self.k_force_bigcar = style_params['k_force_bigcar'] # 换道时大车对换道间距的影响系数
        self.k_free_bigcar = style_params['k_free_bigcar'] # 换道时大车对换道间距的影响系数
        # 对HV跟车动作的预计折减
        self.hv_idm_discount = style_params['hv_idm_discount']
        # 是否启用博弈结果用于换道安全检测
        self.is_hv_gamepred = style_params['is_hv_gamepred']
        # 与前车/冲突前车相关参数
        self.idm_T = style_params['idm_T'] # 安全跟车时距，不能调很低
        self.idm_a = style_params['idm_a']
        self.v_control = style_params['v_control']
        # 最大速度 # 前处理模块 + 博弈内部处理
        self.v0 = (self.limits[self.type_mapping['AV']]['MAX_SPEED'] + style_params['v0+']) * self.v_control
        # 初始化casadi
        self.max_n_players = 6 # 最大玩家数量 含有ego
        if self.decision_config:
            self.max_iters = int(self.decision_config.get('max_game_iterations', 3))
            self.converged_eps = float(self.decision_config.get('acceleration_tolerance_ms2', 0.05))
            self.t_list = np.asarray(self.decision_config.get('reward_times_s', [0.1, 0.5, 1., 2., 4., 8.]), dtype=float)
            self.LONG_REWARD = True
            self.n_time = len(self.t_list)
            self.dt = float(self.decision_config.get('control_dt_s', 0.1))
            self.t_output_av = np.arange(self.dt, 8. + self.dt / 2, self.dt)
            self.max_solve_time = float(self.decision_config.get('solve_budget_ms', 100.)) / 1000.
            self.collision_margin_m = float(self.decision_config.get('collision_margin_m', 0.))
            self.force_ignore_idm_on_av_lc = bool(self.decision_config.get('force_ignore_idm_on_av_lc', False))
        self._init_solvers()

        self.decel_from_speedlimit = False
    
    def init_config(self, AV_style="Normal"):
        self.AV_style = AV_style
        style_params = STYLE_PARAMS[AV_style]
        self.style_params = style_params
        
        # 换道长度
        self.L_min = style_params['L_min']
        self.change_t = style_params['change_t']
        # 换道动作diff阈值
        self.path_diff_action_threshold = style_params['path_diff_action_threshold']
        # 换道危险判断条件
        self.safechange_threshold = style_params['safechange_threshold'] # 静止距离
        self.k_force_lead_TTC = style_params['k_force_lead_TTC'] # 换道时与前车的TTC
        self.k_force_follow_TTC = style_params['k_force_follow_TTC'] # 换道时与后车的TTC
        self.k_free_lead_TTC = style_params['k_free_lead_TTC'] # 换道时与前车的TTC
        self.k_free_follow_TTC = style_params['k_free_follow_TTC'] # 换道时与后车的TTC
        self.k_free_TTC = style_params['k_free_TTC'] # 换道时与前后车的TTC统一系数
        self.k_force_bigcar = style_params['k_force_bigcar'] # 换道时大车对换道间距的影响系数
        self.k_free_bigcar = style_params['k_free_bigcar'] # 换道时大车对换道间距的影响系数
        # 对HV跟车动作的预计折减
        self.hv_idm_discount = style_params['hv_idm_discount']
        # 是否启用博弈结果用于换道安全检测
        self.is_hv_gamepred = style_params['is_hv_gamepred']
        # 与前车/冲突前车相关参数
        self.idm_T = style_params['idm_T'] # 安全跟车时距，不能调很低
        self.idm_a = style_params['idm_a']
        self.v_control = style_params['v_control']
        # 最大速度 # 前处理模块 + 博弈内部处理
        self.v0 = (self.limits[self.type_mapping['AV']]['MAX_SPEED'] + style_params['v0+']) * self.v_control
        
    # 初始化 casadi 优化器
    def _init_solvers(self):
        self.SOLVER = {}
        
        for n_p in range(2, self.max_n_players + 1):
            # 重新构建一个 n_p 的优化器（会将结果写到 self.opti / self.A / ...）
            self._build_player_solver(N=len(self.t_list), n_player=n_p)
            # 捕获当前构建的 Opti 对象和关键变量，保存在字典中
            self.SOLVER[n_p] = {
                'opti': self.opti,
                'AA': self.AA,
                'A': self.A,
                's': self.s,
                'v': self.v,
                'player_idx': self.player_idx,
                'a0': self.a0,
                # 'a_prev': self.a_prev,
                # 's_prev': self.s_prev,
                # 'v_prev': self.v_prev,
                'players_types': self.players_types,
                'players_ids': self.players_ids,
                'av_confpos': self.av_confpos,
                'hv_confpos_2av': self.hv_confpos_2av,
                'weight0': self.weight0,
                'weight1': self.weight1,
                'weight2': self.weight2,
                "a_max": self.a_max,
                "a_min": self.a_min,
                "v_max": self.v_max,
                'a_prev_k': self.a_prev_k,
                's_prev_k': self.s_prev_k,
                'v_prev_k': self.v_prev_k,
                'intent_probs': self.intent_probs,
                'cost_function': ca.Function('game_cost_' + str(n_p), [self.opti.x, self.opti.p], [self.opti.f]).expand(),
                'parameter_vector': self.opti.p,
                'initial_speed': self.initial_speed,
            }
    
    def _build_player_solver(self, N, n_player):
        """
        Refactored to construct the solver once and reuse it.
        """
        opti = ca.Opti()
        
        # Optimization variables
        if self.SHARED_ACC: # n_time共享加速度
            
            AA = opti.variable(1)  # Acceleration sequence
            A = ca.repmat(AA, N, 1)
        else: # 独立加速度
            A = opti.variable(N)
        
        # Parameters
        player_idx = opti.parameter()
        a0 = opti.parameter()
        n_time = len(self.t_list)

        # 【贝叶斯改造】将单一预测轨迹扩展为 K 种意图的轨迹列表
        self.K_intents = len(self.intent_types) if hasattr(self, 'intent_types') else 3
        a_prev_k = [opti.parameter(n_player, n_time) for _ in range(self.K_intents)]
        s_prev_k = [opti.parameter(n_player, n_time) for _ in range(self.K_intents)]
        v_prev_k = [opti.parameter(n_player, n_time) for _ in range(self.K_intents)]
        
        # 【贝叶斯改造】信念概率分布矩阵，形状 (n_player, K_intents)
        intent_probs = opti.parameter(n_player, self.K_intents)
        
        players_types = opti.parameter(n_player)

        # a_prev = opti.parameter(n_player, n_time)
        # s_prev = opti.parameter(n_player, n_time)
        # v_prev = opti.parameter(n_player, n_time)
        # players_types = opti.parameter(n_player)
        players_ids = opti.parameter(n_player)
        av_confpos = opti.parameter(n_player)
        hv_confpos_2av = opti.parameter(n_player)
        
        self.weight0 = opti.parameter(n_player)  # safety weight
        self.weight1 = opti.parameter(n_player)  # efficiency weight
        self.weight2 = opti.parameter(n_player)  # comfort weight
        
        # Dynamics
        s0 = sum(ca.if_else(player_idx==i, s_prev_k[0][i,0], 0.0) for i in range(n_player))
        initial_speed = opti.parameter()
        v0 = initial_speed
        s_list = [0.0] # TODO：修改为0开始，是否有问题？
        s_list = [0.0] # TODO：修改为0开始，是否有问题？
        v_list = [v0]
        for t in range(N):
            dt = self.t_list[t]-self.t_list[t-1] if t>0 else self.t_list[t]
            active_dt = ca.if_else(A[t] < 0, ca.fmin(dt, v_list[-1] / ca.fmax(-A[t], 1e-6)), dt)
            s_next = s_list[-1] + v_list[-1]*active_dt + 0.5*A[t]*active_dt**2
            v_next = ca.fmax(v_list[-1] + A[t]*dt, 0.)
            # v_next = ca.fmax(v_next, 0.0)  # 速度非负约束
            # v_next = ca.fmin(v_next, v_max)
            s_list.append(s_next)
            v_list.append(v_next)
        s = ca.vertcat(*s_list[1:])
        v = ca.vertcat(*v_list[1:])
        # Cost function
        J = self._evaluate_reward_symbolic((A, s, v), (player_idx, a0, a_prev_k, s_prev_k, v_prev_k, intent_probs, players_types, players_ids, av_confpos, hv_confpos_2av))

        opti.minimize(J)
        # Constraints
        # 将加速度与速度上限改为可设置的参数（外部可通过 opti.set_value 修改）
        a_max = opti.parameter()   # 可覆盖的最大加速度
        a_min = opti.parameter()   # 可覆盖的最小加速度（负值）
        v_max = opti.parameter()   # 可覆盖的最大速度
        
        for t in range(N):
            opti.subject_to(A[t] <= a_max)
            opti.subject_to(A[t] >= a_min)
            opti.subject_to(v[t] <= v_max)
            opti.subject_to(v[t] >= 0)
        
        # opti.set_initial(A, np.zeros(N))
        # Solver options
        opti.solver('ipopt', {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-8,
            'ipopt.max_cpu_time': self.max_solve_time,
            # 'ipopt.max_cpu_time': self.max_solve_time,
            })
        # opti.solver('ipopt', {
        #     'print_time': 1,
        #     'ipopt.print_level': 5,     # 更详细
        #     'ipopt.max_cpu_time': 2.0,  # 设短时间看看是否超时
        #     'ipopt.tol': 1e-4,
        #     'ipopt.acceptable_tol': 1e-3,
        #     'ipopt.acceptable_constr_viol_tol': 1e-3,
        # })
        # Store the optimizer and variables
        self.opti = opti
        self.initial_speed = initial_speed
        self.AA = AA
        self.A = A
        self.s = s
        self.v = v
        self.player_idx = player_idx
        self.a0 = a0

        self.a_prev_k = a_prev_k
        self.s_prev_k = s_prev_k
        self.v_prev_k = v_prev_k
        self.intent_probs = intent_probs
        
        self.players_types = players_types
        self.players_types = players_types
        self.players_ids = players_ids
        self.av_confpos = av_confpos
        self.hv_confpos_2av = hv_confpos_2av
        self.a_max = a_max
        self.a_min = a_min
        self.v_max = v_max
    
    # 动态更新权重 TODO
    # def _update_weights_old(self, other_ids):
    #     if not other_ids:
    #         return
    #     for other_id in other_ids:
    #         if other_id not in self.weight_dict and other_id!= 'AV':
    #             self.weight_dict[other_id] = self.other_init_weight
    #         elif other_id in self.weight_dict and other_id!= 'AV':
    #             pass # TODO: 动态更新权重?
        
    #     for other_id in list(self.weight_dict.keys()):
    #         if other_id not in other_ids and other_id!= self.id_mapping['AV']:
    #             del self.weight_dict[other_id]
        
    #     self.weight_keys = np.array(list(self.weight_dict.keys()), dtype=float)
    #     self.weight_values = np.array([self.weight_dict[k] for k in self.weight_dict], dtype=float)
    def inject_llm_beliefs(self, probs_dict):
        """
        外部接口：用于随时更新 LLM 或其他预测模块的结果。
        格式要求: { vehicle_id: [P(Yield), P(Normal), P(Force)] }
        """
        validated = {}
        for vehicle_id, values in probs_dict.items():
            probabilities = np.asarray(values, dtype=float)
            if probabilities.shape != (3,) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or probabilities.sum() <= 0:
                raise ValueError(f"Invalid belief for vehicle {vehicle_id}")
            validated[vehicle_id] = (probabilities / probabilities.sum()).tolist()
        self.external_probs = validated
        print(f"[GAME模块] 已接收外部 LLM 信念注入: {self.external_probs}")


    def _update_beliefs(self, others_info):
        """
        【重构】第一阶段：在线更新人类驾驶车辆（HV）的意图信念分布 (Belief Distribution)。
        接管并替代原有的 _update_weights 功能。
        
        参数:
            others_info: 包含 HV 当前状态和预测轨迹的字典
            external_probs: (预留给LLM) 外部传入的概率字典，例如 {'car_1': [0.1, 0.7, 0.2]}
        """
        if others_info['ids'] is None:
            return 
            
        other_ids = others_info['ids']
        other_states = others_info['states']
        
        # 0. 初始化他车信念
        for other_id in other_ids:
            if other_id != self.id_mapping['AV'] and other_id not in self.belief_dict:
                # 初始化新出现 HV 的权重（确保是可变列表）
                self.belief_dict[other_id] = list(self.other_init_belief)

        # 1. 清理已离开场景的他车信念
        for other_id in list(self.belief_dict.keys()):
            if other_id != self.id_mapping['AV'] and other_id not in other_ids:
                del self.belief_dict[other_id]
                
        # 2. 为当前场景中的每辆他车更新信念
        for idx, other_id in enumerate(other_ids):
            if other_id == self.id_mapping['AV']:
                continue
                
            # 接入了外部 LLM 
            if self.external_probs is not None and other_id in self.external_probs:
                raw_probs = self.external_probs[other_id]
            else:
                # 【当前测试用】伪造一个基于速度差的启发式概率生成器
                # 获取他车当前速度 (索引 3)
                v_other = other_states[idx][3] if other_states is not None else 10.0
                
                # 简单规则：速度很快 -> 更有可能抢行；速度很慢 -> 更有可能让行
                if v_other > 15.0:  # > 54km/h 偏激进
                    raw_probs = [0.1, 0.3, 0.6] # [P(Yield), P(Normal), P(Force)]
                elif v_other < 5.0: # < 18km/h 偏保守
                    raw_probs = [0.6, 0.3, 0.1]
                else:               # 正常速度
                    raw_probs = [0.2, 0.6, 0.2]
                    
            # 3. 概率归一化 (确保和为 1.0)
            prob_sum = sum(raw_probs)
            if prob_sum > 0:
                normalized_probs = [p / prob_sum for p in raw_probs]
            else:
                normalized_probs = [1.0, 0.0, 0.0] # 兜底策略
            


            # 4. 存入信念字典
            self.belief_dict[other_id] = normalized_probs
            
        # ==========================================================
        # 【向下兼容补丁】为了防止第三阶段重构前，旧的 CasADi 报错，
        # 暂时维持 weight_dict 的存在，统一赋予 Normal 权重
        # ==========================================================
        for other_id in other_ids:
            if other_id != self.id_mapping['AV']:
                self.weight_dict[other_id] = self.intent_profiles['Normal']['weights']
        self.weight_dict[self.id_mapping['AV']] = self.av_weights
        
        # self.weight_keys = np.array(list(self.weight_dict.keys()), dtype=float)
        # self.weight_values = np.array([self.weight_dict[k] for k in self.weight_dict], dtype=float)
        # ==========================================================

        if self.roslog:
            rospy.loginfo(f"[GAME模块] 他车意图信念分布: {self.belief_dict}")
        else:
            print(f"[GAME模块] 他车意图信念分布: {self.belief_dict}")

    # def _update_weights(self, others_info):
    #     """
    #     在线更新人类驾驶车辆（HV）的多目标权重（安全、效率、舒适性）。
        
    #     参数:
    #         others_info: 包含 HV 当前状态和预测轨迹的字典，结构为：
    #             {
    #                 'states': np.ndarray (M, 6),   # [x, y, heading, speed, acc, ?]
    #                 'pred': np.ndarray (N+1, M, 5),# 预测轨迹（匀速模型）
    #                 'ids': list of vehicle IDs,
    #                 'types': ...
    #             }
    #     """
    #     if others_info['ids'] is None:
    #         return 
    #     # ========== 第一部分：准备状态数据 ==========
    #     other_ids = others_info['ids']
    #     others_pred = others_info['pred']         # 形状 (N+1, M, 5)
    #     other_ids_from_info = others_info['ids']  # 顺序与 states 对应
    #     # ========== 第一部分：维护 weight_dict 的存在性（原始逻辑） ==========
    #     for other_id in other_ids:
    #         if other_id != self.id_mapping['AV'] and other_id not in self.weight_dict:
    #             # 初始化新出现 HV 的权重（确保是可变列表）
    #             self.weight_dict[other_id] = list(self.other_init_weight)  # 如 [10.0, 0.5, 0.02]

    #     # 删除已离开场景的车辆权重
    #     for other_id in list(self.weight_dict.keys()):
    #         if other_id != self.id_mapping['AV'] and other_id not in other_ids:
    #             del self.weight_dict[other_id]

    #     # 构建 ID 到索引的映射
    #     id_to_idx = {vid: i for i, vid in enumerate(other_ids_from_info)}

    #     # ========== 第四部分：对每个 HV 进行权重在线更新 ==========
    #     for other_id in other_ids:
    #         if other_id == self.id_mapping['AV'] or other_id not in id_to_idx:
    #             self.weight_dict[self.id_mapping['AV']] = self.av_weights
    #             continue

    #         # 获取当前权重（3维）
    #         w = np.array(self.weight_dict[other_id])

    #         # 计算真实与 PG 假设下的奖励
    #         if not (other_id in self.old_style_reward_dict and other_id in self.style_reward_dict):
    #             continue
    #         r_real = self.old_style_reward_dict[other_id] * np.array([0, 1, 0]) # 安全\舒适都不是线性的, 因此只用效率进行计算
    #         r_pg = self.style_reward_dict[other_id] * np.array([0, 1, 0])

    #         # 计算损失：L = (r_pg · w) - (r_real · w)
    #         loss_orig = np.dot(r_pg, w) - np.dot(r_real, w)

    #         # 数值梯度（有限差分）
    #         h = 1e-5
    #         grad = np.zeros_like(w)
    #         for i in range(len(w)):
    #             w_perturb = w.copy()
    #             w_perturb[i] += h
    #             loss_perturb = np.dot(r_pg, w_perturb) - np.dot(r_real, w_perturb)
    #             grad[i] = (loss_perturb - loss_orig) / h

    #         # 梯度下降更新权重（学习率 0.1）
    #         new_w = w.copy()
    #         for i in range(len(w)):
    #             if not (grad[i] == 0 or math.isnan(grad[i]) or math.isinf(grad[i])) and i == 1: # i == 1表示只更新效率权重
    #                 new_w[i] = w[i] - 0.1 * grad[i]

    #         # 更新权重字典
    #         self.weight_dict[other_id] = new_w.tolist()

        # # 根据风格辨识进行权重更新
        # for key, value in self.other_style_dict.items():
        #     if key in self.weight_dict:
        #         self.weight_dict[key][1] += 0.01 * self.style_map[value]
        
        # # ========== 第五部分：更新用于外部访问的数组 ==========
        # self.weight_keys = np.array(list(self.weight_dict.keys()), dtype=float)
        # self.weight_values = np.array([self.weight_dict[k] for k in self.weight_dict], dtype=float)
        # if self.roslog:
        #     rospy.loginfo(f"[GAME模块]他车权重: {self.weight_dict}")
        # else:
        #     print(f"[GAME模块]他车权重: {self.weight_dict}")

    
    # 静态函数：用于处理优化问题
    @staticmethod
    def _smooth_min(a, b, alpha=20.0):
        """
        稳定的 soft-min 实现（数值平移以避免 exp 溢出）。
        返回近似 min(a,b)。a,b 可为 CasADi 符号或数值。
        """
        eps = 1e-12
        # m = min(a,b)
        m = ca.fmin(a, b)
        # use stable log-sum-exp shift by m
        return m - (1.0/alpha) * ca.log(ca.exp(-alpha*(a - m)) + ca.exp(-alpha*(b - m)) + eps)
    
    def _smooth_clip(self, x, xmin, xmax, alpha=10.0):
        """平滑裁剪到 [xmin, xmax]"""
        # x_clipped = xmin + (xmax-xmin) * sigmoid(alpha*(x - xmin)/(xmax-xmin))
        return xmin + (xmax - xmin) / (1 + ca.exp(-alpha*(x - (xmin+xmax)/2)/(xmax-xmin)))
    
    # 路径生成：避免np.grandient Error
    def _get_sigmoid_path(self, start_pose, end_lane, speed, step=0.2, lat_ratio=1.0):
        """
        生成局部平滑换道轨迹，衔接车辆当前位置与目标车道
        Args:
            start_pose: (x0, y0, heading0) in global coordinates
            end_lane: np.ndarray [[x, y], ...] of target lane
            speed: 当前速度，用于计算换道长度
            step: 采样间隔
        Returns:
            np.ndarray: shape (N, 4) -> [x, y, heading, curvature]
        """
        x0, y0, heading0, v0 = start_pose
        
        # 1. 找到目标车道上距离车辆当前位置最近的点
        distances = np.linalg.norm(end_lane[:, :2] - np.array([x0, y0]), axis=1)
        idx_closest = np.argmin(distances)
        
        # 2. 根据速度确定换道长度 L
        L = np.clip(speed * self.change_t, self.L_min, self.L_max)
        L = np.clip(distances[idx_closest] / self.L_delta, 0, 1) * L #

        # 3. 找到距离 L 最近的目标点索引
        s_cum = np.cumsum(np.hypot(np.diff(end_lane[:,0], prepend=end_lane[0,0]),
                                    np.diff(end_lane[:,1], prepend=end_lane[0,1])))
        
        s0 = s_cum[idx_closest]
        s_target = s0 + L
        idx_target = np.searchsorted(s_cum, s_target)
        idx_target = min(idx_target, len(end_lane)-1)
        
        x1, y1 = end_lane[idx_target][:2]
        # 目标航向沿车道切线
        if idx_target < len(end_lane)-1:
            dx = end_lane[idx_target+1,0] - end_lane[idx_target,0]
            dy = end_lane[idx_target+1,1] - end_lane[idx_target,1]
        else:
            dx = x1 - end_lane[idx_target-1,0]
            dy = y1 - end_lane[idx_target-1,1]
        heading1 = np.arctan2(dy, dx)
        
        # 4. 构建局部坐标系: 起点为原点, x轴沿车辆初始航向
        cos_h0, sin_h0 = np.cos(heading0), np.sin(heading0)
        def to_local(xg, yg):
            dxg = xg - x0
            dyg = yg - y0
            xl =  cos_h0 * dxg + sin_h0 * dyg
            yl = -sin_h0 * dxg + cos_h0 * dyg
            return xl, yl
        
        def to_global(xl, yl):
            xg = cos_h0 * xl - sin_h0 * yl + x0
            yg = sin_h0 * xl + cos_h0 * yl + y0
            return xg, yg
        
        x1_local, y1_local = to_local(x1, y1)
        heading1_local = heading1 - heading0
        y1_local = y1_local * lat_ratio 
        heading1_local = (heading1 - heading0) * lat_ratio 

        # 5. 使用三次 Hermite 插值
        # 起点斜率 m0, 终点斜率 m1
        m0 = np.tan(0)  # 局部坐标下初始沿x轴
        m1 = np.tan(heading1_local)
        
        # Hermite 多项式: y(s) = a*s^3 + b*s^2 + c*s + d
        # 边界条件: y(0)=0, y(L)=y1_local, y'(0)=m0, y'(L)=m1
        L_local = x1_local
        A = np.array([[0,     0,   0, 1],
                    [L_local**3, L_local**2, L_local, 1],
                    [0,     0,   1, 0],
                    [3*L_local**2, 2*L_local, 1, 0]])
        B = np.array([0, y1_local, m0, m1])
        coeffs = np.linalg.solve(A, B)
        
        a, b, c, d = coeffs
        
        # 6. 采样
        s_vals = np.arange(0, L_local+step, step)
        y_vals = a*s_vals**3 + b*s_vals**2 + c*s_vals + d
        dy_ds = 3*a*s_vals**2 + 2*b*s_vals + c
        d2y_ds2 = 6*a*s_vals + 2*b
        
        # heading & curvature（heading用dy,dx推算）
        x_global, y_global = to_global(s_vals, y_vals)
        # 安全处理：避免 np.gradient 输入长度不足
        if x_global.size > 1:
            dx_global = np.gradient(x_global)
        else:
            dx_global = np.zeros_like(x_global)
        if y_global.size > 1:
            dy_global = np.gradient(y_global)
        else:
            dy_global = np.zeros_like(y_global)
        headings = np.arctan2(dy_global, dx_global)
        curvature = d2y_ds2 / (1 + dy_ds**2)**1.5

        # 8. 衔接目标车道后续点
        if (end_lane[idx_target][0] - end_lane[-1][0])**2 + (end_lane[idx_target][1] - end_lane[-1][1])**2 > 10.0**2:
            remaining_pts = end_lane[idx_target+1:]
            # 安全处理：避免 np.gradient 输入长度不足
            if remaining_pts.shape[0] > 1:
                dx_rem = np.gradient(remaining_pts[:,0])
                dy_rem = np.gradient(remaining_pts[:,1])
            else:
                dx_rem = np.zeros_like(remaining_pts[:,0])
                dy_rem = np.zeros_like(remaining_pts[:,1])
            # heading用end_lane的第三维
            if remaining_pts.shape[1] >= 3:
                headings_rem = remaining_pts[:,2]
            else:
                headings_rem = np.arctan2(dy_rem, dx_rem)

            if lat_ratio != 1.0:
                # 获取目标车道在拼接点处的绝对坐标和航向
                ref_x, ref_y = end_lane[idx_target, 0], end_lane[idx_target, 1]
                ref_heading = heading1
                
                # 获取我们生成的轨迹末端的绝对坐标
                last_x, last_y = x_global[-1], y_global[-1]
                
                # 计算法向偏差 (投影到目标车道的法向量 [-sin(θ), cos(θ)] 上)
                nx = -np.sin(ref_heading)
                ny = np.cos(ref_heading)
                offset_mag = (last_x - ref_x) * nx + (last_y - ref_y) * ny
                
                # 根据偏差值，对所有 remaining_pts 进行法向平移
                # 注意：每个点沿着自己的法向量 [-sin(headings_rem), cos(headings_rem)] 平移
                shift_x = offset_mag * (-np.sin(headings_rem))
                shift_y = offset_mag * np.cos(headings_rem)
                
                # 创建平移后的新点阵
                remaining_pts_shifted = np.copy(remaining_pts)
                remaining_pts_shifted[:, 0] += shift_x
                remaining_pts_shifted[:, 1] += shift_y
            else:
                remaining_pts_shifted = remaining_pts
                
            if remaining_pts.shape[0] > 1:
                ddx_rem = np.gradient(remaining_pts[:,0])
                ddy_rem = np.gradient(remaining_pts[:,1])
            else:
                ddx_rem = np.zeros_like(remaining_pts[:,0])
                ddy_rem = np.zeros_like(remaining_pts[:,1])
            curvature_rem = (dx_rem*ddy_rem - dy_rem*ddx_rem)/(dx_rem**2 + dy_rem**2 + 1e-6)
            traj_ext = np.column_stack([remaining_pts_shifted[:,0], remaining_pts_shifted[:,1], headings_rem, curvature_rem])
            traj = np.vstack([np.column_stack([x_global, y_global, headings, curvature]), traj_ext])
            return traj, len(s_vals)
        else:
            if self.roslog:
                rospy.loginfo("[GAME模块]  'None' from [_get_sigmoid_path] no enough points.")
            else:
                print("'None' from [_get_sigmoid_path] no enough points.")
            traj = np.column_stack([x_global, y_global, headings, curvature])
            return traj, len(s_vals)
    
    def _get_target_path(self, target_idx: int, lanes: dict, curr_lane_idx: int, ego_pose: tuple, lat_ratio=1.0):
        """
        获取目标轨迹（直行或换道），以车辆当前位置为起点
        Args:
            target_idx: 目标车道索引
            lanes: {lane_id: [Point, ...]} 格式的车道信息
            curr_lane_idx: 当前车道索引
            ego_pose: 车辆当前位置 (x, y, heading, v)
        Returns:
            np.ndarray: shape (N, 4) -> [x, y, heading, curvature]
        """
        ego_x, ego_y, _, ego_v = ego_pose
        t1 = time.time()
        # === Case 1: 同车道，无需换道 ===
        if target_idx == curr_lane_idx and self.changing == False:
            lane_points = lanes[curr_lane_idx]
            N = len(lane_points)
            lane_arr = np.empty((N,4))  # [x, y, heading(rad), curvature]
            for i in range(N):
                lane_arr[i,0] = lane_points[i].x
                lane_arr[i,1] = lane_points[i].y
                lane_arr[i,2] = lane_points[i].heading
                lane_arr[i,3] = lane_points[i].curvature
            
            _, start_idx = nearest_point_on_traj(np.array([ego_x, ego_y]), lane_arr[:,:2])
            return lane_arr[start_idx:], None
        
        # === Case 2: 换道 ===
        lane_to = lanes[target_idx]
        speed = ego_v  # 或使用车辆当前速度
        N = len(lane_to)
        end_lane_array = np.empty((N,4))
        for i in range(N):
            end_lane_array[i,0] = lane_to[i].x
            end_lane_array[i,1] = lane_to[i].y
            end_lane_array[i,2] = lane_to[i].heading
            end_lane_array[i,3] = lane_to[i].curvature

        _, start_idx = nearest_point_on_traj(np.array([ego_x, ego_y]), end_lane_array[:,:2])
        N = min(len(lane_to), self.path_foresee_num)  # 限制最大点数
        end_lane_array = end_lane_array[min(int(start_idx-5), 0):start_idx+N]
        # ==========================================================
        # 【核心修复：将“相对探头”改为瞄准一条“绝对的虚拟目标线”】
        # ==========================================================
        if lat_ratio != 1.0:
            # 1. 提取自车所在的原始车道线
            lane_from = lanes[curr_lane_idx]
            curr_lane_array = np.empty((len(lane_from), 4))
            for i in range(len(lane_from)):
                curr_lane_array[i,0] = lane_from[i].x
                curr_lane_array[i,1] = lane_from[i].y
                curr_lane_array[i,2] = lane_from[i].heading
            
            # 2. 找到车辆在“原始车道”和“目标车道”上对应的物理基准点
            _, idx_curr = nearest_point_on_traj(np.array([ego_x, ego_y]), curr_lane_array[:,:2])
            curr_ref_x, curr_ref_y = curr_lane_array[idx_curr, 0], curr_lane_array[idx_curr, 1]
            
            # 这里的 5 对应上面切片时的 start_idx 的相对位置
            target_ref_x, target_ref_y, target_ref_h = end_lane_array[5, 0], end_lane_array[5, 1], end_lane_array[5, 2] 
            
            # 3. 计算从目标车道指向原始车道的法向距离 (绝对车道宽度，带正负号)
            dx = curr_ref_x - target_ref_x
            dy = curr_ref_y - target_ref_y
            nx = -np.sin(target_ref_h)
            ny = np.cos(target_ref_h)
            lane_dist_signed = dx * nx + dy * ny  
            
            # 4. 计算目标线需要平移的量 
            # (1 - lat_ratio) 代表虚拟线距离目标车道中心的比例
            # 例如 lat_ratio=0.4，这条线将永远固定在距离目标车道中心 60% 车道宽度的位置！
            shift_mag = (1.0 - lat_ratio) * lane_dist_signed
            
            # 5. 对整条目标车道线进行几何平移，生成一条绝对存在的“压线轨迹”
            end_lane_array[:, 0] += shift_mag * (-np.sin(end_lane_array[:, 2]))
            end_lane_array[:, 1] += shift_mag * (np.cos(end_lane_array[:, 2]))
        # if self.roslog:
        #     rospy.loginfo(f"换道前准备耗时 {(time.time()-t1)*1000} ms")
        # 调用优化后的 _get_sigmoid_path
        # v2 already encodes the partial lateral displacement in the absolute
        # virtual target line above. Scaling again would produce ratio**2,
        # e.g. only 0.36 m rather than 1.2 m for a 4 m lane and ratio=0.3.
        interpolation_ratio = 1.0 if self.decision_config else lat_ratio
        traj, change_end_idx = self._get_sigmoid_path(ego_pose, end_lane_array, speed, lat_ratio=interpolation_ratio)
        return traj, change_end_idx
    
    def _find_lead_and_idm( self,
                            ego_state,
                            ego_path,
                            path_s,
                            other_states,
                            other_lengths,
                            av_confpos,
                            hv_confpos_2av,
                            other_ids=None,
                            max_lateral_dist=2.0,
                            idm_params=None,
                            is_other='AV'):
        """
        查找 ego 在所给路径上的前车并用 IDM 计算跟驰加速度。
        Args:
            ego_state: array-like [x, y, heading, v, a, ...] （速度位于索引3）
            ego_path: np.ndarray (N,2) or (N,>=2) 轨迹点（至少包含 x,y）
            path_s: np.ndarray (N,) 对应 ego_path 的弧长（与 ego_path 索引对应）
            other_states: iterable of states, 每项至少包含 [x, y, ..., v, ...] 且速度在索引3
            other_ids: 可选，other_states 对应的 id 列表
            max_lateral_dist: 认定为同道的横向阈值 (m)
            idm_params: dict 可选 IDM 参数覆盖，支持 keys: a, b, s0, T, delta, v0
        Returns:
            dict:
                'lead_index': index in other_states of chosen lead (or None)
                'lead_id': 对应 id 或 None
                'lead_state': lead 状态或 None
                'lead_s': lead 在 path_s 上的位置或 None
                'a_idm': IDM 计算得到的加速度（已裁剪到 AV 限制）
        """
        # prepare arrays
        ego = np.asarray(ego_state)
        others = np.asarray(other_states)
        if others.ndim == 1:
            others = others[np.newaxis, :]
        # xy path
        ego_path = np.asarray(ego_path)
        if ego_path.shape[1] >= 2:
            path_xy = ego_path[:, :2]
        else:
            raise ValueError("ego_path 必须包含 x,y 列")
        path_s = np.asarray(path_s)
        
        # ego longitudinal position on path: assume path 起点为当前车辆位置 -> s_ego = path_s[0]
        s_ego = float(path_s[0]) if path_s.size > 0 else 0.0
        v_ego = float(ego[3])
        
        max_a = float(self.limits[self.type_mapping['AV']]['MAX_ACCELERATION'])
        min_a = float(self.limits[self.type_mapping['AV']]['MAX_DECELERATION'])
        max_v = float(self.limits[self.type_mapping['AV']]['MAX_SPEED'])

        # IDM 默认参数（可被 idm_params 覆盖）
        idm_defaults = {
            'a': self.idm_a,        # 最大加速度 m/s^2 (typical comfortable accel)
            'b': 2.0,        # 期望舒适减速度 m/s^2
            's0': self.s0,       # 最小间距 m
            'T': self.idm_T,        # 时距 s
            'delta': 4.0,    # 速度指数
            'v0': float(self.v0)  # 期望速度
        }
        
        if idm_params is not None:
            idm_defaults.update(idm_params)
        a_idm_param = idm_defaults['a']
        b_idm_param = idm_defaults['b']
        s0 = idm_defaults['s0']
        T = idm_defaults['T']
        # delta = idm_defaults['delta']
        # 1228
        if v_ego < 0.7 * self.v0:
            delta = 4
        elif 0.7 * self.v0 <= v_ego < self.v0:
            delta = 4-2*( v_ego- 0.7 *self.v0)/(0.3*self.v0)
        else:
            delta = 2
        if self.roslog:
            rospy.loginfo(f"delta: {delta}")
        else:
            print(f"delta: {delta}")
        v0 = idm_defaults['v0']
        dt = self.dt
        k = 0.1

        best_idx = None
        best_s = np.inf
        best_state = None
        lead_id = None
        best_ol = 0
        
        # 如果周围没有车辆
        if other_states is None:
            if self.roslog:
                rospy.loginfo(f"Leader id:{lead_id}, Leads{best_s}, IDM加速度: {min(max_a, (max_v - v_ego)/dt):.2f} m/s²")
            else:
                print(f"Leader id:{lead_id}, Leads{best_s}, IDM加速度: {min(max_a, (max_v - v_ego)/dt):.2f} m/s²")
            return {
                'lead_index': best_idx,
                'lead_id': lead_id,
                'lead_state': best_state,
                'lead_s': best_s,
                'a_idm': max_a,
            }
        
        same_lane_otherids = []
        # 遍历所有其他车辆，判断是否在本 path 上并且位于前方
        for idx in range(others.shape[0]):
            ox = float(others[idx, 0])
            oy = float(others[idx, 1])
            # nearest_point_on_traj 返回 (min_dist, nearest_idx)
            try:
                d, nearest_idx = nearest_point_on_traj(np.array([ox, oy], dtype=np.float64), path_xy.astype(np.float64))
            except Exception:
                # 回退：用简单欧氏距离检查（当 nearest_point_on_traj 不可用时）
                dists = np.hypot(path_xy[:, 0] - ox, path_xy[:, 1] - oy)
                nearest_idx = int(np.argmin(dists))
                d = float(dists[nearest_idx])
            
            # 横向阈值判断为“同道”
            if d <= max_lateral_dist:
                if is_other == 'AV' and self.now_cal_path_idx != self.now_current_path_idx:
                    if self.others_laneid[idx] != self.now_cal_path_idx:
                        continue

                same_lane_otherids.append(other_ids[idx])
                s_other = float(path_s[nearest_idx])
                # 只考虑前方车辆（严格大于 ego s）
                if s_other > s_ego + 1e-6:
                    if s_other < best_s:
                        best_s = s_other
                        best_idx = idx
                        best_state = others[idx]
                        best_ol = other_lengths[idx]
            
            # 如何计算同车道后方的车
            # 计算车辆与ego的夹角
            heading_ego = float(ego[2])
            o2e = math.atan2(ego[1]-oy, ego[0]-ox)
            angle_diff = abs((heading_ego - o2e + math.pi) % (2*math.pi) - math.pi)
            if angle_diff < math.radians(20): # 10度范围内认为是同车道
                same_lane_otherids.append(other_ids[idx])
        
        # 若未找到前车，则返回自由路况下的 IDM 加速度（无车影响项）
        if best_idx is None:
            # 自由流加速度 (no interaction term)
            # 使用 IDM: a * (1 - (v/v0)^delta)
            if self.force_flag:
                a_free = a_idm_param * (1.0 - (v_ego / v0) ** delta)
            else:
                a_free = a_idm_param * (1.0 - (min(v_ego, v0) / v0) ** delta)
            # 裁剪到 AV 加速度限制
            a_free = float(np.clip(a_free, min_a, max_a))
            acc = a_free
        else:
            l_ego = self.ego_length   # 或从 ego_state 中取
            l_lead = best_ol

            delta_s = best_s - s_ego - 0.5*l_ego - 0.5*l_lead
            delta_s = max(delta_s, 1e-3)
            
            # 计算相对量并应用 IDM
            v_lead = float(best_state[3])
            delta_v = v_ego - v_lead  # closing speed (positive: approaching)
            # delta_s = max(1e-3, best_s - s_ego)  # 与前车的距离，防除零
            
            
            # effective desired gap s_star
            o_l = best_ol # 获取车辆长度
            sqrt_term = 2.0 * math.sqrt(max(1e-9, a_idm_param * b_idm_param))
            
            # s_star = (s0 + k * v_lead) + v_ego * T + (v_ego * delta_v) / sqrt_term
            s_star = (s0) + v_ego * T + (v_ego * delta_v) / sqrt_term
            # s_star = (s0 + o_l/2 - 5/2) + v_ego * T + (v_ego * delta_v) / sqrt_term # 以小车为5m进行计算，根据其他车辆车长调整
            
            # IDM 加速度
            acc = a_idm_param * (1.0 - (v_ego / v0) ** delta - (s_star / delta_s) ** 2)
            
            # 裁剪到 AV 的物理加速度限制
            acc = float(np.clip(acc, min_a, max_a))
            
            # 强制停车逻辑
            if delta_s <= s0/2 + 1e-3 and v_ego < 0.5:  # 距离接近s0且速度很低
                acc = -v_ego / dt 
            
            lead_id = None
            if other_ids is not None and len(other_ids) > best_idx:
                lead_id = other_ids[best_idx]
        
        # 找到与ego存在冲突的车辆
        detect_range = max(v_ego *5, 40)
        acc_conf_min = 100 
        safe_dist_2_rear = 5 + s0  # 与后车的安全距离
        if_disc = True
        _disc_k = 2
        if is_other != 'AV':
            av_confpos_alter = {'AV': hv_confpos_2av[is_other]}
            hv_confpos_2av_alter = {'AV': av_confpos[is_other]}
            av_confpos = av_confpos_alter
            hv_confpos_2av = hv_confpos_2av_alter
        for vid, confpos in av_confpos.items():
            if confpos == -1.0 or vid in same_lane_otherids:
                pass
            else:
                # 计算冲突点到 ego 的距离
                dist_ego = confpos
                dist_other = hv_confpos_2av[vid]
                # print(f"detect_range:{detect_range}, dist_ego:{dist_ego}, dist_other:{dist_other}")
                if is_other == 'AV' and self.now_cal_path_idx != self.now_current_path_idx:
                    if self.others_laneid[other_ids.index(vid)] != self.now_cal_path_idx:
                        if dist_other <= 0.1 and dist_ego > 0.1:
                            continue
                if detect_range >= dist_ego:
                    if dist_other <  dist_ego + safe_dist_2_rear:
                        other_i = others[other_ids.index(vid)] if other_ids is not None else None
                        v_other = float(other_i[3])
                        v_other = max(v_other, 1e-7)
                        o_l = other_lengths[other_ids.index(vid)]
                        # o_l = other_i[-2] if other_i is not None else 5
                        
                        # 1219修复：对于冲突车辆，减速度数值按照到冲突最小距离折减
                        _2conf_discount = 1.0

                        delta_v = v_ego - v_other  # closing speed (positive: approaching)
                        if dist_other < dist_ego:
                            # 计算IDM加速度
                            # 其他车速度（避免为0）
                            l_ego = self.ego_length   # 或从 ego_state 中取
                            l_lead = o_l

                            delta_s = dist_ego - dist_other - 0.5*l_ego - 0.5*l_lead
                            delta_s = max(delta_s, 1e-3)
                            if dist_ego>0 and dist_other>0 and if_disc == True:
                                _2conf_discount = min((detect_range - dist_ego) / (detect_range) * _disc_k, 1.0)
                                _2conf_discount = max(_2conf_discount, 0.0)
                            # delta_s = max(1e-7, dist_ego - dist_other)  # 与前车的距离，防除零

                        # TODO
                        elif (dist_other < dist_ego + safe_dist_2_rear) and (v_ego <= v_other) and self.changing == False: # 虽然AV位于前车但是速度小于other时，保险起见也减速避让
                            delta_s = max(1e-7, 0)  # 与前车的距离，防除零
                            if dist_ego>0 and dist_other>0 and if_disc == True:
                                _2conf_discount =  min((detect_range - dist_ego) / (detect_range) * _disc_k , 1.0)
                                _2conf_discount = max(_2conf_discount, 0.0)

                        
                        else:
                            continue
                        # effective desired gap s_star
                        # sqrt_term = 2.0 * math.sqrt(max(1e-12, a_idm_param * b_idm_param))
                        # # s_star =  (s0 + o_l/2 - 5/2) + v_ego * T + (v_ego * delta_v) / sqrt_term
                        # s_star = (s0 + k * v_other) + v_ego * T + (v_ego * delta_v) / sqrt_term

                        # # IDM 加速度
                        # acc_conf = a_idm_param * (1.0 - (v_ego / v0) ** delta - (s_star / delta_s) ** 2)
                        # # ==========================================================
                        # # 【新增】解除并行时的 IDM 封印
                        # # ==========================================================
                        # 1. 判断是否处于“并行/近乎并行”状态（到达冲突点的距离差小于 8.0 米）
                        is_parallel = abs(dist_ego - dist_other) < 8.0 
                        
                        # 2. 从信念字典中获取该交互车辆的“保守(Yield)”意图概率
                        p_yield = 0.0
                        if vid in self.belief_dict:
                            p_yield = self.belief_dict[vid][0]  # 索引 0 对应 P(Yield)
                            
                        # 3. 如果处于并行且对方大概率会让行，则解除封印
                        if is_parallel and p_yield > 0.5:
                            # 骗过 IDM：人为将安全间距拉大，假装前方很空旷
                            delta_s = max(delta_s, 20.0) 
                            
                            if self.roslog:
                                rospy.loginfo(f"[GAME模块] 检测到与车辆 {vid} 并行且对方倾向保守(P_Yield={p_yield:.2f})，解除IDM减速封印！")
                            else:
                                print(f"[GAME模块] 检测到与车辆 {vid} 并行且对方倾向保守(P_Yield={p_yield:.2f})，解除IDM减速封印！")
                        # ==========================================================

                        # effective desired gap s_star
                        sqrt_term = 2.0 * math.sqrt(max(1e-12, a_idm_param * b_idm_param))
                        s_star = (s0 + k * v_other) + v_ego * T + (v_ego * delta_v) / sqrt_term

                        # IDM 加速度
                        acc_conf = a_idm_param * (1.0 - (v_ego / v0) ** delta - (s_star / delta_s) ** 2)                        
                        
                        
                        # 强制停车逻辑
                        if delta_s <= s0 + 1e-3 and v_ego < 0.5:  # 距离接近s0且速度很低
                            acc_conf = -v_ego / dt 
                        
                        # 裁剪到 AV 的物理加速度限制
                        acc_conf = float(np.clip(acc_conf, min_a, max_a))
                        
                        # 1219修复：冲突车缓慢减速
                        if acc_conf < 0:
                            acc_conf = acc_conf * _2conf_discount
                        
                        if acc_conf < acc_conf_min:
                            acc_conf_min = acc_conf
                            conf_idx = other_ids.index(vid)
                            conf_id = vid
                            conf_state = other_states[conf_idx]
                            conf_s = dist_ego
        
        if acc_conf_min < acc:
            acc = acc_conf_min
            best_idx = conf_idx
            lead_id = conf_id
            best_state = conf_state
            best_s = conf_s
        
        if best_state is not None:
            if best_state[3] > v_ego and acc < 0:
                acc = acc * 0.2

        if is_other=='AV':
            if self.roslog:
                rospy.loginfo(f"[GAME模块] Leader id:{lead_id}, Leads{best_s}, IDM加速度: {acc:.2f} m/s²")
            else:
                print(f"[GAME模块] Leader id:{lead_id}, Leads{best_s}, IDM加速度: {acc:.2f} m/s²")
        return {
            'lead_index': best_idx,
            'lead_id': lead_id,
            'lead_state': best_state,
            'lead_s': best_s,
            'a_idm': acc
        }
    
    # 收益相关
    def __reward_TTC(self, s1, confpos1, v1, s2, confpos2, v2):
        """
        计算两车在冲突点的 TTC（Time To Collision）收益
        支持符号计算 (CasADi MX/SX)
        双向考虑 delta_x/delta_v 同号的情况
        """
        TTC_max = float(self.TTC_max)
        min_denom = 1e-3  # 分母下限，防除零
        TTC_clip = TTC_max * 10.0
        
        # 符号条件：冲突点不存在，直接返回 TTC_max
        cond_no_conflict = ca.logic_or(confpos1 == -1.0, confpos2 == -1.0)
        
        # delta 计算
        delta_v = v1 - v2
        delta_x = (confpos1 - s1) - (confpos2 - s2)
        
        # 数值安全分母
        denom_sign = ca.sign(delta_v) + 1e-12
        denom_abs = ca.fmax(ca.fabs(delta_v), min_denom)
        denom = denom_sign * denom_abs
        
        # TTC 候选 TODO减去车长
        ttc_candidate = delta_x / denom - 4.0/ ca.fabs(denom_sign)
        
        # 条件：delta_x/delta_v 同号则有效
        cond_A = ca.logic_and(delta_x > 0.0, delta_v > 0.0)
        cond_B = ca.logic_and(delta_x < 0.0, delta_v < 0.0)
        cond_valid = ca.logic_or(cond_A, cond_B)
        
        # Clamp 值
        ttc_safe = ca.fmax(ca.fmin(ttc_candidate, TTC_clip), -TTC_clip)
        
        # 符号条件选择：如果没有冲突点 -> TTC_max
        ttc_branch = ca.if_else(cond_no_conflict, TTC_max,
                                ca.if_else(cond_valid, ttc_safe, TTC_max))
        
        # 最终平滑裁剪到 [0, TTC_max]
        ttc = self._smooth_clip(ttc_branch, 0.0, TTC_max, alpha=20.0)
        
        return ttc
    
    def __reward_delta_TTCP(self, s1, confpos1, v1, s2, confpos2, v2):
        """平滑化 TTCP 计算"""
        min_v = 1e-3
        k_cond = 50.0  # 平滑度
        
        t1 = (confpos1 - s1) / ca.fmax(ca.fabs(v1), min_v) * ca.sign(v1 + 1e-12)
        t2 = (confpos2 - s2) / ca.fmax(ca.fabs(v2), min_v) * ca.sign(v2 + 1e-12)
        ttcp_raw = ca.fabs(t1 - t2)
        
        # 平滑裁剪
        ttcp = self._smooth_clip(ttcp_raw, 0.0, float(self.TTCP_max), alpha=20.0)
        
        # 平滑冲突判断：confpos1/2 == -1 时
        # Positive conflict coordinates indicate an actual upcoming conflict.
        # A sigmoid of their product incorrectly labelled almost every such
        # interaction as conflict-free, saturating the safety reward.
        cond_no_conflict = ca.logic_or(confpos1 < 0., confpos2 < 0.)
        ttcp_smooth = ca.if_else(cond_no_conflict, float(self.TTCP_max), ttcp)
        return ttcp_smooth
    
    def __reward_safety(self, s1, confpos1, v1, s2, confpos2, v2, margin=1.0):
        ttc = self.__reward_TTC(s1, confpos1, v1, s2, confpos2, v2)
        ttcp = self.__reward_delta_TTCP(s1, confpos1, v1, s2, confpos2, v2)
        # smooth max via -smooth_min(-a,-b)
        # 计算物理距离差
        # dist_diff = (confpos1 - s1) - (confpos2 - s2)
        
        # # 核心：空间侵入惩罚。如果对方离冲突点更近且距离我小于 10 米
        # # 这种惩罚对识别出的意图概率非常敏感
        # spatial_penalty = ca.if_else(
        #     ca.logic_and(dist_diff > 0, dist_diff < 10.0 * margin),
        #     -20.0 * ca.exp(-0.5 * dist_diff / margin), # 距离越近，扣分呈指数级爆炸
        #     0.0
        # )
        return - self._smooth_min(-ttc, -ttcp, alpha=20.0)
    
    def __reward_eff(self, v):
        return v
    
    def __reward_comfort(self, a_old, a_new, dt):
        # 确保 dt 是 CasADi 符号或数值，不要强制 float()
        if not isinstance(dt, ca.MX):
            dt = ca.MX(dt)
        # 将 a_old、a_new 统一为 MX 类型（即便是 parameter）
        a_old = ca.MX(a_old)
        a_new = ca.MX(a_new)
        diff = a_new - a_old
        comfort = -ca.sqrt(diff**2 + 1e-4)
        return comfort
    
    def __reward_total_symbolic(self, Asv_t, A_t_1, dt, t_idx, player_idx, a_prev_k, s_prev_k, v_prev_k, intent_probs, players_types, players_ids, av_confpos, hv_confpos_2av):
        A_t, s_t, v_t = Asv_t
        ego_s = s_t
        ego_v = v_t
        ego_type = sum(ca.if_else(player_idx == i, players_types[i], 0.0) for i in range(players_types.size1()))
        ego_id = sum(ca.if_else(player_idx == i, players_ids[i], 0.0) for i in range(players_ids.size1()))
        
        safe_reward = max(self.TTC_max, self.TTCP_max)
        n_players = players_ids.size1() 
        
        for i in range(n_players):
            is_current_player = ca.if_else(i == player_idx, 1, 0.0)
            other_type = sum(ca.if_else(i == j, players_types[j], 0.0) for j in range(players_types.size1()))
            is_ego_av = ca.if_else(ego_type == self.type_mapping['AV'], 1, 0.0)
            is_other_av = ca.if_else(other_type == self.type_mapping['AV'], 1, 0.0)
            
            # 【贝叶斯核心】计算主车对第 i 辆交互车的 "期望安全收益"
            expected_safe_reward_i = 0.0
            
            for k in range(self.K_intents):
                # 提取意图 k 下，他车在 t_idx 时刻的预测状态
                other_s_k = sum(ca.if_else(i == j, s_prev_k[k][j, t_idx], 0.0) for j in range(n_players))
                other_v_k = sum(ca.if_else(i == j, v_prev_k[k][j, t_idx], 0.0) for j in range(n_players))
                
                # 计算该意图分支下的物理碰撞惩罚
                safe_reward_1_k = ca.if_else(
                    is_ego_av,
                    self.__reward_safety(s1=ego_s, confpos1=av_confpos[i], v1=ego_v, s2=other_s_k, confpos2=hv_confpos_2av[i], v2=other_v_k),
                    self.__reward_safety(s1=ego_s, confpos1=sum(ca.if_else(player_idx== k1, hv_confpos_2av[k1], 0) for k1 in range(n_players)), v1=ego_v, s2=other_s_k, confpos2=sum(ca.if_else(player_idx== k2, av_confpos[k2], 0) for k2 in range(n_players)), v2=other_v_k)
                )
                
                # 获取该车发生该意图的概率并累加期望
                prob_k = intent_probs[i, k]
                expected_safe_reward_i += prob_k * safe_reward_1_k
            
            # 各个交互车辆的 "期望收益" 取最小 (最危险原则)
            safe_reward = ca.if_else(
                ca.logic_and(ca.logic_not(is_current_player), ca.logic_or(is_ego_av, is_other_av)),
                ca.fmin(safe_reward, expected_safe_reward_i),
                safe_reward,
            )
        
        eff_reward = self.__reward_eff(ego_v)
        comf_reward = self.__reward_comfort(A_t_1, A_t, dt)
        
        weight_0 = sum(ca.if_else(players_ids[j] == ego_id, self.weight0[j], 0.0) for j in range(n_players))
        weight_1 = sum(ca.if_else(players_ids[j] == ego_id, self.weight1[j], 0.0) for j in range(n_players))
        weight_2 = sum(ca.if_else(players_ids[j] == ego_id, self.weight2[j], 0.0) for j in range(n_players))
        
        return weight_0 * safe_reward + weight_1 * eff_reward + weight_2 * comf_reward
    
    def _evaluate_reward_symbolic(self, Asv, p):
        player_idx, a0, a_prev_k, s_prev_k, v_prev_k, intent_probs, players_types, players_ids, av_confpos, hv_confpos_2av = p
        sum_t_reward = 0.0
        A, s, v = Asv
        n_time = self.n_time
        
        for t_idx in range(n_time):
            t = self.t_list[t_idx]
            dt = self.t_list[t_idx] - self.t_list[t_idx - 1] if t_idx > 0 else self.t_list[t_idx]
            timestep_weight = ca.exp(-self.decay_rate * t)
            
            A_t   = A[t_idx]
            s_t   = s[t_idx]
            v_t   = v[t_idx]
            A_t_1 = a0 if t_idx == 0 else A[t_idx - 1]
            Asv_t = (A_t, s_t, v_t)
            
            # 【直接透传全量意图矩阵 s_prev_k，在底层函数中索引】
            reward_t = self.__reward_total_symbolic(Asv_t, A_t_1, dt, t_idx, player_idx, a_prev_k, s_prev_k, v_prev_k, intent_probs, players_types, players_ids, av_confpos, hv_confpos_2av)
            sum_t_reward += reward_t * timestep_weight
            
        J = -1 * sum_t_reward
        return J
    
    # def __reward_total_symbolic(self, Asv_t, A_t_1, dt, t_idx, player_idx, a_prev, s_prev_t, v_prev_t, players_types, players_ids, av_confpos, hv_confpos_2av):
    #     A_t, s_t, v_t = Asv_t
    #     ego_s = s_t
    #     ego_v = v_t
    #     # 修复 ego_type 和 ego_id 的动态索引
    #     ego_type = sum(ca.if_else(player_idx == i, players_types[i], 0.0) for i in range(players_types.size1()))
    #     ego_id = sum(ca.if_else(player_idx == i, players_ids[i], 0.0) for i in range(players_ids.size1()))
    #     # 1. 安全收益计算
    #     safe_reward = max(self.TTC_max, self.TTCP_max)
    #     n_players = players_ids.size1()  # 获取玩家数量
    #     for i in range(n_players):
    #         # 修复布尔表达式
    #         is_current_player = ca.if_else(i == player_idx, 1, 0.0)
    #         # 修复 other_id 和 other_type 的动态索引
    #         other_id = sum(ca.if_else(i == j, players_ids[j], 0.0) for j in range(players_ids.size1()))
    #         other_type = sum(ca.if_else(i == j, players_types[j], 0.0) for j in range(players_types.size1()))
    #         # 修复 s_prev_t 和 v_prev_t 的动态索引
    #         other_s = sum(ca.if_else(i == j, s_prev_t[j], 0.0) for j in range(s_prev_t.size1()))
    #         other_v = sum(ca.if_else(i == j, v_prev_t[j], 0.0) for j in range(v_prev_t.size1()))
    #         # 判断是否为 AV
    #         is_ego_av = ca.if_else(ego_type == self.type_mapping['AV'], 1, 0.0)
    #         is_other_av = ca.if_else(other_type == self.type_mapping['AV'], 1, 0.0)
            
    #         # 计算安全收益
    #         safe_reward_1 = ca.if_else(
    #             is_ego_av,
    #             self.__reward_safety(
    #                 s1=ego_s,
    #                 confpos1=av_confpos[i],
    #                 v1=ego_v,
    #                 s2=other_s,
    #                 confpos2=hv_confpos_2av[i],
    #                 v2=other_v
    #             ),
    #             self.__reward_safety(
    #                 s1=ego_s,
    #                 confpos1=sum(ca.if_else(player_idx== k1, hv_confpos_2av[k1], 0) for k1 in range(n_players)),
    #                 v1=ego_v,
    #                 s2=other_s,
    #                 confpos2=sum(ca.if_else(player_idx== k2, av_confpos[k2], 0) for k2 in range(n_players)),
    #                 v2=other_v
    #             )
    #         )
    #         safe_reward = ca.if_else(
    #             ca.logic_and(ca.logic_not(is_current_player), ca.logic_or(is_ego_av, is_other_av)),
    #             ca.fmin(safe_reward, safe_reward_1),
    #             safe_reward,
    #         )
        
    #     # 2. 效率收益计算
    #     eff_reward = self.__reward_eff(ego_v)
    #     # 3. 舒适度收益计算
    #     comf_reward = self.__reward_comfort(A_t_1, A_t, dt)
    #     # 修复 weight_dict 的动态访问
    #     weight_0 = sum(ca.if_else(players_ids[j] == ego_id, self.weight0[j], 0.0) for j in range(players_ids.size1()))
    #     weight_1 = sum(ca.if_else(players_ids[j] == ego_id, self.weight1[j], 0.0) for j in range(players_ids.size1()))
    #     weight_2 = sum(ca.if_else(players_ids[j] == ego_id, self.weight2[j], 0.0) for j in range(players_ids.size1()))
    #     # 计算总奖励
    #     total_reward = weight_0 * safe_reward + weight_1 * eff_reward + weight_2 * comf_reward
        
    #     return total_reward
    
    # def _evaluate_reward_symbolic(self, Asv, p):
    #     player_idx, a0, a_prev, s_prev, v_prev, players_types, players_ids, av_confpos, hv_confpos_2av = p
    #     sum_t_reward = 0.0
    #     A, s, v = Asv
    #     n_player = players_types.size1()  # 获取玩家数量
    #     n_time = self.n_time
    #     for t_idx in range(n_time):
    #         t = self.t_list[t_idx]
    #         dt = self.t_list[t_idx] - self.t_list[t_idx - 1] if t_idx > 0 else self.t_list[t_idx]
    #         timestep_weight = ca.exp(-self.decay_rate * t)  # 使用 CasADi 的 exp
    #         # 修复 s_prev 和 v_prev 的切片
    #         s_prev_t = ca.vertcat(*[s_prev[i, t_idx] for i in range(s_prev.size1())])
    #         v_prev_t = ca.vertcat(*[v_prev[i, t_idx] for i in range(v_prev.size1())])
    #         A_t   = A[t_idx]
    #         s_t   = s[t_idx]
    #         v_t   = v[t_idx]
    #         if t_idx == 0:
    #             A_t_1 = a0
    #         else:
    #             A_t_1 = A[t_idx - 1]
    #         Asv_t = (A_t, s_t, v_t)
    #         reward_t = self.__reward_total_symbolic(Asv_t, A_t_1, dt, t_idx, player_idx, a_prev, s_prev_t, v_prev_t, players_types, players_ids, av_confpos, hv_confpos_2av)
    #         sum_t_reward += reward_t * timestep_weight
    #     J = -1 * sum_t_reward
    #     return J
    
    # 风格识别所需子项收益
    def _get_sub_reward_hv(self, a_all, a0_list, v0_list, players_types, players_ids, av_confpos, hv_confpos_2av):
        style_reward_dict = {}
        av_a = a_all[0]
        av_s = [0.0]
        av_v = [v0_list[0]]
        
        n_time = self.n_time
        
        for t in range(n_time):
            dt = self.t_list[t] - self.t_list[t - 1] if t > 0 else self.t_list[t]
            s_next = av_s[-1] + av_v[-1] * dt + 0.5 * av_a[t] * dt**2
            v_next = av_v[-1] + av_a[t] * dt
            av_s.append(s_next)
            av_v.append(v_next)
        av_s = av_s[1:]
        av_v = av_v[1:]
        
        for i in range(len(players_types)):
            if i > 0: # TODO：需要判断需要更新的车辆类型
                A = a_all[i]
                s_list = [0.0]
                v_list = [v0_list[i]]
                for t in range(len(self.t_list)):
                    dt = self.t_list[t]-self.t_list[t-1] if t>0 else self.t_list[t]
                    s_next = s_list[-1] + v_list[-1]*dt + 0.5*A[t]*dt**2
                    v_next = v_list[-1] + A[t]*dt
                    s_list.append(s_next)
                    v_list.append(v_next)
                s = s_list[1:]
                v = v_list[1:]
                
                i_safe_r = 0.0
                i_eff_r = 0.0
                i_comf_r = 0.0
                
                for t_idx in range(n_time):
                    t = self.t_list[t_idx]
                    dt = self.t_list[t_idx] - self.t_list[t_idx - 1] if t_idx > 0 else self.t_list[t_idx]
                    timestep_weight = ca.exp(-self.decay_rate * t)  # 使用 CasADi 的 exp
                    A_t   = A[t_idx]
                    s_t   = s[t_idx]
                    v_t   = v[t_idx]
                    if t_idx == 0:
                        A_t_1 = a0_list[i]
                    else:
                        A_t_1 = A[t_idx - 1]
                    
                    # 计算风格识别所需子项收益
                    i_safe_r_t = max(self.TTC_max, self.TTCP_max)
                    for j in range(len(players_types)):
                        if j == 0: # 只计算与AV的收益
                            ttc = self.__reward_TTC(  s1 = s_t, 
                                                                confpos1 = hv_confpos_2av[i], 
                                                                v1 = v_t, 
                                                                s2 = av_s[t_idx], 
                                                                confpos2 = av_confpos[i], 
                                                                v2 = av_v[t_idx])
                            ttcp = self.__reward_delta_TTCP(  s1 = s_t, 
                                                                confpos1 = hv_confpos_2av[i], 
                                                                v1 = v_t, 
                                                                s2 = av_s[t_idx], 
                                                                confpos2 = av_confpos[i], 
                                                                v2 = av_v[t_idx])
                            i_safe_r_t = max(ttc, ttcp)
                    i_eff_r_t = self.__reward_eff(v_t)
                    i_comf_r_t = self.__reward_comfort(A_t_1, A_t, dt)
                    i_safe_r += i_safe_r_t * timestep_weight
                    i_eff_r += i_eff_r_t * timestep_weight
                    i_comf_r += i_comf_r_t * timestep_weight
                style_reward_dict[players_ids[i]] = (float(i_safe_r), float(i_eff_r), float(i_comf_r))
        return style_reward_dict
    
    # 离散动作计算
    def _cal_discrete_reward(self, i, a, a_prev, v_prev, a0_list, v0_list, players_types, players_ids, av_confpos_array, hv_confpos_2av_array):
        av_a = a_prev[0]
        av_s = [0.0]
        av_v = [v0_list[0]]
        
        n_time = self.n_time
        
        for t in range(n_time):
            dt = self.t_list[t] - self.t_list[t - 1] if t > 0 else self.t_list[t]
            active_dt = min(dt, av_v[-1] / -av_a[t]) if av_a[t] < 0 else dt
            s_next = av_s[-1] + av_v[-1] * active_dt + 0.5 * av_a[t] * active_dt**2
            v_next = max(0., av_v[-1] + av_a[t] * dt)
            av_s.append(s_next)
            av_v.append(v_next)
        av_s = av_s[1:]
        av_v = av_v[1:]
        
        A = a
        s_list = [0.0]
        v_list = [v0_list[i]]
        for t in range(len(self.t_list)):
            dt = self.t_list[t]-self.t_list[t-1] if t>0 else self.t_list[t]
            active_dt = min(dt, v_list[-1] / -A[t]) if A[t] < 0 else dt
            s_next = s_list[-1] + v_list[-1]*active_dt + 0.5*A[t]*active_dt**2
            v_next = max(0., v_list[-1] + A[t]*dt)
            s_list.append(s_next)
            v_list.append(v_next)
        s = s_list[1:]
        v = v_list[1:]
        
        i_safe_r = 0.0
        i_eff_r = 0.0
        i_comf_r = 0.0
        
        for t_idx in range(n_time):
            t = self.t_list[t_idx]
            dt = self.t_list[t_idx] - self.t_list[t_idx - 1] if t_idx > 0 else self.t_list[t_idx]
            timestep_weight = ca.exp(-self.decay_rate * t)  # 使用 CasADi 的 exp
            A_t   = A[t_idx]
            s_t   = s[t_idx]
            v_t   = v[t_idx]
            if t_idx == 0:
                A_t_1 = a0_list[i]
            else:
                A_t_1 = A[t_idx - 1]
            
            # 计算风格识别所需子项收益
            i_safe_r_t = max(self.TTC_max, self.TTCP_max)
            for j in range(len(players_types)):
                if j == 0: # 只计算与AV的收益
                    ttc = self.__reward_TTC(  s1 = s_t, 
                                                        confpos1 = hv_confpos_2av_array[i], 
                                                        v1 = v_t, 
                                                        s2 = av_s[t_idx], 
                                                        confpos2 = av_confpos_array[i], 
                                                        v2 = av_v[t_idx])
                    ttcp = self.__reward_delta_TTCP(  s1 = s_t, 
                                                        confpos1 = hv_confpos_2av_array[i], 
                                                        v1 = v_t, 
                                                        s2 = av_s[t_idx], 
                                                        confpos2 = av_confpos_array[i], 
                                                        v2 = av_v[t_idx])
                    i_safe_r_t = max(ttc, ttcp)
            i_eff_r_t = self.__reward_eff(v_t)
            i_comf_r_t = self.__reward_comfort(A_t_1, A_t, dt)
            
            i_safe_r += i_safe_r_t * timestep_weight
            i_eff_r += i_eff_r_t * timestep_weight
            i_comf_r += i_comf_r_t * timestep_weight
            
            sum_reward = i_safe_r *self.weight_dict[players_ids[i]][0] + \
                        i_eff_r*self.weight_dict[players_ids[i]][1] + \
                        i_comf_r*self.weight_dict[players_ids[i]][2]
        return -1 *sum_reward
    
    # 迭代博弈
    def _solve_nash_equilibrium_symbolic_discrete(self, players_states, players_ids, players_types, players_lengths, av_confpos, hv_confpos_2av, ego_path, ego_path_s, hvs_path, hvs_s, tol=1e-3):
        n_players = len(players_ids)
        N = len(self.t_list)
        a_all = [np.zeros(N) for _ in range(n_players)]
        a_all_k = [[np.zeros(N) for _ in range(self.K_intents)] for _ in range(n_players)]
        AV_reward = 0
        a0_list = [players_states[i][4] for i in range(n_players)]
        v0_list = [players_states[i][3] for i in range(n_players)]
        for it in range(self.max_iters):
            # # 更新player的意愿
            # a_prev = []
            # s_prev = []
            # v_prev = []
            # for i in range(n_players):
            #     a = a_all[i].copy()
            #     s = [0.0]
            #     v = [players_states[i][3]]
            #     v_max_i = max(self.limits[self.type_mapping[players_types[i]]]['MAX_SPEED'], v[0])
            #     for t in range(N):
            #         dt = self.t_list[t] - self.t_list[t - 1] if t > 0 else self.t_list[t]
            #         s_next = s[-1] + v[-1] * dt + 0.5 * a[t] * dt**2
            #         v_next = v[-1] + a[t] * dt
            #         v_next = np.clip(v_next, 0, v_max_i)
            #         s.append(s_next)
            #         v.append(v_next)
            #     s = s[1:]
            #     v = v[1:]
            #     a_prev.append(a)
            #     s_prev.append(s)
            #     v_prev.append(v)
            # a_prev = np.asarray(a_prev)
            # s_prev = np.asarray(s_prev) 
            # v_prev = np.asarray(v_prev)

            # 【贝叶斯改造】推导所有玩家在 K 种意图下的运动状态
            av_action_prev_iter = a_all[0].copy()
            a_prev_k_iter = []
            s_prev_k_iter = []
            v_prev_k_iter = []
            for k in range(self.K_intents):
                a_prev_temp = []
                s_prev_temp = []
                v_prev_temp = []
                for i in range(n_players):
                    a = a_all_k[i][k].copy()  # 取第 i 辆车在第 k 种意图下的动作
                    s = [0.0]
                    v = [players_states[i][3]]
                    v_max_i = max(self.limits[self.type_mapping[players_types[i]]]['MAX_SPEED'], v[0])
                    for t in range(N):
                        dt = self.t_list[t] - self.t_list[t - 1] if t > 0 else self.t_list[t]
                        s_next = s[-1] + v[-1] * dt + 0.5 * a[t] * dt**2
                        v_next = v[-1] + a[t] * dt
                        v_next = np.clip(v_next, 0, v_max_i)
                        s.append(s_next)
                        v.append(v_next)
                    s_prev_temp.append(s[1:])
                    v_prev_temp.append(v[1:])
                    a_prev_temp.append(a)
                a_prev_k_iter.append(np.asarray(a_prev_temp))
                s_prev_k_iter.append(np.asarray(s_prev_temp))
                v_prev_k_iter.append(np.asarray(v_prev_temp))

            # 将字典转换为数组
            av_confpos_array = np.array([av_confpos[player_id] for player_id in players_ids])
            hv_confpos_2av_array = np.array([hv_confpos_2av[player_id] for player_id in players_ids])
            
            # 新的迭代轮次博弈求解
            player_order = list(range(1, n_players)) + [0] if self.decision_config else list(range(n_players))
            for i in player_order:
                t_i = time.time()
                if players_types[i] == 'obs':
                    continue
                a0 = players_states[i][4]
                # 将 players_types 和 players_ids 转换为数值表示
                players_types_numeric = np.array([self.type_mapping[player_type] for player_type in players_types])
                players_ids_numeric = np.array([
                                        self.id_mapping[players_id] if isinstance(players_id, str) and players_id in self.id_mapping else int(players_id)
                                        for players_id in players_ids
                                    ])
                
                a_min = -v0_list[i]/self.dt
                a_max = (self.limits[players_types_numeric[i]]['MAX_SPEED'] - v0_list[i]) / self.dt
                a_from_speedlimt = (self.limits[players_types_numeric[i]]['MAX_SPEED'] - v0_list[i]) / self.dt
                a_min = np.clip(a_min, self.limits[players_types_numeric[i]]['MAX_DECELERATION'], self.limits[players_types_numeric[i]]['MAX_ACCELERATION'])
                a_max = np.clip(a_max, self.limits[players_types_numeric[i]]['MAX_DECELERATION'], self.limits[players_types_numeric[i]]['MAX_ACCELERATION'])
                if i == 0:
                    # Refresh the type-conditioned responses BEFORE the AV solves.
                    if self.decision_config:
                        for k in range(self.K_intents):
                            for j in range(1, n_players):
                                a = np.asarray(a_all_k[j][k], dtype=float)
                                vv, ss = v0_list[j], 0.0
                                for t in range(N):
                                    delta_t = self.t_list[t] - (self.t_list[t-1] if t else 0.)
                                    # Integrate a stopped vehicle without reversing it.
                                    active_t = min(delta_t, vv / -a[t]) if a[t] < 0 else delta_t
                                    ss += vv * active_t + 0.5 * a[t] * active_t**2
                                    vv = max(0., vv + a[t] * delta_t)
                                    s_prev_k_iter[k][j, t] = ss
                                    v_prev_k_iter[k][j, t] = vv
                                a_prev_k_iter[k][j] = a
                    # 调用对应数量的求解器
                    solver = self.SOLVER[n_players]
                    
                    for name, var in solver.items():
                        setattr(self, name, var)
                    
                    # 动态参数赋值
                    self.opti.set_value(self.player_idx, i)
                    self.opti.set_value(self.a0, a0)
                    self.opti.set_value(self.initial_speed, v0_list[i])
                    # self.opti.set_value(self.a_prev, a_prev)
                    # self.opti.set_value(self.s_prev, s_prev)
                    # self.opti.set_value(self.v_prev, v_prev)
                    # 【贝叶斯兼容补丁】将当前的单一轨迹直接复制 K 份传入
                    for k in range(self.K_intents):
                        self.opti.set_value(self.a_prev_k[k], a_prev_k_iter[k])
                        self.opti.set_value(self.s_prev_k[k], s_prev_k_iter[k])
                        self.opti.set_value(self.v_prev_k[k], v_prev_k_iter[k])
                        
                    # 将外部维护的意图信念字典转为矩阵传入求解器
                    intent_probs_matrix = np.zeros((n_players, self.K_intents))
                    for j, pid in enumerate(players_ids_numeric):
                        if pid in self.belief_dict:
                            intent_probs_matrix[j] = self.belief_dict[pid]
                        else:
                            intent_probs_matrix[j] = [0.0, 1.0, 0.0] # 默认全部视为 Normal

                    self.opti.set_value(self.intent_probs, intent_probs_matrix)
                    self.opti.set_value(self.players_types, players_types_numeric)
                    self.opti.set_value(self.players_ids, players_ids_numeric)
                    self.opti.set_value(self.av_confpos, av_confpos_array)
                    self.opti.set_value(self.hv_confpos_2av, hv_confpos_2av_array)
                    
                    weight0_arr = np.zeros(n_players, dtype=float)
                    weight1_arr = np.zeros(n_players, dtype=float)
                    weight2_arr = np.zeros(n_players, dtype=float)
                    for j, pid in enumerate(players_ids_numeric):
                        if pid in self.weight_dict:
                            w0, w1, w2 = self.weight_dict[pid]
                        else:
                            w0, w1, w2 = self.other_init_weight
                        weight0_arr[j] = w0
                        weight1_arr[j] = w1
                        weight2_arr[j] = w2
                    self.opti.set_value(self.weight0, weight0_arr)
                    self.opti.set_value(self.weight1, weight1_arr)
                    self.opti.set_value(self.weight2, weight2_arr)
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]  av_confpos: {av_confpos}")
                        rospy.loginfo(f"[GAME模块]  hv_confpos_2av: {hv_confpos_2av}")
                    else:
                        print(f"   >>>> av_confpos: {av_confpos}")
                        print(f"   >>>> hv_confpos_2av: {hv_confpos_2av}")
                    a_follow_detect = self._find_lead_and_idm( ego_state=players_states[i],
                                                    ego_path=ego_path,
                                                    path_s=ego_path_s,
                                                    other_states=[players_states[j] for j in range(n_players) if j != i],
                                                    other_lengths=[players_lengths[j] for j in range(n_players) if j != i],
                                                    other_ids=[players_ids[j] for j in range(n_players) if j != i],
                                                    av_confpos = av_confpos,
                                                    hv_confpos_2av = hv_confpos_2av,
                                                    max_lateral_dist = self.max_lateral_dist
                                                    )
                    a_follow = a_follow_detect['a_idm']
                    real_follow = a_follow
                    
                    # a_follow = np.clip(a_follow, a_min, a_max)
                    a_max = min(a_max, a_follow)
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]     >>>> Player {i} acceleration limits: [{a_min:.2f}, {a_max:.2f}] m/s², a_idm: {a_follow:.2f} m/s²")
                    else:
                        print(f"   >>>> Player {i} acceleration limits: [{a_min:.2f}, {a_max:.2f}] m/s², a_idm: {a_follow:.2f} m/s²")
                    v_max = max(self.limits[players_types_numeric[i]]['MAX_SPEED'], v0_list[i])
                    a_max = min(a_max, (v_max-v0_list[i]) / self.t_list[-1])
                    bounds_infeasible = a_max < a_min
                    # Physical bounds take precedence when the IDM bound is infeasible.
                    idm_upper = float(a_max)
                    a_max = max(a_min, a_max)
                    self.opti.set_value(self.a_max, a_max)
                    self.opti.set_value(self.a_min, a_min)
                    self.opti.set_value(self.v_max, v_max)
                    
                    time11 = time.time()
                    try:
                        if self.decision_config.get('av_optimizer') == 'scalar_grid':
                            # SHARED_ACC makes this an exactly one-dimensional problem.
                            # Evaluate the same CasADi objective; deterministic grid plus
                            # bounded refinements avoids expensive NLP derivative setup.
                            parameters = self.opti.value(self.parameter_vector)
                            def objective(value):
                                return float(self.cost_function([value], parameters))
                            grid = np.linspace(a_min, a_max, 17)
                            values = [objective(value) for value in grid]
                            options = [(values[j], float(grid[j])) for j in range(len(grid))]
                            for j in range(1, len(grid)-1):
                                if values[j] <= values[j-1] and values[j] <= values[j+1]:
                                    refined = minimize_scalar(objective, bounds=(grid[j-1],grid[j+1]),
                                                              method='bounded', options={'xatol':1e-5, 'maxiter':40})
                                    options.append((float(refined.fun), float(refined.x)))
                            _, optimum = min(options)
                            raw_action = np.full(N, optimum)
                            solve_status = 'scalar_grid_refined'
                        else:
                            self.opti.set_initial(self.AA, a_max)
                            sol = self.opti.solve()
                            raw_action = np.asarray(sol.value(self.A), dtype=float).reshape(-1)
                            solve_status = self.opti.stats()['return_status']
                        if not np.isfinite(raw_action).all():
                            raise RuntimeError('Nonfinite optimizer action')
                        a_all[i] = np.clip(raw_action, a_min, a_max)
                    except RuntimeError as e:
                        if self.roslog:
                            rospy.logwarn(f"求解超时或失败: {e}")
                        else:
                            print(f"求解超时或失败: {e}")
                        # 部分解，直接用 debug.value 获取数值
                        # a_all[i] = self.opti.debug.value(self.A)
                        a_all[i] = np.array([a_max for _ in range(N)])
                        raw_action = a_all[i].copy()
                        solve_status = 'fallback:' + str(e).splitlines()[-1]
                    # 打印求解状态
                    reward = float(self.cost_function(np.asarray(a_all[i][:1]), self.opti.value(self.parameter_vector)))
                    self.solve_diagnostics.append({
                        'iteration': it, 'status': solve_status,
                        'a_raw': raw_action.tolist(), 'a_output': a_all[i].tolist(),
                        'a_min': float(a_min), 'a_max': float(a_max), 'idm_upper': idm_upper,
                        'infeasible_bounds': bool(bounds_infeasible),
                        'objective': reward,
                        'solve_ms': (time.time()-time11)*1000.,
                        'response_accelerations': [[list(map(float, aa)) for aa in row] for row in a_all_k],
                    })
                    if a_max < real_follow and a_from_speedlimt <= a_max and a_from_speedlimt <= 0:
                        self.decel_from_speedlimit = True

                    AV_reward = reward
                    # 更新AV的动作，作为HV的先验
                    if self.AV_LEADER:
                        a = a_all[0].copy()
                        s = [0.0]
                        v = [players_states[0][3]]
                        a_max_av = max(self.limits[self.type_mapping['AV']]['MAX_SPEED'], v[0])
                        for t in range(N):
                            dt = self.t_list[t] - self.t_list[t - 1] if t > 0 else self.t_list[t]
                            s_next = s[-1] + v[-1] * dt + 0.5 * a[t] * dt**2
                            v_next = v[-1] + a[t] * dt
                            v_next = np.clip(v_next, 0, a_max_av)
                            s.append(s_next)
                            v.append(v_next)
                        s = s[1:]
                        v = v[1:]
                        # 主车的动作是确定的，将其同步到所有 k 分支，供他车预测时作为背景车
                        for k in range(self.K_intents):
                            a_prev_k_iter[k][0] = a
                            s_prev_k_iter[k][0] = s
                            v_prev_k_iter[k][0] = v
                            a_all_k[0][k] = a.copy()
                        a_all[0] = a.copy()
                
                elif i != 0:
                    if self.decision_config.get('response_mode') == 'constant_velocity':
                        a_all[i] = np.zeros(N)
                        continue
                    # # 使用 arange 并确保包含 a_max（避免浮点精度问题）
                    # a_follow_detect = self._find_lead_and_idm( ego_state=players_states[i],
                    #                                 ego_path=hvs_path[i-1],
                    #                                 path_s=hvs_s[i-1],
                    #                                 other_states=[players_states[j] for j in range(n_players) if j != i],
                    #                                 other_lengths=[players_lengths[j] for j in range(n_players) if j != i],
                    #                                 other_ids=[players_ids[j] for j in range(n_players) if j != i],
                    #                                 av_confpos = av_confpos,
                    #                                 hv_confpos_2av = hv_confpos_2av,
                    #                                 max_lateral_dist = self.max_lateral_dist,
                    #                                 is_other = players_ids[i], 
                    #                                 )
                    # a_follow = a_follow_detect['a_idm'] * self.hv_idm_discount if a_follow_detect['a_idm'] < 0 else a_follow_detect['a_idm']
                    # a_max = min(a_max, a_follow)

                    # discrete_actions_1 = list(np.arange(a_min, a_max, self.other_discrete_precision, dtype=float)) if a_max - a_min > self.other_discrete_precision else [a_min]
                    # discrete_actions_1.append(a_max)
                    # discrete_actions = [[da for _ in range(N)] for da in discrete_actions_1]
                    # reward_list = []
                    # for a_idx, a in enumerate(discrete_actions):
                    #     reward = self._cal_discrete_reward(i, a, a_prev, v_prev, a0_list, v0_list, players_types, players_ids, av_confpos_array, hv_confpos_2av_array)
                    #     reward_list.append(reward)
                    # best_a_idx = np.argmin(reward_list)
                    # a_all[i] = discrete_actions[best_a_idx]
                    # print(players_ids[i], ">>>>> ",  a_all[i][0])

                    # ==========================================================
                    # 【贝叶斯核心】他车 (HV) 在 K 种意图下的独立推演
                    # ==========================================================
                    for k, intent_name in enumerate(self.intent_types):
                        intent_profile = self.intent_profiles[intent_name]
                        
                        # 1. 临时套上该意图参数
                        original_weight = self.weight_dict[players_ids_numeric[i]]
                        
                        
                        
                        idm_params_k = {
                            'T': intent_profile['idm_T'], 
                            's0': intent_profile['s0']
                        }
                        
                        # 2. IDM 计算出特有的加速度极限
                        a_follow_detect = self._find_lead_and_idm( ego_state=players_states[i],
                                                        ego_path=hvs_path[i-1],
                                                        path_s=hvs_s[i-1],
                                                        other_states=[players_states[j] for j in range(n_players) if j != i],
                                                        other_lengths=[players_lengths[j] for j in range(n_players) if j != i],
                                                        other_ids=[players_ids[j] for j in range(n_players) if j != i],
                                                        av_confpos = av_confpos,
                                                        hv_confpos_2av = hv_confpos_2av,
                                                        max_lateral_dist = self.max_lateral_dist,
                                                        is_other = players_ids[i], 
                                                        idm_params = idm_params_k 
                                                        )
                        a_follow = a_follow_detect['a_idm'] * self.hv_idm_discount if a_follow_detect['a_idm'] < 0 else a_follow_detect['a_idm']
                        a_max_k = max(a_min, min(a_max, a_follow))

                        # --- 【参数 1 注入】：yield_to_lc (保守让行硬约束) ---
                        is_av_changing = self.changing or (self.now_cal_path_idx != self.now_current_path_idx)
                        if is_av_changing and intent_profile.get('yield_to_lc', False):
                            # 保守型车辆在发现自车变道时，最大加速度被强行限制为不大于0（即必须减速或匀速让行）
                            a_max_k = max(a_min, min(a_max_k, 0.0))

                        # --- 【参数 3 注入】：Force 意图在自车换道时不接受跟车限速 ---
                        # 激进车辆的预测轨迹应体现"继续行驶"的敌意分支，而不是像
                        # 普通跟车一样对加塞自车减速。该开关只改变博弈内部对其他
                        # 车辆轨迹的预测，不改变自车的任何决策规则。
                        if is_av_changing and intent_name == 'Force' and self.force_ignore_idm_on_av_lc:
                            a_max_k = a_max

                        # 3. 生成动作搜索空间
                        discrete_actions_1 = list(np.arange(a_min, a_max_k, self.other_discrete_precision, dtype=float)) if a_max_k - a_min > self.other_discrete_precision else [a_min]
                        discrete_actions_1.append(a_max_k)
                        discrete_actions = [[da for _ in range(N)] for da in discrete_actions_1]
                        
                        # --- 【参数 2 注入】：lc_efficiency_decay (路怒/避让收益调整) ---
                        hv_numeric_id = players_ids_numeric[i]
                        original_weight = self.weight_dict[hv_numeric_id]
                        temp_weight = intent_profile['weights'].copy()
                        
                        if is_av_changing:
                            # 乘以效率衰减/激增系数（索引 1 代表效率分）
                            decay_factor = intent_profile.get('lc_efficiency_decay', 1.0)
                            temp_weight[1] = temp_weight[1] * decay_factor
                            
                        self.weight_dict[hv_numeric_id] = temp_weight
                        
                        # 4. 打分评估
                        reward_list = []
                        try:
                            for a_idx, a in enumerate(discrete_actions):
                                reward = self._cal_discrete_reward(i, a, a_prev_k_iter[k], v_prev_k_iter[k], a0_list, v0_list, players_types, players_ids, av_confpos_array, hv_confpos_2av_array)
                                reward_list.append(float(reward))
                        finally:
                            self.weight_dict[hv_numeric_id] = original_weight
                            
                        # 5. 选出该意图下的最佳应对策略
                        best_a_idx = np.argmin(reward_list)
                        a_all_k[i][k] = discrete_actions[best_a_idx]
                        
                        self.weight_dict[players_ids_numeric[i]] = original_weight

                    # ==========================================================
                    # 基于最大后验概率输出最具代表性的轨迹
                    # ==========================================================
                    # 获取该车当前的意图概率分布
                    current_probs = self.belief_dict.get(players_ids_numeric[i], [0.0, 1.0, 0.0])

                    # 找到概率最大（最可能发生）的意图索引
                    # 例如：[0.1, 0.2, 0.7] -> max_prob_idx = 2 (对应 Force)
                    max_prob_idx = np.argmax(current_probs)
                    
                    # 将最有可能发生的轨迹作为对外输出的唯一轨迹
                    a_all[i] = a_all_k[i][max_prob_idx].copy()
                # rospy.loginfo(f"[GAME模块]     >>>> Player {i} solve time: {(time.time() - t_i)*1000} ms")
            
            # 收敛判定
            max_diff_av = np.max(np.abs(a_all[0] - av_action_prev_iter))
            if max_diff_av < self.converged_eps and (it > 0 or not self.decision_config):
                break
                
        a_max = (self.limits[self.type_mapping['AV']]['MAX_SPEED'] - v0_list[0]) / self.dt
        a_from_speedlimt = (self.limits[self.type_mapping['AV']]['MAX_SPEED'] - v0_list[0]) / self.dt
        a_max = np.clip(a_max, self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], self.limits[self.type_mapping['AV']]['MAX_ACCELERATION'])
        v_max = max(self.limits[self.type_mapping['AV']]['MAX_SPEED'], v0_list[0])
        AV_reward_max = [
            (self.av_weights[0] * max(self.TTC_max, self.TTCP_max) + \
                        self.av_weights[1] * v0_list[0] + self.limits[self.type_mapping['AV']]['MAX_ACCELERATION'] * t+ \
                        self.av_weights[2] * 0.0) * np.exp(-self.decay_rate * t)
            for t in self.t_list[:self.n_time]
        ]
        
        # 计算风格识别所需子项收益
        style_reward_dict = self._get_sub_reward_hv(a_all, a0_list, v0_list, players_types, players_ids, av_confpos_array, hv_confpos_2av_array)
        if self.roslog:
            rospy.loginfo(f"[GAME模块]     >>>> 车辆当前速度: {players_states[0][3]:.2f} m/s")
            rospy.loginfo(f"[GAME模块]     >>>> 计算最优加速度: {a_all[0][0]:.2f} m/s")
            rospy.loginfo(f"[GAME模块]     >>>> AV_reward_max: {sum(AV_reward_max)}")
            rospy.loginfo(f"[GAME模块]     >>>> AV_reward: {AV_reward}")
        else:
            print(f"   >>>> 车辆当前速度: {players_states[0][3]:.2f} m/s")
            print(f"   >>>> 计算最优加速度: {a_all[0][0]:.2f} m/s")
            print(f"   >>>> AV_reward_max: {sum(AV_reward_max)}")
            print(f"   >>>> AV_reward: {AV_reward}")
        return a_all, AV_reward, AV_reward + sum(AV_reward_max), style_reward_dict, real_follow
    
    # === 定义单条轨迹求解函数 ===
    def solve_for_one_target(self, target_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths, lat_ratio=1.0):
        t1 = time.time()
        # ==================== 【修改点 2】 ====================
        # 新增：每进入一次单支轨迹评估，分支计数器 +1
        self.evaluated_branches_count += 1

        # 1️⃣ 获取目标轨迹
        ego_pose = ego_state[:4]  # x, y, heading
        
        if self.DFpath is not None:
            path = self.DFpath
            change_end_idx = 0
        else:
            if target_idx is None: # 当前车道不在目标车道 且其他车道都不可行时启用
                path, change_end_idx = self._get_target_path(curr_lane_idx, lanes, curr_lane_idx, ego_pose, lat_ratio=lat_ratio)
                # path = self.current_path
                # min_idx = None
                # min_dist, min_idx = nearest_point_on_traj(ego_state[:2], path)
                # if min_idx is not None:
                #     path = path[min_idx:]  # 取当前位置之后的路径
            else:
                path, change_end_idx = self._get_target_path(target_idx, lanes, curr_lane_idx, ego_pose, lat_ratio=lat_ratio)
                # if self.changing == False:
                #     path, change_end_idx = self._get_target_path(target_idx, lanes, curr_lane_idx, ego_pose)
                # # TODO：如果当前规划路径覆盖不了换道终点，return None
                # elif self.changing == True: # 查找当前位置之后的点
                #     # path = self.current_path
                #     # min_idx = None
                #     # min_dist, min_idx = nearest_point_on_traj(ego_state[:2], path)
                #     # if min_idx is not None:
                #     #     path = path[min_idx:]  # 取当前位置之后的路径
                #     # 1221x修改：每个时刻重新规划换道路径
                #     path, change_end_idx = self._get_target_path(self.current_target_idx, lanes, curr_lane_idx, ego_pose)
                #     # 
        
        if path is None:
            if self.roslog:
                rospy.loginfo("[GAME模块]  'None' from [_get_sigmoid_path] no enough points.")
            else:
                print("'None' from [_get_sigmoid_path] no enough points.")
            return None
        
        s = np.cumsum(np.hypot(np.diff(path[:,0]), np.diff(path[:,1])))
        s = np.insert(s, 0, 0)
        
        # 如果没有交互对象，直接按照最大加速度沿路径走
        if others_pred is None:
            # a_max = min(self.limits[self.type_mapping['AV']]['MAX_ACCELERATION'], (self.limits[self.type_mapping['AV']]['MAX_SPEED'] - ego_state[3]) / self.t_list[-1])
            a_max = (self.limits[self.type_mapping['AV']]['MAX_SPEED'] - ego_state[3]) / self.dt
            a_from_speedlimit = (self.limits[self.type_mapping['AV']]['MAX_SPEED'] - ego_state[3]) / self.dt
            if a_from_speedlimit < self.limits[self.type_mapping['AV']]['MAX_DECELERATION']:
                self.decel_from_speedlimit = True
            a_max = np.clip(a_max, self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], self.limits[self.type_mapping['AV']]['MAX_ACCELERATION'])
            players_actions = [np.array([a_max for _ in range(len(self.t_list))])]
            AV_reward = - sum([
                (self.av_weights[0] * max(self.TTC_max, self.TTCP_max) + \
                            self.av_weights[1] * ( ego_state[3] + a_max * t ) + \
                            self.av_weights[2] * 0.0) * np.exp(-self.decay_rate * t)
                for t in self.t_list
            ])
            return {
                'target_idx': target_idx,
                'reward': AV_reward,
                'players_ids': ['AV'],
                'action': players_actions,
                'other_trajs_dense': [], 
                'other_s_dense': [],
                'path': path,
                's': s,
                'AV_reward_diff2max':  0,
                'av_confpos': {},
                'hv_confpos_2av': {},
                'a_follow': self.limits[self.type_mapping['AV']]['MAX_ACCELERATION']
            }
        
        t2 = time.time()
        # 2️⃣ hv轨迹处理
        num_points = 10 # 轨迹加密
        n_vehicles, N_steps, _ = others_pred.shape
        hv_trajs_dense = []
        hv_s_dense = []
        for j in range(n_vehicles):
            traj_dense = _densify_trajectory(others_pred[j,:,:], num_points)
            hv_trajs_dense.append(traj_dense)
            s_dense = np.cumsum(np.hypot(np.diff(traj_dense[:,0]), np.diff(traj_dense[:,1])))
            s_dense = np.insert(s_dense, 0, 0)
            hv_s_dense.append(s_dense)
        hv_trajs_dense = np.asarray(hv_trajs_dense) 
        hv_s_dense = np.asarray(hv_s_dense) 
        
        hv_trajs_dense = np.asarray(hv_trajs_dense, dtype=np.float64)  # e.g. (n_vehicles, N_hv, 2)
        hv_s_dense = np.asarray(hv_s_dense, dtype=np.float64)          # e.g. (n_vehicles, N_hv) or (N_hv, n_vehicles)
        if hv_trajs_dense.shape[0] != hv_trajs_dense.shape[1] and hv_trajs_dense.shape[2] == 2:
            pass
        
        ego_path_xy = np.asarray(path[:, :2], dtype=np.float64)
        ego_s = np.asarray(s, dtype=np.float64)
        
        players_states = np.concatenate([np.expand_dims(ego_state, axis=0) , other_states]) # 拼接矩阵
        players_ids = ['AV'] + other_ids
        players_types = ['AV'] + other_types
        players_lengths = [self.ego_length] + other_lengths
        
        # 3️⃣ 冲突计算
        t3 = time.time()
        # 障碍物轮廓投影折算
        new_width_list = []
        for i in range(len(other_states)):
            l, w = others_pred[i][0][-2], others_pred[i][0][-1]
            new_width = calculate_diagonal_projection(l, w, veh_heading = other_states[i][2], ref_heading = ego_state[2])
            new_width_list.append(new_width)
        av_confpos, hv_confpos_2av = _compute_conf_s(ego_path_xy, ego_s, hv_trajs_dense, hv_s_dense, new_width_list, self.confcal_threshold)
        av_confpos = {players_ids[i]: av_confpos[i] for i in range(n_vehicles + 1)}
        hv_confpos_2av = {players_ids[i]: hv_confpos_2av[i] for i in range(n_vehicles + 1)}
        t4 = time.time()
        # # rospy.loginfo("path生成",(t2 - t1)*1000,"ms")
        # # rospy.loginfo("hv轨迹处理耗时",(t3 - t2)*1000,"ms")
        # # rospy.loginfo("冲突计算耗时",(t4 - t3)*1000,"ms")
        # # rospy.loginfo("生成轨迹、冲突计算耗时",(t4 - t1)*1000,"ms")
        
        # RULE【1】: 如果两车到冲突点的距离不为0和-1，则不允许换道，因为认为比较危险
        flag_stopchange = False
        if target_idx != curr_lane_idx and target_idx is not None:
            for hv_id, av_cp in av_confpos.items():
                corr_hv_cp = hv_confpos_2av[hv_id]
                if av_cp == -1:
                    continue
                
                if av_cp> ego_state[3]*5 or corr_hv_cp>other_states[other_ids.index(hv_id)][3] * 5 : # 冲突点在规划终点之后，不考虑
                    continue
                
                if av_cp != 0 and corr_hv_cp != 0 :
                    av_former = - np.sign(av_cp - corr_hv_cp)
                    av_v = ego_state[3]
                    other_v = other_states[other_ids.index(hv_id)][3]
                    o_l = others_pred[other_ids.index(hv_id)][0][-2]
                    # if self.roslog:
                    #     rospy.loginfo(f"[GAME模块]  conf_diff: {-av_former*(av_cp - corr_hv_cp)} VS safe_thred: {self.safechange_threshold + max(-av_former*(av_v - other_v), 0) * 3.0}")
                    # else:
                    #     print(f"conf_diff: {-av_former*(av_cp - corr_hv_cp)} VS safe_thred: {self.safechange_threshold + max(-av_former*(av_v - other_v), 0) * 3.0}")
                    if self.changing == False:
                        if self.force_flag:
                            # if av_former:
                            #     flag = -av_former*(av_cp - corr_hv_cp) <  (self.safechange_threshold + max(0, (o_l/2 - 5/2))*self.k_force_bigcar) + max(-av_former*(av_v - other_v), 0) * self.k_force_follow_TTC # 以小车5m为车辆长度为基准进行调整。
                            # else:
                            #     flag = -av_former*(av_cp - corr_hv_cp) <  (self.safechange_threshold + max(0, (o_l/2 - 5/2))*self.k_force_bigcar) + max(-av_former*(av_v - other_v), 0) * self.k_force_lead_TTC # 以小车5m为车辆长度为基准进行调整。
                            # 【修改1：强制变道放开硬约束】
                            # 取消基于 TTC 的长距离拦截，只保留绝对物理防碰撞底线 (例如 1.5 米 + 车辆尺寸差异)
                            # 将安全评估的权力交还给后续的贝叶斯 Expected Reward
                            absolute_min_gap = 1.5 + max(0, (o_l/2 - 5/2))
                            flag = -av_former*(av_cp - corr_hv_cp) < absolute_min_gap
                        else:
                            if av_former:
                                flag = -av_former*(av_cp - corr_hv_cp) <  (self.safechange_threshold + max(0, (o_l/2 - 5/2))*self.k_free_bigcar) + max(-av_former*(av_v - other_v), 0) * self.k_free_follow_TTC # 以小车5m为车辆长度为基准进行调整。
                            else:
                                flag = -av_former*(av_cp - corr_hv_cp) <  (self.safechange_threshold + max(0, (o_l/2 - 5/2))*self.k_free_bigcar) + max(-av_former*(av_v - other_v), 0) * self.k_free_lead_TTC # 以小车5m为车辆长度为基准进行调整。
                    else:
                        if av_former:
                            flag = -av_former*(av_cp - corr_hv_cp) <  (self.safechange_threshold + max(0, (o_l/2 - 5/2))*self.k_force_bigcar) + max(-av_former*(av_v - other_v), 0) * self.k_force_follow_TTC / 5 # 以小车5m为车辆长度为基准进行调整。
                        else:
                            flag = False
                    if flag:
                        if self.changing == False:
                            if self.roslog:
                                rospy.loginfo("[GAME模块]  'None' from [【RULE1】change should be safe enough (after pos)].")
                            else:
                                print("'None' from [【RULE1】change should be safe enough (after pos)].")
                            return None
                        elif self.changing == True:
                            # if not self.force_flag:
                            flag_stopchange = True
                            self.stopchange_memory_1 = self.stopchange_memory_1[-1:] + [flag_stopchange]
                            if self.stopchange_memory_1 == [True, True]:
                                path, change_end_idx = self._get_target_path(curr_lane_idx, lanes, curr_lane_idx, ego_pose)
                                s = np.cumsum(np.hypot(np.diff(path[:,0]), np.diff(path[:,1])))
                                s = np.insert(s, 0, 0)
                                ego_path_xy = np.asarray(path[:, :2], dtype=np.float64)
                                ego_s = np.asarray(s, dtype=np.float64)
                                # 重置 3️⃣ 冲突计算
                                # 障碍物轮廓投影折算
                                new_width_list = []
                                for i in range(len(other_states)):
                                    l, w = others_pred[i][0][-2], others_pred[i][0][-1]
                                    new_width = calculate_diagonal_projection(l, w, veh_heading = other_states[i][2], ref_heading = ego_state[2])
                                    new_width_list.append(new_width)
                                av_confpos, hv_confpos_2av = _compute_conf_s(ego_path_xy, ego_s, hv_trajs_dense, hv_s_dense, new_width_list, self.confcal_threshold)
                                av_confpos = {players_ids[i]: av_confpos[i] for i in range(n_vehicles + 1)}
                                hv_confpos_2av = {players_ids[i]: hv_confpos_2av[i] for i in range(n_vehicles + 1)}
                                
                                target_idx = curr_lane_idx
                                
                                if self.roslog:
                                    rospy.loginfo("[GAME模块]  由于换道轨迹不安全 ，强制取消换道，改为当前车道行驶.")
                                else:
                                    print("由于换道轨迹不安全 ，强制取消换道，改为当前车道行驶.")
        self.stopchange_memory_1 = self.stopchange_memory_1[-1:] + [flag_stopchange]
        
        # 5️⃣ 博弈实施:
        self.old_style_reward_dict = self.style_reward_dict.copy()
        t3 = time.time()
        if self.roslog:
            rospy.loginfo(f"[GAME模块]  players_ids: {players_ids}")
        else:
            print(f"players_ids: {players_ids}")
        self.now_cal_path_idx = target_idx if target_idx is not None else curr_lane_idx
        self.now_current_path_idx = curr_lane_idx
        players_actions, AV_reward, AV_reward_diff2max, style_reward_dict, a_follow = self._solve_nash_equilibrium_symbolic_discrete(players_states, 
                                                                            players_ids, 
                                                                            players_types, 
                                                                            players_lengths,
                                                                            av_confpos, 
                                                                            hv_confpos_2av,
                                                                            ego_path = path,
                                                                            ego_path_s = s,
                                                                            hvs_path = hv_trajs_dense,
                                                                            hvs_s= hv_s_dense,)
        
        # 【RULE2】以跟驰加速度为上限
        if players_actions[0][0] > a_follow:
            players_actions[0] = [a_follow for _ in range(len(self.t_list))]
        # rospy.loginfo(f"[GAME模块]  Single target solving time: {(time.time() - t3) * 1000:.2f} ms")
        self.style_reward_dict = style_reward_dict.copy()
        
        # 【RULE3】
        # 如果是换道且规划时域没有覆盖换道终点
        if target_idx != curr_lane_idx and target_idx is not None:
            if self.changing == False and not self.force_flag:
                # 计算按照最优加速度行驶的状态更新t_list
                a = players_actions[0]
                s_t = 0.0 + ego_state[3]*self.t_list[-1] + 0.5*a[0]*np.array(self.t_list[-1])**2 # TODO: 这里假设匀加速
                s_idx = np.argmin(np.abs(s_t - s))
                if s_idx < change_end_idx:
                    if self.roslog:
                        rospy.loginfo("[GAME模块]  'None' from [【RULE3】] 换道没有覆盖到终点.")
                    else:
                        print("'None' from [【RULE3】] 换道没有覆盖到终点.")
                    return None
        
        # 【RULE4】
        # 检测沿轨迹是否存在碰撞
        if target_idx != curr_lane_idx and self.changing == False and target_idx is not None:
            if self.force_flag and self.is_hv_gamepred: # 只有强制换道时启用博弈更改后的预测轨迹
                # === HV预测轨迹生成 ===
                other_trajs = []
                for i, o_traj in enumerate([hv_trajs_dense[i-1] for i in range(len(players_states)) if i>0 and players_ids[i] in players_ids]):
                    if other_ids[i] not in players_ids:
                        # 非交互对象，保持原始预测不变
                        other_trajs.append(others_pred[i, :, :])
                        continue
                    x, y, heading, v,  = other_states[i][:4]
                    length, width = others_pred[i][0][-2:]
                    best_traj = o_traj
                    path_s = hv_s_dense[i]  # 原始路径的弧长序列
                    o_ori_best_a = players_actions[i + 1]
                    o_interp_best_a = np.interp(self.t_output_other, self.t_list, o_ori_best_a)
                    
                    v_list = [v]
                    v_max_i = max(self.limits[self.type_mapping[other_types[i]]]['MAX_SPEED'], v)
                    s_list = [0.0]
                    length_list = [length]
                    width_list = [width]
                    dt_list = np.diff(np.insert(self.t_output_other, 0, 0))
                    for j, dt in enumerate(dt_list):
                        a = o_interp_best_a[j]
                        v_next = v_list[-1] + a * dt
                        
                        # 1223 取原有结果和博弈计算结果小的那个，以避免静止障碍物的预测错误
                        v_pred = others_pred[i][j][3]
                        v_next = min(v_next, v_pred)  
                        
                        v_next = np.clip(v_next, 0.0, v_max_i)
                        s_next = s_list[-1] + v_list[-1] * dt + 0.5 * a * dt**2
                        v_list.append(v_next)
                        s_list.append(s_next)
                    
                    # 对应路径上的坐标插值
                    x_list = np.interp(s_list, path_s, best_traj[:, 0])
                    y_list = np.interp(s_list, path_s, best_traj[:, 1])
                    heading_list = np.interp(s_list, path_s, best_traj[:, 2])
                    
                    o_traj = np.stack([x_list, y_list, heading_list, v_list, np.array(length_list*len(x_list)), np.array(width_list*len(x_list))], axis=1)
                    other_trajs.append(o_traj)
                other_trajs = np.array(other_trajs)  if len(other_trajs) > 0 else None # 转换为NumPy数组
                
                if_collision = path_collision_detect(path, s, players_actions[0], ego_state, self.t_list, other_trajs,
                                                     margin=self.collision_margin_m)
            else:
                if_collision = path_collision_detect(path, s, players_actions[0], ego_state, self.t_list, others_pred,
                                                     margin=self.collision_margin_m)
            if if_collision:
                if self.roslog:
                    rospy.loginfo("[GAME模块]  'None' from [path_collision_detect] 换道轨迹存在碰撞.")
                else:
                    print("'None' from [path_collision_detect] 换道轨迹存在碰撞.")
                
                AV_reward -= 10000.0  # 手动施加惩罚，确保期望为负
                AV_reward_diff2max += 10000.0
        
        # 6️⃣ 更新对交互对象的风格识别和权重
        if self.roslog:
            rospy.loginfo(f"[GAME模块]   AV reward: {AV_reward}, AV action: {players_actions[0]}")
        else:
            print(" AV reward: ", AV_reward, "AV action:", players_actions[0])
        return {
            'target_idx': target_idx,
            'reward': AV_reward,
            'players_ids': players_ids,
            'action': players_actions,
            'other_trajs_dense': [hv_trajs_dense[i-1] for i in range(len(players_states)) if i>0 and players_ids[i] in players_ids], 
            'other_s_dense': [hv_s_dense[i-1] for i in range(len(players_states)) if i>0 and players_ids[i] in players_ids],
            'path': path,
            's': s,
            'AV_reward_diff2max': AV_reward_diff2max,
            'av_confpos': av_confpos,
            'hv_confpos_2av': hv_confpos_2av,
            'a_follow': a_follow,
        }
    
    # AEB：独立于博弈逻辑，判断车辆行驶方向前方有近距离障碍物急刹车
    def check_aeb(self, ego_state, others_info):
        """
        新AEB判定逻辑：
        1. 获取自车轨迹（current_path），默认取前N点
        2. 遍历障碍物，判断是否在前方且横向距离小于车宽+0.5m
        3. 计算障碍物heading与自车轨迹交点，得到TTC
        4. 计算主车在TTC内减速至0的aeb减速度
        5. 若aeb减速度大于阈值，触发AEB
        返回: (aeb_flag, aeb_acc)
        """
        # 可调节参数
        ego_width = self.ego_width
        ego_length = self.ego_length

        TTC_threshold = 2
        lateral_threshold = 1.5 * ego_width  # 横向距离阈值
        safe_speed_threshold = 2  # 如触发AEB需要使得速度小于对方该阈值，而不是减速到与他一致
        safe_distance = 2.0  # 安全距离
        dt = 0.1
        pred_horizon = 1.5 * TTC_threshold # 预测时域
        max_brake_acc = -5.0 # 最大刹车加速度
        # ----------

        aeb_flag = False
        aeb_acc = None
        x, y, heading, v = ego_state[:4]
        
        if self.current_path is None or len(self.current_path) < 2:
            return aeb_flag, aeb_acc
        
        # 允许others_info为None
        if others_info is None:
            return aeb_flag, aeb_acc
        other_states = others_info['states']
        other_types = others_info.get('types', None)
        other_ids = others_info.get('ids', None)
        others_pred = others_info.get('pred', None)
        
        if other_states is None:
            return aeb_flag, aeb_acc
        
        # 支持障碍物状态为[x, y, heading, v, ...]
        min_aeb_acc = 0
        for idx, obs in enumerate(other_states):
            # print("[FSY DEBUG] Checking obs:", obs, ego_state)
            ox, oy, oheading = obs[:3]
            ov = obs[3] if len(obs) > 3 else 0.0
            obs_length = others_pred[idx][0][-2]
            obs_width = others_pred[idx][0][-1]
            
            # 1. 判断障碍物是否在自车前方
            rel_x = ox - x
            rel_y = oy - y
            forward_proj = rel_x * np.cos(heading) + rel_y * np.sin(heading)
            if forward_proj < 0:
                continue  # 不在前方
            # 2. 横向距离
            lateral_proj = abs(-rel_x * np.sin(heading) + rel_y * np.cos(heading))
            # if self.roslog:
            #     rospy.loginfo(f"[GAME模块][AEBcheck]  rel_x: {rel_x}, rel_y: {rel_y}, forward_proj: {forward_proj}, lateral_proj: {lateral_proj}")
            # else:
            #     print("[FSY DEBUG] rel_x, rel_y, forward_proj:", rel_x, rel_y, forward_proj, lateral_proj)
            if other_types is not None and idx < len(other_types):
                # 可根据类型自定义宽度
                pass
            if lateral_proj > (ego_width + obs_width) / 2 + lateral_threshold:
                continue  # 横向距离太远
            # 3. 外推障碍物轨迹，找与自车轨迹的第一个交点
            # print("[FSY DEBUG] Generating trajectories for collision check...")
            obs_traj = []
            for k in range(int(pred_horizon/dt)):
                t = k * dt
                px = ox + ov * t * np.cos(oheading)
                py = oy + ov * t * np.sin(oheading)
                obs_traj.append([px, py])
            obs_traj = np.array(obs_traj)
            
            ego_traj = []
            for k in range(int(pred_horizon/dt)):
                t = k * dt
                px = x + v * t * np.cos(heading)
                py = y + v * t * np.sin(heading)
                ego_traj.append([px, py])
            ego_traj = np.array(ego_traj)
            
            cross_idx = None
            min_idx = None
            # print("[FSY DEBUG] obs_traj:", obs_traj, "ego_traj:", ego_traj)
            for i, ego_pt in enumerate(ego_traj):
                ex, ey = ego_pt[0], ego_pt[1]
                dists = np.linalg.norm(obs_traj - np.array([ex, ey]), axis=1)
                min_idx_tmp = np.argmin(dists)
                # print("[FSY DEBUG] min_dist:", dists[min_idx_tmp], "min_idx_tmp:", min_idx_tmp)
                if dists[min_idx_tmp] < ego_width/2 + obs_width/2:
                    cross_idx = i
                    min_idx = min_idx_tmp
                    break
            # if self.roslog:
            #     rospy.loginfo(f"[GAME模块][AEBcheck]  cross_idx: {cross_idx}, min_idx: {min_idx}")
            # else:
            #     print("[FSY DEBUG] cross_idx, min_idx:", cross_idx, min_idx)
            if cross_idx is None or min_idx is None:
                print("[FSY DEBUG] No crossing point found.")
                continue  # 没有交点
            # 4. 计算TTC
            ego_s = np.cumsum(np.hypot(np.diff(ego_traj[:,0]), np.diff(ego_traj[:,1])))
            ego_s = np.insert(ego_s, 0, 0)
            ego_dist2cross = ego_s[cross_idx]
            # 速度为0时不触发
            if v < 0.1:
                continue
            rel_v = v - ov + safe_speed_threshold
            rel_v = max(rel_v, 1e-3)  # 防止除零
            ttc_ego = (ego_dist2cross - ego_length/2 - obs_length/2 - safe_distance) / max(rel_v, 1e-3)
            ttc = max(ttc_ego, 0.1)
            # b_ttc = (ego_dist2cross - ego_length/2) / max(v, 1e-3)
            # print(f"[FSY DEBUG] last version ttc {b_ttc}, bev acc {v/b_ttc}")
            # print("[FSY DEBUG] vel:", v, "ov:", ov)
            # print("[FSY DEBUG] dist2cross:", ego_dist2cross - ego_length/2 - obs_length/2, "rel_v:", rel_v, "ttc:", ttc, "bev acc:", -rel_v / ttc)
            # 5. 计算主车在TTC内减速至0的aeb减速度
            if TTC_threshold >= ttc >= 0.1:
                req_aeb_acc = -rel_v / ttc
                # 只要比当前AEB阈值更紧急就触发
                if req_aeb_acc < min_aeb_acc:
                    aeb_flag = True
                    aeb_acc = req_aeb_acc
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块][AEBcheck]  AEB Triggered! req_aeb_acc: {req_aeb_acc}, TTC: {ttc}")
                    else:
                        print("[FSY DEBUG] AEB Triggered! req_aeb_acc:", req_aeb_acc, "TTC:", ttc)
        if self.roslog:
            rospy.loginfo(f"[GAME模块][AEBcheck]  aeb_flag: {aeb_flag}, aeb_acc: {aeb_acc}")
        else:
            print("[FSY DEBUG]aeb_flag:", aeb_flag, "aeb_acc:", aeb_acc)
        aeb_acc = np.clip(aeb_acc, max_brake_acc, 0.0) if aeb_acc is not None else None
        return aeb_flag, aeb_acc
    
    # 获取到车道终点的加速度
    def get_end_acc(self, ego_state, result, other_states, other_lengths, cal_end = False):
        # if self.waiting:
        #     idm_params = {
        #             'a': 1.0,        # 最大加速度 m/s^2 (typical comfortable accel)
        #             'b': 2.0,        # 期望舒适减速度 m/s^2
        #             's0': max(self.L_min, self.s0),       # 最小间距 m
        #             'T': self.idm_T,        # 时距 s
        #             'delta': 4.0,    # 速度指数
        #             'v0': float(self.limits[self.type_mapping['AV']]['MAX_SPEED'])  # 期望速度
        #         }
        #     a_follow_detect = self._find_lead_and_idm( ego_state=ego_state,
        #                                     ego_path=result['path'],
        #                                     path_s=result['s'],
        #                                     other_states=other_states,
        #                                     other_lengths=other_lengths,
        #                                     other_ids=result['players_ids'][1:],
        #                                     av_confpos = result['av_confpos'],
        #                                     hv_confpos_2av = result['hv_confpos_2av'],
        #                                     max_lateral_dist = self.max_lateral_dist,
        #                                     idm_params = idm_params,)
        #     path_s = result['s']
        #     if path_s[-1] > self.dis2end_threshold:
        #         a_2end = - ego_state[3]**2/(2*(path_s[-1] - self.dis2end_threshold))
        #     else:
        #         a_2end = self.limits[self.type_mapping['AV']]['MAX_DECELERATION']
        #     if a_2end <= self.limits[self.type_mapping['AV']]['MAX_DECELERATION'] * 1:
        #         a_2end = a_2end
        #     else:
        #         a_2end = a_follow_detect['a_idm']
        #     a_min = max(self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], -ego_state[3]/self.t_output_av[0])
        #     # a_ = max(min(a_follow_detect, a_2end), a_min)
        #     a_ = max(min(a_follow_detect['a_idm'], a_2end), a_min)
        #     result['action'][0] = [a_ for _ in range(len(self.t_list))]
        #     result['reward'] = result['reward'] + 100
        #     result['AV_reward_diff2max'] = result['AV_reward_diff2max'] + 100
        # else:
        #     pass
        if cal_end:
            a_now = ego_state[-2]
            max_jerk = 1.0  # m/s^3
            path_s = result['s']
            if path_s[-1] > self.dis2end_threshold:
                a_2end = - ego_state[3]**2/(2*(path_s[-1] - self.dis2end_threshold))
            else:
                a_2end = self.limits[self.type_mapping['AV']]['MAX_DECELERATION']
            if a_2end <= self.limits[self.type_mapping['AV']]['MAX_DECELERATION'] * 0.5:
                a_2end = a_2end
                a_2end = max(a_2end, a_now - max_jerk)
            else:
                a_2end = result['action'][0][0]
            a_min = max(self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], -ego_state[3]/self.t_output_av[0])
            # a_ = max(min(a_follow_detect, a_2end), a_min)
            a_ = max(min(result['action'][0][0], a_2end), a_min)
            result['action'][0] = [a_ for _ in range(len(self.t_list))]
            result['reward'] = result['reward'] + 100
            result['AV_reward_diff2max'] = result['AV_reward_diff2max'] + 100
        

        return result

    # 主要调用
    def _prallel_solve_game(self,
                            data_process,   # 输入数据处理函数
                            others_info,
                            DFpath=None,    # 与东风融合：0118
                            
                            ):
        """
        在多候选目标车道上并行求解孪生博弈问题，选出最优轨迹及控制动作。
        所有轨迹和其他车辆状态均基于 Frenet 坐标系。
        返回：
            best_traj: np.ndarray [x, y, heading, v]
            best_action: dict {'acceleration': a*, 'steering': δ*}
        """
        t0 = time.time()
        if self.roslog:
            rospy.loginfo(f"[GAME模块]  当前测试轮次风格: {self.AV_style}")
        else:
            print(f"[GAME模块]  当前测试轮次风格: {self.AV_style}")
        
        # 初始化本次控制周期的分支计数器
        self.evaluated_branches_count = 0

        self.decel_from_speedlimit = False
        self.DFpath = DFpath
        # 对齐数据处理
        ego_state = copy.deepcopy(data_process.ego_state)
        target_idxes = list(data_process.near_target_idxes) if DFpath is None else [data_process.curr_lane_idx]
        # =========================================================
        # 视野截断 (Spatial Truncation)
        # =========================================================
        ego_x, ego_y = data_process.ego_state[0], data_process.ego_state[1]
        
        # 初始化上一帧的索引缓存（如果还没有的话）
        if not hasattr(self, 'last_ego_idxes'):
            self.last_ego_idxes = {}
            
        truncated_lanes = {}
        for l_idx, pts in data_process.lanes.items():
            total_points = len(pts)
            if total_points == 0:
                continue

            search_start = 0
            search_end = total_points
            
            # 1. 极速定位：时序缓存 (Temporal Coherence)
            # 车辆每帧只移动一两米，直接用上一帧的索引前后找 20 个点即可，O(1) 复杂度！
            if l_idx in self.last_ego_idxes:
                last_idx = self.last_ego_idxes[l_idx]
                search_start = max(0, last_idx - 10)
                search_end = min(total_points, last_idx + 30)
            else:
                # 只有第一帧才会走这里，做一次粗略搜索
                stride = 20
                min_dist_sq = float('inf')
                coarse_idx = 0
                for i in range(0, total_points, stride):
                    dist_sq = (pts[i].x - ego_x)**2 + (pts[i].y - ego_y)**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        coarse_idx = i
                search_start = max(0, coarse_idx - stride)
                search_end = min(total_points, coarse_idx + stride)

            # 2. 精确搜索（只在极小范围内遍历）
            exact_idx = search_start
            min_dist_sq = float('inf')
            for i in range(search_start, search_end):
                dist_sq = (pts[i].x - ego_x)**2 + (pts[i].y - ego_y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    exact_idx = i

            # 记录当前帧索引，留给下一帧使用
            self.last_ego_idxes[l_idx] = exact_idx

            # 3. 提取视野 (后方 50，前方 300)
            slice_start = max(0, exact_idx - 50)
            slice_end = min(total_points, exact_idx + 300)


            fast_copy = []
            for p in pts[slice_start:slice_end]:
                # 手动实例化新的 Point 对象，打破原对象的内存引用
                fast_copy.append(
                    Point(
                        x=p.x, y=p.y, heading=p.heading, speed=p.speed, 
                        curvature=p.curvature, to_left=p.to_left, to_right=p.to_right, 
                        point_type=p.point_type, left_traverse_flag=p.left_traverse_flag, 
                        right_traverse_flag=p.right_traverse_flag
                    )
                )
            truncated_lanes[l_idx] = fast_copy

        # 5. 加密插值
        lanes = _dense_refline(truncated_lanes)
        t_copy = time.time()

        curr_lane_idx = data_process.curr_lane_idx
        # permit_idxes = list(data_process.near_permit_idxes) # 绕行开启
        permit_idxes = list(data_process.near_target_idxes) if DFpath is None else [data_process.curr_lane_idx]# 绕行关闭
        AV_Cruise = data_process.auto_state # 自动驾驶状态为1，接管状态为0

        # 更新v0       
        self.v0 = self.limits[self.type_mapping['AV']]['MAX_SPEED'] + self.style_params['v0+']
        if self.roslog:
            rospy.loginfo(f"[GAME模块] 当前最大速度: {self.v0 * 3.6} km/h")
        else:
            print(f"当前最大速度: {self.v0 * 3.6} km/h")

        if others_info is None:
            others_info = {'pred': None, 'ids': None, 'types': None,'states': None}
        other_states = others_info['states']; others_pred = others_info['pred']; other_ids = others_info['ids']; other_types = others_info['types']
        other_lengths = [op[0][-2] for op in others_pred] if others_pred is not None else []
        # other_widths = [op[0][-1] for op in others_pred] if others_pred is not None else []
        
        self.others_laneid = others_info['laneids']
        

        # 接管状态换道重置：要求驾驶员在车道中心线附近重启自动驾驶
        if AV_Cruise == 0:
            self.changing = False
            self.current_target_idx = curr_lane_idx
            if self.roslog:
                rospy.loginfo(f"[GAME模块]  自动驾驶接管! 重置换道状态。") 
            else:
                print(f"[GAME模块]  自动驾驶接管! 重置换道状态。")

        start_time = time.time()
        #【RULE0】如果正在换道 & 目标车道的车道索引发生变化，需要重新赋值
        if self.changing == True:
            # 换道后就从所有车道中找距离参考点最近的车道作为目标车道
            # 找到lanes中与self.current_target_idx_point距离最近的索引，作为当前的self.current_target_idx
            closest_dis = float('inf')
            finded_nearest_target_idx = None
            for l_idx, ps in lanes.items():
                if l_idx in target_idxes:
                    target_points = lanes[l_idx]
                    ref_point = np.array(self.current_target_idx_point)
                    distances = np.array([np.hypot(pt.x - ref_point[0], pt.y - ref_point[1]) for pt in target_points])
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < closest_dis:
                        finded_nearest_target_idx = l_idx
                        closest_dis = distances[closest_idx]
            self.current_target_idx = finded_nearest_target_idx
            # 更新参考点位置
            mid_idx = len(lanes[self.current_target_idx]) // 2
            self.current_target_idx_point = (lanes[self.current_target_idx][mid_idx].x, lanes[self.current_target_idx][mid_idx].y)
        
        results = {}
        
        
        self._update_beliefs(others_info) # 【贝叶斯】更新意图信念分布
        # self._update_weights(others_info) # 更新效率权重
        # self._update_weights_old(other_ids) # 不更新权重 TODO

        if self.roslog:
            rospy.loginfo(f"[GAME模块]  ----- Current changing: {self.changing} -----")
        else:
            print(f"----- Current changing: {self.changing} -----")
        if self.Changing_Permitted == False:
            target_idxes = [curr_lane_idx]
        
        # 判断是否需要启用备选车道
        if not self.changing and target_idxes ==[curr_lane_idx]:
            if ego_state[3] < 0.1: # 静止状态下启用
                target_idxes = permit_idxes.copy()
                if self.roslog:
                    rospy.loginfo(f"[GAME模块]  当前车道为唯一目标车道，且车辆静止，启用备选车道：{target_idxes}")
                else:
                    print(f"当前车道为唯一目标车道，且车辆静止，启用备选车道：{target_idxes}")
                 
        # 判断是否强制换道
        if self.changing and curr_lane_idx != self.current_target_idx:
            if self.force_flag and curr_lane_idx in target_idxes:
                target_idxes.remove(curr_lane_idx)
        
        if curr_lane_idx not in target_idxes:
            self.force_flag = True
        else:
            self.force_flag = False

        # # 测试加塞用
        self.force_flag = curr_lane_idx not in target_idxes

        if self.force_flag:
            # 强制换道：根据目标车道前后车状态调整自车速度，创造换道机会
            fc_leadid, fc_followid = others_info['force_change_id']
            fc_leadid_idx = other_ids.index(fc_leadid) if fc_leadid is not None else None
            fc_followid_idx = other_ids.index(fc_followid) if fc_followid is not None else None
            fc_lead_v = other_states[fc_leadid_idx][3] if fc_leadid_idx is not None else None
            fc_follow_v = other_states[fc_followid_idx][3] if fc_followid_idx is not None else None
            fc_lead_s = np.hypot(ego_state[0] - other_states[fc_leadid_idx][0], ego_state[1] - other_states[fc_leadid_idx][1]) if fc_leadid_idx is not None else None
            fc_follow_s = np.hypot(ego_state[0] - other_states[fc_followid_idx][0], ego_state[1] - other_states[fc_followid_idx][1]) if fc_followid_idx is not None else None
            ttc_follow = (fc_follow_s) / max(1e-3, fc_follow_v - ego_state[3]) if fc_followid_idx is not None else None
            ego_v = ego_state[3]
            v_force = max(ego_v, fc_follow_v* 0.8) if fc_followid_idx is not None else ego_v
            safe_upper_v = fc_lead_v if fc_leadid_idx is not None else self.limits[self.type_mapping['AV']]['MAX_SPEED']
            if fc_leadid_idx is not None:
                dis2fc_lead = np.hypot(ego_state[0] - other_states[fc_leadid_idx][0], ego_state[1] - other_states[fc_leadid_idx][1]) if fc_leadid_idx is not None else None
                if dis2fc_lead < ego_v * 3.0:
                    v_force = min(v_force, fc_lead_v)
            
            # if ego_v < v_force * 0.6 and ttc_follow < 4.0:
            #     self.waiting = True
            #     self.limits[self.type_mapping['AV']]['MAX_SPEED'] = self.limits[self.type_mapping['AV']]['MAX_SPEED'] * 0.6
            if ego_v < 10:
                if ego_v < v_force * 0.6 and ttc_follow is not None and ttc_follow < 4.0:
                    # 1. 太慢且极度危险：重刹让行
                    self.waiting = True
                    self.v0 = self.v0 * 0.6  # 降速等待后车过去 (结合我们上一次的软控修改)
                elif ego_v < v_force:
                    # 2. 偏慢：继续在当前车道待命
                    self.waiting = True
                elif ego_v > safe_upper_v + 2.0:  # 容忍 2m/s (约 7km/h) 的速度差
                    # 3. 【新增】太快：启动主动速度同步
                    self.waiting = True       # 不允许立刻打方向变道
                    self.v0 = max(safe_upper_v, 0.0) # 强制降低期望速度 v0，诱导底层求解器在当前车道踩刹车减速
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]  自车过快 (自车:{ego_v:.1f}m/s, 目标:{safe_upper_v:.1f}m/s)，主动执行减速同步！")
                    else:
                        print(f"[GAME模块]  自车过快 (自车:{ego_v:.1f}m/s, 目标:{safe_upper_v:.1f}m/s)，主动执行减速同步！")
                else:
                    # 4. 速度匹配良好（既不太快也不太慢），允许执行横向博弈变道
                    self.waiting = False
        
        
        
    
        # 如果正在换道
        changing = self.changing
        target_idxes_init = target_idxes.copy()
        # flag2 = False
        if self.changing == True:
            target_idxes = [self.current_target_idx]
            target_idx = self.current_target_idx
            if self.roslog:
                rospy.loginfo(f"[GAME模块]  ---------->>>>>>>>>>>>>>>>>   正在换道 Solving for target {target_idx}...")
            else:
                print(f"---------->>>>>>>>>>>>>>>>>   正在换道 Solving for target {target_idx}...")
            result = self.solve_for_one_target(target_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths)
            if is_near_road_end(
                    ego_state[:2], 
                    (lanes[result['target_idx']][-1].x, lanes[result['target_idx']][-1].y), 
                    threshold=max(-ego_state[3]**2/2/self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], ego_state[3]*self.t_list[-1])+max(self.L_min, self.s0)*3):
                result = self.get_end_acc(ego_state, result, other_states, other_lengths)
            if self.roslog:
                rospy.loginfo(f"[GAME模块]  result: {result['reward'], result['action'][0]}")
            else:
                print(f"result: {result['reward'], result['action'][0]}")
            results[result['target_idx']] = result
            if result['target_idx'] != target_idx: # 换道过程不安全，换道中止
                changing = False
                self.stopchanging = True

        else:
            # self.stopchange_memory_2 = self.stopchange_memory_2[-1:] + [flag2]
            # === TODO 循环求解 ===

            # 要求：当前车道在目标车道中
            if curr_lane_idx in target_idxes:
                # 先求当前车道的
                if self.roslog:
                    rospy.loginfo(f"[GAME模块]  -------------------------->>>>>>>>>>>>>>>>>   Solving for current target... <<{curr_lane_idx}>>")
                else:
                    print(f"---------->>>>>>>>>>>>>>>>>   Solving for current target... <<{curr_lane_idx}>>...")
                
                result = self.solve_for_one_target(curr_lane_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths)
                current_result = result
                # 【RULE5-1】判断当前车道是否到达尽头
                if is_near_road_end(
                    ego_state[:2], 
                    (lanes[curr_lane_idx][-1].x, lanes[curr_lane_idx][-1].y), 
                    threshold=max(-ego_state[3]**2/2/self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], ego_state[3]*self.t_list[-1])+max(self.L_min, self.s0)*3):
                    
                    result = self.get_end_acc(ego_state, result, other_states, other_lengths)
                    
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]      >>>>> Approaching road end, applying deceleration")
                    else:
                        print(f"    >>>>> Approaching road end, applying deceleration")
                
                results[curr_lane_idx] = result
                
                if self.roslog:
                    rospy.loginfo(f"[GAME模块]  result: {result['reward'], result['action'][0]}")
                else:
                    print(f"result: {result['reward'], result['action'][0]}")

                if result['AV_reward_diff2max'] <= self.AV_reward_diff2max_threshold:
                    target_idxes = []
                
                else:
                    for target_idx in target_idxes:
                        if target_idx == curr_lane_idx:
                            continue
                        
                        if is_near_road_end(
                            ego_state[:2], 
                            (lanes[target_idx][-1].x, lanes[target_idx][-1].y), 
                            threshold=max(-ego_state[3]**2/2/self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], ego_state[3]*self.t_list[-1])+max(self.L_min, self.s0)*3):
                            if self.roslog:
                                rospy.loginfo(f"[GAME模块]  Skipping target <<{target_idx}>> due to proximity to road end.")
                            else:
                                print(f"Skipping target <<{target_idx}>> due to proximity to road end.")
                            continue
                        
                        if self.roslog:
                            rospy.loginfo(f"[GAME模块]  -------------------------->>>>>>>>>>>>>>>>>   Solving for target <<{target_idx}>>...")
                        else:
                            print(f"---------->>>>>>>>>>>>>>>>>   Solving for target <<{target_idx}>>...")
                        
                        result = self.solve_for_one_target(target_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths)
                        if result is not None:
                            results[target_idx] = result
                            if self.roslog:
                                rospy.loginfo(f"[GAME模块]  result: {result['reward'], result['action'][0]}")
                            else:
                                print(f"result: {result['reward'], result['action'][0]}")
            
            # 如果当前车道不在目标车道中
            else:
                if self.force_flag == False:
                    # ==========================================================
                    # 【新增】并行博弈护栏：纵向解绑，侧向拦截 (Delayed Cut-in)
                    # ==========================================================
                    is_parallel_interacting = False
                    blocking_vid = None
                    # 定义一个安全超车余量（超出对方半个车身 + 自车半个车身 + 1.5米安全缓冲）
                    safe_cut_in_margin = 1.5 
                    self.waiting = False
                    if self.changing == False and curr_lane_idx not in target_idxes:
                        for idx, vid in enumerate(other_ids):
                            o_state = other_states[idx]
                            o_length = others_pred[idx][0][-2]
                            
                            # 简单计算相对位置 (投影到自车航向)
                            dx = o_state[0] - ego_state[0]
                            dy = o_state[1] - ego_state[1]
                            forward_dist = dx * math.cos(ego_state[2]) + dy * math.sin(ego_state[2])
                            lateral_dist = abs(-dx * math.sin(ego_state[2]) + dy * math.cos(ego_state[2]))
                            
                            # 检查是否有一辆车在相邻车道 (横向距离约一车道宽)，并且处于“并行或刚刚落后”的危险区域
                            if 1.5 < lateral_dist < 5.0:
                                # 定义前后的并行感知边界
                                # 下界：对方车头落后于我们车尾的安全线
                                rear_bound = - (o_length/2 + self.ego_length/2 + safe_cut_in_margin)
                                # 上界：对方车尾超出身前的一定距离（比如前方 5 米，超过这个距离说明他已经走远了，不再是并行交互）
                                front_bound = (o_length/2 + self.ego_length/2 + 15) 
                                
                                # 如果对方在这两个边界之间，说明正处于“并行或刚刚落后/超前”的黏滞区域
                                if rear_bound < forward_dist < front_bound:
                                    is_parallel_interacting = True
                                    blocking_vid = vid
                                    break

                        if is_parallel_interacting:
                            # 获取 LLM/贝叶斯对这辆车的保守意图概率
                            p_yield = self.belief_dict.get(blocking_vid, [0.0, 1.0, 0.0])[0]
                            
                            if p_yield > 0.5:
                                # 【战术：加速抢行】对方是保守派，我们决定加速超车
                                self.waiting = True  # 【横向拦截】禁止当前帧打方向盘切入！
                                self.v0 = self.limits[self.type_mapping['AV']]['MAX_SPEED'] + 5.0 # 【纵向刺激】强行拔高期望速度，诱导 CasADi 全力加速
                                if self.roslog:
                                    rospy.loginfo(f"[GAME模块] 并行超车中... 压制侧向换道，等待完全超越车辆 {blocking_vid}。")
                                else:
                                    print(f"[GAME模块] 并行超车中... 压制侧向换道，等待完全超越车辆 {blocking_vid}。")
                            else:
                                # 【战术：减速让行】对方是激进派，我们决定减速退让
                                self.waiting = True
                                self.v0 = max(0.0, other_states[other_ids.index(blocking_vid)][3] - 3.0) # 降速到比他慢，等他先走
                                if self.roslog:
                                    rospy.loginfo(f"[GAME模块] 并行博弈：对方激进，主动减速退让。")
                                else:
                                    print(f"[GAME模块] 并行博弈：对方激进，主动减速退让。")
                for target_idx in target_idxes:
                    if target_idx == curr_lane_idx:
                        continue # 当前车道前面已经算过了
                    # 动作 1：探入 (Probe)
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]  -----> 评估探入动作 (Probe) 目标车道 <<{target_idx}>>...")
                    if ego_state[3] < 10: # 低速条件才允许探入动作
                        result_probe = self.solve_for_one_target(target_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths, lat_ratio=0.3)
                    
                        if result_probe is not None:
                            # 终点减速逻辑
                            if is_near_road_end(ego_state[:2], (lanes[target_idx][-1].x, lanes[target_idx][-1].y), threshold=50.0):
                                result_probe = self.get_end_acc(ego_state, result_probe, other_states, other_lengths)
                            result_probe['action_type'] = 'probe'
                            results[f"{target_idx}_probe"] = result_probe
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]  -----> 评估探入动作 (Probe) 目标车道 <<{target_idx}>>...")
                    
                    # # 动作 3：直行 (Keep)
                    # if ego_state[3] < 100: # 低速条件才允许探入动作
                    #     result_keep = self.solve_for_one_target(target_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths, lat_ratio=0.0)
                    
                    #     if result_keep is not None:
                    #         # 终点减速逻辑
                    #         if is_near_road_end(ego_state[:2], (lanes[target_idx][-1].x, lanes[target_idx][-1].y), threshold=50.0):
                    #             result_keep = self.get_end_acc(ego_state, result_keep, other_states, other_lengths)
                    #         result_probe['action_type'] = 'keep'
                    #         results[f"{target_idx}_probe"] = result_keep

                    if self.waiting == False:    
                    # 动作 2：完整换道 (Commit)
                        if self.roslog:
                            rospy.loginfo(f"[GAME模块]  -----> 评估完整换道 (Commit) 目标车道 <<{target_idx}>>...")
                        result_commit = self.solve_for_one_target(target_idx, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths, lat_ratio=1.0)
                        
                        if result_commit is not None:
                            if is_near_road_end(ego_state[:2], (lanes[target_idx][-1].x, lanes[target_idx][-1].y), threshold=50.0):
                                result_commit = self.get_end_acc(ego_state, result_commit, other_states, other_lengths)
                            result_commit['action_type'] = 'commit'
                            results[f"{target_idx}_commit"] = result_commit
                        
            # 处理无解的情况
            if results == {}:
                # 没有任何解，强制计算当前车道停车解
                if self.roslog:
                    rospy.loginfo(f"[GAME模块]  ---------->>>>>>>>>>>>>>>>>   换道路径均无解: Solving for current <<{curr_lane_idx}>>...")
                else:
                    print(f"-------------------------->>>>>>>>>>>>>>>>>   换道路径均无解: Solving for current <<{curr_lane_idx}>>...")
                result = self.solve_for_one_target(None, ego_state, lanes, curr_lane_idx, others_pred, other_states, other_ids, other_types, other_lengths)
                result['target_idx'] = curr_lane_idx
                
                if is_near_road_end(
                        ego_state[:2], 
                        (lanes[curr_lane_idx][-1].x, lanes[curr_lane_idx][-1].y), 
                        threshold=max(-ego_state[3]**2/2/self.limits[self.type_mapping['AV']]['MAX_DECELERATION'], ego_state[3]*self.t_list[-1])+max(self.L_min, self.s0)*3):
                    result = self.get_end_acc(ego_state, result, other_states, other_lengths, cal_end = True)
                    if self.roslog:
                        rospy.loginfo(f"[GAME模块]      >>>>> 强制换道过程中接近道路终点，加速度{result['action'][0][0]}")
                    else:
                        print(f"    >>>>> 强制换道过程中接近道路终点，加速度{result['action'][0][0]}")
                results[curr_lane_idx] = result
                if self.roslog:
                    rospy.loginfo(f"[GAME模块]  Forced stop solution on current lane as no other solutions found.")
                    rospy.loginfo(f"[GAME模块]  result: {result['reward'], result['action'][0]}")
                else:
                    print(f"Forced stop solution on current lane as no other solutions found.")
                    print(f"result: {result['reward'], result['action'][0]}")

        self.changing = changing

        # 选择策略：比较各候选方案的正向期望 reward
        if not results:
            raise ValueError("No solution found!")
        else:

            # 1. 计算所有候选方案的正向 reward 
            pos_rewards = {k: -res['AV_reward_diff2max'] for k, res in results.items()}

            # ==========================================================
            # 【建立战术奖励阶梯】
            # ==========================================================
            hysteresis_bonus = 100.0  # 动作粘性分
            commit_tactical_bonus = 500.0  # 换道成功战术分 (必须大于迟滞分)

            for k in pos_rewards.keys():
                # A. 给予 Commit 最高的基础战术奖励，鼓励它只要安全就赶紧进去
                if 'commit' in str(k):
                    pos_rewards[k] += commit_tactical_bonus
                elif 'probe' in str(k):
                        # 可选：适当鼓励探头试探动作，打破僵局
                        pos_rewards[k] += 150.0

                # C. 如果是强制变道，直接枪毙 Keep 动作
                if self.force_flag and ('probe' not in str(k) and 'commit' not in str(k)) and (ego_state[3] < 10):
                    pos_rewards[k] -= 8000.0

            # # 【修改3：施加强制变道的 Routing 惩罚】
            # if self.force_flag:
            #     for k in pos_rewards.keys():
            #         # 识别出“保持当前车道”的动作 (可能是带 _ keep 后缀，或是纯数字的 curr_lane_idx)
            #         is_keep_action = str(curr_lane_idx) in str(k) and ('probe' not in str(k) and 'commit' not in str(k))
            #         if is_keep_action:
            #             # 给予极大的负向惩罚，迫使算法只要 Probe 或 Commit 没撞，就优先选它们
            #             pos_rewards[k] -= 8000.0  
                    

            # 2. 找到期望收益最大的方案
            best_key = max(pos_rewards, key=pos_rewards.get)
            best_pos_reward = pos_rewards[best_key]
            selected_result = results[best_key]
            selected_idx = selected_result['target_idx']
            
            # 3. 获取当前车道(基准)的结果，用于防抖判断
            keep_key = f"{curr_lane_idx}_keep"
            if keep_key in results:
                current_result = results[keep_key]
                current_a = current_result['action']
            elif curr_lane_idx in results: # 兼容正在换道时的旧逻辑(直接存的数字key)
                keep_key = curr_lane_idx
                current_result = results[curr_lane_idx]
                current_a = current_result['action']
            else:
                current_result = None
                current_a = [np.array([self.limits[self.type_mapping['AV']]['MAX_DECELERATION']] * len(self.t_list))]
            
            # 4. 动作防抖：判断收益最大的动作（探入/切入）是否在加速度上显著优于保持当前车道
            flag = selected_result['action'][0][0] - current_a[0][0] > self.path_diff_action_threshold
            self.change_memory = self.change_memory[-1:] + [flag]
            print(">>> 不满足防抖要求")
            # 如果没有连续满足阈值条件，则退回 current_result (即放弃探入或换道)
            if self.change_memory != [True]*len(self.change_memory) and current_result is not None:
                selected_result = current_result
                selected_idx = current_result['target_idx']
                best_key = keep_key
            
            # 只保留选中的结果
            results = {best_key: selected_result}
            
        best_result = selected_result
        # 打印当前实际选出的动作类型
        action_type = best_result.get('action_type', 'keep')
        if self.roslog:
            rospy.loginfo(f"[GAME模块]  >>> 贝叶斯决策胜出方案: [{best_key}], 动作类型: [{action_type}], 期望收益: {pos_rewards.get(best_key, 0.0):.2f}")
        else:
            print(f">>> 贝叶斯决策胜出方案: [{best_key}], 动作类型: [{action_type}], 期望收益: {pos_rewards.get(best_key, 0.0):.2f}")

        if action_type in ['probe', 'commit']:
            print(" ")

        best_traj = best_result['path']
        self.current_target_idx = best_result['target_idx']
        self.current_path = best_traj.copy()
        best_actions = best_result['action']
        others_trajs_dense = best_result['other_trajs_dense']
        av_follow = best_result['a_follow']

        # === AV粗轨迹生成 ===
        x, y, heading, v = ego_state[:4]
        best_traj = best_result['path']
        path_s = best_result['s']  # 原始路径的弧长序列
        av_ori_best_a = best_actions[0]
        
        # # 1222对低速&加速时的加速度度值进行修正 TODO 根据当前控制版本稳定性确定是否启用
        # if v < 5/3.6 and av_ori_best_a[0] >= 0:
        #     av_ori_best_a= np.array(av_ori_best_a) * 3
        
        # 1222判断是否使用AEB
        if self.if_use_aeb:
            aeb_flag, aeb_acc = self.check_aeb(ego_state, others_info)
            if aeb_flag:
                av_ori_best_a = np.array([min(av_ori_best_a[0], aeb_acc)] * len(av_ori_best_a))
                if self.roslog:
                    rospy.loginfo(f"[GAME模块]  启用AEB: {aeb_acc:.3f} m/s²")
                else:
                    print(f"AEB: {aeb_acc:.3f} m/s²")
        
        av_interp_best_a = np.interp(self.t_output_av, self.t_list, av_ori_best_a)
        
        # 纵向按加加速度平滑 TODO：有待改进
        if self.smooth_long_acc == True:
            if (av_interp_best_a[0] - ego_state[-2]) / (self.t_output_av[1] - self.t_output_av[0])> self.max_long_jerk:
                _, av_interp_best_a, _ = generate_acceleration_sequence(ego_state[-2], av_interp_best_a[0], self.t_output_av[-1], self.max_long_jerk, self.t_output_av[1] - self.t_output_av[0])
                av_interp_best_a = av_interp_best_a[1:]
        
        v_list = [v]
        v_safe_lists = [v]
        v_max_av = max(self.limits[self.type_mapping['AV']]['MAX_SPEED'], v)
        v_min_av = self.limits[self.type_mapping['AV']]['MAX_SPEED'] if self.decel_from_speedlimit == True else 0.0
        if self.smooth_cal_a:
            a_now = self.memory_cal_a
        else:
            a_now = ego_state[-2]
        max_jerk = 1
        
        if self.roslog and self.decel_from_speedlimit == True:
            rospy.loginfo(f"[GAME模块]  是否由于限速减速：{self.decel_from_speedlimit}， 原始加速度：{av_interp_best_a[0]}， 当前加速度：{a_now}， 平滑后加速度：{min(0, a_now - max_jerk)}")
        elif self.decel_from_speedlimit == True:
            print(f"[GAME模块]  是否由于限速减速：{self.decel_from_speedlimit}， 原始加速度：{av_interp_best_a[0]}，当前加速度：{a_now}， 平滑后加速度：{min(0, a_now - max_jerk)}")
        s_list = [0.0]
        dt_list = np.diff(np.insert(self.t_output_av, 0, 0))
        for i, dt in enumerate(dt_list):
            if self.decel_from_speedlimit == True:
                a = max(a_now - max_jerk, av_interp_best_a[i])
                a = min(0, a)
                av_interp_best_a[i] = a
                v_next = v_list[-1] + a * dt
                v_next = np.clip(v_next, v_min_av, v_max_av)
                v_safe_next = v_safe_lists[-1] + av_follow * dt
                if self.smooth_a:
                    if av_follow < 0:
                        v_safe_next = v_safe_lists[-1] + max(a_now - max_jerk, av_follow) * dt
                v_next = min(v_next, v_safe_next)
                v_next = np.clip(v_next, 0, v_max_av)

                v_safe_lists.append(v_safe_next)

            else:
                a = av_interp_best_a[i]
                if self.smooth_a:
                    if a < 0:
                        a = max(a_now - max_jerk, av_interp_best_a[i])
                av_interp_best_a[i] = a
                v_next = v_list[-1] + a * dt
                v_next = np.clip(v_next, v_min_av, v_max_av)

            s_next = s_list[-1] + v_list[-1] * dt + 0.5 * a * dt**2
            v_list.append(v_next)
            s_list.append(s_next)
        
        self.memory_cal_a = av_interp_best_a[0]

        # 对应路径上的坐标插值（支持对超出 path_s 的 s 值按末端两点线性外推）
        def interp_extrap(s_vals, path_s_arr, vals, is_angle=False):
            s_vals = np.asarray(s_vals)
            out = np.interp(s_vals, path_s_arr, vals)
            max_s = path_s_arr[-1]
            if np.any(s_vals > max_s):
                s1, s2 = path_s_arr[-2], path_s_arr[-1]
                v1, v2 = vals[-2], vals[-1]
                ds = s2 - s1 if (s2 - s1) != 0 else 1e-6
                mask = s_vals > max_s
                if is_angle:
                    diff = v2 - v1
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    slope = diff / ds
                    ext = v2 + slope * (s_vals[mask] - s2)
                    out[mask] = (ext + np.pi) % (2 * np.pi) - np.pi
                else:
                    slope = (v2 - v1) / ds
                    out[mask] = v2 + slope * (s_vals[mask] - s2)
            return out

        x_list = interp_extrap(s_list, path_s, best_traj[:, 0], is_angle=False)
        y_list = interp_extrap(s_list, path_s, best_traj[:, 1], is_angle=False)
        heading_list = interp_extrap(s_list, path_s, best_traj[:, 2], is_angle=True)
        if self.roslog:
            rospy.loginfo(f"[GAME模块] 博弈规划速度 v_list[1]: {v_list[1]}, v_list[-1]: {v_list[-1]}")
        else:
            print(f"[GAME模块] 博弈规划速度 v_list[1]: {v_list[1]}, v_list[-1]: {v_list[-1]}")

        av_traj = np.stack([x_list, y_list, heading_list, v_list, np.insert(self.t_output_av, 0, 0)], axis=1)
        
        # === HV预测轨迹生成 ===
        other_trajs = []
        players_ids = best_result['players_ids']
        for i, o_traj in enumerate(others_trajs_dense):
            if other_ids[i] not in players_ids:
                # 非交互对象，保持原始预测不变
                other_trajs.append(others_pred[i, :, :])
                continue
            x, y, heading, v,  = other_states[i][:4]
            length, width = others_pred[i][0][-2:]
            best_traj = o_traj
            path_s = best_result['other_s_dense'][i]  # 原始路径的弧长序列
            o_ori_best_a = best_actions[i + 1]
            o_interp_best_a = np.interp(self.t_output_other, self.t_list, o_ori_best_a)
            
            v_list = [v]
            v_max_i = max(self.limits[self.type_mapping[other_types[i]]]['MAX_SPEED'], v)
            s_list = [0.0]
            length_list = [length]
            width_list = [width]
            dt_list = np.diff(np.insert(self.t_output_other, 0, 0))
            for j, dt in enumerate(dt_list):
                a = o_interp_best_a[j]
                v_next = v_list[-1] + a * dt
                v_next = np.clip(v_next, 0.0, v_max_i)
                s_next = s_list[-1] + v_list[-1] * dt + 0.5 * a * dt**2
                v_list.append(v_next)
                s_list.append(s_next)
            
            # 对应路径上的坐标插值（支持对超出 path_s 的 s 值按末端两点线性外推）
            def interp_extrap(s_vals, path_s_arr, vals, is_angle=False):
                s_vals = np.asarray(s_vals)
                out = np.interp(s_vals, path_s_arr, vals)
                max_s = path_s_arr[-1]
                if np.any(s_vals > max_s):
                    s1, s2 = path_s_arr[-2], path_s_arr[-1]
                    v1, v2 = vals[-2], vals[-1]
                    ds = s2 - s1 if (s2 - s1) != 0 else 1e-6
                    mask = s_vals > max_s
                    if is_angle:
                        diff = v2 - v1
                        diff = (diff + np.pi) % (2 * np.pi) - np.pi
                        slope = diff / ds
                        ext = v2 + slope * (s_vals[mask] - s2)
                        out[mask] = (ext + np.pi) % (2 * np.pi) - np.pi
                    else:
                        slope = (v2 - v1) / ds
                        out[mask] = v2 + slope * (s_vals[mask] - s2)
                return out

            x_list = interp_extrap(s_list, path_s, best_traj[:, 0], is_angle=False)
            y_list = interp_extrap(s_list, path_s, best_traj[:, 1], is_angle=False)
            heading_list = interp_extrap(s_list, path_s, best_traj[:, 2], is_angle=True)
            
            o_traj = np.stack([x_list, y_list, heading_list, v_list, np.array(length_list*len(x_list)), np.array(width_list*len(x_list))], axis=1)
            other_trajs.append(o_traj)
        other_trajs = np.array(other_trajs)  if len(other_trajs) > 0 else None # 转换为NumPy数组
        
        # 换道状态判断
        if best_result['target_idx'] != curr_lane_idx and action_type != 'probe':
            self.changing = True
            # rospy.loginfo("判断处于换道状态.")
            mid_idx = len(lanes[best_result['target_idx']]) // 2
            self.current_target_idx_point = (lanes[best_result['target_idx']][mid_idx].x, lanes[best_result['target_idx']][mid_idx].y)
        lane_points = lanes[self.current_target_idx]
        ego_x, ego_y = ego_state[:2]
        distances = min([np.hypot(p.x - ego_x, p.y - ego_y) for p in lane_points])
        if distances < self.changing_finish_threshold:
            self.changing = False  
        
        # 中止换道状态判断
        if best_result['target_idx'] == curr_lane_idx and self.stopchanging == True:
            # rospy.loginfo("判断处于中止换道状态.")
            lane_points = lanes[self.current_target_idx]
            ego_x, ego_y = ego_state[:2]
            distances = min([np.hypot(p.x - ego_x, p.y - ego_y) for p in lane_points])
            if distances < self.changing_finish_threshold:
                self.stopchanging = False
        
        if self.roslog:
            rospy.loginfo(f"[GAME模块]  ---- GAME计算加速度: {av_interp_best_a[0]:.3f} m/s², 当前速度：{ego_state[3]}, 目标车道: {best_result['target_idx']}, 换道状态: {self.changing} ----")
            rospy.loginfo(f"[GAME模块]  ==== GAME总计算时长: {(time.time()-start_time)*1000:.5f} ms ====")
        else:
            print(f"---- GAME计算加速度: {av_interp_best_a[0]:.3f} m/s², 当前速度：{ego_state[3]}, 目标车道: {best_result['target_idx']}, 换道状态: {self.changing} ----")
            print(f"==== GAME计算时长: {(time.time()-t_copy)*1000:.5f} ms, 前处理时长：{(t_copy-t0)*1000:.5f} ms, 共评估博弈分支数: {self.evaluated_branches_count} 个 ====")
        if self.evaluated_branches_count > 1:
            print(" ")
        return av_traj, av_interp_best_a, other_trajs, self.current_target_idx

@njit
def _densify_trajectory(traj, num_points):
    T = traj.shape[0]
    width = traj[0,5]
    dense_len = (T-1)*num_points + 1
    traj_dense = np.empty((dense_len, 4), dtype=np.float64)
    idx = 0
    for i in range(T-1):
        x0, y0, heading0 = traj[i,0], traj[i,1], traj[i,2]
        x1, y1, heading1 = traj[i+1,0], traj[i+1,1], traj[i+1,2]
        width = traj[i,5]
        for t_idx in range(num_points):
            t = t_idx / num_points
            traj_dense[idx,0] = x0 + t*(x1-x0)
            traj_dense[idx,1] = y0 + t*(y1-y0)
            traj_dense[idx,2] = heading0 + t*(heading1-heading0)
            traj_dense[idx,3] = width
            idx += 1
    traj_dense[idx,0] = traj[T-1,0]
    traj_dense[idx,1] = traj[T-1,1]
    traj_dense[idx,2] = traj[T-1,2]
    traj_dense[idx,3] = traj[T-1,5]
    return traj_dense

@njit
def _compute_conf_s(ego_path, ego_s, hv_trajs_dense, hv_s_dense, new_width_list, safe_distance=1.0):
    N_ego = ego_path.shape[0]
    n_vehicles = hv_trajs_dense.shape[0]
    N_hv = hv_trajs_dense.shape[1]
    
    av_confpos = np.full(n_vehicles + 1, -1.0)
    hv_confpos_2av = np.full(n_vehicles + 1, -1.0)
    
    for j in prange(n_vehicles):
        for t in range(N_ego):
            ex, ey = ego_path[t]
            for k in range(N_hv):
                hx, hy, hheading, _ = hv_trajs_dense[j, k]
                w = new_width_list[j]
                dx = ex - hx
                dy = ey - hy
                dist = np.sqrt(dx * dx + dy * dy)
                if dist - w/2 < safe_distance and av_confpos[j + 1] < 0:
                    av_confpos[j + 1] = ego_s[t]
                    hv_confpos_2av[j + 1] = hv_s_dense[j, k]
    return av_confpos, hv_confpos_2av

@njit
def nearest_point_on_traj(point, traj):
    """
    计算给定点到轨迹的最短距离及最近点索引
    
    参数:
        point: np.ndarray [2,] 目标点 [x, y]
        traj: np.ndarray [N, 2] 轨迹点集 [[x1,y1], [x2,y2], ...]
    
    返回:
        min_dist: 最短距离
        nearest_idx: 最近点索引
        nearest_point: 最近点坐标 [x, y]
    """
    min_dist = np.inf
    nearest_idx = -1
    nearest_point = np.zeros(2)
    
    for i in range(traj.shape[0]):
        dx = point[0] - traj[i, 0]
        dy = point[1] - traj[i, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
            nearest_point[0] = traj[i, 0]
            nearest_point[1] = traj[i, 1]
    return min_dist, nearest_idx

# 为实现加加速度限制对输出加速度进行处理的辅助函数
def calculate_jerk_time(a0, a1, T, jerk):
    """
    计算jerk作用时间t
    
    参数:
    a0: 当前加速度
    a1: 目标匀加速度
    T: 总时域
    jerk: 最大加加速度
    
    返回:
    t: jerk作用时间
    """
    if abs(jerk) < 1e-10:
        # 如果jerk为0，直接返回T（整个时域都是匀加速）
        return T
    
    # 解一元二次方程: t × (T - t/2) = (a1 - a0) × T / jerk
    # 整理为: -0.5 × t² + T × t - (a1 - a0) × T / jerk = 0
    A = -0.5
    B = T
    C = -(a1 - a0) * T / jerk
    
    # 解二次方程
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        raise ValueError("无实数解，请检查参数")
    
    t1 = (-B + np.sqrt(discriminant)) / (2*A)
    t2 = (-B - np.sqrt(discriminant)) / (2*A)
    
    # 选择在[0, T]范围内的合理解
    valid_solutions = []
    for t in [t1, t2]:
        if 0 <= t <= T:
            valid_solutions.append(t)
    
    if not valid_solutions:
        raise ValueError(f"在[0, {T}]范围内无有效解，解为: t1={t1:.3f}, t2={t2:.3f}")
    
    # 如果有多个解，选择较小的那个（更平滑）
    t = min(valid_solutions)
    
    return t

def generate_acceleration_sequence(a0, a1, T, jerk, dt):
    """
    生成满足jerk约束的加速度序列
    
    参数:
    a0: 当前加速度
    a1: 目标匀加速度
    T: 总时域
    jerk: 最大加加速度
    dt: 时间间隔
    
    返回:
    time_array: 时间数组
    accel_array: 加速度数组
    t_jerk: jerk作用时间
    """
    # 计算jerk作用时间
    t_jerk = calculate_jerk_time(a0, a1, T, jerk)
    
    # 生成时间序列
    n_points = int(T / dt) + 1
    time_array = np.linspace(0, T, n_points)
    accel_array = np.zeros(n_points)
    
    for i, time in enumerate(time_array):
        if time <= t_jerk:
            # jerk阶段：加速度线性变化
            accel_array[i] = a0 + jerk * time
        else:
            # 匀加速阶段：加速度恒定
            accel_array[i] = a0 + jerk * t_jerk
    
    return time_array, accel_array, t_jerk

# 判断车辆当前位置与道路尽头的距离是不是小于阈值
def is_near_road_end(current_position, road_end_position, threshold):
    distance = np.linalg.norm(current_position - road_end_position)
    if distance < threshold:
        # rospy.loginfo(f"[GAME模块]  距离道路尽头还有 {distance:.2f} 米，小于阈值 {threshold:.2f} 米。")
        pass
    return distance < threshold

from dataclasses import dataclass
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

# 车道中心线加密
def _dense_refline(lanes):
    dense_lanes = {}
    for lane_id, lane_points in lanes.items():
        dense_lane_points = []
        for i in range(len(lane_points) - 1):
            p1 = lane_points[i]
            p2 = lane_points[i + 1]
            # 原点
            dense_lane_points.append(p1)
            # 插值点（均值）
            avg_point = Point(
                x=(p1.x + p2.x) / 2,
                y=(p1.y + p2.y) / 2,
                heading=(p1.heading + p2.heading) / 2,
                speed=(p1.speed + p2.speed) / 2,
                curvature=(p1.curvature + p2.curvature) / 2,
                to_left=(abs(p1.to_left) + abs(p2.to_left)) / 2,
                to_right=(abs(p1.to_right) + abs(p2.to_right)) / 2,
                point_type=p1.point_type,
                left_traverse_flag=p1.left_traverse_flag,
                right_traverse_flag=p1.right_traverse_flag,
            )
            dense_lane_points.append(avg_point)
        # 最后一个点
        dense_lane_points.append(lane_points[-1])
        dense_lanes[lane_id] = dense_lane_points
    return dense_lanes

def calculate_diagonal_projection(l, w, veh_heading, ref_heading):
    """
    计算矩形斜边在垂直于直线方向上的投影长度
    
    参数:
        l: 矩形长度
        w: 矩形宽度
        veh_heading: 矩形角度（弧度）
        ref_heading: 直线角度（弧度）
    
    返回:
        最大投影长度
    """
    # 对角线长度
    diagonal = math.sqrt(l**2 + w**2)
    
    # 对角线相对于矩形长边的角度
    alpha = math.atan2(w, l)
    
    # 两条对角线的方向
    theta1 = veh_heading + alpha      # 第一条对角线
    theta2 = veh_heading - alpha      # 第二条对角线
    
    # 计算与参考线的角度差
    delta1 = theta1 - ref_heading
    delta2 = theta2 - ref_heading
    
    # 计算投影长度
    proj1 = diagonal * abs(math.sin(delta1))
    proj2 = diagonal * abs(math.sin(delta2))
    
    # 返回最大投影
    return max(proj1, proj2)
