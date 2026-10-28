STYLE_PARAMS = {
    "Aggressive": {
        # 最大速度
        "v0+": 30
        / 3.6,  # TODO：1. 仅前处理; 2. 仅博弈IDM-v0; 3. 前处理+博弈IDM-v0 (可能会加速到很高)
        # 换道长度
        "L_min": 5.0,
        "change_t": 3.0,  # TODO：仿真测试-停止障碍物停车后是否可以绕行
        # 换道动作diff阈值
        "path_diff_action_threshold": 0.25,  # TODO：换道灵敏，需测试 低速静止跟停 + 高速静止跟停 + 慢速大车
        # 换道危险判断条件 # TODO：仿真测试 - 不同交通流密度下的 1. 汇入 + 2. 汇出 + 3. 低速前车
        "safechange_threshold": 1.0,  # 静止距离
        "k_force_lead_TTC": 1.0,  # 换道时与前车的TTC
        "k_force_follow_TTC": 5.0,  # 换道时与后车的TTC
        "k_free_lead_TTC": 1.0,  # 换道时与前车的TTC
        "k_free_follow_TTC": 5.0,  # 换道时与后车的TTC
        "k_free_TTC": 5.0,  # 换道时与前后车的TTC统一系数
        "k_force_bigcar": 5,  # 换道时大车对换道间距的影响系数
        "k_free_bigcar": 5,  # 换道时大车对换道间距的影响系数
        # 对HV跟车动作的预计折减
        "hv_idm_discount": 0.8,
        # 是否启用博弈结果用于换道安全检测
        "is_hv_gamepred": True,
        # 与前车/冲突前车相关参数
        "idm_T": 1.0,  # 安全跟车时距，不能调很低
        "idm_a": 2,  # idm加速度
        "v_control": 1.0,
    },
    "Normal": {
        # 最大速度
        "v0+": 30
        / 3.6,  # TODO：1. 仅前处理; 2. 仅博弈IDM-v0; 3. 前处理+博弈IDM-v0 (可能会加速到很高)
        # 换道长度
        "L_min": 5.0,
        "change_t": 3.5,  # TODO：仿真测试-停止障碍物停车后是否可以绕行
        # 换道动作diff阈值
        "path_diff_action_threshold": 0.25,  # TODO：换道灵敏，需测试 低速静止跟停 + 高速静止跟停 + 慢速大车
        # 换道危险判断条件 # TODO：仿真测试 - 不同交通流密度下的 1. 汇入 + 2. 汇出 + 3. 低速前车
        "safechange_threshold": 10.0,  # 静止距离
        "k_force_lead_TTC": 1.0,  # 换道时与前车的TTC
        "k_force_follow_TTC": 5.0,  # 换道时与后车的TTC
        "k_free_lead_TTC": 1.0,  # 换道时与前车的TTC
        "k_free_follow_TTC": 5.0,  # 换道时与后车的TTC
        "k_free_TTC": 5.0,  # 换道时与前后车的TTC统一系数
        "k_force_bigcar": 5,  # 换道时大车对换道间距的影响系数
        "k_free_bigcar": 5,  # 换道时大车对换道间距的影响系数
        # 对HV跟车动作的预计折减
        "hv_idm_discount": 0.3,
        # 是否启用博弈结果用于换道安全检测
        "is_hv_gamepred": True,
        # 与前车/冲突前车相关参数
        "idm_T": 1.0,  # 安全跟车时距，不能调很低
        "idm_a": 2,  # idm加速度
        "v_control": 1.0,
    },
    "Conservative": {
        # 最大速度
        "v0+": 20
        / 3.6,  # TODO：1. 仅前处理; 2. 仅博弈IDM-v0; 3. 前处理+博弈IDM-v0 (可能会加速到很高)
        # 换道长度
        "L_min": 5.0,
        "change_t": 4.0,  # TODO：仿真测试-停止障碍物停车后是否可以绕行
        # 换道动作diff阈值
        "path_diff_action_threshold": 0.4,  # TODO：换道灵敏，需测试 低速静止跟停 + 高速静止跟停 + 慢速大车
        # 换道危险判断条件 # TODO：仿真测试 - 不同交通流密度下的 1. 汇入 + 2. 汇出 + 3. 低速前车
        "safechange_threshold": 15.0,  # 静止距离
        "k_force_lead_TTC": 3.0,  # 换道时与前车的TTC
        "k_force_follow_TTC": 5.0,  # 换道时与后车的TTC
        "k_free_lead_TTC": 3.0,  # 换道时与前车的TTC
        "k_free_follow_TTC": 5.0,  # 换道时与后车的TTC
        "k_free_TTC": 5.0,  # 换道时与前后车的TTC统一系数
        "k_force_bigcar": 5,  # 换道时大车对换道间距的影响系数
        "k_free_bigcar": 5,  # 换道时大车对换道间距的影响系数
        # 对HV跟车动作的预计折减
        "hv_idm_discount": 0.1,
        # 是否启用博弈结果用于换道安全检测
        "is_hv_gamepred": False,
        # 与前车/冲突前车相关参数
        "idm_T": 1.0,  # 安全跟车时距，不能调很低
        "idm_a": 1.4,  # idm加速度
        "v_control": 0.8,
    },
}
