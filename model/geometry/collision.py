# 对AV结果进行插值计算
import numpy as np
from model.settings import *


# === AV轨迹插值 ===
def interp_av_traj(path, s, av_a, ego_state, t_list):

    t_output_av = np.arange(0.0792, 8.0792, 0.0792)  # 输出的规划时间

    t_output_av = t_output_av  # 截断到与av_a长度一致

    x, y, heading, v = ego_state[:4]
    best_traj = path
    path_s = s  # 原始路径的弧长序列
    av_ori_best_a = av_a

    av_interp_best_a = np.interp(t_output_av, t_list, av_ori_best_a)

    v_list = [v]
    s_list = [0.0]
    dt_list = np.diff(np.insert(t_output_av, 0, 0))
    for i, dt in enumerate(dt_list):
        a = av_interp_best_a[i]
        v_next = max(0.0, v_list[-1] + a * dt)
        s_next = s_list[-1] + v_list[-1] * dt + 0.5 * a * dt**2
        v_list.append(v_next)
        s_list.append(s_next)

    # 对应路径上的坐标插值
    x_list = np.interp(s_list, path_s, best_traj[:, 0])
    y_list = np.interp(s_list, path_s, best_traj[:, 1])
    heading_list = np.interp(s_list, path_s, best_traj[:, 2])

    av_traj = np.stack([x_list, y_list, heading_list, v_list], axis=1)
    return av_traj


def interp_other_preds(others_pred):
    t_input = np.arange(0.0, 8.5, 0.5)
    t_output_ov = np.arange(0.0792, 8.0792, 0.0792)
    t_output_ov = np.insert(t_output_ov, 0, 0)
    n_other, N, dim = others_pred.shape
    # 输出 shape: (n_other, len(t_output_ov), dim)
    other_preds_interp = np.empty((n_other, len(t_output_ov), dim))
    for i in range(n_other):
        for j in range(dim):
            other_preds_interp[i, :, j] = np.interp(
                t_output_ov, t_input, others_pred[i, :, j]
            )
    return other_preds_interp


def get_rectangle_vertices(x, y, heading, length, width):
    """
    输入：车辆中心(x, y)，heading（弧度），length，width
    输出：四个顶点坐标，顺序为左下、右下、右上、左上
    """
    # 车辆中心到各顶点的相对坐标（未旋转）
    dx = length / 2
    dy = width / 2
    corners = np.array(
        [
            [-dx, -dy],  # 左下
            [dx, -dy],  # 右下
            [dx, dy],  # 右上
            [-dx, dy],  # 左上
        ]
    )
    # 构造旋转矩阵
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    # 旋转并平移
    rotated = corners @ R.T
    vertices = rotated + np.array([x, y])
    return vertices  # shape: (4, 2)


# 矩形-椭圆碰撞检测函数
# @nb.njit(cache=True)
def rect_intersects_any_ellipse(ll, rl, ru, lu, ox, oy, a, b, otheta=None):
    ox = np.asarray(ox, dtype=np.float32)
    n = ox.shape[0]
    if otheta is None:
        otheta = np.zeros(n)
    for i in range(n):
        cx = ox[i]
        cy = oy[i]
        ai = a[i]
        bi = b[i]
        th = otheta[i]
        if ai < 1e-9:
            ai = 1e-9
        if bi < 1e-9:
            bi = 1e-9
        # 矩形四个顶点（世界坐标）
        rx0 = ll[0]
        ry0 = ll[1]
        rx1 = rl[0]
        ry1 = rl[1]
        rx2 = ru[0]
        ry2 = ru[1]
        rx3 = lu[0]
        ry3 = lu[1]
        # 平移至椭圆中心
        rx0 -= cx
        ry0 -= cy
        rx1 -= cx
        ry1 -= cy
        rx2 -= cx
        ry2 -= cy
        rx3 -= cx
        ry3 -= cy
        # 旋转到椭圆主轴坐标系（绕原点旋转 -th）
        cth = np.cos(-th)
        sth = np.sin(-th)

        def rot(px, py, c, s):
            return px * c - py * s, px * s + py * c

        x0, y0 = rot(rx0, ry0, cth, sth)
        x1, y1 = rot(rx1, ry1, cth, sth)
        x2, y2 = rot(rx2, ry2, cth, sth)
        x3, y3 = rot(rx3, ry3, cth, sth)
        # 顶点在椭圆内判定
        if (x0 / ai) * (x0 / ai) + (y0 / bi) * (y0 / bi) <= 1.0:
            return True
        if (x1 / ai) * (x1 / ai) + (y1 / bi) * (y1 / bi) <= 1.0:
            return True
        if (x2 / ai) * (x2 / ai) + (y2 / bi) * (y2 / bi) <= 1.0:
            return True
        if (x3 / ai) * (x3 / ai) + (y3 / bi) * (y3 / bi) <= 1.0:
            return True

    return False


# === 路径与障碍物相交 ===
def path_collision_detect(path, s, av_a, ego_state, t_list, others_pred, margin=0.0):

    c = np.sqrt(VEHICLE_LENGTH**2 + VEHICLE_WIDTH**2) / 2.0  # 车身中心到轴线距离
    a = VEHICLE_LENGTH * 0.5 + c / 2
    b = VEHICLE_WIDTH * 0.5 + c / 2

    av_traj = interp_av_traj(path, s, av_a, ego_state, t_list)
    others_pred = interp_other_preds(others_pred)  # (n_other, len(t_output_ov), 6)
    collision = False
    for t in range(len(av_traj)):
        x, y, heading, v = av_traj[t]
        x_min = x - VEHICLE_LENGTH * 0.5
        x_max = x + VEHICLE_LENGTH * 0.5
        y_min = y - VEHICLE_WIDTH * 0.5
        y_max = y + VEHICLE_WIDTH * 0.5
        cx_arr = []
        cy_arr = []
        a_arr = []
        b_arr = []
        theta_arr = []
        for i in range(len(others_pred)):
            ox, oy, oh, _, o_l, o_w = others_pred[i][t]  # TODO
            c = np.sqrt(o_l**2 + o_w**2) / 2.0  # 障碍物中心到轴线距离
            # Optional physical safety buffer (metres) on both ellipse axes.
            # The default margin=0.0 reproduces the historical geometry.
            a = (o_l / 2 + c / 2) / 2 + margin
            b = (o_w / 2 + c / 2) / 2 + margin
            h = oh  # + np.pi/2
            cx_arr.append(ox)
            cy_arr.append(oy)
            a_arr.append(a)
            b_arr.append(b)
            theta_arr.append(h)
        theta_arr = np.asarray(theta_arr)
        ll, rl, ru, lu = get_rectangle_vertices(
            x, y, heading, VEHICLE_LENGTH, VEHICLE_WIDTH
        )
        t_coll = rect_intersects_any_ellipse(
            ll, rl, ru, lu, cx_arr, cy_arr, a_arr, b_arr, theta_arr
        )

        # if 1:

        if t_coll:
            collision = True
            break

    return collision
