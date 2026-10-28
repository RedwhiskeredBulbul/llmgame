"""Convert game trajectories to the simulator's trajectory/action format."""

import numpy as np


def build_action(trajectory, acceleration, ego_state, controller):
    count = len(trajectory)
    formatted = np.zeros((count, 6))
    formatted[:, :4] = trajectory[:, [0, 1, 3, 2]]
    available = min(len(acceleration), count)
    formatted[:available, 4] = acceleration[:available]
    if available < count:
        formatted[available:, 4] = acceleration[-1]
    formatted[:, 5] = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))
    ]
    x, y, heading, speed = ego_state[:4]
    controller.route = formatted[:, :2]
    steering = controller.act_stanley([x, y, speed, heading], plan_idx=0)
    controls = np.column_stack((formatted[:, 4], np.full(count, steering)))
    return {"trajectory": formatted, "action": controls.tolist()}
