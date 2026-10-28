"""Candidate-isolated adapter around the existing path and game solver.

This is the v2 repaired controller, not a reproduction of the September runs.
"""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np
import casadi as ca

from Game.Game_solver import GameSolver
from utils._from_env import Point, get_hv_preds


def scene_data(env, cache=None):
    ego = env.vehicle
    lane_id = int(ego.lane_index[2])
    if cache is None:
        lanes = {}
        default_speed = 10. if env.kind == "merge" else 30.
        speed = float(getattr(env, "scene_speed_limit_ms", default_speed))
        for index, lane in enumerate(env.road.network.graph["a"]["b"]):
            lanes[index] = [Point(float(x), float(index * 4.), 0., speed=speed)
                            for x in np.arange(0., lane.length + .1, 1.)]
    else:
        lanes = cache
    state, pred, ids, types = get_hv_preds(env.road.vehicles, ego.id)
    # Lane topology priority; the simulator generates no future information here.
    selected = []
    for index in lanes:
        eligible = [j for j in range(len(ids)) if abs(state[j, 1] - 4. * index) <= 2.5]
        for front in (True, False):
            subset = [j for j in eligible if (state[j, 0] >= ego.position[0]) == front]
            if subset:
                j = min(subset, key=lambda z: abs(state[z, 0] - ego.position[0]))
                if j not in selected:
                    selected.append(j)
    nearest = sorted(range(len(ids)), key=lambda j: np.linalg.norm(state[j, :2] - ego.position))
    selected += [j for j in nearest if j not in selected and np.linalg.norm(state[j, :2] - ego.position) < 100.]
    selected = selected[:5]
    other_ids = [ids[j] for j in selected]
    others = {
        "ids": other_ids, "states": state[selected], "pred": pred[selected],
        "types": [types[j] for j in selected],
        "laneids": [int(round(state[j, 1] / 4.)) for j in selected],
        "frenet_relative_coords": [(float(state[j, 0]-ego.position[0]), float(state[j, 1]-ego.position[1])) for j in selected],
    }
    current = np.array([*ego.position, ego.heading, ego.speed, ego.action['acceleration'], ego.action['steering']], dtype=float)
    target = [0] if env.kind == "merge" else list(lanes)
    # Optional observation phase for the type-observable protocols: the lane
    # change is not offered before the ego reaches the scene's opening x, so
    # the LLM can observe the key vehicle before the decision boundary.  Merge
    # scenes keep using ``merge_open_x``; other scenes may set
    # ``decision_open_x``.  Old runs set neither attribute and keep the
    # original behaviour.  Non-merge scenes may also constrain the target to
    # one ``required_target_lane`` after the opening x.
    open_x = getattr(env, "merge_open_x", None)
    if open_x is None:
        open_x = getattr(env, "decision_open_x", None)
    if open_x is not None and ego.position[0] < float(open_x):
        target = [lane_id]
    elif env.kind != "merge":
        required_lane = getattr(env, "required_target_lane", None)
        if required_lane is not None:
            target = [int(required_lane)]
    return SimpleNamespace(ego_state=current, others_info=others, lanes=lanes, curr_lane_idx=lane_id,
                           near_target_idxes=target, auto_state=1,
                           route_end_x=getattr(env, 'route_end_x', None))


class ExperimentGameSolver(GameSolver):
    def __init__(self, config, response_mode="interactive", enable_probe=True):
        params = dict(config)
        params["response_mode"] = response_mode
        super().__init__(AV_style="Normal", roslog=False, decision_config=params)
        self.enable_probe = enable_probe
        self.branch_rows = []
        self.selected_action = "keep"
        self.evaluated_branches_count = 0
        self.DFpath = None
        # Shared across all v2 variants: no giant action-specific tactical bonuses.
        self.route_progress_bonus = float(params.get("route_progress_bonus", 2.))
        self.switch_penalty = float(params.get("switch_penalty", .1))
        self.v0 = 10.
        pair = ca.MX.sym('pair',6)
        ttc = self._GameSolver__reward_TTC(pair[0],pair[4],pair[1],pair[2],pair[5],pair[3])
        ttcp = self._GameSolver__reward_delta_TTCP(pair[0],pair[4],pair[1],pair[2],pair[5],pair[3])
        self._numeric_pair_reward = ca.Function('numeric_pair_reward',[pair],[ca.fmax(ttc,ttcp)])
        self._numeric_pair_reward_mapped = self._numeric_pair_reward.map(self.n_time)

    def _cal_discrete_reward(self,i,a,a_prev,v_prev,a0_list,v0_list,players_types,players_ids,av_confpos_array,hv_confpos_2av_array):
        # Same scalar safety/efficiency/comfort formula as the parent, compiled
        # once instead of constructing hundreds of CasADi expression graphs/step.
        av_s, hv_s = 0.,0.
        av_v,hv_v = float(v0_list[0]),float(v0_list[i])
        safety_reward=efficiency_reward=comfort_reward=0.
        pair_inputs=[]
        for t in range(self.n_time):
            dt=self.t_list[t]-(self.t_list[t-1] if t else 0.)
            av_a, hv_a = float(a_prev[0][t]),float(a[t])
            av_dt=min(dt,av_v/-av_a) if av_a<0 else dt
            hv_dt=min(dt,hv_v/-hv_a) if hv_a<0 else dt
            av_s+=av_v*av_dt+.5*av_a*av_dt**2
            hv_s+=hv_v*hv_dt+.5*hv_a*hv_dt**2
            av_v=max(0.,av_v+av_a*dt)
            hv_v=max(0.,hv_v+hv_a*dt)
            pair=[hv_s,hv_v,av_s,av_v,hv_confpos_2av_array[i],av_confpos_array[i]]
            weight=np.exp(-self.decay_rate*self.t_list[t])
            pair_inputs.append(pair)
            efficiency_reward+=hv_v*weight
            old_a=float(a[t-1]) if t else float(a0_list[i])
            comfort_reward+=-np.sqrt((hv_a-old_a)**2+1e-4)*weight
        pair_rewards=np.asarray(self._numeric_pair_reward_mapped(np.asarray(pair_inputs).T)).reshape(-1)
        safety_reward=float(np.dot(pair_rewards,np.exp(-self.decay_rate*self.t_list[:self.n_time])))
        weights=self.weight_dict[players_ids[i]]
        return -(safety_reward*weights[0]+efficiency_reward*weights[1]+comfort_reward*weights[2])

    def _get_sub_reward_hv(self,*args,**kwargs):
        # Legacy style-estimation diagnostics are not read by the v2 controller.
        return {}

    def decide(self, data, beliefs, reverse_candidates=False):
        self.inject_llm_beliefs(beliefs)
        self._update_beliefs(data.others_info)
        self.solve_diagnostics = []
        self.evaluated_branches_count = 0
        self.branch_rows = []
        self.force_flag = data.curr_lane_idx not in data.near_target_idxes
        self.others_laneid = data.others_info['laneids']
        self.DFpath = None
        limit = data.lanes[data.curr_lane_idx][0].speed
        self.limits[0]['MAX_SPEED'] = limit
        self.v0 = limit
        # Local map shared by all candidates, never modified in-place.
        x = data.ego_state[0]
        lanes = {idx: pts[max(0, int(x)-20):min(len(pts), int(x)+260)] for idx, pts in data.lanes.items()}
        candidates = [("keep", data.curr_lane_idx, 1.)]
        if self.changing and self.current_target_idx in lanes:
            candidates = [("commit", self.current_target_idx, 1.)]
        else:
            for target in data.near_target_idxes:
                if target == data.curr_lane_idx:
                    continue
                if self.enable_probe and data.ego_state[3] < 10.:
                    candidates.append(("probe", target, .3))
                candidates.append(("commit", target, 1.))
        if reverse_candidates:
            candidates.reverse()
        memory_keys = ["changing", "stopchanging", "current_path", "current_target_idx", "current_target_idx_point",
                       "change_memory", "stopchange_memory_1", "stopchange_memory_2", "waiting", "decel_from_speedlimit",
                       "weight_dict", "style_reward_dict", "old_style_reward_dict"]
        before = {key: copy.deepcopy(getattr(self, key)) for key in memory_keys}
        results = []
        other = data.others_info
        for action_type, target, ratio in candidates:
            for key, value in before.items():
                setattr(self, key, copy.deepcopy(value))
            diagnostic_start = len(self.solve_diagnostics)
            started = time.perf_counter()
            result = self.solve_for_one_target(target, data.ego_state, lanes, data.curr_lane_idx,
                       other['pred'] if len(other['ids']) else None,
                       other['states'], other['ids'], other['types'],
                       [float(p[0, 4]) for p in other['pred']], lat_ratio=ratio)
            row = {"action_type": action_type, "target": target, "accepted": result is not None,
                   "evaluation_ms": (time.perf_counter()-started)*1000.,
                   "solves": copy.deepcopy(self.solve_diagnostics[diagnostic_start:])}
            if result is not None:
                # The legacy collision penalty is added to diff2max (cost), but
                # subtracted from reward (also a cost). Use the consistent channel.
                raw_score = -float(result['AV_reward_diff2max'])
                score = raw_score
                if self.force_flag:
                    route_end = getattr(data, 'route_end_x', None)
                    route_end = data.lanes[data.curr_lane_idx][-1].x if route_end is None else float(route_end)
                    remaining = max(1., route_end-x)
                    urgency = min(3., 30. / remaining)
                    score += self.route_progress_bonus * urgency * (0.3 if action_type == 'probe' else 1. if action_type == 'commit' else 0.)
                if action_type != self.selected_action:
                    score -= self.switch_penalty
                row.update(raw_score=raw_score, score=score, acceleration=float(result['action'][0][0]))
                results.append((score, action_type, target, result))
            self.branch_rows.append(row)
        for key, value in before.items():
            setattr(self, key, value)
        if not results:
            raise RuntimeError("No feasible candidate, including Keep")
        # Deterministic tie rule, independent of candidate evaluation order.
        _, action_type, target, best = max(results, key=lambda row: (row[0], {'keep': 2, 'probe': 1, 'commit': 0}[row[1]], -row[2]))
        self.selected_action = action_type
        self.changing = action_type == 'commit' and abs(data.ego_state[1]-4.*target) > .5
        self.current_target_idx = target
        self.current_path = best['path'].copy()
        self.current_target_idx_point = (float(best['path'][-1, 0]), float(best['path'][-1, 1]))
        a = np.asarray(best['action'][0], dtype=float).reshape(-1)
        times = np.r_[0., self.t_output_av]
        acceleration = np.r_[a[0], np.interp(self.t_output_av, self.t_list, a)]
        if self.force_flag and action_type == 'keep':
            # Road-end stopping constraint shared by all variants.
            distance = max(.1, data.lanes[data.curr_lane_idx][-1].x-x-5.)
            acceleration = np.minimum(acceleration, -data.ego_state[3]**2/(2.*distance))
        acceleration = np.clip(acceleration, self.limits[0]['MAX_DECELERATION'], self.limits[0]['MAX_ACCELERATION'])
        speeds, distances = [float(data.ego_state[3])], [0.]
        for j in range(1, len(times)):
            dt, aa, vv = times[j]-times[j-1], acceleration[j-1], speeds[-1]
            active = min(dt, vv/-aa) if aa < 0 else dt
            distances.append(distances[-1] + vv*active+.5*aa*active**2)
            speeds.append(max(0., min(limit, vv+aa*dt)))
        path, arc = best['path'], best['s']
        traj = np.column_stack([np.interp(distances, arc, path[:, 0]), np.interp(distances, arc, path[:, 1]),
                                np.interp(distances, arc, np.unwrap(path[:, 2])), speeds, times])
        return traj, acceleration, target
