from __future__ import annotations

import numpy as np


def corners(vehicle):
    angle = vehicle.heading
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = np.array([[1,1], [-1,1], [-1,-1], [1,-1]]) * [vehicle.LENGTH/2, vehicle.WIDTH/2]
    return local @ rotation.T + vehicle.position


def rectangle_clearance(first, second):
    a, b = corners(first), corners(second)
    separated = False
    for polygon in (a, b):
        for edge in (polygon[1]-polygon[0], polygon[2]-polygon[1]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            separated |= bool(pa.max() < pb.min() or pb.max() < pa.min())
    if not separated:
        return 0.
    result = np.inf
    for points, segments in ((a,b),(b,a)):
        starts, ends = segments, np.roll(segments, -1, axis=0)
        edges = ends-starts
        for p in points:
            t = np.clip(np.sum((p-starts)*edges, axis=1)/np.sum(edges*edges, axis=1),0,1)
            result = min(result, float(np.linalg.norm(p-(starts+t[:,None]*edges),axis=1).min()))
    return result


def safety(env):
    ego = env.vehicle
    clearance, ttc = np.inf, np.inf
    forward = np.array([np.cos(ego.heading), np.sin(ego.heading)])
    lateral = np.array([-forward[1], forward[0]])
    for other in env.road.vehicles:
        if other is ego:
            continue
        clearance = min(clearance, rectangle_clearance(ego, other))
        delta = other.position-ego.position
        ds, dl = float(delta @ forward), float(delta @ lateral)
        relative_heading = other.heading-ego.heading
        half_long = ego.LENGTH/2 + abs(np.cos(relative_heading))*other.LENGTH/2 + abs(np.sin(relative_heading))*other.WIDTH/2
        half_lat = ego.WIDTH/2 + abs(np.sin(relative_heading))*other.LENGTH/2 + abs(np.cos(relative_heading))*other.WIDTH/2
        closing = float((ego.velocity-other.velocity) @ forward) * np.sign(ds)
        if abs(dl) < half_lat and closing > .1:
            ttc = min(ttc, max(0., abs(ds)-half_long)/closing)
    return {"clearance_m": float(clearance) if np.isfinite(clearance) else None,
            "ttc_s": float(ttc) if np.isfinite(ttc) else None,
            "collision": bool(ego.crashed)}


def summarize_steps(rows, status, horizon, completed_time):
    speeds = np.asarray([r['speed_after_ms'] for r in rows])
    accelerations = np.asarray([r['a_executed'] for r in rows])
    times = np.asarray([r['control_total_ms'] for r in rows])
    jerks = np.diff(accelerations)/.1
    gaps = [r['clearance_m'] for r in rows if r['clearance_m'] is not None]
    ttcs = [r['ttc_s'] for r in rows if r['ttc_s'] is not None]
    return dict(success=int(status=='completed'), collision=int(status=='collision'), steps=len(rows),
                simulation_duration_s=len(rows)*.1, completion_time_s=completed_time,
                restricted_task_time_s=completed_time if status=='completed' else horizon,
                average_speed_ms=float(speeds.mean()) if len(rows) else None,
                acceleration_variance=float(accelerations.var()) if len(rows) else None,
                jerk_rms=float(np.sqrt(np.mean(jerks**2))) if len(jerks) else None,
                jerk_abs_p95=float(np.percentile(abs(jerks),95)) if len(jerks) else None,
                min_clearance_m=min(gaps) if gaps else None,
                min_valid_ttc_s=min(ttcs) if ttcs else None,
                ttc_valid_fraction=len(ttcs)/len(rows) if rows else 0.,
                near_conflict=int(any(t<1.5 for t in ttcs)),
                waiting_time_s=.1*sum(s<.5 for s in speeds),
                probe_selected_steps=sum(r['action_type']=='probe' for r in rows),
                control_p95_ms=float(np.percentile(times,95)) if len(rows) else None,
                control_p99_ms=float(np.percentile(times,99)) if len(rows) else None,
                deadline_miss_rate=float(np.mean(times>100.)) if len(rows) else None)
