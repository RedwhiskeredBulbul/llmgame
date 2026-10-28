"""Collect labelled trajectory windows from the CutinEnv simulator."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gymnasium as gym

import highway_env  # noqa: F401 - registers local environments
from experiments.vehicle_types.evaluation import save_cases_jsonl, vehicle_ground_truth


def snapshot(env) -> dict:
    unwrapped = env.unwrapped
    ego = unwrapped.controlled_vehicles[0]
    cos_h, sin_h = math.cos(ego.heading), math.sin(ego.heading)
    vehicles = []
    for vehicle in unwrapped.road.vehicles:
        if vehicle is ego:
            continue
        label = vehicle_ground_truth(vehicle)
        if label is None:
            continue
        dx = float(vehicle.position[0] - ego.position[0])
        dy = float(vehicle.position[1] - ego.position[1])
        ds = dx * cos_h + dy * sin_h
        dl = -dx * sin_h + dy * cos_h
        vehicles.append((math.hypot(ds, dl), vehicle, ds, dl, label))
    vehicles.sort(key=lambda item: item[0])

    ids, states, coords, lanes, labels = [], [], [], [], {}
    for _, vehicle, ds, dl, label in vehicles:
        acceleration = float(getattr(vehicle, "action", {}).get("acceleration", 0.0))
        ids.append(int(vehicle.id))
        states.append([
            float(vehicle.position[0]),
            float(vehicle.position[1]),
            float(vehicle.heading),
            float(vehicle.speed),
            acceleration,
            0.0,
        ])
        coords.append([ds, dl])
        lanes.append(int(vehicle.lane_index[2]))
        labels[int(vehicle.id)] = label
    return {
        "ego_state": [
            float(ego.position[0]),
            float(ego.position[1]),
            float(ego.heading),
            float(ego.speed),
            float(getattr(ego, "action", {}).get("acceleration", 0.0)),
            float(getattr(ego, "action", {}).get("steering", 0.0)),
        ],
        "curr_lane_idx": int(ego.lane_index[2]),
        "others_info": {
            "ids": ids,
            "states": states,
            "frenet_relative_coords": coords,
            "laneids": lanes,
        },
        "_labels": labels,
    }


def collect_cases(seeds, windows_per_seed: int, history_len: int, dt: float) -> list:
    cases = []
    for seed in seeds:
        env = gym.make("CutinEnv")
        env.reset(seed=seed)
        try:
            for window_index in range(windows_per_seed):
                history = []
                for _ in range(history_len):
                    # Advance the traffic model without invoking the expensive game planner.
                    env.unwrapped.road.act()
                    env.unwrapped.road.step(dt)
                    frame = snapshot(env)
                    labels = frame.pop("_labels")
                    history.append(frame)
                latest_ids = history[-1]["others_info"]["ids"][:3]
                truth = {vehicle_id: labels[vehicle_id] for vehicle_id in latest_ids if vehicle_id in labels}
                if truth:
                    cases.append(
                        {
                            "case_id": f"CutinEnv-seed{seed}-window{window_index}",
                            "seed": seed,
                            "history": history,
                            "ground_truth": truth,
                        }
                    )
        finally:
            env.close()
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--windows-per-seed", type=int, default=1)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("results/vehicle_type_dataset_cutin.jsonl"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    cases = collect_cases(seeds, args.windows_per_seed, args.history_len, args.dt)
    save_cases_jsonl(cases, args.output)
    labels = {"Cautious": 0, "Normal": 0, "Aggressive": 0}
    for case in cases:
        for label in case["ground_truth"].values():
            labels[label] += 1
    print(f"cases={len(cases)} vehicle_samples={sum(labels.values())} labels={labels}")
    print(f"dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
