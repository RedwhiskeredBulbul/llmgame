"""Fit scene-specific behaviour fingerprints from labelled training seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from model.reasoning.worker import LLMReasoningWorker
from experiments.vehicle_types.evaluation import BEHAVIOR_LABELS, load_cases_jsonl


FEATURES = (
    "speed_mean",
    "speed_end",
    "speed_slope",
    "acceleration_mean",
    "headway_min",
    "ttc_min",
)


def fit(cases, history_len: int, dt: float) -> dict:
    worker = LLMReasoningWorker(history_len=history_len, dt=dt)
    values = {label: {feature: [] for feature in FEATURES} for label in BEHAVIOR_LABELS}
    supports = {label: 0 for label in BEHAVIOR_LABELS}
    seeds = set()
    for case in cases:
        if "seed" in case:
            seeds.add(case["seed"])
        for vehicle_id, label in case["ground_truth"].items():
            if label not in values:
                continue
            summary = worker._summarize_vehicle_history(case["history"], int(vehicle_id))
            if not summary:
                continue
            supports[label] += 1
            for feature in FEATURES:
                value = summary.get(feature)
                if value is not None and math.isfinite(value):
                    values[label][feature].append(value)

    classes = {}
    for label in BEHAVIOR_LABELS:
        classes[label] = {}
        for feature in FEATURES:
            observed = values[label][feature]
            classes[label][feature] = {
                "mean": statistics.fmean(observed) if observed else None,
                "std": statistics.stdev(observed) if len(observed) > 1 else 0.0,
                "count": len(observed),
            }
    return {
        "metadata": {
            "history_len": history_len,
            "dt": dt,
            "seeds": sorted(seeds),
            "supports": supports,
            "note": "Fitted only from the labelled training dataset; do not fit on test seeds.",
        },
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("simulation_results/vehicle_type_prototypes.json"))
    args = parser.parse_args()
    result = fit(load_cases_jsonl(args.dataset), args.history_len, args.dt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metadata"], ensure_ascii=False))
    print(f"prototypes: {args.output.resolve()}")


if __name__ == "__main__":
    main()
