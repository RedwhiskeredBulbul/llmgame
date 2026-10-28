"""Evaluate vehicle behaviour classification while CutinEnv runs continuously."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path

import gymnasium as gym

import highway_env  # noqa: F401 - register local environments
from experiments.vehicle_types.collect import snapshot
from experiments.vehicle_types.evaluate import heuristic_prediction
from model.reasoning.worker import LLMReasoningWorker, runtime_setting
from experiments.vehicle_types.evaluation import (
    BEHAVIOR_LABELS,
    EvaluationRecord,
    calculate_metrics,
    prediction_from_probs,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "heuristic"), default="qwen")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--steps", type=int, default=60, help="Simulation steps per seed")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--temporal-alpha", type=float, default=0.7)
    parser.add_argument("--stable-count", type=int, default=3)
    parser.add_argument("--prototype-file", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results/vehicle_type_online"))
    return parser.parse_args()


def _record(request_id, vehicle_id, truth, probs, latency, error):
    padded = list(probs) if probs is not None else [None, None, None]
    record_error = error or ("missing_vehicle_prediction" if probs is None else None)
    return EvaluationRecord(
        case_id=request_id,
        vehicle_id=vehicle_id,
        true_label=truth,
        predicted_label=prediction_from_probs(probs),
        p_cautious=padded[0],
        p_normal=padded[1],
        p_aggressive=padded[2],
        latency_s=latency,
        error=record_error,
    )


def _mean_ci(values):
    if not values:
        return {"mean": 0.0, "95ci": [0.0, 0.0]}
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "95ci": [mean, mean]}
    margin = 1.96 * statistics.stdev(values) / (len(values) ** 0.5)
    return {"mean": mean, "95ci": [max(0.0, mean - margin), min(1.0, mean + margin)]}


def main() -> int:
    args = parse_args()
    if args.query_interval <= 0 or args.history_len <= 0:
        raise ValueError("query-interval and history-len must be positive")
    if args.backend == "qwen" and not runtime_setting("DASHSCOPE_API_KEY"):
        print("error: DASHSCOPE_API_KEY is required for --backend qwen", file=sys.stderr)
        return 2
    prototypes = None
    if args.prototype_file:
        prototypes = json.loads(args.prototype_file.read_text(encoding="utf-8"))

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    raw_records, smoothed_records, online_rows = [], [], []
    vehicle_states = {}
    seed_counts = defaultdict(lambda: {"raw_total": 0, "raw_correct": 0, "smooth_correct": 0})
    request_count = 0

    for seed in seeds:
        env = gym.make("CutinEnv")
        env.reset(seed=seed)
        worker = LLMReasoningWorker(
            history_len=args.history_len,
            dt=args.dt,
            model=args.model,
            behavior_prototypes=prototypes,
            temporal_alpha=args.temporal_alpha,
        )
        history = deque(maxlen=args.history_len)
        try:
            for step in range(args.steps):
                env.unwrapped.road.act()
                env.unwrapped.road.step(args.dt)
                frame = snapshot(env)
                labels = frame.pop("_labels")
                history.append(frame)
                observed_steps = step + 1
                if observed_steps < max(args.warmup_steps, args.history_len):
                    continue
                if (observed_steps - max(args.warmup_steps, args.history_len)) % args.query_interval:
                    continue

                prompt, target_ids = worker._generate_prompt(list(history))
                if not prompt:
                    continue
                request_id = f"seed{seed}-step{observed_steps:04d}"
                request_count += 1
                if args.backend == "qwen":
                    raw_beliefs = worker.get_response(prompt, target_ids) or {}
                    latency = worker.last_latency_s
                    error = worker.last_error
                else:
                    raw_beliefs = heuristic_prediction(list(history), target_ids)
                    latency = 0.0
                    error = None
                smooth_beliefs = worker.smooth_probabilities(raw_beliefs) or {}

                for vehicle_id in target_ids:
                    truth = labels.get(vehicle_id)
                    if truth not in BEHAVIOR_LABELS:
                        continue
                    raw_probs = raw_beliefs.get(vehicle_id)
                    smooth_probs = smooth_beliefs.get(vehicle_id)
                    raw_record = _record(request_id, vehicle_id, truth, raw_probs, latency, error)
                    smooth_record = _record(request_id, vehicle_id, truth, smooth_probs, latency, error)
                    raw_records.append(raw_record)
                    smoothed_records.append(smooth_record)

                    seed_counts[seed]["raw_total"] += 1
                    seed_counts[seed]["raw_correct"] += int(raw_record.correct)
                    seed_counts[seed]["smooth_correct"] += int(smooth_record.correct)
                    key = (seed, vehicle_id)
                    state = vehicle_states.setdefault(
                        key,
                        {
                            "seed": seed,
                            "vehicle_id": vehicle_id,
                            "true_label": truth,
                            "observations": 0,
                            "raw_correct": 0,
                            "smoothed_correct": 0,
                            "first_correct_time_s": None,
                            "stable_time_s": None,
                            "class_switches": 0,
                            "last_prediction": None,
                            "streak_label": None,
                            "streak_count": 0,
                            "final_probs": None,
                        },
                    )
                    state["observations"] += 1
                    state["raw_correct"] += int(raw_record.correct)
                    state["smoothed_correct"] += int(smooth_record.correct)
                    simulation_time = observed_steps * args.dt
                    if smooth_record.correct and state["first_correct_time_s"] is None:
                        state["first_correct_time_s"] = simulation_time
                    prediction = smooth_record.predicted_label
                    if state["last_prediction"] is not None and prediction != state["last_prediction"]:
                        state["class_switches"] += 1
                    state["last_prediction"] = prediction
                    if prediction == state["streak_label"]:
                        state["streak_count"] += 1
                    else:
                        state["streak_label"] = prediction
                        state["streak_count"] = 1
                    if state["streak_count"] >= args.stable_count and state["stable_time_s"] is None:
                        state["stable_time_s"] = simulation_time
                    state["final_probs"] = list(smooth_probs) if smooth_probs is not None else None

                    online_rows.append(
                        {
                            "seed": seed,
                            "simulation_step": observed_steps,
                            "simulation_time_s": simulation_time,
                            "request_id": request_id,
                            "vehicle_id": vehicle_id,
                            "true_label": truth,
                            "raw_prediction": raw_record.predicted_label,
                            "smoothed_prediction": smooth_record.predicted_label,
                            "raw_correct": raw_record.correct,
                            "smoothed_correct": smooth_record.correct,
                            "raw_probabilities": json.dumps(raw_probs),
                            "smoothed_probabilities": json.dumps(smooth_probs),
                            "latency_s": latency,
                            "error": error,
                        }
                    )
        finally:
            env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics = calculate_metrics(raw_records)
    smooth_metrics = calculate_metrics(smoothed_records)
    for metrics, name in ((raw_metrics, "online-raw"), (smooth_metrics, "online-ema")):
        metrics.update(
            {
                "backend": args.backend,
                "model": args.model or ("qwen-turbo" if args.backend == "qwen" else "motion-heuristic-baseline"),
                "dataset": f"continuous-CutinEnv-{name}",
                "seeds": seeds,
                "requests": request_count,
                "temporal_alpha": args.temporal_alpha,
            }
        )
    save_results(raw_records, raw_metrics, args.output_dir / "raw")
    save_results(smoothed_records, smooth_metrics, args.output_dir / "smoothed")

    final_records = []
    vehicle_rows = []
    for state in vehicle_states.values():
        final_probs = state.pop("final_probs")
        final_record = _record(
            f"seed{state['seed']}-final",
            state["vehicle_id"],
            state["true_label"],
            final_probs,
            None,
            None,
        )
        final_records.append(final_record)
        row = dict(state)
        row.update(
            {
                "final_prediction": final_record.predicted_label,
                "final_correct": final_record.correct,
                "final_probabilities": json.dumps(final_probs),
                "raw_accuracy": state["raw_correct"] / state["observations"],
                "smoothed_accuracy": state["smoothed_correct"] / state["observations"],
            }
        )
        vehicle_rows.append(row)
    final_metrics = calculate_metrics(final_records)
    final_metrics.update(
        {
            "backend": args.backend,
            "model": args.model or ("qwen-turbo" if args.backend == "qwen" else "motion-heuristic-baseline"),
            "dataset": "continuous-CutinEnv-final-vehicle",
            "seeds": seeds,
            "temporal_alpha": args.temporal_alpha,
        }
    )
    save_results(final_records, final_metrics, args.output_dir / "vehicle_final")

    with (args.output_dir / "online_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(online_rows[0]) if online_rows else ["request_id"])
        writer.writeheader()
        writer.writerows(online_rows)
    with (args.output_dir / "vehicle_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(vehicle_rows[0]) if vehicle_rows else ["vehicle_id"])
        writer.writeheader()
        writer.writerows(vehicle_rows)

    per_seed = {}
    for seed, counts in seed_counts.items():
        total = counts["raw_total"]
        per_seed[str(seed)] = {
            **counts,
            "raw_accuracy": counts["raw_correct"] / total if total else 0.0,
            "smoothed_accuracy": counts["smooth_correct"] / total if total else 0.0,
        }
    summary = {
        "configuration": vars(args) | {"output_dir": str(args.output_dir), "prototype_file": str(args.prototype_file)},
        "requests": request_count,
        "observation_predictions": len(raw_records),
        "unique_vehicles": len(final_records),
        "raw_metrics": raw_metrics,
        "smoothed_metrics": smooth_metrics,
        "final_vehicle_metrics": final_metrics,
        "per_seed": per_seed,
        "seed_level_raw_accuracy": _mean_ci([value["raw_accuracy"] for value in per_seed.values()]),
        "seed_level_smoothed_accuracy": _mean_ci([value["smoothed_accuracy"] for value in per_seed.values()]),
    }
    (args.output_dir / "online_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        f"requests={request_count} observations={len(raw_records)} vehicles={len(final_records)} "
        f"raw_accuracy={raw_metrics['accuracy']:.3%} ema_accuracy={smooth_metrics['accuracy']:.3%} "
        f"final_vehicle_accuracy={final_metrics['accuracy']:.3%}"
    )
    print(f"results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
