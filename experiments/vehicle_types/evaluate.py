"""Run a reproducible vehicle behaviour-type classification experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from model.reasoning.worker import LLMReasoningWorker
from experiments.vehicle_types.evaluation import (
    BEHAVIOR_LABELS,
    EvaluationRecord,
    calculate_metrics,
    load_cases_jsonl,
    prediction_from_probs,
    save_cases_jsonl,
    save_results,
    synthetic_cases,
)


def _softmax(values):
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


def heuristic_prediction(history, target_ids):
    """Non-LLM baseline based only on observed motion features."""
    result = {}
    latest = history[-1]
    ego_speed = float(latest["ego_state"][3])
    for vehicle_id in target_ids:
        samples = []
        for frame in history:
            ids = frame["others_info"]["ids"]
            if vehicle_id in ids:
                idx = ids.index(vehicle_id)
                state = frame["others_info"]["states"][idx]
                ds = frame["others_info"]["frenet_relative_coords"][idx][0]
                samples.append((float(state[3]), float(ds)))
        dv = samples[-1][0] - ego_speed
        acceleration = (samples[-1][0] - samples[0][0]) / max(0.1, 0.5 * (len(samples) - 1))
        gap = abs(samples[-1][1])
        # Fixed centres derived from the textual behaviour definitions, not labels.
        cautious_score = -((dv + 1.8) / 1.6) ** 2 - ((acceleration + 0.25) / 0.55) ** 2 - ((gap - 28) / 16) ** 2
        normal_score = -(dv / 1.6) ** 2 - (acceleration / 0.55) ** 2 - ((gap - 23) / 16) ** 2
        aggressive_score = -((dv - 1.8) / 1.6) ** 2 - ((acceleration - 0.25) / 0.55) ** 2 - ((gap - 18) / 16) ** 2
        result[vehicle_id] = _softmax([cautious_score, normal_score, aggressive_score])
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "heuristic"), default="qwen")
    parser.add_argument("--dataset", type=Path, help="JSONL cases; omit to generate a seeded balanced smoke-test set")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.0,
        help="EMA weight for the previous probability of the same seed/vehicle (0 disables fusion)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--prototype-file", type=Path, help="Scene-calibrated class fingerprints fitted on separate seeds")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results/vehicle_type_eval"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prototypes = None
    if args.prototype_file:
        with args.prototype_file.open("r", encoding="utf-8") as handle:
            prototypes = json.load(handle)
    worker = LLMReasoningWorker(
        history_len=args.history_len,
        dt=args.dt,
        model=args.model,
        behavior_prototypes=prototypes,
    )
    if args.backend == "qwen" and not worker.api_key:
        print("error: DASHSCOPE_API_KEY is required for --backend qwen", file=sys.stderr)
        return 2

    if args.dataset:
        cases = load_cases_jsonl(args.dataset)
    else:
        cases = synthetic_cases(args.samples_per_class, args.history_len, args.dt, args.seed)
        save_cases_jsonl(cases, args.output_dir / "dataset.jsonl")

    records = []
    temporal_beliefs = {}
    for case_index, case in enumerate(cases, 1):
        prompt, target_ids = worker._generate_prompt(case["history"])
        if not prompt:
            continue
        if args.backend == "qwen":
            beliefs = worker.get_response(prompt, target_ids) or {}
            latency = worker.last_latency_s
            error = worker.last_error
        else:
            beliefs = heuristic_prediction(case["history"], target_ids)
            latency = 0.0
            error = None

        for vehicle_id in target_ids:
            true_label = case["ground_truth"].get(vehicle_id)
            if true_label not in BEHAVIOR_LABELS:
                continue
            probs = beliefs.get(vehicle_id)
            if probs is not None and args.temporal_alpha > 0:
                alpha = min(max(args.temporal_alpha, 0.0), 1.0)
                sequence_id = case.get("seed", case.get("case_id", case_index))
                temporal_key = (sequence_id, vehicle_id)
                previous = temporal_beliefs.get(temporal_key)
                if previous is not None:
                    probs = [alpha * old + (1 - alpha) * new for old, new in zip(previous, probs)]
                temporal_beliefs[temporal_key] = list(probs)
            predicted = prediction_from_probs(probs)
            padded = list(probs) if probs is not None else [None, None, None]
            records.append(
                EvaluationRecord(
                    case_id=str(case.get("case_id", case_index)),
                    vehicle_id=vehicle_id,
                    true_label=true_label,
                    predicted_label=predicted,
                    p_cautious=padded[0],
                    p_normal=padded[1],
                    p_aggressive=padded[2],
                    latency_s=latency,
                    error=error if predicted is None else None,
                )
            )

    metrics = calculate_metrics(records)
    metrics["backend"] = args.backend
    metrics["model"] = worker.model if args.backend == "qwen" else "motion-heuristic-baseline"
    metrics["dataset"] = str(args.dataset) if args.dataset else "synthetic-balanced-smoke-test"
    metrics["seed"] = args.seed
    metrics["temporal_alpha"] = args.temporal_alpha
    metrics["prototype_file"] = str(args.prototype_file) if args.prototype_file else None
    save_results(records, metrics, args.output_dir)
    print(f"backend={metrics['backend']} samples={metrics['samples']} accuracy={metrics['accuracy']:.3%} "
          f"macro_f1={metrics['macro_f1']:.3f} coverage={metrics['coverage']:.3%}")
    print(f"results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
