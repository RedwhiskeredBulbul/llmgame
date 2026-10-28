"""Metrics and data utilities for vehicle behaviour-type experiments."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


BEHAVIOR_LABELS = ("Cautious", "Normal", "Aggressive")
INTENT_LABELS = ("Yield", "Normal", "Force")
BEHAVIOR_TO_INTENT = dict(zip(BEHAVIOR_LABELS, INTENT_LABELS))
INTENT_TO_BEHAVIOR = dict(zip(INTENT_LABELS, BEHAVIOR_LABELS))


def normalize_behavior_label(value: object) -> Optional[str]:
    """Map simulator class names and intent labels to one evaluation ontology."""
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "cautious": "Cautious",
        "cautiousvehicle_df": "Cautious",
        "yield": "Cautious",
        "normal": "Normal",
        "normalvehicle_df": "Normal",
        "aggressive": "Aggressive",
        "aggressivevehicle_df": "Aggressive",
        "cutinvehicle_aggressive": "Aggressive",
        "force": "Aggressive",
    }
    return aliases.get(text)


def vehicle_ground_truth(vehicle: object) -> Optional[str]:
    """Read a labelled simulator vehicle without changing its physical ``type``."""
    explicit = normalize_behavior_label(getattr(vehicle, "behavior_type", None))
    if explicit:
        return explicit
    class_label = normalize_behavior_label(getattr(vehicle.__class__, "BEHAVIOR_TYPE", None))
    if class_label:
        return class_label
    return normalize_behavior_label(vehicle.__class__.__name__)


@dataclass
class EvaluationRecord:
    case_id: str
    vehicle_id: int
    true_label: str
    predicted_label: Optional[str]
    p_cautious: Optional[float]
    p_normal: Optional[float]
    p_aggressive: Optional[float]
    latency_s: Optional[float] = None
    error: Optional[str] = None

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.true_label


def prediction_from_probs(probs: Optional[Sequence[float]]) -> Optional[str]:
    if probs is None or len(probs) != 3:
        return None
    return BEHAVIOR_LABELS[max(range(3), key=lambda idx: probs[idx])]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def calculate_metrics(records: Sequence[EvaluationRecord]) -> dict:
    """Compute metrics with missing/invalid predictions counted as incorrect."""
    total = len(records)
    parsed = [record for record in records if record.predicted_label in BEHAVIOR_LABELS]
    correct = sum(record.correct for record in records)
    parsed_correct = sum(record.correct for record in parsed)

    matrix = {
        actual: {predicted: 0 for predicted in (*BEHAVIOR_LABELS, "Unparsed")}
        for actual in BEHAVIOR_LABELS
    }
    for record in records:
        predicted = record.predicted_label if record.predicted_label in BEHAVIOR_LABELS else "Unparsed"
        matrix[record.true_label][predicted] += 1

    per_class = {}
    f1_values = []
    for label in BEHAVIOR_LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in BEHAVIOR_LABELS)
        precision = _safe_div(tp, predicted_count)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    probability_records = [
        record
        for record in parsed
        if None not in (record.p_cautious, record.p_normal, record.p_aggressive)
    ]
    brier_values = []
    log_losses = []
    confidences = []
    for record in probability_records:
        probs = [record.p_cautious, record.p_normal, record.p_aggressive]
        true_idx = BEHAVIOR_LABELS.index(record.true_label)
        brier_values.append(sum((prob - float(idx == true_idx)) ** 2 for idx, prob in enumerate(probs)))
        log_losses.append(-math.log(max(probs[true_idx], 1e-15)))
        confidences.append(max(probs))

    latencies = [record.latency_s for record in records if record.latency_s is not None]
    return {
        "samples": total,
        "parsed_predictions": len(parsed),
        "coverage": _safe_div(len(parsed), total),
        "accuracy": _safe_div(correct, total),
        "accuracy_95ci_wilson": _wilson_interval(correct, total),
        "conditional_accuracy": _safe_div(parsed_correct, len(parsed)),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "mean_brier_score": _safe_div(sum(brier_values), len(brier_values)),
        "mean_log_loss": _safe_div(sum(log_losses), len(log_losses)),
        "mean_confidence": _safe_div(sum(confidences), len(confidences)),
        "mean_latency_s": _safe_div(sum(latencies), len(latencies)),
        "errors": sum(record.error is not None for record in records),
    }


def synthetic_cases(samples_per_class: int = 10, history_len: int = 5, dt: float = 0.5, seed: int = 42) -> List[dict]:
    """Create a balanced, seeded smoke-test set with overlapping behaviour cues.

    This set validates the experiment pipeline. It is not a replacement for a
    held-out real/simulator trajectory set when publishing LLM accuracy.
    """
    rng = random.Random(seed)
    profiles = {
        "Cautious": {"dv": (-1.8, 1.1), "acc": (-0.25, 0.28), "gap": (28.0, 8.0)},
        "Normal": {"dv": (0.0, 1.1), "acc": (0.0, 0.25), "gap": (23.0, 8.0)},
        "Aggressive": {"dv": (1.8, 1.1), "acc": (0.25, 0.28), "gap": (18.0, 8.0)},
    }
    cases = []
    case_number = 0
    for label in BEHAVIOR_LABELS:
        profile = profiles[label]
        for _ in range(samples_per_class):
            vehicle_id = 1000 + case_number
            ego_speed = rng.uniform(23.0, 27.0)
            acceleration = rng.gauss(*profile["acc"])
            initial_dv = rng.gauss(*profile["dv"])
            initial_gap = max(6.0, rng.gauss(*profile["gap"]))
            lateral = rng.choice([0.0, -4.0, 4.0])
            target_speed = max(0.0, ego_speed + initial_dv)
            gap = initial_gap
            history = []
            for frame_idx in range(history_len):
                if frame_idx:
                    target_speed = max(0.0, target_speed + acceleration * dt)
                    gap += (target_speed - ego_speed) * dt
                lane_id = 1 if lateral == 0 else (0 if lateral > 0 else 2)
                history.append(
                    {
                        "ego_state": [0.0, 0.0, 0.0, ego_speed, 0.0, 0.0],
                        "curr_lane_idx": 1,
                        "others_info": {
                            "ids": [vehicle_id],
                            "states": [[gap, lateral, 0.0, target_speed, acceleration, 0.0]],
                            "frenet_relative_coords": [[gap, lateral]],
                            "laneids": [lane_id],
                        },
                    }
                )
            cases.append(
                {
                    "case_id": f"synthetic-{case_number:04d}",
                    "history": history,
                    "ground_truth": {vehicle_id: label},
                }
            )
            case_number += 1
    rng.shuffle(cases)
    return cases


def save_cases_jsonl(cases: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def load_cases_jsonl(path: Path) -> List[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            if "history" not in case or "ground_truth" not in case:
                raise ValueError(f"{path}:{line_number}: history and ground_truth are required")
            case["ground_truth"] = {int(key): value for key, value in case["ground_truth"].items()}
            cases.append(case)
    return cases


def save_results(records: Sequence[EvaluationRecord], metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*asdict(records[0]).keys(), "correct"] if records else ["correct"])
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["correct"] = record.correct
            writer.writerow(row)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    columns = [*BEHAVIOR_LABELS, "Unparsed"]
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *columns])
        for actual in BEHAVIOR_LABELS:
            writer.writerow([actual, *(metrics["confusion_matrix"][actual][column] for column in columns)])

    report = [
        "# Vehicle behaviour-type experiment",
        "",
        f"- Backend: {metrics.get('backend', 'unknown')}",
        f"- Model: {metrics.get('model', 'unknown')}",
        f"- Dataset: {metrics.get('dataset', 'unknown')}",
        f"- Samples: {metrics['samples']}",
        f"- Accuracy (unparsed counted wrong): {metrics['accuracy']:.3%}",
        f"- Parse coverage: {metrics['coverage']:.3%}",
        f"- Conditional accuracy: {metrics['conditional_accuracy']:.3%}",
        f"- Macro F1: {metrics['macro_f1']:.3f}",
        f"- Mean Brier score: {metrics['mean_brier_score']:.3f}",
        f"- Mean latency: {metrics['mean_latency_s']:.3f} s",
        "",
        "| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |",
        "|---|---:|---:|---:|---:|",
    ]
    for actual in BEHAVIOR_LABELS:
        row = metrics["confusion_matrix"][actual]
        report.append(f"| {actual} | {row['Cautious']} | {row['Normal']} | {row['Aggressive']} | {row['Unparsed']} |")
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
