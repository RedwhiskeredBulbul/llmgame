"""Paired style-recognition diagnostics for prompt versions v5 and v6.

The v6 prompt is deliberately allowed to abstain when the target's own-leader
evidence is unavailable.  A hard argmax therefore mixes genuine errors with
honest uncertainty.  This script reports both the hard-label metrics used by
the existing pipeline and confidence-gated metrics, and it pairs v5/v6
predictions on the same (episode, vehicle, query) keys.

Usage::

    python -m experiments.ablation_v2.style_diagnostics \
        --run df_v5=simulation_results/.../v6_style_diag_df_v5_001 \
        --run df_v6=simulation_results/.../v6_style_diag_df_v6_001 \
        --run merge_v5=... --run merge_v6=... \
        --output simulation_results/.../v6_style_diag_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

LABELS = ("Cautious", "Normal", "Aggressive")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
CONFIDENCE_THRESHOLDS = (0.45, 0.50, 0.60)


def argmax_label(probs: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda index: float(probs[index]))]


def normalize(probs: Sequence[float]) -> Optional[List[float]]:
    values = [float(value) for value in probs]
    if len(values) != 3 or any(not math.isfinite(value) or value < 0 for value in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    return [value / total for value in values]


def brier(probs: Sequence[float], truth: str) -> float:
    true_index = LABEL_INDEX[truth]
    return sum((float(prob) - float(index == true_index)) ** 2
               for index, prob in enumerate(probs))


def log_loss(probs: Sequence[float], truth: str) -> float:
    return -math.log(max(float(probs[LABEL_INDEX[truth]]), 1e-15))


def macro_f1(confusion: Dict[str, Dict[str, int]]) -> float:
    values = []
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[actual][label] for actual in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        values.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return sum(values) / len(values)


def binary_scores(records: Sequence[dict], predict_positive) -> dict:
    truth_positive = [record["truth"] == "Aggressive" for record in records]
    predicted_positive = [bool(predict_positive(record)) for record in records]
    tp = sum(t and p for t, p in zip(truth_positive, predicted_positive))
    tn = sum((not t) and (not p) for t, p in zip(truth_positive, predicted_positive))
    fp = sum((not t) and p for t, p in zip(truth_positive, predicted_positive))
    fn = sum(t and (not p) for t, p in zip(truth_positive, predicted_positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall,
                specificity=specificity, balanced_accuracy=(recall + specificity) / 2,
                accuracy=(tp + tn) / len(records) if records else None,
                predicted_positive=sum(predicted_positive),
                truth_positive=sum(truth_positive))


def metrics(records: Sequence[dict]) -> dict:
    n = len(records)
    if not n:
        return {"n": 0}
    correct = sum(record["correct"] for record in records)
    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for record in records:
        confusion[record["truth"]][record["pred"]] += 1
    by_truth = {}
    for label in LABELS:
        part = [record for record in records if record["truth"] == label]
        by_truth[label] = dict(
            n=len(part),
            correct=sum(record["correct"] for record in part),
            accuracy=(sum(record["correct"] for record in part) / len(part)) if part else None,
            predictions={predicted: sum(record["pred"] == predicted for record in part)
                         for predicted in LABELS},
            mean_confidence=(float(np.mean([record["max_prob"] for record in part]))
                             if part else None),
            mean_p_true=(float(np.mean([record["probs"][LABEL_INDEX[label]]
                                        for record in part])) if part else None),
            brier=(float(np.mean([record["brier"] for record in part])) if part else None),
            log_loss=(float(np.mean([record["log_loss"] for record in part]))
                      if part else None))
    confidence = [record["max_prob"] for record in records]
    truth_counts = Counter(record["truth"] for record in records)
    majority_label, majority_count = truth_counts.most_common(1)[0]
    result = dict(
        n=n, correct=correct, accuracy=correct / n,
        majority_label=majority_label, majority_accuracy=majority_count / n,
        macro_f1=macro_f1(confusion), confusion=confusion, by_truth=by_truth,
        brier=float(np.mean([record["brier"] for record in records])),
        log_loss=float(np.mean([record["log_loss"] for record in records])),
        mean_confidence=float(np.mean(confidence)),
        mean_margin=float(np.mean([record["margin"] for record in records])),
        mean_latency_ms=float(np.mean([record["latency_ms"] for record in records
                                       if record["latency_ms"] is not None]))
        if any(record["latency_ms"] is not None for record in records) else None,
        binary_hard=binary_scores(records, lambda record: record["pred"] == "Aggressive"),
        binary_threshold=binary_scores(records,
                                       lambda record: record["probs"][2] >= 0.5),
    )
    result["coverage"] = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        part = [record for record in records if record["max_prob"] >= threshold]
        result["coverage"][f"{threshold:.2f}"] = dict(
            n=len(part), coverage=len(part) / n,
            accuracy=(sum(record["correct"] for record in part) / len(part)) if part else None,
            brier=(float(np.mean([record["brier"] for record in part])) if part else None))
    order = sorted(range(n), key=lambda index: confidence[index], reverse=True)
    result["risk_coverage"] = {}
    for coverage in (0.25, 0.50, 0.75, 0.90, 1.00):
        count = max(1, int(round(coverage * n)))
        selected = [records[index] for index in order[:count]]
        result["risk_coverage"][f"{coverage:.2f}"] = dict(
            n=count, coverage=count / n,
            accuracy=sum(record["correct"] for record in selected) / count,
            brier=float(np.mean([record["brier"] for record in selected])))
    return result


def load_run(path: Path) -> dict:
    expected_episodes = None
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config") or {}
        if config.get("seeds"):
            expected_episodes = len(config["seeds"])
    cases = {}
    cases_path = path / "cases.json"
    if cases_path.exists():
        for case in json.loads(cases_path.read_text(encoding="utf-8")):
            cases[case.get("case_id")] = case
    query_records: List[dict] = []
    vehicle_records: List[dict] = []
    missing_truth = 0
    episode_count = 0
    all_response_events = 0
    error_events = 0
    empty_probability_events = 0
    for episode_dir in sorted((path / "episodes").iterdir()):
        if not episode_dir.is_dir():
            continue
        result_path = episode_dir / "result.json"
        belief_path = episode_dir / "belief_events.json"
        initial_path = episode_dir / "initial_state.json"
        if not (result_path.exists() and belief_path.exists() and initial_path.exists()):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("variant") != "full_model":
            continue
        initial_doc = json.loads(initial_path.read_text(encoding="utf-8"))
        initial = initial_doc.get("initial_state", [])
        truth = {int(vehicle["id"]): vehicle.get("behaviour") for vehicle in initial
                 if vehicle.get("behaviour") in LABELS}
        ego = next((vehicle for vehicle in initial
                    if vehicle.get("behaviour") == "AV"), None)
        if ego is None or not truth:
            continue
        episode_count += 1
        case_id = result.get("case_id") or episode_dir.name.rsplit("-", 1)[0]
        case = cases.get(case_id, {})
        key_id = case.get("key_id", result.get("key_id"))
        if key_id is None and 1 in truth:
            # Merge scenes keep the key vehicle at id 1.
            key_id = 1
        if key_id is not None:
            key_id = int(key_id)
        positions = {}
        for vehicle in initial:
            vehicle_id = int(vehicle["id"])
            if vehicle_id not in truth:
                continue
            dx = float(vehicle["position"][0]) - float(ego["position"][0])
            dy = float(vehicle["position"][1]) - float(ego["position"][1])
            positions[vehicle_id] = (
                "ahead" if dx > 1.0 else ("behind" if dx < -1.0 else "alongside"),
                dx, dy)
        events = [event for event in json.loads(belief_path.read_text(encoding="utf-8"))
                  if event.get("event") == "response"]
        all_response_events += len(events)
        error_events += sum(bool(event.get("error")) for event in events)
        empty_probability_events += sum(not event.get("probabilities") for event in events)
        per_vehicle = defaultdict(list)
        for query_index, event in enumerate(events):
            probabilities = event.get("probabilities") or {}
            for raw_id, raw_probs in probabilities.items():
                vehicle_id = int(raw_id)
                if vehicle_id not in truth:
                    missing_truth += 1
                    continue
                probs = normalize(raw_probs)
                if probs is None:
                    continue
                label = argmax_label(probs)
                sorted_probs = sorted(probs, reverse=True)
                record = dict(
                    episode=result.get("episode_id", episode_dir.name),
                    episode_dir=episode_dir.name,
                    case_id=case_id, seed=result.get("seed"),
                    vehicle_id=vehicle_id, query_index=query_index,
                    sim_time=event.get("observation_sim_time"),
                    latency_ms=event.get("latency_ms"),
                    truth=truth[vehicle_id], probs=probs, pred=label,
                    max_prob=sorted_probs[0], margin=sorted_probs[0] - sorted_probs[1],
                    correct=label == truth[vehicle_id],
                    binary_hard=(label == "Aggressive") == (truth[vehicle_id] == "Aggressive"),
                    binary_threshold=(probs[2] >= 0.5) == (truth[vehicle_id] == "Aggressive"),
                    is_key=(vehicle_id == key_id),
                    position=positions.get(vehicle_id, ("unknown", 0.0, 0.0))[0],
                    dx=positions.get(vehicle_id, ("unknown", 0.0, 0.0))[1],
                    dy=positions.get(vehicle_id, ("unknown", 0.0, 0.0))[2],
                )
                record["brier"] = brier(probs, truth[vehicle_id])
                record["log_loss"] = log_loss(probs, truth[vehicle_id])
                query_records.append(record)
                per_vehicle[vehicle_id].append(record)
        for vehicle_id, sequence in per_vehicle.items():
            sequence.sort(key=lambda record: record["query_index"])
            first = sequence[0]
            mean_probs = [float(np.mean([record["probs"][index] for record in sequence]))
                          for index in range(3)]
            mean_pred = argmax_label(mean_probs)
            majority = Counter(record["pred"] for record in sequence).most_common()
            majority_label = majority[0][0] if majority else None
            vehicle_records.append(dict(
                episode=first["episode"], case_id=case_id, seed=first["seed"],
                vehicle_id=vehicle_id, truth=first["truth"], is_key=first["is_key"],
                position=first["position"], n_queries=len(sequence),
                first_pred=first["pred"], first_correct=first["correct"],
                first_max_prob=first["max_prob"], first_margin=first["margin"],
                first_probs=list(first["probs"]),
                mean_pred=mean_pred, mean_correct=(mean_pred == first["truth"]),
                mean_probs=mean_probs,
                majority_pred=majority_label,
                majority_correct=(majority_label == first["truth"])))
    return dict(path=str(path), episodes=episode_count,
                expected_episodes=expected_episodes, cases=cases,
                query_records=query_records, vehicle_records=vehicle_records,
                missing_truth=missing_truth,
                all_response_events=all_response_events,
                error_events=error_events,
                empty_probability_events=empty_probability_events)


def subset(records: Sequence[dict], **conditions) -> List[dict]:
    selected = list(records)
    for field, expected in conditions.items():
        selected = [record for record in selected if record[field] == expected]
    return selected


def paired_metric(records_v5: Sequence[dict], records_v6: Sequence[dict],
                  field: str) -> dict:
    lookup_v5 = {(record["episode"], record["vehicle_id"], record["query_index"]): record
                 for record in records_v5}
    lookup_v6 = {(record["episode"], record["vehicle_id"], record["query_index"]): record
                 for record in records_v6}
    keys = sorted(set(lookup_v5) & set(lookup_v6))
    only_v5 = len(set(lookup_v5) - set(lookup_v6))
    only_v6 = len(set(lookup_v6) - set(lookup_v5))
    better = worse = 0
    for key in keys:
        left, right = lookup_v5[key][field], lookup_v6[key][field]
        if right and not left:
            better += 1
        elif left and not right:
            worse += 1
    p_value = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    return dict(n_pairs=len(keys), only_v5=only_v5, only_v6=only_v6,
                v6_better=better, v6_worse=worse, exact_mcnemar_p=float(p_value))


def paired_mean_difference(records_v5: Sequence[dict], records_v6: Sequence[dict],
                           field: str, seed: int = 20261023) -> dict:
    lookup_v5 = {(record["episode"], record["vehicle_id"], record["query_index"]): record
                 for record in records_v5}
    lookup_v6 = {(record["episode"], record["vehicle_id"], record["query_index"]): record
                 for record in records_v6}
    by_episode = defaultdict(list)
    for key in set(lookup_v5) & set(lookup_v6):
        by_episode[lookup_v5[key]["episode"]].append(
            float(lookup_v6[key][field]) - float(lookup_v5[key][field]))
    values = np.asarray([np.mean(differences) for differences in by_episode.values()])
    if not len(values):
        return dict(n_episodes=0, mean_difference=None)
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(n_episodes=len(values), mean_difference=float(values.mean()),
                ci_low=float(np.percentile(bootstrap, 2.5)),
                ci_high=float(np.percentile(bootstrap, 97.5)))


def trajectory_gate(path_v5: Path, path_v6: Path) -> dict:
    mismatches = 0
    compared_steps = 0
    compared_episodes = 0
    max_position_delta = 0.0
    max_speed_delta = 0.0
    for left_dir in sorted((path_v5 / "episodes").glob("*-full_model")):
        right_dir = path_v6 / "episodes" / left_dir.name
        left_path = left_dir / "steps.csv"
        right_path = right_dir / "steps.csv"
        if not (left_path.exists() and right_path.exists()):
            continue
        with left_path.open(encoding="utf-8-sig") as handle:
            left_rows = list(csv.DictReader(handle))
        with right_path.open(encoding="utf-8-sig") as handle:
            right_rows = list(csv.DictReader(handle))
        compared_episodes += 1
        for left, right in zip(left_rows, right_rows):
            compared_steps += 1
            position_delta = max(abs(float(left["x"]) - float(right["x"])),
                                 abs(float(left["y"]) - float(right["y"])))
            speed_delta = abs(float(left["speed_after_ms"]) - float(right["speed_after_ms"]))
            max_position_delta = max(max_position_delta, position_delta)
            max_speed_delta = max(max_speed_delta, speed_delta)
            if (position_delta > 1e-6 or speed_delta > 1e-6
                    or left["action_type"] != right["action_type"]
                    or left["target_lane"] != right["target_lane"]):
                mismatches += 1
    return dict(compared_episodes=compared_episodes, compared_steps=compared_steps,
                mismatch_steps=mismatches,
                max_position_delta_m=max_position_delta,
                max_speed_delta_ms=max_speed_delta,
                identical=(mismatches == 0 and compared_steps > 0))


def summarize_run(run: dict) -> dict:
    query_records = run["query_records"]
    vehicle_records = run["vehicle_records"]
    key_records = [record for record in query_records if record["is_key"]]
    background_records = [record for record in query_records if not record["is_key"]]
    return dict(
        path=run["path"], episodes=run["episodes"],
        expected_episodes=run["expected_episodes"],
        complete=(run["expected_episodes"] is None
                  or run["episodes"] >= run["expected_episodes"]),
        query_samples=len(query_records), unique_vehicle_episodes=len(vehicle_records),
        missing_truth=run["missing_truth"],
        response_events=run["all_response_events"],
        error_events=run["error_events"],
        empty_probability_events=run["empty_probability_events"],
        all=metrics(query_records),
        key=metrics(key_records),
        background=metrics(background_records),
        by_position={position: metrics(subset(query_records, position=position))
                     for position in ("ahead", "behind", "alongside")},
        vehicle_level=dict(
            n=len(vehicle_records),
            first_correct=sum(record["first_correct"] for record in vehicle_records),
            first_accuracy=(sum(record["first_correct"] for record in vehicle_records)
                            / len(vehicle_records)) if vehicle_records else None,
            mean_accuracy=(sum(record["mean_correct"] for record in vehicle_records)
                           / len(vehicle_records)) if vehicle_records else None,
            majority_accuracy=(sum(record["majority_correct"] for record in vehicle_records)
                               / len(vehicle_records)) if vehicle_records else None,
            first_brier=float(np.mean([brier(record["first_probs"], record["truth"])
                                       for record in vehicle_records]))
            if vehicle_records else None,
            key_n=len([record for record in vehicle_records if record["is_key"]]),
            key_first_accuracy=(sum(record["first_correct"] for record in vehicle_records
                                    if record["is_key"])
                                / len([record for record in vehicle_records
                                       if record["is_key"]]))
            if any(record["is_key"] for record in vehicle_records) else None,
            background_n=len([record for record in vehicle_records if not record["is_key"]]),
            background_first_accuracy=(sum(record["first_correct"] for record in vehicle_records
                                           if not record["is_key"])
                                       / len([record for record in vehicle_records
                                              if not record["is_key"]]))
            if any(not record["is_key"] for record in vehicle_records) else None,
        ))


def pair_runs(label_v5: str, run_v5: dict, label_v6: str, run_v6: dict) -> dict:
    records_v5 = run_v5["query_records"]
    records_v6 = run_v6["query_records"]
    pair = dict(
        label_v5=label_v5, label_v6=label_v6,
        trajectory=trajectory_gate(Path(run_v5["path"]), Path(run_v6["path"])),
        query_pairs={
            "three_class": paired_metric(records_v5, records_v6, "correct"),
            "binary_hard": paired_metric(records_v5, records_v6, "binary_hard"),
            "binary_threshold": paired_metric(records_v5, records_v6, "binary_threshold"),
        },
        accuracy_difference=paired_mean_difference(records_v5, records_v6, "correct"),
        brier_difference=paired_mean_difference(records_v5, records_v6, "brier"),
        log_loss_difference=paired_mean_difference(records_v5, records_v6, "log_loss"),
    )
    lookup_v5 = {(record["episode"], record["vehicle_id"]): record
                 for record in run_v5["vehicle_records"]}
    lookup_v6 = {(record["episode"], record["vehicle_id"]): record
                 for record in run_v6["vehicle_records"]}
    keys = sorted(set(lookup_v5) & set(lookup_v6))
    better = sum(lookup_v6[key]["first_correct"] and not lookup_v5[key]["first_correct"]
                 for key in keys)
    worse = sum(lookup_v5[key]["first_correct"] and not lookup_v6[key]["first_correct"]
                for key in keys)
    pair["vehicle_first"] = dict(
        n_pairs=len(keys), v6_better=better, v6_worse=worse,
        exact_mcnemar_p=float(binomtest(better, better + worse, 0.5).pvalue)
        if better + worse else 1.0)
    pair["key_vehicles"] = []
    for key in keys:
        left, right = lookup_v5[key], lookup_v6[key]
        if not left["is_key"]:
            continue
        truth = left["truth"]
        true_index = LABEL_INDEX[truth]
        pair["key_vehicles"].append(dict(
            seed=left.get("seed"), episode=left["episode"], truth=truth,
            v5_pred=left["first_pred"], v5_p_true=left["first_probs"][true_index],
            v5_max_prob=left["first_max_prob"],
            v6_pred=right["first_pred"], v6_p_true=right["first_probs"][true_index],
            v6_max_prob=right["first_max_prob"]))
    pair["key_vehicles"].sort(key=lambda row: (row["seed"] is None, row["seed"]))
    return pair


def markdown(summary: dict) -> str:
    lines = ["# v5 vs v6 style-recognition diagnostics", ""]
    lines.append("## Runs")
    lines.append("")
    lines.append("| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        all_metrics = run["all"]
        confident = all_metrics["coverage"]["0.50"]
        lines.append(
            f"| {label} | {run['episodes']} | {run['query_samples']} | "
            f"{run['unique_vehicle_episodes']} | {all_metrics['accuracy']:.3f} | "
            f"{all_metrics['majority_accuracy']:.3f} | "
            f"{confident['accuracy']:.3f} ({confident['coverage']:.3f}) | "
            f"{all_metrics['binary_hard']['accuracy']:.3f} | "
            f"{all_metrics['brier']:.3f} | {all_metrics['log_loss']:.3f} |")
    lines.append("")
    lines.append("## Vehicle-level recognition (one prediction per vehicle-episode)")
    lines.append("")
    lines.append("| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        values = run["vehicle_level"]
        if not values.get("n"):
            continue
        lines.append(
            f"| {label} | {values['n']} | {values['first_accuracy']:.3f} | "
            f"{values['mean_accuracy']:.3f} | {values['majority_accuracy']:.3f} | "
            f"{values['first_brier']:.3f} | "
            f"{values['key_first_accuracy'] if values['key_first_accuracy'] is not None else float('nan'):.3f} "
            f"({values['key_n']}) | "
            f"{values['background_first_accuracy'] if values['background_first_accuracy'] is not None else float('nan'):.3f} "
            f"({values['background_n']}) |")
    lines.append("")
    lines.append("## LLM response health")
    lines.append("")
    lines.append("| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        lines.append(
            f"| {label} | {run['episodes']} ({run['expected_episodes']}) | "
            f"{run['complete']} | {run['response_events']} | {run['error_events']} | "
            f"{run['empty_probability_events']} | {run['missing_truth']} |")
    lines.append("")
    lines.append("## Risk-coverage curve (all query samples; select the most confident records)")
    lines.append("")
    lines.append("| Run | Coverage | n | Accuracy | Brier |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        for coverage, values in run["all"]["risk_coverage"].items():
            lines.append(
                f"| {label} | {coverage} | {values['n']} | "
                f"{values['accuracy']:.3f} | {values['brier']:.3f} |")
    lines.append("")
    lines.append("## Key vs background (all query samples)")
    lines.append("")
    lines.append("| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        for subset_name in ("all", "key", "background"):
            values = run[subset_name]
            if not values.get("n"):
                continue
            confident = values["coverage"]["0.50"]
            lines.append(
                f"| {label} | {subset_name} | {values['n']} | {values['accuracy']:.3f} | "
                f"{values['macro_f1']:.3f} | "
                f"{confident['accuracy'] if confident['accuracy'] is not None else float('nan'):.3f} "
                f"({confident['coverage']:.3f}) | "
                f"{values['binary_hard']['accuracy']:.3f} | "
                f"{values['binary_threshold']['accuracy']:.3f} | "
                f"{values['brier']:.3f} | {values['log_loss']:.3f} |")
    lines.append("")
    lines.append("## Per-truth-class breakdown (all query samples)")
    lines.append("")
    lines.append("| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        for truth_label in LABELS:
            values = run["all"]["by_truth"][truth_label]
            if not values.get("n"):
                continue
            lines.append(
                f"| {label} | {truth_label} | {values['n']} | "
                f"{values['accuracy']:.3f} | "
                f"{values['mean_confidence']:.3f} | "
                f"{values['mean_p_true']:.3f} | "
                f"{values['brier']:.3f} |")
    lines.append("")
    lines.append("## Paired comparisons")
    lines.append("")
    for pair_label, pair in summary["pairs"].items():
        lines.append(f"### {pair_label}")
        lines.append("")
        lines.append(f"Trajectory identity: {pair['trajectory']}")
        lines.append("")
        lines.append("| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for target in ("all", "key", "background"):
            part = pair.get("subsets", {}).get(target)
            if not part:
                continue
            left_metrics = part["metrics_v5"]
            right_metrics = part["metrics_v6"]
            lines.append(
                f"| {target} | {left_metrics['n']} | {right_metrics['n']} | "
                f"{left_metrics['accuracy']:.3f} | {right_metrics['accuracy']:.3f} | "
                f"{right_metrics['accuracy'] - left_metrics['accuracy']:+.3f} | "
                f"{left_metrics['brier']:.3f} | {right_metrics['brier']:.3f} | "
                f"{right_metrics['brier'] - left_metrics['brier']:+.3f} | "
                f"{left_metrics['log_loss']:.3f} | {right_metrics['log_loss']:.3f} | "
                f"{right_metrics['log_loss'] - left_metrics['log_loss']:+.3f} |")
        lines.append("")
        lines.append("| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for target in ("all", "key", "background"):
            part = pair.get("subsets", {}).get(target)
            if not part:
                continue
            for metric_name in ("three_class", "binary_hard", "binary_threshold"):
                row = part[metric_name]
                lines.append(
                    f"| {target} | {metric_name} | {row['v6_better']} | "
                    f"{row['v6_worse']} | {row['only_v5']} | {row['only_v6']} | "
                    f"{row['exact_mcnemar_p']:.6g} |")
        lines.append("")
        lines.append(f"Accuracy difference (v6 - v5, episode-clustered): {pair['accuracy_difference']}")
        lines.append(f"Brier difference (v6 - v5): {pair['brier_difference']}")
        lines.append(f"Log-loss difference (v6 - v5): {pair['log_loss_difference']}")
        lines.append("")
        if pair.get("by_position"):
            lines.append("| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for position, part in pair["by_position"].items():
                left_metrics = part["metrics_v5"]
                right_metrics = part["metrics_v6"]
                lines.append(
                    f"| {position} | {left_metrics['n']} | {right_metrics['n']} | "
                    f"{left_metrics['accuracy']:.3f} | {right_metrics['accuracy']:.3f} | "
                    f"{right_metrics['accuracy'] - left_metrics['accuracy']:+.3f} | "
                    f"{left_metrics['brier']:.3f} | {right_metrics['brier']:.3f} |")
            lines.append("")
        if pair.get("by_truth"):
            lines.append("| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
            for truth_label, part in pair["by_truth"].items():
                left_metrics = part["metrics_v5"]
                right_metrics = part["metrics_v6"]
                row = part["three_class"]
                lines.append(
                    f"| {truth_label} | {left_metrics['n']} | {right_metrics['n']} | "
                    f"{left_metrics['accuracy']:.3f} | {right_metrics['accuracy']:.3f} | "
                    f"{right_metrics['accuracy'] - left_metrics['accuracy']:+.3f} | "
                    f"{left_metrics['brier']:.3f} | {right_metrics['brier']:.3f} | "
                    f"{row['v6_better']}/{row['v6_worse']} | "
                    f"{row['exact_mcnemar_p']:.6g} |")
            lines.append("")
        if pair.get("key_vehicles"):
            lines.append("| Key seed | Truth | v5 first | v5 p_true | v6 first | v6 p_true |")
            lines.append("|---:|---|---|---:|---|---:|")
            for row in pair["key_vehicles"]:
                lines.append(
                    f"| {row['seed']} | {row['truth']} | {row['v5_pred']} | "
                    f"{row['v5_p_true']:.3f} | {row['v6_pred']} | "
                    f"{row['v6_p_true']:.3f} |")
            lines.append("")
    lines.append("## Position breakdown (all query samples)")
    lines.append("")
    lines.append("| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for label, run in summary["runs"].items():
        for position, values in run["by_position"].items():
            if not values.get("n"):
                continue
            confident = values["coverage"]["0.50"]
            lines.append(
                f"| {label} | {position} | {values['n']} | {values['accuracy']:.3f} | "
                f"{confident['accuracy'] if confident['accuracy'] is not None else float('nan'):.3f} | "
                f"{values['brier']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="LABEL=PATH; repeat for each run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair", action="append", default=[],
                        help="PAIR_LABEL:V5_LABEL:V6_LABEL; defaults to pairing "
                             "runs whose labels end in v5/v6 with the same prefix")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, raw_path = item.split("=", 1)
        runs[label] = load_run(Path(raw_path))
    summary = dict(runs={label: summarize_run(run) for label, run in runs.items()},
                   pairs={})
    pairs = []
    if args.pair:
        for item in args.pair:
            pair_label, left, right = item.split(":", 2)
            pairs.append((pair_label, left, right))
    else:
        for label in runs:
            if label.endswith("_v5"):
                counterpart = label[:-3] + "_v6"
                if counterpart in runs:
                    pairs.append((label[:-3], label, counterpart))
    for pair_label, left, right in pairs:
        pair = pair_runs(left, runs[left], right, runs[right])
        pair["subsets"] = {}
        for subset_name, condition in (("all", {}), ("key", {"is_key": True}),
                                       ("background", {"is_key": False})):
            part_v5 = subset(runs[left]["query_records"], **condition)
            part_v6 = subset(runs[right]["query_records"], **condition)
            pair["subsets"][subset_name] = {
                "three_class": paired_metric(part_v5, part_v6, "correct"),
                "binary_hard": paired_metric(part_v5, part_v6, "binary_hard"),
                "binary_threshold": paired_metric(part_v5, part_v6, "binary_threshold"),
                "metrics_v5": metrics(part_v5),
                "metrics_v6": metrics(part_v6),
            }
        pair["by_truth"] = {}
        for truth_label in LABELS:
            part_v5 = subset(runs[left]["query_records"], truth=truth_label)
            part_v6 = subset(runs[right]["query_records"], truth=truth_label)
            if part_v5 or part_v6:
                pair["by_truth"][truth_label] = {
                    "three_class": paired_metric(part_v5, part_v6, "correct"),
                    "metrics_v5": metrics(part_v5),
                    "metrics_v6": metrics(part_v6),
                }
        pair["by_position"] = {}
        for position in ("ahead", "behind", "alongside"):
            part_v5 = subset(runs[left]["query_records"], position=position)
            part_v6 = subset(runs[right]["query_records"], position=position)
            if part_v5 or part_v6:
                pair["by_position"][position] = {
                    "three_class": paired_metric(part_v5, part_v6, "correct"),
                    "metrics_v5": metrics(part_v5),
                    "metrics_v6": metrics(part_v6),
                }
        summary["pairs"][pair_label] = pair
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output.with_suffix(".md")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(output)
    print(report_path)


if __name__ == "__main__":
    main()
