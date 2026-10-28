"""Offline query-level type recognition on the DF_Cruise dataset.

The cruise dataset stores frozen 20-frame histories and ground-truth native
behaviour labels for the three nearest vehicles.  This script rebuilds the v5
and v6b prompts from those histories, queries the frozen LLM protocol once per
case, and reports query-level accuracy.  No simulator or controller is
involved, so the same histories are used for both prompt versions.

Usage::

    python -m experiments.ablation_v2.cruise_offline_type_eval \
        --dataset simulation_results/vehicle_type_dataset_v2_20f.jsonl \
        --output-dir simulation_results/ablation_v2/cruise_offline_type_eval_001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from experiments.ablation_v2.style_diagnostics import (
    LABELS, argmax_label, brier, log_loss, metrics, normalize, paired_mean_difference,
    paired_metric)
from model.reasoning.worker import LLMReasoningWorker

VERSIONS = {
    "v5": "relative_evidence_v5",
    "v6b": "relative_evidence_v6b",
}


def read_cases(path: Path) -> List[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def latest_position(frame: dict, vehicle_id: int) -> tuple:
    others = frame.get("others_info") or {}
    ids = list(others.get("ids") or [])
    if vehicle_id not in ids:
        return "unknown", 0.0, None
    index = ids.index(vehicle_id)
    coords = others.get("frenet_relative_coords") or []
    lanes = others.get("laneids") or []
    ds = float(coords[index][0]) if index < len(coords) else 0.0
    lane = lanes[index] if index < len(lanes) else None
    position = "ahead" if ds > 1.0 else ("behind" if ds < -1.0 else "alongside")
    return position, ds, lane


def make_records(cases: Sequence[dict], version_key: str, model: str,
                 temperature: float, timeout_s: float, pause_s: float,
                 dt: float) -> List[dict]:
    worker = LLMReasoningWorker(
        model=model, temperature=temperature, timeout_s=timeout_s, max_retries=0,
        prompt_version=VERSIONS[version_key], enable_thinking=False, dt=dt)
    records: List[dict] = []
    target_mismatches = 0
    prompt_failures = 0
    try:
        for case in cases:
            history = case["history"]
            truth = {int(key): value for key, value in case["ground_truth"].items()}
            prompt, target_ids = worker._generate_prompt(history)
            if not prompt or not target_ids:
                prompt_failures += 1
                continue
            if set(target_ids) != set(truth):
                target_mismatches += 1
            started = time.perf_counter()
            beliefs = worker.get_response(prompt, target_ids)
            latency_ms = 1000.0 * (time.perf_counter() - started)
            for vehicle_id in target_ids:
                true_label = truth.get(vehicle_id)
                probs = normalize(beliefs[vehicle_id]) if beliefs and vehicle_id in beliefs else None
                style = worker._summarize_vehicle_history_relative(history, vehicle_id) or {}
                headway = style.get("headway_mean")
                leader_visible = headway is not None and headway < 4.0
                record = dict(
                    episode=case["case_id"], case_id=case["case_id"],
                    seed=case.get("seed"), vehicle_id=vehicle_id, query_index=0,
                    sim_time=None, latency_ms=latency_ms,
                    truth=true_label, probs=probs,
                    pred=argmax_label(probs) if probs is not None else None,
                    max_prob=max(probs) if probs is not None else None,
                    margin=(sorted(probs, reverse=True)[0] - sorted(probs, reverse=True)[1]
                            if probs is not None else None),
                    correct=(argmax_label(probs) == true_label) if probs is not None else False,
                    binary_hard=((argmax_label(probs) == "Aggressive") == (true_label == "Aggressive"))
                    if probs is not None else False,
                    binary_threshold=((probs[2] >= 0.5) == (true_label == "Aggressive"))
                    if probs is not None else False,
                    is_key=False,
                    position=latest_position(history[-1], vehicle_id)[0],
                    dx=latest_position(history[-1], vehicle_id)[1],
                    lane_id=latest_position(history[-1], vehicle_id)[2],
                    leader_visible=leader_visible,
                    headway_mean=headway,
                    relative_speed_mean=style.get("relative_speed_mean"),
                    stream_relative_speed_mean=style.get("stream_relative_speed_mean"),
                    prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    error=worker.last_error,
                )
                if probs is not None and true_label in LABELS:
                    record["brier"] = brier(probs, true_label)
                    record["log_loss"] = log_loss(probs, true_label)
                else:
                    record["brier"] = None
                    record["log_loss"] = None
                records.append(record)
            if pause_s:
                time.sleep(pause_s)
    finally:
        if worker._client is not None:
            worker._client.close()
    return records, dict(prompt_failures=prompt_failures,
                         target_mismatches=target_mismatches)


def valid(records: Sequence[dict]) -> List[dict]:
    return [record for record in records
            if record.get("probs") is not None and record.get("truth") in LABELS]


def summarize(records: Sequence[dict], health: dict) -> dict:
    usable = valid(records)
    result = dict(
        health=health,
        n_records=len(records),
        n_usable=len(usable),
        errors=sum(record.get("error") is not None for record in records),
        accuracy=(sum(record["correct"] for record in usable) / len(usable)) if usable else None,
        metrics=metrics(usable) if usable else {},
        latency_ms=(float(np.mean([record["latency_ms"] for record in usable]))
                    if usable else None),
        by_position={},
        by_lane={},
        by_leader_visibility={},
        by_truth_distribution=dict(Counter(record["truth"] for record in usable)),
    )
    for position in sorted({record["position"] for record in usable}):
        part = [record for record in usable if record["position"] == position]
        result["by_position"][position] = dict(
            n=len(part), accuracy=sum(record["correct"] for record in part) / len(part))
    for lane in sorted({str(record["lane_id"]) for record in usable}):
        part = [record for record in usable if str(record["lane_id"]) == lane]
        result["by_lane"][lane] = dict(
            n=len(part), accuracy=sum(record["correct"] for record in part) / len(part))
    for label, predicate in (("leader_visible", lambda record: record.get("leader_visible")),
                             ("leaderless", lambda record: not record.get("leader_visible"))):
        part = [record for record in usable if predicate(record)]
        result["by_leader_visibility"][label] = dict(
            n=len(part),
            accuracy=(sum(record["correct"] for record in part) / len(part)) if part else None,
            predicted={name: sum(record["pred"] == name for record in part) for name in LABELS},
        )
    return result


def render_markdown(summaries: Dict[str, dict], pair: dict) -> str:
    labels = LABELS
    lines = ["# DF_Cruise offline query-level type recognition", ""]
    lines.append("Frozen 20-frame histories, one LLM query per case, "
                 "qwen-flash-2025-07-28, temperature 0, thinking off, timeout 10 s, "
                 "no retries. Frames are 0.5 s apart and the worker uses dt=0.5. "
                 "The dataset advances traffic around a fixed ego observer, so "
                 "ego-relative speed is biased; leader-relative evidence is the "
                 "primary cue for v6b.")
    lines.append("")

    lines.append("## Query-level accuracy")
    lines.append("")
    lines.append("| Version | Usable / total | Accuracy | Macro-F1 | Binary hard | "
                 "Binary threshold | Brier | NLL | Mean confidence | Latency mean ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in summaries.items():
        values = summary["metrics"]
        if not values.get("n"):
            lines.append(f"| {name} | 0 / {summary['n_records']} | NA | NA | NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {name} | {summary['n_usable']} / {summary['n_records']} | "
            f"{values['accuracy']:.3f} | {values['macro_f1']:.3f} | "
            f"{values['binary_hard']['accuracy']:.3f} | "
            f"{values['binary_threshold']['accuracy']:.3f} | {values['brier']:.3f} | "
            f"{values['log_loss']:.3f} | {values['mean_confidence']:.3f} | "
            f"{summary['latency_ms']:.1f} |")
    lines.append("")

    lines.append("## Per-truth-class accuracy")
    lines.append("")
    lines.append("| Version | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, summary in summaries.items():
        for label in labels:
            values = summary["metrics"].get("by_truth", {}).get(label)
            if not values or not values.get("n"):
                continue
            lines.append(f"| {name} | {label} | {values['n']} | {values['accuracy']:.3f} | "
                         f"{values['mean_confidence']:.3f} | {values['mean_p_true']:.3f} | "
                         f"{values['brier']:.3f} |")
    lines.append("")

    lines.append("## Accuracy by position")
    lines.append("")
    lines.append("| Version | Position | n | Accuracy |")
    lines.append("|---|---|---:|---:|")
    for name, summary in summaries.items():
        for position, values in summary["by_position"].items():
            lines.append(f"| {name} | {position} | {values['n']} | {values['accuracy']:.3f} |")
    lines.append("")

    lines.append("## Accuracy by lane")
    lines.append("")
    lines.append("| Version | Lane | n | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for name, summary in summaries.items():
        for lane, values in summary["by_lane"].items():
            lines.append(f"| {name} | {lane} | {values['n']} | {values['accuracy']:.3f} |")
    lines.append("")

    lines.append("## Accuracy by own-leader visibility")
    lines.append("")
    lines.append("| Version | Subset | n | Accuracy | Predicted counts |")
    lines.append("|---|---|---:|---:|---|")
    for name, summary in summaries.items():
        for subset, values in summary["by_leader_visibility"].items():
            accuracy = "NA" if values["accuracy"] is None else f"{values['accuracy']:.3f}"
            lines.append(f"| {name} | {subset} | {values['n']} | {accuracy} | "
                         f"{values['predicted']} |")
    lines.append("")

    if pair:
        lines.append("## Paired v5 vs v6b")
        lines.append("")
        lines.append(f"Paired query records: {pair['n_pairs']} "
                     f"(only v5: {pair['only_v5']}, only v6b: {pair['only_v6']}).")
        lines.append("")
        lines.append("| Metric | Paired n | v6b better | v6b worse | exact McNemar p |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric_name in ("three_class", "binary_hard", "binary_threshold"):
            values = pair[metric_name]
            lines.append(f"| {metric_name} | {values['n_pairs']} | {values['v6_better']} | "
                         f"{values['v6_worse']} | {values['exact_mcnemar_p']:.6g} |")
        lines.append("")
        lines.append(f"Accuracy difference (v6b - v5, case-clustered): "
                     f"{pair['accuracy_difference']['mean_difference']:+.3f} "
                     f"[{pair['accuracy_difference']['ci_low']:+.3f}, "
                     f"{pair['accuracy_difference']['ci_high']:+.3f}]")
        lines.append("")
        lines.append(f"Brier difference: "
                     f"{pair['brier_difference']['mean_difference']:+.3f} "
                     f"[{pair['brier_difference']['ci_low']:+.3f}, "
                     f"{pair['brier_difference']['ci_high']:+.3f}]")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="simulation_results/vehicle_type_dataset_v2_20f.jsonl")
    parser.add_argument("--output-dir", default="simulation_results/ablation_v2/cruise_offline_type_eval_001")
    parser.add_argument("--model", default="qwen-flash-2025-07-28")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--pause-s", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=0.5,
                        help="frame spacing of the dataset (seconds)")
    parser.add_argument("--versions", nargs="+", choices=tuple(VERSIONS), default=list(VERSIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = read_cases(Path(args.dataset))
    print(f"dataset={args.dataset} cases={len(cases)} versions={args.versions}")
    if args.dry_run:
        worker = LLMReasoningWorker(prompt_version=VERSIONS[args.versions[0]], dt=args.dt)
        mismatches = 0
        for case in cases:
            prompt, target_ids = worker._generate_prompt(case["history"])
            truth = {int(key) for key in case["ground_truth"]}
            if prompt is None or set(target_ids) != truth:
                mismatches += 1
                print("mismatch", case["case_id"], sorted(target_ids), sorted(truth))
        print(f"target mismatches: {mismatches}/{len(cases)}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    all_records = {}
    for version_key in args.versions:
        records, health = make_records(cases, version_key, args.model, args.temperature,
                                       args.timeout_s, args.pause_s, args.dt)
        summary = summarize(records, health)
        summaries[version_key] = summary
        all_records[version_key] = records
        version_dir = output_dir / version_key
        version_dir.mkdir(parents=True, exist_ok=True)
        with (version_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (version_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{version_key}: usable={summary['n_usable']}/{summary['n_records']} "
              f"accuracy={summary['accuracy']} errors={summary['errors']}")

    pair = None
    if {"v5", "v6b"}.issubset(all_records):
        left = valid(all_records["v5"])
        right = valid(all_records["v6b"])
        pair = dict(
            n_pairs=len({(record["case_id"], record["vehicle_id"]) for record in left}
                        & {(record["case_id"], record["vehicle_id"]) for record in right}),
            only_v5=len({(record["case_id"], record["vehicle_id"]) for record in left}
                        - {(record["case_id"], record["vehicle_id"]) for record in right}),
            only_v6=len({(record["case_id"], record["vehicle_id"]) for record in right}
                        - {(record["case_id"], record["vehicle_id"]) for record in left}),
            three_class=paired_metric(left, right, "correct"),
            binary_hard=paired_metric(left, right, "binary_hard"),
            binary_threshold=paired_metric(left, right, "binary_threshold"),
            accuracy_difference=paired_mean_difference(left, right, "correct"),
            brier_difference=paired_mean_difference(left, right, "brier"),
        )
    combined = dict(dataset=args.dataset, model=args.model, summaries=summaries, pair=pair)
    (output_dir / "summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(summaries, pair), encoding="utf-8")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
