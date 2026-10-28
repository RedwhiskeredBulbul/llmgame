"""Query-level type-recognition report for completed simulation runs.

This is a read-only analysis.  It loads the ``full_model`` episodes of one or
more runs, reuses the record construction in :mod:`style_diagnostics`, and
reports query-level and vehicle-level recognition separately for the key
vehicle and the background vehicles.

Usage::

    python -m experiments.ablation_v2.query_level_report \
        --run cutin_v5_50=simulation_results/.../run_001 \
        --run cutin_v6b_50=simulation_results/.../run_002 \
        --markdown simulation_results/.../query_level.md \
        --output simulation_results/.../query_level.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from experiments.ablation_v2.style_diagnostics import load_run, metrics, subset


def _latency(records: Sequence[dict]) -> dict:
    values = [float(record["latency_ms"]) for record in records
              if record.get("latency_ms") is not None]
    if not values:
        return dict(n=0, mean=None, median=None, p95=None, p99=None)
    return dict(
        n=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )


def _first(vehicle_records: Sequence[dict], **conditions) -> dict:
    selected = [record for record in vehicle_records
                if all(record.get(key) == value for key, value in conditions.items())]
    if not selected:
        return dict(n=0, accuracy=None, brier=None)
    correct = sum(record["first_correct"] for record in selected)
    briers = [
        sum((prob - float(index == ("Cautious", "Normal", "Aggressive").index(record["truth"]))) ** 2
            for index, prob in enumerate(record["first_probs"]))
        for record in selected
    ]
    return dict(n=len(selected), correct=correct, accuracy=correct / len(selected),
                brier=float(np.mean(briers)))


def _query_index(records: Sequence[dict]) -> List[dict]:
    rows = []
    for index in sorted({record["query_index"] for record in records}):
        part = [record for record in records if record["query_index"] == index]
        rows.append(dict(query_index=index, n=len(part),
                         accuracy=sum(record["correct"] for record in part) / len(part),
                         binary_hard=sum(record["binary_hard"] for record in part) / len(part)))
    return rows


def summary(path: Path) -> dict:
    run = load_run(path)
    records = run["query_records"]
    vehicles = run["vehicle_records"]
    result = dict(
        path=str(path),
        episodes=run["episodes"],
        expected_episodes=run["expected_episodes"],
        response_events=run["all_response_events"],
        error_events=run["error_events"],
        empty_probability_events=run["empty_probability_events"],
        missing_truth=run["missing_truth"],
        query_samples=len(records),
        vehicle_episodes=len(vehicles),
        subsets={},
        vehicle_first={
            "all": _first(vehicles),
            "key": _first(vehicles, is_key=True),
            "background": _first(vehicles, is_key=False),
        },
        by_truth=metrics(records)["by_truth"] if records else {},
        by_query_index=_query_index(records),
        latency=_latency(records),
    )
    for name, part in (
        ("all", records),
        ("key", subset(records, is_key=True)),
        ("background", subset(records, is_key=False)),
        ("ahead", subset(records, position="ahead")),
        ("behind", subset(records, position="behind")),
    ):
        result["subsets"][name] = metrics(part)
    result["confusion"] = result["subsets"]["all"].get("confusion")
    return result


def _fmt(value, digits=3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(results: Dict[str, dict]) -> str:
    lines = ["# Query-level type-recognition report", ""]
    lines.append("## Run health")
    lines.append("")
    lines.append("| Run | Episodes (expected) | Complete | Response events | Errors | "
                 "Empty probabilities | Missing truth | Query samples | Vehicle-episodes |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for name, result in results.items():
        expected = result["expected_episodes"]
        complete = (expected is not None and result["episodes"] == expected)
        lines.append(
            f"| {name} | {result['episodes']} ({expected if expected is not None else 'NA'}) | "
            f"{complete} | {result['response_events']} | {result['error_events']} | "
            f"{result['empty_probability_events']} | {result['missing_truth']} | "
            f"{result['query_samples']} | {result['vehicle_episodes']} |")
    lines.append("")

    lines.append("## Query-level recognition")
    lines.append("")
    lines.append("| Run | Subset | n | Accuracy | Macro-F1 | Binary hard | Binary threshold | "
                 "Brier | NLL | Mean confidence | Conf>=0.5 acc (coverage) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, result in results.items():
        for subset_name in ("all", "key", "background"):
            values = result["subsets"].get(subset_name) or {}
            if not values.get("n"):
                continue
            coverage = values["coverage"]["0.50"]
            lines.append(
                f"| {name} | {subset_name} | {values['n']} | {_fmt(values['accuracy'])} | "
                f"{_fmt(values['macro_f1'])} | {_fmt(values['binary_hard']['accuracy'])} | "
                f"{_fmt(values['binary_threshold']['accuracy'])} | {_fmt(values['brier'])} | "
                f"{_fmt(values['log_loss'])} | {_fmt(values['mean_confidence'])} | "
                f"{_fmt(coverage['accuracy'])} ({_fmt(coverage['coverage'])}) |")
    lines.append("")

    lines.append("## Vehicle-level first-query recognition")
    lines.append("")
    lines.append("| Run | Vehicles | First acc | First Brier | Key first acc | Background first acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, result in results.items():
        first = result["vehicle_first"]
        lines.append(
            f"| {name} | {result['vehicle_episodes']} | "
            f"{_fmt(first['all'].get('accuracy'))} | {_fmt(first['all'].get('brier'))} | "
            f"{_fmt(first['key'].get('accuracy'))} (n={first['key']['n']}) | "
            f"{_fmt(first['background'].get('accuracy'))} "
            f"(n={first['background']['n']}) |")
    lines.append("")

    lines.append("## Per-truth-class accuracy (query level)")
    lines.append("")
    labels = ("Cautious", "Normal", "Aggressive")
    lines.append("| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, result in results.items():
        for label in labels:
            values = result["by_truth"].get(label)
            if not values or not values.get("n"):
                continue
            lines.append(
                f"| {name} | {label} | {values['n']} | {_fmt(values['accuracy'])} | "
                f"{_fmt(values['mean_confidence'])} | {_fmt(values['mean_p_true'])} | "
                f"{_fmt(values['brier'])} |")
    lines.append("")

    lines.append("## Accuracy by query index")
    lines.append("")
    lines.append("| Run | Query index | n | Accuracy | Binary hard |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, result in results.items():
        for row in result["by_query_index"]:
            lines.append(f"| {name} | {row['query_index']} | {row['n']} | "
                         f"{_fmt(row['accuracy'])} | {_fmt(row['binary_hard'])} |")
    lines.append("")

    lines.append("## LLM latency (successful responses)")
    lines.append("")
    lines.append("| Run | n | Mean ms | Median ms | p95 ms | p99 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, result in results.items():
        value = result["latency"]
        lines.append(f"| {name} | {value['n']} | {_fmt(value['mean'], 1)} | "
                     f"{_fmt(value['median'], 1)} | {_fmt(value['p95'], 1)} | "
                     f"{_fmt(value['p99'], 1)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="name=path; repeat for multiple runs")
    parser.add_argument("--output", required=True, help="JSON summary path")
    parser.add_argument("--markdown", required=True, help="Markdown report path")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, raw_path = item.split("=", 1)
        runs[name] = summary(Path(raw_path).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    Path(args.markdown).write_text(render_markdown(runs), encoding="utf-8")
    print(json.dumps({name: dict(query_samples=value["query_samples"],
                                 all_accuracy=value["subsets"]["all"].get("accuracy"),
                                 key_accuracy=value["subsets"]["key"].get("accuracy"),
                                 background_accuracy=value["subsets"]["background"].get("accuracy"),
                                 key_first=value["vehicle_first"]["key"].get("accuracy"))
                      for name, value in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
