"""Benchmark full JSON response latency on fixed project prompts, not token slogans."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from LLMreasoning.llmreason import LLMReasoningWorker
from LLMreasoning.vehicle_type_evaluation import synthetic_cases
from experiments.ablation_v2.config import ROOT, canonical_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='simulation_results/ablation_v2/model_benchmark_001')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--models', nargs='+', default=['qwen-flash', 'qwen3.5-flash', 'qwen3.8-flash', 'qwen-turbo'])
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Only freshly generated fictional trajectories. No dataset files are read or exported.
    singles = synthetic_cases(samples_per_class=4, history_len=20, dt=.1, seed=90817)
    cases = []
    for index in range(0, len(singles), 3):
        group = singles[index:index+3]
        history = []
        for frame in range(20):
            merged = copy.deepcopy(group[0]['history'][frame])
            for item in group[1:]:
                for field in ('ids','states','frenet_relative_coords','laneids'):
                    merged['others_info'][field] += copy.deepcopy(item['history'][frame]['others_info'][field])
            history.append(merged)
        cases.append({'history':history})
    workers = {model: LLMReasoningWorker(model=model, dt=.5, timeout_s=10., max_retries=0) for model in args.models}
    if not all(w.api_key for w in workers.values()):
        raise RuntimeError('DASHSCOPE_API_KEY not configured')
    results = []
    for trial in range(args.repeats):
        for offset in range(len(args.models)):
            model = args.models[(trial + offset) % len(args.models)]
            w = workers[model]
            prompt, ids = w._generate_prompt(cases[trial % len(cases)]['history'])
            probs = w.get_response(prompt, ids)
            row = dict(model=model, trial=trial, prompt_hash=canonical_hash(prompt),
                       latency_ms=1000.*w.last_latency_s, valid=bool(probs and set(probs)==set(ids)),
                       error=w.last_error, response_model=w.last_response_model,
                       usage=w.last_usage, response=w.last_raw_response)
            results.append(row)
            (out/'requests.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"{model} trial={trial} valid={row['valid']} latency={row['latency_ms']:.0f}ms error={w.last_error or ''}", flush=True)
    summaries = []
    for model in args.models:
        rows = [r for r in results if r['model']==model]
        valid = [r['latency_ms'] for r in rows if r['valid']]
        summaries.append(dict(model=model, trials=len(rows), valid=len(valid),
                         median_ms=float(np.median(valid)) if valid else None,
                         p95_ms=float(np.percentile(valid,95)) if valid else None))
    eligible = [r for r in summaries if r['valid']==args.repeats]
    selected = min(eligible, key=lambda r:r['median_ms'])['model'] if eligible else None
    report = dict(task='fresh fictional synthetic trajectories, complete JSON latency', selected=selected,
                  data_source='synthetic_cases(seed=90817); no existing project datasets read',
                  selection_rule='all trials parse completely, then minimum median latency',
                  time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), summaries=summaries,
                  limitations='Small sequential benchmark; not a universal fastest-model claim.')
    (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    for w in workers.values():
        if w._client:
            w._client.close()


if __name__ == '__main__':
    main()
