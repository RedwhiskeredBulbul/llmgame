# Core-module ablation experiment

The experiment is implemented by `ablation_study.py`. Every method is run on
the same scenario/seed pairs. Numerical kernels receive one untimed warm-up
step per scenario before measurements begin.

## Variants

| CLI name | Paper name | Operational definition |
|---|---|---|
| `deterministic_type` | Deterministic Type | Every observed surrounding vehicle is assigned `[0, 1, 0]` (Normal). |
| `uniform_belief` | Uniform Belief | Every observed surrounding vehicle is assigned `[1/3, 1/3, 1/3]`. |
| `without_fast_slow` | w/o Fast--Slow | LLM inference is synchronous. Its selected target-lane identifier is applied directly as the only high-level candidate; the request time is included in decision latency. |
| `full_model` | Full Model | The rolling-history LLM runs asynchronously; its probabilistic beliefs are consumed by the fast game solver without blocking it. |

Both LLM variants use the same history length, inference interval, model,
probability parser, temporal smoothing, and fallback uniform belief. This keeps
the fast--slow coordination mechanism as the controlled difference.

## Default scenarios and success

- `MergeEnv`: the ego vehicle travels on the main road and interacts with a
  vehicle entering from the access ramp. Success requires collision-free
  traversal of the merge area and normal road-end termination.
- `HighwayEnv`: four-lane dense highway driving. Success means collision-free
  survival to termination or the configured evaluation horizon.

## Metrics

- SR is the percentage of valid episodes that complete the scenario objective
  without collision.
- Min TTC is first computed per episode using heading-aligned vehicle geometry,
  then averaged across episodes. No closing encounter is capped at 50 s and a
  collision is 0 s.
- Average Speed is the mean of episode-average ego speeds.
- Acceleration Variance is the mean of within-episode population variances.
- P95 Decision Latency pools step latencies. A step starts before belief
  handling and ends after low-level acceleration/steering generation.

## Running

Set the API key only in the current shell; never add it to source files:

```powershell
$env:DASHSCOPE_API_KEY = Read-Host "DashScope API key" -MaskInput
```

Recommended paper run with 30 paired seeds:

```powershell
python -m experiments.ablation `
  --variants deterministic_type uniform_belief without_fast_slow full_model `
  --scenarios MergeEnv HighwayEnv `
  --seeds 0:30 `
  --max-steps 500 `
  --history-len 20 `
  --llm-interval-steps 10 `
  --model qwen-turbo `
  --output-dir simulation_results/ablation_30seeds
```

For a local pipeline check that makes no API calls:

```powershell
python -m experiments.ablation `
  --variants deterministic_type uniform_belief `
  --scenarios MergeEnv HighwayEnv `
  --seeds 0 `
  --max-steps 5 `
  --output-dir simulation_results/ablation_smoke
```

The output directory contains:

- `manifest.json`: protocol and exact arguments, without API secrets.
- `episodes.csv`: one row per scenario/seed/variant.
- `step_metrics.csv`: raw step measurements for auditing and new plots.
- `scenario_summary.csv`: metrics split by scenario.
- `ablation_summary.csv`: the four rows aggregated for the paper table.
- `ablation_table_rows.tex`: ready-to-paste LaTeX table rows.

Failed episodes are reported separately and excluded from metric aggregation;
the program returns a non-zero exit status if any episode failed.
