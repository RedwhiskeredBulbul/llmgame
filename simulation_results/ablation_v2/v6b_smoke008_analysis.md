# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v6b | 2 | 6 | 6 | 0.333 | 0.500 | 0.333 (1.000) | 0.500 | 1.000 | 6.889 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| df_v6b | 6 | 0.333 | 0.333 | 0.333 | 1.000 | 0.500 (2) | 0.250 (4) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| df_v6b | 2 (2) | True | 2 | 0 | 0 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| df_v6b | 0.25 | 2 | 0.500 | 0.770 |
| df_v6b | 0.50 | 3 | 0.333 | 1.000 |
| df_v6b | 0.75 | 4 | 0.250 | 1.070 |
| df_v6b | 0.90 | 5 | 0.200 | 1.172 |
| df_v6b | 1.00 | 6 | 0.333 | 1.000 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v6b | all | 6 | 0.333 | 0.300 | 0.333 (1.000) | 0.500 | 0.500 | 1.000 | 6.889 |
| df_v6b | key | 2 | 0.500 | 0.222 | 0.500 (1.000) | 1.000 | 1.000 | 0.680 | 0.916 |
| df_v6b | background | 4 | 0.250 | 0.133 | 0.250 (1.000) | 0.250 | 0.250 | 1.160 | 9.875 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| df_v6b | Cautious | 1 | 1.000 | 0.800 | 0.800 | 0.080 |
| df_v6b | Normal | 3 | 0.000 | 0.800 | 0.133 | 1.400 |
| df_v6b | Aggressive | 2 | 0.500 | 0.700 | 0.350 | 0.860 |

## Paired comparisons

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v6b | behind | 6 | 0.333 | 0.333 | 1.000 |
