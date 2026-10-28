# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v6 | 2 | 6 | 6 | 0.500 | 0.500 | 0.400 (0.833) | 0.500 | 0.803 | 1.333 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| df_v6 | 6 | 0.500 | 0.500 | 0.500 | 0.803 | 1.000 (2) | 0.250 (4) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| df_v6 | 2 (2) | True | 2 | 0 | 0 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| df_v6 | 0.25 | 2 | 0.500 | 0.763 |
| df_v6 | 0.50 | 3 | 0.333 | 0.995 |
| df_v6 | 0.75 | 4 | 0.500 | 0.781 |
| df_v6 | 0.90 | 5 | 0.400 | 0.854 |
| df_v6 | 1.00 | 6 | 0.500 | 0.803 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v6 | all | 6 | 0.500 | 0.522 | 0.400 (0.833) | 0.500 | 0.500 | 0.803 | 1.333 |
| df_v6 | key | 2 | 1.000 | 0.667 | 1.000 (0.500) | 1.000 | 1.000 | 0.305 | 0.570 |
| df_v6 | background | 4 | 0.250 | 0.133 | 0.250 (1.000) | 0.250 | 0.250 | 1.051 | 1.715 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| df_v6 | Cautious | 1 | 1.000 | 0.800 | 0.800 | 0.065 |
| df_v6 | Normal | 3 | 0.333 | 0.667 | 0.200 | 1.155 |
| df_v6 | Aggressive | 2 | 0.500 | 0.650 | 0.425 | 0.643 |

## Paired comparisons

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v6 | behind | 6 | 0.500 | 0.400 | 0.803 |
