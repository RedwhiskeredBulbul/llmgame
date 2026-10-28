# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 2 | 6 | 6 | 0.500 | 0.400 (0.833) | 0.500 | 0.803 | 1.333 |
| df_v6 | 2 | 6 | 6 | 0.500 | 0.400 (0.833) | 0.500 | 0.803 | 1.333 |

## LLM response health

| Run | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---:|---:|---:|
| df_v5 | 2 | 0 | 0 | 0 |
| df_v6 | 2 | 0 | 0 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| df_v5 | 0.25 | 2 | 0.500 | 0.763 |
| df_v5 | 0.50 | 3 | 0.333 | 0.995 |
| df_v5 | 0.75 | 4 | 0.500 | 0.781 |
| df_v5 | 0.90 | 5 | 0.400 | 0.854 |
| df_v5 | 1.00 | 6 | 0.500 | 0.803 |
| df_v6 | 0.25 | 2 | 0.500 | 0.763 |
| df_v6 | 0.50 | 3 | 0.333 | 0.995 |
| df_v6 | 0.75 | 4 | 0.500 | 0.781 |
| df_v6 | 0.90 | 5 | 0.400 | 0.854 |
| df_v6 | 1.00 | 6 | 0.500 | 0.803 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | all | 6 | 0.500 | 0.522 | 0.400 (0.833) | 0.500 | 0.500 | 0.803 | 1.333 |
| df_v5 | key | 2 | 1.000 | 0.667 | 1.000 (0.500) | 1.000 | 1.000 | 0.305 | 0.570 |
| df_v5 | background | 4 | 0.250 | 0.133 | 0.250 (1.000) | 0.250 | 0.250 | 1.051 | 1.715 |
| df_v6 | all | 6 | 0.500 | 0.522 | 0.400 (0.833) | 0.500 | 0.500 | 0.803 | 1.333 |
| df_v6 | key | 2 | 1.000 | 0.667 | 1.000 (0.500) | 1.000 | 1.000 | 0.305 | 0.570 |
| df_v6 | background | 4 | 0.250 | 0.133 | 0.250 (1.000) | 0.250 | 0.250 | 1.051 | 1.715 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| df_v5 | Cautious | 1 | 1.000 | 0.800 | 0.800 | 0.065 |
| df_v5 | Normal | 3 | 0.333 | 0.667 | 0.200 | 1.155 |
| df_v5 | Aggressive | 2 | 0.500 | 0.650 | 0.425 | 0.643 |
| df_v6 | Cautious | 1 | 1.000 | 0.800 | 0.800 | 0.065 |
| df_v6 | Normal | 3 | 0.333 | 0.667 | 0.200 | 1.155 |
| df_v6 | Aggressive | 2 | 0.500 | 0.650 | 0.425 | 0.643 |

## Paired comparisons

### smoke

Trajectory identity: {'compared_episodes': 2, 'compared_steps': 50, 'mismatch_steps': 0, 'max_position_delta_m': 0.0, 'max_speed_delta_ms': 0.0, 'identical': True}

| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 6 | 6 | 0.500 | 0.500 | +0.000 | 0.803 | 0.803 | +0.000 | 1.333 | 1.333 | +0.000 |
| key | 2 | 2 | 1.000 | 1.000 | +0.000 | 0.305 | 0.305 | +0.000 | 0.570 | 0.570 | +0.000 |
| background | 4 | 4 | 0.250 | 0.250 | +0.000 | 1.051 | 1.051 | +0.000 | 1.715 | 1.715 | +0.000 |

| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| all | three_class | 0 | 0 | 0 | 0 | 1 |
| all | binary_hard | 0 | 0 | 0 | 0 | 1 |
| all | binary_threshold | 0 | 0 | 0 | 0 | 1 |
| key | three_class | 0 | 0 | 0 | 0 | 1 |
| key | binary_hard | 0 | 0 | 0 | 0 | 1 |
| key | binary_threshold | 0 | 0 | 0 | 0 | 1 |
| background | three_class | 0 | 0 | 0 | 0 | 1 |
| background | binary_hard | 0 | 0 | 0 | 0 | 1 |
| background | binary_threshold | 0 | 0 | 0 | 0 | 1 |

Accuracy difference (v6 - v5, episode-clustered): {'n_episodes': 2, 'mean_difference': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
Brier difference (v6 - v5): {'n_episodes': 2, 'mean_difference': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
Log-loss difference (v6 - v5): {'n_episodes': 2, 'mean_difference': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}

| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| behind | 6 | 6 | 0.500 | 0.500 | +0.000 | 0.803 | 0.803 |

| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Cautious | 1 | 1 | 1.000 | 1.000 | +0.000 | 0.065 | 0.065 | 0/0 | 1 |
| Normal | 3 | 3 | 0.333 | 0.333 | +0.000 | 1.155 | 1.155 | 0/0 | 1 |
| Aggressive | 2 | 2 | 0.500 | 0.500 | +0.000 | 0.643 | 0.643 | 0/0 | 1 |

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v5 | behind | 6 | 0.500 | 0.400 | 0.803 |
| df_v6 | behind | 6 | 0.500 | 0.400 | 0.803 |
