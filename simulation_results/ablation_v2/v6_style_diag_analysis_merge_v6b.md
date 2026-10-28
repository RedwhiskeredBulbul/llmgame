# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_v5 | 24 | 156 | 72 | 0.500 | 0.442 | 0.493 (0.949) | 0.590 | 0.753 | 2.870 |
| merge_v6b | 24 | 141 | 72 | 0.801 | 0.433 | 0.824 (0.929) | 0.865 | 0.334 | 0.842 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| merge_v5 | 72 | 0.528 | 0.528 | 0.528 | 0.750 | 0.750 (24) | 0.417 (48) |
| merge_v6b | 72 | 0.806 | 0.819 | 0.806 | 0.340 | 0.958 (24) | 0.729 (48) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| merge_v5 | 24 (24) | True | 52 | 0 | 0 | 0 |
| merge_v6b | 24 (24) | True | 47 | 0 | 0 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| merge_v5 | 0.25 | 39 | 0.462 | 0.928 |
| merge_v5 | 0.50 | 78 | 0.513 | 0.795 |
| merge_v5 | 0.75 | 117 | 0.496 | 0.799 |
| merge_v5 | 0.90 | 140 | 0.500 | 0.769 |
| merge_v5 | 1.00 | 156 | 0.500 | 0.753 |
| merge_v6b | 0.25 | 35 | 0.914 | 0.175 |
| merge_v6b | 0.50 | 70 | 0.943 | 0.151 |
| merge_v6b | 0.75 | 106 | 0.868 | 0.256 |
| merge_v6b | 0.90 | 127 | 0.827 | 0.306 |
| merge_v6b | 1.00 | 141 | 0.801 | 0.334 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_v5 | all | 156 | 0.500 | 0.418 | 0.493 (0.949) | 0.590 | 0.558 | 0.753 | 2.870 |
| merge_v5 | key | 52 | 0.731 | 0.707 | 0.731 (1.000) | 1.000 | 1.000 | 0.336 | 0.564 |
| merge_v5 | background | 104 | 0.385 | 0.185 | 0.365 (0.923) | 0.385 | 0.337 | 0.962 | 4.022 |
| merge_v6b | all | 141 | 0.801 | 0.781 | 0.824 (0.929) | 0.865 | 0.879 | 0.334 | 0.842 |
| merge_v6b | key | 47 | 0.915 | 0.926 | 0.951 (0.872) | 1.000 | 1.000 | 0.220 | 0.434 |
| merge_v6b | background | 94 | 0.745 | 0.608 | 0.767 (0.957) | 0.798 | 0.819 | 0.391 | 1.045 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| merge_v5 | Cautious | 69 | 0.319 | 0.805 | 0.308 | 1.062 |
| merge_v5 | Normal | 30 | 0.133 | 0.716 | 0.263 | 1.068 |
| merge_v5 | Aggressive | 57 | 0.912 | 0.693 | 0.675 | 0.214 |
| merge_v6b | Cautious | 61 | 0.721 | 0.714 | 0.574 | 0.432 |
| merge_v6b | Normal | 29 | 0.621 | 0.572 | 0.446 | 0.568 |
| merge_v6b | Aggressive | 51 | 1.000 | 0.801 | 0.801 | 0.084 |

## Paired comparisons

### merge_v6b

Trajectory identity: {'compared_episodes': 24, 'compared_steps': 1817, 'mismatch_steps': 0, 'max_position_delta_m': 0.0, 'max_speed_delta_ms': 0.0, 'identical': True}

| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 156 | 141 | 0.500 | 0.801 | +0.301 | 0.753 | 0.334 | -0.420 | 2.870 | 0.842 | -2.028 |
| key | 52 | 47 | 0.731 | 0.915 | +0.184 | 0.336 | 0.220 | -0.116 | 0.564 | 0.434 | -0.130 |
| background | 104 | 94 | 0.385 | 0.745 | +0.360 | 0.962 | 0.391 | -0.571 | 4.022 | 1.045 | -2.977 |

| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| all | three_class | 39 | 1 | 27 | 12 | 7.45786e-11 |
| all | binary_hard | 32 | 0 | 27 | 12 | 4.65661e-10 |
| all | binary_threshold | 34 | 0 | 27 | 12 | 1.16415e-10 |
| key | three_class | 12 | 1 | 9 | 4 | 0.00341797 |
| key | binary_hard | 0 | 0 | 9 | 4 | 1 |
| key | binary_threshold | 0 | 0 | 9 | 4 | 1 |
| background | three_class | 27 | 0 | 18 | 8 | 1.49012e-08 |
| background | binary_hard | 32 | 0 | 18 | 8 | 4.65661e-10 |
| background | binary_threshold | 34 | 0 | 18 | 8 | 1.16415e-10 |

Accuracy difference (v6 - v5, episode-clustered): {'n_episodes': 24, 'mean_difference': 0.27777777777777773, 'ci_low': 0.17361111111111108, 'ci_high': 0.3819444444444444}
Brier difference (v6 - v5): {'n_episodes': 24, 'mean_difference': -0.4054614027777778, 'ci_low': -0.5366817961805554, 'ci_high': -0.27152642638888896}
Log-loss difference (v6 - v5): {'n_episodes': 24, 'mean_difference': -1.8481106106582843, 'ci_low': -4.001764612168377, 'ci_high': 0.0784440276217383}

| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ahead | 104 | 94 | 0.385 | 0.745 | +0.360 | 0.962 | 0.391 |
| behind | 52 | 47 | 0.731 | 0.915 | +0.184 | 0.336 | 0.220 |

| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Cautious | 69 | 61 | 0.319 | 0.721 | +0.402 | 1.062 | 0.432 | 25/1 | 8.04663e-07 |
| Normal | 30 | 29 | 0.133 | 0.621 | +0.487 | 1.068 | 0.568 | 13/0 | 0.000244141 |
| Aggressive | 57 | 51 | 0.912 | 1.000 | +0.088 | 0.214 | 0.084 | 1/0 | 1 |

| Key seed | Truth | v5 first | v5 p_true | v6 first | v6 p_true |
|---:|---|---|---:|---|---:|
| 10300 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10301 | Cautious | Cautious | 0.800 | Normal | 0.330 |
| 10302 | Normal | Normal | 0.670 | Normal | 0.670 |
| 10303 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 10304 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 10305 | Cautious | Cautious | 0.800 | Cautious | 0.700 |
| 10306 | Normal | Cautious | 0.100 | Normal | 0.400 |
| 10307 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 10308 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10309 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 10310 | Aggressive | Aggressive | 0.800 | Aggressive | 0.800 |
| 10311 | Aggressive | Aggressive | 0.800 | Aggressive | 1.000 |
| 10312 | Aggressive | Aggressive | 0.800 | Aggressive | 1.000 |
| 10313 | Normal | Cautious | 0.200 | Normal | 0.600 |
| 10314 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 10315 | Normal | Cautious | 0.200 | Normal | 0.600 |
| 10316 | Aggressive | Aggressive | 0.800 | Aggressive | 1.000 |
| 10317 | Normal | Cautious | 0.300 | Normal | 0.600 |
| 10318 | Normal | Normal | 0.670 | Normal | 0.600 |
| 10319 | Aggressive | Aggressive | 0.800 | Aggressive | 0.800 |
| 10320 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 10321 | Cautious | Cautious | 0.800 | Cautious | 0.700 |
| 10322 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10323 | Cautious | Cautious | 0.800 | Cautious | 0.800 |

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| merge_v5 | ahead | 104 | 0.385 | 0.365 | 0.962 |
| merge_v5 | behind | 52 | 0.731 | 0.731 | 0.336 |
| merge_v6b | ahead | 94 | 0.745 | 0.767 | 0.391 |
| merge_v6b | behind | 47 | 0.915 | 0.951 | 0.220 |
