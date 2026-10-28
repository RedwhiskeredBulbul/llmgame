# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 21 | 282 | 64 | 0.372 | 0.408 | 0.371 (0.965) | 0.755 | 0.895 | 2.095 |
| df_v6 | 21 | 279 | 64 | 0.358 | 0.394 | 0.210 (0.222) | 0.509 | 0.722 | 1.425 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 64 | 0.391 | 0.391 | 0.391 | 0.944 | 0.571 (21) | 0.302 (43) |
| df_v6 | 64 | 0.234 | 0.344 | 0.328 | 0.790 | 0.095 (21) | 0.302 (43) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| df_v5 | 21 (24) | False | 102 | 8 | 8 | 0 |
| df_v6 | 21 (24) | False | 97 | 4 | 4 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| df_v5 | 0.25 | 70 | 0.386 | 0.980 |
| df_v5 | 0.50 | 141 | 0.440 | 0.875 |
| df_v5 | 0.75 | 212 | 0.448 | 0.847 |
| df_v5 | 0.90 | 254 | 0.390 | 0.897 |
| df_v5 | 1.00 | 282 | 0.372 | 0.895 |
| df_v6 | 0.25 | 70 | 0.229 | 0.888 |
| df_v6 | 0.50 | 140 | 0.343 | 0.778 |
| df_v6 | 0.75 | 209 | 0.364 | 0.741 |
| df_v6 | 0.90 | 251 | 0.378 | 0.728 |
| df_v6 | 1.00 | 279 | 0.358 | 0.722 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | all | 282 | 0.372 | 0.284 | 0.371 (0.965) | 0.755 | 0.755 | 0.895 | 2.095 |
| df_v5 | key | 94 | 0.479 | 0.577 | 0.488 (0.915) | 1.000 | 1.000 | 0.663 | 1.103 |
| df_v5 | background | 188 | 0.319 | 0.171 | 0.317 (0.989) | 0.633 | 0.633 | 1.011 | 2.591 |
| df_v6 | all | 279 | 0.358 | 0.360 | 0.210 (0.222) | 0.509 | 0.663 | 0.722 | 1.425 |
| df_v6 | key | 93 | 0.473 | 0.438 | 0.167 (0.065) | 0.484 | 0.871 | 0.636 | 1.049 |
| df_v6 | background | 186 | 0.301 | 0.269 | 0.214 (0.301) | 0.522 | 0.559 | 0.765 | 1.613 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| df_v5 | Cautious | 89 | 0.978 | 0.799 | 0.783 | 0.096 |
| df_v5 | Normal | 115 | 0.061 | 0.716 | 0.166 | 1.246 |
| df_v5 | Aggressive | 78 | 0.141 | 0.762 | 0.171 | 1.290 |
| df_v6 | Cautious | 89 | 0.348 | 0.427 | 0.338 | 0.688 |
| df_v6 | Normal | 110 | 0.309 | 0.424 | 0.310 | 0.746 |
| df_v6 | Aggressive | 80 | 0.438 | 0.438 | 0.321 | 0.727 |

## Paired comparisons

### df

Trajectory identity: {'compared_episodes': 21, 'compared_steps': 2160, 'mismatch_steps': 0, 'max_position_delta_m': 0.0, 'max_speed_delta_ms': 0.0, 'identical': True}

| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 282 | 279 | 0.372 | 0.358 | -0.014 | 0.895 | 0.722 | -0.173 | 2.095 | 1.425 | -0.671 |
| key | 94 | 93 | 0.479 | 0.473 | -0.006 | 0.663 | 0.636 | -0.027 | 1.103 | 1.049 | -0.055 |
| background | 188 | 186 | 0.319 | 0.301 | -0.018 | 1.011 | 0.765 | -0.246 | 2.591 | 1.613 | -0.978 |

| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| all | three_class | 60 | 64 | 30 | 27 | 0.787747 |
| all | binary_hard | 33 | 94 | 30 | 27 | 5.68628e-08 |
| all | binary_threshold | 6 | 29 | 30 | 27 | 0.000116842 |
| key | three_class | 23 | 26 | 10 | 9 | 0.77545 |
| key | binary_hard | 0 | 46 | 10 | 9 | 2.84217e-14 |
| key | binary_threshold | 0 | 11 | 10 | 9 | 0.000976562 |
| background | three_class | 37 | 38 | 20 | 18 | 1 |
| background | binary_hard | 33 | 48 | 20 | 18 | 0.119274 |
| background | binary_threshold | 6 | 18 | 20 | 18 | 0.0226558 |

Accuracy difference (v6 - v5, episode-clustered): {'n_episodes': 21, 'mean_difference': -0.06666666666666665, 'ci_low': -0.22540343915343916, 'ci_high': 0.1015873015873016}
Brier difference (v6 - v5): {'n_episodes': 21, 'mean_difference': -0.12167640843050645, 'ci_low': -0.29690302730951423, 'ci_high': 0.04857802142283738}
Log-loss difference (v6 - v5): {'n_episodes': 21, 'mean_difference': -0.6030338359674351, 'ci_low': -1.5955249351272474, 'ci_high': 0.3106675309358453}

| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ahead | 2 | 1 | 0.000 | 1.000 | +1.000 | 1.460 | 0.135 |
| behind | 280 | 278 | 0.375 | 0.356 | -0.019 | 0.891 | 0.724 |

| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Cautious | 89 | 89 | 0.978 | 0.348 | -0.629 | 0.096 | 0.688 | 1/53 | 6.10623e-15 |
| Normal | 115 | 110 | 0.061 | 0.309 | +0.248 | 1.246 | 0.746 | 27/2 | 1.62423e-06 |
| Aggressive | 78 | 80 | 0.141 | 0.438 | +0.296 | 1.290 | 0.727 | 32/9 | 0.000430857 |

| Key seed | Truth | v5 first | v5 p_true | v6 first | v6 p_true |
|---:|---|---|---:|---|---:|
| 10200 | Cautious | Cautious | 0.800 | Aggressive | 0.330 |
| 10201 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10202 | Cautious | Cautious | 0.800 | Aggressive | 0.330 |
| 10203 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10204 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10205 | Aggressive | Aggressive | 0.800 | Normal | 0.300 |
| 10206 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10207 | Aggressive | Aggressive | 0.800 | Normal | 0.300 |
| 10208 | Cautious | Cautious | 0.800 | Aggressive | 0.330 |
| 10209 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10210 | Aggressive | Aggressive | 0.800 | Normal | 0.300 |
| 10211 | Cautious | Cautious | 0.800 | Aggressive | 0.330 |
| 10212 | Aggressive | Aggressive | 0.800 | Normal | 0.300 |
| 10213 | Aggressive | Aggressive | 0.800 | Normal | 0.300 |
| 10214 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10215 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10216 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10217 | Cautious | Cautious | 0.800 | Cautious | 0.400 |
| 10218 | Normal | Normal | 0.800 | Aggressive | 0.330 |
| 10219 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10220 | Aggressive | Aggressive | 0.800 | Aggressive | 0.450 |

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v5 | ahead | 2 | 0.000 | 0.000 | 1.460 |
| df_v5 | behind | 280 | 0.375 | 0.374 | 0.891 |
| df_v6 | ahead | 1 | 1.000 | 1.000 | 0.135 |
| df_v6 | behind | 278 | 0.356 | 0.197 | 0.724 |
