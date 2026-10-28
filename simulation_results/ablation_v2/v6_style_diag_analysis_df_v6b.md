# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 24 | 339 | 73 | 0.348 | 0.469 | 0.349 (0.965) | 0.796 | 0.913 | 2.000 |
| df_v6b | 24 | 312 | 74 | 0.481 | 0.446 | 0.468 (0.747) | 0.657 | 0.689 | 1.691 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 73 | 0.370 | 0.370 | 0.370 | 0.974 | 0.500 (24) | 0.306 (49) |
| df_v6b | 74 | 0.514 | 0.486 | 0.541 | 0.720 | 1.000 (24) | 0.280 (50) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| df_v5 | 24 (24) | True | 121 | 8 | 8 | 0 |
| df_v6b | 24 (24) | True | 104 | 0 | 0 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| df_v5 | 0.25 | 85 | 0.353 | 1.022 |
| df_v5 | 0.50 | 170 | 0.453 | 0.854 |
| df_v5 | 0.75 | 254 | 0.425 | 0.871 |
| df_v5 | 0.90 | 305 | 0.367 | 0.915 |
| df_v5 | 1.00 | 339 | 0.348 | 0.913 |
| df_v6b | 0.25 | 78 | 0.667 | 0.545 |
| df_v6b | 0.50 | 156 | 0.558 | 0.662 |
| df_v6b | 0.75 | 234 | 0.466 | 0.701 |
| df_v6b | 0.90 | 281 | 0.466 | 0.695 |
| df_v6b | 1.00 | 312 | 0.481 | 0.689 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | all | 339 | 0.348 | 0.267 | 0.349 (0.965) | 0.796 | 0.796 | 0.913 | 2.000 |
| df_v5 | key | 113 | 0.398 | 0.534 | 0.408 (0.912) | 1.000 | 1.000 | 0.748 | 1.224 |
| df_v5 | background | 226 | 0.323 | 0.170 | 0.321 (0.991) | 0.695 | 0.695 | 0.996 | 2.388 |
| df_v6b | all | 312 | 0.481 | 0.461 | 0.468 (0.747) | 0.657 | 0.689 | 0.689 | 1.691 |
| df_v6b | key | 104 | 0.817 | 0.845 | 0.881 (0.644) | 0.990 | 1.000 | 0.333 | 0.598 |
| df_v6b | background | 208 | 0.312 | 0.251 | 0.301 (0.798) | 0.490 | 0.534 | 0.868 | 2.238 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| df_v5 | Cautious | 102 | 0.980 | 0.799 | 0.785 | 0.091 |
| df_v5 | Normal | 159 | 0.044 | 0.722 | 0.164 | 1.255 |
| df_v5 | Aggressive | 78 | 0.141 | 0.762 | 0.171 | 1.290 |
| df_v6b | Cautious | 95 | 0.789 | 0.705 | 0.628 | 0.325 |
| df_v6b | Normal | 139 | 0.360 | 0.563 | 0.330 | 0.795 |
| df_v6b | Aggressive | 78 | 0.321 | 0.645 | 0.282 | 0.945 |

## Paired comparisons

### df_v6b

Trajectory identity: {'compared_episodes': 24, 'compared_steps': 2526, 'mismatch_steps': 0, 'max_position_delta_m': 0.0, 'max_speed_delta_ms': 0.0, 'identical': True}

| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 339 | 312 | 0.348 | 0.481 | +0.133 | 0.913 | 0.689 | -0.224 | 2.000 | 1.691 | -0.309 |
| key | 113 | 104 | 0.398 | 0.817 | +0.419 | 0.748 | 0.333 | -0.415 | 1.224 | 0.598 | -0.626 |
| background | 226 | 208 | 0.323 | 0.312 | -0.011 | 0.996 | 0.868 | -0.128 | 2.388 | 2.238 | -0.150 |

| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| all | three_class | 57 | 22 | 43 | 16 | 0.000102596 |
| all | binary_hard | 12 | 51 | 43 | 16 | 7.45972e-07 |
| all | binary_threshold | 9 | 38 | 43 | 16 | 2.4904e-05 |
| key | three_class | 42 | 2 | 14 | 5 | 1.12664e-10 |
| key | binary_hard | 0 | 1 | 14 | 5 | 1 |
| key | binary_threshold | 0 | 0 | 14 | 5 | 1 |
| background | three_class | 15 | 20 | 29 | 11 | 0.49956 |
| background | binary_hard | 12 | 50 | 29 | 11 | 1.21438e-06 |
| background | binary_threshold | 9 | 38 | 29 | 11 | 2.4904e-05 |

Accuracy difference (v6 - v5, episode-clustered): {'n_episodes': 24, 'mean_difference': 0.0982843137254902, 'ci_low': 0.003839869281045752, 'ci_high': 0.1951797385620915}
Brier difference (v6 - v5): {'n_episodes': 24, 'mean_difference': -0.1984620460861282, 'ci_low': -0.3157378404612677, 'ci_high': -0.08197044153780426}
Log-loss difference (v6 - v5): {'n_episodes': 24, 'mean_difference': -0.44801625784549143, 'ci_low': -1.357362843927869, 'ci_high': 0.4234044566883733}

| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ahead | 2 | 3 | 0.000 | 0.667 | +0.667 | 1.460 | 0.683 |
| behind | 337 | 309 | 0.350 | 0.479 | +0.129 | 0.910 | 0.689 |

| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Cautious | 102 | 95 | 0.980 | 0.789 | -0.191 | 0.091 | 0.325 | 1/19 | 4.00543e-05 |
| Normal | 159 | 139 | 0.044 | 0.360 | +0.316 | 1.255 | 0.795 | 45/3 | 1.31259e-10 |
| Aggressive | 78 | 78 | 0.141 | 0.321 | +0.179 | 1.290 | 0.945 | 11/0 | 0.000976562 |

| Key seed | Truth | v5 first | v5 p_true | v6 first | v6 p_true |
|---:|---|---|---:|---|---:|
| 10200 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 10201 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 10202 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 10203 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 10204 | Normal | Cautious | 0.100 | Normal | 0.400 |
| 10205 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10206 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 10207 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10208 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 10209 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 10210 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10211 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 10212 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10213 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10214 | Normal | Cautious | 0.100 | Normal | 0.440 |
| 10215 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 10216 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 10217 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 10218 | Normal | Normal | 0.800 | Normal | 0.670 |
| 10219 | Normal | Cautious | 0.100 | Normal | 0.700 |
| 10220 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 10221 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 10222 | Normal | Cautious | 0.100 | Normal | 0.540 |
| 10223 | Normal | Cautious | 0.100 | Normal | 0.670 |

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v5 | ahead | 2 | 0.000 | 0.000 | 1.460 |
| df_v5 | behind | 337 | 0.350 | 0.351 | 0.910 |
| df_v6b | ahead | 3 | 0.667 | 0.667 | 0.683 |
| df_v6b | behind | 309 | 0.479 | 0.465 | 0.689 |
