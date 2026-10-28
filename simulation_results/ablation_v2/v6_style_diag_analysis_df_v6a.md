# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 24 | 339 | 73 | 0.348 | 0.469 | 0.349 (0.965) | 0.796 | 0.913 | 2.000 |
| df_v6 | 24 | 330 | 73 | 0.336 | 0.452 | 0.191 (0.206) | 0.482 | 0.720 | 1.382 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | 73 | 0.370 | 0.370 | 0.370 | 0.974 | 0.500 (24) | 0.306 (49) |
| df_v6 | 73 | 0.205 | 0.342 | 0.329 | 0.794 | 0.083 (24) | 0.265 (49) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| df_v5 | 24 (24) | True | 121 | 8 | 8 | 0 |
| df_v6 | 24 (24) | True | 114 | 4 | 4 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| df_v5 | 0.25 | 85 | 0.353 | 1.022 |
| df_v5 | 0.50 | 170 | 0.453 | 0.854 |
| df_v5 | 0.75 | 254 | 0.425 | 0.871 |
| df_v5 | 0.90 | 305 | 0.367 | 0.915 |
| df_v5 | 1.00 | 339 | 0.348 | 0.913 |
| df_v6 | 0.25 | 82 | 0.220 | 0.882 |
| df_v6 | 0.50 | 165 | 0.315 | 0.777 |
| df_v6 | 0.75 | 248 | 0.363 | 0.737 |
| df_v6 | 0.90 | 297 | 0.367 | 0.725 |
| df_v6 | 1.00 | 330 | 0.336 | 0.720 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v5 | all | 339 | 0.348 | 0.267 | 0.349 (0.965) | 0.796 | 0.796 | 0.913 | 2.000 |
| df_v5 | key | 113 | 0.398 | 0.534 | 0.408 (0.912) | 1.000 | 1.000 | 0.748 | 1.224 |
| df_v5 | background | 226 | 0.323 | 0.170 | 0.321 (0.991) | 0.695 | 0.695 | 0.996 | 2.388 |
| df_v6 | all | 330 | 0.336 | 0.337 | 0.191 (0.206) | 0.482 | 0.697 | 0.720 | 1.382 |
| df_v6 | key | 110 | 0.473 | 0.435 | 0.167 (0.055) | 0.482 | 0.891 | 0.634 | 1.046 |
| df_v6 | background | 220 | 0.268 | 0.249 | 0.194 (0.282) | 0.482 | 0.600 | 0.763 | 1.550 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| df_v5 | Cautious | 102 | 0.980 | 0.799 | 0.785 | 0.091 |
| df_v5 | Normal | 159 | 0.044 | 0.722 | 0.164 | 1.255 |
| df_v5 | Aggressive | 78 | 0.141 | 0.762 | 0.171 | 1.290 |
| df_v6 | Cautious | 101 | 0.307 | 0.425 | 0.330 | 0.702 |
| df_v6 | Normal | 149 | 0.302 | 0.416 | 0.317 | 0.728 |
| df_v6 | Aggressive | 80 | 0.438 | 0.438 | 0.321 | 0.727 |

## Paired comparisons

### df_v6a

Trajectory identity: {'compared_episodes': 24, 'compared_steps': 2526, 'mismatch_steps': 0, 'max_position_delta_m': 0.0, 'max_speed_delta_ms': 0.0, 'identical': True}

| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 339 | 330 | 0.348 | 0.336 | -0.012 | 0.913 | 0.720 | -0.193 | 2.000 | 1.382 | -0.617 |
| key | 113 | 110 | 0.398 | 0.473 | +0.074 | 0.748 | 0.634 | -0.114 | 1.224 | 1.046 | -0.178 |
| background | 226 | 220 | 0.323 | 0.268 | -0.055 | 0.996 | 0.763 | -0.233 | 2.388 | 1.550 | -0.837 |

| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| all | three_class | 71 | 76 | 36 | 27 | 0.741592 |
| all | binary_hard | 33 | 128 | 36 | 27 | 2.07536e-14 |
| all | binary_threshold | 6 | 35 | 36 | 27 | 4.8736e-06 |
| key | three_class | 31 | 26 | 12 | 9 | 0.596642 |
| key | binary_hard | 0 | 55 | 12 | 9 | 5.55112e-17 |
| key | binary_threshold | 0 | 11 | 12 | 9 | 0.000976562 |
| background | three_class | 40 | 50 | 24 | 18 | 0.342833 |
| background | binary_hard | 33 | 73 | 24 | 18 | 0.000128221 |
| background | binary_threshold | 6 | 24 | 24 | 18 | 0.00143091 |

Accuracy difference (v6 - v5, episode-clustered): {'n_episodes': 24, 'mean_difference': -0.059259259259259255, 'ci_low': -0.20278356481481477, 'ci_high': 0.08703703703703704}
Brier difference (v6 - v5): {'n_episodes': 24, 'mean_difference': -0.14484975395927802, 'ci_low': -0.3042548030686803, 'ci_high': 0.01413288348094559}
Log-loss difference (v6 - v5): {'n_episodes': 24, 'mean_difference': -0.5770223212908764, 'ci_low': -1.4728903338964616, 'ci_high': 0.2174865482398473}

| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ahead | 2 | 1 | 0.000 | 1.000 | +1.000 | 1.460 | 0.135 |
| behind | 337 | 329 | 0.350 | 0.334 | -0.016 | 0.910 | 0.722 |

| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Cautious | 102 | 101 | 0.980 | 0.307 | -0.673 | 0.091 | 0.702 | 1/65 | 1.81604e-18 |
| Normal | 159 | 149 | 0.044 | 0.302 | +0.258 | 1.255 | 0.728 | 38/2 | 1.49339e-09 |
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
| 10221 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10222 | Normal | Cautious | 0.100 | Aggressive | 0.330 |
| 10223 | Normal | Cautious | 0.100 | Aggressive | 0.330 |

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v5 | ahead | 2 | 0.000 | 0.000 | 1.460 |
| df_v5 | behind | 337 | 0.350 | 0.351 | 0.910 |
| df_v6 | ahead | 1 | 1.000 | 1.000 | 0.135 |
| df_v6 | behind | 329 | 0.334 | 0.179 | 0.722 |
