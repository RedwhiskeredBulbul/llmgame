# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Majority base | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cutin50_v5 | 50 | 870 | 157 | 0.493 | 0.395 | 0.511 (0.947) | 0.775 | 0.701 | 1.529 |
| cutin50_v6b | 50 | 885 | 158 | 0.560 | 0.384 | 0.582 (0.782) | 0.741 | 0.584 | 1.174 |

## Vehicle-level recognition (one prediction per vehicle-episode)

| Run | Vehicles | First acc | Mean-prob acc | Majority acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| cutin50_v5 | 157 | 0.478 | 0.478 | 0.478 | 0.811 | 0.640 (50) | 0.402 (107) |
| cutin50_v6b | 158 | 0.563 | 0.557 | 0.601 | 0.641 | 0.960 (50) | 0.380 (108) |

## LLM response health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth samples |
|---|---:|---|---:|---:|---:|---:|
| cutin50_v5 | 50 (50) | True | 290 | 0 | 0 | 0 |
| cutin50_v6b | 50 (50) | True | 295 | 0 | 0 | 0 |

## Risk-coverage curve (all query samples; select the most confident records)

| Run | Coverage | n | Accuracy | Brier |
|---|---:|---:|---:|---:|
| cutin50_v5 | 0.25 | 218 | 0.647 | 0.580 |
| cutin50_v5 | 0.50 | 435 | 0.614 | 0.613 |
| cutin50_v5 | 0.75 | 652 | 0.578 | 0.648 |
| cutin50_v5 | 0.90 | 783 | 0.535 | 0.681 |
| cutin50_v5 | 1.00 | 870 | 0.493 | 0.701 |
| cutin50_v6b | 0.25 | 221 | 0.760 | 0.383 |
| cutin50_v6b | 0.50 | 442 | 0.665 | 0.503 |
| cutin50_v6b | 0.75 | 664 | 0.589 | 0.562 |
| cutin50_v6b | 0.90 | 796 | 0.565 | 0.577 |
| cutin50_v6b | 1.00 | 885 | 0.560 | 0.584 |

## Key vs background (all query samples)

| Run | Subset | n | Accuracy | Macro-F1 | Conf>=0.5 acc (cov) | Binary hard acc | Binary threshold acc | Brier | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cutin50_v5 | all | 870 | 0.493 | 0.374 | 0.511 (0.947) | 0.775 | 0.774 | 0.701 | 1.529 |
| cutin50_v5 | key | 290 | 0.634 | 0.531 | 0.669 (0.948) | 0.962 | 0.962 | 0.461 | 0.822 |
| cutin50_v5 | background | 580 | 0.422 | 0.225 | 0.432 (0.947) | 0.681 | 0.679 | 0.821 | 1.883 |
| cutin50_v6b | all | 885 | 0.560 | 0.532 | 0.582 (0.782) | 0.741 | 0.756 | 0.584 | 1.174 |
| cutin50_v6b | key | 295 | 0.861 | 0.860 | 0.911 (0.797) | 0.997 | 1.000 | 0.258 | 0.489 |
| cutin50_v6b | background | 590 | 0.410 | 0.333 | 0.414 (0.775) | 0.614 | 0.634 | 0.748 | 1.516 |

## Per-truth-class breakdown (all query samples)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| cutin50_v5 | Cautious | 344 | 0.983 | 0.780 | 0.772 | 0.101 |
| cutin50_v5 | Normal | 249 | 0.024 | 0.718 | 0.163 | 1.259 |
| cutin50_v5 | Aggressive | 277 | 0.307 | 0.703 | 0.316 | 0.944 |
| cutin50_v6b | Cautious | 340 | 0.815 | 0.698 | 0.631 | 0.313 |
| cutin50_v6b | Normal | 262 | 0.302 | 0.555 | 0.315 | 0.814 |
| cutin50_v6b | Aggressive | 283 | 0.495 | 0.637 | 0.401 | 0.698 |

## Paired comparisons

### cutin50

Trajectory identity: {'compared_episodes': 50, 'compared_steps': 6204, 'mismatch_steps': 0, 'max_position_delta_m': 0.0, 'max_speed_delta_ms': 0.0, 'identical': True}

| Subset | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | delta Brier | v5 NLL | v6 NLL | delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 870 | 885 | 0.493 | 0.560 | +0.067 | 0.701 | 0.584 | -0.116 | 1.529 | 1.174 | -0.355 |
| key | 290 | 295 | 0.634 | 0.861 | +0.227 | 0.461 | 0.258 | -0.204 | 0.822 | 0.489 | -0.333 |
| background | 580 | 590 | 0.422 | 0.410 | -0.012 | 0.821 | 0.748 | -0.073 | 1.883 | 1.516 | -0.366 |

| Target | Metric | v6 better | v6 worse | only v5 | only v6 | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| all | three_class | 128 | 58 | 37 | 52 | 3.08688e-07 |
| all | binary_hard | 56 | 83 | 37 | 52 | 0.0270735 |
| all | binary_threshold | 40 | 54 | 37 | 52 | 0.179665 |
| key | three_class | 68 | 0 | 11 | 16 | 6.77626e-21 |
| key | binary_hard | 7 | 1 | 11 | 16 | 0.0703125 |
| key | binary_threshold | 7 | 0 | 11 | 16 | 0.015625 |
| background | three_class | 60 | 58 | 26 | 36 | 0.926704 |
| background | binary_hard | 49 | 82 | 26 | 36 | 0.00498135 |
| background | binary_threshold | 33 | 54 | 26 | 36 | 0.0314181 |

Accuracy difference (v6 - v5, episode-clustered): {'n_episodes': 50, 'mean_difference': 0.08621381886087769, 'ci_low': 0.019126202147525678, 'ci_high': 0.15217375116713353}
Brier difference (v6 - v5): {'n_episodes': 50, 'mean_difference': -0.14146567516513897, 'ci_low': -0.22116078578918275, 'ci_high': -0.0595532226327749}
Log-loss difference (v6 - v5): {'n_episodes': 50, 'mean_difference': -0.4739350666354948, 'ci_low': -0.9009486797161034, 'ci_high': -0.13442235037098563}

| Position | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ahead | 4 | 5 | 0.000 | 0.400 | +0.400 | 1.460 | 0.948 |
| behind | 866 | 880 | 0.495 | 0.561 | +0.066 | 0.697 | 0.582 |

| Truth | n v5 | n v6 | v5 acc | v6 acc | delta acc | v5 Brier | v6 Brier | v6 better/worse | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Cautious | 344 | 340 | 0.983 | 0.815 | -0.168 | 0.101 | 0.313 | 3/58 | 3.28574e-14 |
| Normal | 249 | 262 | 0.024 | 0.302 | +0.277 | 1.259 | 0.814 | 71/0 | 8.47033e-22 |
| Aggressive | 277 | 283 | 0.307 | 0.495 | +0.188 | 0.944 | 0.698 | 54/0 | 1.11022e-16 |

| Key seed | Truth | v5 first | v5 p_true | v6 first | v6 p_true |
|---:|---|---|---:|---|---:|
| 9950 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9951 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9952 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9953 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9954 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9955 | Normal | Cautious | 0.100 | Normal | 0.670 |
| 9956 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9957 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9958 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 9959 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9960 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9961 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 9962 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9963 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9964 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9965 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9966 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9967 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9968 | Normal | Cautious | 0.100 | Normal | 0.670 |
| 9969 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9970 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9971 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9972 | Normal | Cautious | 0.100 | Normal | 0.600 |
| 9973 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9974 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9975 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 9976 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9977 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9978 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9979 | Cautious | Cautious | 0.800 | Cautious | 0.900 |
| 9980 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9981 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9982 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9983 | Normal | Cautious | 0.100 | Normal | 0.450 |
| 9984 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9985 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 9986 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9987 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9988 | Normal | Cautious | 0.100 | Cautious | 0.200 |
| 9989 | Normal | Cautious | 0.100 | Normal | 0.340 |
| 9990 | Aggressive | Aggressive | 0.800 | Aggressive | 0.700 |
| 9991 | Normal | Cautious | 0.100 | Cautious | 0.200 |
| 9992 | Cautious | Cautious | 0.800 | Cautious | 0.800 |
| 9993 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9994 | Cautious | Cautious | 0.800 | Cautious | 0.900 |
| 9995 | Normal | Cautious | 0.100 | Normal | 0.540 |
| 9996 | Normal | Cautious | 0.100 | Normal | 0.670 |
| 9997 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9998 | Cautious | Cautious | 0.800 | Cautious | 0.850 |
| 9999 | Cautious | Cautious | 0.800 | Cautious | 0.850 |

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| cutin50_v5 | ahead | 4 | 0.000 | 0.000 | 1.460 |
| cutin50_v5 | behind | 866 | 0.495 | 0.513 | 0.697 |
| cutin50_v6b | ahead | 5 | 0.400 | 0.400 | 0.948 |
| cutin50_v6b | behind | 880 | 0.561 | 0.584 | 0.582 |
