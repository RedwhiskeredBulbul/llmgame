# Query-level type-recognition report

## Run health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth | Query samples | Vehicle-episodes |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cutin_v5_50 | 50 (50) | True | 290 | 0 | 0 | 0 | 870 | 157 |
| cutin_v6b_50 | 50 (50) | True | 295 | 0 | 0 | 0 | 885 | 158 |
| cutin_v5_diag24 | 24 (24) | True | 121 | 8 | 8 | 0 | 339 | 73 |
| cutin_v6b_diag24 | 24 (24) | True | 104 | 0 | 0 | 0 | 312 | 74 |
| merge_v5_diag24 | 24 (24) | True | 52 | 0 | 0 | 0 | 156 | 72 |
| merge_v6b_diag24 | 24 (24) | True | 47 | 0 | 0 | 0 | 141 | 72 |

## Query-level recognition

| Run | Subset | n | Accuracy | Macro-F1 | Binary hard | Binary threshold | Brier | NLL | Mean confidence | Conf>=0.5 acc (coverage) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cutin_v5_50 | all | 870 | 0.493 | 0.374 | 0.775 | 0.774 | 0.701 | 1.529 | 0.738 | 0.511 (0.947) |
| cutin_v5_50 | key | 290 | 0.634 | 0.531 | 0.962 | 0.962 | 0.461 | 0.822 | 0.730 | 0.669 (0.948) |
| cutin_v5_50 | background | 580 | 0.422 | 0.225 | 0.681 | 0.679 | 0.821 | 1.883 | 0.741 | 0.432 (0.947) |
| cutin_v6b_50 | all | 885 | 0.560 | 0.532 | 0.741 | 0.756 | 0.584 | 1.174 | 0.636 | 0.582 (0.782) |
| cutin_v6b_50 | key | 295 | 0.861 | 0.860 | 0.997 | 1.000 | 0.258 | 0.489 | 0.677 | 0.911 (0.797) |
| cutin_v6b_50 | background | 590 | 0.410 | 0.333 | 0.614 | 0.634 | 0.748 | 1.516 | 0.616 | 0.414 (0.775) |
| cutin_v5_diag24 | all | 339 | 0.348 | 0.267 | 0.796 | 0.796 | 0.913 | 2.000 | 0.754 | 0.349 (0.965) |
| cutin_v5_diag24 | key | 113 | 0.398 | 0.534 | 1.000 | 1.000 | 0.748 | 1.224 | 0.704 | 0.408 (0.912) |
| cutin_v5_diag24 | background | 226 | 0.323 | 0.170 | 0.695 | 0.695 | 0.996 | 2.388 | 0.779 | 0.321 (0.991) |
| cutin_v6b_diag24 | all | 312 | 0.481 | 0.461 | 0.657 | 0.689 | 0.689 | 1.691 | 0.626 | 0.468 (0.747) |
| cutin_v6b_diag24 | key | 104 | 0.817 | 0.845 | 0.990 | 1.000 | 0.333 | 0.598 | 0.606 | 0.881 (0.644) |
| cutin_v6b_diag24 | background | 208 | 0.312 | 0.251 | 0.490 | 0.534 | 0.868 | 2.238 | 0.637 | 0.301 (0.798) |
| merge_v5_diag24 | all | 156 | 0.500 | 0.418 | 0.590 | 0.558 | 0.753 | 2.870 | 0.747 | 0.493 (0.949) |
| merge_v5_diag24 | key | 52 | 0.731 | 0.707 | 1.000 | 1.000 | 0.336 | 0.564 | 0.760 | 0.731 (1.000) |
| merge_v5_diag24 | background | 104 | 0.385 | 0.185 | 0.385 | 0.337 | 0.962 | 4.022 | 0.740 | 0.365 (0.923) |
| merge_v6b_diag24 | all | 141 | 0.801 | 0.781 | 0.865 | 0.879 | 0.334 | 0.842 | 0.716 | 0.824 (0.929) |
| merge_v6b_diag24 | key | 47 | 0.915 | 0.926 | 1.000 | 1.000 | 0.220 | 0.434 | 0.680 | 0.951 (0.872) |
| merge_v6b_diag24 | background | 94 | 0.745 | 0.608 | 0.798 | 0.819 | 0.391 | 1.045 | 0.734 | 0.767 (0.957) |

## Vehicle-level first-query recognition

| Run | Vehicles | First acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|
| cutin_v5_50 | 157 | 0.478 | 0.811 | 0.640 (n=50) | 0.402 (n=107) |
| cutin_v6b_50 | 158 | 0.563 | 0.641 | 0.960 (n=50) | 0.380 (n=108) |
| cutin_v5_diag24 | 73 | 0.370 | 0.974 | 0.500 (n=24) | 0.306 (n=49) |
| cutin_v6b_diag24 | 74 | 0.514 | 0.720 | 1.000 (n=24) | 0.280 (n=50) |
| merge_v5_diag24 | 72 | 0.528 | 0.750 | 0.750 (n=24) | 0.417 (n=48) |
| merge_v6b_diag24 | 72 | 0.806 | 0.340 | 0.958 (n=24) | 0.729 (n=48) |

## Per-truth-class accuracy (query level)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| cutin_v5_50 | Cautious | 344 | 0.983 | 0.780 | 0.772 | 0.101 |
| cutin_v5_50 | Normal | 249 | 0.024 | 0.718 | 0.163 | 1.259 |
| cutin_v5_50 | Aggressive | 277 | 0.307 | 0.703 | 0.316 | 0.944 |
| cutin_v6b_50 | Cautious | 340 | 0.815 | 0.698 | 0.631 | 0.313 |
| cutin_v6b_50 | Normal | 262 | 0.302 | 0.555 | 0.315 | 0.814 |
| cutin_v6b_50 | Aggressive | 283 | 0.495 | 0.637 | 0.401 | 0.698 |
| cutin_v5_diag24 | Cautious | 102 | 0.980 | 0.799 | 0.785 | 0.091 |
| cutin_v5_diag24 | Normal | 159 | 0.044 | 0.722 | 0.164 | 1.255 |
| cutin_v5_diag24 | Aggressive | 78 | 0.141 | 0.762 | 0.171 | 1.290 |
| cutin_v6b_diag24 | Cautious | 95 | 0.789 | 0.705 | 0.628 | 0.325 |
| cutin_v6b_diag24 | Normal | 139 | 0.360 | 0.563 | 0.330 | 0.795 |
| cutin_v6b_diag24 | Aggressive | 78 | 0.321 | 0.645 | 0.282 | 0.945 |
| merge_v5_diag24 | Cautious | 69 | 0.319 | 0.805 | 0.308 | 1.062 |
| merge_v5_diag24 | Normal | 30 | 0.133 | 0.716 | 0.263 | 1.068 |
| merge_v5_diag24 | Aggressive | 57 | 0.912 | 0.693 | 0.675 | 0.214 |
| merge_v6b_diag24 | Cautious | 61 | 0.721 | 0.714 | 0.574 | 0.432 |
| merge_v6b_diag24 | Normal | 29 | 0.621 | 0.572 | 0.446 | 0.568 |
| merge_v6b_diag24 | Aggressive | 51 | 1.000 | 0.801 | 0.801 | 0.084 |

## Accuracy by query index

| Run | Query index | n | Accuracy | Binary hard |
|---|---:|---:|---:|---:|
| cutin_v5_50 | 0 | 150 | 0.487 | 0.787 |
| cutin_v5_50 | 1 | 150 | 0.487 | 0.787 |
| cutin_v5_50 | 2 | 150 | 0.487 | 0.787 |
| cutin_v5_50 | 3 | 150 | 0.513 | 0.793 |
| cutin_v5_50 | 4 | 147 | 0.483 | 0.776 |
| cutin_v5_50 | 5 | 72 | 0.556 | 0.806 |
| cutin_v5_50 | 6 | 36 | 0.444 | 0.583 |
| cutin_v5_50 | 7 | 15 | 0.400 | 0.533 |
| cutin_v6b_50 | 0 | 150 | 0.580 | 0.680 |
| cutin_v6b_50 | 1 | 150 | 0.573 | 0.693 |
| cutin_v6b_50 | 2 | 150 | 0.587 | 0.760 |
| cutin_v6b_50 | 3 | 150 | 0.587 | 0.787 |
| cutin_v6b_50 | 4 | 150 | 0.527 | 0.787 |
| cutin_v6b_50 | 5 | 75 | 0.493 | 0.733 |
| cutin_v6b_50 | 6 | 48 | 0.500 | 0.750 |
| cutin_v6b_50 | 7 | 9 | 0.556 | 0.778 |
| cutin_v6b_50 | 8 | 3 | 0.667 | 0.667 |
| cutin_v5_diag24 | 0 | 72 | 0.375 | 0.792 |
| cutin_v5_diag24 | 1 | 69 | 0.377 | 0.797 |
| cutin_v5_diag24 | 2 | 54 | 0.333 | 0.778 |
| cutin_v5_diag24 | 3 | 51 | 0.373 | 0.804 |
| cutin_v5_diag24 | 4 | 45 | 0.356 | 0.822 |
| cutin_v5_diag24 | 5 | 36 | 0.222 | 0.750 |
| cutin_v5_diag24 | 6 | 12 | 0.333 | 0.917 |
| cutin_v6b_diag24 | 0 | 72 | 0.514 | 0.611 |
| cutin_v6b_diag24 | 1 | 72 | 0.514 | 0.611 |
| cutin_v6b_diag24 | 2 | 54 | 0.537 | 0.630 |
| cutin_v6b_diag24 | 3 | 54 | 0.444 | 0.722 |
| cutin_v6b_diag24 | 4 | 54 | 0.352 | 0.722 |
| cutin_v6b_diag24 | 5 | 6 | 0.667 | 0.833 |
| merge_v5_diag24 | 0 | 72 | 0.528 | 0.611 |
| merge_v5_diag24 | 1 | 60 | 0.500 | 0.600 |
| merge_v5_diag24 | 2 | 18 | 0.444 | 0.556 |
| merge_v5_diag24 | 3 | 3 | 0.333 | 0.333 |
| merge_v5_diag24 | 4 | 3 | 0.333 | 0.333 |
| merge_v6b_diag24 | 0 | 72 | 0.806 | 0.847 |
| merge_v6b_diag24 | 1 | 57 | 0.825 | 0.877 |
| merge_v6b_diag24 | 2 | 12 | 0.667 | 0.917 |

## LLM latency (successful responses)

| Run | n | Mean ms | Median ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| cutin_v5_50 | 870 | 1135.4 | 1097.5 | 1571.2 | 2045.5 |
| cutin_v6b_50 | 885 | 1143.2 | 1102.7 | 1551.5 | 1861.5 |
| cutin_v5_diag24 | 339 | 1045.5 | 1014.0 | 1456.5 | 1838.2 |
| cutin_v6b_diag24 | 312 | 1186.3 | 1172.8 | 1575.9 | 1673.4 |
| merge_v5_diag24 | 156 | 1101.9 | 1044.6 | 1626.1 | 2830.5 |
| merge_v6b_diag24 | 141 | 1086.4 | 1052.4 | 1429.6 | 1734.9 |
