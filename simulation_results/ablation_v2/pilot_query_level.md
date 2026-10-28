# Query-level type-recognition report

## Run health

| Run | Episodes (expected) | Complete | Response events | Errors | Empty probabilities | Missing truth | Query samples | Vehicle-episodes |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| pilot_v5 | 6 (6) | True | 51 | 0 | 0 | 0 | 153 | 24 |
| pilot_v6b | 6 (6) | True | 46 | 0 | 0 | 0 | 138 | 23 |

## Query-level recognition

| Run | Subset | n | Accuracy | Macro-F1 | Binary hard | Binary threshold | Brier | NLL | Mean confidence | Conf>=0.5 acc (coverage) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot_v5 | all | 153 | 0.346 | 0.303 | 0.431 | 0.425 | 0.835 | 1.336 | 0.705 | 0.338 (0.869) |
| pilot_v5 | key | 39 | 0.436 | 0.424 | 0.641 | 0.692 | 0.593 | 0.958 | 0.639 | 0.484 (0.795) |
| pilot_v5 | background | 114 | 0.316 | 0.221 | 0.360 | 0.333 | 0.918 | 1.466 | 0.728 | 0.294 (0.895) |
| pilot_v6b | all | 138 | 0.413 | 0.354 | 0.630 | 0.652 | 0.739 | 1.657 | 0.684 | 0.445 (0.862) |
| pilot_v6b | key | 33 | 0.424 | 0.381 | 0.667 | 0.758 | 0.704 | 1.176 | 0.635 | 0.520 (0.758) |
| pilot_v6b | background | 105 | 0.410 | 0.344 | 0.619 | 0.619 | 0.750 | 1.808 | 0.699 | 0.426 (0.895) |

## Vehicle-level first-query recognition

| Run | Vehicles | First acc | First Brier | Key first acc | Background first acc |
|---|---:|---:|---:|---:|---:|
| pilot_v5 | 24 | 0.500 | 0.730 | 1.000 (n=6) | 0.333 (n=18) |
| pilot_v6b | 23 | 0.565 | 0.688 | 0.667 (n=6) | 0.529 (n=17) |

## Per-truth-class accuracy (query level)

| Run | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| pilot_v5 | Cautious | 36 | 0.417 | 0.735 | 0.373 | 0.855 |
| pilot_v5 | Normal | 60 | 0.067 | 0.671 | 0.180 | 1.189 |
| pilot_v5 | Aggressive | 57 | 0.596 | 0.723 | 0.581 | 0.450 |
| pilot_v6b | Cautious | 38 | 0.947 | 0.773 | 0.766 | 0.115 |
| pilot_v6b | Normal | 54 | 0.037 | 0.660 | 0.177 | 1.190 |
| pilot_v6b | Aggressive | 46 | 0.413 | 0.638 | 0.396 | 0.725 |

## Accuracy by query index

| Run | Query index | n | Accuracy | Binary hard |
|---|---:|---:|---:|---:|
| pilot_v5 | 0 | 18 | 0.556 | 0.556 |
| pilot_v5 | 1 | 18 | 0.556 | 0.556 |
| pilot_v5 | 2 | 18 | 0.333 | 0.444 |
| pilot_v5 | 3 | 18 | 0.333 | 0.444 |
| pilot_v5 | 4 | 18 | 0.389 | 0.500 |
| pilot_v5 | 5 | 18 | 0.444 | 0.667 |
| pilot_v5 | 6 | 12 | 0.333 | 0.417 |
| pilot_v5 | 7 | 12 | 0.167 | 0.333 |
| pilot_v5 | 8 | 9 | 0.000 | 0.000 |
| pilot_v5 | 9 | 6 | 0.000 | 0.000 |
| pilot_v5 | 10 | 6 | 0.000 | 0.000 |
| pilot_v6b | 0 | 18 | 0.667 | 0.833 |
| pilot_v6b | 1 | 18 | 0.611 | 0.833 |
| pilot_v6b | 2 | 18 | 0.556 | 0.833 |
| pilot_v6b | 3 | 18 | 0.444 | 0.667 |
| pilot_v6b | 4 | 18 | 0.333 | 0.500 |
| pilot_v6b | 5 | 18 | 0.222 | 0.389 |
| pilot_v6b | 6 | 12 | 0.167 | 0.417 |
| pilot_v6b | 7 | 12 | 0.167 | 0.500 |
| pilot_v6b | 8 | 6 | 0.333 | 0.500 |

## LLM latency (successful responses)

| Run | n | Mean ms | Median ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| pilot_v5 | 153 | 1081.8 | 1061.1 | 1347.9 | 1708.0 |
| pilot_v6b | 138 | 1161.1 | 1129.1 | 1488.0 | 1566.3 |
