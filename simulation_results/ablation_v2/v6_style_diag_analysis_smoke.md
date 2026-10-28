# v5 vs v6 style-recognition diagnostics

## Runs

| Run | Episodes | Query samples | Vehicle-episodes | All acc | Conf>=0.5 acc (cov) | Binary acc | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_v6 | 2 | 6 | 6 | 0.500 | 0.400 (0.833) | 0.500 | 0.803 | 1.333 |
| merge_v6 | 2 | 6 | 6 | 0.333 | 1.000 (0.333) | 1.000 | 0.496 | 0.859 |

## Paired comparisons

## Position breakdown (all query samples)

| Run | Position | n | Accuracy | Conf>=0.5 acc | Brier |
|---|---|---:|---:|---:|---:|
| df_v6 | behind | 6 | 0.500 | 0.400 | 0.803 |
| merge_v6 | ahead | 4 | 0.500 | 1.000 | 0.407 |
| merge_v6 | behind | 2 | 0.000 | nan | 0.673 |
