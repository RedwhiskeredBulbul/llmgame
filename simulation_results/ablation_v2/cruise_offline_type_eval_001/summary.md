# DF_Cruise offline query-level type recognition

Frozen 20-frame histories, one LLM query per case, qwen-flash-2025-07-28, temperature 0, thinking off, timeout 10 s, no retries. Frames are 0.5 s apart and the worker uses dt=0.5. The dataset advances traffic around a fixed ego observer, so ego-relative speed is biased; leader-relative evidence is the primary cue for v6b.

## Query-level accuracy

| Version | Usable / total | Accuracy | Macro-F1 | Binary hard | Binary threshold | Brier | NLL | Mean confidence | Latency mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 | 150 / 150 | 0.393 | 0.188 | 0.680 | 0.680 | 0.909 | 1.485 | 0.800 | 1192.2 |
| v6b | 150 / 150 | 0.447 | 0.328 | 0.680 | 0.673 | 0.885 | 7.531 | 0.813 | 1158.1 |

## Per-truth-class accuracy

| Version | Truth | n | Accuracy | Mean confidence | Mean p_true | Brier |
|---|---|---:|---:|---:|---:|---:|
| v5 | Cautious | 59 | 1.000 | 0.800 | 0.800 | 0.060 |
| v5 | Normal | 43 | 0.000 | 0.800 | 0.100 | 1.460 |
| v5 | Aggressive | 48 | 0.000 | 0.800 | 0.100 | 1.460 |
| v6b | Cautious | 59 | 0.864 | 0.849 | 0.753 | 0.247 |
| v6b | Normal | 43 | 0.000 | 0.824 | 0.143 | 1.428 |
| v6b | Aggressive | 48 | 0.333 | 0.757 | 0.231 | 1.181 |

## Accuracy by position

| Version | Position | n | Accuracy |
|---|---|---:|---:|
| v5 | ahead | 150 | 0.393 |
| v6b | ahead | 150 | 0.447 |

## Accuracy by lane

| Version | Lane | n | Accuracy |
|---|---:|---:|---:|
| v5 | 0 | 50 | 0.280 |
| v5 | 1 | 46 | 0.435 |
| v5 | 2 | 54 | 0.463 |
| v6b | 0 | 50 | 0.360 |
| v6b | 1 | 46 | 0.478 |
| v6b | 2 | 54 | 0.500 |

## Accuracy by own-leader visibility

| Version | Subset | n | Accuracy | Predicted counts |
|---|---|---:|---:|---|
| v5 | leader_visible | 13 | 0.077 | {'Cautious': 13, 'Normal': 0, 'Aggressive': 0} |
| v5 | leaderless | 137 | 0.423 | {'Cautious': 137, 'Normal': 0, 'Aggressive': 0} |
| v6b | leader_visible | 13 | 0.615 | {'Cautious': 1, 'Normal': 0, 'Aggressive': 12} |
| v6b | leaderless | 137 | 0.431 | {'Cautious': 115, 'Normal': 2, 'Aggressive': 20} |

## Paired v5 vs v6b

Paired query records: 150 (only v5: 0, only v6b: 0).

| Metric | Paired n | v6b better | v6b worse | exact McNemar p |
|---|---:|---:|---:|---:|
| three_class | 150 | 16 | 8 | 0.15159 |
| binary_hard | 150 | 16 | 16 | 1 |
| binary_threshold | 150 | 15 | 16 | 1 |

Accuracy difference (v6b - v5, case-clustered): +0.053 [-0.007, +0.120]

Brier difference: -0.025 [-0.114, +0.066]
