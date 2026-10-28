# Vehicle behaviour-type experiment

- Backend: qwen
- Model: qwen-turbo
- Dataset: continuous-DF_Cruise-online-ema
- Samples: 330
- Accuracy (unparsed counted wrong): 33.030%
- Parse coverage: 99.091%
- Conditional accuracy: 33.333%
- Macro F1: 0.296
- Mean Brier score: 0.783
- Mean latency: 2.189 s

| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |
|---|---:|---:|---:|---:|
| Cautious | 45 | 65 | 8 | 2 |
| Normal | 41 | 56 | 7 | 0 |
| Aggressive | 31 | 66 | 8 | 1 |
