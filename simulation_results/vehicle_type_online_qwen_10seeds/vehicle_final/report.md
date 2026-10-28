# Vehicle behaviour-type experiment

- Backend: qwen
- Model: qwen-turbo
- Dataset: continuous-DF_Cruise-final-vehicle
- Samples: 34
- Accuracy (unparsed counted wrong): 38.235%
- Parse coverage: 100.000%
- Conditional accuracy: 38.235%
- Macro F1: 0.342
- Mean Brier score: 0.796
- Mean latency: 0.000 s

| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |
|---|---:|---:|---:|---:|
| Cautious | 5 | 6 | 1 | 0 |
| Normal | 3 | 7 | 1 | 0 |
| Aggressive | 5 | 5 | 1 | 0 |
