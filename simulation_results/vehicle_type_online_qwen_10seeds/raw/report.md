# Vehicle behaviour-type experiment

- Backend: qwen
- Model: qwen-turbo
- Dataset: continuous-DF_Cruise-online-raw
- Samples: 330
- Accuracy (unparsed counted wrong): 34.242%
- Parse coverage: 99.091%
- Conditional accuracy: 34.557%
- Macro F1: 0.314
- Mean Brier score: 0.879
- Mean latency: 2.189 s

| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |
|---|---:|---:|---:|---:|
| Cautious | 43 | 63 | 12 | 2 |
| Normal | 34 | 59 | 11 | 0 |
| Aggressive | 29 | 65 | 11 | 1 |
