# Vehicle behaviour-type experiment

- Backend: qwen
- Model: qwen-turbo
- Dataset: simulation_results\vehicle_type_dataset_v2_20f.jsonl
- Samples: 150
- Accuracy (unparsed counted wrong): 30.667%
- Parse coverage: 100.000%
- Conditional accuracy: 30.667%
- Macro F1: 0.262
- Mean Brier score: 0.822
- Mean latency: 2.198 s

| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |
|---|---:|---:|---:|---:|
| Cautious | 19 | 38 | 2 | 0 |
| Normal | 17 | 25 | 1 | 0 |
| Aggressive | 14 | 32 | 2 | 0 |
