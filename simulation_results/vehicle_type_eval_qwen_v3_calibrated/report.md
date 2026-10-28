# Vehicle behaviour-type experiment

- Backend: qwen
- Model: qwen-turbo
- Dataset: simulation_results\vehicle_type_dataset_v2_20f.jsonl
- Samples: 150
- Accuracy (unparsed counted wrong): 32.000%
- Parse coverage: 100.000%
- Conditional accuracy: 32.000%
- Macro F1: 0.286
- Mean Brier score: 0.838
- Mean latency: 2.290 s

| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |
|---|---:|---:|---:|---:|
| Cautious | 17 | 35 | 7 | 0 |
| Normal | 15 | 27 | 1 | 0 |
| Aggressive | 12 | 32 | 4 | 0 |
