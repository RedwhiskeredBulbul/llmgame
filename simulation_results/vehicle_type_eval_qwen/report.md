# Vehicle behaviour-type experiment

- Backend: qwen
- Model: qwen-turbo
- Dataset: simulation_results\vehicle_type_dataset.jsonl
- Samples: 150
- Accuracy (unparsed counted wrong): 28.667%
- Parse coverage: 100.000%
- Conditional accuracy: 28.667%
- Macro F1: 0.248
- Mean Brier score: 0.927
- Mean latency: 2.150 s

| Actual / predicted | Cautious | Normal | Aggressive | Unparsed |
|---|---:|---:|---:|---:|
| Cautious | 21 | 35 | 3 | 0 |
| Normal | 21 | 20 | 2 | 0 |
| Aggressive | 19 | 27 | 2 | 0 |
