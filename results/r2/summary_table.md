| Model | CV Top-1 (classifier) | ST Top-1 (classifier) | ST macro-F1 | ST log loss | Train-CV gap (pts) |
|---|---|---|---|---|---|
| LightGBM | 0.9123 | 0.8981 | 0.9411 | 0.0569 | 2.75 |
| XGBoost (selected) | 0.9117 | 0.8994 | 0.9413 | 0.0565 | 2.72 |
| RandomForest | 0.9092 | 0.8973 | 0.9342 | 0.0608 | 2.50 |
| MLP | 0.9053 | 0.9059 | 0.9388 | 0.0647 | 2.49 |
| LogisticRegression | 0.8960 | 0.8756 | 0.8631 | 0.1494 | 0.25 |

| Model | Evaluated on | Top-1 classifier -> HMM (reliable) | Top-1 (all points) | Route F1 | Continuity |
|---|---|---|---|---|---|
| LightGBM | train (OOF emissions; decoder tuned here) | 0.9143 -> 0.9283 | 0.8086 -> 0.8211 | 0.899 -> 0.926 | 0.769 -> 0.887 |
| LightGBM | student_test | 0.8944 -> 0.9021 | 0.7954 -> 0.8055 | 0.909 -> 0.932 | 0.792 -> 0.898 |
| XGBoost | train (OOF emissions; decoder tuned here) | 0.9131 -> 0.9292 | 0.8069 -> 0.8221 | 0.898 -> 0.927 | 0.769 -> 0.888 |
| XGBoost | student_test | 0.8984 -> 0.9060 | 0.7986 -> 0.8088 | 0.908 -> 0.933 | 0.796 -> 0.900 |

Inference: largest trip (19,313 points) 13.1 s; student_test trips max 1.02 s.
Sources: results/r2/model_comparison_v2_full.csv, hmm_comparison.csv, inference_check.csv. 'reliable' = points within 50 m of the reference route in trips with label_quality ok.
