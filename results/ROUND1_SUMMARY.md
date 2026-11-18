# PathSnap - Round 1 summary (preprocessing, features, model comparison)

All numbers below are read from `results/` by `scripts/write_summary.py`. Seeds = 42.
Scope of this round: preprocessing, labels, candidates, features and 6 classifier models with one selected.
**Not done yet:** Bayesian HPO, HMM/Viterbi, API, UI.

## 1. Pipeline

| Step | Script | Output |
|---|---|---|
| Load, project, quality checks | `python -m scripts.preprocess` | `results/eda/inventory.csv`, `maps/trip_00000001_raw.html`, `data/processed/points_*.parquet` |
| Per-point labels (monotonic DP) | `python -m scripts.make_labels` | `data/processed/labels_*.parquet`, `results/eda/label_alignment.csv`, `label_threshold.json` |
| Candidates + features | `python -m scripts.build_features` | `data/processed/{train,student_test}_feats.parquet`, `results/eda/candidates.json` |
| Models | `python -m scripts.train_models` | `results/model_comparison.csv`, `results/experiments.csv`, `reports/figures/*.png` |
| This file | `python -m scripts.write_summary` | `results/ROUND1_SUMMARY.md` |

- **Projection:** every trip uses its own UTM zone, chosen from the track centroid via pyproj (25 distinct EPSG codes across the 90 trips). All distances are in metres. Arcs are straight segments between their two nodes.
- **Gaps:** time gaps > 30 s start a new `segment_id`; no points are dropped. Heading and speed windows never cross a segment boundary.
- **Labels:** `.route` is an ordered arc list. Each point gets a route position by DP: positions are non-decreasing in time and the summed point-to-segment distance is minimised. Back-pointers are banded around each point's nearest route arc; the band doubles whenever the optimum touches its edge, so the band never changes the answer. A toy-graph test (out-and-back loop) passes. Monotonicity is asserted for every trip.
- **Candidates:** STRtree per trip, arcs within R = 75 m, top K = 12 by distance. Points with no arc in range get their nearest 3.
- **Features (13):** perp_dist, dist_rank, dist_minus_min, proj_fraction, arc_length, heading_cos / heading_abs (GPS heading over ±2 points vs directed-arc bearing; NaN with heading_reliable = 0 when window displacement < 3 m), heading_reliable, speed, n_candidates, has_reverse_twin, prev/next_point_dist_to_cand. No lon/lat, no trip id. `candidates.py` and `features.py` do not import `pathsnap.labels`; labels are joined only afterwards.

## 2. Data quality (`results/eda/inventory.csv`)

| partition | trips | GPS points | route arcs | max gap (s) | trips with gap > 30 s | invalid coords | non-monotonic t | routes with a connectivity break | repeated route arcs |
|---|---|---|---|---|---|---|---|---|---|
| student_test | 10 | 19160 | 3477 | 210.0000 | 2 | 0 | 0 | 1 | 255 |
| train | 80 | 208417 | 31336 | 153064.0000 | 28 | 0 | 0 | 10 | 67 |

Finding: CLAUDE.md says route arcs are always connected and never repeat. This does **not** hold for every trip: see the last two columns. The DP aligns to route *positions*, so repeated arcs are handled correctly.

## 3. Label alignment

- align_dist_m percentiles (50 / 90 / 95 / 99): **5.72 / 93.04 / 148.08 / 231.99 m**.
- 11.15% of points exceed 50 m (> 5%), so per the rule the unreliable threshold is the **95th percentile = 148.08 m** instead of 50 m.
- Unreliable points (excluded from training **and** scoring): train 9,493 (4.55%), student_test 1,886 (9.84%).
- The excess is concentrated in a few trips:
  - student_test/00000056: 47.7% unreliable, median 105.6 m, p95 1879.7 m
  - train/00000054: 44.9% unreliable, median 140.1 m, p95 210.1 m
  - train/00000002: 7.0% unreliable, median 5.8 m, p95 242.4 m

## 4. Candidates

| Partition | Candidate rows | Points | Positive rate (reliable rows) | Miss rate R=50 m | Miss rate R=75 m |
|---|---|---|---|---|---|
| train | 1,674,932 | 208,417 | 0.1136 (~1 in 8.8) | 6.90% | 6.75% |
| student_test | 172,845 | 19,160 | 0.1112 (~1 in 9.0) | 1.48% | 1.46% |

The train miss rate was > 5% at R = 50 m, so R was raised to 75 m, but it barely moved. The misses are not a radius problem: they are points 100-148 m from their label arc that the loosened (148 m) reliability threshold lets in, mostly trips 00000054 and 00000033. For points aligned within 25 m the miss rate is about 0.15% (checked ad hoc during the run; not yet a saved artifact). Every missed point counts as a Top-1 error, so it caps achievable train Top-1 at roughly 1 - miss rate.

## 5. Model comparison (`results/model_comparison.csv`)

Protocol: GroupKFold(5) by trip on the 80 train trips. Training rows = candidate rows of every 3rd point (same subsample for all models; RandomForest is capped at 200k rows, KNN and MLP at 150k, by a fixed seed-42 draw). Scoring uses **all** reliable points. Non-tree models use a median imputer + StandardScaler inside each fold's Pipeline. KNN is scored on a fixed 20% point subsample; the `ST Top-1 (20% sub)` column gives every model on that same subsample. Macro-F1 / precision / recall are row-level at threshold 0.5. Log loss uses probabilities clipped to [1e-6, 1-1e-6]. Route continuity = N/A until HMM decoding.

| Model | Algorithm Family | CV Macro F1-score | CV Top-1 Accuracy | CV Top-1 std | CV Log Loss | ST Top-1 | ST Macro F1 | ST Precision | ST Recall | ST Log Loss | ST Top-1 (20% sub) | Train time (s) | Infer ms/1k pts | Scored on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | Linear | 0.8168 | 0.8123 | 0.1041 | 0.2165 | 0.8306 | 0.8318 | 0.5759 | 0.9350 | 0.1969 | 0.8225 | 1.5000 | 1.2000 | all reliable points |
| KNN | Instance-based | 0.9099 | 0.8078 | 0.1085 | 0.1383 | 0.8324 | 0.9100 | 0.8598 | 0.8199 | 0.1300 | 0.8324 | 0.4000 | 995.3000 | 20% point subsample |
| RandomForest | Ensemble (bagging) | 0.9221 | 0.8204 | 0.1052 | 0.0865 | 0.8581 | 0.9194 | 0.8755 | 0.8380 | 0.0814 | 0.8532 | 8.9000 | 12.4000 | all reliable points |
| LightGBM | Boosting | 0.9251 | 0.8237 | 0.1065 | 0.0812 | 0.8601 | 0.9221 | 0.8699 | 0.8531 | 0.0783 | 0.8584 | 3.3000 | 20.4000 | all reliable points |
| MLP | Neural network | 0.9108 | 0.8176 | 0.1032 | 0.0947 | 0.8485 | 0.9145 | 0.8783 | 0.8185 | 0.0856 | 0.8422 | 8.8000 | 2.9000 | all reliable points |
| XGBoost | Boosting (GPU) | 0.9251 | 0.8220 | 0.1063 | 0.0796 | 0.8602 | 0.9219 | 0.8683 | 0.8538 | 0.0774 | 0.8570 | 4.7000 | 4.9000 | all reliable points |

Reference heuristic (not a trained model): nearest arc, Top-1 CV 0.5115 / student_test 0.4933.

Figures: `reports/figures/cv_macro_f1.png`, `student_test_top1.png`, `confusion_matrix_LightGBM.png`, `mlp_loss_curve.png`.

## 6. Selected model: **LightGBM**

- Rule: highest CV observation Top-1 (GroupKFold by trip); CV log loss breaks ties within 0.001.
- LightGBM: CV Top-1 0.8237 ± 0.1065, CV log loss 0.0812; student_test Top-1 0.8601. Runner-up XGBoost: CV Top-1 0.8220, CV log loss 0.0796.
- **Why Top-1 and not raw accuracy:** each point has many candidates and only one is correct (positive rate ~1 in 9). A model that predicts "not the label arc" everywhere already gets ~89% row accuracy. Top-1 measures what the map-matcher actually needs: whether the right arc gets the highest score for each GPS point. It is also exactly the quantity the HMM emission step consumes. Log loss is the tie-break because the HMM needs calibrated probabilities, not just rankings.
- Model selection used only train CV; student_test is reported for comparison and was not used to choose.
- Leakage check: no model exceeded 99% CV Top-1.

## 7. Known issues

1. The reliability threshold is 148 m (the p95 rule). It is inflated by two trips with poor route/track agreement (train/00000054, student_test/00000056). A 50 m cap plus per-trip exclusion would give cleaner labels; worth deciding before HPO.
   This is also why every model drops to roughly 0.65 Top-1 on one CV fold. That fold holds train/00000054, where most "reliable" points have no correct candidate. The large CV Top-1 std reflects this trip, not model instability.
2. The train candidate miss rate (6.75%) is dominated by those far-from-route points, not by R or K.
3. Some routes have connectivity breaks or repeated arcs (section 2); the HMM transition model must tolerate a missing link.
4. Row caps (RF/KNN/MLP) mean those models saw less data than LR/LightGBM/XGBoost. The comparison is fair on protocol but not on data volume.
5. XGBoost device: cuda. Prediction falls back to a DMatrix copy (CPU input, CUDA model), which slightly inflates XGBoost inference time. KNN inference is about 1000x slower than the trees, which matters for the API.
6. The selection margin is tiny. If the runner-up's CV Top-1 is within 0.001 of the winner's, the choice was made by the log-loss tie-break. Both boosters are effectively tied, so HPO may reverse the order.

## 8. Next steps

1. Optuna (TPE) HPO for LightGBM with GroupKFold on train; objective = CV Top-1 (log loss secondary); log trials to `results/optuna.db`.
2. HMM/Viterbi: hidden states = candidate arcs per point; emission = classifier probability; transitions from network connectivity (shortest-path vs GPS distance) with gap handling per `segment_id`. Compare classifier-only vs classifier+HMM on Top-1, sequence accuracy and route continuity.
3. FastAPI backend on Laptop 1 (`--eval-dir`; ground truth stays server-side); UI on Laptop 2 with Folium output.
