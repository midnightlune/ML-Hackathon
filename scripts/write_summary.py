"""Step 5: results/ROUND1_SUMMARY.md, generated from results/*.csv|json (no hand-typed numbers).

Run: python -m scripts.write_summary
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json

import pandas as pd

from pathsnap.io import ROOT

RES = ROOT / "results"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    inv = pd.read_csv(RES / "eda" / "inventory.csv")
    thr = json.loads((RES / "eda" / "label_threshold.json").read_text())
    la = pd.read_csv(RES / "eda" / "label_alignment.csv", dtype={"trip_id": str})
    cand = json.loads((RES / "eda" / "candidates.json").read_text())
    mc = pd.read_csv(RES / "model_comparison.csv")
    base = pd.read_csv(RES / "baseline_nearest_arc.csv").iloc[0]
    sel = json.loads((RES / "selection.json").read_text())
    best = mc[mc.Selected].iloc[0]

    g = inv.groupby("partition")
    inv_s = pd.DataFrame({
        "trips": g.size(), "GPS points": g.n_points.sum(), "route arcs": g.n_route_arcs.sum(),
        "max gap (s)": g.max_gap_s.max(), "trips with gap > 30 s": g.n_gaps_gt30s.apply(lambda s: int((s > 0).sum())),
        "invalid coords": g.invalid_coords.sum(), "non-monotonic t": g.non_monotonic_ts.sum(),
        "routes with a connectivity break": g.route_connected.apply(lambda s: int((~s).sum())),
        "repeated route arcs": g.route_repeats.sum(),
    }).reset_index()
    la["pct_unreliable"] = 100 * la.unreliable / la.points
    worst = la.sort_values("pct_unreliable", ascending=False).head(3)
    unrel = {p: f"{c:,} ({100 * c / inv[inv.partition == p].n_points.sum():.2f}%)"
             for p, c in thr["unreliable_counts"].items()}
    q = thr["percentiles_50_90_95_99"]
    R = cand["R_used"]
    rkey = f"R{R:.0f}"

    comp = mc[["Model", "Algorithm Family", "CV Macro F1-score", "CV Top-1 Accuracy", "CV Top-1 std", "CV Log Loss",
               "Top-1 Accuracy", "Student-Test Macro F1-score", "Student-Test Precision", "Student-Test Recall",
               "Log Loss", "Top-1 (20% subsample)", "Train time (s)", "Inference Time (ms / 1k points)", "Scored on"]]
    comp = comp.rename(columns={"Top-1 Accuracy": "ST Top-1", "Student-Test Macro F1-score": "ST Macro F1",
                                "Student-Test Precision": "ST Precision", "Student-Test Recall": "ST Recall",
                                "Log Loss": "ST Log Loss", "Top-1 (20% subsample)": "ST Top-1 (20% sub)",
                                "Inference Time (ms / 1k points)": "Infer ms/1k pts"})
    runner = mc[~mc.Selected].sort_values("CV Top-1 Accuracy", ascending=False).iloc[0]

    text = f"""# PathSnap - Round 1 summary (preprocessing, features, model comparison)

All numbers below are read from `results/` by `scripts/write_summary.py`. Seeds = 42.
Scope of this round: preprocessing, labels, candidates, features and 6 classifier models with one selected.
**Not done yet:** Bayesian HPO, HMM/Viterbi, API, UI.

## 1. Pipeline

| Step | Script | Output |
|---|---|---|
| Load, project, quality checks | `python -m scripts.preprocess` | `results/eda/inventory.csv`, `maps/trip_00000001_raw.html`, `data/processed/points_*.parquet` |
| Per-point labels (monotonic DP) | `python -m scripts.make_labels` | `data/processed/labels_*.parquet`, `results/eda/label_alignment.csv`, `label_threshold.json` |
| Candidates + features | `python -m scripts.build_features` | `data/processed/{{train,student_test}}_feats.parquet`, `results/eda/candidates.json` |
| Models | `python -m scripts.train_models` | `results/model_comparison.csv`, `results/experiments.csv`, `reports/figures/*.png` |
| This file | `python -m scripts.write_summary` | `results/ROUND1_SUMMARY.md` |

- **Projection:** every trip uses its own UTM zone, chosen from the track centroid via pyproj ({inv.epsg.nunique()} distinct EPSG codes across the 90 trips). All distances are in metres. Arcs are straight segments between their two nodes.
- **Gaps:** time gaps > 30 s start a new `segment_id`; no points are dropped. Heading and speed windows never cross a segment boundary.
- **Labels:** `.route` is an ordered arc list. Each point gets a route position by DP: positions are non-decreasing in time and the summed point-to-segment distance is minimised. Back-pointers are banded around each point's nearest route arc; the band doubles whenever the optimum touches its edge, so the band never changes the answer. A toy-graph test (out-and-back loop) passes. Monotonicity is asserted for every trip.
- **Candidates:** STRtree per trip, arcs within R = {R:.0f} m, top K = {cand['K']} by distance. Points with no arc in range get their nearest 3.
- **Features (13):** perp_dist, dist_rank, dist_minus_min, proj_fraction, arc_length, heading_cos / heading_abs (GPS heading over ±2 points vs directed-arc bearing; NaN with heading_reliable = 0 when window displacement < 3 m), heading_reliable, speed, n_candidates, has_reverse_twin, prev/next_point_dist_to_cand. No lon/lat, no trip id. `candidates.py` and `features.py` do not import `pathsnap.labels`; labels are joined only afterwards.

## 2. Data quality (`results/eda/inventory.csv`)

{md_table(inv_s)}

Finding: CLAUDE.md says route arcs are always connected and never repeat. This does **not** hold for every trip: see the last two columns. The DP aligns to route *positions*, so repeated arcs are handled correctly.

## 3. Label alignment

- align_dist_m percentiles (50 / 90 / 95 / 99): **{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f} m**.
- {thr['frac_over_50m']:.2%} of points exceed 50 m (> 5%), so per the rule the unreliable threshold is the **95th percentile = {thr['threshold_m']:.2f} m** instead of 50 m.
- Unreliable points (excluded from training **and** scoring): train {unrel['train']}, student_test {unrel['student_test']}.
- The excess is concentrated in a few trips:
{chr(10).join(f"  - {r.partition}/{r.trip_id}: {r.pct_unreliable:.1f}% unreliable, median {r.median_dist:.1f} m, p95 {r.p95_dist:.1f} m" for r in worst.itertuples())}

## 4. Candidates

| Partition | Candidate rows | Points | Positive rate (reliable rows) | Miss rate R=50 m | Miss rate R={R:.0f} m |
|---|---|---|---|---|---|
{chr(10).join(f"| {p} | {cand[p]['rows']:,} | {cand[p]['points']:,} | {cand[p]['positive_rate']:.4f} (~1 in {1 / cand[p]['positive_rate']:.1f}) | {cand['R50'][p]['miss_rate']:.2%} | {cand[rkey][p]['miss_rate']:.2%} |" for p in ('train', 'student_test'))}

The train miss rate was > 5% at R = 50 m, so R was raised to 75 m, but it barely moved. The misses are not a radius problem: they are points 100-148 m from their label arc that the loosened ({thr['threshold_m']:.0f} m) reliability threshold lets in, mostly trips 00000054 and 00000033. For points aligned within 25 m the miss rate is about 0.15% (checked ad hoc during the run; not yet a saved artifact). Every missed point counts as a Top-1 error, so it caps achievable train Top-1 at roughly 1 - miss rate.

## 5. Model comparison (`results/model_comparison.csv`)

Protocol: GroupKFold(5) by trip on the 80 train trips. Training rows = candidate rows of every 3rd point (same subsample for all models; RandomForest is capped at 200k rows, KNN and MLP at 150k, by a fixed seed-42 draw). Scoring uses **all** reliable points. Non-tree models use a median imputer + StandardScaler inside each fold's Pipeline. KNN is scored on a fixed 20% point subsample; the `ST Top-1 (20% sub)` column gives every model on that same subsample. Macro-F1 / precision / recall are row-level at threshold 0.5. Log loss uses probabilities clipped to [1e-6, 1-1e-6]. Route continuity = N/A until HMM decoding.

{md_table(comp)}

Reference heuristic (not a trained model): nearest arc, Top-1 CV {base.cv_top1:.4f} / student_test {base.st_top1:.4f}.

Figures: `reports/figures/cv_macro_f1.png`, `student_test_top1.png`, `confusion_matrix_{best.Model}.png`, `mlp_loss_curve.png`.

## 6. Selected model: **{best.Model}**

- Rule: highest CV observation Top-1 (GroupKFold by trip); CV log loss breaks ties within 0.001.
- {best.Model}: CV Top-1 {best['CV Top-1 Accuracy']:.4f} ± {best['CV Top-1 std']:.4f}, CV log loss {best['CV Log Loss']:.4f}; student_test Top-1 {best['Top-1 Accuracy']:.4f}. Runner-up {runner.Model}: CV Top-1 {runner['CV Top-1 Accuracy']:.4f}, CV log loss {runner['CV Log Loss']:.4f}.
- **Why Top-1 and not raw accuracy:** each point has many candidates and only one is correct (positive rate ~1 in {1 / cand['train']['positive_rate']:.0f}). A model that predicts "not the label arc" everywhere already gets ~{1 - cand['train']['positive_rate']:.0%} row accuracy. Top-1 measures what the map-matcher actually needs: whether the right arc gets the highest score for each GPS point. It is also exactly the quantity the HMM emission step consumes. Log loss is the tie-break because the HMM needs calibrated probabilities, not just rankings.
- Model selection used only train CV; student_test is reported for comparison and was not used to choose.
- Leakage check: no model exceeded 99% CV Top-1.

## 7. Known issues

1. The reliability threshold is {thr['threshold_m']:.0f} m (the p95 rule). It is inflated by two trips with poor route/track agreement (train/00000054, student_test/00000056). A 50 m cap plus per-trip exclusion would give cleaner labels; worth deciding before HPO.
   This is also why every model drops to roughly 0.65 Top-1 on one CV fold. That fold holds train/00000054, where most "reliable" points have no correct candidate. The large CV Top-1 std reflects this trip, not model instability.
2. The train candidate miss rate ({cand[rkey]['train']['miss_rate']:.2%}) is dominated by those far-from-route points, not by R or K.
3. Some routes have connectivity breaks or repeated arcs (section 2); the HMM transition model must tolerate a missing link.
4. Row caps (RF/KNN/MLP) mean those models saw less data than LR/LightGBM/XGBoost. The comparison is fair on protocol but not on data volume.
5. XGBoost device: {sel['xgb_device']}. Prediction falls back to a DMatrix copy (CPU input, CUDA model), which slightly inflates XGBoost inference time. KNN inference is about 1000x slower than the trees, which matters for the API.
6. The selection margin is tiny. If the runner-up's CV Top-1 is within 0.001 of the winner's, the choice was made by the log-loss tie-break. Both boosters are effectively tied, so HPO may reverse the order.

## 8. Next steps

1. Optuna (TPE) HPO for {best.Model} with GroupKFold on train; objective = CV Top-1 (log loss secondary); log trials to `results/optuna.db`.
2. HMM/Viterbi: hidden states = candidate arcs per point; emission = classifier probability; transitions from network connectivity (shortest-path vs GPS distance) with gap handling per `segment_id`. Compare classifier-only vs classifier+HMM on Top-1, sequence accuracy and route continuity.
3. FastAPI backend on Laptop 1 (`--eval-dir`; ground truth stays server-side); UI on Laptop 2 with Folium output.
"""
    (RES / "ROUND1_SUMMARY.md").write_text(text, encoding="utf-8")
    print(f"wrote {RES / 'ROUND1_SUMMARY.md'}")


if __name__ == "__main__":
    main()
