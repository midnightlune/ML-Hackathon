"""Step 4: train + compare candidate-scoring models (classifier only; no HMM yet).

Protocol (identical for every model):
- Rows from points flagged `unreliable` (label alignment) are excluded from training AND scoring.
- Training rows = candidate rows of every 3rd point (point_idx % 3 == 0) -- the same subsample for
  all models. Models with a row cap draw a fixed (seed 42) random subset of that subsample.
- CV = GroupKFold(5) by trip_id on the 80 train trips; scoring on ALL reliable points of the
  held-out trips. KNN is scored on a fixed 20% subsample of points (seed 42); every model is also
  reported on that same subsample (`*_sub20` columns) for a like-for-like comparison.
- Non-tree models use sklearn Pipelines (median imputer + StandardScaler) fitted inside each fold.
- Top-1 = per point, argmax predicted probability over its candidates; correct iff that candidate
  is the label arc. Points whose label arc is not among the candidates count as wrong.
- Final fit on all train-subsample rows, evaluation on the 10 student_test trips.

Run: python -m scripts.train_models
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import logging
import sys
import time
import traceback

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss,  # noqa: E402
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402

SEED = 42
EPS = 1e-6
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "reports" / "figures"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")


# ----------------------------------------------------------------------------- data
def load(part):
    df = pd.read_parquet(PROC / f"{part}_feats.parquet")
    df = df[~df.unreliable].reset_index(drop=True)
    df["pkey"] = df.trip_id.astype(np.int64) * 1_000_000 + df.point_idx
    rng = np.random.default_rng(SEED)
    pk = df.pkey.unique()
    sub = set(rng.choice(pk, size=int(0.2 * len(pk)), replace=False).tolist())
    df["sub20"] = df.pkey.isin(sub)
    return df


# ----------------------------------------------------------------------------- metrics
def top1(df, p):
    o = np.lexsort((-p, df.pkey.to_numpy()))
    k = df.pkey.to_numpy()[o]
    first = np.r_[True, k[1:] != k[:-1]]
    return float(df.label.to_numpy()[o][first].mean())


def metrics(df, p, prefix):
    y = df.label.to_numpy()
    pc = np.clip(p, EPS, 1 - EPS)
    yh = (p >= 0.5).astype(int)
    return {f"{prefix}top1": top1(df, p), f"{prefix}macro_f1": f1_score(y, yh, average="macro"),
            f"{prefix}precision": precision_score(y, yh, zero_division=0),
            f"{prefix}recall": recall_score(y, yh, zero_division=0),
            f"{prefix}accuracy": float((yh == y).mean()), f"{prefix}log_loss": log_loss(y, pc, labels=[0, 1])}


# ----------------------------------------------------------------------------- models
def imputed(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


XGB_DEVICE = "cuda"


def fit_boost(kind, X, y, groups):
    """Boosters: early stopping on a group-held-out 10% of the training trips (never the eval fold)."""
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(X, y, groups))
    if kind == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", device=XGB_DEVICE, early_stopping_rounds=30, eval_metric="logloss",
                              random_state=SEED)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    return m


MODELS = {
    # name: (family, key config, max train rows or None, builder(X, y, groups) -> fitted model, knn_only_sub20)
    "LogisticRegression": ("Linear", "class_weight=balanced, C=1, lbfgs, median-impute+scale", None,
                           lambda X, y, g: imputed(LogisticRegression(class_weight="balanced", max_iter=1000,
                                                                      random_state=SEED)).fit(X, y), False),
    "KNN": ("Instance-based", "k=25, weights=distance, euclidean on scaled features, <=150k train rows", 150_000,
            lambda X, y, g: imputed(KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)).fit(X, y), True),
    "RandomForest": ("Ensemble (bagging)", "200 trees, max_depth=20, min_samples_leaf=5, <=200k train rows", 200_000,
                     lambda X, y, g: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
                         n_estimators=200, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=SEED)).fit(X, y),
                     False),
    "LightGBM": ("Boosting", "<=500 rounds, lr=0.05, 63 leaves, early stop 30 on 10% group holdout", None,
                 lambda X, y, g: fit_boost("lgbm", X, y, g), False),
    "MLP": ("Neural network", "hidden=(64,32), relu, adam, early_stopping, <=150k train rows", 150_000,
            lambda X, y, g: imputed(MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200,
                                                  random_state=SEED)).fit(X, y), False),
    "XGBoost": ("Boosting (GPU)", "<=500 rounds, lr=0.05, depth=8, hist, early stop 30 on 10% group holdout", None,
                lambda X, y, g: fit_boost("xgb", X, y, g), False),
}


def train_rows(df, cap):
    t = df[df.point_idx % 3 == 0]
    if cap is not None and len(t) > cap:
        t = t.sample(n=cap, random_state=SEED)
    return t


def predict(m, X):
    return m.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- main
def main():
    global XGB_DEVICE
    t_start = time.time()
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(np.random.rand(200, 3), np.arange(200) % 2)
        log.info("XGBoost CUDA init OK (RTX 3050)")
    except Exception as e:  # noqa: BLE001
        XGB_DEVICE = "cpu"
        log.warning("XGBoost CUDA init failed (%s) -> device=cpu", e)
    MODELS["XGBoost"] = (MODELS["XGBoost"][0], MODELS["XGBoost"][1] + f", device={XGB_DEVICE}", *MODELS["XGBoost"][2:])

    tr, te = load("train"), load("student_test")
    log.info("train: %d reliable rows / %d points; student_test: %d rows / %d points",
             len(tr), tr.pkey.nunique(), len(te), te.pkey.nunique())
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip_id))

    rows, oof_store = [], {}
    # reference baseline (not a trained model): nearest candidate arc
    base = {"Model": "Baseline: nearest arc (not a model)", "Algorithm Family": "Heuristic", "Key Configuration": "argmin perp_dist"}
    base.update({"cv_top1": top1(tr, -tr.perp_dist.to_numpy()), "st_top1": top1(te, -te.perp_dist.to_numpy()),
                 "st_top1_sub20": top1(te[te.sub20], -te.perp_dist[te.sub20].to_numpy())})
    log.info("baseline nearest-arc top1: cv %.4f  student_test %.4f", base["cv_top1"], base["st_top1"])

    for name, (fam, cfg, cap, build, knn_sub) in MODELS.items():
        try:
            log.info("=== %s ===", name)
            t0 = time.time()
            fold_m, fold_sub = [], []
            for k, (itr, iev) in enumerate(folds):
                ftr = train_rows(tr.iloc[itr], cap)
                m = build(ftr[FEATURES], ftr.label.to_numpy(), ftr.trip_id.to_numpy())
                ev = tr.iloc[iev]
                ev_sub = ev[ev.sub20]
                p_sub = predict(m, ev_sub[FEATURES])
                fold_sub.append(top1(ev_sub, p_sub))
                if knn_sub:
                    fold_m.append(metrics(ev_sub, p_sub, "cv_"))
                else:
                    fold_m.append(metrics(ev, predict(m, ev[FEATURES]), "cv_"))
                log.info("  fold %d: top1 %.4f (%.0fs)", k, fold_m[-1]["cv_top1"], time.time() - t0)
            cv_time = time.time() - t0
            if cv_time > 240:
                log.warning("  %s CV took %.0fs > 4 min budget -> cut its training rows next run", name, cv_time)
            cv = pd.DataFrame(fold_m)
            r = {"Model": name, "Algorithm Family": fam, "Key Configuration": cfg}
            r.update(cv.mean().to_dict())
            r["cv_top1_std"] = float(cv.cv_top1.std())
            r["cv_top1_sub20"] = float(np.mean(fold_sub))
            r["cv_scored_on"] = "20% point subsample" if knn_sub else "all reliable points"
            r["cv_time_s"] = cv_time

            ftr = train_rows(tr, cap)
            t1 = time.time()
            m = build(ftr[FEATURES], ftr.label.to_numpy(), ftr.trip_id.to_numpy())
            r["train_time_s"] = time.time() - t1
            r["train_rows"] = len(ftr)
            ev = te[te.sub20] if knn_sub else te
            t2 = time.time()
            p = predict(m, ev[FEATURES])
            r["inference_ms_per_1k_points"] = (time.time() - t2) * 1000 / ev.pkey.nunique() * 1000  # ms per 1,000 points
            r.update(metrics(ev, p, "st_"))
            r["st_scored_on"] = "20% point subsample" if knn_sub else "all reliable points"
            p_sub = predict(m, te.loc[te.sub20, FEATURES])
            r["st_top1_sub20"] = top1(te[te.sub20], p_sub)
            if name == "MLP":
                mlp = m[-1]
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(mlp.loss_curve_, label="training loss")
                ax2 = ax.twinx()
                ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy (10% early-stop split)")
                ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax2.set_ylabel("accuracy")
                fig.legend(loc="upper right", fontsize=8); ax.set_title("MLP (64,32) training history")
                fig.tight_layout(); fig.savefig(FIG / "mlp_loss_curve.png", dpi=150); plt.close(fig)
            oof_store[name] = (m, ev, p)
            rows.append(r)
            log.info("  %s: CV top1 %.4f±%.4f  CV F1 %.4f  ST top1 %.4f  (%.0fs total)", name, r["cv_top1"],
                     r["cv_top1_std"], r["cv_macro_f1"], r["st_top1"], time.time() - t0)
        except Exception:
            log.error("model %s failed:\n%s", name, traceback.format_exc())

    res = pd.DataFrame(rows)
    leak = res[res.cv_top1 > 0.99]
    if len(leak):
        log.error("LEAKAGE CHECK FAILED: CV top1 > 99%% for %s -> stop and audit features", list(leak.Model))
        res.to_csv(RES / "model_comparison_LEAK_AUDIT.csv", index=False)
        sys.exit(2)

    # selection: CV top1, log loss as tie-break (within 0.001 top1)
    best_top = res.cv_top1.max()
    tied = res[res.cv_top1 >= best_top - 0.001]
    best = tied.sort_values("cv_log_loss").iloc[0]["Model"]
    res["selected"] = res.Model == best

    table = pd.DataFrame({
        "Model": res.Model, "Algorithm Family": res["Algorithm Family"], "Key Configuration": res["Key Configuration"],
        "CV Macro F1-score": res.cv_macro_f1.round(4), "CV Top-1 Accuracy": res.cv_top1.round(4),
        "Top-1 Accuracy": res.st_top1.round(4),
        "Route continuity": "N/A (needs HMM/Viterbi decoding; round 2)",
        "Student-Test Macro F1-score": res.st_macro_f1.round(4), "Log Loss": res.st_log_loss.round(4),
        "Inference Time (ms / 1k points)": res.inference_ms_per_1k_points.round(1),
        "CV Log Loss": res.cv_log_loss.round(4), "CV Top-1 std": res.cv_top1_std.round(4),
        "Student-Test Precision": res.st_precision.round(4), "Student-Test Recall": res.st_recall.round(4),
        "Student-Test Accuracy (row-level)": res.st_accuracy.round(4),
        "CV Top-1 (20% subsample)": res.cv_top1_sub20.round(4), "Top-1 (20% subsample)": res.st_top1_sub20.round(4),
        "Scored on": res.st_scored_on, "Train rows": res.train_rows, "Train time (s)": res.train_time_s.round(1),
        "CV time (s)": res.cv_time_s.round(0), "Selected": res.selected,
    })
    table.to_csv(RES / "model_comparison.csv", index=False)
    pd.DataFrame([base]).to_csv(RES / "baseline_nearest_arc.csv", index=False)
    exp = table.copy()
    exp.insert(0, "experiment", "round1_model_comparison")
    exp.insert(1, "timestamp", pd.Timestamp.now().isoformat(timespec="seconds"))
    exp.insert(2, "seed", SEED)
    p_exp = RES / "experiments.csv"
    exp.to_csv(p_exp, mode="a", header=not p_exp.exists(), index=False)

    # plots
    FIG.mkdir(parents=True, exist_ok=True)
    for col, fname, title in (("CV Macro F1-score", "cv_macro_f1.png", "GroupKFold(5) CV macro-F1 (candidate rows)"),
                              ("Top-1 Accuracy", "student_test_top1.png", "Student-test observation Top-1 accuracy")):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        c = ["tab:green" if s else "tab:blue" for s in table.Selected]
        ax.bar(table.Model, table[col], color=c)
        if col == "Top-1 Accuracy":
            ax.axhline(base["st_top1"], color="grey", ls="--", label=f"nearest-arc baseline {base['st_top1']:.3f}")
            ax.legend(fontsize=8)
        for i, v in enumerate(table[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(min(table[col].min(), base["st_top1"]) - 0.05, 1.0)
        ax.set_title(title + " (green = selected)"); ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)
    m, ev, p = oof_store[best]
    cm = confusion_matrix(ev.label, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["not label arc", "label arc"]).plot(ax=ax, values_format=",d", colorbar=False)
    ax.set_title(f"{best} - student_test (threshold 0.5)")
    fig.tight_layout(); fig.savefig(FIG / f"confusion_matrix_{best}.png", dpi=150); plt.close(fig)

    (RES / "selection.json").write_text(json.dumps(dict(
        selected=best, rule="max CV observation Top-1 (GroupKFold 5 by trip); CV log loss breaks ties within 0.001",
        cv_top1=float(res.loc[res.selected, "cv_top1"].iloc[0]), baseline_nearest_arc=base,
        xgb_device=XGB_DEVICE, seed=SEED, total_runtime_s=time.time() - t_start), indent=2))
    print(table[["Model", "CV Macro F1-score", "CV Top-1 Accuracy", "CV Top-1 std", "CV Log Loss", "Top-1 Accuracy",
                 "Student-Test Macro F1-score", "Log Loss", "Inference Time (ms / 1k points)", "Train time (s)",
                 "CV time (s)", "Selected"]].to_string(index=False))
    print(f"baseline nearest arc: CV top1 {base['cv_top1']:.4f}, student_test top1 {base['st_top1']:.4f}")
    print(f"SELECTED: {best}; total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
