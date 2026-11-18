"""Round 2 / Phase 3: model comparison v2 (8 models, same rows, same features).

Protocol: GroupKFold(5) by trip on the 80 train trips; every model is fitted on the SAME shared sample
(36,000 eligible points per fit, seed 42) with the Phase 2 feature set. Out-of-fold scoring on eligible
points (headline) and on all points; KNN is scored on a fixed 20% point subsample (every model is also
reported on that subsample). Train-fold Top-1 on a fixed 10% subsample of training-fold eligible points.
Final fit on 36,000 eligible points of all 80 train trips -> student_test.
Isotonic calibration (fitted inside CV on a 20% group split of the training fold) for the top-2
classifiers by CV Top-1. The selection rule is written to results/r2/selection_rule_v2.json BEFORE
any model is trained.

Writes results/r2/{model_comparison_v2.csv, model_comparison_v2_full.csv, cv_folds_v2.csv,
calibration_v2.csv, selection_v2.json} and reports/figures/r2/{cv_macro_f1_v2, st_top1_f1_v2,
log_loss_v2, confusion_matrix_v2, reliability_v2, mlp_loss_curve_v2}.png
Run: python scripts\\r2_model_comparison.py   (~15 min)
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # thermal limit: <= 4 threads
    _os.environ.setdefault(_v, "4")

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402

from pathsnap.eval.metrics import argmax_per_point, point_key, route_continuity  # noqa: E402
from pathsnap.io import RAW, ROOT  # noqa: E402
from pathsnap.models import zoo  # noqa: E402
from pathsnap.models.sampling import eligible, sample_points  # noqa: E402
from scripts.r2_common import PROC, load_feats_v2  # noqa: E402

RES, FIG = ROOT / "results" / "r2", ROOT / "reports" / "figures" / "r2"
SEED, EPS = 42, 1e-6
N_POINTS = json.loads((RES / "train_size_decision.json").read_text())["shared_train_points_per_fit"]
FEATS = json.loads((RES / "feature_set_v2.json").read_text())["features"]
# deadline scope (12:30 decision): exactly these 5 models, run one at a time
MODELS = ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost", "MLP"]
RULE = dict(
    primary="CV observation Top-1 on eligible train points (GroupKFold(5) by trip)",
    tie_band=0.005,
    rule="If the best models are within 0.005 CV Top-1, the final choice among them is made after Phase 5 by CV "
         "Top-1 of classifier+HMM; the best by CV Top-1 is the provisional pick for Phase 4 HPO. Otherwise: highest CV "
         "Top-1, with CV log loss as tie-break within 0.001.",
    student_test_used_for_selection=False, logged_before_training=True)


def subsample_mask(df, frac, salt):
    pk = np.unique(point_key(df))
    keep = np.random.default_rng(SEED + salt).choice(pk, size=int(frac * len(pk)), replace=False)
    return np.isin(point_key(df), keep)


def to_prob(model, df, s):
    """Classifier -> its probability; ranker -> softmax of scores within each point."""
    if not getattr(model, "is_ranker", False):
        return s
    k = point_key(df)
    t = pd.DataFrame({"k": k, "s": s})
    mx = t.groupby("k").s.transform("max")
    e = np.exp(t.s - mx)
    return (e / e.groupby(t.k).transform("sum")).to_numpy()


def row_metrics(df, prob, is_ranker, prefix):
    y = df.label.to_numpy()
    if is_ranker:   # positive = the point's top-1 candidate
        top = argmax_per_point(df, prob)
        pos = set(zip(top.trip_id, top.point_idx, top.pred_arc))
        yh = np.fromiter(((t, p, a) in pos for t, p, a in zip(df.trip_id, df.point_idx, df.arc_id)), bool, len(df)).astype(int)
    else:
        yh = (prob >= 0.5).astype(int)
    return {f"{prefix}macro_f1": f1_score(y, yh, average="macro"), f"{prefix}precision": precision_score(y, yh, zero_division=0),
            f"{prefix}recall": recall_score(y, yh, zero_division=0),
            f"{prefix}log_loss": log_loss(y, np.clip(prob, EPS, 1 - EPS), labels=[0, 1])}


def top1(df, s):
    return float(argmax_per_point(df, s).correct.mean())


def fit_on(spec, d):
    return spec.fit(d[FEATS], d.label.to_numpy(), d.trip_id.to_numpy(), point_key(d))


def main():
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    (RES / "selection_rule_v2.json").write_text(json.dumps(RULE | {"timestamp": pd.Timestamp.now().isoformat(timespec="seconds")}, indent=2))
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(np.random.rand(50, 2), np.arange(50) % 2)
    except Exception as e:  # noqa: BLE001
        print("CUDA unavailable -> cpu:", e); zoo.set_xgb_device("cpu")
    t_all = time.time()
    tr, te = load_feats_v2("train"), load_feats_v2("student_test")
    tr["sub20"] = subsample_mask(tr, 0.2, 0); te["sub20"] = subsample_mask(te, 0.2, 0)
    tr["sub10_trainfold"] = subsample_mask(tr, 0.1, 1)
    el_tr, el_te = eligible(tr).to_numpy(), eligible(te).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip_id))
    arcs = {(p, t): np.loadtxt(RAW / p / t / f"{t}.arcs", dtype=np.int64, ndmin=2)
            for p, d in (("train", tr), ("student_test", te)) for t in d.trip_id.unique()}
    segs = {}
    for p in ("train", "student_test"):
        pts = pd.read_parquet(PROC / f"points_{p}.parquet", columns=["trip_id", "point_idx", "segment_id"])
        for t, g in pts.groupby("trip_id"):
            segs[(p, t)] = g.sort_values("point_idx").segment_id.to_numpy()

    def cont(pred, part):
        return route_continuity(pred, {t: arcs[(part, t)] for t in pred.trip_id.unique()},
                                {t: segs[(part, t)] for t in pred.trip_id.unique()})

    rows, fold_rows, oof_store, models_final = [], [], {}, {}
    run = [s for s in zoo.ZOO if s.name in MODELS]
    print("models:", [s.name for s in run], "| n_jobs:", zoo.N_JOBS, "| features:", len(FEATS), flush=True)
    for spec in run:
        try:
            knn = spec.name == "KNN"
            t0 = time.time(); fm = []
            oof = np.full(len(tr), np.nan)
            for k, (itr, iev) in enumerate(folds):
                f = tr.iloc[itr]
                m = fit_on(spec, f[sample_points(f, N_POINTS)])
                ev_mask = np.zeros(len(tr), bool); ev_mask[iev] = True
                if knn:
                    ev_mask &= tr.sub20.to_numpy()
                ev = tr[ev_mask]
                s = m.score(ev[FEATS]); oof[ev_mask] = s
                ef = el_tr[ev_mask]
                trf = f[eligible(f) & f.sub10_trainfold]
                r = dict(model=spec.name, fold=k, cv_top1=top1(ev[ef], s[ef]), cv_top1_all=top1(ev, s),
                         train_fold_top1=top1(trf, m.score(trf[FEATS])))
                r |= row_metrics(ev[ef], to_prob(m, ev[ef], s[ef]), getattr(m, "is_ranker", False), "cv_")
                fm.append(r); fold_rows.append(r)
                print(f"  {spec.name} fold {k}: top1 {r['cv_top1']:.4f} train-fold {r['train_fold_top1']:.4f} [{time.time() - t0:.0f}s]", flush=True)
            cv_time = time.time() - t0
            F = pd.DataFrame(fm)
            done = ~np.isnan(oof)
            oof_pred = argmax_per_point(tr[done & el_tr], oof[done & el_tr])
            res = dict(Model=spec.name, family=spec.family, config=spec.config, scored_on="20% point subsample" if knn else "all points",
                       cv_top1=float(F.cv_top1.mean()), cv_top1_std=float(F.cv_top1.std()), cv_top1_all=float(F.cv_top1_all.mean()),
                       train_fold_top1=float(F.train_fold_top1.mean()), cv_macro_f1=float(F.cv_macro_f1.mean()),
                       cv_macro_f1_std=float(F.cv_macro_f1.std()), cv_log_loss=float(F.cv_log_loss.mean()),
                       cv_route_continuity=cont(oof_pred, "train"), cv_time_s=cv_time,
                       cv_top1_sub20=top1(tr[done & el_tr & tr.sub20.to_numpy()], oof[done & el_tr & tr.sub20.to_numpy()]))
            res["train_minus_cv_pts"] = 100 * (res["train_fold_top1"] - res["cv_top1"])
            res["overfit_flag"] = res["train_minus_cv_pts"] > 5
            oof_store[spec.name] = oof
            # final fit -> student_test
            t1 = time.time(); m = fit_on(spec, tr[sample_points(tr, N_POINTS)]); res["train_time_s"] = time.time() - t1
            ev = te[te.sub20] if knn else te
            t2 = time.time(); s = m.score(ev[FEATS]); dt = time.time() - t2
            res["inference_ms_per_1k_points"] = dt * 1e6 / len(np.unique(point_key(ev)))  # seconds -> ms per 1,000 points
            e = el_te[te.sub20.to_numpy()] if knn else el_te
            prob = to_prob(m, ev, s)
            res |= {"st_top1": top1(ev[e], s[e]), "st_top1_all": top1(ev, s)}
            res |= row_metrics(ev[e], prob[e], getattr(m, "is_ranker", False), "st_")
            res["st_route_continuity"] = cont(argmax_per_point(ev, s), "student_test")
            sub = te.sub20.to_numpy() & el_te
            res["st_top1_sub20"] = top1(te[sub], m.score(te[sub][FEATS]))
            models_final[spec.name] = (m, ev, s, e)
            rows.append(res)
            print(f"{spec.name}: CV top1 {res['cv_top1']:.4f}±{res['cv_top1_std']:.4f} train-fold {res['train_fold_top1']:.4f} "
                  f"ST {res['st_top1']:.4f} cont {res['st_route_continuity']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        except Exception:  # noqa: BLE001
            print(f"model {spec.name} FAILED:\n{traceback.format_exc()}", flush=True)

    R = pd.DataFrame(rows)
    pd.DataFrame(fold_rows).to_csv(RES / "cv_folds_v2.csv", index=False)

    # isotonic calibration inside CV for the top-2 classifiers (by CV Top-1)
    cls = R[~R.Model.isin(["XGBRanker"])].sort_values("cv_top1", ascending=False).head(2).Model.tolist()
    cal_rows, rel = [], {}
    for name in cls:
        spec = next(s for s in zoo.ZOO if s.name == name)
        raw_all, cal_all, y_all = [], [], []
        for itr, iev in folds:
            f = tr.iloc[itr]
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(f, groups=f.trip_id))
            fa, fb = f.iloc[a], f.iloc[b]
            m = fit_on(spec, fa[sample_points(fa, N_POINTS)])
            fb = fb[eligible(fb)]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(m.score(fb[FEATS]), fb.label)
            ev = tr.iloc[iev]; ev = ev[eligible(ev)]
            pr = m.score(ev[FEATS]); raw_all.append(pr); cal_all.append(iso.predict(pr)); y_all.append(ev.label.to_numpy())
        pr, pc, y = np.concatenate(raw_all), np.concatenate(cal_all), np.concatenate(y_all)
        cal_rows.append(dict(model=name, cv_log_loss_raw=log_loss(y, np.clip(pr, EPS, 1 - EPS)), cv_log_loss_isotonic=log_loss(y, pc),
                             brier_raw=float(np.mean((pr - y) ** 2)), brier_isotonic=float(np.mean((pc - y) ** 2))))
        rel[name] = (calibration_curve(y, pr, n_bins=15, strategy="quantile"), calibration_curve(y, pc, n_bins=15, strategy="quantile"))
        print("calibration", cal_rows[-1], flush=True)
    C = pd.DataFrame(cal_rows); C.to_csv(RES / "calibration_v2.csv", index=False)

    # selection (rule logged before training)
    R = R.sort_values("cv_top1", ascending=False).reset_index(drop=True)
    best = R.cv_top1.iloc[0]
    contenders = R[R.cv_top1 >= best - RULE["tie_band"]].Model.tolist()
    if len(contenders) > 1:
        provisional = R.Model.iloc[0]
        decision = dict(status="deferred_to_phase5", contenders=contenders, provisional_for_hpo=provisional)
    else:
        decision = dict(status="selected", selected=R.Model.iloc[0], provisional_for_hpo=R.Model.iloc[0])
    R["selected_or_contender"] = R.Model.isin(contenders)
    (RES / "selection_v2.json").write_text(json.dumps(RULE | decision | {"cv_top1": dict(zip(R.Model, R.cv_top1.round(5)))}, indent=2))

    full = R.copy(); full.to_csv(RES / "model_comparison_v2_full.csv", index=False)
    table = pd.DataFrame({
        "Model": R.Model, "Algorithm Family": R.family, "Key Configuration": R.config + f"; {N_POINTS:,} train points/fit, {len(FEATS)} features",
        "CV Macro F1-score": R.cv_macro_f1.map(lambda v: f"{v:.4f}") + " ± " + R.cv_macro_f1_std.map(lambda v: f"{v:.4f}"),
        "Top-1 Accuracy": R.st_top1.round(4),
        "Route continuity": R.st_route_continuity.round(4).astype(str) + " (classifier-only, student_test; HMM in Phase 5)",
        "Student-Test Macro F1-score": R.st_macro_f1.round(4),
        "Log Loss": np.where(R.Model == "XGBRanker", R.st_log_loss.round(4).astype(str) + " (softmax of ranker scores per point)", R.st_log_loss.round(4).astype(str)),
        "Inference Time": R.inference_ms_per_1k_points.round(1).astype(str) + " ms / 1k points",
        "CV Top-1": R.cv_top1.round(4), "CV Top-1 std": R.cv_top1_std.round(4), "Train-fold Top-1": R.train_fold_top1.round(4),
        "Train-CV gap (pts)": R.train_minus_cv_pts.round(2), "Overfit flag (>5 pts)": R.overfit_flag,
        "CV Log Loss": R.cv_log_loss.round(4), "CV Top-1 (all points)": R.cv_top1_all.round(4), "ST Top-1 (all points)": R.st_top1_all.round(4),
        "ST Precision": R.st_precision.round(4), "ST Recall": R.st_recall.round(4),
        "CV Top-1 (20% sub)": R.cv_top1_sub20.round(4), "ST Top-1 (20% sub)": R.st_top1_sub20.round(4),
        "Scored on": R.scored_on, "Train time (s)": R.train_time_s.round(1), "CV time (s)": R.cv_time_s.round(0),
        "Selected / contender": R.selected_or_contender})
    table.to_csv(RES / "model_comparison_v2.csv", index=False)

    # figures
    o = R.sort_values("cv_top1")
    hl = ["tab:green" if c else "tab:blue" for c in o.selected_or_contender]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.barh(o.Model, o.cv_macro_f1, xerr=o.cv_macro_f1_std, color=hl, capsize=3)
    ax.set_xlim(o.cv_macro_f1.min() - 0.05, 1); ax.set_title("CV macro-F1 ± std across 5 folds (green = selected/contender)")
    fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1_v2.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(o)); w = 0.38
    ax.bar(x - w / 2, o.st_top1, w, label="student_test Top-1"); ax.bar(x + w / 2, o.st_macro_f1, w, label="student_test macro-F1")
    ax.errorbar(x - w / 2, o.cv_top1, yerr=o.cv_top1_std, fmt="k_", ms=14, label="CV Top-1 ± std")
    ax.set_xticks(x, o.Model, rotation=20); ax.set_ylim(0.6, 1); ax.legend(fontsize=8)
    ax.set_title("student_test Top-1 and macro-F1 (eligible points); CV Top-1 for reference")
    fig.tight_layout(); fig.savefig(FIG / "st_top1_f1_v2.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, o.cv_log_loss, w, label="CV"); ax.bar(x + w / 2, o.st_log_loss, w, label="student_test")
    ax.set_xticks(x, o.Model, rotation=20); ax.set_ylabel("log loss (lower is better)"); ax.legend()
    ax.set_title("Log loss (XGBRanker: softmax of scores within each point)")
    fig.tight_layout(); fig.savefig(FIG / "log_loss_v2.png", dpi=150); plt.close(fig)
    bname = R.Model.iloc[0]; m, ev, s, e = models_final[bname]
    prob = to_prob(m, ev[e], s[e])
    yh = (prob >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(ev[e].label, yh), display_labels=["not label arc", "label arc"]).plot(ax=ax, values_format=",d", colorbar=False)
    ax.set_title(f"{bname}, student_test (threshold 0.5)"); fig.tight_layout(); fig.savefig(FIG / "confusion_matrix_v2.png", dpi=150); plt.close(fig)
    if rel:
        fig, axs = plt.subplots(1, len(rel), figsize=(5 * len(rel), 4.2))
        for ax, (name, ((t0_, p0_), (t1_, p1_))) in zip(np.atleast_1d(axs), rel.items()):
            ax.plot([0, 1], [0, 1], "k:", lw=1); ax.plot(p0_, t0_, "o-", label="raw"); ax.plot(p1_, t1_, "s-", label="isotonic (fit inside CV)")
            cr = C[C.model == name].iloc[0]
            ax.set_title(f"{name}: CV log loss {cr.cv_log_loss_raw:.4f} -> {cr.cv_log_loss_isotonic:.4f}")
            ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "reliability_v2.png", dpi=150); plt.close(fig)
    if "MLP" in models_final:
        mlp = models_final["MLP"][0].m[-1]
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(mlp.loss_curve_, label="training loss")
        ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy (10% early-stop split)")
        ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax2.set_ylabel("accuracy"); fig.legend(loc="upper right", fontsize=8)
        ax.set_title("MLP (64,32) training history, v2 features"); fig.tight_layout(); fig.savefig(FIG / "mlp_loss_curve_v2.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 250)
    print(R[["Model", "cv_top1", "cv_top1_std", "train_fold_top1", "train_minus_cv_pts", "cv_log_loss", "st_top1", "st_top1_all",
             "st_macro_f1", "st_log_loss", "st_route_continuity", "inference_ms_per_1k_points"]].round(4).to_string(index=False))
    print(json.dumps(decision)); print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
