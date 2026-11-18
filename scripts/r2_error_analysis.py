"""Round 2 / Phase 1.3: error analysis of the Round 1 XGBoost config re-run on the Round 2 labels.

Protocol: GroupKFold(5) by trip over all 80 train trips (seed 42). Training rows = Round 1 protocol
(candidate rows of every 3rd point) restricted to eligible points (reliable, label_quality ok).
Scores on every train point out-of-fold; final fit on all eligible train rows -> student_test.
Scopes reported: reliable points in ok trips (headline), all points in ok trips, all points,
low_quality trips. Error taxonomy: candidate_miss / reverse_twin / adjacent_arc / other.

Writes results/r2/{error_analysis.csv, top1_by_scope_r1model.csv, per_trip_top1_r1model.csv,
error_analysis_points.parquet} and reports/figures/r2/{error_taxonomy.png, per_trip_top1.png}.
Run: python scripts\\r2_error_analysis.py   (~1 min on GPU)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import scripts.train_models as r1  # noqa: E402  (Round 1 config, unchanged)
from pathsnap.eval.errors import CATEGORIES, classify  # noqa: E402
from pathsnap.eval.metrics import argmax_per_point  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import RAW, ROOT  # noqa: E402
from pathsnap.models.sampling import eligible  # noqa: E402
from scripts.r2_common import load_feats  # noqa: E402

RES, FIG = ROOT / "results" / "r2", ROOT / "reports" / "figures" / "r2"
SCOPES = {
    "reliable_ok_trips": lambda p: (~p.unreliable) & (p.label_quality == "ok"),
    "all_ok_trips": lambda p: p.label_quality == "ok",
    "all_points": lambda p: p.label_quality.notna(),
    "low_quality_trips_all": lambda p: p.label_quality == "low_quality",
    "reliable_all_trips": lambda p: ~p.unreliable,
}


def train_rows(df):
    return df[eligible(df) & (df.point_idx % 3 == 0)]


def fit(df):
    t = train_rows(df)
    return r1.fit_boost("xgb", t[FEATURES], t.label.to_numpy(), t.trip_id.to_numpy())


def point_table(df, score, part):
    pred = argmax_per_point(df, score)
    meta = df.groupby(["trip_id", "point_idx"]).agg(label_in_cands=("label", "max"), unreliable=("unreliable", "first"),
                                                     label_quality=("label_quality", "first")).reset_index()
    pred = pred.merge(meta, on=["trip_id", "point_idx"], validate="one_to_one")
    pred["label_in_cands"] = pred.label_in_cands.astype(bool)
    arcs = {t: np.loadtxt(RAW / part / t / f"{t}.arcs", dtype=np.int64, ndmin=2) for t in pred.trip_id.unique()}
    pred["category"] = classify(pred, arcs, pred.label_in_cands)
    pred["partition"] = part
    return pred


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(np.random.rand(50, 2), np.arange(50) % 2)
    except Exception as e:  # noqa: BLE001
        print("CUDA unavailable -> cpu:", e); r1.XGB_DEVICE = "cpu"
    t0 = time.time()
    tr, te = load_feats("train"), load_feats("student_test")
    oof = np.zeros(len(tr)); fold_rows = []
    for k, (itr, iev) in enumerate(GroupKFold(n_splits=5).split(tr, groups=tr.trip_id)):
        m = fit(tr.iloc[itr])
        oof[iev] = m.predict_proba(tr.iloc[iev][FEATURES])[:, 1]
        ft = tr.iloc[itr]; ft = ft[eligible(ft)]
        tp = argmax_per_point(ft, m.predict_proba(ft[FEATURES])[:, 1]).correct.mean()
        ev = tr.iloc[iev]; ev = ev[eligible(ev)]
        cp = argmax_per_point(ev, oof[iev][eligible(tr.iloc[iev]).to_numpy()]).correct.mean()
        fold_rows.append(dict(fold=k, train_fold_top1=tp, cv_top1=cp, gap_pts=100 * (tp - cp),
                              held_out_trips=" ".join(sorted(tr.iloc[iev].trip_id.unique()))))
        print(fold_rows[-1] | {"held_out_trips": "..."}, flush=True)
    m = fit(tr)
    st_score = m.predict_proba(te[FEATURES])[:, 1]
    pts = pd.concat([point_table(tr, oof, "train"), point_table(te, st_score, "student_test")], ignore_index=True)
    base = pd.concat([point_table(tr, -tr.perp_dist.to_numpy(), "train"), point_table(te, -te.perp_dist.to_numpy(), "student_test")])
    pts.to_parquet(RES / "error_analysis_points.parquet", index=False)

    rows, tax = [], []
    for part in ("train", "student_test"):
        P, B = pts[pts.partition == part], base[base.partition == part]
        for s, f in SCOPES.items():
            mk, mb = f(P), f(B)
            rows.append(dict(partition=part, eval="CV out-of-fold" if part == "train" else "student_test",
                             scope=s, points=int(mk.sum()), xgb_r1config_top1=float(P[mk].correct.mean()),
                             nearest_arc_top1=float(B[mb].correct.mean()),
                             candidate_miss_rate=float((~P[mk].label_in_cands).mean())))
            vc = P[mk].category.value_counts()
            n_err = int((P[mk].category != "correct").sum())
            for c in CATEGORIES:
                tax.append(dict(partition=part, scope=s, category=c, points=int(vc.get(c, 0)),
                                pct_of_points=100 * vc.get(c, 0) / max(mk.sum(), 1),
                                pct_of_errors=np.nan if c == "correct" else 100 * vc.get(c, 0) / max(n_err, 1)))
    sc = pd.DataFrame(rows); tx = pd.DataFrame(tax); fr = pd.DataFrame(fold_rows)
    sc.to_csv(RES / "top1_by_scope_r1model.csv", index=False)
    tx.to_csv(RES / "error_analysis.csv", index=False)
    fr.to_csv(RES / "cv_folds_r1model.csv", index=False)

    pt = pts.groupby(["partition", "trip_id"]).agg(
        label_quality=("label_quality", "first"), points=("correct", "size"),
        top1_all=("correct", "mean"),
        top1_reliable=("correct", lambda s: s[~pts.loc[s.index, "unreliable"]].mean()),
        reliable_points=("unreliable", lambda s: int((~s).sum())),
        candidate_miss_rate_reliable=("label_in_cands", lambda s: float((~s[~pts.loc[s.index, "unreliable"]]).mean())),
        reverse_twin_err=("category", lambda s: float((s == "reverse_twin").mean())),
        adjacent_err=("category", lambda s: float((s == "adjacent_arc").mean()))).reset_index()
    pt = pt.sort_values("top1_reliable").reset_index(drop=True)
    pt.to_csv(RES / "per_trip_top1_r1model.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cats = CATEGORIES[1:]
    w = 0.38
    for j, part in enumerate(("train", "student_test")):
        d = tx[(tx.partition == part) & (tx.scope == "reliable_ok_trips")].set_index("category").loc[cats]
        b = ax.bar(np.arange(len(cats)) + (j - 0.5) * w, d.pct_of_errors, w, label=f"{'CV (train, out-of-fold)' if part == 'train' else 'student_test'}: {int(d.points.sum()):,} errors")
        ax.bar_label(b, fmt="%.1f%%", fontsize=8)
    ax.set_xticks(range(len(cats)), ["label arc not\nin candidates", "reverse twin\nof label arc", "adjacent arc\n(shares a node)", "other"])
    ax.set_ylabel("% of errors"); ax.legend(fontsize=8)
    ax.set_title("Where the Round 1 XGBoost config goes wrong (Round 2 labels, reliable points in ok trips)")
    fig.tight_layout(); fig.savefig(FIG / "error_taxonomy.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    lab = [f"{'ST ' if p == 'student_test' else ''}{t[-3:]}" for p, t in zip(pt.partition, pt.trip_id)]
    col = ["tab:red" if q == "low_quality" else ("tab:orange" if p == "student_test" else "tab:blue") for q, p in zip(pt.label_quality, pt.partition)]
    ax.bar(range(len(pt)), pt.top1_reliable.fillna(0), color=col)
    ax.set_xticks(range(len(pt)), lab, rotation=90, fontsize=6)
    ax.set_ylim(0, 1); ax.set_ylabel("Top-1 (reliable points)")
    ax.set_title("Per-trip Top-1, sorted ascending (blue = train CV, orange = student_test, red = low_quality trip)")
    fig.tight_layout(); fig.savefig(FIG / "per_trip_top1.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 250)
    print(fr.drop(columns="held_out_trips").round(4).to_string(index=False))
    print(sc.round(4).to_string(index=False))
    print(tx[tx.scope == "reliable_ok_trips"].round(2).to_string(index=False))
    print(pt.head(10).round(3).to_string(index=False))
    (RES / "error_analysis_meta.json").write_text(json.dumps(dict(model="XGBoost, Round 1 config (scripts/train_models.py fit_boost)",
        xgb_device=r1.XGB_DEVICE, train_rows="eligible & point_idx % 3 == 0", seed=42, runtime_s=time.time() - t0), indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
