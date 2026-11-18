"""Round 2 / Phase 2: feature-group ablation + importance of the kept set.

Protocol: XGBoost (Round 1 config, GPU), GroupKFold(5) by trip on the 80 train trips, the shared
training sample (36,000 eligible points per fit, seed 42), scored out-of-fold on all eligible points
(headline) and on all points. student_test is NOT used here.

DECISION RULE (fixed before running, logged to results/r2/ablation_rule.json):
  keep feature group g  iff  CV Top-1(all groups) - CV Top-1(all groups minus g) >= 0.001
  (headline scope). Final set = original 13 + kept groups.

Writes results/r2/{ablation.csv, ablation_rule.json, feature_set_v2.json, feature_importance_v2.csv},
reports/figures/r2/{ablation.png, feature_importance_v2.png}.
Run: python scripts\\r2_ablation.py   (~5 min on GPU)
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

import scripts.train_models as r1  # noqa: E402
from pathsnap.eval.metrics import top1  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.features_v2 import GROUPS  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402
from pathsnap.models.sampling import eligible, sample_points  # noqa: E402
from scripts.r2_common import load_feats_v2  # noqa: E402

RES, FIG = ROOT / "results" / "r2", ROOT / "reports" / "figures" / "r2"
N_POINTS = json.loads((RES / "train_size_decision.json").read_text())["shared_train_points_per_fit"]
MIN_GAIN = 0.001
RULE = dict(rule="keep group g iff CV_top1(all) - CV_top1(all minus g) >= 0.001 (headline scope: eligible points)",
            min_gain=MIN_GAIN, model="XGBoost Round 1 config", train_points_per_fit=N_POINTS, seed=42,
            logged_before_results=True)


def cv(tr, folds, cols):
    oof = np.full(len(tr), np.nan); tp = []
    for itr, iev in folds:
        f = tr.iloc[itr]; t = f[sample_points(f, N_POINTS)]
        m = r1.fit_boost("xgb", t[cols], t.label.to_numpy(), t.trip_id.to_numpy())
        oof[iev] = m.predict_proba(tr.iloc[iev][cols])[:, 1]
        fe = f[eligible(f)]; tp.append(top1(fe, m.predict_proba(fe[cols])[:, 1]))
    el = eligible(tr).to_numpy()
    per_fold = [top1(tr.iloc[iev][el[iev]], oof[iev][el[iev]]) for _, iev in folds]
    return dict(cv_top1=top1(tr[el], oof[el]), cv_top1_std=float(np.std(per_fold)),
                cv_top1_all_points=top1(tr, oof), train_fold_top1=float(np.mean(tp)),
                n_features=len(cols))


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    (RES / "ablation_rule.json").write_text(json.dumps(RULE, indent=2))
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(np.random.rand(50, 2), np.arange(50) % 2)
    except Exception as e:  # noqa: BLE001
        print("CUDA unavailable -> cpu:", e); r1.XGB_DEVICE = "cpu"
    t0 = time.time()
    tr = load_feats_v2("train")
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip_id))
    all_v2 = [c for g in GROUPS.values() for c in g]
    configs = {"original_13": FEATURES}
    configs |= {f"+{g}": FEATURES + cols for g, cols in GROUPS.items()}
    configs["all"] = FEATURES + all_v2
    configs |= {f"all_minus_{g}": FEATURES + [c for h, cols in GROUPS.items() if h != g for c in cols] for g in GROUPS}
    rows = []
    for name, cols in configs.items():
        r = dict(config=name) | cv(tr, folds, cols)
        rows.append(r)
        print(f"{name:22s} CV top1 {r['cv_top1']:.4f} (all pts {r['cv_top1_all_points']:.4f}, train-fold {r['train_fold_top1']:.4f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    ab = pd.DataFrame(rows)
    full = float(ab.loc[ab.config == "all", "cv_top1"].iloc[0])
    base = float(ab.loc[ab.config == "original_13", "cv_top1"].iloc[0])
    ab["gain_vs_original"] = ab.cv_top1 - base
    ab["loo_drop"] = np.where(ab.config.str.startswith("all_minus_"), full - ab.cv_top1, np.nan)
    ab["train_minus_cv_pts"] = 100 * (ab.train_fold_top1 - ab.cv_top1)
    kept = [g for g in GROUPS if float(ab.loc[ab.config == f"all_minus_{g}", "loo_drop"].iloc[0]) >= MIN_GAIN]
    final_cols = FEATURES + [c for g in kept for c in GROUPS[g]]
    if kept and set(kept) != set(GROUPS):
        r = dict(config="final_kept") | cv(tr, folds, final_cols)
        r |= dict(gain_vs_original=r["cv_top1"] - base, train_minus_cv_pts=100 * (r["train_fold_top1"] - r["cv_top1"]))
        ab = pd.concat([ab, pd.DataFrame([r])], ignore_index=True)
    ab.to_csv(RES / "ablation.csv", index=False)
    (RES / "feature_set_v2.json").write_text(json.dumps(dict(kept_groups=kept, dropped_groups=[g for g in GROUPS if g not in kept],
                                                             features=final_cols, n_features=len(final_cols)), indent=2))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    show = ab[~ab.config.str.startswith("all_minus_")]
    c = ["0.6" if n == "original_13" else ("tab:green" if n in ("all", "final_kept") else "tab:blue") for n in show.config]
    b = ax.bar(show.config, show.cv_top1, color=c)
    ax.bar_label(b, fmt="%.4f", fontsize=8)
    lo = ab[ab.config.str.startswith("all_minus_")]
    ax2 = ax.twinx(); ax2.plot([f"+{n[10:]}" for n in lo.config], lo.loo_drop, "o", color="tab:red", label="drop when removed from 'all'")
    ax2.axhline(MIN_GAIN, color="tab:red", ls=":", lw=1); ax2.set_ylabel("leave-one-group-out drop", color="tab:red")
    ax.set_ylim(min(show.cv_top1) - 0.02, max(show.cv_top1) + 0.01); ax.set_ylabel("CV Top-1 (eligible points)")
    ax.set_title(f"Feature-group ablation, XGBoost, GroupKFold(5) on train. Kept: {', '.join(kept) or 'none'}")
    ax.tick_params(axis="x", rotation=25); ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ablation.png", dpi=150); plt.close(fig)

    # importance of the final set: gain + mean |SHAP| (XGBoost pred_contribs) on a fixed sample
    t = tr[sample_points(tr, N_POINTS)]
    m = r1.fit_boost("xgb", t[final_cols], t.label.to_numpy(), t.trip_id.to_numpy())
    bst = m.get_booster()
    gain = bst.get_score(importance_type="gain")
    smp = tr[eligible(tr)].sample(n=50_000, random_state=42)
    contrib = bst.predict(xgb.DMatrix(smp[final_cols]), pred_contribs=True)[:, :-1]
    imp = pd.DataFrame({"feature": final_cols, "gain": [gain.get(f, 0.0) for f in final_cols],
                        "mean_abs_shap": np.abs(contrib).mean(0)})
    imp["gain_share"] = imp.gain / imp.gain.sum(); imp["shap_share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(RES / "feature_importance_v2.csv", index=False)
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    ax[0].barh(top.feature, top.shap_share, color="tab:blue"); ax[0].set_title("mean |SHAP| share")
    ax[1].barh(top.feature, top.gain_share, color="tab:gray"); ax[1].set_title("gain share")
    fig.suptitle(f"Top-20 feature importance, final v2 set ({len(final_cols)} features), XGBoost on train")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance_v2.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 250)
    print(ab.round(4).to_string(index=False)); print("kept:", kept)
    print(imp.head(12).round(4).to_string(index=False)); print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
