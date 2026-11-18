"""EDA + feature-engineering figures for the report (deliverables section 2.2). No retraining of the comparison;
feature importance comes from one XGBoost fit with exactly the round-1 config on the same training rows.

Writes reports/figures/eda_*.png, results/eda/feature_missing.csv, results/eda/feature_importance.csv.
Run: python scripts\\eda_plots.py   (~1 min; needs build_features to have run)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.io import RAW, ROOT, load_trip  # noqa: E402
from pathsnap.labels.align import load_route  # noqa: E402  (plot of the reference route only)

PROC, EDA, FIG = ROOT / "data" / "processed", ROOT / "results" / "eda", ROOT / "reports" / "figures"
SEED = 42
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def sampling_and_gaps(pts):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    dt = pts.groupby("trip_id").t.diff().dropna()
    bins = np.r_[0.5, 1.5, 2.5, 5, 10, 30, 60, 300, 1e3, 1e4, 2e5]
    cnt, _ = np.histogram(dt, bins=bins)
    labels = ["1 s", "2 s", "3-5", "5-10", "10-30", "30-60", "1-5 min", "5-17 min", "17 min-3 h", "> 3 h"]
    ax[0].bar(range(len(cnt)), cnt, color=["tab:blue" if b < 30 else "tab:red" for b in bins[1:]])
    ax[0].set_yscale("log"); ax[0].set_xticks(range(len(cnt)), labels, rotation=45, ha="right")
    ax[0].set_title(f"Time between GPS fixes (all 90 trips)\n{(dt == 1).mean():.1%} are exactly 1 s; red = gap > 30 s -> new segment")
    ax[0].set_ylabel("count (log)")
    n = pts.groupby(["part", "trip_id"]).size().reset_index(name="n")
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        ax[1].hist(n[n.part == part].n, bins=np.logspace(2, 4.4, 25), color=c, alpha=0.7, label=part)
    ax[1].set_xscale("log"); ax[1].set_xlabel("GPS points per trip (log)"); ax[1].set_ylabel("trips")
    ax[1].set_title("Trip length varies 74x (262 to 19,313 points)"); ax[1].legend()
    save(fig, "eda_sampling_and_gaps.png")


def road_network(inv):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        d = inv[inv.partition == part]
        ax[0].scatter(d.n_nodes, d.n_arcs, s=14, color=c, label=part, alpha=0.8)
        ax[1].scatter(d.n_arcs, d.largest_component_frac, s=14, color=c, label=part, alpha=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs (directed)")
    ax[0].set_title(f"Road graph size per trip\n{inv.n_arcs.min():,} to {inv.n_arcs.max():,} arcs"); ax[0].legend()
    ax[1].set_xscale("log"); ax[1].set_xlabel("arcs"); ax[1].set_ylabel("share of nodes in largest component")
    ax[1].set_title(f"Connectivity: median {inv.largest_component_frac.median():.1%} of nodes in one component\n"
                    f"reverse twins: {inv.frac_arcs_with_reverse.mean():.0%} of arcs on average")
    save(fig, "eda_road_network.png")


def trip_map(trip_id="00000001", part="train"):
    d = RAW / part / trip_id
    trip = load_trip(d)
    pt = project_trip(trip)
    route = load_route(d)
    near = np.unique(pt.arc_tree.query(shapely.points(pt.pxy), predicate="dwithin", distance=300)[1])
    o = pt.pxy.min(0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    c = (pt.pxy[-150] - o)  # zoom window near the end of the trip (in town), 800 m wide
    for k, ax in enumerate(axs):
        ax.add_collection(LineCollection(np.stack([pt.a_xy[near], pt.b_xy[near]], 1) - o, colors="0.7", linewidths=0.7,
                                         label=f"road network ({len(near):,} arcs within 300 m)"))
        ax.scatter(*(pt.pxy - o).T, s=4 if k == 0 else 14, color="tab:blue", zorder=3, label=f"raw GPS ({len(pt.pxy):,} points)")
        ax.add_collection(LineCollection(np.stack([pt.a_xy[route], pt.b_xy[route]], 1) - o, colors="tab:green",
                                         linewidths=1.2 if k == 0 else 2.2, zorder=4, label=f"reference route ({len(route)} arcs)"))
        ax.set_aspect("equal"); ax.set_xlabel("x (m, UTM)"); ax.set_ylabel("y (m, UTM)")
    axs[0].autoscale(); axs[0].set_title(f"Whole trip (EPSG {pt.crs.to_epsg()}); box = zoom")
    axs[0].add_patch(plt.Rectangle(c - 400, 800, 800, fill=False, ec="tab:red", lw=1.2, zorder=5))
    axs[1].set_xlim(c[0] - 400, c[0] + 400); axs[1].set_ylim(c[1] - 400, c[1] + 400)
    axs[1].set_title("Zoom (800 m): GPS points sit beside the road, not on it"); axs[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Trip {trip_id} ({part}) projected to metres in its local UTM zone")
    save(fig, "eda_trip_00000001_metres.png")


def alignment(lab, thr):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    d = lab.align_dist_m[np.isfinite(lab.align_dist_m)].clip(lower=0.1)
    ax.hist(d, bins=np.logspace(-1, 4, 60), color="tab:blue")
    ax.axvline(50, color="tab:orange", ls="--", label="50 m (planned threshold)")
    ax.axvline(thr["threshold_m"], color="tab:red", ls="--", label=f"{thr['threshold_m']:.0f} m (p95, used: {thr['frac_over_50m']:.1%} > 50 m)")
    ax.set_xscale("log"); ax.set_xlabel("distance from GPS point to its aligned route arc (m, log)"); ax.set_ylabel("points")
    ax.set_title(f"Label alignment quality: median {np.median(d):.1f} m"); ax.legend(fontsize=8)
    save(fig, "eda_label_alignment.png")


def missing(feats):
    m = pd.DataFrame({"feature": FEATURES, "pct_missing": [100 * feats[f].isna().mean() for f in FEATURES]})
    m["why"] = m.feature.map({"heading_cos": "car stationary (window < 3 m) -> heading undefined",
                              "heading_abs": "car stationary (window < 3 m) -> heading undefined",
                              "speed": "single-point segment (no neighbour)",
                              "prev_point_dist_to_cand": "first point of a segment",
                              "next_point_dist_to_cand": "last point of a segment"}).fillna("never missing")
    m.to_csv(EDA / "feature_missing.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(m.feature, m.pct_missing, color=["tab:red" if v > 0 else "0.7" for v in m.pct_missing])
    for i, v in enumerate(m.pct_missing):
        ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("% of train candidate rows missing")
    ax.set_title("Missing values per feature (raw GPS: 0 invalid coords, 0 bad timestamps)\n"
                 "NaN kept as-is for trees; median-imputed inside CV for linear/KNN/MLP")
    save(fig, "eda_feature_missing.png")


def distributions(feats):
    rel = feats[~feats.unreliable].sample(n=min(400_000, len(feats)), random_state=SEED)
    show = [("perp_dist", "distance to road (m)", (0, 75)), ("heading_abs", "|heading difference| (rad)", (0, np.pi)),
            ("dist_rank", "distance rank (0 = closest)", (-0.5, 11.5)), ("prev_point_dist_to_cand", "prev point to road (m)", (0, 100)),
            ("speed", "speed (m/s)", (0, 40)), ("proj_fraction", "position along arc (0-1)", (0, 1))]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.5))
    for ax, (f, lab, rng) in zip(axs.ravel(), show):
        bins = np.arange(-0.5, 12.5) if f == "dist_rank" else np.linspace(*rng, 40)
        for y, c, name in ((0, "0.6", "wrong road"), (1, "tab:blue", "right road")):
            v = rel.loc[rel.label == y, f].dropna()
            ax.hist(v, bins=bins, density=True, histtype="stepfilled" if y == 0 else "step", color=c, lw=1.6, alpha=0.6 if y == 0 else 1, label=name)
        ax.set_title(lab)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions: right road vs wrong road (400k reliable train rows, density)")
    save(fig, "eda_feature_distributions.png")


def importance(feats):
    import xgboost as xgb  # same config as the round-1 comparison
    from scripts.train_models import fit_boost, train_rows
    rel = feats[~feats.unreliable].reset_index(drop=True)
    tr = train_rows(rel, None)
    try:
        m = fit_boost("xgb", tr[FEATURES], tr.label.to_numpy(), tr.trip_id.to_numpy())
    except xgb.core.XGBoostError:
        import scripts.train_models as tm
        tm.XGB_DEVICE = "cpu"
        m = fit_boost("xgb", tr[FEATURES], tr.label.to_numpy(), tr.trip_id.to_numpy())
    g = m.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": FEATURES, "gain": [g.get(f, 0.0) for f in FEATURES]})
    imp["share"] = imp.gain / imp.gain.sum()
    imp = imp.sort_values("share", ascending=False)
    imp.to_csv(EDA / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp.feature, imp.share, color="tab:blue")
    for i, v in enumerate(imp.share):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("share of total gain")
    ax.set_title("XGBoost feature importance (gain), selected model config, train trips only")
    save(fig, "eda_feature_importance.png")
    return imp


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pts = pd.concat([pd.read_parquet(PROC / f"points_{p}.parquet").assign(part=p) for p in ("train", "student_test")])
    inv = pd.read_csv(EDA / "inventory.csv", dtype={"trip_id": str})
    lab = pd.concat([pd.read_parquet(PROC / f"labels_{p}.parquet") for p in ("train", "student_test")])
    thr = json.loads((EDA / "label_threshold.json").read_text())
    feats = pd.read_parquet(PROC / "train_feats.parquet")
    sampling_and_gaps(pts)
    road_network(inv)
    trip_map()
    alignment(lab, thr)
    missing(feats)
    distributions(feats)
    imp = importance(feats)
    print("\nfeature importance (gain share):\n" + imp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nmissing values:\n" + pd.read_csv(EDA / "feature_missing.csv").to_string(index=False))


if __name__ == "__main__":
    main()
