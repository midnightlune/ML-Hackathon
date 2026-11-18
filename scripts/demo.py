"""Review demo: print the Round 1 evidence from results/ and open the map + figures.

    python -m scripts.demo            # print + open map and figures in the browser / image viewer
    python -m scripts.demo --no-open  # print only
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import json
import os
import webbrowser

import pandas as pd

from pathsnap.io import ROOT

RES = ROOT / "results"


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 30)

    inv = pd.read_csv(RES / "eda" / "inventory.csv")
    section("1. DATA: inventory and quality checks (results/eda/inventory.csv)")
    print(inv.groupby("partition").agg(trips=("trip_id", "size"), points=("n_points", "sum"),
                                       arcs_max=("n_arcs", "max"), max_gap_s=("max_gap_s", "max"),
                                       trips_gap30=("n_gaps_gt30s", lambda s: int((s > 0).sum())),
                                       invalid=("invalid_coords", "sum"), nonmono_t=("non_monotonic_ts", "sum"),
                                       route_breaks=("route_connected", lambda s: int((~s).sum())),
                                       route_repeats=("route_repeats", "sum")).to_string())
    print(f"distinct UTM zones (EPSG): {inv.epsg.nunique()}")

    thr = json.loads((RES / "eda" / "label_threshold.json").read_text())
    section("2. LABELS: monotonic DP alignment of points to the ordered route")
    q = thr["percentiles_50_90_95_99"]
    print(f"align distance p50/p90/p95/p99 = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} / {q[3]:.1f} m")
    print(f"{thr['frac_over_50m']:.1%} of points > 50 m  ->  unreliable threshold = {thr['threshold_m']:.1f} m")
    print("unreliable points (excluded from training and scoring):", thr["unreliable_counts"])

    c = json.loads((RES / "eda" / "candidates.json").read_text())
    section(f"3. CANDIDATES + FEATURES (R = {c['R_used']:.0f} m, K = {c['K']})")
    for p in ("train", "student_test"):
        print(f"{p:13s} rows {c[p]['rows']:>9,}  points {c[p]['points']:>7,}  positive rate {c[p]['positive_rate']:.3f}"
              f"  miss rate R50 {c['R50'][p]['miss_rate']:.2%}  R75 {c['R75'][p]['miss_rate']:.2%}")

    mc = pd.read_csv(RES / "model_comparison.csv")
    base = pd.read_csv(RES / "baseline_nearest_arc.csv").iloc[0]
    section("4. MODELS (results/model_comparison.csv), GroupKFold(5) by trip")
    print(mc[["Model", "Algorithm Family", "CV Top-1 Accuracy", "CV Log Loss", "CV Macro F1-score", "Top-1 Accuracy",
              "Student-Test Macro F1-score", "Log Loss", "Inference Time (ms / 1k points)", "Selected"]].to_string(index=False))
    print(f"\nbaseline (nearest arc): CV Top-1 {base.cv_top1:.3f}, student_test Top-1 {base.st_top1:.3f}")
    sel = json.loads((RES / "selection.json").read_text())
    print(f"SELECTED: {sel['selected']}  ({sel['rule']})")

    if not args.no_open:
        webbrowser.open((ROOT / "maps" / "trip_00000001_raw.html").as_uri())
        for f in sorted((ROOT / "reports" / "figures").glob("*.png")):
            os.startfile(f) if os.name == "nt" else webbrowser.open(f.as_uri())
        print("\nopened: maps/trip_00000001_raw.html and reports/figures/*.png")


if __name__ == "__main__":
    main()
