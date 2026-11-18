"""Round 2 / Phase 1.1: frozen label-reliability rules (chosen from TRAIN only), applied to both partitions.

Reads the Round 1 DP alignment (data/processed/labels_*.parquet, read-only; the alignment itself is
unchanged) and writes data/processed/r2/labels_*.parquet with `unreliable` (50 m rule) and the per-trip
`label_quality` flag. Outputs: results/r2/label_threshold.json, results/r2/label_quality_trips.csv,
results/r2/label_changes.csv, reports/figures/r2/label_threshold.png.

Run: python scripts\\r2_labels.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.io import PARTITIONS, ROOT  # noqa: E402
from pathsnap.labels import quality  # noqa: E402

PROC, OUT, RES, FIG = ROOT / "data" / "processed", ROOT / "data" / "processed" / "r2", ROOT / "results" / "r2", ROOT / "reports" / "figures" / "r2"


def threshold_figure(train: pd.DataFrame, trips: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    t = trips.sort_values("median_align_m").reset_index(drop=True)
    col = np.where(t.label_quality == "low_quality", "tab:red", "tab:blue")
    ax[0].bar(np.arange(len(t)), t.median_align_m, color=col)
    ax[0].axhline(quality.TRIP_MEDIAN_MAX_M, color="tab:red", ls="--", lw=1, label=f"trip rule: median > {quality.TRIP_MEDIAN_MAX_M:.0f} m")
    ax[0].set_yscale("log"); ax[0].set_xlabel("train trips, sorted"); ax[0].set_ylabel("median align distance (m, log)")
    for _, r in t[t.label_quality == "low_quality"].iterrows():
        ax[0].annotate(r.trip_id, (r.name, r.median_align_m), textcoords="offset points", xytext=(-60, -4), fontsize=8, color="tab:red")
    ax[0].set_title("Per-trip median alignment distance (TRAIN, 80 trips)\nred = low_quality"); ax[0].legend(fontsize=8, loc="upper left")

    good = train[train.label_quality == "ok"].align_dist_m.clip(lower=0.1)
    bins = np.logspace(-1, 3.5, 70)
    ax[1].hist(good, bins=bins, color="tab:blue", alpha=0.85, label="points in ok trips")
    ax[1].hist(train[train.label_quality != "ok"].align_dist_m.clip(lower=0.1), bins=bins, color="tab:red", alpha=0.6, label="points in low_quality trips")
    ax[1].axvline(quality.POINT_THRESHOLD_M, color="k", ls="--", lw=1.2, label=f"frozen threshold {quality.POINT_THRESHOLD_M:.0f} m")
    ax[1].axvline(148.08, color="0.5", ls=":", lw=1.2, label="Round 1 threshold 148 m (p95 rule)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("align distance (m, log)"); ax[1].set_ylabel("points (log)")
    ax[1].set_title("Point-level distribution (TRAIN)\nok trips: mode ~6 m, trough ~40-50 m, then a misaligned bump")
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "label_threshold.png", dpi=150); plt.close(fig)


def main():
    for d in (OUT, RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    r1 = {p: pd.read_parquet(PROC / f"labels_{p}.parquet") for p in PARTITIONS}
    new, trips = {}, []
    for p in PARTITIONS:   # same frozen rules, applied independently per partition
        lab, tq = quality.apply_rules(r1[p].drop(columns="unreliable"))
        lab = lab.merge(r1[p][["trip_id", "point_idx", "unreliable"]].rename(columns={"unreliable": "unreliable_r1"}),
                        on=["trip_id", "point_idx"])
        new[p] = lab
        trips.append(tq.assign(partition=p))
        lab.drop(columns="unreliable_r1").to_parquet(OUT / f"labels_{p}.parquet", index=False)
    trips = pd.concat(trips, ignore_index=True)
    trips.to_csv(RES / "label_quality_trips.csv", index=False)
    threshold_figure(new["train"], trips[trips.partition == "train"])

    rows = []
    for p, lab in new.items():
        ok = lab.label_quality == "ok"
        rows.append(dict(partition=p, trips=lab.trip_id.nunique(), points=len(lab),
                         low_quality_trips=int((trips[trips.partition == p].label_quality == "low_quality").sum()),
                         low_quality_trip_ids=" ".join(trips[(trips.partition == p) & (trips.label_quality == "low_quality")].trip_id),
                         points_in_low_quality_trips=int((~ok).sum()),
                         unreliable_r1_148m=int(lab.unreliable_r1.sum()), unreliable_r2_50m=int(lab.unreliable.sum()),
                         newly_unreliable=int((lab.unreliable & ~lab.unreliable_r1).sum()),
                         unreliable_in_ok_trips=int((lab.unreliable & ok).sum()),
                         train_eligible_points=int((~lab.unreliable & ok).sum()),
                         pct_points_train_eligible=100 * float((~lab.unreliable & ok).mean())))
    ch = pd.DataFrame(rows)
    ch.to_csv(RES / "label_changes.csv", index=False)
    tr = new["train"]
    good = tr[tr.label_quality == "ok"].align_dist_m
    info = dict(rules=quality.RULES, chosen_from="train only",
                train_point_pct_over_threshold=100 * float((tr.align_dist_m > quality.POINT_THRESHOLD_M).mean()),
                train_ok_trips_point_pct_over_threshold=100 * float((good > quality.POINT_THRESHOLD_M).mean()),
                train_ok_trips_point_percentiles_50_90_95_99=[float(x) for x in np.percentile(good, [50, 90, 95, 99])],
                round1_threshold_m=148.08)
    (RES / "label_threshold.json").write_text(json.dumps(info, indent=2))
    pd.set_option("display.width", 250)
    print(json.dumps(info, indent=2))
    print(ch.to_string(index=False))
    print(trips[trips.label_quality == "low_quality"].to_string(index=False))


if __name__ == "__main__":
    main()
