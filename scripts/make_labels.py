"""Step 2: per-point labels by monotonic DP alignment of track points to the ordered .route arcs.
Writes data/processed/labels_<partition>.parquet and results/eda/label_alignment.csv.

Run: python -m scripts.make_labels
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
import time
import traceback

import numpy as np
import pandas as pd

from pathsnap.geo import project_trip
from pathsnap.io import PARTITIONS, ROOT, load_trip, trip_dirs
from pathsnap.labels.align import align_points, load_route

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("labels")
OUT = ROOT / "data" / "processed"
BASE_THRESH = 50.0


def main():
    frames, fails = [], []
    t0 = time.time()
    for part in PARTITIONS:
        for d in trip_dirs(part):
            try:
                trip = load_trip(d)
                pt = project_trip(trip)
                route = load_route(d)
                v = np.flatnonzero(pt.valid)
                pos, dist = align_points(pt.pxy[v], pt.a_xy[route], pt.b_xy[route])
                assert np.all(np.diff(pos) >= 0), f"non-monotonic alignment in {trip.trip_id}"
                n = len(pt.t)
                rp = np.full(n, -1, np.int64); rp[v] = pos
                ad = np.full(n, np.inf, np.float32); ad[v] = dist
                frames.append(pd.DataFrame(dict(
                    partition=part, trip_id=trip.trip_id, point_idx=np.arange(n, dtype=np.int32),
                    route_pos=rp, label_arc_id=np.where(rp >= 0, route[np.maximum(rp, 0)], -1),
                    align_dist_m=ad)))
            except Exception:
                log.error("trip %s/%s failed:\n%s", part, d.name, traceback.format_exc())
                fails.append((part, d.name))
    lab = pd.concat(frames, ignore_index=True)
    print(f"aligned {lab.trip_id.nunique()} trips, {len(lab)} points in {time.time() - t0:.1f}s; failures: {fails or 'none'}")

    # monotonic check over the stored table (per trip, in point order)
    mono = lab.groupby(["partition", "trip_id"]).route_pos.apply(lambda s: bool(np.all(np.diff(s.values) >= 0)))
    assert mono.all(), mono[~mono]
    print("monotonic route_pos in every trip: OK")

    q = np.percentile(lab.align_dist_m[np.isfinite(lab.align_dist_m)], [50, 90, 95, 99])
    print("align_dist_m percentiles (50/90/95/99): " + " / ".join(f"{x:.2f}" for x in q))
    frac_over = float((lab.align_dist_m > BASE_THRESH).mean())
    if frac_over > 0.05:
        thresh = float(q[2])
        print(f"{frac_over:.2%} of points exceed {BASE_THRESH} m (>5%) -> using 95th percentile = {thresh:.2f} m as threshold")
    else:
        thresh = BASE_THRESH
        print(f"{frac_over:.2%} of points exceed {BASE_THRESH} m (<=5%) -> threshold stays {BASE_THRESH} m")
    lab["unreliable"] = ~(lab.align_dist_m <= thresh)  # inf (invalid coords) -> unreliable
    summ = lab.groupby("partition").agg(points=("point_idx", "size"), unreliable=("unreliable", "sum"))
    summ["pct_unreliable"] = 100 * summ.unreliable / summ.points
    print(summ.to_string())
    per_trip = lab.groupby(["partition", "trip_id"]).agg(
        points=("point_idx", "size"), unreliable=("unreliable", "sum"),
        median_dist=("align_dist_m", "median"), p95_dist=("align_dist_m", lambda s: s.quantile(0.95)),
        route_arcs_used=("route_pos", "nunique")).reset_index()
    per_trip.to_csv(ROOT / "results" / "eda" / "label_alignment.csv", index=False)
    print("worst trips by % unreliable:")
    per_trip["pct"] = 100 * per_trip.unreliable / per_trip.points
    print(per_trip.sort_values("pct", ascending=False).head(5).to_string(index=False))
    (ROOT / "results" / "eda" / "label_threshold.json").write_text(json.dumps(
        dict(percentiles_50_90_95_99=[float(x) for x in q], frac_over_50m=frac_over, threshold_m=thresh,
             unreliable_counts=summ.unreliable.astype(int).to_dict()), indent=2))
    for part in PARTITIONS:
        lab[lab.partition == part].drop(columns="partition").to_parquet(OUT / f"labels_{part}.parquet", index=False)


if __name__ == "__main__":
    main()
