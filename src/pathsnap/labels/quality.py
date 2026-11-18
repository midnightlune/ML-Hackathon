"""Round 2 label-reliability rules. Chosen ONCE from TRAIN data in Phase 1 and frozen.

Evidence (scripts/r2_labels.py -> reports/figures/r2/label_threshold.png):
- point level: within trips whose median alignment distance is <= 30 m, the distribution of
  align_dist_m peaks near 6 m and has its trough at ~40-50 m, followed by a second bump at
  ~58-72 m (a separate population of misaligned points). 50 m sits in that trough.
- trip level: 78/80 train trips have a median <= 11 m; two have a median > 100 m.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POINT_THRESHOLD_M = 50.0      # point is unreliable if align_dist_m > this (or not finite)
TRIP_MEDIAN_MAX_M = 30.0      # trip is low_quality if its median align_dist_m exceeds this ...
TRIP_UNRELIABLE_MAX = 0.25    # ... or more than this share of its points is unreliable

RULES = dict(point_threshold_m=POINT_THRESHOLD_M, trip_median_max_m=TRIP_MEDIAN_MAX_M,
             trip_unreliable_max=TRIP_UNRELIABLE_MAX)


def apply_rules(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """labels: one row per point with trip_id, align_dist_m. Returns (labels + `unreliable`,
    `label_quality`), per-trip table)."""
    out = labels.copy()
    out["unreliable"] = ~(out.align_dist_m <= POINT_THRESHOLD_M)   # inf/NaN -> unreliable
    trips = out.groupby("trip_id").agg(points=("align_dist_m", "size"),
                                       median_align_m=("align_dist_m", "median"),
                                       unreliable_frac=("unreliable", "mean")).reset_index()
    low = (trips.median_align_m > TRIP_MEDIAN_MAX_M) | (trips.unreliable_frac > TRIP_UNRELIABLE_MAX)
    trips["label_quality"] = np.where(low, "low_quality", "ok")
    out = out.merge(trips[["trip_id", "label_quality"]], on="trip_id", how="left")
    return out, trips
