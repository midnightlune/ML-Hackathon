"""Per-(point, candidate) features. Uses only the GPS track and road network (never .route).
No lon/lat and no trip id in the feature set."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import ProjTrip, point_seg_dist

HEADING_WIN = 2       # heading from points i-2 .. i+2 (clipped to the gap-free segment)
STATIONARY_M = 3.0    # window displacement below this -> heading unreliable (NaN)
FEATURES = [
    "perp_dist", "dist_rank", "dist_minus_min", "proj_fraction", "arc_length",
    "heading_cos", "heading_abs", "heading_reliable", "speed", "n_candidates",
    "has_reverse_twin", "prev_point_dist_to_cand", "next_point_dist_to_cand",
]


def _seg_bounds(seg: np.ndarray):
    n = len(seg)
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
    lens = np.diff(np.r_[starts, n])
    s = np.repeat(starts, lens)
    return s, s + np.repeat(lens, lens) - 1


def point_kinematics(pt: ProjTrip) -> pd.DataFrame:
    n = len(pt.t)
    i = np.arange(n)
    s, e = _seg_bounds(pt.segment_id)
    lo, hi = np.maximum(i - HEADING_WIN, s), np.minimum(i + HEADING_WIN, e)
    v = pt.pxy[hi] - pt.pxy[lo]
    disp = np.hypot(v[:, 0], v[:, 1])
    reliable = disp >= STATIONARY_M
    heading = np.where(reliable, np.arctan2(v[:, 1], v[:, 0]), np.nan)
    lo1, hi1 = np.maximum(i - 1, s), np.minimum(i + 1, e)
    dt = pt.t[hi1] - pt.t[lo1]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, np.linalg.norm(pt.pxy[hi1] - pt.pxy[lo1], axis=1) / dt, np.nan)
    return pd.DataFrame(dict(heading=heading, heading_reliable=reliable.astype(np.int8), speed=speed,
                             has_prev=(i > s), has_next=(i < e)))


def build(pt: ProjTrip, cand: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(cand)
    p, a = df.point_idx.to_numpy(), df.arc_id.to_numpy()
    kin = point_kinematics(pt)
    grp = df.groupby("point_idx").perp_dist
    df["dist_minus_min"] = (df.perp_dist - grp.transform("min")).astype(np.float32)
    df["n_candidates"] = grp.transform("size").astype(np.int16)
    df["arc_length"] = pt.arc_len[a].astype(np.float32)
    diff = np.angle(np.exp(1j * (kin.heading.to_numpy()[p] - pt.arc_bearing[a])))  # wrap to (-pi, pi]
    df["heading_cos"] = np.cos(diff).astype(np.float32)
    df["heading_abs"] = np.abs(diff).astype(np.float32)
    df["heading_reliable"] = kin.heading_reliable.to_numpy()[p]
    df["speed"] = kin.speed.to_numpy()[p].astype(np.float32)
    M = len(pt.nxy)
    key, rkey = pt.arcs[:, 0] * M + pt.arcs[:, 1], pt.arcs[:, 1] * M + pt.arcs[:, 0]
    df["has_reverse_twin"] = np.isin(rkey, key)[a].astype(np.int8)
    for name, off, ok in (("prev", -1, kin.has_prev), ("next", 1, kin.has_next)):
        q = np.clip(p + off, 0, len(pt.t) - 1)
        d, _ = point_seg_dist(pt.pxy[q], pt.a_xy[a], pt.b_xy[a])
        df[f"{name}_point_dist_to_cand"] = np.where(ok.to_numpy()[p], d, np.nan).astype(np.float32)
    return df
