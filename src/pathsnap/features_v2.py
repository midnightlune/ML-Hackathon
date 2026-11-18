"""Round 2 features (v2_*), computed per trip from the GPS track and road graph only (never .route).

Groups (each a list of column names, used by the ablation):
- smooth:   Savitzky-Golay-smoothed track (per gap-free segment) -> perp distance and heading match
- heading:  multi-scale heading (+-2, +-5, +-10 points) vs arc bearing, and heading consistency
            (circular variance of step directions in +-5)
- temporal: mean/min distance of points i-k..i+k (k=3, 7; same segment) to the candidate arc, and the
            share of those points whose nearest candidate is this arc
- along:    projected position along the arc in metres: to start / to end node, unclamped overshoot
- graph:    connected (correct direction) to a top-3-by-distance candidate of point i-1 / i+1,
            in/out degree, dead end
- relative: z-scores and ranks of perp_dist / heading_cos within the point's candidate set,
            is_reverse_twin_of_nearest
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .features import _seg_bounds
from .geo import ProjTrip, point_seg_dist

STATIONARY_M = 3.0
SG_WINDOW, SG_ORDER = 9, 2
HEADING_SCALES = (2, 5, 10)
TEMPORAL_K = (3, 7)

GROUPS = {
    "smooth": ["v2_s_perp_dist", "v2_s_heading_cos", "v2_s_heading_abs", "v2_s_shift_m"],
    "heading": [f"v2_hcos_{w}" for w in HEADING_SCALES] + [f"v2_habs_{w}" for w in HEADING_SCALES] + ["v2_hvar_5"],
    "temporal": [f"v2_win{k}_{s}" for k in TEMPORAL_K for s in ("mean_dist", "min_dist", "frac_nearest")],
    "along": ["v2_dist_to_start_m", "v2_dist_to_end_m", "v2_overshoot_m"],
    "graph": ["v2_conn_prev", "v2_conn_next", "v2_out_deg_end", "v2_in_deg_start", "v2_dead_end"],
    "relative": ["v2_perp_z", "v2_hcos_rank", "v2_hcos_z", "v2_hcos_minus_max", "v2_is_twin_of_nearest"],
}
ALL_V2 = [c for g in GROUPS.values() for c in g]


def _wrap(a):
    return np.angle(np.exp(1j * a))


def _heading(pxy, s, e, w):
    i = np.arange(len(pxy))
    lo, hi = np.maximum(i - w, s), np.minimum(i + w, e)
    v = pxy[hi] - pxy[lo]
    ok = np.hypot(v[:, 0], v[:, 1]) >= STATIONARY_M
    return np.where(ok, np.arctan2(v[:, 1], v[:, 0]), np.nan)


def _smooth(pt: ProjTrip, s, e):
    out = pt.pxy.copy()
    for a in np.unique(s):
        b = e[a] + 1
        n = b - a
        if n >= 5:
            wl = min(SG_WINDOW, n if n % 2 else n - 1)
            out[a:b] = savgol_filter(pt.pxy[a:b], wl, min(SG_ORDER, wl - 1), axis=0)
    return out


def _circ_var(pxy, s, e, w):
    """1 - |mean unit step vector| over steps inside [i-w, i+w] (same segment)."""
    n = len(pxy)
    d = np.diff(pxy, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    moving = L > 0.5
    ux = np.where(moving, d[:, 0] / np.where(moving, L, 1), 0)
    uy = np.where(moving, d[:, 1] / np.where(moving, L, 1), 0)
    cx, cy, cc = (np.r_[0, np.cumsum(x)] for x in (ux, uy, moving.astype(float)))
    i = np.arange(n)
    lo, hi = np.maximum(i - w, s), np.minimum(i + w, e)        # steps lo .. hi-1
    sx, sy, cnt = cx[hi] - cx[lo], cy[hi] - cy[lo], cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.hypot(sx, sy) / cnt
    return np.where(cnt >= 2, 1 - r, np.nan)


def build_v2(pt: ProjTrip, cand: pd.DataFrame) -> pd.DataFrame:
    """cand: this trip's candidate rows with point_idx, arc_id, perp_dist, proj_fraction, dist_rank,
    heading_cos (Round 1 columns). Returns a frame aligned to cand.index with the v2 columns."""
    p = cand.point_idx.to_numpy()
    a = cand.arc_id.to_numpy()
    n = len(pt.t)
    s, e = _seg_bounds(pt.segment_id)
    A, B, bear, alen = pt.a_xy[a], pt.b_xy[a], pt.arc_bearing[a], pt.arc_len[a]
    out = {}

    # smooth
    sxy = _smooth(pt, s, e)
    d, _ = point_seg_dist(sxy[p], A, B)
    out["v2_s_perp_dist"] = d
    sh = _wrap(_heading(sxy, s, e, 2)[p] - bear)
    out["v2_s_heading_cos"], out["v2_s_heading_abs"] = np.cos(sh), np.abs(sh)
    out["v2_s_shift_m"] = np.hypot(*(sxy - pt.pxy)[p].T)

    # heading (multi-scale)
    for w in HEADING_SCALES:
        h = _wrap(_heading(pt.pxy, s, e, w)[p] - bear)
        out[f"v2_hcos_{w}"], out[f"v2_habs_{w}"] = np.cos(h), np.abs(h)
    out["v2_hvar_5"] = _circ_var(pt.pxy, s, e, 5)[p]

    # temporal support
    nearest = np.full(n, -1, np.int64)
    r0 = cand.dist_rank.to_numpy() == 0
    nearest[p[r0]] = a[r0]
    for k in TEMPORAL_K:
        tot = np.zeros(len(p)); cnt = np.zeros(len(p)); mn = np.full(len(p), np.inf); fn = np.zeros(len(p))
        for o in range(-k, k + 1):
            q = p + o
            ok = (q >= s[p]) & (q <= e[p])
            qc = np.clip(q, 0, n - 1)
            dq, _ = point_seg_dist(pt.pxy[qc], A, B)
            tot += np.where(ok, dq, 0); cnt += ok; mn = np.where(ok, np.minimum(mn, dq), mn)
            fn += ok & (nearest[qc] == a)
        out[f"v2_win{k}_mean_dist"] = tot / cnt
        out[f"v2_win{k}_min_dist"] = mn
        out[f"v2_win{k}_frac_nearest"] = fn / cnt

    # along-track (unclamped projection, metres)
    ab = B - A
    den = np.maximum((ab * ab).sum(1), 1e-9)
    tu = ((pt.pxy[p] - A) * ab).sum(1) / den
    out["v2_dist_to_start_m"] = np.clip(tu, 0, 1) * alen
    out["v2_dist_to_end_m"] = (1 - np.clip(tu, 0, 1)) * alen
    out["v2_overshoot_m"] = np.where(tu < 0, -tu, np.where(tu > 1, tu - 1, 0)) * alen

    # graph context
    fr, to = pt.arcs[:, 0], pt.arcs[:, 1]
    top3 = np.full((n, 3), -1, np.int64)
    rk = cand.dist_rank.to_numpy()
    m3 = rk < 3
    top3[p[m3], rk[m3]] = a[m3]
    for name, off in (("prev", -1), ("next", 1)):
        q = p + off
        ok = (q >= s[p]) & (q <= e[p])
        c = top3[np.clip(q, 0, n - 1)]                         # (rows, 3)
        valid = (c >= 0) & ok[:, None]
        cc = np.where(valid, c, 0)
        if name == "prev":   # candidate of i-1 is this arc or leads into it
            link = (cc == a[:, None]) | (to[cc] == fr[a][:, None])
        else:                # this arc is the same as, or leads into, a candidate of i+1
            link = (cc == a[:, None]) | (fr[cc] == to[a][:, None])
        out[f"v2_conn_{name}"] = np.where(ok, (link & valid).any(1), np.nan)
    M = len(pt.nxy)
    out_deg = np.bincount(fr, minlength=M)
    in_deg = np.bincount(to, minlength=M)
    out["v2_out_deg_end"] = out_deg[to[a]]
    out["v2_in_deg_start"] = in_deg[fr[a]]
    key = fr * M + to
    has_twin = np.isin(to * M + fr, key)
    out["v2_dead_end"] = (out_deg[to[a]] - has_twin[a]) <= 0

    df = pd.DataFrame(out, index=cand.index)

    # relative, within the point's candidate set
    g = cand.groupby("point_idx")
    pdist, hc = cand.perp_dist, cand.heading_cos
    sd = g.perp_dist.transform("std").fillna(0)
    df["v2_perp_z"] = np.where(sd > 0, (pdist - g.perp_dist.transform("mean")) / sd.where(sd > 0, 1), 0)
    df["v2_hcos_rank"] = g.heading_cos.rank(ascending=False, method="min")
    hsd = g.heading_cos.transform("std")
    df["v2_hcos_z"] = np.where(hsd > 0, (hc - g.heading_cos.transform("mean")) / hsd.where(hsd > 0, 1), np.nan)
    df["v2_hcos_minus_max"] = hc - g.heading_cos.transform("max")
    near = nearest[p]
    nn = np.where(near >= 0, near, 0)
    df["v2_is_twin_of_nearest"] = (near >= 0) & (fr[a] == to[nn]) & (to[a] == fr[nn])
    return df[ALL_V2].astype(np.float32)
