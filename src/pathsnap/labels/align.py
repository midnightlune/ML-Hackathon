"""Per-point labels from the ordered .route arc list via monotonic DP alignment.

For each GPS point i we choose a route position j_i (index into .route) such that
j_0 <= j_1 <= ... (the vehicle never goes back along the route) and the summed
point-to-segment distance sum_i d(p_i, route_arc[j_i]) is minimal.

Recurrence: D[i, j] = c[i, j] + min_{k <= j} D[i-1, k]   (prefix minimum => O(N*L) time)
To keep memory small, back-pointers are stored only inside a band of width B around each
point's nearest route arc. If the optimal path touches a band edge (the true optimum could lie
outside), or the band makes the problem infeasible, the band is doubled and the DP re-run, up to
the full route width -- so the band never silently changes the answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely

from ..geo import point_seg_dist


def load_route(trip_dir: Path) -> np.ndarray:
    tid = trip_dir.name
    p = trip_dir / f"{tid}.route"
    if not p.exists():
        return np.empty(0, dtype=np.int64)
    return np.loadtxt(p, dtype=np.int64, ndmin=1).reshape(-1)


def _banded_dp(cost_fn, n: int, L: int, centre: np.ndarray, half: int):
    B = min(2 * half + 1, L)
    lo = np.clip(centre - half, 0, L - B).astype(np.int64)
    cols = lo[:, None] + np.arange(B)[None, :]          # (n,B)
    cost = cost_fn(cols)                                 # (n,B) float32
    bp = np.empty((n, B), dtype=np.int32)
    ar = np.arange(L)
    D = np.full(L, np.inf, dtype=np.float64)
    D[cols[0]] = cost[0]
    bp[0] = cols[0]
    for i in range(1, n):
        pm = np.minimum.accumulate(D)
        idx = np.maximum.accumulate(np.where(D == pm, ar, -1))
        c = cols[i]
        D = np.full(L, np.inf, dtype=np.float64)
        D[c] = cost[i] + pm[c]
        bp[i] = idx[c]
    if not np.isfinite(D).any():
        return None
    path = np.empty(n, dtype=np.int64)
    j = int(np.argmin(D))
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = int(bp[i, j - lo[i]])
    on_edge = ((path == lo) & (lo > 0)) | ((path == lo + B - 1) & (lo + B - 1 < L - 1))
    if B < L and on_edge.any():
        return None
    return path


def align_points(pxy: np.ndarray, ra: np.ndarray, rb: np.ndarray, half: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """pxy (n,2) points in metres; ra/rb (L,2) route segment endpoints in route order.
    Returns (route_pos (n,), dist_m (n,) float32)."""
    n, L = len(pxy), len(ra)
    if n == 0 or L == 0:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.nan, dtype=np.float32)
    tree = shapely.STRtree(shapely.linestrings(np.stack([ra, rb], axis=1)))
    qi, qj = tree.query_nearest(shapely.points(pxy), all_matches=False)
    centre = np.zeros(n, dtype=np.int64)
    centre[qi] = qj

    def cost_fn(cols):
        d, _ = point_seg_dist(pxy[:, None, :], ra[cols], rb[cols])
        return d.astype(np.float32)

    while True:
        path = _banded_dp(cost_fn, n, L, centre, half)
        if path is not None:
            break
        if 2 * half + 1 >= L:
            raise RuntimeError("full-width DP infeasible (should not happen)")
        half *= 2
    d, _ = point_seg_dist(pxy, ra[path], rb[path])
    return path, d.astype(np.float32)
