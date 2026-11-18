"""Candidate road arcs per GPS point (STRtree, metric CRS). Never reads ground truth."""
from __future__ import annotations

import numpy as np
import shapely

from .geo import ProjTrip, point_seg_dist

R_DEFAULT = 50.0
K_DEFAULT = 12
FALLBACK_K = 3


def _rank_within(groups: np.ndarray, key: np.ndarray):
    """Sort by (group, key); return (order, rank of each sorted row within its group)."""
    order = np.lexsort((key, groups))
    g = groups[order]
    if len(g) == 0:
        return order, np.zeros(0, np.int64)
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    return order, rank


def generate(pt: ProjTrip, R: float = R_DEFAULT, K: int = K_DEFAULT) -> dict[str, np.ndarray]:
    """Arcs within R metres of each valid point, top K by distance. Points with none get their
    nearest FALLBACK_K arcs. Returns equal-length arrays."""
    pidx = np.flatnonzero(pt.valid)
    pts = shapely.points(pt.pxy[pidx])
    tree = pt.arc_tree
    qi, qa = tree.query(pts, predicate="dwithin", distance=R)
    fb = np.zeros(len(qi), bool)
    missing = np.setdiff1d(np.arange(len(pidx)), qi)
    if len(missing):
        mi, ma = tree.query(pts[missing], predicate="dwithin", distance=500.0)
        still = np.setdiff1d(np.arange(len(missing)), mi)
        if len(still):
            si, sa = tree.query_nearest(pts[missing][still], all_matches=True)
            mi, ma = np.r_[mi, still[si]], np.r_[ma, sa]
        d, _ = point_seg_dist(pt.pxy[pidx[missing[mi]]], pt.a_xy[ma], pt.b_xy[ma])
        o, r = _rank_within(mi, d)
        keep = o[r < FALLBACK_K]
        qi, qa = np.r_[qi, missing[mi[keep]]], np.r_[qa, ma[keep]]
        fb = np.r_[fb, np.ones(len(keep), bool)]
    d, t = point_seg_dist(pt.pxy[pidx[qi]], pt.a_xy[qa], pt.b_xy[qa])
    o, r = _rank_within(qi, d)
    sel = r < K
    keep = o[sel]
    return dict(point_idx=pidx[qi[keep]].astype(np.int32), arc_id=qa[keep].astype(np.int64),
                perp_dist=d[keep].astype(np.float32), proj_fraction=t[keep].astype(np.float32),
                dist_rank=r[sel].astype(np.int16), fallback=fb[keep])
