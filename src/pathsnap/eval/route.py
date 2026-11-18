"""Route-level evaluation against the reference .route (evaluation only; reads ground truth)."""
from __future__ import annotations

import numpy as np


def collapse(arcs: np.ndarray) -> np.ndarray:
    """Predicted per-point arcs -> ordered arc sequence without consecutive repeats."""
    if len(arcs) == 0:
        return arcs
    return arcs[np.r_[True, arcs[1:] != arcs[:-1]]]


def route_prf(pred_arcs: np.ndarray, route: np.ndarray) -> dict:
    """Set-based precision / recall / F1 of the collapsed predicted arc sequence vs the reference route."""
    P, R = set(collapse(pred_arcs).tolist()), set(route.tolist())
    tp = len(P & R)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(R) if R else 0.0
    return dict(route_precision=prec, route_recall=rec, route_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray, same_segment_next: np.ndarray) -> tuple[int, int]:
    """(connected changes, total changes) between consecutive points of the same segment."""
    a0, a1 = pred_arcs[:-1], pred_arcs[1:]
    m = same_segment_next[:-1] & (a0 != a1)
    return int((arcs[a0[m], 1] == arcs[a1[m], 0]).sum()), int(m.sum())
