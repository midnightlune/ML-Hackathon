"""Server-side evaluator for the API: the ONLY place in the serving path that reads .route.
Its output is aggregate metrics; per-point labels and the route itself never leave this module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..labels.align import align_points, load_route
from ..labels.quality import apply_rules
from .route import continuity, route_prf


def evaluate_trip(trip_dir: Path, pred: dict) -> dict:
    trip_dir = Path(trip_dir)
    route = load_route(trip_dir)
    pt, arcs = pred["proj"], pred["trip"].arcs
    pidx = pred["point_idx"]
    same_next = np.r_[(np.diff(pidx) == 1) & (pt.segment_id[pidx[1:]] == pt.segment_id[pidx[:-1]]), False]
    out = {"has_reference": bool(len(route))}
    for nm, key in (("hmm", "point_arcs"), ("classifier_only", "classifier_only_arcs")):
        c, n = continuity(pred[key], arcs, same_next)
        out[f"{nm}_continuity"] = c / n if n else None
    if not len(route) or route.max() >= len(arcs):
        return out
    pos, dist = align_points(pt.pxy[pidx], pt.a_xy[route], pt.b_xy[route])
    lab, tq = apply_rules(pd.DataFrame({"trip_id": "x", "point_idx": pidx, "align_dist_m": dist}))
    label = route[pos]
    rel = ~lab.unreliable.to_numpy()
    for nm, key in (("hmm", "point_arcs"), ("classifier_only", "classifier_only_arcs")):
        ok = pred[key] == label
        out[f"{nm}_top1_reliable"] = float(ok[rel].mean()) if rel.any() else None
        out[f"{nm}_top1_all"] = float(ok.mean())
        out |= {f"{nm}_{k}": v for k, v in route_prf(pred[key], route).items()}
    out["reliable_points"] = int(rel.sum())
    out["label_quality"] = str(tq.label_quality.iloc[0])
    return out
