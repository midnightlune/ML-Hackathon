"""Error taxonomy for point-level predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["correct", "candidate_miss", "reverse_twin", "adjacent_arc", "other"]


def classify(pred: pd.DataFrame, arcs_by_trip: dict[str, np.ndarray], label_in_cands: pd.Series) -> pd.Series:
    """pred: argmax_per_point output. label_in_cands: bool per row of pred (label arc among candidates).
    Categories (first match wins): correct; candidate_miss (label arc not in the candidate set);
    reverse_twin (pred = label reversed); adjacent_arc (pred shares a node with the label arc);
    other."""
    cat = np.full(len(pred), "other", dtype=object)
    for tid, idx in pred.groupby("trip_id").indices.items():
        arcs = arcs_by_trip[tid]
        p = pred.pred_arc.to_numpy()[idx]
        t = pred.label_arc_id.to_numpy()[idx]
        pu, pv = arcs[p, 0], arcs[p, 1]
        tu, tv = arcs[t, 0], arcs[t, 1]
        c = np.full(len(idx), "other", dtype=object)
        c[(pu == tv) & (pv == tu)] = "reverse_twin"
        adj = ((pu == tu) | (pu == tv) | (pv == tu) | (pv == tv)) & (c == "other")
        c[adj] = "adjacent_arc"
        c[~label_in_cands.to_numpy()[idx]] = "candidate_miss"
        c[pred.correct.to_numpy()[idx]] = "correct"
        cat[idx] = c
    return pd.Series(cat, index=pred.index, name="category")
