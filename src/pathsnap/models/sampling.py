"""Shared training-row selection so every model is trained on exactly the same rows.

Points (not rows) are sampled, so each selected point keeps its full candidate set (needed by
rankers and for a fair per-point comparison).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


def eligible(df: pd.DataFrame) -> pd.Series:
    """Rows usable for training: reliable label in a trip whose label_quality is ok."""
    return (~df.unreliable) & (df.label_quality == "ok")


def sample_points(df: pd.DataFrame, n_points: int | None, seed: int = SEED) -> pd.Series:
    """Boolean row mask selecting all candidate rows of `n_points` random eligible points
    (all eligible points if n_points is None or larger than available)."""
    el = eligible(df)
    keys = df.trip_id.astype(np.int64) * 1_000_000 + df.point_idx
    pk = np.unique(keys[el])
    if n_points is not None and n_points < len(pk):
        pk = np.random.default_rng(seed).choice(pk, size=n_points, replace=False)
    return el & keys.isin(pk)
