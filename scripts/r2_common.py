"""Shared loaders for Round 2 scripts (read-only use of Round 1 feature tables)."""
from __future__ import annotations

import pandas as pd

from pathsnap.io import ROOT

PROC = ROOT / "data" / "processed"


def load_feats(part: str, feats_path=None) -> pd.DataFrame:
    """Round 1 candidate/feature table (label-independent) joined with the Round 2 reliability
    flags. Candidate features never read .route; only `label_arc_id`, `label`, `unreliable` and
    `label_quality` come from the labels module."""
    df = pd.read_parquet(feats_path or PROC / f"{part}_feats.parquet").drop(columns=["unreliable"])
    lab = pd.read_parquet(PROC / "r2" / f"labels_{part}.parquet")[["trip_id", "point_idx", "unreliable", "label_quality"]]
    df = df.merge(lab, on=["trip_id", "point_idx"], how="left", validate="many_to_one")
    return df


def load_feats_v2(part: str) -> pd.DataFrame:
    """Round 1 table + Round 2 labels + v2 features (row order identical, checked)."""
    df = load_feats(part)
    v2 = pd.read_parquet(PROC / "r2" / f"{part}_feats_v2.parquet")
    keys = ["trip_id", "point_idx", "arc_id"]
    assert (df[keys].to_numpy() == v2[keys].to_numpy()).all(), "v2 rows not aligned with Round 1 candidates"
    return pd.concat([df, v2.drop(columns=keys)], axis=1)
