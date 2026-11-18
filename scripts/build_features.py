"""Step 3: candidates + features for train / student_test -> data/processed/<part>_feats.parquet.
Labels are joined AFTER feature construction (candidate/feature code never sees .route).

Run: python -m scripts.build_features
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import logging
import time
import traceback

import numpy as np
import pandas as pd

from pathsnap import candidates, features
from pathsnap.geo import project_trip
from pathsnap.io import PARTITIONS, ROOT, load_trip, trip_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("features")
OUT = ROOT / "data" / "processed"
COLS = ["trip_id", "point_idx", "arc_id", *features.FEATURES, "fallback", "label_arc_id", "unreliable", "label"]


def build_partition(part: str, R: float):
    frames = []
    for d in trip_dirs(part):
        try:
            pt = project_trip(load_trip(d))
            df = features.build(pt, candidates.generate(pt, R=R))
            df.insert(0, "trip_id", pt.trip_id)
            frames.append(df)
        except Exception:
            log.error("trip %s/%s failed:\n%s", part, d.name, traceback.format_exc())
    df = pd.concat(frames, ignore_index=True)
    lab = pd.read_parquet(OUT / f"labels_{part}.parquet")[["trip_id", "point_idx", "label_arc_id", "unreliable"]]
    df = df.merge(lab, on=["trip_id", "point_idx"], how="left", validate="many_to_one")
    df["label"] = (df.arc_id == df.label_arc_id).astype(np.int8)
    return df, lab


def miss_rate(df, lab):
    rel = lab[~lab.unreliable]
    hit = df.loc[df.label == 1, ["trip_id", "point_idx"]].drop_duplicates()
    m = rel.merge(hit, on=["trip_id", "point_idx"], how="left", indicator=True)
    k = int((m._merge == "left_only").sum())
    return k / len(rel), k, len(rel)


def main():
    t0 = time.time()
    info = {}
    for R in (candidates.R_DEFAULT, 75.0):
        res = {p: build_partition(p, R) for p in PARTITIONS}
        rates = {p: miss_rate(*res[p]) for p in PARTITIONS}
        for p, (r, k, n) in rates.items():
            print(f"R={R:.0f} m  {p}: candidate miss rate {r:.2%} ({k}/{n} reliable points)")
        info[f"R{R:.0f}"] = {p: dict(miss_rate=r, missed=k, reliable_points=n) for p, (r, k, n) in rates.items()}
        if max(r for r, _, _ in rates.values()) <= 0.05:
            break
        if R < 75.0:
            print("miss rate > 5% -> raising R to 75 m")
        else:
            print("miss rate still > 5% at R=75 m (the maximum) -> keeping 75 m; remaining misses are points far "
                  "from their label arc (label noise), not a radius problem - see ROUND1_SUMMARY.md")
    info["R_used"], info["K"] = R, candidates.K_DEFAULT
    for p, (df, lab) in res.items():
        df[COLS].to_parquet(OUT / f"{p}_feats.parquet", index=False)
        rel = df[~df.unreliable]
        npts = df.groupby("trip_id").point_idx.nunique().sum()
        print(f"{p}: {len(df):,} rows ({npts:,} points); reliable rows {len(rel):,}, positive rate "
              f"{rel.label.mean():.4f} (~1 in {1 / rel.label.mean():.1f}); fallback rows {df.fallback.mean():.2%}; "
              f"NaN-heading rows {df.heading_cos.isna().mean():.2%}")
        info[p] = dict(rows=len(df), points=int(npts), reliable_rows=len(rel), positive_rate=float(rel.label.mean()))
    (ROOT / "results" / "eda" / "candidates.json").write_text(json.dumps(info, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
