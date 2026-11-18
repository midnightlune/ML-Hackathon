"""Round 2 / Phase 2: compute v2 features for the Round 1 candidate sets (label-free), per trip in
parallel. Writes data/processed/r2/<part>_feats_v2.parquet (keys + v2 columns only). Cached: skips a
partition if its output is newer than the inputs (use --force to rebuild).

Run: python scripts\\r2_build_features_v2.py [--force]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import time
import traceback

import pandas as pd
from joblib import Parallel, delayed

from pathsnap.features_v2 import ALL_V2, build_v2
from pathsnap.geo import project_trip
from pathsnap.io import PARTITIONS, RAW, ROOT, load_trip

PROC = ROOT / "data" / "processed"
CAND_COLS = ["trip_id", "point_idx", "arc_id", "perp_dist", "proj_fraction", "dist_rank", "heading_cos"]


def one_trip(part: str, tid: str, cand: pd.DataFrame):
    try:
        pt = project_trip(load_trip(RAW / part / tid))
        v2 = build_v2(pt, cand.reset_index(drop=True))
        return pd.concat([cand[["trip_id", "point_idx", "arc_id"]].reset_index(drop=True), v2], axis=1), None
    except Exception:  # noqa: BLE001
        return None, f"{part}/{tid}:\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    (PROC / "r2").mkdir(parents=True, exist_ok=True)
    for part in PARTITIONS:
        src, dst = PROC / f"{part}_feats.parquet", PROC / "r2" / f"{part}_feats_v2.parquet"
        if dst.exists() and not args.force and dst.stat().st_mtime > src.stat().st_mtime:
            print(f"{part}: cached ({dst.name})"); continue
        t0 = time.time()
        cand = pd.read_parquet(src, columns=CAND_COLS)
        res = Parallel(n_jobs=-1, backend="loky")(delayed(one_trip)(part, tid, g) for tid, g in cand.groupby("trip_id"))
        fails = [f for _, f in res if f]
        for f in fails:
            print("FAILED", f)
        out = pd.concat([r for r, _ in res if r is not None], ignore_index=True)
        out.to_parquet(dst, index=False)
        nan = out[ALL_V2].isna().mean().sort_values(ascending=False)
        print(f"{part}: {len(out):,} rows, {len(ALL_V2)} v2 features, {len(fails)} failed trips, {time.time() - t0:.0f}s")
        print("  highest NaN share:", nan.head(4).round(4).to_dict())


if __name__ == "__main__":
    main()
