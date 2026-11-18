"""Round 2 / Phase 1.2: choose ONE shared training size that every model can handle.

Times the slowest models (KNN, MLP, RandomForest) on fold 0 of GroupKFold(5) at several shared
training sizes (sampled by points, seed 42). Writes results/r2/train_size_benchmark.csv.
Run: python scripts\\r2_size_benchmark.py   (~5 min)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.features import FEATURES
from pathsnap.io import ROOT
from pathsnap.models.sampling import eligible, sample_points
from scripts.r2_common import load_feats

SEED = 42


def main():
    tr = load_feats("train")
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip_id))
    itr, iev = folds[0]
    ftr, ev = tr.iloc[itr], tr.iloc[iev]
    ev = ev[eligible(ev)]
    ev_sub = ev[ev.point_idx % 5 == 0]  # fixed 20% of points for the KNN predict timing
    rows = []
    for n_pts in (18_000, 36_000):
        t = ftr[sample_points(ftr, n_pts)]
        X, y = t[FEATURES], t.label.to_numpy()
        models = {
            "KNN": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(25, weights="distance", n_jobs=-1)),
            "MLP": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 MLPClassifier((64, 32), early_stopping=True, max_iter=200, random_state=SEED)),
            "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
                200, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        }
        for name, m in models.items():
            t0 = time.time(); m.fit(X, y); fit = time.time() - t0
            t0 = time.time(); m.predict_proba(ev_sub[FEATURES]); pred = time.time() - t0
            pred_all = pred * len(ev) / len(ev_sub)
            rows.append(dict(points=n_pts, rows=len(t), model=name, fit_s=fit, predict_20pct_s=pred,
                             est_predict_all_fold_s=pred_all, est_cv_s=5 * (fit + pred_all)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "r2" / "train_size_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
