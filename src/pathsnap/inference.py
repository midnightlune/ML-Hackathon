"""End-to-end inference for one trip folder: candidates -> features -> classifier -> HMM/Viterbi.

Reads ONLY <trip>.track, <trip>.nodes, <trip>.arcs. Must never import pathsnap.labels or pathsnap.eval
(enforced by tests/test_no_label_import.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import candidates, features
from .features_v2 import build_v2
from .geo import project_trip
from .hmm.transitions import build_lattice
from .hmm.viterbi import HMMParams, decode
from .io import ROOT, load_trip

ART = ROOT / "artifacts" / "r2"


class _XGB:
    """XGBoost booster wrapper with the same predict(DataFrame) -> P(label arc) interface as lightgbm.Booster."""

    def __init__(self, path: Path):
        import xgboost as xgb
        self.b = xgb.Booster()
        self.b.load_model(str(path))
        self.b.set_param({"device": "cpu", "nthread": 4})   # inference on CPU: works on any laptop

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.b.inplace_predict(X)


@dataclass
class Predictor:
    booster: object
    feats: list[str]
    hmm: HMMParams
    R: float
    K: int

    @classmethod
    def load(cls, art_dir: Path = ART) -> "Predictor":
        spec = json.loads((art_dir / "feature_spec.json").read_text())
        h = json.loads((art_dir / "hmm_config.json").read_text())
        path = art_dir / spec["model_file"]
        model = _XGB(path) if path.suffix == ".json" else lgb.Booster(model_file=str(path))
        return cls(booster=model, feats=spec["features"],
                   hmm=HMMParams(tau=h["tau"], beta=h["beta"], v_max=h["v_max"]), R=spec["candidate_radius_m"],
                   K=spec["candidates_per_point"])

    def predict_trip(self, trip_dir: Path) -> dict:
        tm = {}
        t0 = time.perf_counter()
        trip = load_trip(Path(trip_dir)); pt = project_trip(trip); tm["load_project_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        df = features.build(pt, candidates.generate(pt, R=self.R, K=self.K))
        df = pd.concat([df, build_v2(pt, df)], axis=1); tm["candidates_features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        prob = self.booster.predict(df[self.feats]); tm["classifier_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        o = np.lexsort((df.arc_id.to_numpy(), df.point_idx.to_numpy()))
        lat = build_lattice(pt, df.point_idx.to_numpy(), df.arc_id.to_numpy(), df.proj_fraction.to_numpy())
        tm["lattice_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = prob[o]
        hmm_arcs, breaks = decode(lat, p, self.hmm); tm["viterbi_s"] = time.perf_counter() - t0
        off = lat.cand_off
        clf_arcs = np.empty(len(lat.point_idx), np.int64); conf = np.empty(len(lat.point_idx))
        for t in range(len(lat.point_idx)):
            q = p[off[t]:off[t + 1]]
            clf_arcs[t] = lat.cand_arc[off[t] + int(np.argmax(q))]
            j = int(np.flatnonzero(lat.cand_arc[off[t]:off[t + 1]] == hmm_arcs[t])[0])
            conf[t] = q[j] / q.sum()            # renormalised classifier probability of the HMM-chosen arc
        keep = np.r_[True, hmm_arcs[1:] != hmm_arcs[:-1]]
        return dict(point_idx=lat.point_idx, point_arcs=hmm_arcs, classifier_only_arcs=clf_arcs, confidences=conf,
                    route_arcs=hmm_arcs[keep], breaks=breaks, timings=tm, trip=trip, proj=pt)
