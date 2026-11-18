"""Round 2 / Phase 6: write the inference artifacts to artifacts/r2/ (model is produced by r2_hpo.py,
HMM config by r2_hmm.py): feature_spec.json, versions.json. Then stress-test predict_trip on the largest
train trip and run every student_test trip end to end -> results/r2/inference_check.csv.
Run: python scripts\\r2_export_artifacts.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import platform
import subprocess
import time
from importlib.metadata import version

import pandas as pd

from pathsnap import candidates
from pathsnap.inference import Predictor
from pathsnap.io import RAW, ROOT, trip_dirs

ART, RES = ROOT / "artifacts" / "r2", ROOT / "results" / "r2"


def train_selected_xgb(feats):
    """Refit the HMM-selected XGBoost on all eligible train rows (same call as scripts/r2_hmm.py) and save it."""
    import numpy as np
    import xgboost as xgb
    from pathsnap.eval.metrics import point_key
    from pathsnap.models import zoo
    from pathsnap.models.sampling import eligible
    from scripts.r2_common import load_feats_v2
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(np.random.rand(50, 2), np.arange(50) % 2)
    except Exception:  # noqa: BLE001
        zoo.set_xgb_device("cpu")
    tr = load_feats_v2("train"); d = tr[eligible(tr)]
    m = zoo.fit_xgb(d[feats], d.label.to_numpy(), d.trip_id.to_numpy(), point_key(d))
    m.m.get_booster().save_model(str(ART / "xgb_selected.json"))
    return len(d)


def main():
    hpo = json.loads((RES / "hpo_best.json").read_text())
    fs = json.loads((RES / "feature_set_v2.json").read_text())
    sel = json.loads((RES / "hmm_selection.json").read_text())["selected"]
    if sel == "XGBoost":
        n = train_selected_xgb(fs["features"])
        model_file, note = "xgb_selected.json", f"XGBoost (Phase 3 config), refit on {n:,} eligible train rows"
    else:
        model_file, note = f"lgbm_{hpo['model_used_for_hmm']}.txt", f"LightGBM ({hpo['model_used_for_hmm']})"
    spec = dict(model_file=model_file, model=sel, description=note, selection="results/r2/hmm_selection.json",
                trained_on="all eligible train rows (reliable points in ok trips, 80 train trips)",
                features=fs["features"], candidate_radius_m=75.0, candidates_per_point=candidates.K_DEFAULT)
    (ART / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    vers = {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "shapely", "pyproj", "scipy")}
    (ART / "versions.json").write_text(json.dumps(dict(python=platform.python_version(), packages=vers, git_commit=commit), indent=2))

    pr = Predictor.load()
    rows = []
    biggest = max(trip_dirs("train"), key=lambda d: (d / f"{d.name}.track").stat().st_size)
    for part, d in [("train (largest)", biggest)] + [("student_test", d) for d in trip_dirs("student_test")]:
        t0 = time.time()
        r = pr.predict_trip(d)
        rows.append(dict(partition=part, trip_id=d.name, points=len(r["point_idx"]), route_arcs=len(r["route_arcs"]),
                         breaks=r["breaks"], total_s=time.time() - t0, **{k: round(v, 2) for k, v in r["timings"].items()}))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "inference_check.csv", index=False)
    print("largest trip under 30 s:", bool(df.total_s.iloc[0] < 30), "| all student_test ran:", len(df) == 11)


if __name__ == "__main__":
    main()
