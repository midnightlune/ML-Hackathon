"""Run the Round 2 pipeline in order (Round 1's run_all.py is unchanged).

    python scripts\\run_all_r2.py           # full: ~45 min (ablation ~8, comparison ~11, HPO ~7, HMM ~5)
    python scripts\\run_all_r2.py --quick   # data + HMM + artifacts + maps only (~8 min); reuses committed
                                           # feature set, model comparison and HPO results
Needs the Round 1 data steps first: python scripts\\run_all.py --skip-train
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import subprocess
import sys
import time

from pathsnap.io import ROOT

STEPS = [  # (name, module, run in --quick)
    ("labels: frozen 50 m rule + trip quality", "scripts.r2_labels", True),
    ("v2 features", "scripts.r2_build_features_v2", True),
    ("error analysis (Round 1 config)", "scripts.r2_error_analysis", False),
    ("feature-group ablation", "scripts.r2_ablation", False),
    ("model comparison v2 (5 models)", "scripts.r2_model_comparison", False),
    ("HPO (Optuna, LightGBM)", "scripts.r2_hpo", False),
    ("HMM + Viterbi", "scripts.r2_hmm", True),
    ("artifacts + inference check", "scripts.r2_export_artifacts", True),
    ("Folium maps", "scripts.r2_make_maps", True),
    ("summary table", "scripts.r2_summary_table", True),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    for name, mod, quick in STEPS:
        if args.quick and not quick:
            print(f"--- {name}: skipped (--quick, uses committed results)"); continue
        print(f"--- {name}: python -m {mod}", flush=True)
        t0 = time.time()
        if subprocess.run([sys.executable, "-m", mod], cwd=ROOT).returncode != 0:
            sys.exit(f"step failed: {mod}")
        print(f"--- {name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
