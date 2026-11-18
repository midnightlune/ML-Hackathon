"""Run the whole Round 1 pipeline in order (Windows-friendly replacement for a shell script).

    python -m scripts.run_all               # everything (about 12 min; training is about 10)
    python -m scripts.run_all --skip-train  # data steps only (about 2 min), reuse existing model results
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import subprocess
import sys
import time

from pathsnap.io import ROOT

STEPS = [
    ("1. preprocess + inventory + map", "scripts.preprocess"),
    ("2. label alignment", "scripts.make_labels"),
    ("3. candidates + features", "scripts.build_features"),
    ("3b. EDA + feature figures", "scripts.eda_plots"),
    ("4. train + compare models", "scripts.train_models"),
    ("5. summary", "scripts.write_summary"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="skip step 4 (uses existing results/)")
    args = ap.parse_args()
    for name, mod in STEPS:
        if args.skip_train and mod == "scripts.train_models":
            print(f"--- {name}: skipped (--skip-train)")
            continue
        print(f"--- {name}: python -m {mod}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"step failed: {mod} (exit {r.returncode})")
        print(f"--- {name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
