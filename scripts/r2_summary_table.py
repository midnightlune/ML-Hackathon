"""Write results/r2/summary_table.md from the Round 2 result files (no hand-typed numbers).
Run: python scripts\\r2_summary_table.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json

import pandas as pd

from pathsnap.io import ROOT

RES = ROOT / "results" / "r2"


def main():
    mc = pd.read_csv(RES / "model_comparison_v2_full.csv")
    hm = pd.read_csv(RES / "hmm_comparison.csv")
    sel = json.loads((RES / "hmm_selection.json").read_text())["selected"]
    inf = pd.read_csv(RES / "inference_check.csv")
    L = ["| Model | CV Top-1 (classifier) | ST Top-1 (classifier) | ST macro-F1 | ST log loss | Train-CV gap (pts) |",
         "|---|---|---|---|---|---|"]
    for r in mc.sort_values("cv_top1", ascending=False).itertuples():
        L.append(f"| {r.Model}{' (selected)' if r.Model == sel else ''} | {r.cv_top1:.4f} | {r.st_top1:.4f} | "
                 f"{r.st_macro_f1:.4f} | {r.st_log_loss:.4f} | {r.train_minus_cv_pts:.2f} |")
    L += ["", "| Model | Evaluated on | Top-1 classifier -> HMM (reliable) | Top-1 (all points) | Route F1 | Continuity |",
          "|---|---|---|---|---|---|"]
    for r in hm.itertuples():
        L.append(f"| {r.model} | {r.evaluated_on} | {r.clf_top1_eligible:.4f} -> {r.hmm_top1_eligible:.4f} | "
                 f"{r.clf_top1_all:.4f} -> {r.hmm_top1_all:.4f} | {r.clf_route_f1_mean:.3f} -> {r.hmm_route_f1_mean:.3f} | "
                 f"{r.clf_continuity:.3f} -> {r.hmm_continuity:.3f} |")
    L += ["", f"Inference: largest trip ({int(inf.points.iloc[0]):,} points) {inf.total_s.iloc[0]:.1f} s; "
              f"student_test trips max {inf.total_s.iloc[1:].max():.2f} s.",
          "Sources: results/r2/model_comparison_v2_full.csv, hmm_comparison.csv, inference_check.csv. "
          "'reliable' = points within 50 m of the reference route in trips with label_quality ok."]
    (RES / "summary_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
