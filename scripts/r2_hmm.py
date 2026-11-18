"""Round 2 / Phase 5: HMM + Viterbi on top of the classifier.

1. Lattices (network distances for consecutive candidate pairs) built once per trip, cached in
   data/processed/r2/hmm_cache/ (joblib, n_jobs=4). Label-free.
2. Emissions: train = OUT-OF-FOLD probabilities (GroupKFold(5) by trip, shared 36k-point sample per fit);
   student_test = model refit on all eligible train rows. Models: LightGBM (HPO choice) and XGBoost.
3. Decoder params (tau, beta, v_max) tuned with Optuna TPE (seed 42) on TRAIN trips only, objective = HMM
   Top-1 on eligible train points using LightGBM OOF emissions; the same params are used for XGBoost.
4. Report classifier-only vs classifier+HMM: Top-1 (eligible / all), route continuity, route P/R/F1 vs
   .route, breaks, runtime. Per-trip table.
Writes results/r2/{hmm_comparison.csv, hmm_per_trip.csv, hmm_params.json, hmm_trials.csv, hmm_selection.json},
reports/figures/r2/{hmm_comparison.png, hmm_per_trip.png}, artifacts/r2/hmm_config.json
Run: python scripts\\r2_hmm.py
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "4")

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import time
import traceback

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.eval.metrics import point_key  # noqa: E402
from pathsnap.eval.route import continuity, route_prf  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.hmm.transitions import TripLattice, build_lattice  # noqa: E402
from pathsnap.hmm.viterbi import HMMParams, decode  # noqa: E402
from pathsnap.io import RAW, ROOT, load_trip  # noqa: E402
from pathsnap.labels.align import load_route  # noqa: E402  (evaluation only)
from pathsnap.models import zoo  # noqa: E402
from pathsnap.models.sampling import eligible, sample_points  # noqa: E402
from scripts.r2_common import PROC, load_feats_v2  # noqa: E402

RES, FIG, ART = ROOT / "results" / "r2", ROOT / "reports" / "figures" / "r2", ROOT / "artifacts" / "r2"
CACHE = PROC / "r2" / "hmm_cache"
SEED = 42
N_POINTS = json.loads((RES / "train_size_decision.json").read_text())["shared_train_points_per_fit"]
FEATS = json.loads((RES / "feature_set_v2.json").read_text())["features"]
HPO = json.loads((RES / "hpo_best.json").read_text())
LGBM_PARAMS = HPO["best_params"] if HPO["model_used_for_hmm"] == "tuned" else {}
N_TRIALS, TIMEOUT = 25, 300


def lattice_job(part, tid, cand):
    path = CACHE / f"{part}_{tid}.npz"
    if path.exists():
        return tid, None, 0.0
    try:
        t0 = time.time()
        pt = project_trip(load_trip(RAW / part / tid))
        lat = build_lattice(pt, cand.point_idx.to_numpy(), cand.arc_id.to_numpy(), cand.proj_fraction.to_numpy())
        lat.save(path)
        return tid, None, time.time() - t0
    except Exception:  # noqa: BLE001
        return tid, traceback.format_exc(), 0.0


def build_all(part, df):
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = Parallel(n_jobs=4, backend="loky")(delayed(lattice_job)(part, t, g[["point_idx", "arc_id", "proj_fraction"]])
                                             for t, g in df.groupby("trip_id"))
    for t, err, _ in res:
        if err:
            print(f"lattice FAILED {part}/{t}:\n{err}", flush=True)
    print(f"{part}: lattices ready in {time.time() - t0:.0f}s (max per trip {max(s for *_, s in res):.1f}s)", flush=True)


def scores_oof_and_st(name, tr, te):
    fit = (lambda d: zoo.fit_lgbm(d[FEATS], d.label.to_numpy(), d.trip_id.to_numpy(), point_key(d), **LGBM_PARAMS)) \
        if name == "LightGBM" else (lambda d: zoo.fit_xgb(d[FEATS], d.label.to_numpy(), d.trip_id.to_numpy(), point_key(d)))
    oof = np.zeros(len(tr))
    for itr, iev in GroupKFold(n_splits=5).split(tr, groups=tr.trip_id):
        f = tr.iloc[itr]
        oof[iev] = fit(f[sample_points(f, N_POINTS)]).score(tr.iloc[iev][FEATS])
    if name == "LightGBM":   # the HPO script saved the refit-on-all-eligible model
        bst = lgb.Booster(model_file=str(ART / f"lgbm_{HPO['model_used_for_hmm']}.txt"))
        st = bst.predict(te[FEATS])
    else:
        st = fit(tr[eligible(tr)]).score(te[FEATS])
    return oof, st


class TripData:
    def __init__(self, part, tid, g, score):
        o = np.lexsort((g.arc_id.to_numpy(), g.point_idx.to_numpy()))
        self.lat = TripLattice.load(CACHE / f"{part}_{tid}.npz")
        self.score = score[o]
        gg = g.iloc[o]
        self.label = gg.label.to_numpy().astype(bool)
        pts = self.lat.point_idx
        first = np.r_[0, self.lat.cand_off[1:-1]]
        self.elig = (~gg.unreliable.to_numpy() & (gg.label_quality.to_numpy() == "ok"))[first]
        self.label_arc = gg.label_arc_id.to_numpy()[first]
        self.same_next = self.lat.pair_off >= 0
        self.part, self.tid, self.pts = part, tid, pts

    def classifier_only(self):
        off = self.lat.cand_off
        return np.array([self.lat.cand_arc[off[t] + np.argmax(self.score[off[t]:off[t + 1]])] for t in range(len(self.pts))])


def evaluate(trips, prm, arcs_cache, routes, with_routes=True):
    rows = []
    for td in trips:
        t0 = time.time()
        hmm, breaks = decode(td.lat, td.score, prm); dec = time.time() - t0
        clf = td.classifier_only()
        arcs = arcs_cache[(td.part, td.tid)]
        r = dict(partition=td.part, trip_id=td.tid, points=len(td.pts), eligible_points=int(td.elig.sum()),
                 breaks=breaks, decode_s=dec)
        for nm, pred in (("clf", clf), ("hmm", hmm)):
            ok = pred == td.label_arc
            r[f"{nm}_correct_eligible"] = int(ok[td.elig].sum()); r[f"{nm}_correct_all"] = int(ok.sum())
            c, n = continuity(pred, arcs, td.same_next); r[f"{nm}_cont_ok"] = c; r[f"{nm}_cont_n"] = n
            if with_routes:
                for k, v in route_prf(pred, routes[(td.part, td.tid)]).items():
                    r[f"{nm}_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(df):
    out = {}
    for nm in ("clf", "hmm"):
        out[f"{nm}_top1_eligible"] = df[f"{nm}_correct_eligible"].sum() / df.eligible_points.sum()
        out[f"{nm}_top1_all"] = df[f"{nm}_correct_all"].sum() / df.points.sum()
        out[f"{nm}_continuity"] = df[f"{nm}_cont_ok"].sum() / max(df[f"{nm}_cont_n"].sum(), 1)
        if f"{nm}_route_f1" in df:
            for k in ("route_precision", "route_recall", "route_f1"):
                out[f"{nm}_{k}_mean"] = df[f"{nm}_{k}"].mean()
    out["breaks"] = int(df.breaks.sum()); out["decode_s_mean_per_trip"] = df.decode_s.mean()
    return out


def main():
    for d in (RES, FIG, ART):
        d.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    tr, te = load_feats_v2("train"), load_feats_v2("student_test")
    build_all("train", tr); build_all("student_test", te)
    try:
        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(np.random.rand(50, 2), np.arange(50) % 2)
    except Exception:  # noqa: BLE001
        zoo.set_xgb_device("cpu")
    arcs_cache = {(p, t): np.loadtxt(RAW / p / t / f"{t}.arcs", dtype=np.int64, ndmin=2)
                  for p, d in (("train", tr), ("student_test", te)) for t in d.trip_id.unique()}
    routes = {(p, t): load_route(RAW / p / t) for p, d in (("train", tr), ("student_test", te)) for t in d.trip_id.unique()}

    scores = {}
    for name in ("LightGBM", "XGBoost"):   # sequential: CPU model, then GPU model
        t0 = time.time(); scores[name] = scores_oof_and_st(name, tr, te)
        print(f"{name}: OOF + student_test scores in {time.time() - t0:.0f}s", flush=True)

    def trips_for(part, df, s):
        return [TripData(part, t, g, s[g.index.to_numpy()]) for t, g in df.groupby("trip_id")]

    tr_trips = {n: trips_for("train", tr, scores[n][0]) for n in scores}
    te_trips = {n: trips_for("student_test", te, scores[n][1]) for n in scores}

    # decoder tuning on TRAIN only (LightGBM OOF emissions)
    def objective(trial):
        prm = HMMParams(tau=trial.suggest_float("tau", 0.3, 3.0, log=True), beta=trial.suggest_float("beta", 1.0, 100.0, log=True),
                        v_max=trial.suggest_float("v_max", 20.0, 80.0))
        s = summarise(evaluate(tr_trips["LightGBM"], prm, arcs_cache, routes, with_routes=False))
        print(f"  hmm trial {trial.number}: {trial.params} -> train HMM top1 {s['hmm_top1_eligible']:.4f} [{time.time() - t_all:.0f}s]", flush=True)
        return s["hmm_top1_eligible"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"tau": 1.0, "beta": 10.0, "v_max": 60.0})
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    study.trials_dataframe().to_csv(RES / "hmm_trials.csv", index=False)
    prm = HMMParams(**study.best_params)
    cfg = dict(tau=prm.tau, beta=prm.beta, v_max=prm.v_max, gap_threshold_s=30.0, subgraph_m=400.0, cutoff_m=2000.0,
               back_eps_m=15.0, tuned_on="train trips only (LightGBM out-of-fold emissions)", trials=len(study.trials),
               best_train_hmm_top1=study.best_value)
    (RES / "hmm_params.json").write_text(json.dumps(cfg, indent=2)); (ART / "hmm_config.json").write_text(json.dumps(cfg, indent=2))

    comp, per = [], []
    for name in scores:
        for part, trips in (("train (OOF emissions; decoder tuned here)", tr_trips[name]), ("student_test", te_trips[name])):
            df = evaluate(trips, prm, arcs_cache, routes)
            df["model"] = name; per.append(df)
            comp.append(dict(model=name, evaluated_on=part) | summarise(df))
    C = pd.DataFrame(comp); P = pd.concat(per, ignore_index=True)
    for nm in ("clf", "hmm"):
        P[f"{nm}_top1_eligible"] = P[f"{nm}_correct_eligible"] / P.eligible_points.where(P.eligible_points > 0)
        P[f"{nm}_top1_all"] = P[f"{nm}_correct_all"] / P.points
    P["hmm_minus_clf_eligible"] = P.hmm_top1_eligible - P.clf_top1_eligible
    C.to_csv(RES / "hmm_comparison.csv", index=False); P.to_csv(RES / "hmm_per_trip.csv", index=False)
    trc = C[C.evaluated_on.str.startswith("train")].set_index("model").hmm_top1_eligible
    sel = trc.idxmax()
    (RES / "hmm_selection.json").write_text(json.dumps(dict(
        rule="Phase 3 contenders within 0.005 CV Top-1 -> choose by train (out-of-fold) Top-1 after HMM",
        deviation="RandomForest not decoded (time); decoder params tuned on LightGBM OOF and shared with XGBoost",
        train_hmm_top1=trc.round(5).to_dict(), selected=sel), indent=2))

    # figures
    st = C[C.evaluated_on == "student_test"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (k, lab) in zip(axs, (("top1_eligible", "Top-1 (eligible points)"), ("top1_all", "Top-1 (all points)"), ("continuity", "route continuity"))):
        x = np.arange(len(st)); w = 0.38
        b1 = ax.bar(x - w / 2, st[f"clf_{k}"], w, label="classifier only"); b2 = ax.bar(x + w / 2, st[f"hmm_{k}"], w, label="classifier + HMM")
        ax.bar_label(b1, fmt="%.3f", fontsize=7); ax.bar_label(b2, fmt="%.3f", fontsize=7)
        ax.set_xticks(x, st.model); ax.set_ylim(0.5, 1.0); ax.set_title(f"student_test: {lab}")
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "hmm_comparison.png", dpi=150); plt.close(fig)
    q = P[P.model == sel]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        d = q[q.partition == part]; ax.scatter(d.clf_top1_eligible, d.hmm_top1_eligible, s=18, c=c, label=part)
    ax.plot([0.4, 1], [0.4, 1], "k:", lw=1); ax.set_xlabel("classifier-only Top-1"); ax.set_ylabel("classifier + HMM Top-1")
    ax.set_title(f"Per trip ({sel}); above the line = HMM helped"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "hmm_per_trip.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 250)
    print(C.round(4).to_string(index=False)); print("selected:", sel, "| params", cfg)
    print(q.sort_values("hmm_minus_clf_eligible")[["partition", "trip_id", "clf_top1_eligible", "hmm_top1_eligible", "breaks"]].head(5).round(3).to_string(index=False))
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
