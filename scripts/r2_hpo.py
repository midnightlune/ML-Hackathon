"""Round 2 / Phase 4: Bayesian HPO (Optuna TPE, seed 42) for LightGBM, the provisional pick from Phase 3.

Deadline-scoped: 10 trials, 3-fold GroupKFold by trip on the 80 train trips, MedianPruner (prunes after
each fold), hard timeout 480 s, n_jobs 4. Each trial fits on the shared 36,000-point sample of the
training folds and is scored on eligible held-out points. Objective: mean CV Top-1 (maximise); the
train-fold Top-1 gap is recorded per trial (10% fixed subsample of training-fold points).
Then tuned and untuned are BOTH evaluated on the same 3 folds and BOTH refit on all eligible train rows
-> student_test. The untuned model is kept if the tuned one is not better on CV Top-1.

Writes results/r2/{optuna.db, hpo_trials.csv, hpo_best.json, hpo_tuned_vs_untuned.csv},
reports/figures/r2/{hpo_history.png, hpo_param_importance.png}, artifacts/r2/lgbm_{tuned,untuned}.txt
Run: python scripts\\r2_hpo.py   (~10 min)
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, log_loss  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.eval.metrics import argmax_per_point, point_key  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402
from pathsnap.models import zoo  # noqa: E402
from pathsnap.models.sampling import eligible, sample_points  # noqa: E402
from scripts.r2_common import load_feats_v2  # noqa: E402

RES, FIG, ART = ROOT / "results" / "r2", ROOT / "reports" / "figures" / "r2", ROOT / "artifacts" / "r2"
SEED, EPS, N_TRIALS, TIMEOUT = 42, 1e-6, 10, 480
N_POINTS = json.loads((RES / "train_size_decision.json").read_text())["shared_train_points_per_fit"]
FEATS = json.loads((RES / "feature_set_v2.json").read_text())["features"]
SPACE = {  # justification: standard LightGBM ranges around the untuned config (lr 0.05, 63 leaves)
    "learning_rate": ("float_log", 0.01, 0.3), "num_leaves": ("int_log", 15, 255),
    "min_child_samples": ("int_log", 5, 200), "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0), "reg_lambda": ("float_log", 1e-3, 10.0),
    "reg_alpha": ("float_log", 1e-8, 1.0)}


def top1(df, s):
    return float(argmax_per_point(df, s).correct.mean())


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SPACE.items():
        if kind == "float_log":
            p[k] = trial.suggest_float(k, lo, hi, log=True)
        elif kind == "int_log":
            p[k] = trial.suggest_int(k, lo, hi, log=True)
        else:
            p[k] = trial.suggest_float(k, lo, hi)
    p["n_estimators"] = 1000   # upper bound; early stopping (30 rounds) picks the number of trees
    return p


def cv_eval(tr, folds, params, trial=None):
    tops, tf, f1s, lls = [], [], [], []
    for k, (itr, iev) in enumerate(folds):
        f = tr.iloc[itr]; d = f[sample_points(f, N_POINTS)]
        m = zoo.fit_lgbm(d[FEATS], d.label.to_numpy(), d.trip_id.to_numpy(), point_key(d), **params)
        ev = tr.iloc[iev]; ev = ev[eligible(ev)]
        s = m.score(ev[FEATS])
        tops.append(top1(ev, s)); f1s.append(f1_score(ev.label, s >= 0.5, average="macro"))
        lls.append(log_loss(ev.label, np.clip(s, EPS, 1 - EPS), labels=[0, 1]))
        trf = f[eligible(f) & f.sub10]; tf.append(top1(trf, m.score(trf[FEATS])))
        if trial is not None:
            trial.report(float(np.mean(tops)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return dict(cv_top1=float(np.mean(tops)), cv_top1_std=float(np.std(tops)), cv_macro_f1=float(np.mean(f1s)),
                cv_log_loss=float(np.mean(lls)), train_fold_top1=float(np.mean(tf)),
                train_minus_cv_pts=100 * float(np.mean(tf) - np.mean(tops)))


def refit_eval(tr, te, params, name):
    d = tr[eligible(tr)]
    t0 = time.time()
    m = zoo.fit_lgbm(d[FEATS], d.label.to_numpy(), d.trip_id.to_numpy(), point_key(d), **params)
    fit_s = time.time() - t0
    t0 = time.time(); s = m.score(te[FEATS]); inf = time.time() - t0
    el = eligible(te).to_numpy()
    m.m.booster_.save_model(str(ART / f"lgbm_{name}.txt"))
    return dict(st_top1=top1(te[el], s[el]), st_top1_all=top1(te, s),
                st_macro_f1=f1_score(te[el].label, s[el] >= 0.5, average="macro"),
                st_log_loss=log_loss(te[el].label, np.clip(s[el], EPS, 1 - EPS), labels=[0, 1]),
                inference_ms_per_1k_points=inf * 1e6 / len(np.unique(point_key(te))),
                refit_rows=len(d), refit_time_s=fit_s, n_trees=int(m.m.booster_.num_trees()))


def main():
    for d in (RES, FIG, ART):
        d.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    tr, te = load_feats_v2("train"), load_feats_v2("student_test")
    pk = np.unique(point_key(tr))
    keep = np.random.default_rng(SEED + 1).choice(pk, size=len(pk) // 10, replace=False)
    tr["sub10"] = np.isin(point_key(tr), keep)
    folds = list(GroupKFold(n_splits=3).split(tr, groups=tr.trip_id))

    db = RES / "optuna.db"
    if db.exists():
        db.unlink()   # fresh study each run (reproducible from seed)
    study = optuna.create_study(study_name="lgbm_r2", direction="maximize", storage=f"sqlite:///{db.as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0))
    study.enqueue_trial({k: zoo.LGBM_DEFAULTS[k] for k in ("learning_rate", "num_leaves", "subsample", "colsample_bytree")}
                        | {"min_child_samples": 20, "reg_lambda": 1e-3, "reg_alpha": 1e-8})   # trial 0 ~= untuned config

    def objective(trial):
        r = cv_eval(tr, folds, suggest(trial), trial)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        print(f"trial {trial.number}: CV top1 {r['cv_top1']:.4f} gap {r['train_minus_cv_pts']:.2f} pts [{time.time() - t_all:.0f}s]", flush=True)
        return r["cv_top1"]

    t_hpo = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    hpo_s = time.time() - t_hpo
    trials = study.trials_dataframe()
    trials.to_csv(RES / "hpo_trials.csv", index=False)
    best = dict(study.best_params) | {"n_estimators": 1000}

    # history + importance figures
    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(done.number, done.value, "o", label="trial CV Top-1"); ax.plot(done.number, done.value.cummax(), "-", label="best so far")
    for _, r in trials[trials.state == "PRUNED"].iterrows():
        ax.axvline(r.number, color="0.8", lw=1)
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (3-fold, eligible)"); ax.legend(fontsize=8)
    ax.set_title(f"Optuna TPE on LightGBM: {len(done)} complete, {int((trials.state == 'PRUNED').sum())} pruned (grey), {hpo_s:.0f}s")
    fig.tight_layout(); fig.savefig(FIG / "hpo_history.png", dpi=150); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.4)); ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set_title("Hyperparameter importance (fANOVA, completed trials)"); fig.tight_layout()
        fig.savefig(FIG / "hpo_param_importance.png", dpi=150); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("param importance skipped:", e)

    # tuned vs untuned: same 3 folds, same samples, both refit on all eligible rows
    rows = []
    for name, params in (("untuned", {}), ("tuned", best)):
        r = dict(model=f"LightGBM {name}") | cv_eval(tr, folds, params) | refit_eval(tr, te, params, name)
        rows.append(r); print(r, flush=True)
    T = pd.DataFrame(rows)
    use = "tuned" if T.loc[T.model == "LightGBM tuned", "cv_top1"].iloc[0] > T.loc[T.model == "LightGBM untuned", "cv_top1"].iloc[0] else "untuned"
    T["used_for_hmm"] = T.model == f"LightGBM {use}"
    T.to_csv(RES / "hpo_tuned_vs_untuned.csv", index=False)
    (RES / "hpo_best.json").write_text(json.dumps(dict(
        model="LightGBM", sampler="TPE(seed=42)", pruner="MedianPruner(n_startup_trials=3)", n_trials_requested=N_TRIALS,
        timeout_s=TIMEOUT, trials_complete=int(len(done)), trials_pruned=int((trials.state == "PRUNED").sum()),
        cv="GroupKFold(3) by trip, train only", objective="mean CV Top-1 on eligible points", search_space=SPACE,
        best_params=best, best_cv_top1=study.best_value, hpo_compute_s=hpo_s, model_used_for_hmm=use,
        total_runtime_s=time.time() - t_all), indent=2))
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False)); print("use for HMM:", use, f"| total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
