"""Round 2 model zoo. Every builder takes (X, y, groups, point_keys) and returns a fitted object with
score(X) -> 1-D array (higher = more likely the label arc). Classifiers return P(label arc)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_JOBS = 4  # thermal limit: never more than 4 parallel workers
XGB_DEVICE = {"device": "cuda"}


def set_xgb_device(dev: str):
    XGB_DEVICE["device"] = dev


class Prob:
    def __init__(self, m):
        self.m = m

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class Rank:
    is_ranker = True

    def __init__(self, m):
        self.m = m

    def score(self, X):
        return self.m.predict(X)


def _holdout(groups, frac=0.1):
    idx = np.arange(len(groups))
    return next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=SEED).split(idx, groups=groups))


def _imp_scale(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def fit_logreg(X, y, g, pk):
    return Prob(_imp_scale(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)).fit(X, y))


def fit_knn(X, y, g, pk):
    return Prob(_imp_scale(KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=N_JOBS)).fit(X, y))


def fit_rf(X, y, g, pk):
    return Prob(make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED)).fit(X, y))


def fit_et(X, y, g, pk):
    return Prob(make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
        n_estimators=300, max_depth=24, min_samples_leaf=3, n_jobs=N_JOBS, random_state=SEED)).fit(X, y))


def fit_mlp(X, y, g, pk):
    return Prob(_imp_scale(MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200,
                                         random_state=SEED)).fit(X, y))


LGBM_DEFAULTS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8)


def fit_lgbm(X, y, g, pk, **params):
    """Untuned config by default; keyword params (e.g. from HPO) override LGBM_DEFAULTS."""
    tr, va = _holdout(g)
    p = LGBM_DEFAULTS | params
    m = lgb.LGBMClassifier(**p, n_jobs=N_JOBS, random_state=SEED, verbose=-1)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
    return Prob(m)


def fit_xgb(X, y, g, pk, **params):
    tr, va = _holdout(g)
    p = dict(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8)
    p.update(params)
    m = xgb.XGBClassifier(**p, tree_method="hist", n_jobs=N_JOBS, device=XGB_DEVICE["device"], early_stopping_rounds=30,
                          eval_metric="logloss", random_state=SEED)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    return Prob(m)


def fit_xgb_ranker(X, y, g, pk):
    """Pairwise ranker, query group = GPS point (rows must be grouped by point)."""
    o = np.argsort(pk, kind="stable")
    X, y, g, pk = X.iloc[o], y[o], g[o], pk[o]
    tr, va = _holdout(g)
    m = xgb.XGBRanker(objective="rank:pairwise", n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS, device=XGB_DEVICE["device"],
                      early_stopping_rounds=30, eval_metric="ndcg@1", random_state=SEED)
    m.fit(X.iloc[tr], y[tr], qid=pk[tr], eval_set=[(X.iloc[va], y[va])], eval_qid=[pk[va]], verbose=False)
    return Rank(m)


@dataclass
class Spec:
    name: str
    family: str
    config: str
    fit: Callable


ZOO = [
    Spec("LogisticRegression", "Linear", "class_weight=balanced, C=1, lbfgs; median impute + scale", fit_logreg),
    Spec("KNN", "Instance-based", "k=25, distance-weighted, euclidean on scaled features", fit_knn),
    Spec("RandomForest", "Ensemble (bagging)", "200 trees, max_depth=20, min_samples_leaf=5", fit_rf),
    Spec("ExtraTrees", "Ensemble (randomised trees)", "300 trees, max_depth=24, min_samples_leaf=3", fit_et),
    Spec("LightGBM", "Boosting", "<=500 rounds, lr 0.05, 63 leaves, early stop 30 (10% group holdout)", fit_lgbm),
    Spec("XGBoost", "Boosting (GPU)", "<=500 rounds, lr 0.05, depth 8, hist, early stop 30 (10% group holdout)", fit_xgb),
    Spec("MLP", "Neural network", "hidden (64,32), relu, adam, early_stopping", fit_mlp),
    Spec("XGBRanker", "Learning-to-rank (GPU)", "rank:pairwise, query = GPS point, <=500 rounds, depth 8, early stop on ndcg@1", fit_xgb_ranker),
]
