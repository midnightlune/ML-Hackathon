
from pathlib import Path
import json

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from scripts import train_models as tm


SEED = 42
N_TRIALS = 10

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Detect GPU; fall back to CPU if unavailable.
try:
    xgb.XGBClassifier(
        n_estimators=2,
        tree_method="hist",
        device="cuda"
    ).fit(np.random.rand(100, 3), np.arange(100) % 2)
    DEVICE = "cuda"
    print("XGBoost device: CUDA")
except Exception:
    DEVICE = "cpu"
    print("XGBoost device: CPU")


def objective(trial):
    # Load only the 80 training trips.
    data = tm.load("train")

    folds = GroupKFold(n_splits=5).split(
        data, groups=data.trip_id
    )

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 700),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.15, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.1, 10.0, log=True
        ),
    }

    scores = []

    for fold, (train_idx, valid_idx) in enumerate(folds):
        train_data = tm.train_rows(data.iloc[train_idx], None)
        valid_data = data.iloc[valid_idx]

        X = train_data[tm.FEATURES]
        y = train_data.label.to_numpy()
        groups = train_data.trip_id.to_numpy()

        # Group-based early-stopping split inside training fold.
        inner_train, inner_valid = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=0.1,
                random_state=SEED
            ).split(X, y, groups)
        )

        model = xgb.XGBClassifier(
            **params,
            tree_method="hist",
            device=DEVICE,
            early_stopping_rounds=30,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1
        )

        model.fit(
            X.iloc[inner_train],
            y[inner_train],
            eval_set=[(
                X.iloc[inner_valid],
                y[inner_valid]
            )],
            verbose=False
        )

        probabilities = tm.predict(
            model, valid_data[tm.FEATURES]
        )

        score = tm.top1(valid_data, probabilities)
        scores.append(score)

        trial.report(float(np.mean(scores)), step=fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def main():
    study = optuna.create_study(
        study_name="pathsnap_xgboost_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage="sqlite:///" + str(RESULTS / "optuna.db"),
        load_if_exists=True
    )

    print(f"Starting {N_TRIALS} Optuna trials...")
    study.optimize(objective, n_trials=N_TRIALS)

    study.trials_dataframe().to_csv(
        RESULTS / "optuna_trials.csv", index=False
    )

    best = {
        "best_cv_top1": study.best_value,
        "best_params": study.best_params,
        "device": DEVICE,
        "seed": SEED,
        "trials": len(study.trials)
    }

    with open(RESULTS / "optuna_best.json", "w") as f:
        json.dump(best, f, indent=4)

    print("\nBest CV Top-1:", round(study.best_value * 100, 2), "%")
    print("Best parameters:")
    print(study.best_params)
    print("\nResults saved in results/optuna_best.json")


if __name__ == "__main__":
    main()
