"""
Model training with Optuna hyperparameter tuning.

Models
------
  1. LightGBM              – native SHAP TreeExplainer support ✓
  2. CatBoost              – native SHAP TreeExplainer support ✓
  3. HistGradientBoosting  – ⚠ TreeExplainer NOT supported (see explain.py)

Usage
-----
    python src/train.py
    python src/train.py --trials 30 --no-tune   # skip tuning, use defaults
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from preprocess import preprocess

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = "models"
RANDOM_STATE = 42


# ── default hyper-parameters (used when --no-tune is set) ────────────────────
DEFAULTS = {
    "lgbm": dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
    "catboost": dict(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=0,
        eval_metric="AUC",
    ),
    "histgb": dict(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    ),
}


# ── Optuna objective functions ────────────────────────────────────────────────
def _objective_lgbm(trial, X_train, y_train, X_val, y_val):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 127),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")],
    )
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def _objective_catboost(trial, X_train, y_train, X_val, y_val):
    params = dict(
        iterations=trial.suggest_int("iterations", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        random_seed=RANDOM_STATE,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=30,
    )
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def _objective_histgb(trial, X_train, y_train, X_val, y_val):
    params = dict(
        max_iter=trial.suggest_int("max_iter", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 50),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        random_state=RANDOM_STATE,
    )
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


# ── training functions ────────────────────────────────────────────────────────
def train_lgbm(X_train, y_train, X_val, y_val, n_trials: int = 20) -> LGBMClassifier:
    print(f"\n[LightGBM] Optuna tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda t: _objective_lgbm(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[LightGBM] Best val AUC: {study.best_value:.4f} | params: {best}")
    model = LGBMClassifier(**best, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    return model


def train_catboost(X_train, y_train, X_val, y_val, n_trials: int = 20) -> CatBoostClassifier:
    print(f"\n[CatBoost] Optuna tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _objective_catboost(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[CatBoost] Best val AUC: {study.best_value:.4f} | params: {best}")
    model = CatBoostClassifier(
        **best,
        random_seed=RANDOM_STATE,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def train_histgb(X_train, y_train, X_val, y_val, n_trials: int = 20) -> HistGradientBoostingClassifier:
    print(f"\n[HistGB] Optuna tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _objective_histgb(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[HistGB] Best val AUC: {study.best_value:.4f} | params: {best}")
    model = HistGradientBoostingClassifier(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_with_defaults(X_train, y_train, X_val, y_val):
    """Train all three models with default hyperparameters (no Optuna)."""
    print("\n[LightGBM] Training with defaults...")
    lgbm = LGBMClassifier(**DEFAULTS["lgbm"])
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    print("[CatBoost] Training with defaults...")
    cat = CatBoostClassifier(**DEFAULTS["catboost"])
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    print("[HistGB] Training with defaults...")
    histgb = HistGradientBoostingClassifier(**DEFAULTS["histgb"])
    histgb.fit(X_train, y_train)

    return lgbm, cat, histgb


# ── main ──────────────────────────────────────────────────────────────────────
def main(n_trials: int = 20, tune: bool = True):
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess()

    if tune:
        lgbm  = train_lgbm(X_train, y_train, X_val, y_val, n_trials)
        cat   = train_catboost(X_train, y_train, X_val, y_val, n_trials)
        histgb = train_histgb(X_train, y_train, X_val, y_val, n_trials)
    else:
        lgbm, cat, histgb = train_with_defaults(X_train, y_train, X_val, y_val)

    # ── save models ──────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lgbm,   f"{MODELS_DIR}/lgbm.pkl")
    joblib.dump(cat,    f"{MODELS_DIR}/catboost.pkl")
    joblib.dump(histgb, f"{MODELS_DIR}/histgb.pkl")

    # ── quick validation AUC summary ─────────────────────────────────────────
    results = {}
    for name, model in [("lgbm", lgbm), ("catboost", cat), ("histgb", histgb)]:
        proba = model.predict_proba(X_val)[:, 1]
        pred  = model.predict(X_val)
        results[name] = {
            "val_auc": round(roc_auc_score(y_val, proba), 4),
            "val_f1":  round(f1_score(y_val, pred), 4),
        }
        print(f"[{name:10s}] val AUC={results[name]['val_auc']}  F1={results[name]['val_f1']}")

    with open(f"{MODELS_DIR}/val_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── persist test split for evaluate.py ───────────────────────────────────
    np.save(f"{MODELS_DIR}/X_test.npy", X_test)
    np.save(f"{MODELS_DIR}/y_test.npy", y_test)
    np.save(f"{MODELS_DIR}/X_val.npy",  X_val)
    np.save(f"{MODELS_DIR}/y_val.npy",  y_val)
    # Save a background sample for SHAP (500 rows from train)
    idx = np.random.default_rng(42).choice(len(X_train), size=500, replace=False)
    np.save(f"{MODELS_DIR}/X_background.npy", X_train[idx])

    print(f"\n[done] Models and test splits saved to {MODELS_DIR}/")
    return lgbm, cat, histgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per model")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna, use defaults")
    args = parser.parse_args()
    main(n_trials=args.trials, tune=not args.no_tune)