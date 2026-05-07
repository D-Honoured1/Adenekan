"""
Baseline model training — Logistic Regression, Random Forest, XGBoost, AttentionMLP.

Models
------
  1. Logistic Regression  – sklearn, L2-regularised
  2. Random Forest        – sklearn, 300 trees
  3. XGBoost              – XGBClassifier with early stopping
  4. AttentionMLP         – PyTorch MLP with a sigmoid-gated feature attention
                            layer that gates each input feature before the dense
                            hidden layers; fully sklearn-compatible API.

Usage
-----
    cd phishing-xai
    python src/train_baselines.py
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from attention_mlp import AttentionMLPClassifier
from preprocess import preprocess

MODELS_DIR = "models"
RANDOM_STATE = 42


# ── individual training helpers ───────────────────────────────────────────────

def train_lr(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    print("\n[LR] Training Logistic Regression...")
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    print("\n[RF] Training Random Forest (300 trees)...")
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> XGBClassifier:
    print("\n[XGB] Training XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_attention_mlp(X_train: np.ndarray, y_train: np.ndarray) -> AttentionMLPClassifier:
    print("\n[AttentionMLP] Training MLP with learnable feature attention layer...")
    model = AttentionMLPClassifier(
        hidden_sizes=(256, 128, 64),
        epochs=30,
        batch_size=512,
        lr=1e-3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    top5 = np.argsort(model.feature_weights_)[-5:][::-1]
    print(f"[AttentionMLP] Top-5 attended feature indices: {top5.tolist()}")
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess()

    lr_model   = train_lr(X_train, y_train)
    rf_model   = train_rf(X_train, y_train)
    xgb_model  = train_xgb(X_train, y_train, X_val, y_val)
    amlp_model = train_attention_mlp(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr_model,   f"{MODELS_DIR}/lr.pkl")
    joblib.dump(rf_model,   f"{MODELS_DIR}/rf.pkl")
    joblib.dump(xgb_model,  f"{MODELS_DIR}/xgb.pkl")
    joblib.dump(amlp_model, f"{MODELS_DIR}/attention_mlp.pkl")

    new_results: dict = {}
    for tag, model in [
        ("lr",            lr_model),
        ("rf",            rf_model),
        ("xgb",           xgb_model),
        ("attention_mlp", amlp_model),
    ]:
        proba = model.predict_proba(X_val)[:, 1]
        pred  = model.predict(X_val)
        new_results[tag] = {
            "val_auc": round(roc_auc_score(y_val, proba), 4),
            "val_f1":  round(f1_score(y_val, pred), 4),
        }
        print(f"[{tag:14s}] val AUC={new_results[tag]['val_auc']}  F1={new_results[tag]['val_f1']}")

    # Merge with existing val_results.json so all 7 entries are present
    results_path = f"{MODELS_DIR}/val_results.json"
    if os.path.exists(results_path):
        with open(results_path) as fh:
            existing: dict = json.load(fh)
    else:
        existing = {}

    existing.update(new_results)
    with open(results_path, "w") as fh:
        json.dump(existing, fh, indent=2)

    print(f"\n[done] Baseline models saved to {MODELS_DIR}/")
    print(f"[done] val_results.json now contains {len(existing)} models")


if __name__ == "__main__":
    main()
