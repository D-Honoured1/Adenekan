"""
Evaluation module — computes all metrics on the held-out test set.

Metrics reported per model
--------------------------
  Accuracy, Precision, Recall, F1-score, AUC-ROC,
  SHAP explanation generation time per instance (target < 100 ms)

Models evaluated
----------------
  Tuned  : lgbm, catboost, histgb
  Baseline: lr, rf, xgb, attention_mlp

Usage
-----
    python src/evaluate.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from attention_mlp import AttentionMLPClassifier  # noqa: F401 — required for joblib unpickling

MODELS_DIR = "models"

# Maps model name → pkl filename (all seven models)
_MODEL_FILES = {
    "lgbm":          "lgbm.pkl",
    "catboost":      "catboost.pkl",
    "histgb":        "histgb.pkl",
    "lr":            "lr.pkl",
    "rf":            "rf.pkl",
    "xgb":           "xgb.pkl",
    "attention_mlp": "attention_mlp.pkl",
}

# SHAP strategy per model
#   tree   → shap.TreeExplainer      (fast, exact)
#   linear → shap.LinearExplainer    (fast, exact)
#   perm   → shap.PermutationExplainer (slow, model-agnostic)
_SHAP_STRATEGY = {
    "lgbm":          "tree",
    "catboost":      "tree",
    "histgb":        "perm",   # TreeExplainer unsupported for HistGB
    "lr":            "linear",
    "rf":            "tree",
    "xgb":           "perm",   # shap TreeExplainer incompatible with XGBoost ≥ 2.1 base_score format
    "attention_mlp": "perm",   # custom PyTorch model → model-agnostic fallback
}


def _shap_timing(model_name: str, model, X_test: np.ndarray, n_samples: int = 100) -> float:
    """Return mean SHAP explanation time per instance (ms)."""
    import shap

    X_bg   = np.load(f"{MODELS_DIR}/X_background.npy")
    subset = X_test[:n_samples]
    strategy = _SHAP_STRATEGY.get(model_name, "perm")

    if strategy == "linear":
        explainer = shap.LinearExplainer(model, X_bg)
        start = time.perf_counter()
        explainer.shap_values(subset)

    elif strategy == "perm":
        explainer = shap.PermutationExplainer(model.predict_proba, X_bg[:50])
        start = time.perf_counter()
        explainer(subset)

    else:  # "tree"
        explainer = shap.TreeExplainer(model)
        start = time.perf_counter()
        explainer.shap_values(subset)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return round(elapsed_ms / n_samples, 2)


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return all evaluation metrics for one model."""
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    return {
        "accuracy":  round(accuracy_score(y_test, pred),                  4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, pred, zero_division=0),    4),
        "f1":        round(f1_score(y_test, pred, zero_division=0),        4),
        "auc_roc":   round(roc_auc_score(y_test, proba),                   4),
    }


def evaluate_all(models_dir: str = MODELS_DIR, time_shap: bool = True) -> dict:
    X_test = np.load(f"{models_dir}/X_test.npy")
    y_test = np.load(f"{models_dir}/y_test.npy")

    # Load whichever models exist on disk; gracefully skip missing ones
    models: dict = {}
    for name, fname in _MODEL_FILES.items():
        path = f"{models_dir}/{fname}"
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"[skip] {fname} not found — run train.py / train_baselines.py first")

    results: dict = {}
    for name, model in models.items():
        print(f"\n[{name}] evaluating on test set ({len(X_test):,} samples)...")
        m = evaluate_model(name, model, X_test, y_test)

        if time_shap:
            print(f"[{name}] timing SHAP (100 samples)...")
            m["shap_ms_per_instance"] = _shap_timing(name, model, X_test)

        results[name] = m

        print(f"  Accuracy : {m['accuracy']}")
        print(f"  Precision: {m['precision']}")
        print(f"  Recall   : {m['recall']}")
        print(f"  F1       : {m['f1']}")
        print(f"  AUC-ROC  : {m['auc_roc']}")
        if time_shap:
            flag = " ⚠ exceeds 100ms target" if m["shap_ms_per_instance"] > 100 else " ✓"
            print(f"  SHAP/inst: {m['shap_ms_per_instance']} ms{flag}")

        print("\n  Classification report:")
        pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        print(classification_report(y_test, pred, target_names=["Legitimate", "Phishing"]))

    # Best model by AUC-ROC + F1 combined
    best_name = max(results, key=lambda k: results[k]["auc_roc"] + results[k]["f1"])
    results["best_model"] = best_name
    print(f"\n[best model] '{best_name}'  (highest AUC-ROC + F1)")

    with open(f"{models_dir}/test_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[saved] test_results.json → {models_dir}/")

    return results


if __name__ == "__main__":
    evaluate_all()
