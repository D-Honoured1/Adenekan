"""
SHAP explanation module.

⚠  HISTGRADIENTBOOSTING COMPATIBILITY WARNING
──────────────────────────────────────────────
shap.TreeExplainer does NOT support sklearn's HistGradientBoostingClassifier.
Reason: sklearn's HistGB does not expose its internal tree structure in the
format SHAP expects (it uses a Cython-level histogram representation).

Effect: HistGB falls back to shap.PermutationExplainer, which is model-agnostic
but O(n_features × n_background_samples) per call — typically 5–50× slower than
TreeExplainer. This will likely exceed the 100 ms/instance target for 50 features.

Recommendation: Only select HistGB as the final model if it substantially
outperforms LightGBM and CatBoost on AUC-ROC + F1. Prefer LightGBM or CatBoost
for the SHAP integration layer.

Usage
-----
    python src/explain.py --model lgbm          # global + local plots
    python src/explain.py --model catboost
    python src/explain.py --model histgb        # will use PermutationExplainer
    python src/explain.py --best                # auto-select from test_results.json
"""

import argparse
import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

MODELS_DIR = "models"
OUTPUT_DIR = "models/shap_plots"


def _sigmoid(x: float) -> float:
    """Convert a log-odds value to probability."""
    return float(1 / (1 + np.exp(-x)))


# ── explainer factory ─────────────────────────────────────────────────────────
def build_explainer(model_name: str, model, X_background: np.ndarray):
    """
    Return (explainer, explainer_type_str).
    TreeExplainer for LightGBM / CatBoost; PermutationExplainer for HistGB.
    """
    if model_name == "lgbm":
        # LightGBM TreeExplainer only supports model_output="raw" (log-odds).
        # Probability conversion is applied at plot time via sigmoid.
        try:
            explainer = shap.TreeExplainer(
                model,
                X_background,
                feature_perturbation="interventional",
                model_output="raw",
            )
            return explainer, "TreeExplainer(raw)"
        except Exception as e:
            print(f"[warn] TreeExplainer failed ({e}), falling back to Explainer")
            explainer = shap.Explainer(model, X_background)
            return explainer, "Explainer(auto)"

    if model_name == "catboost":
        try:
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="probability",
            )
            return explainer, "TreeExplainer"
        except Exception as e:
            print(f"[warn] TreeExplainer failed ({e}), falling back to Explainer")
            explainer = shap.Explainer(model, X_background)
            return explainer, "Explainer(auto)"

    # HistGB — TreeExplainer is not supported
    print(
        "[warn] HistGradientBoosting: shap.TreeExplainer is not supported. "
        "Using shap.PermutationExplainer. Expect slower SHAP times."
    )
    explainer = shap.PermutationExplainer(
        model.predict_proba, X_background, max_evals=2 * X_background.shape[1] + 1
    )
    return explainer, "PermutationExplainer"


# ── SHAP value computation ────────────────────────────────────────────────────
def compute_shap_values(explainer, explainer_type: str, X: np.ndarray):
    """
    Return raw SHAP values array (n_samples × n_features) for the positive class.
    Handles the different output shapes of TreeExplainer vs PermutationExplainer.
    """
    raw = explainer(X) if "Permutation" in explainer_type else explainer.shap_values(X)

    # TreeExplainer on binary classifiers returns list[neg_class, pos_class]
    if isinstance(raw, list):
        return raw[1]

    # shap.Explanation object (from Explainer / PermutationExplainer)
    if isinstance(raw, shap.Explanation):
        vals = raw.values
        # shape may be (n, features, 2) for predict_proba output
        if vals.ndim == 3:
            return vals[:, :, 1]
        return vals

    return raw


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_global_summary(shap_values: np.ndarray, feature_names: list[str],
                        model_name: str, save: bool = True):
    """Bar chart of mean |SHAP| per feature (global importance)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, feature_names=feature_names,
        plot_type="bar", show=False, max_display=20,
    )
    plt.title(f"Global Feature Importance — {model_name.upper()} (mean |SHAP|)")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{model_name}_global_summary.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.show()
    plt.close()


def plot_local_waterfall(
    explainer,
    explainer_type: str,
    X_instance: np.ndarray,
    feature_names: list[str],
    model_name: str,
    instance_idx: int = 0,
    save: bool = True,
):
    """Waterfall plot for a single prediction (local explanation)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Compute explanation for this single instance
    exp = explainer(X_instance[[instance_idx]])

    if isinstance(exp, shap.Explanation):
        # Handle (1, features, 2) from predict_proba explainers
        if exp.values.ndim == 3:
            sv   = exp.values[0, :, 1]
            base = exp.base_values[0, 1] if exp.base_values.ndim > 1 else exp.base_values[0]
        else:
            sv   = exp.values[0]
            base = exp.base_values[0]
        single_exp = shap.Explanation(
            values=sv,
            base_values=base,
            data=exp.data[0],
            feature_names=feature_names,
        )
    else:
        # TreeExplainer.shap_values returns a list or array
        raw = explainer.shap_values(X_instance[[instance_idx]])
        sv  = raw[1][0] if isinstance(raw, list) else raw[0]
        single_exp = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=X_instance[instance_idx],
            feature_names=feature_names,
        )

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(single_exp, max_display=15, show=False)
    if "raw" in explainer_type:
        pred_prob = _sigmoid(float(single_exp.base_values) + float(single_exp.values.sum()))
        title = (
            f"Local Explanation — {model_name.upper()} | Instance {instance_idx}"
            f"\nSHAP values in log-odds space · predicted P(phishing) = {pred_prob:.4f}"
        )
    else:
        title = f"Local Explanation — {model_name.upper()} | Instance {instance_idx}"
    plt.title(title)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{model_name}_local_waterfall_{instance_idx}.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.show()
    plt.close()

    return single_exp


# ── timing benchmark ──────────────────────────────────────────────────────────
def benchmark_shap_time(explainer, explainer_type: str, X_test: np.ndarray,
                        n_samples: int = 100) -> float:
    """Return mean SHAP explanation time per instance in milliseconds."""
    subset = X_test[:n_samples]
    start  = time.perf_counter()
    compute_shap_values(explainer, explainer_type, subset)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return round(elapsed_ms / n_samples, 2)


# ── main ──────────────────────────────────────────────────────────────────────
def explain(model_name: str, n_global_samples: int = 500):
    X_test       = np.load(f"{MODELS_DIR}/X_test.npy")
    X_background = np.load(f"{MODELS_DIR}/X_background.npy")
    feature_names: list[str] = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    model        = joblib.load(f"{MODELS_DIR}/{model_name}.pkl")

    print(f"\n[explain] Building explainer for '{model_name}'...")
    explainer, explainer_type = build_explainer(model_name, model, X_background)
    print(f"[explain] Using: {explainer_type}")

    # ── timing ───────────────────────────────────────────────────────────────
    ms = benchmark_shap_time(explainer, explainer_type, X_test)
    flag = "⚠  EXCEEDS 100ms target" if ms > 100 else "✓  within target"
    print(f"[timing] {ms} ms/instance — {flag}")

    # ── global explanation ────────────────────────────────────────────────────
    print(f"[explain] Computing global SHAP values ({n_global_samples} test samples)...")
    X_global = X_test[:n_global_samples]
    shap_vals = compute_shap_values(explainer, explainer_type, X_global)
    plot_global_summary(shap_vals, feature_names, model_name)

    # ── local explanation (first test instance) ───────────────────────────────
    print("[explain] Generating local waterfall plot for instance 0...")
    plot_local_waterfall(explainer, explainer_type, X_test, feature_names, model_name, instance_idx=0)

    # ── save explainer ────────────────────────────────────────────────────────
    joblib.dump((explainer, explainer_type), f"{MODELS_DIR}/{model_name}_explainer.pkl")
    print(f"[saved] {model_name}_explainer.pkl")

    return explainer, explainer_type, shap_vals, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgbm",
                        choices=["lgbm", "catboost", "histgb"])
    parser.add_argument("--best", action="store_true",
                        help="Auto-select best model from test_results.json")
    parser.add_argument("--n-global", type=int, default=500,
                        help="Number of test samples for global SHAP")
    args = parser.parse_args()

    if args.best:
        with open(f"{MODELS_DIR}/test_results.json") as f:
            results = json.load(f)
        model_name = results.get("best_model", "lgbm")
        print(f"[auto] Best model from test_results.json: {model_name}")
    else:
        model_name = args.model

    explain(model_name, n_global_samples=args.n_global)
