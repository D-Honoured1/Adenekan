"""
Notebook 02 — Full Training, Evaluation & SHAP Pipeline
Run as a script or open in Jupyter with: jupytext --to notebook 02_training.py

Executes the entire pipeline end-to-end:
  1. Preprocessing
  2. Model training (with or without Optuna tuning)
  3. Evaluation (all metrics + SHAP timing)
  4. SHAP global + local explanations for best model
  5. Summary table
"""

# %%
import sys
from pathlib import Path

ROOT = Path().resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import json
import time

import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## 1. Preprocessing

# %%
from preprocess import preprocess

X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess(
    path=str(ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"),
    smote=True,
    save_artifacts=True,
    models_dir=str(ROOT / "models"),
)
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# %% [markdown]
# ## 2. Model Training
#
# Set `tune=False` to skip Optuna and use default hyperparameters (faster for dev).

# %%
from train import main as run_training
import numpy as np

# Change tune=True for full Optuna search (recommended for final results)
lgbm, catboost, histgb = run_training(n_trials=10, tune=False)

# %% [markdown]
# ## 3. Evaluation

# %%
from evaluate import evaluate_all

results = evaluate_all(models_dir=str(ROOT / "models"), time_shap=True)

# Summary table
rows = []
for name, m in results.items():
    if name == "best_model":
        continue
    rows.append({"Model": name.upper(), **m})

df_results = pd.DataFrame(rows).set_index("Model")
print("\n=== Test-Set Results ===")
print(df_results.to_string())

# %% [markdown]
# ## 4. SHAP Explanations for Best Model

# %%
from explain import explain

best_model = results.get("best_model", "lgbm")
print(f"\nGenerating SHAP explanations for: {best_model}")

explainer, explainer_type, shap_vals, feat_names = explain(
    model_name=best_model,
    n_global_samples=300,
)

# %% [markdown]
# ## 5. Plain-Language Explanation Demo

# %%
from templates import build_explanation
import joblib

model = joblib.load(ROOT / "models" / f"{best_model}.pkl")
X_test_arr = np.load(ROOT / "models" / "X_test.npy")
y_test_arr  = np.load(ROOT / "models" / "y_test.npy")

# Pick first phishing instance from test set
phish_idx = int(np.where(y_test_arr == 1)[0][0])
instance  = X_test_arr[[phish_idx]]

prob  = float(model.predict_proba(instance)[0, 1])
pred  = int(prob >= 0.5)

from explain import compute_shap_values, build_explainer
X_bg = np.load(ROOT / "models" / "X_background.npy")
exp, exp_type = build_explainer(best_model, model, X_bg)
shap_single = compute_shap_values(exp, exp_type, instance)[0]
shap_dict   = dict(zip(feat_names, shap_single.tolist()))
feat_vals   = dict(zip(feat_names, instance[0].tolist()))

explanation = build_explanation(shap_dict, feat_vals, pred, prob)
print("\n" + "=" * 60)
print(explanation["full_text"])
print("=" * 60)

# %% [markdown]
# ## 6. Results Summary

# %%
print("\n=== Final Summary ===")
print(f"Best model: {best_model.upper()}")
print(df_results.to_string())
print(f"\nSHAP plots saved to: {ROOT / 'models' / 'shap_plots'}")
