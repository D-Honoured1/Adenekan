---
title: PhishGuard XAI
emoji: 🎣
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app/streamlit_app.py
pinned: false
---

# Phishing XAI — SHAP-Based Explainable Phishing Detection

BSc Final Year Project · Redeemer's University
Dataset: PhiUSIIL Phishing URL Dataset (UCI ML Repository) — 235,795 URLs, 50 features

---

## Project Structure

```
phishing-xai/
├── data/                          # Place PhiUSIIL_Phishing_URL_Dataset.csv here
├── models/                        # Saved models, scalers, SHAP artifacts (auto-created)
├── notebooks/
│   ├── 01_eda.py                  # Exploratory Data Analysis
│   └── 02_training.py             # End-to-end pipeline notebook
├── src/
│   ├── preprocess.py              # Data loading, scaling, SMOTE, split
│   ├── train.py                   # Train LightGBM, CatBoost, HistGB + Optuna
│   ├── evaluate.py                # Accuracy, Precision, Recall, F1, AUC-ROC, SHAP timing
│   ├── explain.py                 # SHAP TreeExplainer + plots
│   ├── templates.py               # Plain-language explanation layer
│   └── feature_extractor.py      # URL lexical feature extraction (inference)
├── api/
│   └── main.py                    # FastAPI: /predict + /predict/url endpoints
├── app/
│   └── streamlit_app.py           # Streamlit frontend
└── requirements.txt
```

---

## ⚠ HistGradientBoosting + SHAP Compatibility Warning

`shap.TreeExplainer` **does not support** `sklearn.ensemble.HistGradientBoostingClassifier`.

| Model | SHAP Explainer | Speed | Notes |
|---|---|---|---|
| LightGBM | `TreeExplainer` | Fast (~1–5 ms/instance) | Full support ✓ |
| CatBoost | `TreeExplainer` | Fast (~1–5 ms/instance) | Full support ✓ |
| HistGradientBoosting | `PermutationExplainer` | Slow (50–500 ms/instance) | TreeExplainer raises `InvalidModelError` |

**Recommendation**: Select LightGBM or CatBoost as the final model for SHAP integration.
HistGB is included for benchmark comparison only. If HistGB wins on AUC-ROC + F1, consider
wrapping it in a `PermutationExplainer` and accepting the latency trade-off.

---

## Setup

```bash
# 1. Clone / navigate to project
cd phishing-xai

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
#    Download PhiUSIIL_Phishing_URL_Dataset.csv from UCI ML Repository
#    and put it in data/
```

---

## Running the Pipeline

All commands should be run from the `phishing-xai/` root directory.

### Step 1 — Preprocess only (optional check)
```bash
cd src && python preprocess.py
```

### Step 2 — Train models
```bash
# With Optuna hyperparameter tuning (20 trials per model, ~30–60 min)
cd src && python train.py --trials 20

# Without tuning (uses defaults, fast — good for development)
cd src && python train.py --no-tune
```

### Step 3 — Evaluate all models
```bash
cd src && python evaluate.py
# Outputs test_results.json to models/ and selects best model
```

### Step 4 — Generate SHAP explanations
```bash
# Best model (auto-selected from test_results.json)
cd src && python explain.py --best

# Specific model
cd src && python explain.py --model lgbm
cd src && python explain.py --model catboost
cd src && python explain.py --model histgb   # will use PermutationExplainer
```

### Step 5 — Start API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Step 6 — Start Streamlit frontend
```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

---

## Preprocessing Pipeline

| Step | Detail |
|---|---|
| Drop columns | `FILENAME`, `URL`, `Domain`, `TLD`, `Title` (raw strings) |
| Retained features | 50 numerical features |
| Imputation | Mean imputation (dataset has no missing values; kept for robustness) |
| Scaling | Min-Max scaling to [0, 1] |
| Split | Stratified 70 / 15 / 15 (train / val / test), `random_state=42` |
| SMOTE | Applied only if minority class < 40% of training set |

Dataset class distribution: 57.2% phishing / 42.8% legitimate → SMOTE not triggered.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score (macro)
- AUC-ROC
- SHAP explanation time per instance (target: < 100 ms)

Best model selected by: `argmax(AUC-ROC + F1)`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness + model status |
| GET | `/model-info` | Feature names, metrics, explainer type |
| POST | `/predict` | Predict from pre-extracted feature dict |
| POST | `/predict/url` | Extract lexical features from raw URL, then predict |

Example request:
```bash
curl -X POST http://localhost:8000/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-secure.verify-account.net/login"}'
```

---

## Notebooks

The `.py` notebooks can be run as scripts or converted to Jupyter format:

```bash
pip install jupytext
jupytext --to notebook notebooks/01_eda.py
jupytext --to notebook notebooks/02_training.py
```

---

## Limitations

1. Content-based features (page HTML, JavaScript, iframes, etc.) require fetching the URL.
   The `/predict/url` endpoint sets these to training-set medians for demo purposes.
2. HistGB SHAP is model-agnostic (PermutationExplainer) and will exceed the 100 ms target.
3. The `URLSimilarityIndex` feature is approximated in the feature extractor — the
   dataset's original value was computed against a reference corpus not available here.
