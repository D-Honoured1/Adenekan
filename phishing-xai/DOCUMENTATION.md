# Phishing XAI — Detailed Project Documentation

**BSc Final Year Project · Redeemer's University**

---

## 1. Overview

**Phishing XAI** is an end-to-end explainable machine learning system that detects phishing URLs and justifies every prediction in plain English. It trains three gradient-boosted tree classifiers on the **PhiUSIIL Phishing URL Dataset** (235,795 URLs, 50 features) from the UCI ML Repository, selects the best model, attaches a **SHAP (SHapley Additive exPlanations)** layer for interpretability, and exposes the whole pipeline through a **FastAPI** REST backend and a **Streamlit** web frontend.

The core research question is whether tree-based models can achieve near-perfect phishing detection *and* satisfy a hard real-time explainability constraint (< 100 ms per SHAP explanation), a requirement that has practical implications for browser extensions and security tooling.

---

## 2. Project Structure

```
phishing-xai/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv   # 235,795 rows, 56 columns
├── models/                                   # All saved artifacts
│   ├── lgbm.pkl / catboost.pkl / histgb.pkl
│   ├── lgbm_explainer.pkl                   # Cached SHAP explainer
│   ├── imputer.pkl / scaler.pkl             # Preprocessing artifacts
│   ├── feature_names.pkl
│   ├── X_test.npy / y_test.npy             # Held-out test split
│   ├── X_val.npy / y_val.npy
│   ├── X_background.npy                     # 500-sample SHAP background
│   ├── test_results.json                    # Evaluation metrics + best model
│   ├── val_results.json
│   └── shap_plots/
│       ├── lgbm_global_summary.png
│       └── lgbm_local_waterfall_0.png
├── src/
│   ├── preprocess.py        # Data loading, imputation, scaling, SMOTE, split
│   ├── train.py             # Optuna tuning + model training
│   ├── evaluate.py          # All metrics + SHAP timing benchmark
│   ├── explain.py           # SHAP explainer factory + plot generation
│   ├── templates.py         # Plain-English explanation layer
│   └── feature_extractor.py # URL lexical feature extraction (inference)
├── api/
│   └── main.py              # FastAPI application
├── app/
│   └── streamlit_app.py     # Interactive web UI
├── notebooks/
│   ├── 01_eda.py            # Exploratory Data Analysis
│   └── 02_training.py       # Full end-to-end pipeline notebook
└── requirements.txt
```

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | PhiUSIIL Phishing URL Dataset (UCI ML Repository) |
| Total samples | 235,795 URLs |
| Raw columns | 56 |
| Dropped columns | `FILENAME`, `URL`, `Domain`, `TLD`, `Title` (raw strings) |
| Feature columns retained | **50 numerical features** |
| Target label | `label` (1 = phishing, 0 = legitimate) |
| Class distribution | 57.2% phishing / 42.8% legitimate |

The 50 features fall into six groups:

- **URL structure**: `URLLength`, `DomainLength`, `IsDomainIP`, `URLSimilarityIndex`, `NoOfSubDomain`, `IsHTTPS`, `TLDLength`, `TLDLegitimateProb`, `CharContinuationRate`, `URLCharProb`
- **Obfuscation**: `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`
- **Character composition**: `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`
- **Page content**: `LineOfCode`, `LargestLineLength`, `HasTitle`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`
- **Keywords**: `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`
- **Resource counts**: `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`

---

## 4. Preprocessing Pipeline (`src/preprocess.py`)

Executed in strict sequence to prevent data leakage:

1. **Load CSV** with BOM-safe UTF-8 encoding (`utf-8-sig`).
2. **Drop** non-numerical identifier columns (`FILENAME`, `URL`, `Domain`, `TLD`, `Title`).
3. **Stratified 70 / 15 / 15 train/val/test split** using `random_state=42`. The split happens *before* fitting any transformer.
4. **Mean imputation** (`sklearn.SimpleImputer`) — fit on train only, then applied to val and test. The dataset has no missing values, but the imputer is kept in the pipeline to handle missing features at inference time.
5. **Min-Max scaling** to [0, 1] (`sklearn.MinMaxScaler`) — fit on train only.
6. **SMOTE** (`imbalanced-learn`) — applied to the training set *only if* the minority class ratio falls below 40%. Given the 57/43 split, SMOTE is **not triggered**.

**Artifacts saved** to `models/`: `imputer.pkl`, `scaler.pkl`, `feature_names.pkl`, `X_test.npy`, `y_test.npy`, `X_val.npy`, `y_val.npy`, `X_background.npy` (500-row background for SHAP).

**Inference helper** — `transform_single(raw_features: dict)` loads the saved imputer and scaler and applies them to a single feature dict, producing a model-ready `(1, 50)` array.

---

## 5. Model Training (`src/train.py`)

Three gradient-boosted tree classifiers are trained and compared:

| Model | Library | SHAP Compatibility |
|---|---|---|
| **LightGBM** | `lightgbm.LGBMClassifier` | `TreeExplainer` (fast, native) |
| **CatBoost** | `catboost.CatBoostClassifier` | `TreeExplainer` (fast, native) |
| **HistGradientBoosting** | `sklearn.ensemble.HistGradientBoostingClassifier` | `PermutationExplainer` only (slow) |

### Hyperparameter Optimisation (Optuna)

Each model is tuned independently using **Optuna** (`optuna.create_study(direction="maximize")`), maximising **AUC-ROC** on the validation set. Default is 20 trials per model; overridable via `--trials N`.

**LightGBM search space**: `n_estimators` [200–800], `learning_rate` [0.01–0.2, log], `num_leaves` [31–127], `max_depth` [4–10], `min_child_samples` [10–50], `subsample` [0.6–1.0], `colsample_bytree` [0.6–1.0], `reg_alpha`/`reg_lambda` [0.001–10, log]. Uses `LightGBMPruningCallback` for early stopping.

**CatBoost search space**: `iterations` [200–800], `learning_rate` [0.01–0.2, log], `depth` [4–10], `l2_leaf_reg` [1.0–10.0]. Uses `early_stopping_rounds=30` on validation AUC.

**HistGB search space**: `max_iter` [100–500], `learning_rate` [0.01–0.2, log], `max_depth` [3–9], `min_samples_leaf` [10–50], `l2_regularization` [0.001–10, log].

Skipping tuning is possible with `--no-tune`, which uses pre-set defaults and trains in seconds.

**Usage**:
```bash
cd phishing-xai/src
python train.py --trials 20     # with Optuna (~30–60 min)
python train.py --no-tune       # default params (fast)
```

---

## 6. Evaluation (`src/evaluate.py`)

Metrics computed on the **held-out test set** (`X_test.npy`, `y_test.npy`):

| Metric | Description |
|---|---|
| Accuracy | Overall fraction correct |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-score | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve |
| SHAP ms/instance | Mean SHAP computation time over 100 test samples |

**Best model selection**: `argmax(AUC-ROC + F1)` across the three models. The winner is written to `test_results.json` under the key `"best_model"`.

### Actual Test Results (Achieved)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | SHAP ms/inst |
|---|---|---|---|---|---|---|
| **LightGBM** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 7.66 ms ✓ |
| **CatBoost** | 0.9998 | 0.9997 | 1.0000 | 0.9998 | 1.0000 | 1.46 ms ✓ |
| HistGB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3071 ms ✗ |

> **Best model selected: LightGBM** (tied AUC-ROC + F1 = 2.0, tiebreak by declaration order).

HistGradientBoosting achieves perfect accuracy but its `PermutationExplainer` takes **~3 seconds per instance**, failing the 100 ms real-time target by 30×.

---

## 7. Explainability Layer (`src/explain.py`)

### Explainer Factory (`build_explainer`)

Selects the right SHAP explainer for each model:

- **LightGBM** → `shap.TreeExplainer(model, X_background, feature_perturbation="interventional", model_output="raw")`. Outputs SHAP values in **log-odds** space. Probabilities are recovered via `sigmoid(base_value + sum(shap_values))`.
- **CatBoost** → `shap.TreeExplainer(model, feature_perturbation="interventional", model_output="probability")`. Outputs values directly in **probability** space.
- **HistGB** → `shap.PermutationExplainer(model.predict_proba, X_background)`. Model-agnostic, but O(n_features × n_background) per call.

### Outputs

- **Global explanation**: SHAP summary bar chart (`mean |SHAP|` per feature) for 500 test samples — saved to `models/shap_plots/{model}_global_summary.png`.
- **Local explanation**: Waterfall plot for a single instance — saved to `models/shap_plots/{model}_local_waterfall_0.png`.
- **Cached explainer**: Saved to `models/{model}_explainer.pkl` for fast API startup.

---

## 8. Plain-Language Templates (`src/templates.py`)

Converts raw SHAP values into human-readable explanations via `build_explanation(shap_dict, feature_values, prediction, probability)`.

**Process**:
1. Map `probability` to a risk level: `SAFE` (< 40%), `LOW` (40–60%), `MEDIUM` (60–80%), `HIGH` (> 80%).
2. Rank all 50 features by `|SHAP value|`, take the top 5.
3. Separate top features into *pushers* (SHAP > 0, toward phishing) and *pullers* (SHAP < 0, toward legitimate).
4. Generate one sentence per feature using the `FEATURE_DESCRIPTIONS` map (e.g., `IsDomainIP` → "IP address as domain (is active) increases the phishing risk.").

**Returns** a dict with: `risk_level`, `verdict`, `summary`, `top_features`, `detail_sentences`, `full_text`.

---

## 9. URL Feature Extractor (`src/feature_extractor.py`)

Used at **inference time** when a raw URL string is provided (not pre-extracted features). It extracts the 22 features derivable purely from the URL string without any network request:

| Feature | How computed |
|---|---|
| `URLLength` | `len(url)` |
| `DomainLength` | `len(netloc)` |
| `IsDomainIP` | Regex `^\d{1,3}(\.\d{1,3}){3}$` on host |
| `URLSimilarityIndex` | Heuristic: `TLD_score × 100 − url_length × 0.05` |
| `CharContinuationRate` | Fraction of adjacent identical chars |
| `TLDLegitimateProb` | Lookup table (`.com`=0.52, `.gov`=0.70, etc.) |
| `URLCharProb` | Alphanumeric ratio in URL |
| `TLDLength` | `len(tld)` |
| `NoOfSubDomain` | `max(0, host_parts − 2)` |
| `HasObfuscation` | Presence of `%XX` percent-encoded sequences |
| `NoOfObfuscatedChar` | Count of `%XX` occurrences |
| `ObfuscationRatio` | `NoOfObfuscatedChar / URLLength` |
| `NoOfLettersInURL` | Alpha char count |
| `LetterRatioInURL` | Alpha / total |
| `NoOfDegitsInURL` | Digit char count |
| `DegitRatioInURL` | Digit / total |
| `NoOfEqualsInURL` | Count of `=` |
| `NoOfQMarkInURL` | Count of `?` |
| `NoOfAmpersandInURL` | Count of `&` |
| `NoOfOtherSpecialCharsInURL` | Count of chars in the special-char set |
| `SpacialCharRatioInURL` | Special / total |
| `IsHTTPS` | `scheme == "https"` |

The **remaining 28 content-based features** (HTML page metrics, form fields, keywords, etc.) are set to approximate training-set medians (e.g., `LineOfCode=200`, `NoOfImage=5`, `HasFavicon=1`). This is a documented limitation — a production system would use a headless browser (e.g., Playwright) to extract them.

---

## 10. FastAPI Backend (`api/main.py`)

Started with:
```bash
cd phishing-xai
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Startup Behaviour

On launch, the API reads `models/test_results.json`, loads the `best_model` (LightGBM by default), its preprocessing artifacts, and the cached SHAP explainer. If no cached explainer exists, one is built from `X_background.npy`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok", "model_loaded": bool, "model": str}` |
| `GET` | `/model-info` | Returns model name, explainer type, feature names, test metrics |
| `POST` | `/predict` | Accepts `{"features": {name: value}}` → full prediction + SHAP |
| `POST` | `/predict/scaled` | Same but skips the preprocessing step (for demo mode with test-set data) |
| `POST` | `/predict/url` | Accepts `{"url": "..."}` → lexical extraction → predict |

### Response Schema (`PredictionResponse`)

```json
{
  "prediction": 1,
  "probability": 0.9723,
  "risk_level": "HIGH",
  "verdict": "This URL is very likely a phishing site.",
  "summary": "The model classified this URL as PHISHING with 97.2% confidence...",
  "top_features": [{"feature": "IsHTTPS", "shap": -0.1234}, "..."],
  "detail_sentences": ["Factors increasing phishing risk:", "  • ..."],
  "shap_values": {"URLLength": 0.0123, "...": "..."},
  "shap_scale": "log-odds",
  "prediction_caveat": "",
  "model_used": "lgbm",
  "inference_ms": 12.4
}
```

The `shap_scale` field tells the frontend whether values are in log-odds (LightGBM) or probability space (CatBoost), so the UI can apply the correct sigmoid conversion before display.

---

## 11. Streamlit Frontend (`app/streamlit_app.py`)

Started with:
```bash
cd phishing-xai
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### Features

| Component | Description |
|---|---|
| **URL input** | Text field accepting any URL string; sends to `/predict/url` |
| **Demo mode** | Two buttons load a random phishing or legitimate sample from `X_test.npy` and send it to `/predict/scaled` |
| **Probability gauge** | Semicircular Plotly indicator: green (< 40%), amber (40–70%), red (> 70%) |
| **Risk badge** | Colour-coded `SAFE / LOW / MEDIUM / HIGH` label |
| **SHAP waterfall chart** | Horizontal Plotly bar chart; log-odds values are sigmoid-converted to a centred probability scale for readability |
| **Plain-language explanations** | 32 bespoke feature templates, each with a "sus" and "safe" variant, rendered as bullet points with icons |
| **Top features table** | DataFrame of top 5 SHAP contributors with direction labels |
| **Sidebar** | API status indicator; model name, explainer type, feature count; test-set metrics dashboard |

---

## 12. Full Pipeline: Step-by-Step Execution

```bash
cd phishing-xai

# Step 1 — Preprocess (optional standalone check)
cd src && python preprocess.py && cd ..

# Step 2 — Train (with Optuna tuning)
cd src && python train.py --trials 20 && cd ..
# or fast dev mode:
cd src && python train.py --no-tune && cd ..

# Step 3 — Evaluate all three models
cd src && python evaluate.py && cd ..

# Step 4 — Generate SHAP plots for best model
cd src && python explain.py --best && cd ..

# Step 5 — Start API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Step 6 — Start frontend (new terminal)
streamlit run app/streamlit_app.py
```

---

## 13. Dependencies (`requirements.txt`)

| Category | Libraries |
|---|---|
| ML models | `lightgbm`, `catboost`, `scikit-learn`, `imbalanced-learn` |
| Hyperparameter tuning | `optuna` |
| Explainability | `shap`, `matplotlib`, `seaborn` |
| Data | `pandas`, `numpy`, `joblib` |
| API | `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart` |
| Frontend | `streamlit`, `plotly`, `requests` |
| Notebooks | `ipykernel` |

---

## 14. Known Limitations

1. **Content features are approximated at inference time.** The `/predict/url` endpoint can only derive 22 of the 50 features from the URL string. The remaining 28 page-content features (HTML line count, favicons, form fields, etc.) are set to training-set medians. A production system would need a headless browser (e.g., Playwright) to extract them.

2. **HistGB SHAP is too slow for real-time use.** It achieves perfect accuracy but takes ~3,071 ms/instance — 30× over the 100 ms target. It is included for benchmark comparison only.

3. **`URLSimilarityIndex` is approximated.** The original dataset value was computed against a reference corpus of known-good URLs that is not publicly available. The extractor uses a TLD-score heuristic as a proxy.

4. **Perfect test scores warrant scrutiny.** LightGBM and HistGB report accuracy, precision, recall, F1, and AUC-ROC all at exactly 1.0. While this may reflect genuine separability in the PhiUSIIL dataset (which is known to be clean and feature-rich), it should be validated on real-world out-of-distribution URLs before deployment.