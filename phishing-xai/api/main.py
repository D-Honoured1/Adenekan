"""
FastAPI backend — phishing detection + SHAP explanation endpoint.

Endpoints
---------
  GET  /health           — liveness check
  GET  /model-info       — loaded model metadata
  POST /predict          — full prediction + SHAP explanation
  POST /predict/url      — convenience: extract features from raw URL then predict

Run
---
    cd phishing-xai
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── path setup so src/ modules are importable ─────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import transform_single          # noqa: E402
from explain import build_explainer, compute_shap_values  # noqa: E402
from templates import build_explanation          # noqa: E402
from live_extractor import extract as live_extract, ExtractionResult  # noqa: E402

MODELS_DIR = ROOT / "models"

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phishing XAI API",
    description="SHAP-based explainable phishing URL detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── model loading (once at startup) ──────────────────────────────────────────
class _ModelStore:
    model = None
    explainer = None
    explainer_type = None
    feature_names: list[str] = []
    model_name: str = ""
    test_metrics: dict = {}
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None


store = _ModelStore()


@app.on_event("startup")
def load_model():
    results_path = MODELS_DIR / "test_results.json"
    if not results_path.exists():
        print("[startup] test_results.json not found — run evaluate.py first")
        return

    with open(results_path) as f:
        results = json.load(f)

    model_name = results.get("best_model", "lgbm")
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        print(f"[startup] {model_path} not found — run train.py first")
        return

    store.model_name   = model_name
    store.model        = joblib.load(model_path)
    store.feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    store.test_metrics  = results.get(model_name, {})

    # Try loading cached explainer, else build fresh
    explainer_path = MODELS_DIR / f"{model_name}_explainer.pkl"
    if explainer_path.exists():
        store.explainer, store.explainer_type = joblib.load(explainer_path)
    else:
        X_background = _load_background()
        store.explainer, store.explainer_type = build_explainer(
            model_name, store.model, X_background
        )

    print(f"[startup] Loaded model='{model_name}'  explainer='{store.explainer_type}'")

    x_path = MODELS_DIR / "X_test.npy"
    y_path = MODELS_DIR / "y_test.npy"
    if x_path.exists() and y_path.exists():
        store.X_test = np.load(x_path)
        store.y_test = np.load(y_path)
        print(f"[startup] Loaded test set: {len(store.X_test):,} samples")
    else:
        print("[startup] X_test.npy / y_test.npy not found — demo mode unavailable")


# ── schemas ───────────────────────────────────────────────────────────────────
class FeatureInput(BaseModel):
    """Send a dict of {feature_name: value} — missing features default to 0."""
    features: dict[str, float] = Field(..., description="Feature name → value mapping")


class URLInput(BaseModel):
    url: str = Field(..., description="Raw URL to analyse")


class PredictionResponse(BaseModel):
    prediction: int          # 0 = legitimate, 1 = phishing
    probability: float       # probability of phishing
    risk_level: str          # SAFE / LOW / MEDIUM / HIGH
    verdict: str             # one-sentence verdict
    summary: str             # plain-language summary
    top_features: list[dict] # [{"feature": ..., "shap": ...}]
    detail_sentences: list[str]
    shap_values: dict[str, float]
    shap_scale: str          # "log-odds" or "probability" — tells frontend how to interpret shap_values
    prediction_caveat: str   # non-empty when prediction may be unreliable (e.g. missing content features)
    features_extracted: int = 50   # how many of the 50 features were extracted from the live page
    fallback_features: list[str] = Field(default_factory=list)  # names that fell back to training means
    model_used: str
    inference_ms: float


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_background() -> np.ndarray:
    """Load SHAP background data, falling back to first 500 rows of X_test."""
    bg_path = MODELS_DIR / "X_background.npy"
    if bg_path.exists():
        return np.load(bg_path)
    x_path = MODELS_DIR / "X_test.npy"
    if x_path.exists():
        return np.load(x_path)[:500]
    raise FileNotFoundError("No background data found (X_background.npy or X_test.npy).")


def _check_ready():
    if store.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py and evaluate.py first.",
        )


def _run_prediction(
    features: dict[str, float],
    prediction_caveat: str = "",
    already_scaled: bool = False,
    features_extracted: int = 50,
    fallback_features: list[str] | None = None,
) -> PredictionResponse:
    _check_ready()

    t0 = time.perf_counter()

    if already_scaled:
        # Features are already imputed + Min-Max scaled (e.g. from X_test.npy).
        # Build array in the correct feature order; missing values default to 0.
        X = np.array(
            [[features.get(f, 0.0) for f in store.feature_names]],
            dtype=np.float32,
        )
    else:
        # Transform to scaled array
        X = transform_single(features, models_dir=str(MODELS_DIR))

    # Predict
    probability = float(store.model.predict_proba(X)[0, 1])
    prediction  = int(probability >= 0.5)

    # SHAP
    shap_raw = compute_shap_values(store.explainer, store.explainer_type, X)
    # shap_raw shape: (1, n_features)
    shap_vec = shap_raw[0] if shap_raw.ndim == 2 else shap_raw
    shap_dict = dict(zip(store.feature_names, shap_vec.tolist()))

    feature_values = {
        feat: features.get(feat, 0.0) for feat in store.feature_names
    }

    explanation = build_explanation(shap_dict, feature_values, prediction, probability)

    inference_ms = (time.perf_counter() - t0) * 1000

    shap_scale = "log-odds" if "raw" in store.explainer_type else "probability"

    return PredictionResponse(
        prediction         = prediction,
        probability        = round(probability, 6),
        risk_level         = explanation["risk_level"],
        verdict            = explanation["verdict"],
        summary            = explanation["summary"],
        top_features       = explanation["top_features"],
        detail_sentences   = explanation["detail_sentences"],
        shap_values        = {k: round(v, 5) for k, v in shap_dict.items()},
        shap_scale         = shap_scale,
        prediction_caveat  = prediction_caveat,
        features_extracted = features_extracted,
        fallback_features  = fallback_features or [],
        model_used         = store.model_name,
        inference_ms       = round(inference_ms, 2),
    )


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": store.model is not None,
        "model": store.model_name,
    }


@app.get("/model-info")
def model_info():
    _check_ready()
    return {
        "model":          store.model_name,
        "explainer_type": store.explainer_type,
        "n_features":     len(store.feature_names),
        "feature_names":  store.feature_names,
        "test_metrics":   store.test_metrics,
    }


class SwitchModelRequest(BaseModel):
    model: str


@app.post("/switch-model")
def switch_model(body: SwitchModelRequest):
    model_name = body.model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    try:
        loaded = joblib.load(model_path)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot load '{model_name}': {exc}. "
                   "The serving environment only supports lgbm, histgb, lr, and rf.",
        )

    store.model_name    = model_name
    store.model         = loaded
    store.feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

    results_path = MODELS_DIR / "test_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        store.test_metrics = results.get(model_name, {})

    try:
        explainer_path = MODELS_DIR / f"{model_name}_explainer.pkl"
        if explainer_path.exists():
            store.explainer, store.explainer_type = joblib.load(explainer_path)
        else:
            X_background = _load_background()
            store.explainer, store.explainer_type = build_explainer(
                model_name, store.model, X_background
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' loaded but explainer failed: {exc}",
        )

    print(f"[switch] model='{model_name}'  explainer='{store.explainer_type}'")
    return {"status": "ok", "model": store.model_name, "explainer_type": store.explainer_type}


@app.get("/demo-sample/{label}")
def demo_sample(label: int):
    """Return a random sample from the held-out test set for the given label (0=legitimate, 1=phishing)."""
    if store.X_test is None or store.y_test is None:
        raise HTTPException(status_code=503, detail="Test set not loaded on this server.")
    if label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 (legitimate) or 1 (phishing).")
    indices = np.where(store.y_test == label)[0]
    if len(indices) == 0:
        raise HTTPException(status_code=404, detail=f"No samples with label={label} in test set.")
    idx = int(np.random.default_rng().choice(indices))
    features = dict(zip(store.feature_names, store.X_test[idx].tolist()))
    return {"features": features, "label": label}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: FeatureInput):
    """
    Predict phishing probability from pre-extracted features.
    Returns prediction, SHAP explanation, and plain-language summary.
    """
    return _run_prediction(body.features)


@app.post("/predict/scaled", response_model=PredictionResponse)
def predict_scaled(body: FeatureInput):
    """
    Predict from features that are already imputed and Min-Max scaled (e.g. from X_test.npy).
    Skips the transform_single step. Used by the demo mode in the Streamlit frontend.
    """
    return _run_prediction(body.features, already_scaled=True)


@app.post("/predict/url", response_model=PredictionResponse)
def predict_url(body: URLInput):
    """
    Fetch the live page, extract all 50 PhiUSIIL features, then predict.

    - 22 lexical features are derived from the URL string alone.
    - 28 content features are extracted from the fetched HTML page.
    - Features that cannot be extracted fall back to training-set means;
      their names are listed in the response's fallback_features field.
    - features_extracted (out of 50) reflects extraction confidence.

    Handled failure cases (all return a valid prediction using lexical features only):
      page timeout · connection refused · SSL error · non-HTML response
    """
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    result: ExtractionResult = live_extract(url, models_dir=str(MODELS_DIR))

    if result.fetch_error:
        caveat = (
            f"Page could not be fetched ({result.fetch_error}). "
            f"Prediction uses 22/50 lexical URL features only; "
            f"28 content features defaulted to training means."
        )
    elif result.fallback_features:
        n_fb = len(result.fallback_features)
        caveat = (
            f"{result.features_extracted}/50 features extracted from live page. "
            f"{n_fb} feature(s) could not be determined and used training means "
            f"({', '.join(result.fallback_features[:4])}"
            + (" …" if n_fb > 4 else "") + ")."
        )
    else:
        caveat = ""

    return _run_prediction(
        result.features,
        prediction_caveat  = caveat,
        features_extracted = result.features_extracted,
        fallback_features  = result.fallback_features,
    )
