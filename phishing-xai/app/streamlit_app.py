"""
PhishGuard XAI — Professional dark-themed Streamlit frontend.

Tabs
----
  1. Live URL Scanner    — fetch live page, extract all 50 features, explain
  2. Dataset Validation  — batch CSV upload with progress bar
  3. Evaluation Mode     — demo from held-out test set (no live fetch)
  4. Model Comparison    — all 7 models side-by-side with styled metric table

Run
---
    cd phishing-xai
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

API_BASE   = os.getenv("API_BASE_URL", "http://localhost:8000")
MODELS_DIR = ROOT / "models"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Palette (Catppuccin Mocha-inspired — designed to be easy on the eyes):
#   BG base   #1E1E2E  |  surface  #252535  |  border  #353558
#   text      #CDD6F4  |  subtext  #9399B2  |  muted   #6C7086
#   green     #A6E3A1  |  red      #F38BA8  |  amber   #FAB387
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: #1E1E2E !important;
}
[data-testid="stHeader"] { background-color: #1E1E2E !important; }
[data-testid="stSidebar"] {
    background-color: #181825 !important;
    border-right: 1px solid #313244;
}
[data-testid="stSidebar"] > div { background-color: #181825 !important; }

/* ── Typography ───────────────────────────────────── */
body, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p {
    color: #CDD6F4 !important;
}
h1, h2, h3, h4 { color: #E6E9F4 !important; }
label, .stSelectbox label, .stSlider label { color: #BAC2DE !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #9399B2 !important; }

/* ── Tabs ─────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #313244;
    gap: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    color: #9399B2;
    border-radius: 6px 6px 0 0;
    padding: 8px 22px;
    font-weight: 500;
    font-size: 0.92rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(166,227,161,0.07) !important;
    color: #A6E3A1 !important;
    border-bottom: 2px solid #A6E3A1 !important;
}

/* ── Cards ─────────────────────────────────────────── */
.pg-card {
    background: #252535;
    border: 1px solid #353558;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ── Metric chips ──────────────────────────────────── */
.pg-metric {
    background: #252535;
    border: 1px solid #353558;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.pg-metric .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #A6E3A1;
}
.pg-metric .lbl {
    font-size: 0.71rem;
    color: #9399B2;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}

/* ── Risk badges ───────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-HIGH   { background: rgba(243,139,168,0.15); color:#F38BA8; border:1px solid rgba(243,139,168,0.6); }
.badge-MEDIUM { background: rgba(250,179,135,0.15); color:#FAB387; border:1px solid rgba(250,179,135,0.6); }
.badge-LOW    { background: rgba(249,226,175,0.10); color:#F9E2AF; border:1px solid rgba(249,226,175,0.4); }
.badge-SAFE   { background: rgba(166,227,161,0.13); color:#A6E3A1; border:1px solid rgba(166,227,161,0.5); }

/* ── API status dot ─────────────────────────────────── */
.sdot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: middle;
}
.sdot-g { background:#A6E3A1; box-shadow:0 0 5px rgba(166,227,161,0.6); }
.sdot-r { background:#F38BA8; box-shadow:0 0 5px rgba(243,139,168,0.6); }

/* ── Inputs ─────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #252535 !important;
    border: 1px solid #353558 !important;
    color: #CDD6F4 !important;
    border-radius: 8px;
}
[data-testid="stTextInput"] input:focus {
    border-color: #A6E3A1 !important;
    box-shadow: 0 0 0 2px rgba(166,227,161,0.12) !important;
}

/* ── Primary button — dark-fill, easy on the eyes ─── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #2A3E35 !important;
    color: #A6E3A1 !important;
    border: 1px solid #3D6654 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #324840 !important;
    border-color: #A6E3A1 !important;
    box-shadow: none !important;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: #313244 !important; }

/* ── File uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background: #252535 !important;
    border: 1px dashed #353558 !important;
    border-radius: 10px;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploaderDropzone"] button {
    background: #2A3E35 !important;
    color: #A6E3A1 !important;
    border: 1px solid #3D6654 !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #324840 !important;
    border-color: #A6E3A1 !important;
}

/* ── Streamlit metric widget ─────────────────────────── */
[data-testid="stMetric"] { background: #252535; border-radius: 8px; padding: 10px 14px; }
[data-testid="stMetricLabel"] p { color: #9399B2 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"]   { color: #CDD6F4 !important; }

/* ── DataFrame ───────────────────────────────────────── */
[data-testid="stDataFrame"] iframe { background: #252535 !important; }

/* ── Status widget ───────────────────────────────────── */
[data-testid="stStatus"] {
    background: #252535 !important;
    border: 1px solid #353558 !important;
    border-radius: 10px;
}

/* ── Warning / info ──────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(250,179,135,0.08) !important;
    border-color: rgba(250,179,135,0.4) !important;
    border-radius: 8px;
}

/* ── Sidebar text overrides ──────────────────────────── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #BAC2DE !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3  { color: #E6E9F4 !important; }

/* ── Selectbox — read-only, no typing ───────────────── */
[data-testid="stSelectbox"] input {
    pointer-events: none !important;
    caret-color: transparent !important;
}

/* ── Header / menu buttons ───────────────────────────── */
[data-testid="stHeader"] button,
button[data-testid="stBaseButton-headerNoPadding"] {
    color: #9399B2 !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stHeader"] button:hover,
button[data-testid="stBaseButton-headerNoPadding"]:hover {
    color: #CDD6F4 !important;
    background: rgba(205,214,244,0.06) !important;
}

/* ══ RESPONSIVE ══════════════════════════════════════════

   Breakpoints
   ─────────────────────────────────────────────────────
   ≤ 960px  tablet   — tighten tab labels, reduce padding
   ≤ 640px  mobile   — stack all columns, shrink text
*/

/* ── Tablet (≤ 960px) ───────────────────────────────── */
@media (max-width: 960px) {
    .main .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 8px 14px;
        font-size: 0.83rem;
    }
    .pg-metric .val { font-size: 1.25rem; }
}

/* ── Mobile (≤ 640px) ───────────────────────────────── */
@media (max-width: 640px) {
    /* Reduce outer padding */
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1.5rem !important;
    }

    /* Stack every horizontal column block */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.4rem;
    }
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Tabs — horizontal scroll, no wrap, hidden scrollbar */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 7px 10px !important;
        font-size: 0.72rem !important;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* Cards */
    .pg-card { padding: 12px 14px; }

    /* Metric chips */
    .pg-metric { padding: 10px 10px; }
    .pg-metric .val { font-size: 1.05rem; }
    .pg-metric .lbl { font-size: 0.62rem; letter-spacing: 0.04em; }

    /* Headings */
    h1 { font-size: 1.55rem !important; }
    h3 { font-size: 1.05rem !important; }

    /* Badge — smaller on mobile */
    .badge { padding: 4px 12px; font-size: 0.78rem; }

    /* Plotly charts — allow horizontal scroll instead of squishing */
    [data-testid="stPlotlyChart"] > div {
        overflow-x: auto !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ── feature templates (plain-language) ───────────────────────────────────────
FEATURE_TEMPLATES: dict[str, dict[str, str]] = {
    "URLSimilarityIndex":    {"sus": "Low similarity to known legitimate URLs — a common phishing trait.",
                              "safe": "URL closely matches patterns of legitimate sites."},
    "IsHTTPS":               {"sus": "No HTTPS — unusual for legitimate services handling sensitive data.",
                              "safe": "Site uses HTTPS, providing basic connection security."},
    "HasSocialNet":          {"sus": "No social network links — phishing pages typically lack these trust signals.",
                              "safe": "Social network links present — a hallmark of genuine organisations."},
    "HasCopyrightInfo":      {"sus": "No copyright notice — phishing pages rarely include this trust signal.",
                              "safe": "Copyright notice present — suggests an established, legitimate site."},
    "HasDescription":        {"sus": "No meta description — hastily assembled phishing pages often skip this.",
                              "safe": "Meta description present — typical of professionally maintained sites."},
    "DomainTitleMatchScore": {"sus": "Page title doesn't match the domain — a sign of brand impersonation.",
                              "safe": "Page title matches the domain — consistent, legitimate branding."},
    "URLTitleMatchScore":    {"sus": "URL and page title are inconsistent — red flag for impersonation.",
                              "safe": "URL and page title are consistent, suggesting a coherent, legitimate site."},
    "IsDomainIP":            {"sus": "Domain is a raw IP address — legitimate sites almost never use this.",
                              "safe": "Domain uses a proper hostname, not a raw IP address."},
    "NoOfSubDomain":         {"sus": "Multiple subdomains — a tactic to disguise the real domain.",
                              "safe": "Few or no extra subdomains — consistent with a straightforward address."},
    "URLLength":             {"sus": "Unusually long URL — phishing sites use length to hide suspicious parts.",
                              "safe": "Short, concise URL — typical of well-established legitimate sites."},
    "HasObfuscation":        {"sus": "Obfuscated characters in URL — used to disguise phishing links.",
                              "safe": "No URL obfuscation detected."},
    "HasPasswordField":      {"sus": "Password field present — combined with other signals, may indicate credential harvesting.",
                              "safe": "No password field — reduces credential-harvesting risk."},
    "HasExternalFormSubmit": {"sus": "Forms submit to an external domain — a classic phishing technique.",
                              "safe": "Form submissions stay on the same domain — as expected from a legitimate site."},
    "NoOfURLRedirect":       {"sus": "Multiple redirects — phishing sites chain redirects to obscure destinations.",
                              "safe": "No unusual redirects detected."},
    "SpacialCharRatioInURL": {"sus": "High special-character ratio in URL — a common obfuscation indicator.",
                              "safe": "Few special characters in URL — typical of well-formed addresses."},
    "TLDLegitimateProb":     {"sus": "Low-legitimacy top-level domain — phishing sites often use obscure TLDs.",
                              "safe": "Top-level domain commonly associated with legitimate sites."},
    "DomainLength":          {"sus": "Unusually long domain name — atypical of well-known organisations.",
                              "safe": "Concise domain name — typical of well-known organisations."},
    "IsResponsive":          {"sus": "No responsive design — hastily built phishing pages often skip this.",
                              "safe": "Responsive design present — consistent with a professionally built site."},
    "HasFavicon":            {"sus": "No favicon — legitimate sites almost always include one for branding.",
                              "safe": "Favicon present — a small but consistent sign of a maintained site."},
    "HasHiddenFields":       {"sus": "Hidden form fields found — sometimes used to smuggle data silently.",
                              "safe": "No hidden form fields detected."},
    "NoOfPopup":             {"sus": "Popup windows detected — phishing sites use these to pressure users.",
                              "safe": "No popup windows — consistent with a non-aggressive, legitimate site."},
    "NoOfiFrame":            {"sus": "iFrames found — can embed content from another site, masking its origin.",
                              "safe": "No iFrames detected — straightforward page structure."},
    "NoOfEmptyRef":          {"sus": "Empty href links — placeholder links are common on cloned phishing pages.",
                              "safe": "No empty href links — page was not hastily cloned."},
    "NoOfExternalRef":       {"sus": "High external link count — may indicate a scraped or cloned page.",
                              "safe": "External links within normal range for a legitimate page."},
    "NoOfJS":                {"sus": "High JavaScript file count — can be used to run malicious code.",
                              "safe": "JavaScript usage within normal range."},
    "CharContinuationRate":  {"sus": "Repeated character sequences in URL — used to confuse automated filters.",
                              "safe": "No unusual repeated character patterns in the URL."},
    "Robots":                {"sus": "No robots.txt — legitimate sites typically include this.",
                              "safe": "robots.txt present — consistent with a well-managed site."},
    "Bank":                  {"sus": "Banking keywords on page — combined with other signals, may indicate banking phishing.",
                              "safe": "Banking keywords not prominent on this page."},
    "Pay":                   {"sus": "Payment keywords on page — phishing pages often use these to create urgency.",
                              "safe": "Payment keywords not prominent on this page."},
    "Crypto":                {"sus": "Cryptocurrency keywords on page — common in modern phishing and scam campaigns.",
                              "safe": "No cryptocurrency keywords detected."},
    "NoOfSelfRedirect":      {"sus": "Self-redirects detected — can stall security scanners and confuse users.",
                              "safe": "No self-redirects — consistent with a straightforward site."},
}
_GENERIC = {
    "sus":  "The value of '{label}' is atypical and raises the phishing risk.",
    "safe": "The value of '{label}' is within normal range.",
}


def _rich_bullets(top_features: list[dict], shap_values: dict) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for entry in top_features:
        feat = entry["feature"]
        shap = shap_values.get(feat, entry.get("shap", 0.0))
        direction = "sus" if shap > 0 else "safe"
        tmpl = FEATURE_TEMPLATES.get(feat)
        sentence = (
            tmpl[direction] if tmpl
            else _GENERIC[direction].format(label=feat.replace("_", " "))
        )
        if sentence in seen:
            sentence += f" ({feat})"
        seen.add(sentence)
        icon = "🔴" if direction == "sus" else "🟢"
        lines.append(f"{icon} {sentence}")
    return lines


# ── API helpers ───────────────────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


@st.cache_data(ttl=5)
def _api_status() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "offline", "model_loaded": False, "model": ""}


@st.cache_data
def _load_test_results() -> dict:
    path = MODELS_DIR / "test_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _predict_url(url: str) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict/url", json={"url": url}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE}. Check that API_BASE_URL is set correctly.")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def _predict_scaled(features: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict/scaled", json={"features": features}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE}. Check that API_BASE_URL is set correctly.")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def _switch_model(model_name: str) -> bool:
    try:
        r = requests.post(f"{API_BASE}/switch-model", json={"model": model_name}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to switch model: {e}")
        return False


def _load_demo_sample(label: int) -> dict | None:
    x_path  = MODELS_DIR / "X_test.npy"
    y_path  = MODELS_DIR / "y_test.npy"
    fn_path = MODELS_DIR / "feature_names.pkl"
    if not (x_path.exists() and y_path.exists() and fn_path.exists()):
        st.error("Test-set files missing in models/. Run evaluate.py first.")
        return None
    X_test        = np.load(x_path)
    y_test        = np.load(y_path)
    feature_names = joblib.load(fn_path)
    indices = np.where(y_test == label)[0]
    if len(indices) == 0:
        st.error(f"No samples with label={label} in test set.")
        return None
    idx = np.random.default_rng().choice(indices)
    return dict(zip(feature_names, X_test[idx].tolist()))


# ── Plotly chart builders ─────────────────────────────────────────────────────
def _gauge(prob: float) -> go.Figure:
    if prob >= 0.7:
        color = "#F38BA8"
    elif prob >= 0.4:
        color = "#FAB387"
    else:
        color = "#A6E3A1"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 2),
        number={"suffix": "%", "font": {"size": 42, "color": color},
                "valueformat": ".2f"},
        title={"text": "Phishing Probability", "font": {"size": 13, "color": "#9399B2"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#313244",
                "tickfont": {"color": "#9399B2", "size": 11},
            },
            "bar":      {"color": color, "thickness": 0.25},
            "bgcolor":  "#252535",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "rgba(166,227,161,0.07)"},
                {"range": [40, 70],  "color": "rgba(250,179,135,0.07)"},
                {"range": [70, 100], "color": "rgba(243,139,168,0.07)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "value": round(prob * 100, 1),
            },
        },
    ))
    fig.update_layout(
        height=265,
        margin=dict(l=20, r=20, t=55, b=0),
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
    )
    return fig


def _waterfall(shap_values: dict, shap_scale: str = "probability", top_n: int = 15) -> go.Figure:
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features = [item[0] for item in sorted_items]
    raw      = [item[1] for item in sorted_items]

    if shap_scale == "log-odds":
        values      = [_sigmoid(v) - 0.5 for v in raw]
        xaxis_title = "SHAP (probability scale, sigmoid-converted)"
    else:
        values      = raw
        xaxis_title = "SHAP value (probability)"

    colors = ["#F38BA8" if v > 0 else "#A6E3A1" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        marker_opacity=0.85,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont={"color": "#BAC2DE", "size": 11},
    ))
    fig.update_layout(
        title=dict(
            text="SHAP Feature Contributions  "
                 "<span style='font-size:12px;color:#9399B2'>"
                 "(red = toward phishing · green = toward safe)</span>",
            font={"color": "#E6E9F4", "size": 14},
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font={"color": "#9399B2", "size": 11}),
            tickfont={"color": "#9399B2"},
            gridcolor="#313244",
            zerolinecolor="#45475A",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont={"color": "#BAC2DE", "size": 11},
            gridcolor="#313244",
        ),
        height=max(420, top_n * 33),
        margin=dict(l=10, r=110, t=60, b=40),
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#252535",
    )
    return fig


def _bar_chart(
    models: list[str],
    values: list[float],
    title: str,
    yaxis_label: str,
    threshold_high: float = 0.99,
    threshold_mid: float = 0.97,
) -> go.Figure:
    colors = [
        "#A6E3A1" if v >= threshold_high else
        ("#FAB387" if v >= threshold_mid else "#F38BA8")
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=[m.upper() for m in models],
        y=values,
        marker_color=colors,
        marker_opacity=0.85,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont={"color": "#BAC2DE", "size": 11},
    ))
    fig.update_layout(
        title=dict(text=title, font={"color": "#E6E9F4", "size": 14}),
        yaxis=dict(
            range=[min(values) * 0.995 if values else 0, 1.004],
            title=dict(text=yaxis_label, font={"color": "#9399B2"}),
            tickfont={"color": "#9399B2"},
            gridcolor="#313244",
        ),
        xaxis=dict(tickfont={"color": "#BAC2DE"}),
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#252535",
        margin=dict(l=40, r=40, t=55, b=40),
        height=360,
    )
    return fig


# ── render prediction result ──────────────────────────────────────────────────
def _render_result(result: dict, context_key: str = "") -> None:
    prob       = float(result["probability"])
    risk       = result["risk_level"]
    shap_scale = result.get("shap_scale", "probability")
    caveat     = result.get("prediction_caveat", "")
    n_extracted = result.get("features_extracted", 50)
    inf_ms      = result.get("inference_ms", 0)
    model_name  = result.get("model_used", "—")

    if caveat:
        st.warning(f"⚠  {caveat}")

    col_gauge, col_info = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(
            _gauge(prob),
            width="stretch",
            key=f"gauge_{context_key}_{prob}",
        )

    with col_info:
        st.markdown(
            f'<span class="badge badge-{risk}">{risk} RISK</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {result['verdict']}")
        st.markdown(result["summary"])

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(
            f'<div class="pg-metric"><div class="val">{prob*100:.2f}%</div>'
            f'<div class="lbl">Phishing Prob</div></div>',
            unsafe_allow_html=True,
        )
        mc2.markdown(
            f'<div class="pg-metric"><div class="val">{n_extracted}/50</div>'
            f'<div class="lbl">Features Extracted</div></div>',
            unsafe_allow_html=True,
        )
        mc3.markdown(
            f'<div class="pg-metric"><div class="val">{inf_ms:.0f} ms</div>'
            f'<div class="lbl">Inference Time</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model: **{model_name.upper()}** · SHAP scale: **{shap_scale}**")

    st.divider()

    col_shap, col_explain = st.columns([3, 2])

    with col_shap:
        st.plotly_chart(
            _waterfall(result["shap_values"], shap_scale=shap_scale),
            width="stretch",
            key=f"wf_{context_key}_{prob}",
        )

    with col_explain:
        st.subheader("Plain-Language Explanation")
        st.caption("Key factors that drove this prediction:")
        for bullet in _rich_bullets(result["top_features"], result["shap_values"]):
            st.markdown(bullet)

        fallback = result.get("fallback_features", [])
        if fallback:
            with st.expander(f"Features defaulted to training means ({len(fallback)})"):
                st.write(", ".join(fallback))


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 18px;">
        <div style="font-size:2.2rem;">🛡️</div>
        <div style="font-size:1.2rem;font-weight:700;color:#FFFFFF;letter-spacing:0.03em;">
            PhishGuard XAI
        </div>
        <div style="font-size:0.68rem;color:#8B8FA8;margin-top:6px;line-height:1.5;">
            BSc Final Year Project<br>Redeemer's University
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    health = _api_status()
    api_up = health.get("model_loaded", False)
    dot_cls = "sdot-g" if api_up else "sdot-r"
    status_label = "API Online" if api_up else "API Offline"
    st.markdown(
        f'<div style="margin-bottom:14px;">'
        f'<span class="sdot {dot_cls}"></span>'
        f'<span style="color:#BAC2DE;font-size:0.9rem;">{status_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if not api_up:
        st.caption(f"Cannot reach `{API_BASE}`")

    ALL_MODELS = ["lgbm", "catboost", "histgb", "lr", "rf", "xgb", "attention_mlp"]
    MODEL_LABELS = {
        "lgbm": "LightGBM", "catboost": "CatBoost", "histgb": "HistGradientBoosting",
        "lr": "Logistic Regression", "rf": "Random Forest",
        "xgb": "XGBoost", "attention_mlp": "Attention MLP",
    }
    active_model = health.get("model", ALL_MODELS[0]) or ALL_MODELS[0]
    active_idx   = ALL_MODELS.index(active_model) if active_model in ALL_MODELS else 0

    def _fmt_model(m: str) -> str:
        suffix = "  ✓ active" if m == active_model else ""
        return MODEL_LABELS.get(m, m) + suffix

    selected_model = st.selectbox(
        "Model",
        options=ALL_MODELS,
        index=active_idx,
        format_func=_fmt_model,
        help="Select a model and click Switch to change the active prediction model.",
    )
    if selected_model != active_model:
        if st.button(
            f"⇄ Switch to {MODEL_LABELS[selected_model]}",
            type="primary",
            width="stretch",
            disabled=not api_up,
        ):
            with st.spinner(f"Switching to {MODEL_LABELS[selected_model]}…"):
                if _switch_model(selected_model):
                    _api_status.clear()
                    st.rerun()

    results_all = _load_test_results()
    if selected_model in results_all and isinstance(results_all[selected_model], dict):
        m_data = results_all[selected_model]
        st.markdown("**Test-set Metrics**")
        c1, c2 = st.columns(2)
        for k, label in [("accuracy", "Accuracy"), ("f1", "F1-Score"),
                         ("auc_roc", "AUC-ROC"), ("precision", "Precision")]:
            val = m_data.get(k)
            if val is not None:
                (c1 if k in ("accuracy", "auc_roc") else c2).metric(label, f"{val:.4f}")
        if "shap_ms_per_instance" in m_data:
            flag = " ⚠" if m_data["shap_ms_per_instance"] > 100 else " ✓"
            st.metric("SHAP ms/inst", f"{m_data['shap_ms_per_instance']}{flag}")

    st.divider()
    st.markdown(
        '<div style="font-size:0.73rem;color:#6C7086;line-height:1.6;">'
        'Dataset: PhiUSIIL (235,795 URLs)<br>'
        'Features: 50 PhiUSIIL features<br>'
        'XAI: SHAP (Tree · Linear · Permutation)'
        '</div>',
        unsafe_allow_html=True,
    )


# ── main heading ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:2px;">PhishGuard <span style="color:#A6E3A1;">XAI</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#9399B2;margin-top:0;margin-bottom:20px;">'
    'Explainable Phishing URL Detection · PhiUSIIL Dataset · 7 Models</p>',
    unsafe_allow_html=True,
)

tab_scanner, tab_validation, tab_demo, tab_comparison = st.tabs([
    "🔍 Live URL Scanner",
    "📋 Dataset Validation",
    "🧪 Evaluation Mode",
    "📊 Model Comparison",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live URL Scanner
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scanner:
    st.markdown("### Scan a URL in Real Time")
    st.caption(
        "Enter any URL. The backend fetches the live page, extracts all 50 PhiUSIIL features, "
        "and returns a SHAP-based explanation."
    )

    url_col, btn_col = st.columns([5, 1])
    with url_col:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com",
            label_visibility="collapsed",
            key="url_input_field",
        )
    with btn_col:
        scan_clicked = st.button(
            "Scan", type="primary", disabled=not api_up, width="stretch",
        )

    if scan_clicked:
        url_str = url_input.strip()
        if not url_str:
            st.warning("Please enter a URL.")
        else:
            with st.status("Scanning URL…", expanded=True) as scan_status:
                st.write("🌐 Fetching live page and extracting features…")
                result = _predict_url(url_str)
                if result:
                    st.write("🤖 Running model inference…")
                    st.write("🔍 Computing SHAP values…")
                    scan_status.update(label="Scan complete ✓", state="complete", expanded=False)
                    st.session_state["scanner_result"] = result
                    st.session_state["scanner_url"]    = url_str
                else:
                    scan_status.update(label="Scan failed", state="error")

    if "scanner_result" in st.session_state:
        url_lbl = st.session_state.get("scanner_url", "")
        if url_lbl:
            st.caption(f"Results for `{url_lbl}`")
        _render_result(st.session_state["scanner_result"], context_key="scanner")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Validation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_validation:
    st.markdown("### Batch URL Validation")
    st.caption(
        "Upload a CSV with a **URL** column and an optional **label** column "
        "(0 = legitimate, 1 = phishing). Each URL is scanned live via the API."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            df_raw = None

        if df_raw is not None:
            url_col_candidates   = [c for c in df_raw.columns if c.strip().lower() in ("url", "urls")]
            label_col_candidates = [c for c in df_raw.columns
                                    if c.strip().lower() in ("label", "labels", "class", "phishing")]

            url_col_name   = url_col_candidates[0]   if url_col_candidates   else None
            label_col_name = label_col_candidates[0] if label_col_candidates else None

            if url_col_name is None:
                st.error("No 'URL' column found. Ensure your CSV has a column named 'URL'.")
            else:
                urls   = df_raw[url_col_name].dropna().astype(str).tolist()
                labels = (
                    df_raw[label_col_name].fillna(-1).astype(int).tolist()
                    if label_col_name else [None] * len(urls)
                )

                st.markdown(f"**{len(urls)} URLs detected.** Preview:")
                st.dataframe(df_raw.head(5), width="stretch", hide_index=True)

                max_rows = st.slider(
                    "URLs to validate", 1, min(len(urls), 100), min(10, len(urls))
                )

                if st.button("Run Validation", type="primary", disabled=not api_up):
                    urls_to_scan   = urls[:max_rows]
                    labels_to_scan = labels[:max_rows]
                    progress_bar   = st.progress(0, text="Starting…")
                    rows: list[dict] = []

                    for i, (url, true_label) in enumerate(zip(urls_to_scan, labels_to_scan)):
                        short = url[:55] + ("…" if len(url) > 55 else "")
                        progress_bar.progress(
                            (i + 1) / max_rows,
                            text=f"Scanning {i+1}/{max_rows}: {short}",
                        )
                        res = _predict_url(url)
                        if res:
                            pred  = res["prediction"]
                            prob  = res["probability"]
                            risk  = res["risk_level"]
                            correct: str | None = None
                            if true_label is not None and true_label in (0, 1):
                                correct = "✅" if pred == true_label else "❌"
                            rows.append({
                                "URL":          short,
                                "Risk":         risk,
                                "Probability":  f"{prob*100:.1f}%",
                                "Predicted":    "Phishing" if pred == 1 else "Legitimate",
                                "True Label":   (
                                    "Phishing" if true_label == 1 else "Legitimate"
                                ) if true_label in (0, 1) else "—",
                                "Correct":      correct or "—",
                                "Inference ms": res["inference_ms"],
                            })
                        else:
                            rows.append({
                                "URL": short, "Risk": "ERROR", "Probability": "—",
                                "Predicted": "—", "True Label": "—",
                                "Correct": "—", "Inference ms": "—",
                            })

                    progress_bar.empty()
                    st.session_state["val_results"] = pd.DataFrame(rows)

    if "val_results" in st.session_state:
        df_r = st.session_state["val_results"]
        st.markdown("#### Results")
        st.dataframe(df_r, width="stretch", hide_index=True)

        correct_mask = df_r["Correct"] == "✅"
        wrong_mask   = df_r["Correct"] == "❌"
        if correct_mask.any() or wrong_mask.any():
            total_labeled = (correct_mask | wrong_mask).sum()
            n_correct     = correct_mask.sum()
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Scanned",      len(df_r))
            mc2.metric("Correct Predictions", f"{n_correct}/{total_labeled}")
            mc3.metric("Accuracy",  f"{n_correct/total_labeled*100:.1f}%" if total_labeled else "—")
            mc4.metric("Phishing Detected", str((df_r["Predicted"] == "Phishing").sum()))

        csv_out = df_r.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download Results CSV", data=csv_out,
            file_name="phishguard_validation.csv", mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation Mode (Demo)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("### Evaluation Mode")
    st.caption(
        "Load random samples directly from the held-out test set (X_test.npy). "
        "All 50 pre-scaled features are sent to the model — no live page fetch required."
    )

    dcol_ph, dcol_le = st.columns(2)

    with dcol_ph:
        st.markdown(
            '<div class="pg-card" style="border-color:rgba(243,139,168,0.5);text-align:center;min-height:120px;">'
            '<div style="font-size:2rem;margin-bottom:6px;">🎣</div>'
            '<div style="font-weight:700;color:#F38BA8;font-size:1rem;margin-bottom:4px;">Known Phishing</div>'
            '<div style="font-size:0.78rem;color:#9399B2;">'
            'Loads a random phishing sample from the held-out test set'
            '</div></div>',
            unsafe_allow_html=True,
        )
        demo_phish = st.button(
            "Load Phishing Sample", type="primary",
            disabled=not api_up, width="stretch", key="btn_demo_phish",
        )

    with dcol_le:
        st.markdown(
            '<div class="pg-card" style="border-color:rgba(166,227,161,0.5);text-align:center;min-height:120px;">'
            '<div style="font-size:2rem;margin-bottom:6px;">✅</div>'
            '<div style="font-weight:700;color:#A6E3A1;font-size:1rem;margin-bottom:4px;">Known Legitimate</div>'
            '<div style="font-size:0.78rem;color:#9399B2;">'
            'Loads a random legitimate sample from the held-out test set'
            '</div></div>',
            unsafe_allow_html=True,
        )
        demo_legit = st.button(
            "Load Legitimate Sample", type="primary",
            disabled=not api_up, width="stretch", key="btn_demo_legit",
        )

    if demo_phish:
        with st.spinner("Loading phishing sample from test set…"):
            feats = _load_demo_sample(label=1)
            if feats:
                res = _predict_scaled(feats)
                if res:
                    st.session_state["demo_result"]   = res
                    st.session_state["demo_expected"]  = "Phishing"

    if demo_legit:
        with st.spinner("Loading legitimate sample from test set…"):
            feats = _load_demo_sample(label=0)
            if feats:
                res = _predict_scaled(feats)
                if res:
                    st.session_state["demo_result"]   = res
                    st.session_state["demo_expected"]  = "Legitimate"

    if "demo_result" in st.session_state:
        res_d     = st.session_state["demo_result"]
        expected  = st.session_state.get("demo_expected", "")
        predicted = "Phishing" if res_d["prediction"] == 1 else "Legitimate"
        correct   = predicted == expected

        st.divider()
        banner_color = "166,227,161" if correct else "243,139,168"
        border_color = "#A6E3A1"     if correct else "#F38BA8"
        icon = "✅" if correct else "❌"
        st.markdown(
            f'<div style="border:1px solid {border_color};border-radius:8px;'
            f'padding:12px 20px;background:rgba({banner_color},0.08);margin-bottom:16px;">'
            f'<span style="font-weight:700;color:{border_color};">'
            f'{icon} Expected: {expected}  |  Predicted: {predicted}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        _render_result(res_d, context_key="demo")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_comparison:
    st.markdown("### Model Comparison — Test-Set Performance")
    st.caption("All seven models evaluated on the held-out test set (evaluate.py).")

    results_all = _load_test_results()

    MODEL_ORDER  = ["lgbm", "catboost", "histgb", "lr", "rf", "xgb", "attention_mlp"]
    METRIC_COLS  = ["accuracy", "precision", "recall", "f1", "auc_roc", "shap_ms_per_instance"]
    METRIC_HDR   = {
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1-Score", "auc_roc": "AUC-ROC", "shap_ms_per_instance": "SHAP ms/inst",
    }
    HIGHER_BETTER = {"accuracy", "precision", "recall", "f1", "auc_roc"}

    present = [m for m in MODEL_ORDER
               if m in results_all and isinstance(results_all[m], dict)]

    if not present:
        st.warning("No test results found. Run `python src/evaluate.py` first.")
    else:
        best_model = results_all.get("best_model", "")

        def _cell_color(val: object, metric_key: str) -> tuple[str, str]:
            """Return (bg_css, text_css) for a metric cell."""
            if not isinstance(val, (int, float)):
                return "#252535", "#CDD6F4"
            if metric_key in HIGHER_BETTER:
                if val >= 0.99:
                    return "rgba(166,227,161,0.10)", "#A6E3A1"
                if val >= 0.97:
                    return "rgba(250,179,135,0.10)", "#FAB387"
                return "rgba(243,139,168,0.10)", "#F38BA8"
            else:  # SHAP ms — lower is better
                if val <= 10:
                    return "rgba(166,227,161,0.10)", "#A6E3A1"
                if val <= 100:
                    return "rgba(250,179,135,0.10)", "#FAB387"
                return "rgba(243,139,168,0.10)", "#F38BA8"

        # Build pure-HTML table — avoids Streamlit's iframe dataframe chrome
        th_style = (
            "padding:10px 14px;text-align:left;font-size:0.78rem;"
            "font-weight:600;color:#9399B2;text-transform:uppercase;"
            "letter-spacing:0.06em;border-bottom:1px solid #313244;"
            "background:#1E1E2E;"
        )
        td_base = (
            "padding:9px 14px;font-size:0.85rem;"
            "border-bottom:1px solid #2A2A3E;"
        )

        headers = ["Model"] + list(METRIC_HDR.values())
        html = (
            '<div style="overflow-x:auto;border:1px solid #313244;'
            'border-radius:10px;margin-bottom:4px;">'
            '<table style="width:100%;border-collapse:collapse;">'
            "<thead><tr>"
        )
        for h in headers:
            html += f'<th style="{th_style}">{h}</th>'
        html += "</tr></thead><tbody>"

        for i, m in enumerate(present):
            mdata   = results_all[m]
            row_bg  = "#1E1E2E" if i % 2 == 0 else "#252535"
            is_best = (m == best_model)
            label   = m.upper() + (" ★" if is_best else "")
            name_color = "#A6E3A1" if is_best else "#CDD6F4"

            html += "<tr>"
            html += (
                f'<td style="{td_base}background:{row_bg};'
                f'font-weight:{"700" if is_best else "400"};color:{name_color};">'
                f"{label}</td>"
            )
            for col_key, col_hdr in METRIC_HDR.items():
                val = mdata.get(col_key)
                bg, fg = _cell_color(val, col_key)
                fmt    = f"{val:.4f}" if isinstance(val, float) else (str(val) if val is not None else "—")
                html += (
                    f'<td style="{td_base}background:{bg};color:{fg};'
                    f'font-variant-numeric:tabular-nums;">{fmt}</td>'
                )
            html += "</tr>"

        html += "</tbody></table></div>"

        st.markdown(
            '<div style="font-size:0.78rem;color:#6C7086;margin-bottom:8px;">'
            "★ best model · "
            '<span style="color:#A6E3A1;">■</span> ≥ 0.99 · '
            '<span style="color:#FAB387;">■</span> ≥ 0.97 · '
            '<span style="color:#F38BA8;">■</span> below threshold'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(html, unsafe_allow_html=True)

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            f1_vals = [results_all[m].get("f1", 0) for m in present]
            st.plotly_chart(
                _bar_chart(present, f1_vals, "F1-Score by Model", "F1-Score"),
                width="stretch", key="f1_bar",
            )

        with chart_col2:
            auc_vals = [results_all[m].get("auc_roc", 0) for m in present]
            st.plotly_chart(
                _bar_chart(present, auc_vals, "AUC-ROC by Model", "AUC-ROC"),
                width="stretch", key="auc_bar",
            )
