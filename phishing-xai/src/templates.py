"""
Plain-language explanation translation layer.

Converts raw SHAP values for a single prediction into human-readable sentences
that a non-technical user can understand.

Usage (standalone)
------------------
    from templates import build_explanation
    text = build_explanation(shap_dict, prediction=1, probability=0.93)
    print(text)
"""

from __future__ import annotations

# ── feature → human-readable description map ──────────────────────────────────
# Covers all 50 features retained after dropping FILENAME/URL/Domain/TLD/Title.
FEATURE_DESCRIPTIONS: dict[str, dict] = {
    # URL structure
    "URLLength":                 {"label": "URL length",              "direction": "high=sus"},
    "DomainLength":              {"label": "domain name length",       "direction": "high=sus"},
    "IsDomainIP":                {"label": "IP address as domain",     "direction": "1=sus"},
    "URLSimilarityIndex":        {"label": "URL similarity to known sites", "direction": "low=sus"},
    "CharContinuationRate":      {"label": "repeated character runs",  "direction": "high=sus"},
    "TLDLegitimateProb":         {"label": "TLD trustworthiness",      "direction": "low=sus"},
    "URLCharProb":               {"label": "URL character probability","direction": "low=sus"},
    "TLDLength":                 {"label": "top-level domain length",  "direction": "high=sus"},
    "NoOfSubDomain":             {"label": "number of subdomains",     "direction": "high=sus"},
    # Obfuscation
    "HasObfuscation":            {"label": "URL obfuscation present",  "direction": "1=sus"},
    "NoOfObfuscatedChar":        {"label": "obfuscated characters",    "direction": "high=sus"},
    "ObfuscationRatio":          {"label": "obfuscation ratio",        "direction": "high=sus"},
    # Character composition
    "NoOfLettersInURL":          {"label": "letters in URL",           "direction": "neutral"},
    "LetterRatioInURL":          {"label": "letter ratio in URL",      "direction": "low=sus"},
    "NoOfDegitsInURL":           {"label": "digits in URL",            "direction": "high=sus"},
    "DegitRatioInURL":           {"label": "digit ratio in URL",       "direction": "high=sus"},
    "NoOfEqualsInURL":           {"label": "'=' signs in URL",         "direction": "high=sus"},
    "NoOfQMarkInURL":            {"label": "'?' signs in URL",         "direction": "high=sus"},
    "NoOfAmpersandInURL":        {"label": "'&' signs in URL",         "direction": "high=sus"},
    "NoOfOtherSpecialCharsInURL":{"label": "special characters in URL","direction": "high=sus"},
    "SpacialCharRatioInURL":     {"label": "special char ratio",       "direction": "high=sus"},
    "IsHTTPS":                   {"label": "HTTPS usage",              "direction": "0=sus"},
    # Page content
    "LineOfCode":                {"label": "lines of HTML code",       "direction": "neutral"},
    "LargestLineLength":         {"label": "longest line of code",     "direction": "neutral"},
    "HasTitle":                  {"label": "page title present",       "direction": "0=sus"},
    "DomainTitleMatchScore":     {"label": "domain-to-title match",    "direction": "low=sus"},
    "URLTitleMatchScore":        {"label": "URL-to-title match",       "direction": "low=sus"},
    "HasFavicon":                {"label": "favicon present",          "direction": "0=sus"},
    "Robots":                    {"label": "robots.txt present",       "direction": "0=sus"},
    "IsResponsive":              {"label": "responsive design",        "direction": "0=sus"},
    "NoOfURLRedirect":           {"label": "URL redirects",            "direction": "high=sus"},
    "NoOfSelfRedirect":          {"label": "self-redirects",           "direction": "high=sus"},
    "HasDescription":            {"label": "meta description",         "direction": "0=sus"},
    "NoOfPopup":                 {"label": "popup windows",            "direction": "high=sus"},
    "NoOfiFrame":                {"label": "iframes on page",          "direction": "high=sus"},
    "HasExternalFormSubmit":     {"label": "external form submission",  "direction": "1=sus"},
    "HasSocialNet":              {"label": "social network links",      "direction": "0=sus"},
    "HasSubmitButton":           {"label": "submit button present",     "direction": "neutral"},
    "HasHiddenFields":           {"label": "hidden form fields",        "direction": "1=sus"},
    "HasPasswordField":          {"label": "password input field",      "direction": "1=sus"},
    # Keywords
    "Bank":                      {"label": "banking keyword in page",   "direction": "1=sus"},
    "Pay":                       {"label": "payment keyword in page",   "direction": "1=sus"},
    "Crypto":                    {"label": "crypto keyword in page",    "direction": "1=sus"},
    # Trust signals
    "HasCopyrightInfo":          {"label": "copyright notice",          "direction": "0=sus"},
    # Resource counts
    "NoOfImage":                 {"label": "images on page",            "direction": "neutral"},
    "NoOfCSS":                   {"label": "CSS stylesheets",           "direction": "neutral"},
    "NoOfJS":                    {"label": "JavaScript files",          "direction": "high=sus"},
    "NoOfSelfRef":               {"label": "self-referencing links",    "direction": "neutral"},
    "NoOfEmptyRef":              {"label": "empty href links",          "direction": "high=sus"},
    "NoOfExternalRef":           {"label": "external links",            "direction": "high=sus"},
}

# ── risk thresholds ───────────────────────────────────────────────────────────
RISK_LEVELS = [
    (0.80, "HIGH",    "This URL is very likely a phishing site."),
    (0.60, "MEDIUM",  "This URL shows several suspicious signals."),
    (0.40, "LOW",     "This URL appears mostly safe with minor concerns."),
    (0.00, "SAFE",    "This URL appears to be legitimate."),
]

TOP_N_FEATURES = 5   # number of features to highlight in the explanation


# ── helpers ───────────────────────────────────────────────────────────────────
def _risk_level(probability: float) -> tuple[str, str]:
    for threshold, level, verdict in RISK_LEVELS:
        if probability >= threshold:
            return level, verdict
    return "SAFE", "This URL appears to be legitimate."


def _feature_sentence(feature: str, shap_value: float, feature_value: float) -> str:
    """Convert one (feature, shap_value) pair into a plain English sentence."""
    meta = FEATURE_DESCRIPTIONS.get(feature, {"label": feature.replace("_", " "), "direction": "neutral"})
    label = meta["label"]
    pushes = "increases" if shap_value > 0 else "decreases"
    direction_word = "raising" if shap_value > 0 else "lowering"

    # Special-case binary / boolean features for clearer wording
    if meta["direction"] in ("1=sus", "0=sus"):
        val_str = "is active" if feature_value >= 0.5 else "is absent"
        return f"  • The {label} ({val_str}) {pushes} the phishing risk."
    else:
        magnitude = abs(shap_value)
        impact = "strongly" if magnitude > 0.05 else "slightly"
        return f"  • The {label} {impact} {direction_word} the phishing risk."


# ── main public function ──────────────────────────────────────────────────────
def build_explanation(
    shap_dict: dict[str, float],
    feature_values: dict[str, float],
    prediction: int,
    probability: float,
) -> dict:
    """
    Build a structured plain-language explanation.

    Parameters
    ----------
    shap_dict      : {feature_name: shap_value}  for the predicted instance
    feature_values : {feature_name: raw_scaled_value}  (0-1 scaled)
    prediction     : 0 (legitimate) or 1 (phishing)
    probability    : predicted probability for phishing class

    Returns
    -------
    dict with keys: risk_level, verdict, summary, top_features, detail_sentences
    """
    risk_level, verdict = _risk_level(probability)

    # Sort by absolute SHAP value, take top N
    ranked = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N_FEATURES]

    # Separate pushers and pullers
    pushers = [(f, v) for f, v in ranked if v > 0]   # push toward phishing
    pullers = [(f, v) for f, v in ranked if v < 0]   # pull toward legitimate

    sentences: list[str] = []

    if pushers:
        sentences.append("Factors increasing phishing risk:")
        for feat, sv in pushers:
            sentences.append(_feature_sentence(feat, sv, feature_values.get(feat, 0)))

    if pullers:
        sentences.append("Factors reducing phishing risk:")
        for feat, sv in pullers:
            sentences.append(_feature_sentence(feat, sv, feature_values.get(feat, 0)))

    label_str = "PHISHING" if prediction == 1 else "LEGITIMATE"
    summary = (
        f"The model classified this URL as {label_str} with "
        f"{probability * 100:.1f}% confidence. {verdict}"
    )

    return {
        "risk_level":       risk_level,
        "verdict":          verdict,
        "summary":          summary,
        "top_features":     [{"feature": f, "shap": round(v, 4)} for f, v in ranked],
        "detail_sentences": sentences,
        "full_text":        summary + "\n\n" + "\n".join(sentences),
    }


if __name__ == "__main__":
    # Smoke test
    sample_shap = {
        "IsHTTPS": -0.12,
        "HasPasswordField": 0.18,
        "URLSimilarityIndex": -0.09,
        "NoOfSubDomain": 0.07,
        "IsDomainIP": 0.15,
    }
    sample_vals = {k: 1.0 for k in sample_shap}
    result = build_explanation(sample_shap, sample_vals, prediction=1, probability=0.91)
    print(result["full_text"])
