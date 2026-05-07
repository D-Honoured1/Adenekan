"""
URL lexical feature extractor for live inference in the Streamlit demo.

Extracts the ~22 URL-structure features that can be derived purely from the
URL string itself (no network requests needed). The remaining 28 content-based
features that require fetching the page are set to their training-set medians
so the model still receives a complete 50-feature vector.

Content-based features that default to medians
-----------------------------------------------
LineOfCode, LargestLineLength, HasTitle, DomainTitleMatchScore,
URLTitleMatchScore, HasFavicon, Robots, IsResponsive, NoOfURLRedirect,
NoOfSelfRedirect, HasDescription, NoOfPopup, NoOfiFrame,
HasExternalFormSubmit, HasSocialNet, HasSubmitButton, HasHiddenFields,
HasPasswordField, Bank, Pay, Crypto, HasCopyrightInfo,
NoOfImage, NoOfCSS, NoOfJS, NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef

Note: in a production system these would be extracted via a headless browser.
"""

from __future__ import annotations

import math
import re
import urllib.parse

# Training-set approximate medians for content-based features
# (estimated from the PhiUSIIL dataset statistics)
CONTENT_DEFAULTS: dict[str, float] = {
    "LineOfCode":            200,
    "LargestLineLength":     500,
    "HasTitle":              1,
    "DomainTitleMatchScore": 0,
    "URLTitleMatchScore":    0,
    "HasFavicon":            1,
    "Robots":                0,
    "IsResponsive":          1,
    "NoOfURLRedirect":       0,
    "NoOfSelfRedirect":      0,
    "HasDescription":        0,
    "NoOfPopup":             0,
    "NoOfiFrame":            0,
    "HasExternalFormSubmit": 0,
    "HasSocialNet":          0,
    "HasSubmitButton":       0,
    "HasHiddenFields":       0,
    "HasPasswordField":      0,
    "Bank":                  0,
    "Pay":                   0,
    "Crypto":                0,
    "HasCopyrightInfo":      0,
    "NoOfImage":             5,
    "NoOfCSS":               2,
    "NoOfJS":                3,
    "NoOfSelfRef":           10,
    "NoOfEmptyRef":          0,
    "NoOfExternalRef":       5,
}

# Common legitimate TLDs and their approximate legitimacy probability
TLD_PROB: dict[str, float] = {
    "com": 0.52, "org": 0.50, "net": 0.48, "edu": 0.65, "gov": 0.70,
    "co": 0.40,  "io":  0.35, "uk":  0.45, "us":  0.45,
}

SPECIAL_CHARS = set("!@#$%^&*()_+{}|:<>?`~[]\\;',./=")
OBFUSCATION_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def _tld_from_url(parsed) -> str:
    host = parsed.netloc.lower().split(":")[0]  # strip port
    parts = host.split(".")
    return parts[-1] if parts else ""


def extract(url: str) -> dict[str, float]:
    """
    Extract 50 features from a raw URL string.

    Returns a flat dict {feature_name: value} ready for transform_single().
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = urllib.parse.urlparse("")

    # ── basic components ──────────────────────────────────────────────────────
    host = parsed.netloc.lower().split(":")[0]
    path_query = parsed.path + ("?" + parsed.query if parsed.query else "")
    full_url = url

    tld = _tld_from_url(parsed)
    host_parts = [p for p in host.split(".") if p]

    # ── URLLength ─────────────────────────────────────────────────────────────
    url_length = len(full_url)

    # ── DomainLength ──────────────────────────────────────────────────────────
    domain_length = len(host)

    # ── IsDomainIP ────────────────────────────────────────────────────────────
    ip_re = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    is_domain_ip = int(bool(ip_re.match(host)))

    # ── URLSimilarityIndex ────────────────────────────────────────────────────
    # Rough heuristic: shorter URL with common TLD → more similar to known good
    tld_score = TLD_PROB.get(tld, 0.25)
    url_similarity_index = max(0.0, min(100.0, tld_score * 100 - url_length * 0.05))

    # ── CharContinuationRate ──────────────────────────────────────────────────
    # Ratio of adjacent identical characters in the URL
    if len(full_url) > 1:
        runs = sum(1 for i in range(1, len(full_url)) if full_url[i] == full_url[i - 1])
        char_continuation = runs / len(full_url)
    else:
        char_continuation = 0.0

    # ── TLDLegitimateProb ─────────────────────────────────────────────────────
    tld_legitimate_prob = TLD_PROB.get(tld, 0.25)

    # ── URLCharProb (mean char frequency approximation) ───────────────────────
    # Proxy: ratio of alphanumeric chars (high ratio → more "normal" URL)
    alnum = sum(c.isalnum() for c in full_url)
    url_char_prob = alnum / max(len(full_url), 1)

    # ── TLDLength ─────────────────────────────────────────────────────────────
    tld_length = len(tld)

    # ── NoOfSubDomain ─────────────────────────────────────────────────────────
    # subdomains = total dots in host minus 1 (for domain + TLD)
    no_of_subdomain = max(0, len(host_parts) - 2)

    # ── Obfuscation ───────────────────────────────────────────────────────────
    obf_matches = OBFUSCATION_RE.findall(full_url)
    has_obfuscation = int(len(obf_matches) > 0)
    no_of_obfuscated_char = len(obf_matches)
    obfuscation_ratio = no_of_obfuscated_char / max(url_length, 1)

    # ── Character counts ──────────────────────────────────────────────────────
    letters  = sum(c.isalpha() for c in full_url)
    digits   = sum(c.isdigit() for c in full_url)
    equals   = full_url.count("=")
    qmarks   = full_url.count("?")
    amps     = full_url.count("&")
    specials = sum(1 for c in full_url if c in SPECIAL_CHARS)

    letter_ratio   = letters  / max(url_length, 1)
    digit_ratio    = digits   / max(url_length, 1)
    special_ratio  = specials / max(url_length, 1)

    # ── IsHTTPS ───────────────────────────────────────────────────────────────
    is_https = int(parsed.scheme.lower() == "https")

    # ── assemble feature dict ─────────────────────────────────────────────────
    lexical: dict[str, float] = {
        "URLLength":                  float(url_length),
        "DomainLength":               float(domain_length),
        "IsDomainIP":                 float(is_domain_ip),
        "URLSimilarityIndex":         float(url_similarity_index),
        "CharContinuationRate":       float(char_continuation),
        "TLDLegitimateProb":          float(tld_legitimate_prob),
        "URLCharProb":                float(url_char_prob),
        "TLDLength":                  float(tld_length),
        "NoOfSubDomain":              float(no_of_subdomain),
        "HasObfuscation":             float(has_obfuscation),
        "NoOfObfuscatedChar":         float(no_of_obfuscated_char),
        "ObfuscationRatio":           float(obfuscation_ratio),
        "NoOfLettersInURL":           float(letters),
        "LetterRatioInURL":           float(letter_ratio),
        "NoOfDegitsInURL":            float(digits),
        "DegitRatioInURL":            float(digit_ratio),
        "NoOfEqualsInURL":            float(equals),
        "NoOfQMarkInURL":             float(qmarks),
        "NoOfAmpersandInURL":         float(amps),
        "NoOfOtherSpecialCharsInURL": float(specials),
        "SpacialCharRatioInURL":      float(special_ratio),
        "IsHTTPS":                    float(is_https),
    }

    # Merge with content-based defaults
    features = {**CONTENT_DEFAULTS, **lexical}
    return features


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login?user=admin&pass=1234",
        "https://paypal-secure.verify-account.com/signin%20here",
    ]
    for u in test_urls:
        feats = extract(u)
        print(f"\n{u}")
        print(f"  IsHTTPS={feats['IsHTTPS']}  IsDomainIP={feats['IsDomainIP']}"
              f"  NoOfSubDomain={feats['NoOfSubDomain']}"
              f"  HasObfuscation={feats['HasObfuscation']}")
