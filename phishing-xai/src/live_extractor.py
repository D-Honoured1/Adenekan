"""
Live HTML-fetching feature extractor for the PhiUSIIL 50-feature schema.

Feature split
-------------
  22 lexical  — derived from the URL string alone; always available.
  28 content  — require fetching and parsing the HTML page.

Any content feature that cannot be extracted (page fetch fails, JS-only
dynamic content, etc.) falls back to the feature's training-set mean stored
in models/imputer.pkl, and its name is recorded in ExtractionResult.fallback_features.

Failure cases handled
---------------------
  timeout            → requests.exceptions.Timeout
  connection refused → requests.exceptions.ConnectionError
  SSL error          → requests.exceptions.SSLError
  non-HTML response  → Content-Type check after successful HTTP response
"""

from __future__ import annotations

import re
import time
import urllib.parse
from dataclasses import dataclass, field

import joblib
import requests
from bs4 import BeautifulSoup

# ── constants ─────────────────────────────────────────────────────────────────

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_FETCH_HEADERS: dict[str, str] = {
    "User-Agent":                _USER_AGENT,
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# TLD legitimacy priors (same as feature_extractor.py)
_TLD_PROB: dict[str, float] = {
    "com": 0.52, "org": 0.50, "net": 0.48, "edu": 0.65, "gov": 0.70,
    "co":  0.40, "io":  0.35, "uk":  0.45, "us":  0.45, "de": 0.45,
    "fr":  0.43, "jp":  0.44, "au":  0.46, "ca":  0.48,
}

_SOCIAL_DOMAINS = frozenset({
    "facebook.com", "fb.com", "twitter.com", "x.com", "instagram.com",
    "linkedin.com",  "youtube.com", "tiktok.com", "pinterest.com",
    "reddit.com",    "whatsapp.com", "telegram.org", "snapchat.com",
    "tumblr.com",    "discord.com",  "twitch.tv",
})

_SPECIAL_CHARS = set("!@#$%^&*()_+{}|:<>?`~[]\\;',./=")

_OBFUSCATION_RE = re.compile(r"%[0-9A-Fa-f]{2}")

_BANK_RE    = re.compile(r"\b(bank(?:ing)?|credit|debit|loan|mortgage|financ|invest|brokerage)\b", re.I)
_PAY_RE     = re.compile(r"\b(pay(?:ment)?|checkout|purchase|cart|billing|invoice|transfer)\b", re.I)
_CRYPTO_RE  = re.compile(r"\b(crypto|bitcoin|btc|ethereum|eth|nft|blockchain|wallet|(?<!\w)coin|defi)\b", re.I)
_COPY_RE    = re.compile(r"©|&copy;|\bcopyright\b", re.I)
_POPUP_RE   = re.compile(r"window\.open\s*\(", re.I)
_FAVICON_RELS = {"icon", "shortcut icon", "apple-touch-icon", "apple-touch-icon-precomposed"}


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Return value from :func:`extract`."""
    features:           dict[str, float]
    features_extracted: int          # number of features from live page (out of 50)
    fallback_features:  list[str]    # feature names that used training-mean fallback
    fetch_time_ms:      float
    fetch_error:        str | None   # non-None if page could not be fetched


# ── URL helpers ───────────────────────────────────────────────────────────────

def _host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return ""


def _tld(host: str) -> str:
    parts = host.split(".")
    return parts[-1] if parts else ""


def _domain_stem(host: str) -> str:
    """'www.uni-mainz.de' → 'uni-mainz'  (no www, no TLD)."""
    host = re.sub(r"^www\.", "", host)
    parts = host.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else host


def _is_external(href: str, base_host: str) -> bool:
    try:
        p = urllib.parse.urlparse(href)
        if not p.scheme or not p.netloc:
            return False
        return p.netloc.lower().split(":")[0] != base_host
    except Exception:
        return False


# ── score helpers ─────────────────────────────────────────────────────────────

def _domain_title_match(stem: str, title: str) -> float:
    """
    Percentage of domain token characters that appear as whole words in the title.

    Example: stem='uni-mainz', title='... universität mainz'
      tokens ['uni','mainz'], 'mainz' matched → 5/9 * 100 = 55.55
    """
    if not title:
        return 0.0
    d_tokens = re.findall(r"[a-z0-9]+", stem.lower())
    t_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    if not d_tokens:
        return 0.0
    total   = sum(len(t) for t in d_tokens)
    matched = sum(len(t) for t in d_tokens if t in t_tokens)
    return round(matched / total * 100, 2) if total else 0.0


def _url_title_match(url: str, title: str) -> float:
    """
    Percentage of meaningful URL token characters found as whole words in the title.
    Stop-tokens (http, www, com, …) are excluded.
    """
    if not title:
        return 0.0
    _STOP = {"https", "http", "www", "com", "org", "net", "html", "php", "asp", "htm"}
    u_tokens = [
        t for t in re.findall(r"[a-z0-9]+", url.lower())
        if t not in _STOP and len(t) > 2
    ]
    t_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    if not u_tokens:
        return 0.0
    total   = sum(len(t) for t in u_tokens)
    matched = sum(len(t) for t in u_tokens if t in t_tokens)
    return round(matched / total * 100, 2) if total else 0.0


# ── page fetch ────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 10) -> tuple[requests.Response | None, str | None]:
    """
    GET the URL with browser-like headers.

    Returns (response, None) on success, or (None, error_message) on any failure.
    Never raises.
    """
    try:
        resp = requests.get(
            url,
            headers=_FETCH_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None, f"Non-HTML response (Content-Type: {ct or 'not set'})"
        return resp, None

    except requests.exceptions.Timeout:
        return None, "Request timed out after 10 seconds"
    except requests.exceptions.SSLError as exc:
        return None, f"SSL certificate error: {str(exc)[:100]}"
    except requests.exceptions.ConnectionError as exc:
        msg = str(exc)[:100]
        if "refused" in msg.lower():
            return None, "Connection refused by server"
        return None, f"Connection error: {msg}"
    except Exception as exc:
        return None, f"Unexpected fetch error: {str(exc)[:100]}"


# ── robots.txt probe (separate HEAD request) ──────────────────────────────────

def _has_robots_txt(url: str) -> float:
    """HEAD /robots.txt with a short timeout; returns 1.0 if it exists."""
    try:
        p = urllib.parse.urlparse(url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        r = requests.head(
            robots_url,
            headers={"User-Agent": _USER_AGENT},
            timeout=3,
            allow_redirects=True,
        )
        return float(r.status_code == 200)
    except Exception:
        return 0.0


# ── lexical extraction (URL string only, no network) ─────────────────────────

def _lexical(url: str) -> dict[str, float]:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = urllib.parse.urlparse("")

    host = parsed.netloc.lower().split(":")[0]
    tld  = _tld(host)
    parts = [p for p in host.split(".") if p]

    url_len    = len(url)
    dom_len    = len(host)
    is_ip      = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host)))
    tld_prob   = _TLD_PROB.get(tld, 0.25)
    url_sim    = max(0.0, min(100.0, tld_prob * 100 - url_len * 0.05))

    runs = sum(1 for i in range(1, len(url)) if url[i] == url[i - 1])
    char_cont  = runs / url_len if url_len > 1 else 0.0
    url_char_p = sum(c.isalnum() for c in url) / max(url_len, 1)
    tld_len    = len(tld)
    n_sub      = max(0, len(parts) - 2)

    obf        = _OBFUSCATION_RE.findall(url)
    has_obf    = int(bool(obf))
    n_obf      = len(obf)
    obf_ratio  = n_obf / max(url_len, 1)

    letters    = sum(c.isalpha()  for c in url)
    digits     = sum(c.isdigit()  for c in url)
    specials   = sum(c in _SPECIAL_CHARS for c in url)
    equals     = url.count("=")
    qmarks     = url.count("?")
    amps       = url.count("&")
    is_https   = int(parsed.scheme.lower() == "https")

    return {
        "URLLength":                  float(url_len),
        "DomainLength":               float(dom_len),
        "IsDomainIP":                 float(is_ip),
        "URLSimilarityIndex":         float(url_sim),
        "CharContinuationRate":       float(char_cont),
        "TLDLegitimateProb":          float(tld_prob),
        "URLCharProb":                float(url_char_p),
        "TLDLength":                  float(tld_len),
        "NoOfSubDomain":              float(n_sub),
        "HasObfuscation":             float(has_obf),
        "NoOfObfuscatedChar":         float(n_obf),
        "ObfuscationRatio":           float(obf_ratio),
        "NoOfLettersInURL":           float(letters),
        "LetterRatioInURL":           float(letters / max(url_len, 1)),
        "NoOfDegitsInURL":            float(digits),
        "DegitRatioInURL":            float(digits / max(url_len, 1)),
        "NoOfEqualsInURL":            float(equals),
        "NoOfQMarkInURL":             float(qmarks),
        "NoOfAmpersandInURL":         float(amps),
        "NoOfOtherSpecialCharsInURL": float(specials),
        "SpacialCharRatioInURL":      float(specials / max(url_len, 1)),
        "IsHTTPS":                    float(is_https),
    }


# ── content extraction (requires fetched HTML) ────────────────────────────────

def _content(url: str, response: requests.Response, soup: BeautifulSoup) -> dict[str, float]:
    html      = response.text
    final_url = response.url          # URL after redirects
    base_host = _host(final_url)
    orig_host = _host(url)

    # ── line metrics ──────────────────────────────────────────────────────────
    lines            = html.splitlines()
    line_of_code     = float(len(lines))
    largest_line_len = float(max((len(l) for l in lines), default=0))

    # ── title ─────────────────────────────────────────────────────────────────
    title_tag  = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    has_title  = float(bool(title_text))

    stem               = _domain_stem(orig_host)
    domain_title_score = _domain_title_match(stem, title_text)
    url_title_score    = _url_title_match(url, title_text)

    # ── favicon ───────────────────────────────────────────────────────────────
    has_favicon = float(any(
        set(r.lower() for r in (tag.get("rel") or [])) & _FAVICON_RELS
        for tag in soup.find_all("link")
    ))

    # ── robots.txt (separate HEAD request) ────────────────────────────────────
    robots = _has_robots_txt(url)

    # ── responsive (viewport meta tag) ───────────────────────────────────────
    viewport     = soup.find("meta", attrs={"name": lambda n: n and n.lower() == "viewport"})
    is_responsive = float(bool(viewport))

    # ── redirect counts ───────────────────────────────────────────────────────
    n_redirect      = float(len(response.history))
    n_self_redirect = float(sum(
        1 for r in response.history if _host(r.url) == orig_host
    ))

    # ── meta description ─────────────────────────────────────────────────────
    desc_tag     = soup.find("meta", attrs={"name": lambda n: n and n.lower() == "description"})
    has_desc     = float(bool(desc_tag))

    # ── popups & iframes ─────────────────────────────────────────────────────
    n_popup  = float(len(_POPUP_RE.findall(html)))
    n_iframe = float(len(soup.find_all("iframe")))

    # ── external form submission ──────────────────────────────────────────────
    has_ext_form = float(any(
        _is_external(f.get("action", ""), base_host)
        for f in soup.find_all("form")
        if f.get("action", "").startswith("http")
    ))

    # ── social network links ──────────────────────────────────────────────────
    anchors     = soup.find_all("a", href=True)
    has_social  = float(any(
        any(sd in urllib.parse.urlparse(a["href"]).netloc.lower() for sd in _SOCIAL_DOMAINS)
        for a in anchors
        if a["href"].startswith("http")
    ))

    # ── form inputs ───────────────────────────────────────────────────────────
    inputs          = soup.find_all("input")
    has_submit      = float(
        any(i.get("type", "").lower() == "submit"  for i in inputs)
        or any(b.get("type", "submit").lower() == "submit" for b in soup.find_all("button"))
    )
    has_hidden      = float(any(i.get("type", "").lower() == "hidden"   for i in inputs))
    has_password    = float(any(i.get("type", "").lower() == "password" for i in inputs))

    # ── keyword flags (run on visible text) ──────────────────────────────────
    page_text  = soup.get_text(" ", strip=True)
    bank_flag  = float(bool(_BANK_RE.search(page_text)))
    pay_flag   = float(bool(_PAY_RE.search(page_text)))
    crypto_flag = float(bool(_CRYPTO_RE.search(page_text)))
    has_copy   = float(bool(_COPY_RE.search(html)))

    # ── resource counts ───────────────────────────────────────────────────────
    n_image = float(len(soup.find_all("img")))
    n_css   = float(len([
        t for t in soup.find_all("link")
        if "stylesheet" in (t.get("rel") or []) and t.get("href")
    ]))
    n_js    = float(len([t for t in soup.find_all("script") if t.get("src")]))

    # ── anchor link breakdown ─────────────────────────────────────────────────
    _EMPTY_HREFS = {"", "#", "javascript:void(0)", "javascript:;", "javascript:void(0);"}
    n_self = n_empty = n_ext = 0
    for a in anchors:
        href = a.get("href", "")
        if href.lower() in _EMPTY_HREFS or href.lower().startswith("javascript:"):
            n_empty += 1
        elif href.startswith("http") or href.startswith("//"):
            norm = href if href.startswith("http") else "https:" + href
            if _is_external(norm, base_host):
                n_ext += 1
            else:
                n_self += 1
        else:
            n_self += 1   # relative path → same site

    return {
        "LineOfCode":            line_of_code,
        "LargestLineLength":     largest_line_len,
        "HasTitle":              has_title,
        "DomainTitleMatchScore": domain_title_score,
        "URLTitleMatchScore":    url_title_score,
        "HasFavicon":            has_favicon,
        "Robots":                robots,
        "IsResponsive":          is_responsive,
        "NoOfURLRedirect":       n_redirect,
        "NoOfSelfRedirect":      n_self_redirect,
        "HasDescription":        has_desc,
        "NoOfPopup":             n_popup,
        "NoOfiFrame":            n_iframe,
        "HasExternalFormSubmit": has_ext_form,
        "HasSocialNet":          has_social,
        "HasSubmitButton":       has_submit,
        "HasHiddenFields":       has_hidden,
        "HasPasswordField":      has_password,
        "Bank":                  bank_flag,
        "Pay":                   pay_flag,
        "Crypto":                crypto_flag,
        "HasCopyrightInfo":      has_copy,
        "NoOfImage":             n_image,
        "NoOfCSS":               n_css,
        "NoOfJS":                n_js,
        "NoOfSelfRef":           float(n_self),
        "NoOfEmptyRef":          float(n_empty),
        "NoOfExternalRef":       float(n_ext),
    }


# ── public entry point ────────────────────────────────────────────────────────

def extract(
    url: str,
    models_dir: str = "models",
    fetch_timeout: int = 10,
) -> ExtractionResult:
    """
    Fetch *url* and extract all 50 PhiUSIIL features.

    - 22 lexical features are always extracted from the URL string.
    - 28 content features are extracted from the fetched HTML page.
    - Any feature that cannot be extracted falls back to its training-set mean
      (from models/imputer.pkl) and is listed in ExtractionResult.fallback_features.

    Parameters
    ----------
    url           : Full URL including scheme (e.g. "https://example.com").
    models_dir    : Directory containing feature_names.pkl and imputer.pkl.
    fetch_timeout : HTTP timeout in seconds (default 10).

    Returns
    -------
    ExtractionResult
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    feature_names: list[str] = joblib.load(f"{models_dir}/feature_names.pkl")
    imputer                   = joblib.load(f"{models_dir}/imputer.pkl")
    training_means: dict[str, float] = dict(zip(feature_names, imputer.statistics_))

    extracted: dict[str, float] = {}

    # --- lexical features (always available) ---------------------------------
    extracted.update(_lexical(url))

    # --- fetch page ----------------------------------------------------------
    t0 = time.perf_counter()
    response, fetch_error = _fetch(url, timeout=fetch_timeout)
    fetch_time_ms = (time.perf_counter() - t0) * 1000

    # --- content features (require HTML) -------------------------------------
    if response is not None:
        soup = BeautifulSoup(response.text, "lxml")
        extracted.update(_content(url, response, soup))

    # --- fill any missing features with training means -----------------------
    fallback: list[str] = []
    for feat in feature_names:
        if feat not in extracted:
            extracted[feat] = training_means[feat]
            fallback.append(feat)

    features_extracted = len(feature_names) - len(fallback)

    return ExtractionResult(
        features           = extracted,
        features_extracted = features_extracted,
        fallback_features  = fallback,
        fetch_time_ms      = round(fetch_time_ms, 1),
        fetch_error        = fetch_error,
    )


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.google.com"
    print(f"\nExtracting features for: {test_url}")
    result = extract(test_url)
    print(f"Fetch time       : {result.fetch_time_ms:.0f} ms")
    print(f"Fetch error      : {result.fetch_error or 'none'}")
    print(f"Features extracted: {result.features_extracted}/50")
    if result.fallback_features:
        print(f"Fallback features : {result.fallback_features}")
    print("\nAll features:")
    for k, v in result.features.items():
        flag = " [fallback]" if k in result.fallback_features else ""
        print(f"  {k:30s}: {v:.4f}{flag}")
