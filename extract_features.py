# =====================================
# extract_features.py
# Lexical + SSL/DNS/WHOIS feature extractor
# =====================================
# Old feature extraction function: with TLD encoding at the end and domain extraction
"""
import tldextract
import math
from urllib.parse import urlparse
import ssl, socket, dns.resolver, whois
from datetime import datetime
from collections import Counter

SHORTENERS = {"bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly"}
FULL_LOOKUP = False
print(
    f"[*] Feature Extraction Mode: {'FULL LOOKUP' if FULL_LOOKUP else 'FAST (No SSL/DNS/WHOIS)'}"
)


# -----------------------------
# Helper: Shannon entropy
# -----------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    p = [c / len(s) for c in counts.values()]
    return -sum(pi * math.log2(pi) for pi in p)


# -----------------------------
# Helper: SSL, DNS, WHOIS extraction
# -----------------------------
def extract_ssl_dns_whois(domain: str) -> dict:
    features = {"ssl_days_valid": 0, "has_dns_record": 0, "domain_age_days": 0}

    # SSL validity
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            cert = s.getpeercert()
            not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
            features["ssl_days_valid"] = (not_after - datetime.utcnow()).days
    except Exception:
        pass

    # DNS record
    try:
        dns.resolver.resolve(domain, "A")
        features["has_dns_record"] = 1
    except Exception:
        pass

    # WHOIS domain age
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):  # sometimes list
            created = created[0]
        if created:
            age = (datetime.utcnow() - created).days
            features["domain_age_days"] = max(age, 0)
    except Exception:
        pass

    return features


# -----------------------------
# Main extractor
# -----------------------------
def extract_features(url: str) -> dict:
    u = url.strip()
    parsed = urlparse(u)
    netloc = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""

    ext = tldextract.extract(u)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    tld = ext.suffix or ""

    # Lexical
    url_length = len(u)
    domain_length = len(domain)
    num_dots = u.count(".")
    num_hyphens = u.count("-")
    num_at = u.count("@")
    num_question = u.count("?")
    num_equals = u.count("=")
    num_digits = sum(ch.isdigit() for ch in u)
    num_subdirs = path.count("/")
    has_https = 1 if parsed.scheme.lower() == "https" else 0
    entropy = shannon_entropy(u)
    is_shortened = 1 if any(s in netloc.lower() for s in SHORTENERS) else 0
    has_ip = (
        1
        if any(c.isdigit() for c in netloc)
        and netloc.replace(".", "").replace(":", "").isdigit()
        else 0
    )

    # Combine lexical + host-based
    features = {
        "url": u,
        "url_length": url_length,
        "domain_length": domain_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at": num_at,
        "num_question": num_question,
        "num_equals": num_equals,
        "num_digits": num_digits,
        "num_subdirs": num_subdirs,
        "has_https": has_https,
        "tld": tld,
        "entropy": entropy,
        "is_shortened": is_shortened,
        "has_ip": has_ip,
    }

    # Merge with SSL/DNS/WHOIS features

    if FULL_LOOKUP:
        try:
            domain_part = netloc.split(":")[0]
            features.update(extract_ssl_dns_whois(domain_part))
        except Exception:
            features.update(
                {"ssl_days_valid": 0, "has_dns_record": 0, "domain_age_days": 0}
            )
    else:
        # Skip network lookups completely
        features.update(
            {"ssl_days_valid": 0, "has_dns_record": 0, "domain_age_days": 0}
        )

        # ✅ Add TLD encoding here (before returning)
    import joblib, os

    if os.path.exists("model/tld_encoder.pkl"):
        try:
            le = joblib.load("model/tld_encoder.pkl")
            features["tld_enc"] = int(le.transform([tld])[0])
        except Exception:
            features["tld_enc"] = 0
    else:
        features["tld_enc"] = 0

    return features
"""


# New feature extraction function complete with offline trust metrics
# =====================================
# extract_features.py
# Lexical + Offline Trust (Majestic + Top1M)
# =====================================
import os, math, ssl, socket, dns.resolver, whois, joblib
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
from collections import Counter
import tldextract

SHORTENERS = {"bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly"}
FULL_LOOKUP = False
print(
    f"[*] Feature Extraction Mode: {'FULL LOOKUP' if FULL_LOOKUP else 'FAST (Offline trust only)'}"
)

# -----------------------------
# Load offline trust cache
# -----------------------------
CACHE_PATH = "data/processed/domain_cache.csv"
if os.path.exists(CACHE_PATH):
    cache_df = pd.read_csv(CACHE_PATH)
    print(f"[*] Loaded offline domain cache ({len(cache_df):,} domains)")
else:
    cache_df = pd.DataFrame()
    print("[WARN] domain_cache.csv not found — using zeros as fallback.")


# -----------------------------
# Helper: Shannon entropy
# -----------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    p = [c / len(s) for c in counts.values()]
    return -sum(pi * math.log2(pi) for pi in p)


# -----------------------------
# Helper: SSL/DNS/WHOIS (optional)
# -----------------------------
def extract_ssl_dns_whois(domain: str) -> dict:
    features = {"ssl_days_valid": 0, "has_dns_record": 0, "domain_age_days": 0}
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            cert = s.getpeercert()
            not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
            features["ssl_days_valid"] = (not_after - datetime.utcnow()).days
    except Exception:
        pass

    try:
        dns.resolver.resolve(domain, "A")
        features["has_dns_record"] = 1
    except Exception:
        pass

    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            features["domain_age_days"] = max((datetime.utcnow() - created).days, 0)
    except Exception:
        pass

    return features


# -----------------------------
# Helper: Offline trust lookup
# -----------------------------
def get_offline_trust_features(domain: str) -> dict:
    if cache_df.empty:
        return {
            "popularity_score": 2_000_000,
            "is_top1m": 0,
            "is_majestic": 0,
            "trust_rank_norm": 0.0,
            "domain_age_days": 0,
        }
    row = cache_df.loc[cache_df["domain"].str.lower() == domain.lower()]
    if not row.empty:
        r = row.iloc[0]
        return {
            "popularity_score": r.get("popularity_score", 2_000_000),
            "is_top1m": r.get("is_top1m", 0),
            "is_majestic": r.get("is_majestic", 0),
            "trust_rank_norm": r.get("trust_rank_norm", 0.0),
            "domain_age_days": r.get("domain_age_days", 0),
        }
    return {
        "popularity_score": 2_000_000,
        "is_top1m": 0,
        "is_majestic": 0,
        "trust_rank_norm": 0.0,
        "domain_age_days": 0,
    }


# -----------------------------
# Main extractor
# -----------------------------
def extract_features(url: str) -> dict:
    u = url.strip()
    parsed = urlparse(u)
    netloc = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""

    ext = tldextract.extract(u)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    tld = ext.suffix or ""

    # Lexical
    url_length = len(u)
    domain_length = len(domain)
    num_dots = u.count(".")
    num_hyphens = u.count("-")
    num_at = u.count("@")
    num_question = u.count("?")
    num_equals = u.count("=")
    num_digits = sum(ch.isdigit() for ch in u)
    num_subdirs = path.count("/")
    has_https = 1 if parsed.scheme.lower() == "https" else 0
    entropy = shannon_entropy(u)
    is_shortened = 1 if any(s in netloc.lower() for s in SHORTENERS) else 0
    has_ip = (
        1
        if any(c.isdigit() for c in netloc)
        and netloc.replace(".", "").replace(":", "").isdigit()
        else 0
    )

    features = {
        "url": u,
        "url_length": url_length,
        "domain_length": domain_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at": num_at,
        "num_question": num_question,
        "num_equals": num_equals,
        "num_digits": num_digits,
        "num_subdirs": num_subdirs,
        "has_https": has_https,
        "tld": tld,
        "entropy": entropy,
        "is_shortened": is_shortened,
        "has_ip": has_ip,
    }

    if FULL_LOOKUP:
        try:
            domain_part = netloc.split(":")[0]
            features.update(extract_ssl_dns_whois(domain_part))
        except Exception:
            features.update(
                {"ssl_days_valid": 0, "has_dns_record": 0, "domain_age_days": 0}
            )
    else:
        features.update(
            {"ssl_days_valid": 0, "has_dns_record": 0, "domain_age_days": 0}
        )

    # Add encoded TLD
    if os.path.exists("model/tld_encoder.pkl"):
        try:
            le = joblib.load("model/tld_encoder.pkl")
            features["tld_enc"] = int(le.transform([tld])[0])
        except Exception:
            features["tld_enc"] = 0
    else:
        features["tld_enc"] = 0

    # Merge offline trust metrics (Majestic + Top1M)
    features.update(get_offline_trust_features(domain))

    # -------------------------------------------------------
    # Compatibility patch for old trained models (x/y suffix)
    # -------------------------------------------------------
    """
    if "domain_age_days" in features:
        features["domain_age_days_x"] = features["domain_age_days_y"] = features[
            "domain_age_days"
        ]
    if "popularity_score" in features:
        features["popularity_score_x"] = features["popularity_score_y"] = features[
            "popularity_score"
        ]
    if "is_top1m" in features:
        features["is_top1m_x"] = features["is_top1m_y"] = features["is_top1m"]
    if "is_majestic" in features:
        features["is_majestic_x"] = features["is_majestic_y"] = features["is_majestic"]
    if "trust_rank_norm" in features:
        features["trust_rank_norm_x"] = features["trust_rank_norm_y"] = features[
            "trust_rank_norm"
        ]
    """
    # ✅ Produce only the columns your trained models expect
    for f in [
        "domain_age_days",
        "popularity_score",
        "is_top1m",
        "is_majestic",
        "trust_rank_norm",
    ]:
        features[f + "_x"] = features[f + "_y"] = features[f]
        features.pop(f, None)

    return features
