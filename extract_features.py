"""Feature extraction for phishing URL inference.

This module combines:
- lexical URL features
- optional live SSL/DNS/WHOIS lookups
- offline trust metrics from a cached domain dataset
- TLD label encoding for tree-based models
"""

from __future__ import annotations

import math
import socket
import ssl
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import dns.resolver
import joblib
import pandas as pd
import tldextract
import whois

SHORTENERS = {"bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly"}
FULL_LOOKUP = False

DEFAULT_TRUST_FEATURES = {
    "popularity_score": 2_000_000,
    "is_top1m": 0,
    "is_majestic": 0,
    "trust_rank_norm": 0.0,
    "domain_age_days": 0,
}
DEFAULT_NETWORK_FEATURES = {
    "ssl_days_valid": 0,
    "has_dns_record": 0,
    "domain_age_days": 0,
}

CACHE_PATH = Path("data/processed/domain_cache.csv")
TLD_ENCODER_PATH = Path("model/tld_encoder.pkl")


def _load_trust_lookup() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        print("[WARN] domain_cache.csv not found - using zeroed trust features.")
        return {}

    cache_df = pd.read_csv(CACHE_PATH)
    print(f"[*] Loaded offline domain cache ({len(cache_df):,} domains)")

    if "domain" not in cache_df.columns:
        print("[WARN] domain_cache.csv is missing the 'domain' column.")
        return {}

    lookup: dict[str, dict] = {}
    for row in cache_df.to_dict(orient="records"):
        domain = str(row.get("domain", "")).strip().lower()
        if not domain:
            continue
        lookup[domain] = {
            "popularity_score": row.get(
                "popularity_score", DEFAULT_TRUST_FEATURES["popularity_score"]
            ),
            "is_top1m": row.get("is_top1m", DEFAULT_TRUST_FEATURES["is_top1m"]),
            "is_majestic": row.get(
                "is_majestic", DEFAULT_TRUST_FEATURES["is_majestic"]
            ),
            "trust_rank_norm": row.get(
                "trust_rank_norm", DEFAULT_TRUST_FEATURES["trust_rank_norm"]
            ),
            "domain_age_days": row.get(
                "domain_age_days", DEFAULT_TRUST_FEATURES["domain_age_days"]
            ),
        }
    return lookup


def _load_tld_encoder():
    if not TLD_ENCODER_PATH.exists():
        return None
    try:
        return joblib.load(TLD_ENCODER_PATH)
    except Exception:
        return None


TRUST_LOOKUP = _load_trust_lookup()
TLD_ENCODER = _load_tld_encoder()

print(
    f"[*] Feature Extraction Mode: {'FULL LOOKUP' if FULL_LOOKUP else 'FAST (Offline trust only)'}"
)


def shannon_entropy(value: str) -> float:
    """Return Shannon entropy for a string."""
    if not value:
        return 0.0
    counts = Counter(value)
    probabilities = [count / len(value) for count in counts.values()]
    return -sum(prob * math.log2(prob) for prob in probabilities)


def extract_ssl_dns_whois(domain: str) -> dict:
    """Fetch live network and registration metadata for a domain."""
    features = DEFAULT_NETWORK_FEATURES.copy()

    try:
        context = ssl.create_default_context()
        with context.wrap_socket(socket.socket(), server_hostname=domain) as sock:
            sock.settimeout(3)
            sock.connect((domain, 443))
            cert = sock.getpeercert()
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
        record = whois.whois(domain)
        created = record.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            features["domain_age_days"] = max((datetime.utcnow() - created).days, 0)
    except Exception:
        pass

    return features


def get_offline_trust_features(domain: str) -> dict:
    """Return cached trust metrics for a domain."""
    return TRUST_LOOKUP.get(domain.lower(), DEFAULT_TRUST_FEATURES).copy()


def encode_tld(tld: str) -> int:
    """Encode a TLD using the persisted label encoder when available."""
    if TLD_ENCODER is None:
        return 0
    try:
        return int(TLD_ENCODER.transform([tld])[0])
    except Exception:
        return 0


def extract_features(url: str) -> dict:
    """Build the feature dictionary expected by the trained models."""
    normalized_url = url.strip()
    parsed = urlparse(normalized_url)
    netloc = parsed.netloc
    path = parsed.path or ""

    extracted = tldextract.extract(normalized_url)
    domain = ".".join(part for part in [extracted.domain, extracted.suffix] if part)
    tld = extracted.suffix or ""

    features = {
        "url": normalized_url,
        "url_length": len(normalized_url),
        "domain_length": len(domain),
        "num_dots": normalized_url.count("."),
        "num_hyphens": normalized_url.count("-"),
        "num_at": normalized_url.count("@"),
        "num_question": normalized_url.count("?"),
        "num_equals": normalized_url.count("="),
        "num_digits": sum(char.isdigit() for char in normalized_url),
        "num_subdirs": path.count("/"),
        "has_https": int(parsed.scheme.lower() == "https"),
        "tld": tld,
        "entropy": shannon_entropy(normalized_url),
        "is_shortened": int(
            any(shortener in netloc.lower() for shortener in SHORTENERS)
        ),
        "has_ip": int(
            any(char.isdigit() for char in netloc)
            and netloc.replace(".", "").replace(":", "").isdigit()
        ),
    }

    if FULL_LOOKUP:
        domain_part = netloc.split(":", maxsplit=1)[0]
        features.update(
            extract_ssl_dns_whois(domain_part)
            if domain_part
            else DEFAULT_NETWORK_FEATURES
        )
    else:
        features.update(DEFAULT_NETWORK_FEATURES)

    features["tld_enc"] = encode_tld(tld)
    features.update(get_offline_trust_features(domain))

    # Older trained models expect duplicate *_x / *_y trust columns.
    for name in [
        "domain_age_days",
        "popularity_score",
        "is_top1m",
        "is_majestic",
        "trust_rank_norm",
    ]:
        features[f"{name}_x"] = features[name]
        features[f"{name}_y"] = features[name]
        features.pop(name, None)

    return features
