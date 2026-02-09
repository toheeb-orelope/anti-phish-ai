# FINAL extractor – ONLY features used during model training
import re
import tldextract
from urllib.parse import urlparse
from collections import Counter
import math

def calc_entropy(s: str):
    if not s:
        return 0.0
    p = [freq / len(s) for freq in Counter(s).values()]
    return -sum(pi * math.log2(pi) for pi in p)

def extract_features(url: str):
    p = urlparse(url)
    full = url.lower()

    ex = tldextract.extract(url)
    domain_only = ex.registered_domain.lower()

    features = {
        "url_length": len(full),
        "domain_length": len(domain_only),
        "num_dots": full.count("."),
        "num_hyphens": full.count("-"),
        "num_at": full.count("@"),
        "num_question": full.count("?"),
        "num_equals": full.count("="),
        "num_digits": sum(c.isdigit() for c in full),
        "num_subdirs": p.path.count("/"),
        "has_https": int(full.startswith("https://")),
        "entropy": calc_entropy(full),
        "is_shortened": int("bit.ly" in full or "tinyurl" in full or "goo.gl" in full),
        "has_ip": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain_only))),
        "ssl_days_valid": 0,
        "has_dns_record": 0,
        "domain_age_days": 0,
        "tld_enc": hash(ex.suffix) % 5000,
    }

    return features
