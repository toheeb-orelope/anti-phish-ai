# extract_features.py
import tldextract
import math
from urllib.parse import urlparse

SHORTENERS = {"bit.ly","goo.gl","t.co","tinyurl.com","ow.ly","is.gd","buff.ly"}

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    from collections import Counter
    counts = Counter(s)
    p = [c/len(s) for c in counts.values()]
    return -sum(pi*math.log2(pi) for pi in p)

def extract_features(url: str) -> dict:
    u = url.strip()
    parsed = urlparse(u)
    netloc = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""

    ext = tldextract.extract(u)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    subdomain = ext.subdomain or ""
    tld = ext.suffix or ""

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
    has_ip = 1 if any(c.isdigit() for c in netloc) and netloc.replace(".", "").replace(":", "").isdigit() else 0

    return {
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
