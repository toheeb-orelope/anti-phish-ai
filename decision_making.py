# decision_making.py
def feature_reason(name, value, shap_sign):
    # shap_sign > 0 means pushes toward "phish"
    if name == "has_https":
        return ("Uses HTTPS", "benign") if value == 1 else ("No HTTPS detected", "phish")
    if name == "url_length":
        if value > 90: return ("Unusually long URL", "phish")
    if name == "num_hyphens":
        if value >= 3: return ("Many hyphens in domain/path", "phish")
    if name == "num_at":
        if value >= 1: return ("Contains '@' which can hide real destination", "phish")
    if name == "num_question":
        if value >= 2: return ("Many query parameters", "phish")
    if name == "num_equals":
        if value >= 2: return ("Complex parameter string", "phish")
    if name == "num_digits":
        if value >= 10: return ("Unusually high number of digits", "phish")
    if name == "num_subdirs":
        if value >= 5: return ("Deeply nested path", "phish")
    if name == "tld":
        risky = {"xyz","top","gq","tk","cf"}
        if str(value).lower() in risky: return (f"TLD .{value} often abused", "phish")
    if name == "has_ip":
        if value == 1: return ("Domain looks like an IP address", "phish")
    if name == "is_shortened":
        if value == 1: return ("URL shortener used", "phish")

    # Fallback based on SHAP sign
    return (f"{name} strongly influenced the decision", "phish" if shap_sign>0 else "benign")


def substr_reason(token, sign):
    token = token.strip()
    if not token:
        return None
    suspicious_keywords = {"login","verify","update","secure","account","pay","paypal","bank","confirm"}
    if token.lower() in suspicious_keywords:
        return (f"Suspicious keyword “{token}” in URL", "phish")
    if "." in token and len(token) > 20:
        return (f"Very long domain segment “{token[:25]}…“", "phish")
    return (f"Model focused on “{token}”", "phish" if sign>0 else "benign")
