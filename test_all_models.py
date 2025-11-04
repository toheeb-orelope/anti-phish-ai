# test_individual_models.py
import json, pandas as pd, torch, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from run_xai import (
    extract_features,
    _TREE_MODELS,
    _DEEP_MODELS,
    tree_predict_prob,
    deep_predict_prob,
    get_tree_columns,
    DEVICE,
)
from urllib.parse import urlparse

# -------------------------------------------------
# URLs to test
# -------------------------------------------------
LEGIT = [
    "https://www.facebook.com/",
    "https://www.google.com/",
    "https://www.amazon.co.uk/",
    "https://www.paypal.com/uk/home",
]
PHISH = [
    "http://paypal-secure-login.verify-account123.com/login",
    "https://appleid-login-support.com/",
    "http://update-security-facebook.com/",
    "http://secure-google-auth.info/",
]


# -------------------------------------------------
# Helper
# -------------------------------------------------
def run_single_model(url: str, model_name: str):
    feats = extract_features(url)
    if model_name in ("rf", "xgb", "lgbm"):
        model = _TREE_MODELS[model_name]
        cols = get_tree_columns(model)
        x_row = pd.DataFrame([{k: float(feats.get(k, 0)) for k in cols}], columns=cols)
        prob = tree_predict_prob(model, x_row)
    elif model_name in ("cnn", "ffnn", "lstm"):
        model = _DEEP_MODELS[model_name]
        prob = deep_predict_prob(model, url)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return prob


# -------------------------------------------------
# Run
# -------------------------------------------------
results = []
for urlset, label in [(LEGIT, "Legit"), (PHISH, "Phish")]:
    for url in urlset:
        row = {"url": url, "expected": label}
        for name in ["rf", "xgb", "lgbm", "cnn", "ffnn", "lstm"]:
            try:
                row[name] = run_single_model(url, name)
            except Exception as e:
                row[name] = None
                print(f"[WARN] {name} failed on {url}: {e}")
        results.append(row)

df = pd.DataFrame(results)
df["tree_mean"] = df[["rf", "xgb", "lgbm"]].mean(axis=1)
df["deep_mean"] = df[["cnn", "ffnn", "lstm"]].mean(axis=1)
df["hybrid"] = 0.75 * df["tree_mean"] + 0.25 * df["deep_mean"]

print("\n=== Individual Model Probabilities ===")
print(df.round(3))
df.to_csv("model/indiv_model_test.csv", index=False)
print("\nSaved → model/indiv_model_test.csv")

# Quick sanity
print("\nAverage hybrid (legit):", df[df["expected"] == "Legit"]["hybrid"].mean())
print("Average hybrid (phish):", df[df["expected"] == "Phish"]["hybrid"].mean())

# Get heatmap visualization of the model probabilities
df = pd.read_csv("model/indiv_model_test.csv")
plt.figure(figsize=(10, 6))  # <-- Increase figure size
sns.heatmap(
    df.set_index("url")[["rf", "xgb", "lgbm", "cnn", "ffnn", "lstm", "hybrid"]],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Phishing Probability"},
)
plt.title("Individual Model Probabilities (Legit vs Phish)", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right")
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
