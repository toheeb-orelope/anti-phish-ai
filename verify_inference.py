"""
verify_inference.py
Quick diagnostic to confirm that all models produce valid (0–1) probabilities
and no longer show systematic bias toward phishing or legitimate.
"""

import joblib, torch, numpy as np, pandas as pd
from extract_features import extract_features
from run_xai import get_tree_columns
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# URLs to test
# ----------------------------
TEST_URLS = [
    # Legitimate
    "https://northampton.ac.uk/",
    "https://www.google.com/",
    "https://www.bbc.co.uk/news",
    # Known or suspicious phishing-style
    "http://paypal-login-secure-update.com/",
    "http://verify-bank-account-security.net/",
]

# ----------------------------
# Load models
# ----------------------------
print("[*] Loading calibrated tree models ...")
rf = joblib.load("model/rf_calibrated.pkl")
lgbm = joblib.load("model/lgbm_calibrated.pkl")
xgb_bundle = joblib.load("model/xgb_calibrated.pkl")
xgb, xgb_cal = xgb_bundle["xgb"], xgb_bundle["calibrator"]

print("[*] Loading deep models ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
cnn = torch.load("models/cnn_lightning.ckpt", map_location=device)
ffnn = torch.load("models/ffnn_lightning.ckpt", map_location=device)
lstm = torch.load("models/lstm_lightning.ckpt", map_location=device)


# ----------------------------
# Helper: predict using tree
# ----------------------------
def build_tree_row(model, feats_dict):
    cols = get_tree_columns(model)
    row = {k: float(feats_dict.get(k, 0)) for k in cols}
    x_row = pd.DataFrame([row], columns=cols).astype(float)
    return x_row


def predict_tree(model, url):
    feats = extract_features(url)
    x_row = build_tree_row(model, feats)
    prob = model.predict_proba(x_row)[0, 1]
    return float(prob)


def predict_xgb(xgb, cal, url):
    feats = extract_features(url)
    x_row = build_tree_row(xgb, feats)
    raw = xgb.predict_proba(x_row)[:, 1]
    prob = cal.predict_proba(raw.reshape(-1, 1))[:, 1][0]
    return float(prob)


# ----------------------------
# Helper: predict using deep models
# ----------------------------
def encode_url(url, max_len=200):
    s = str(url)[:max_len].ljust(max_len)
    arr = np.array([ord(c) / 128 for c in s], dtype=np.float32)
    return torch.tensor(arr).unsqueeze(0).to(device)


def predict_deep(model, url):
    model.eval()
    with torch.no_grad():
        x = encode_url(url)
        y = torch.sigmoid(model(x)).item()
    return float(y)


# ----------------------------
# Run diagnostics
# ----------------------------
print("\n================= MODEL DIAGNOSTIC =================")
for url in TEST_URLS:
    print(f"\n🔗 {url}")
    try:
        rf_p = predict_tree(rf, url)
        lgb_p = predict_tree(lgbm, url)
        xgb_p = predict_xgb(xgb, xgb_cal, url)
        cnn_p = predict_deep(cnn, url)
        ffnn_p = predict_deep(ffnn, url)
        lstm_p = predict_deep(lstm, url)

        all_probs = [rf_p, lgb_p, xgb_p, cnn_p, ffnn_p, lstm_p]
        verdicts = ["Phishing" if p >= 0.5 else "Legitimate" for p in all_probs]

        print(
            f"RF={rf_p:.3f} ({verdicts[0]}) | "
            f"LGBM={lgb_p:.3f} ({verdicts[1]}) | "
            f"XGB={xgb_p:.3f} ({verdicts[2]}) | "
            f"CNN={cnn_p:.3f} ({verdicts[3]}) | "
            f"FFNN={ffnn_p:.3f} ({verdicts[4]}) | "
            f"LSTM={lstm_p:.3f} ({verdicts[5]})"
        )

        # Sanity check
        if any(p < 0 or p > 1 for p in all_probs):
            print("⚠️  Warning: probability outside [0,1]")
        elif all(p > 0.8 for p in all_probs) or all(p < 0.2 for p in all_probs):
            print("⚠️  Potential bias detected (all models extreme).")
        else:
            print("✅  Probabilities within expected range.")
    except Exception as e:
        print(f"❌ Error processing {url}: {e}")

print("\n[DONE] Diagnostics complete.")
