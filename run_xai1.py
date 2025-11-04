# =====================================
# run_xai_hybrid_rf_cnn_lstm.py
# =====================================
import os, json, joblib, logging, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from urllib.parse import urlparse
from extract_features import extract_features
from xai_explain import make_plain_english
from explain_tree_with_shap import explain_tree_sample
from phishin_train_cnn import LightningCNN
from phishin_nlp_lstm import LightningLSTM

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.52  # ↑ slightly higher to reduce false positives

RF_PATH = "model/rf_calibrated.pkl"
CNN_CKPT = "models/cnn_lightning.ckpt"
LSTM_CKPT = "models/lstm_best.ckpt"

logging.basicConfig(filename="xai_logs.log", level=logging.INFO)


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def load_rf():
    return joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None


def load_cnn():
    if os.path.exists(CNN_CKPT):
        model = LightningCNN.load_from_checkpoint(CNN_CKPT, map_location=DEVICE)
        return model.to(DEVICE).eval()
    return None


def load_lstm():
    if os.path.exists(LSTM_CKPT):
        model = LightningLSTM.load_from_checkpoint(LSTM_CKPT, map_location="cpu")
        return model.to("cpu").eval()
    return None


RF_MODEL = load_rf()
CNN_MODEL = load_cnn()
LSTM_MODEL = load_lstm()

print(
    "[INFO] Models loaded:",
    {
        "rf": RF_MODEL is not None,
        "cnn": CNN_MODEL is not None,
        "lstm": LSTM_MODEL is not None,
    },
)


# ------------------------------------------------------------
# Deep encoders
# ------------------------------------------------------------
def encode_float(url, max_len=200):
    s = str(url)[:max_len].ljust(max_len)
    return torch.tensor([ord(c) / 128 for c in s], dtype=torch.float32).unsqueeze(0)


def encode_long(url, max_len=200):
    s = str(url)[:max_len].ljust(max_len)
    return torch.tensor([min(ord(c), 127) for c in s], dtype=torch.long).unsqueeze(0)


# ------------------------------------------------------------
# Probability helpers
# ------------------------------------------------------------
def tree_prob(model, feats_df):
    try:
        return float(model.predict_proba(feats_df)[0][1])
    except Exception as e:
        logging.warning(f"RF inference failed: {e}")
        return THRESHOLD


def cnn_prob(model, url):
    try:
        x = encode_float(url)
        if DEVICE.type == "cuda":
            x = x.to(DEVICE)
            model.to(DEVICE)
        with torch.inference_mode():
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim in (0, 1):
            return (
                float(torch.sigmoid(out).item())
                if out.numel() == 1
                else float(F.softmax(out, dim=0)[1].item())
            )
        sm = F.softmax(out, dim=1)
        return float(sm[0, 1].item())
    except Exception as e:
        logging.warning(f"CNN error: {e}")
        return THRESHOLD


def lstm_prob(model, url):
    try:
        x = encode_long(url)
        model_cpu = model.to("cpu")
        with torch.inference_mode():
            torch.backends.cudnn.enabled = False
            out = model_cpu(x)
            torch.backends.cudnn.enabled = True
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim in (0, 1):
            return (
                float(torch.sigmoid(out).item())
                if out.numel() == 1
                else float(F.softmax(out, dim=0)[1].item())
            )
        sm = F.softmax(out, dim=1)
        return float(sm[0, 1].item())
    except Exception as e:
        logging.warning(f"LSTM error: {e}")
        return THRESHOLD


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def run_example(url: str):
    # 1️⃣ Extract numeric features for RF
    feats = extract_features(url)
    cols = getattr(RF_MODEL, "feature_names_in_", list(feats.keys()))
    x_row = pd.DataFrame([{k: float(feats.get(k, 0)) for k in cols}])[cols]

    # 2️⃣ Get probabilities from all three models
    probs = {
        "rf": tree_prob(RF_MODEL, x_row),
        "cnn": cnn_prob(CNN_MODEL, url),
        "lstm": lstm_prob(LSTM_MODEL, url),
    }

    # 3️⃣ Adaptive weighted fusion
    hybrid = 0.6 * probs["rf"] + 0.25 * probs["cnn"] + 0.15 * probs["lstm"]
    probs["hybrid"] = hybrid

    # 4️⃣ Verdict
    verdict = "phishing" if hybrid >= THRESHOLD else "legitimate"

    # 5️⃣ Explainable reasoning
    result = make_plain_english(
        url=url,
        probs=probs,
        tree_model=RF_MODEL,
        tree_columns=cols,
        deep_model=LSTM_MODEL,
        max_reasons=4,
    )
    result["verdict"] = verdict
    result["confidence"] = round(hybrid, 3)

    # Log sanitized result
    logging.info(f"{url} -> {verdict} ({result['confidence']})")

    return result


# ------------------------------------------------------------
# CLI test
# ------------------------------------------------------------
# if __name__ == "__main__":
#     test_url = "https://paypal.com/security-update?session=ab12cd34"
#     out = run_example(test_url)
#     print(json.dumps(out, indent=2))
