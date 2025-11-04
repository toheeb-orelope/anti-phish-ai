# xai_runner.py
import os
import re
import json
import joblib
import torch
import numpy as np
import pandas as pd
import logging
import torch.nn.functional as F
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="No feature names seen")


from xai_explain import make_plain_english
from extract_features import extract_features

# NOTE: make_plain_english imports/uses explain_tree_sample internally; we keep import here for clarity
from explain_tree_with_shap import explain_tree_sample

# Lightning models (your classes)
from phishin_train_cnn import LightningCNN
from phishin_nlp_lstm import LightningLSTM
from phishin_train_ffnn import LightningFFNN


# Dynamic threshold loading
def _load_threshold():
    try:
        with open("model/thresholds.json", "r") as f:
            t = json.load(f)
        # Prefer hybrid Youden's J, then hybrid max-F1, then 0.5
        if "hybrid" in t:
            return float(t["hybrid"].get("youden_j", t["hybrid"].get("max_f1", 0.5)))
    except Exception:
        pass
    return 0.5


THRESHOLD = _load_threshold()


def _load_model_threshold(name, default=0.5):
    try:
        with open("model/thresholds.json") as f:
            t = json.load(f)
        return float(t.get(name, {}).get("youden_j", default))
    except Exception:
        return default


# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RF_PATH = "model/rf_calibrated.pkl"
XGB_PATH = "model/xgb_calibrated.pkl"
LGBM_PATH = "model/lgbm_calibrated.pkl"
# CNN_CKPT = "models/cnn_best.ckpt"
LSTM_CKPT = "models/lstm_best.ckpt"
# FFNN_CKPT = "models/ffnn_best.ckpt"
CNN_CKPT = "models/cnn_lightning.ckpt"  # trained checkpoint
FFNN_CKPT = "models/ffnn_lightning.ckpt"  # trained checkpoint


MAX_URL_LEN = 2048  # basic abuse guard
URL_ALLOWED = re.compile(r"^[\x20-\x7E]+$")  # printable ASCII

# -------------------------
# Logging (sanitized)
# -------------------------
logging.basicConfig(
    filename="xai_logs.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def sanitize_url_for_log(u: str) -> str:
    """
    Redact query values and long token-like segments from path to avoid logging secrets.
    Example: https://a.com/p/VERY-LONG-TOKEN-123?key=abcd -> https://a.com/p/[REDACTED]?key=[REDACTED]
    """
    try:
        p = urlparse(u)
        # redact query values
        q = [(k, "[REDACTED]") for k, _ in parse_qsl(p.query, keep_blank_values=True)]
        redacted_query = urlencode(q)

        # redact long path segments (alnum >= 16)
        path_segs = []
        for seg in p.path.split("/"):
            if re.fullmatch(r"[A-Za-z0-9._~-]{16,}", seg):
                path_segs.append("[REDACTED]")
            else:
                path_segs.append(seg)
        redacted_path = "/".join(path_segs)

        return urlunparse((p.scheme, p.netloc, redacted_path, "", redacted_query, ""))
    except Exception:
        return "[UNPARSABLE_URL]"


# Test function to run only RF model (for debugging)
# ---------------------------------------------------------------------
# Individual model test helpers (with correct thresholds)
# ---------------------------------------------------------------------
def run_rf_only(url: str):
    feats = extract_features(url)
    ref_model = _TREE_MODELS["rf"]
    cols = get_tree_columns(ref_model)
    x_row = pd.DataFrame([{k: float(feats.get(k, 0)) for k in cols}], columns=cols)
    prob = tree_predict_prob(ref_model, x_row)
    RF_THRESHOLD = _load_model_threshold("rf", 0.5)
    verdict = "Phishing" if prob >= RF_THRESHOLD else "Legitimate"
    print(f"\nURL: {url}\nRF Probability: {prob:.3f} → {verdict}")


def run_xgb_only(url: str):
    feats = extract_features(url)
    model = _TREE_MODELS["xgb"]
    cols = get_tree_columns(model)
    x_row = pd.DataFrame([{k: float(feats.get(k, 0)) for k in cols}], columns=cols)
    prob = tree_predict_prob(model, x_row)
    XGB_THRESHOLD = _load_model_threshold("xgb", 0.5)
    verdict = "Phishing" if prob >= XGB_THRESHOLD else "Legitimate"
    print(f"\nURL: {url}\nXGB Probability: {prob:.3f} → {verdict}")


def run_lgbm_only(url: str):
    feats = extract_features(url)
    model = _TREE_MODELS["lgbm"]
    cols = get_tree_columns(model)
    x_row = pd.DataFrame([{k: float(feats.get(k, 0)) for k in cols}], columns=cols)
    prob = tree_predict_prob(model, x_row)
    LGBM_THRESHOLD = _load_model_threshold("lgbm", 0.5)
    verdict = "Phishing" if prob >= LGBM_THRESHOLD else "Legitimate"
    print(f"\nURL: {url}\nLGBM Probability: {prob:.3f} → {verdict}")


def run_cnn_only(url: str):
    model = _DEEP_MODELS.get("cnn")
    prob = deep_predict_prob(model, url)
    CNN_THRESHOLD = _load_model_threshold("cnn", 0.5)
    verdict = "Phishing" if prob >= CNN_THRESHOLD else "Legitimate"
    print(f"\nURL: {url}\nCNN Probability: {prob:.3f} → {verdict}")


def run_ffnn_only(url: str):
    model = _DEEP_MODELS.get("ffnn")
    prob = deep_predict_prob(model, url)
    FFNN_THRESHOLD = _load_model_threshold("ffnn", 0.5)
    verdict = "Phishing" if prob >= FFNN_THRESHOLD else "Legitimate"
    print(f"\nURL: {url}\nFFNN Probability: {prob:.3f} → {verdict}")


def run_lstm_only(url: str):
    model = _DEEP_MODELS.get("lstm")
    prob = deep_predict_prob(model, url)
    LSTM_THRESHOLD = _load_model_threshold("lstm", 0.5)
    verdict = "Phishing" if prob >= LSTM_THRESHOLD else "Legitimate"
    print(f"\nURL: {url}\nLSTM Probability: {prob:.3f} → {verdict}")


# -------------------------
# Simple input validation
# -------------------------
def validate_url(u: str) -> None:
    if not isinstance(u, str) or not u.strip():
        raise ValueError("Empty URL.")
    if len(u) > MAX_URL_LEN:
        raise ValueError("URL too long.")
    if not URL_ALLOWED.match(u):
        raise ValueError("URL contains unsupported characters.")
    p = urlparse(u)
    if p.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed.")
    if not p.netloc:
        raise ValueError("URL must include a domain (netloc).")


# -------------------------
# Model caches (load once)
# -------------------------
_TREE_MODELS = {}
_DEEP_MODELS = {}


def load_tree_model(path):
    return joblib.load(path) if os.path.exists(path) else None


# old version
"""
def load_deep_models():
    
    # Load LightningModule checkpoints with correct architectures.
    # NOTE: This returns *initialized models*; do NOT wrap them with torch.load again.
    models = {}
    if os.path.exists(CNN_CKPT):
        models["cnn"] = (
            LightningCNN.load_from_checkpoint(CNN_CKPT, map_location=DEVICE)
            .to(DEVICE)
            .eval()
        )
    if os.path.exists(LSTM_CKPT):
        models["lstm"] = (
            LightningLSTM.load_from_checkpoint(LSTM_CKPT, map_location=DEVICE)
            .to(DEVICE)
            .eval()
        )
    # if os.path.exists(LSTM_CKPT):
    #     models["lstm"] = (
    #         LightningLSTM.load_from_checkpoint(LSTM_CKPT, map_location="cpu")
    #         .to("cpu")
    #         .eval()
    #     )
    if os.path.exists(FFNN_CKPT):
        models["ffnn"] = (
            LightningFFNN.load_from_checkpoint(FFNN_CKPT, map_location=DEVICE)
            .to(DEVICE)
            .eval()
        )
    return models
"""


# new version: always load to CPU, move to GPU at inference time if available
def load_deep_models():
    """
    Load LightningModule checkpoints with correct architectures.
    Each model is placed on its safest/expected device:
      - CNN & FFNN → GPU (if available)
      - LSTM → CPU only (to avoid cuDNN RNN backward issues)
    """
    models = {}

    # CNN (GPU if available)
    if os.path.exists(CNN_CKPT):
        try:
            cnn_model = LightningCNN.load_from_checkpoint(CNN_CKPT, map_location=DEVICE)
            cnn_model = cnn_model.to(DEVICE).eval()
            models["cnn"] = cnn_model
            print("[INFO] CNN model loaded on", DEVICE)
        except Exception as e:
            print(f"[WARN] Could not load CNN model: {e}")

    # FFNN (GPU if available)
    if os.path.exists(FFNN_CKPT):
        try:
            ffnn_model = LightningFFNN.load_from_checkpoint(
                FFNN_CKPT, map_location=DEVICE
            )
            ffnn_model = ffnn_model.to(DEVICE).eval()
            models["ffnn"] = ffnn_model
            print("[INFO] FFNN model loaded on", DEVICE)
        except Exception as e:
            print(f"[WARN] Could not load FFNN model: {e}")

    # LSTM (CPU only, CuDNN disabled)
    if os.path.exists(LSTM_CKPT):
        try:
            lstm_model = LightningLSTM.load_from_checkpoint(
                LSTM_CKPT, map_location="cpu"
            )
            lstm_model = lstm_model.to("cpu").eval()
            models["lstm"] = lstm_model
            print("[INFO] LSTM model loaded on CPU (CuDNN disabled)")
        except Exception as e:
            print(f"[WARN] Could not load LSTM model: {e}")

    return models


def init_models_once():
    global _TREE_MODELS, _DEEP_MODELS
    if not _TREE_MODELS:
        _TREE_MODELS = {
            "rf": load_tree_model(RF_PATH),
            "xgb": load_tree_model(XGB_PATH),
            "lgbm": load_tree_model(LGBM_PATH),
        }
    if not _DEEP_MODELS:
        _DEEP_MODELS = load_deep_models()


# ---------------------------------------------------------
# Initialize all models immediately when module is loaded
# ---------------------------------------------------------
init_models_once()
print("[DEBUG] Models preloaded at import time:")
print("  Tree:", list(_TREE_MODELS.keys()))
print("  Deep:", list(_DEEP_MODELS.keys()))


# -------------------------
# Probability helpers
# -------------------------


def tree_predict_prob(model, x_row: pd.DataFrame) -> float:
    """Return calibrated probability for a single sample (row 0).

    Handles several saved forms:
      - Plain sklearn tree models with predict_proba
      - CalibratedClassifierCV wrappers (predict_proba works)
      - Dict bundles like {"xgb": base_model, "calibrator": LogisticRegression}
    """
    try:
        X = x_row  # keep as DataFrame for sklearn compatibility

        # Dict bundle: try to locate base model and optional calibrator
        if isinstance(model, dict):
            base = None
            # common keys to check for base model
            for k in ("xgb", "lgbm", "rf", "base", "model"):
                if k in model:
                    base = model[k]
                    break
            calibrator = model.get("calibrator") if isinstance(model, dict) else None

            if base is not None and hasattr(base, "predict_proba"):
                raw = base.predict_proba(X)[:, 1]
                if calibrator is not None and hasattr(calibrator, "predict_proba"):
                    # logistic calibrator trained on raw probs
                    cal = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
                    return float(cal[0])
                return float(raw[0])

        # CalibratedClassifierCV or similar should expose predict_proba
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0][1])

        # Fallbacks
        if hasattr(model, "predict"):
            p = model.predict(X)
            if isinstance(p, np.ndarray) and p.ndim == 2:
                return float(p[0][1])
            return float(p[0])
    except Exception as e:
        logging.warning(f"tree_predict_prob error: {e}")
    return THRESHOLD


# working version
# Old version
"""
def deep_predict_prob(model, url: str, max_len=200) -> float:
    if model is None:
        return THRESHOLD
    s = str(url)[:max_len].ljust(max_len)
    idxs = torch.tensor(
        [min(ord(c), 127) for c in s], dtype=torch.long, device=DEVICE
    ).unsqueeze(
        0
    )  # [1, seq]

    try:
        torch.backends.cudnn.enabled = False
        with torch.no_grad():
            out = model(idxs)
        torch.backends.cudnn.enabled = True

        if isinstance(out, (tuple, list)):
            out = out[0]
        # binary logit -> sigmoid
        if out.ndim == 0 or (out.ndim == 1 and out.numel() == 1):
            return float(torch.sigmoid(out).item())
        # [C] -> softmax
        if out.ndim == 1 and out.shape[0] == 2:
            return float(F.softmax(out, dim=0)[1].item())
        # [B, C]
        if out.ndim == 2:
            probs = F.softmax(out, dim=1)
            return (
                float(probs[0, 1].item())
                if probs.shape[1] > 1
                else float(probs[0, 0].item())
            )

    except Exception as e:
        logging.warning(f"deep_predict_prob error: {e}")
    return THRESHOLD
"""


def deep_predict_prob(model, url: str) -> float:
    if model is None:
        return 0.5
    s = str(url)
    try:
        with torch.no_grad():
            if isinstance(model, (LightningLSTM,)):
                # LSTM expects long input on CPU
                x = (
                    torch.tensor(
                        [min(ord(c), 127) for c in s[:200].ljust(200)], dtype=torch.long
                    )
                    .unsqueeze(0)
                    .to("cpu")
                )
                torch.backends.cudnn.enabled = False
                out = model(x)
                torch.backends.cudnn.enabled = True
            else:
                # CNN/FFNN expect float input on same device as model
                device = next(model.parameters()).device
                x = (
                    torch.tensor(
                        [ord(c) / 128 for c in s[:200].ljust(200)], dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                out = model(x)

            if isinstance(out, (tuple, list)):
                out = out[0]

            if out.ndim == 1:
                out = out.unsqueeze(0)
            p = (
                torch.sigmoid(out).flatten()[0].item()
                if out.shape[1] == 1
                else F.softmax(out, dim=1)[0, 1].item()
            )
            return float(p)
    except Exception as e:
        import logging

        logging.warning(f"deep_predict_prob error: {e}")
        return 0.5


def get_tree_columns(model):
    """Automatically get correct feature names for LightGBM/XGB/RF."""
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        # LightGBM-style
        return [f for f in model.feature_name_ if f not in ("", "default")]
    if hasattr(model, "feature_names_in_"):
        # sklearn-style (RandomForest, XGB)
        return list(model.feature_names_in_)
    # fallback (10 default features)

    return [
        "url_length",
        "domain_length",
        "num_dots",
        "num_hyphens",
        "num_at",
        "num_question",
        "num_equals",
        "num_digits",
        "num_subdirs",
        "has_https",
        "tld_enc",
    ]


# -------------------------
# Main entry
# -------------------------
# Working version
"""
def run_example(url: str):
    # Security: validate first
    validate_url(url)

    # Load models once
    init_models_once()

    # Prepare features/columns (must match training schema)
    # feats = extract_features(url)
    # tree_model = (
    #     _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    # )  # dynamically get the right feature set based on whichever model exists
    # ref_model = (
    #     _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    # )
    # tree_columns = get_tree_columns(ref_model)

    # x_row = pd.DataFrame([{k: feats.get(k, 0) for k in tree_columns}])[tree_columns]

    from sklearn.preprocessing import LabelEncoder

    # Load extracted features
    feats = extract_features(url)

    # Prepare DataFrame for tree-based models
    tree_model = (
        _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    )
    ref_model = (
        _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    )
    tree_columns = get_tree_columns(ref_model)

    # ----------------------------------------------------
    # Build feature DataFrame with guaranteed numeric dtype
    # ----------------------------------------------------
    x_row_dict = {k: float(feats.get(k, 0)) for k in tree_columns}
    x_row = pd.DataFrame([x_row_dict], columns=tree_columns).astype(float)

    print("\n[DEBUG] Feature types before model prediction:\n", x_row.dtypes)

    # Probabilities
    probs = {}
    if _TREE_MODELS.get("rf") is not None:
        probs["rf"] = tree_predict_prob(_TREE_MODELS["rf"], x_row)
    if _TREE_MODELS.get("xgb") is not None:
        probs["xgb"] = tree_predict_prob(_TREE_MODELS["xgb"], x_row)
    if _TREE_MODELS.get("lgbm") is not None:
        probs["lgbm"] = tree_predict_prob(_TREE_MODELS["lgbm"], x_row)

    probs["cnn"] = deep_predict_prob(_DEEP_MODELS.get("cnn"), url)
    probs["lstm"] = deep_predict_prob(_DEEP_MODELS.get("lstm"), url)
    probs["ffnn"] = deep_predict_prob(_DEEP_MODELS.get("ffnn"), url)

    # Pick a representative model for explanations
    tree_model = (
        _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    )
    deep_model = (
        _DEEP_MODELS.get("lstm") or _DEEP_MODELS.get("cnn") or _DEEP_MODELS.get("ffnn")
    )

    # XAI fusion -> plain English
    result = make_plain_english(
        url=url,
        probs=probs,
        tree_model=tree_model,
        tree_columns=tree_columns,
        deep_model=deep_model,
        max_reasons=4,
    )

    # Console view (dev)
    print("\n=== XAI RESULT ===")
    print("URL:", result["url"])
    print("Verdict:", result["verdict"])
    print("Confidence:", result["confidence"])
    print(
        "Model breakdown:",
        {k: round(v, 3) for k, v in result["model_breakdown"].items()},
    )
    print("Top reasons:")
    for i, r in enumerate(result["reasons"], 1):
        print(f" {i}. {r}")

    # Sanitized logging
    safe = sanitize_url_for_log(url)
    logging.info(f"{safe} -> {result['verdict']} ({result['confidence']})")

    # ----------------------------------------------------
    # Save ensemble probabilities for evaluation
    # ----------------------------------------------------
    try:
        ensemble_path = "models/ensemble_probs.npy"
        prob_values = np.array(
            list(result["model_breakdown"].values()), dtype=np.float32
        )

        # Append mode: load old, then stack new one
        if os.path.exists(ensemble_path):
            existing = np.load(ensemble_path, allow_pickle=True)
            if existing.ndim == 1:
                combined = np.vstack([existing, prob_values])
            else:
                combined = np.vstack([existing, prob_values])
        else:
            combined = np.expand_dims(prob_values, axis=0)

        np.save(ensemble_path, combined)
        print(f"✅ Ensemble probabilities saved to {ensemble_path}")
    except Exception as e:
        print(f"[WARN] Could not save ensemble probabilities: {e}")

    # API-friendly return
    return result  # FastAPI/Flask will JSON-serialize this
"""


# New
def run_example(url: str):
    global THRESHOLD
    # -------------------------
    # 1) Validate & init
    # -------------------------
    validate_url(url)
    init_models_once()

    # -------------------------
    # 2) Build tree features (numeric, ordered columns)
    # -------------------------
    feats = extract_features(url)

    # choose any available tree model to derive the correct column order
    ref_model = (
        _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    )
    tree_columns = get_tree_columns(ref_model)
    x_row_dict = {k: float(feats.get(k, 0)) for k in tree_columns}
    x_row = pd.DataFrame([x_row_dict], columns=tree_columns).astype(float)

    # print("\n[DEBUG] Feature types before model prediction:\n", x_row.dtypes)

    # -------------------------
    # 3) Collect probabilities (tree + deep)
    # -------------------------
    probs = {}

    # --- Tree models (safe by default) ---
    try:
        if _TREE_MODELS.get("rf") is not None:
            probs["rf"] = tree_predict_prob(_TREE_MODELS["rf"], x_row)
    except Exception as e:
        logging.warning(f"RF inference failed: {e}")
        probs["rf"] = THRESHOLD

    try:
        if _TREE_MODELS.get("xgb") is not None:
            probs["xgb"] = tree_predict_prob(_TREE_MODELS["xgb"], x_row)
    except Exception as e:
        logging.warning(f"XGB inference failed: {e}")
        probs["xgb"] = THRESHOLD

    try:
        if _TREE_MODELS.get("lgbm") is not None:
            probs["lgbm"] = tree_predict_prob(_TREE_MODELS["lgbm"], x_row)
    except Exception as e:
        logging.warning(f"LGBM inference failed: {e}")
        probs["lgbm"] = THRESHOLD

    # --- Deep models (handled separately) ---
    # Defaults (in case a model is missing or fails)
    probs.setdefault("cnn", THRESHOLD)
    probs.setdefault("ffnn", THRESHOLD)
    probs.setdefault("lstm", THRESHOLD)

    # Helper: float encoding for CNN/FFNN
    def _encode_float(url_str: str, max_len: int = 200):
        s = str(url_str)[:max_len].ljust(max_len)
        return torch.tensor([ord(c) / 128 for c in s], dtype=torch.float32).unsqueeze(0)

    # Helper: integer encoding for LSTM
    def _encode_long(url_str: str, max_len: int = 200):
        s = str(url_str)[:max_len].ljust(max_len)
        return torch.tensor([min(ord(c), 127) for c in s], dtype=torch.long).unsqueeze(
            0
        )

    # CNN (GPU if available)
    try:
        cnn_model = _DEEP_MODELS.get("cnn")
        if cnn_model is not None:
            cnn_model.eval()
            x = _encode_float(url)
            if DEVICE.type == "cuda":
                x = x.to(DEVICE)
                cnn_model.to(DEVICE)
            with torch.inference_mode():
                out = cnn_model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim in (0, 1):  # logit or [2]
                if out.ndim == 0 or (out.ndim == 1 and out.numel() == 1):
                    probs["cnn"] = float(torch.sigmoid(out).item())
                elif out.ndim == 1 and out.shape[0] == 2:
                    probs["cnn"] = float(F.softmax(out, dim=0)[1].item())
            elif out.ndim == 2:
                sm = F.softmax(out, dim=1)
                probs["cnn"] = (
                    float(sm[0, 1].item())
                    if sm.shape[1] > 1
                    else float(sm[0, 0].item())
                )
    except Exception as e:
        logging.warning(f"CNN inference failed: {e}")

    # FFNN (GPU if available)
    try:
        ffnn_model = _DEEP_MODELS.get("ffnn")
        if ffnn_model is not None:
            ffnn_model.eval()
            x = _encode_float(url)
            if DEVICE.type == "cuda":
                x = x.to(DEVICE)
                ffnn_model.to(DEVICE)
            with torch.inference_mode():
                out = ffnn_model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim in (0, 1):  # logit or [2]
                if out.ndim == 0 or (out.ndim == 1 and out.numel() == 1):
                    probs["ffnn"] = float(torch.sigmoid(out).item())
                elif out.ndim == 1 and out.shape[0] == 2:
                    probs["ffnn"] = float(F.softmax(out, dim=0)[1].item())
            elif out.ndim == 2:
                sm = F.softmax(out, dim=1)
                probs["ffnn"] = (
                    float(sm[0, 1].item())
                    if sm.shape[1] > 1
                    else float(sm[0, 0].item())
                )
    except Exception as e:
        logging.warning(f"FFNN inference failed: {e}")

    # LSTM (CPU-only, CuDNN disabled)
    try:
        lstm_model = _DEEP_MODELS.get("lstm")
        if lstm_model is not None:
            lstm_model.eval()
            x = _encode_long(url)  # integer indices
            with torch.inference_mode():
                # Hard-disable CuDNN for RNN to avoid "RNN backward" errors
                prev_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                lstm_model_cpu = lstm_model.to("cpu")
                x_cpu = x.to("cpu")
                # keep weights contiguous if available
                try:
                    if hasattr(lstm_model_cpu, "lstm") and hasattr(
                        lstm_model_cpu.lstm, "flatten_parameters"
                    ):
                        lstm_model_cpu.lstm.flatten_parameters()
                except Exception:
                    pass
                out = lstm_model_cpu(x_cpu)
                # restore flag
                torch.backends.cudnn.enabled = prev_cudnn

            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim in (0, 1):  # logit or [2]
                if out.ndim == 0 or (out.ndim == 1 and out.numel() == 1):
                    probs["lstm"] = float(torch.sigmoid(out).item())
                elif out.ndim == 1 and out.shape[0] == 2:
                    probs["lstm"] = float(F.softmax(out, dim=0)[1].item())
            elif out.ndim == 2:
                sm = F.softmax(out, dim=1)
                probs["lstm"] = (
                    float(sm[0, 1].item())
                    if sm.shape[1] > 1
                    else float(sm[0, 0].item())
                )
    except Exception as e:
        logging.warning(f"LSTM inference failed: {e}")

    # -------------------------
    # 4) Hybrid fusion (optional weights)
    # -------------------------
    tree_vals = [probs[k] for k in ("rf", "xgb", "lgbm") if k in probs]
    deep_vals = [probs[k] for k in ("cnn", "ffnn", "lstm") if k in probs]

    tree_mean = float(np.mean(tree_vals)) if tree_vals else THRESHOLD
    deep_mean = float(np.mean(deep_vals)) if deep_vals else THRESHOLD
    # Fixed hybrid weighting (original): 0.75 tree, 0.25 deep
    hybrid_score = 0.75 * tree_mean + 0.25 * deep_mean
    probs["hybrid"] = float(hybrid_score)

    # -------------------------
    # Trusted-domain override (production safeguard)
    # -------------------------
    # canonicalize domain / tld
    p = urlparse(url)
    domain_only = p.netloc.lower().split(":")[0]
    tld = domain_only.split(".")[-1] if "." in domain_only else domain_only

    # trusted patterns and tld list (extend as needed)
    TRUSTED_TLDS = {
        "edu",
        "ac",
        "gov",
        "mil",
        "nhs",
        "sch",
    }  # 'ac' catches ac.uk when used carefully
    TRUSTED_SUFFIXES = (
        ".ac.uk",
        ".edu",
        ".gov.uk",
        ".gov",
        ".nhs.uk",
        ".edu.au",
        ".ac.nz",
    )
    TRUSTED_DOMAINS = {"google.com", "bbc.co.uk"}  # add high-confidence exceptions here

    trusted_override = False
    # 1) exact domain whitelist
    if domain_only in TRUSTED_DOMAINS:
        trusted_override = True

    # 2) suffix-based rule
    for suf in TRUSTED_SUFFIXES:
        if domain_only.endswith(suf):
            trusted_override = True
            break

    # 3) tld-based fallback (less strict)
    if not trusted_override and tld in TRUSTED_TLDS:
        # only apply if tree_mean & deep_mean are not extremely high
        # this prevents override when model is very confident phishing
        if hybrid_score < 0.85:
            trusted_override = True

    if trusted_override:
        # Stronger damping: pull closer to 0.25 rather than halfway
        final_prob = 0.25 + 0.25 * hybrid_score
        logging.info(
            f"Trusted-domain override applied for {domain_only} "
            f"(hybrid {hybrid_score:.3f} -> final {final_prob:.3f})"
        )
    else:
        final_prob = float(hybrid_score)

    # write adjusted final to probs for XAI & return use
    probs["final"] = final_prob

    # === Set threshold (academic vs production mode) ===
    academic_threshold = THRESHOLD
    production_threshold = 0.6

    MODE = "production"  # or "academic"

    if MODE == "academic":
        THRESHOLD = academic_threshold
    else:
        THRESHOLD = production_threshold
        if trusted_override:
            THRESHOLD = 0.8  # require 80% certainty to flag trusted domains

    # -------------------------
    # 5) Pick models for XAI text
    #    Choose the strongest tree model for SHAP on this URL.
    # -------------------------
    tree_candidates = {k: _TREE_MODELS.get(k) for k in ("rf", "xgb", "lgbm")}
    best_tree_key = None
    best_tree_val = -1.0
    for k, m in tree_candidates.items():
        if m is not None and k in probs:
            if probs[k] >= best_tree_val:
                best_tree_val = probs[k]
                best_tree_key = k
    tree_model = (
        tree_candidates.get(best_tree_key)
        if best_tree_key
        else (
            _TREE_MODELS.get("lgbm")
            or _TREE_MODELS.get("xgb")
            or _TREE_MODELS.get("rf")
        )
    )
    # Ensure SHAP uses the correct column order for the selected tree model
    if tree_model is not None:
        tree_columns = get_tree_columns(tree_model)
    deep_model = (
        _DEEP_MODELS.get("lstm") or _DEEP_MODELS.get("cnn") or _DEEP_MODELS.get("ffnn")
    )

    # -------------------------
    # 6) Plain-English XAI
    # -------------------------
    """
    result = make_plain_english(
        url=url,
        probs={k: float(v) for k, v in probs.items()},
        tree_model=tree_model,
        tree_columns=tree_columns,
        deep_model=deep_model,
        max_reasons=4,
        threshold=THRESHOLD,
        final_prob_override=float(hybrid_score),
    )
    """

    result = make_plain_english(
        url=url,
        probs={k: float(v) for k, v in probs.items()},
        tree_model=tree_model,
        tree_columns=tree_columns,
        deep_model=deep_model,
        max_reasons=4,
        threshold=THRESHOLD,
        final_prob_override=float(final_prob),  # ✅ use overridden score
    )

    # Dev print
    """
    print("\n=== XAI RESULT ===")
    print("URL:", result["url"])
    print("Verdict:", result["verdict"])
    print("Confidence:", result["confidence"])
    print(
        "Model breakdown:",
        {k: round(v, 3) for k, v in result["model_breakdown"].items()},
    )
    print("Top reasons:")
    for i, r in enumerate(result["reasons"], 1):
        print(f" {i}. {r}")
    """
    # Sanitized log
    safe = sanitize_url_for_log(url)
    logging.info(f"{safe} -> {result['verdict']} ({result['confidence']})")

    # -------------------------
    # 7) Save ensemble probabilities (append)
    # -------------------------
    """
    try:
        ensemble_path = "models/ensemble_probs.npy"
        prob_values = np.array(
            list(result["model_breakdown"].values()), dtype=np.float32
        )
        if os.path.exists(ensemble_path):
            existing = np.load(ensemble_path, allow_pickle=True)
            combined = (
                np.vstack([existing, prob_values])
                if existing.ndim > 0
                else np.expand_dims(prob_values, 0)
            )
        else:
            combined = np.expand_dims(prob_values, axis=0)
        np.save(ensemble_path, combined)
        print(f"✅ Ensemble probabilities saved to {ensemble_path}")
    except Exception as e:
        print(f"[WARN] Could not save ensemble probabilities: {e}")
        """

    return result


# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    # test_url = "http://paypal-secure-login.verify-account123.com/login?user=abc&token=XYZ1234567890"
    test_url = "https://northampton.ac.uk/"

    try:
        out = run_example(test_url)
        print("\nJSON:", json.dumps(out, indent=2))
    except Exception as e:
        print("Error:", e)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
