# =====================================================
# run_xai.py - Clean, Production-Ready XAI Runner
# =====================================================

import os
import re
import json
import joblib
import torch
import numpy as np
import pandas as pd
import logging
import torch.nn.functional as F
from time import perf_counter
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="No feature names seen")

from xai_explain import make_plain_english
from tempo import extract_features
from explain_tree_with_shap import explain_tree_sample  # used inside XAI
from phishin_train_cnn import LightningCNN
from phishin_nlp_lstm import LightningLSTM
from phishin_train_ffnn import LightningFFNN

# -------------------------
# Threshold handling
# -------------------------


def _load_threshold():
    """
    Load academic baseline threshold from model/thresholds.json if present.
    Used only for 'academic' mode; production overrides to 0.85.
    """
    try:
        with open("model/thresholds.json", "r") as f:
            t = json.load(f)
        if "hybrid" in t:
            return float(t["hybrid"].get("youden_j", t["hybrid"].get("max_f1", 0.5)))
    except Exception:
        pass
    return 0.5


THRESHOLD_ACADEMIC = _load_threshold()
THRESHOLD_PRODUCTION = 0.85  # ← main decision threshold in production
MODE = "production"  # "academic" or "production"

# -------------------------
# Devices / config
# -------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RF_PATH = "model/rf_calibrated.pkl"
XGB_PATH = "model/xgb_calibrated.pkl"
LGBM_PATH = "model/lgbm_calibrated.pkl"

CNN_CKPT = "models/cnn_lightning.ckpt"
FFNN_CKPT = "models/ffnn_lightning.ckpt"
LSTM_CKPT = "models/lstm_best.ckpt"

MAX_URL_LEN = 2048
URL_ALLOWED = re.compile(r"^[\x20-\x7E]+$")

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
    Example: https://a.com/p/VERY-LONG-TOKEN-123?key=abcd
    -> https://a.com/p/[REDACTED]?key=[REDACTED]
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


# -------------------------
# Input validation
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
# Model caches
# -------------------------

_TREE_MODELS = {}
_DEEP_MODELS = {}


def load_tree_model(path: str):
    return joblib.load(path) if os.path.exists(path) else None


def load_deep_models():
    """
    Load LightningModule checkpoints with correct architectures.
    CNN & FFNN -> GPU if available.
    LSTM      -> CPU only (to avoid cuDNN RNN issues).
    """
    models = {}

    # CNN
    if os.path.exists(CNN_CKPT):
        try:
            cnn_model = LightningCNN.load_from_checkpoint(CNN_CKPT, map_location=DEVICE)
            cnn_model = cnn_model.to(DEVICE).eval()
            models["cnn"] = cnn_model
            print("[INFO] CNN model loaded on", DEVICE)
        except Exception as e:
            print(f"[WARN] Could not load CNN model: {e}")

    # FFNN
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

    # LSTM (CPU only)
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


# Initialize immediately
init_models_once()
print("[DEBUG] Models preloaded at import time:")
print("  Tree:", list(_TREE_MODELS.keys()))
print("  Deep:", list(_DEEP_MODELS.keys()))


# -------------------------
# Tree / deep prediction
# -------------------------


def tree_predict_prob(model, x_row: pd.DataFrame, fallback: float) -> float:
    """Safe probability prediction for tree-based models."""
    try:
        X = x_row
        # Dict bundle (e.g. {'xgb': base_model, 'calibrator': LogisticRegression})
        if isinstance(model, dict):
            base = None
            for k in ("xgb", "lgbm", "rf", "base", "model"):
                if k in model:
                    base = model[k]
                    break
            calibrator = model.get("calibrator") if isinstance(model, dict) else None

            if base is not None and hasattr(base, "predict_proba"):
                raw = base.predict_proba(X)[:, 1]
                if calibrator is not None and hasattr(calibrator, "predict_proba"):
                    cal = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
                    return float(cal[0])
                return float(raw[0])

        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0][1])

        if hasattr(model, "predict"):
            p = model.predict(X)
            if isinstance(p, np.ndarray) and p.ndim == 2:
                return float(p[0][1])
            return float(p[0])
    except Exception as e:
        logging.warning(f"tree_predict_prob error: {e}")
    return float(fallback)


def get_tree_columns(model):
    """Get correct feature order for RF/XGB/LGBM."""
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        return [f for f in model.feature_name_ if f not in ("", "default")]
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # fallback: minimal lexical subset
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
# Trusted domains (T2: soft override)
# -------------------------

SAFE_DOMAINS = {
    "uvicorn.dev",
    "pytorch.org",
    "ultralytics.com",
    "fastapi.tiangolo.com",
    "python.org",
    "numpy.org",
    "djangoproject.com",
    "docs.djangoproject.com",
}

TRUSTED_SUFFIXES = (
    ".ac.uk",
    ".edu",
    ".gov.uk",
    ".gov",
    ".nhs.uk",
    ".edu.au",
    ".ac.nz",
)


def is_trusted_domain(domain: str) -> bool:
    d = domain.lower()
    d = d.split(":")[0]
    d = d.replace("www.", "")
    if d in SAFE_DOMAINS:
        return True
    for suf in TRUSTED_SUFFIXES:
        if d.endswith(suf):
            return True
    return False


# -------------------------
# Main XAI runner
# -------------------------


def run_example(url: str) -> dict:
    # 1) Validate & init
    validate_url(url)
    init_models_once()

    timings = {}

    # 2) Extract features and build tree row
    feats = extract_features(url)

    ref_model = (
        _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    )
    if ref_model is None:
        raise RuntimeError("No tree model available (rf/xgb/lgbm all missing).")

    tree_columns = get_tree_columns(ref_model)
    x_row_dict = {k: float(feats.get(k, 0)) for k in tree_columns}
    x_row = pd.DataFrame([x_row_dict], columns=tree_columns).astype(float)

    # 3) Tree probabilities
    probs = {}
    fallback = THRESHOLD_PRODUCTION if MODE == "production" else THRESHOLD_ACADEMIC

    for name in ("rf", "xgb", "lgbm"):
        t0 = perf_counter()
        model = _TREE_MODELS.get(name)
        if model is not None:
            probs[name] = tree_predict_prob(model, x_row, fallback)
        else:
            probs[name] = fallback
        timings[f"{name}_ms"] = (perf_counter() - t0) * 1000.0

    # 4) Deep probabilities

    # helpers for encoding
    def _encode_float(url_str: str, max_len: int = 200):
        s = str(url_str)[:max_len].ljust(max_len)
        return torch.tensor([ord(c) / 128 for c in s], dtype=torch.float32).unsqueeze(0)

    def _encode_long(url_str: str, max_len: int = 200):
        s = str(url_str)[:max_len].ljust(max_len)
        return torch.tensor([min(ord(c), 127) for c in s], dtype=torch.long).unsqueeze(
            0
        )

    # default deep probs
    probs["cnn"] = fallback
    probs["ffnn"] = fallback
    probs["lstm"] = fallback

    # CNN
    t0 = perf_counter()
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
    timings["cnn_ms"] = (perf_counter() - t0) * 1000.0

    # FFNN
    t0 = perf_counter()
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
    timings["ffnn_ms"] = (perf_counter() - t0) * 1000.0

    # LSTM (CPU only)
    t0 = perf_counter()
    try:
        lstm_model = _DEEP_MODELS.get("lstm")
        if lstm_model is not None:
            lstm_model.eval()
            x = _encode_long(url)
            with torch.inference_mode():
                prev_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                lstm_cpu = lstm_model.to("cpu")
                x_cpu = x.to("cpu")
                try:
                    if hasattr(lstm_cpu, "lstm") and hasattr(
                        lstm_cpu.lstm, "flatten_parameters"
                    ):
                        lstm_cpu.lstm.flatten_parameters()
                except Exception:
                    pass
                out = lstm_cpu(x_cpu)
                torch.backends.cudnn.enabled = prev_cudnn
            if isinstance(out, (tuple, list)):
                out = out[0]
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
    timings["lstm_ms"] = (perf_counter() - t0) * 1000.0

    # 5) Hybrid fusion
    tree_vals = [probs[k] for k in ("rf", "xgb", "lgbm")]
    deep_vals = [probs[k] for k in ("cnn", "ffnn", "lstm")]

    tree_mean = float(np.mean(tree_vals)) if tree_vals else fallback
    deep_mean = float(np.mean(deep_vals)) if deep_vals else fallback

    hybrid_score = 0.75 * tree_mean + 0.25 * deep_mean
    probs["hybrid"] = hybrid_score

    # 6) Soft trusted-domain override (T2)
    p = urlparse(url)
    domain_only = p.netloc.lower().split(":")[0].replace("www.", "")

    trusted = is_trusted_domain(domain_only)
    override_reason = None

    if trusted and hybrid_score < 0.85:
        # soften score towards "benign" but still reflect model
        final_prob = 0.25 + 0.25 * hybrid_score
        override_reason = f"Trusted domain override applied for {domain_only}"
        logging.info(
            f"Trusted override: {domain_only} hybrid={hybrid_score:.3f} -> final={final_prob:.3f}"
        )
    else:
        final_prob = hybrid_score

    probs["final"] = final_prob

    # 7) Choose threshold depending on mode
    if MODE == "academic":
        threshold_used = THRESHOLD_ACADEMIC
    else:
        threshold_used = THRESHOLD_PRODUCTION

    verdict = "Phishing" if final_prob >= threshold_used else "Benign"

    # 8) Pick best tree and deep models for XAI
    tree_candidates = {k: _TREE_MODELS.get(k) for k in ("rf", "xgb", "lgbm")}
    best_tree_key = max(
        (k for k in tree_candidates.keys() if tree_candidates[k] is not None),
        key=lambda k: probs.get(k, 0.0),
        default=None,
    )
    tree_model = (
        tree_candidates.get(best_tree_key)
        if best_tree_key
        else (
            _TREE_MODELS.get("lgbm")
            or _TREE_MODELS.get("xgb")
            or _TREE_MODELS.get("rf")
        )
    )
    if tree_model is not None:
        tree_columns = get_tree_columns(tree_model)
    else:
        tree_columns = tree_columns  # fallback to earlier derived

    deep_model = (
        _DEEP_MODELS.get("lstm") or _DEEP_MODELS.get("cnn") or _DEEP_MODELS.get("ffnn")
    )

    # 9) Plain-English XAI
    result = make_plain_english(
        url=url,
        probs={k: float(v) for k, v in probs.items()},
        tree_model=tree_model,
        tree_columns=tree_columns,
        deep_model=deep_model,
        max_reasons=4,
        threshold=threshold_used,
        final_prob_override=float(final_prob),
    )

    # Inject override info into reasons if needed
    reasons = result.get("reasons", [])
    if override_reason:
        reasons = [override_reason] + reasons
    result["reasons"] = reasons

    # 10) Final JSON fields & logging
    result["verdict"] = verdict
    result["confidence"] = float(final_prob)
    result["final_prob"] = float(final_prob)
    result["threshold_used"] = float(threshold_used)
    result["model_breakdown"] = {k: float(v) for k, v in probs.items()}
    result["timings_ms"] = {k: round(v, 3) for k, v in timings.items()}

    safe = sanitize_url_for_log(url)
    logging.info(f"{safe} -> {verdict} ({final_prob:.3f})")

    return result


# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    # test_url = "https://uvicorn.dev/"
    test_url = "http://testsafebrowsing.appspot.com/s/phishing.html"

    try:
        out = run_example(test_url)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("Error:", e)
