# ============================================================
# run_xai.py  (Production hybrid + XAI)
#
# Uses:
#   - Tree model: model/new_rf_calibrated.pkl  (RF calibrated)
#   - Deep models: models/cnn_lightning.ckpt  (CNN_feature)
#                  models/ffnn_lightning.ckpt (FFNN_feature)
#
#   - Feature extractor: extract_features.extract_features
#   - Plain-English XAI: xai_explain.make_plain_english
#
# Public entrypoint:
#   - run_xai(url: str) -> dict
#     Returns prediction + hybrid prob + explanation bundle.
# ============================================================

import os
import json
import logging
import re
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa

import joblib  # type: ignore

from extract_features import extract_features
from xai_explain import make_plain_english
from phishin_train_cnn import LightningCNN
from phishin_train_ffnn import LightningFFNN

# ------------------------------------------------------------
# Paths & global config
# ------------------------------------------------------------
RF_PATH = "model/rf_calibrated.pkl"
CNN_CKPT = "models/cnn_lightning.ckpt"
FFNN_CKPT = "models/ffnn_lightning.ckpt"

THRESHOLD_CFG = "model/thresholds.json"
DEFAULT_THRESHOLD = 0.5

MAX_URL_LEN = 2048
MAX_SEQ_LEN = 200  # must match DL training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("run_xai")


def _sanitize_url_for_log(url: str) -> str:
    """Hide query tokens etc. just for logs."""
    if not isinstance(url, str):
        return "<non-str-url>"
    u = url.strip()
    # Strip very long query part for logs only
    u = re.sub(r"(\?.{20}).*$", r"\1...[trimmed]", u)
    return u[:256]


# ------------------------------------------------------------
# Global model caches
# ------------------------------------------------------------
_TREE_MODELS: Dict[str, Any] = {}
_DEEP_MODELS: Dict[str, Any] = {}
_MODELS_INITIALISED = False
THRESHOLD = DEFAULT_THRESHOLD  # hybrid decision threshold


# ------------------------------------------------------------
# Threshold loader
# ------------------------------------------------------------
def _load_thresholds() -> None:
    """Load global / per-model thresholds if thresholds.json exists."""
    global THRESHOLD

    if not os.path.exists(THRESHOLD_CFG):
        logger.info(
            "No thresholds.json found - using default threshold=%.3f", DEFAULT_THRESHOLD
        )
        THRESHOLD = DEFAULT_THRESHOLD
        return

    try:
        with open(THRESHOLD_CFG, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.warning(
            "Failed to load thresholds.json (%s) - using default threshold", e
        )
        THRESHOLD = DEFAULT_THRESHOLD
        return

    # Prefer hybrid threshold if present. Handle dict entries from metrics files.
    if isinstance(cfg, dict) and "hybrid" in cfg:
        hybrid_cfg = cfg["hybrid"]

        if isinstance(hybrid_cfg, dict):
            # Try common numeric keys in order of preference
            preferred_keys = ("youden_j", "max_f1", "threshold")
            for key in preferred_keys:
                val = hybrid_cfg.get(key)
                if isinstance(val, (int, float)):
                    THRESHOLD = float(val)
                    break
            else:
                logger.warning(
                    "Hybrid entry in thresholds.json has no numeric threshold - using default threshold=%.3f",
                    DEFAULT_THRESHOLD,
                )
                THRESHOLD = DEFAULT_THRESHOLD
        else:
            try:
                THRESHOLD = float(hybrid_cfg)
            except (TypeError, ValueError):
                logger.warning(
                    "Hybrid threshold entry is not numeric (%s) - using default threshold=%.3f",
                    hybrid_cfg,
                    DEFAULT_THRESHOLD,
                )
                THRESHOLD = DEFAULT_THRESHOLD
    else:
        THRESHOLD = DEFAULT_THRESHOLD

    logger.info("Loaded threshold(s) from thresholds.json - hybrid=%.3f", THRESHOLD)


# ------------------------------------------------------------
# URL validation
# ------------------------------------------------------------
def _validate_url(url: Any) -> str:
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    u = url.strip()
    if not u:
        raise ValueError("URL is empty")
    if len(u) > MAX_URL_LEN:
        raise ValueError(f"URL too long (> {MAX_URL_LEN} chars)")

    # Very lightweight sanity check – you already validate more at frontend
    if not (u.startswith("http://") or u.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    return u


# ------------------------------------------------------------
# Tree model loading & helpers (RF only)
# ------------------------------------------------------------
def _load_tree_model(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        logger.warning("Tree model %s not found.", path)
        return None
    try:
        model = joblib.load(path)
        logger.info("Loaded tree model from %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load tree model from %s: %s", path, e)
        return None


def _get_tree_columns(model: Any, feature_dict: Dict[str, Any]) -> list[str]:
    """
    Get ordered feature columns for RF.
    Prefer model.feature_names_in_ if available, else fallback to keys from feature_dict.
    """
    # sklearn >= 1.0 stores feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)  # type: ignore[attr-defined]

    # Fallback – infer numeric feature columns
    cols = [
        k
        for k, v in feature_dict.items()
        if k not in ("url", "tld") and isinstance(v, (int, float, np.number))
    ]

    cols = sorted(cols)
    logger.warning(
        "Model has no feature_names_in_; using fallback inferred columns (%d).",
        len(cols),
    )
    return cols


def _tree_predict_prob(model: Any, x_df: pd.DataFrame) -> float:
    """
    Generic probability extraction from RF / calibrated wrappers.
    Returns P(phishing=1).
    """
    if model is None:
        return DEFAULT_THRESHOLD

    # If model is a calibrated wrapper, it should expose predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        elif proba.ndim == 1:
            # Just in case, treat as probability of class 1
            return float(proba[0])

    # Fallback: decision_function -> logistic squashing
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_df)
        score = float(scores[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    # Final fallback: predict (hard) -> map to 0.01/0.99
    if hasattr(model, "predict"):
        pred = int(model.predict(x_df)[0])
        return 0.99 if pred == 1 else 0.01

    return DEFAULT_THRESHOLD


# ------------------------------------------------------------
# Deep model loading & helpers (CNN + FFNN)
# ------------------------------------------------------------
def _load_deep_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    # CNN_feature
    if os.path.exists(CNN_CKPT):
        try:
            cnn = LightningCNN.load_from_checkpoint(
                CNN_CKPT, map_location=DEVICE, lr=1e-3
            )
            cnn.eval()
            models["cnn"] = cnn
            logger.info("Loaded CNN model from %s", CNN_CKPT)
        except Exception as e:
            logger.error("Failed to load CNN from %s: %s", CNN_CKPT, e)
    else:
        logger.warning("CNN checkpoint %s not found", CNN_CKPT)

    # FFNN_feature
    if os.path.exists(FFNN_CKPT):
        try:
            ffnn = LightningFFNN.load_from_checkpoint(
                FFNN_CKPT, map_location=DEVICE, lr=1e-3
            )
            ffnn.eval()
            models["ffnn"] = ffnn
            logger.info("Loaded FFNN model from %s", FFNN_CKPT)
        except Exception as e:
            logger.error("Failed to load FFNN from %s: %s", FFNN_CKPT, e)
    else:
        logger.warning("FFNN checkpoint %s not found", FFNN_CKPT)

    return models


def _encode_url_float(url: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """
    Float encoding (ord/128) to match your DL training pipeline.

    Shape: [1, max_len]
    """
    s = url[:max_len].ljust(max_len)
    arr = [min(ord(c), 127) / 128.0 for c in s]
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


def _deep_predict_prob(model: Any, url: str) -> float:
    """
    Get P(phishing=1) for a Lightning CNN / FFNN model using char float encoding.
    """
    if model is None:
        return DEFAULT_THRESHOLD

    x = _encode_url_float(url).to(DEVICE)

    model = model.to(DEVICE)
    model.eval()

    with torch.inference_mode():
        out = model(x)

        # Standard shapes:
        #  - [B, 1] logits  -> sigmoid
        #  - [B] logits     -> sigmoid
        #  - [B, 2] logits  -> softmax[:,1]
        if out.ndim == 2:
            if out.shape[1] == 1:
                prob = torch.sigmoid(out[:, 0])
            elif out.shape[1] == 2:
                prob = torch.softmax(out, dim=1)[:, 1]
            else:
                # Unexpected shape – take last column
                prob = torch.softmax(out, dim=1)[:, -1]
        elif out.ndim == 1:
            prob = torch.sigmoid(out)
        else:
            # Very unusual, flatten then pick first
            prob = torch.sigmoid(out.view(-1)[0])

    return float(prob.squeeze().cpu().item())


# ------------------------------------------------------------
# One-time model init
# ------------------------------------------------------------
def _init_models_once() -> None:
    global _MODELS_INITIALISED, _TREE_MODELS, _DEEP_MODELS

    if _MODELS_INITIALISED:
        return

    logger.info("Initialising models (tree + deep)…")

    # Tree: RF calibrated only
    rf_model = _load_tree_model(RF_PATH)
    _TREE_MODELS = {"rf": rf_model}

    # Deep: CNN + FFNN
    _DEEP_MODELS = _load_deep_models()

    _load_thresholds()
    _MODELS_INITIALISED = True

    logger.info(
        "Model init done → RF: %s, CNN: %s, FFNN: %s, hybrid threshold=%.3f",
        "OK" if _TREE_MODELS.get("rf") is not None else "missing",
        "OK" if _DEEP_MODELS.get("cnn") is not None else "missing",
        "OK" if _DEEP_MODELS.get("ffnn") is not None else "missing",
        THRESHOLD,
    )


# ------------------------------------------------------------
# Core hybrid inference + XAI
# ------------------------------------------------------------
def run_xai(url: str) -> Dict[str, Any]:
    """
    Main production entrypoint.

    Returns a dict with:
      - url
      - label (0/1)
      - label_name ("benign"/"phishing")
      - hybrid_prob
      - threshold
      - tree_probs { "rf": p }
      - deep_probs { "cnn": p?, "ffnn": p? }
      - features (numeric & lexical)
      - explanation (plain English paragraphs from make_plain_english)
    """
    global THRESHOLD

    _init_models_once()

    try:
        u = _validate_url(url)
    except Exception as e:
        logger.error("Invalid URL: %s", e)
        raise

    logger.info("🔎 Inference on URL: %s", _sanitize_url_for_log(u))

    # 1. Extract features
    features = extract_features(u)

    # 2. Tree model (RF)
    rf_model = _TREE_MODELS.get("rf")
    tree_probs: Dict[str, float] = {}

    if rf_model is not None:
        cols = _get_tree_columns(rf_model, features)
        row_dict = {c: float(features.get(c, 0.0)) for c in cols}
        x_row = pd.DataFrame([row_dict], columns=cols)
        rf_prob = _tree_predict_prob(rf_model, x_row)
        tree_probs["rf"] = rf_prob
    else:
        rf_prob = DEFAULT_THRESHOLD

    # 3. Deep models (CNN + FFNN) on raw URL string
    deep_probs: Dict[str, float] = {}

    cnn_model = _DEEP_MODELS.get("cnn")
    if cnn_model is not None:
        try:
            deep_probs["cnn"] = _deep_predict_prob(cnn_model, u)
        except Exception as e:
            logger.error("CNN prediction failed: %s", e)

    ffnn_model = _DEEP_MODELS.get("ffnn")
    if ffnn_model is not None:
        try:
            deep_probs["ffnn"] = _deep_predict_prob(ffnn_model, u)
        except Exception as e:
            logger.error("FFNN prediction failed: %s", e)

    # 4. Hybrid probability – simple average of available sources
    active_probs = []

    if "rf" in tree_probs:
        active_probs.append(tree_probs["rf"])
    if "cnn" in deep_probs:
        active_probs.append(deep_probs["cnn"])
    if "ffnn" in deep_probs:
        active_probs.append(deep_probs["ffnn"])

    if active_probs:
        hybrid_prob = float(np.mean(active_probs))
    else:
        hybrid_prob = rf_prob  # last resort

    # 5. Final label using global THRESHOLD
    label = 1 if hybrid_prob >= THRESHOLD else 0
    label_name = "phishing" if label == 1 else "benign"

    logger.info(
        "RF=%.3f, CNN=%s, FFNN=%s → hybrid=%.3f (thr=%.3f) → %s",
        tree_probs.get("rf", np.nan),
        f"{deep_probs.get('cnn', np.nan):.3f}" if "cnn" in deep_probs else "NA",
        f"{deep_probs.get('ffnn', np.nan):.3f}" if "ffnn" in deep_probs else "NA",
        hybrid_prob,
        THRESHOLD,
        label_name.upper(),
    )

    # 6. Plain-English explanation (reuses your existing helper)
    try:
        explanation = make_plain_english(
            url=u,
            tree_probs=tree_probs,
            deep_probs=deep_probs,
            hybrid_prob=hybrid_prob,
            threshold=THRESHOLD,
            features=features,
            tree_model=rf_model,
        )
    except TypeError:
        # Fallback in case your make_plain_english has older signature
        explanation = make_plain_english(
            url=u,
            probs={
                "rf_prob": tree_probs.get("rf", rf_prob),
                "cnn_prob": deep_probs.get("cnn", None),
                "ffnn_prob": deep_probs.get("ffnn", None),
                "hybrid_prob": hybrid_prob,
            },
            threshold=THRESHOLD,
            features=features,
            tree_model=rf_model,
        )

    result = {
        "url": u,
        "label": int(label),
        "label_name": label_name,
        "hybrid_prob": float(hybrid_prob),
        "threshold": float(THRESHOLD),
        "tree_probs": tree_probs,
        "deep_probs": deep_probs,
        "features": features,
        "explanation": explanation,
    }

    return result


# ------------------------------------------------------------
# CLI helper (manual testing)
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        # else "http://testsafebrowsing.appspot.com/s/phishing.html"
        else "http://example.com/login.php?user=alice"
    )
    out = run_xai(test_url)
    print(json.dumps(out, indent=2))
