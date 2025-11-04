# evaluate_all_thresholds.py
# ---------------------------------------------------------
# Compute optimal thresholds (Youden J and max-F1) for:
# RF, XGB, LGBM, CNN, FFNN, LSTM, and the HYBRID score.
# Uses TEST set by default; if you have a validation CSV,
# point DATA_PATH to that instead to avoid test leakage.
# ---------------------------------------------------------

import os, json, numpy as np, pandas as pd
from typing import Dict
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, auc
import joblib
import torch
import torch.nn.functional as F

# ---------- paths ----------
DATA_PATH = "data/processed/test.csv"
RF_PATH = "model/rf_calibrated.pkl"
XGB_PATH = "model/xgb_calibrated.pkl"
LGBM_PATH = "model/lgbm_calibrated.pkl"

CNN_CKPT = "models/cnn_lightning.ckpt"
FFNN_CKPT = "models/ffnn_lightning.ckpt"
LSTM_CKPT = "models/lstm_best.ckpt"

THRESHOLDS_JSON = "model/thresholds.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- optional: your Lightning classes ----------
try:
    from phishin_train_cnn import LightningCNN
except Exception:
    LightningCNN = None
try:
    from phishin_train_ffnn import LightningFFNN
except Exception:
    LightningFFNN = None
try:
    from phishin_nlp_lstm import LightningLSTM
except Exception:
    LightningLSTM = None


# ---------------------------------------------------------
# Threshold computation helpers
# ---------------------------------------------------------
def _best_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Return thresholds by Youden-J and max-F1."""
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    j_idx = int(np.argmax(j_scores))
    j_thr = float(thr[j_idx])

    prec, rec, thr_pr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    f1_idx = int(np.argmax(f1))
    f1_thr = float(thr_pr[f1_idx])

    return {
        "youden_j": j_thr,
        "max_f1": f1_thr,
        "auc_roc": float(auc(fpr, tpr)),
        "youden_j_tpr": float(tpr[j_idx]),
        "youden_j_fpr": float(fpr[j_idx]),
        "max_f1_value": float(f1[f1_idx]),
    }


# ---------------------------------------------------------
# Model unwrapping + probability functions
# ---------------------------------------------------------
def _unwrap_model(obj):
    """Handle dict or calibrated structures safely."""
    # If it's a dict, look for common keys
    if isinstance(obj, dict):
        for key in ["xgb", "rf", "lgbm", "calibrated", "base", "model"]:
            if key in obj:
                return obj[key]
        first_key = next(iter(obj.keys()), None)
        return obj[first_key] if first_key else obj

    # If it's a CalibratedClassifierCV or similar
    if hasattr(obj, "base_estimator_"):
        return obj.base_estimator_
    if hasattr(obj, "calibrated_classifiers_"):
        return obj.calibrated_classifiers_[0].estimator
    if hasattr(obj, "_get_estimator"):
        return obj._get_estimator()

    return obj


def _select_tree_columns(model, X: pd.DataFrame):
    """Match model’s expected feature order safely."""
    if hasattr(model, "feature_name_") and model.feature_name_:
        cols = [c for c in model.feature_name_ if c and c != "default"]
        return [c for c in cols if c in X.columns]
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return [c for c in cols if c in X.columns]
    return [c for c in X.columns if c not in ("label", "url")]


def _tree_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Safe predict_proba for RF/XGB/LGBM, even if wrapped in dict."""
    m = _unwrap_model(model)
    cols = _select_tree_columns(m, X)
    return m.predict_proba(X[cols])[:, 1]


# ---------------------------------------------------------
# Deep model utilities
# ---------------------------------------------------------
def _encode_float(url: str, max_len: int = 200) -> torch.Tensor:
    s = str(url)[:max_len].ljust(max_len)
    return torch.tensor([[ord(c) / 128 for c in s]], dtype=torch.float32)


def _encode_long(url: str, max_len: int = 200) -> torch.Tensor:
    s = str(url)[:max_len].ljust(max_len)
    return torch.tensor([[min(ord(c), 127) for c in s]], dtype=torch.long)


def _deep_proba(model, urls: pd.Series, kind: str) -> np.ndarray:
    """Compute probabilities for deep models."""
    probs = np.full(len(urls), np.nan, dtype=np.float32)
    if model is None:
        return probs

    model.eval()
    use_gpu = (DEVICE.type == "cuda") and (kind in {"cnn", "ffnn"})
    for i, u in enumerate(urls):
        try:
            if kind in {"cnn", "ffnn"}:
                x = _encode_float(u)
                if use_gpu:
                    x = x.to(DEVICE)
                    model.to(DEVICE)
                with torch.inference_mode():
                    out = model(x)
            else:  # lstm on CPU
                x = _encode_long(u)
                with torch.inference_mode():
                    prev = torch.backends.cudnn.enabled
                    torch.backends.cudnn.enabled = False
                    model_cpu = model.to("cpu")
                    out = model_cpu(x)
                    torch.backends.cudnn.enabled = prev

            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim in (0, 1):
                if out.ndim == 0 or (out.ndim == 1 and out.numel() == 1):
                    p = torch.sigmoid(out).item()
                elif out.ndim == 1 and out.shape[0] == 2:
                    p = F.softmax(out, dim=0)[1].item()
            else:
                sm = F.softmax(out, dim=1)
                p = sm[0, 1].item() if sm.shape[1] > 1 else sm[0, 0].item()
            probs[i] = float(p)
        except Exception:
            pass
    probs[np.isnan(probs)] = 0.5
    return probs


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
def main():
    print("[*] Loading test/validation data …")
    df = pd.read_csv(DATA_PATH)

    # Ensure TLD encoder consistency
    if "tld_enc" not in df.columns:
        le = joblib.load("model/tld_encoder.pkl")
        df["tld_enc"] = [
            le.transform([t if t in le.classes_ else "__unknown__"])[0]
            for t in df["tld"].astype(str)
        ]

    assert "label" in df.columns, "Expected a 'label' column in the dataset."
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"])
    urls = df["url"] if "url" in df.columns else pd.Series([""] * len(df))

    thresholds: Dict[str, Dict[str, float]] = {}
    prob_store: Dict[str, np.ndarray] = {}

    # ---------------- Tree Models ----------------
    for name, path in [("rf", RF_PATH), ("xgb", XGB_PATH), ("lgbm", LGBM_PATH)]:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                p = _tree_proba(model, X)
                prob_store[name] = p
                thresholds[name] = _best_thresholds(y, p)
                print(
                    f"[OK] {name.upper()} AUC: {thresholds[name]['auc_roc']:.4f}  "
                    f"J-thr: {thresholds[name]['youden_j']:.4f}  "
                    f"F1-thr: {thresholds[name]['max_f1']:.4f}"
                )
            except Exception as e:
                print(f"[WARN] {name.upper()} failed: {e}")

    # ---------------- Deep Models ----------------
    cnn = ffnn = lstm = None
    if LightningCNN and os.path.exists(CNN_CKPT):
        try:
            cnn = LightningCNN.load_from_checkpoint(
                CNN_CKPT, map_location=DEVICE
            ).eval()
            print("[OK] CNN loaded.")
        except Exception as e:
            print(f"[WARN] CNN load failed: {e}")
    if LightningFFNN and os.path.exists(FFNN_CKPT):
        try:
            ffnn = LightningFFNN.load_from_checkpoint(
                FFNN_CKPT, map_location=DEVICE
            ).eval()
            print("[OK] FFNN loaded.")
        except Exception as e:
            print(f"[WARN] FFNN load failed: {e}")
    if LightningLSTM and os.path.exists(LSTM_CKPT):
        try:
            lstm = LightningLSTM.load_from_checkpoint(
                LSTM_CKPT, map_location="cpu"
            ).eval()
            print("[OK] LSTM loaded (CPU).")
        except Exception as e:
            print(f"[WARN] LSTM load failed: {e}")

    if len(urls) == len(X):
        if cnn is not None:
            p = _deep_proba(cnn, urls, "cnn")
            prob_store["cnn"] = p
            thresholds["cnn"] = _best_thresholds(y, p)
        if ffnn is not None:
            p = _deep_proba(ffnn, urls, "ffnn")
            prob_store["ffnn"] = p
            thresholds["ffnn"] = _best_thresholds(y, p)
        if lstm is not None:
            p = _deep_proba(lstm, urls, "lstm")
            prob_store["lstm"] = p
            thresholds["lstm"] = _best_thresholds(y, p)
        for k in ("cnn", "ffnn", "lstm"):
            if k in thresholds:
                print(
                    f"[OK] {k.upper()} AUC: {thresholds[k]['auc_roc']:.4f}  "
                    f"J-thr: {thresholds[k]['youden_j']:.4f}  "
                    f"F1-thr: {thresholds[k]['max_f1']:.4f}"
                )

    # ---------------- Hybrid Ensemble ----------------
    if prob_store:
        tree_keys = [k for k in prob_store.keys() if k in ("rf", "xgb", "lgbm")]
        deep_keys = [k for k in prob_store.keys() if k in ("cnn", "ffnn", "lstm")]
        tree_mean = (
            np.mean([prob_store[k] for k in tree_keys], axis=0)
            if tree_keys
            else np.full(len(y), 0.5)
        )
        deep_mean = (
            np.mean([prob_store[k] for k in deep_keys], axis=0)
            if deep_keys
            else np.full(len(y), 0.5)
        )
        hybrid = 0.75 * tree_mean + 0.25 * deep_mean
        thresholds["hybrid"] = _best_thresholds(y, hybrid)
        print(
            f"[OK] HYBRID AUC: {thresholds['hybrid']['auc_roc']:.4f}  "
            f"J-thr: {thresholds['hybrid']['youden_j']:.4f}  "
            f"F1-thr: {thresholds['hybrid']['max_f1']:.4f}"
        )

    # ---------------- Save & Summary ----------------
    os.makedirs(os.path.dirname(THRESHOLDS_JSON), exist_ok=True)
    with open(THRESHOLDS_JSON, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n[SAVED] Thresholds → {THRESHOLDS_JSON}")

    def row(k: str):
        d = thresholds[k]
        return [
            k.upper(),
            d["auc_roc"],
            d["youden_j"],
            d["max_f1"],
            d["youden_j_tpr"],
            d["youden_j_fpr"],
            d["max_f1_value"],
        ]

    cols = ["Model", "AUC", "J-Threshold", "F1-Threshold", "TPR@J", "FPR@J", "Max F1"]
    table = [row(k) for k in thresholds.keys()]
    df_out = pd.DataFrame(table, columns=cols)
    print("\n" + df_out.to_string(index=False))


if __name__ == "__main__":
    main()
