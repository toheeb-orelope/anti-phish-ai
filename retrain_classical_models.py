"""
diagnose_models.py
------------------
Comprehensive diagnostic script for phishing_detection project.
Checks dataset balance, feature consistency, deep model output shape,
and threshold alignment. Summarises likely causes of inconsistent predictions.
"""

import os, json, torch, pandas as pd
from run_xai import (
    extract_features,
    _TREE_MODELS,
    _DEEP_MODELS,
    get_tree_columns,
    _load_model_threshold,
)

print("=== 🔍 Phishing Detection Diagnostic ===\n")

summary = []

# ---------------------------
# 1️⃣ Dataset balance check
# ---------------------------
try:
    train_path = "data/processed/train.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    train = pd.read_csv(train_path)
    balance = train["label"].value_counts(normalize=True).to_dict()
    print("[✔] Dataset loaded:", train.shape)
    print("[ℹ] Label distribution:")
    for k, v in balance.items():
        print(f"  Label {k}: {v*100:.2f}%")
    if abs(balance.get(0, 0) - balance.get(1, 0)) > 0.1:
        summary.append("⚠ Dataset appears imbalanced (>10% difference). Consider rebalancing.")
except Exception as e:
    summary.append(f"❌ Could not check dataset balance: {e}")

print("\n" + "-" * 60 + "\n")

# ---------------------------
# 2️⃣ Feature consistency check
# ---------------------------
try:
    feats = extract_features("https://northampton.ac.uk/")
    print(f"[✔] Extracted {len(feats)} features from test URL.")
    ref_model = _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    if ref_model is not None:
        model_features = get_tree_columns(ref_model)
        print(f"[ℹ] Model expects {len(model_features)} features.")
        missing = set(model_features) - set(feats.keys())
        extra = set(feats.keys()) - set(model_features)
        if missing:
            print(f"⚠ Missing features: {missing}")
            summary.append("⚠ Feature mismatch detected: model trained on different feature schema.")
        if extra:
            print(f"⚠ Extra features: {extra}")
    else:
        summary.append("⚠ No tree model loaded to compare features.")
except Exception as e:
    summary.append(f"❌ Could not verify feature consistency: {e}")

print("\n" + "-" * 60 + "\n")

# ---------------------------
# 3️⃣ Deep model output shape
# ---------------------------
try:
    print("[ℹ] Deep model output shapes:")
    for k, m in _DEEP_MODELS.items():
        if m is not None:
            sample = torch.randn(1, 200) if k != "lstm" else torch.randint(0, 128, (1, 200))
            with torch.no_grad():
                out = m(sample)
            print(f"  {k:5}: output shape -> {tuple(out.shape)}")
            if out.ndim == 1 or (out.ndim == 2 and out.shape[1] not in (1, 2)):
                summary.append(f"⚠ {k} output shape {tuple(out.shape)} unusual; check final layer dimensions.")
        else:
            print(f"  {k:5}: model not loaded.")
except Exception as e:
    summary.append(f"❌ Deep model output check failed: {e}")

print("\n" + "-" * 60 + "\n")

# ---------------------------
# 4️⃣ Threshold alignment check
# ---------------------------
try:
    thresholds_path = "model/thresholds.json"
    if not os.path.exists(thresholds_path):
        summary.append("⚠ thresholds.json missing. Thresholds may not match models.")
    else:
        with open(thresholds_path) as f:
            t = json.load(f)
        print("[ℹ] Thresholds loaded from model/thresholds.json:")
        for k in ("rf", "xgb", "lgbm", "cnn", "ffnn", "lstm", "hybrid"):
            val = _load_model_threshold(k, 0.5)
            print(f"  {k:6}: {val}")
        # simple sanity check
        if any(val < 0.3 or val > 0.7 for val in t.get("rf", {}).values()):
            summary.append("⚠ Some thresholds may be too extreme (<0.3 or >0.7). Recalculate on validation data.")
except Exception as e:
    summary.append(f"❌ Could not check thresholds: {e}")

print("\n" + "-" * 60 + "\n")

# ---------------------------
# 5️⃣ Final summary
# ---------------------------
print("=== 🧭 Diagnostic Summary ===")
if not summary:
    print("✅ No major issues detected. Models and data look consistent.")
else:
    for s in summary:
        print("-", s)

print("\nTip: Fix warnings in order of appearance above for best results.")
print("=============================================================\n")
