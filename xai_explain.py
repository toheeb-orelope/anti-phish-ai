# xai_explain.py
import numpy as np
import pandas as pd
from extract_features import extract_features
from explain_tree_with_shap import explain_tree_sample
from explain_dl_with_captumIG import top_substrings_ig
from decision_making import feature_reason, substr_reason


rf_model = "models/random_forest.pkl"
xgb_model = "models/xgboost_model.pkl"
lgbm_model = "models/lightgbm_model.pkl"
cnn_model = "models/cnn_best.ckpt"
lstm_model = "models/lstm_best.ckpt"
ffnn_model = "models/ffnn_best.ckpt"


# Combines feature extraction, model explanations, and reason generation
def make_plain_english(
    url: str,
    probs: dict,  # {"rf":0.91, "lgbm":0.94, "lstm":0.97, ...}
    tree_model=None,
    tree_columns=None,
    deep_model=None,
    max_reasons=4,
):
    feats = extract_features(url)

    reasons = []

    # 1) Tree reasons (if available)
    if tree_model is not None and tree_columns is not None:
        x_row = pd.DataFrame([{k: feats[k] for k in tree_columns}])[tree_columns]
        contribs = explain_tree_sample(tree_model, x_row, max_reasons=max_reasons)
        for fname, shap_val in contribs:
            msg, tilt = feature_reason(fname, x_row.iloc[0][fname], shap_val)
            reasons.append((abs(shap_val), msg, tilt))

    # 2) Deep reasons (if available)
    if deep_model is not None:
        spans = top_substrings_ig(deep_model, url, k=3)
        for token, score in spans:
            r = substr_reason(token, score)
            if r:
                reasons.append((abs(score), r[0], r[1]))

    # 3) Sort and dedupe
    reasons = sorted(reasons, key=lambda t: t[0], reverse=True)
    seen = set()
    final_msgs = []
    for _, msg, tilt in reasons:
        if msg not in seen:
            final_msgs.append(msg)
            seen.add(msg)
        if len(final_msgs) >= max_reasons:
            break

    # 4) Decision & confidence (you can plug your ensemble here)
    # e.g., weighted average of model probabilities:
    weights = {"lstm": 0.35, "cnn": 0.25, "lgbm": 0.2, "rf": 0.1, "ffnn": 0.1}
    num = sum(weights.get(k, 0.0) * v for k, v in probs.items())
    den = sum(weights.get(k, 0.0) for k in probs.keys())
    final_prob = num / den if den > 0 else np.mean(list(probs.values()))
    verdict = "Phishing" if final_prob >= 0.5 else "Benign"
    confidence = float(final_prob if verdict == "Phishing" else 1 - final_prob)

    return {
        "url": url,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "final_prob": round(float(final_prob), 4),  # <-- Added line
        "model_breakdown": probs,
        "reasons": final_msgs
        or ["Model confidence exceeded threshold based on learned patterns."],
    }
