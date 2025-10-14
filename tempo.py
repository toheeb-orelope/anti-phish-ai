# generate_ensemble_probs.py
import numpy as np
import pandas as pd
from run_xai import (
    init_models_once,
    _TREE_MODELS,
    _DEEP_MODELS,
    get_tree_columns,
    tree_predict_prob,
    deep_predict_prob,
)
from extract_features import extract_features
from xai_explain import make_plain_english

# Load models
init_models_once()

# Load your test dataset
df = pd.read_csv("data/processed/test.csv")
urls = df["url"].values
y_true = df["label"].values

probs = []

for url in urls:
    # Extract features for tree models
    feats = extract_features(url)
    tree_model = (
        _TREE_MODELS.get("lgbm") or _TREE_MODELS.get("xgb") or _TREE_MODELS.get("rf")
    )
    tree_columns = get_tree_columns(tree_model)
    x_row = pd.DataFrame([{k: feats.get(k, 0) for k in tree_columns}])[tree_columns]

    # Collect predictions
    p = {}
    if _TREE_MODELS.get("rf") is not None:
        p["rf"] = tree_predict_prob(_TREE_MODELS["rf"], x_row)
    if _TREE_MODELS.get("xgb") is not None:
        p["xgb"] = tree_predict_prob(_TREE_MODELS["xgb"], x_row)
    if _TREE_MODELS.get("lgbm") is not None:
        p["lgbm"] = tree_predict_prob(_TREE_MODELS["lgbm"], x_row)

    p["cnn"] = deep_predict_prob(_DEEP_MODELS.get("cnn"), url)
    p["lstm"] = deep_predict_prob(_DEEP_MODELS.get("lstm"), url)
    p["ffnn"] = deep_predict_prob(_DEEP_MODELS.get("ffnn"), url)

    # Use your existing ensemble fusion logic from xai_explain.py
    result = make_plain_english(url=url, probs=p)
    probs.append(result["final_prob"])

# Save probabilities
np.save("models/ensemble_probs.npy", np.array(probs))
print("✅ Saved ensemble probabilities to models/ensemble_probs.npy")
import os

print("Working directory:", os.getcwd())
