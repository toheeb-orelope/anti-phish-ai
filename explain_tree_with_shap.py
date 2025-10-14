# explain_tree_with_shap.py
import shap
import numpy as np
import pandas as pd


def explain_tree_sample(model, x_row: pd.DataFrame, max_reasons=4):
    """
    Explain a single sample from a tree-based model (RF/XGB/LGBM) using SHAP.
    Returns top feature contributions as (feature_name, shap_value).
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_row)

    # Handle LightGBM / XGBoost / RF output variations
    if isinstance(shap_vals, list):
        # For binary classification: shap_values can be [neg_class, pos_class]
        if len(shap_vals) == 2:
            shap_values = shap_vals[1][0]  # positive class (phish)
        else:
            shap_values = shap_vals[0][0] if shap_vals[0].ndim > 1 else shap_vals[0]
    elif isinstance(shap_vals, np.ndarray):
        # For RandomForest / single array output
        shap_values = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
    else:
        shap_values = np.array(shap_vals)

    # Rank by absolute magnitude
    feats = x_row.columns.tolist()
    pairs = sorted(zip(feats, shap_values), key=lambda t: abs(t[1]), reverse=True)[
        :max_reasons
    ]
    return pairs  # list of (feature_name, shap_contribution)
