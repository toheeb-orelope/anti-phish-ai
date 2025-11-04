# explain_tree_with_shap.py
import shap
import numpy as np
import pandas as pd

"""
def explain_tree_sample(model, x_row: pd.DataFrame, max_reasons=4):
    
    Explain a single sample from a tree-based model (RF/XGB/LGBM) using SHAP.
    Returns top feature contributions as (feature_name, shap_value).
    
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
"""


# explain_tree_with_shap.py


# explain_tree_with_shap.py

def explain_tree_sample(model, x_row: pd.DataFrame, max_reasons=4):
    """
    Explain a single sample from a tree-based model (RF/XGB/LGBM) using SHAP.
    Automatically unwraps CalibratedClassifierCV and dict-based bundles.
    Returns top feature contributions as (feature_name, shap_value).
    """

    # ------------------------------------------
    # 1️⃣ Unwrap dictionary or calibrated models
    # ------------------------------------------
    base_model = model

    # Case 1: Dict wrapper {"calibrated": ..., "base": ...}
    if isinstance(model, dict):
        if "base" in model:
            base_model = model["base"]
        elif "xgb" in model:  # your earlier structure
            base_model = model["xgb"]
        elif "rf" in model:
            base_model = model["rf"]

    # Case 2: CalibratedClassifierCV (varies by sklearn version)
    elif hasattr(model, "base_estimator_"):
        print(f"[INFO] Unwrapping calibrated model: {type(model).__name__}")
        base_model = model.base_estimator_
    elif hasattr(model, "calibrated_classifiers_"):
        try:
            base_model = model.calibrated_classifiers_[0].estimator
            print(
                f"[INFO] Unwrapped via calibrated_classifiers_: {type(base_model).__name__}"
            )
        except Exception:
            pass
    elif hasattr(model, "_get_estimator"):
        base_model = model._get_estimator()

    # ------------------------------------------
    # 2️⃣ Initialize SHAP safely
    # ------------------------------------------
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_vals = explainer.shap_values(x_row)
    except Exception as e:
        print(f"[WARN] SHAP explanation failed for {type(base_model).__name__}: {e}")
        return [("no_explanation", 0.0)]

    # ------------------------------------------
    # 3️⃣ Handle output shapes (RF/LGBM/XGB)
    # ------------------------------------------
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            shap_values = shap_vals[1][0]  # positive class (phish)
        else:
            shap_values = shap_vals[0][0] if shap_vals[0].ndim > 1 else shap_vals[0]
    elif isinstance(shap_vals, np.ndarray):
        shap_values = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
    else:
        shap_values = np.array(shap_vals)

    # ------------------------------------------
    # 4️⃣ Rank features by absolute contribution
    # ------------------------------------------
    feats = x_row.columns.tolist()
    pairs = sorted(zip(feats, shap_values), key=lambda t: abs(t[1]), reverse=True)[
        :max_reasons
    ]

    return pairs
