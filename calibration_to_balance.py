from sklearn.calibration import CalibratedClassifierCV
import joblib
import pandas as pd
from extract_features import extract_features  # your existing feature builder

train_df = pd.read_csv("data/processed/train.csv")
# X_train = train_df.drop("label", axis=1)
feature_columns = [col for col in train_df.columns if col not in ["url", "label"]]
X_train = train_df[feature_columns]
y_train = train_df["label"]
RF_PATH = "models/random_forest.pkl"
XGB_PATH = "models/xgboost_model.pkl"
LGBM_PATH = "models/lightgbm_model.pkl"


for model_name, path in [("rf", RF_PATH), ("xgb", XGB_PATH), ("lgbm", LGBM_PATH)]:
    base_model = joblib.load(path)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    joblib.dump(calibrated, f"models/{model_name}_calibrated.pkl")
