import joblib, pandas as pd
from extract_features import extract_features

# 1️⃣ What your trained RF model expects:
rf = joblib.load("model/rf_calibrated.pkl")
print(rf.feature_names_in_)

# 2️⃣ What inference actually produces:
print(pd.DataFrame([extract_features("https://northampton.ac.uk/")]).columns)
