# feature_importance_heatmap.py
import joblib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# Load train data and model
train = pd.read_csv("data/processed/train.csv").fillna(0)
model = joblib.load("model/rf_calibrated.pkl")

# Access underlying RandomForest inside CalibratedClassifierCV
rf = model.base_estimator if hasattr(model, "base_estimator") else model.estimator

# --- Feature importance
feature_importances = pd.Series(
    rf.feature_importances_,
    index=train.drop(columns=["url", "label", "tld"], errors="ignore").columns,
)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values[:15], y=feature_importances.index[:15])
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# --- Correlation heatmap
plt.figure(figsize=(12, 10))
corr = train.drop(columns=["url", "label", "tld"], errors="ignore").corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
