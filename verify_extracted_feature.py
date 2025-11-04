import pandas as pd
import matplotlib.pyplot as plt


# df = pd.read_csv("data/processed/train.csv")
df = pd.read_csv("data/processed/test.csv")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print(df.isna().sum())

print("\nSample rows:")
print(df.head(3).T)  # Transposed for better readability


numeric_cols = [c for c in df.columns if df[c].dtype != "object" and c not in ["label"]]
df[numeric_cols].describe().T[["mean", "std", "min", "max"]]

df[numeric_cols].hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.show()
