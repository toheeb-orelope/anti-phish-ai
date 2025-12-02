import pandas as pd

# from extract_features import extract_features
# from run_xai import _TREE_MODELS, get_tree_columns

# cache_path = "data/processed/domain_cache.csv"

# # Load the existing cache
# cache = pd.read_csv(cache_path)

# # Add synthetic domain age (0–3650 days, i.e., 0–10 years)
# cache["domain_age_days"] = (cache["trust_rank_norm"] * 3650).astype(int)

# # Save back to disk
# cache.to_csv(cache_path, index=False)

# print(f"[+] Added synthetic 'domain_age_days' column to {cache_path}")
# print(cache.head(3))


# Load your processed datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Print dataset shapes
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# # Display all columns (not truncated)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)

# # Show first few rows of train/test with all feature columns
# print("\n=== TRAIN SAMPLE ===")
# print(train.head(3))  # You can increase to 5 or 10 if you want

# print("\n=== TEST SAMPLE ===")
# print(test.head(3))

# # List all column names explicitly
# print("\n=== Train Columns ===")
# print(train.columns.tolist())

# print("\n=== Test Columns ===")
# print(test.columns.tolist())


# from explain_tree_with_shap import explain_tree_sample
# from run_xai import _TREE_MODELS, get_tree_columns
# from extract_features import extract_features
# import pandas as pd

# feats = extract_features("https://northampton.ac.uk/")
# model = _TREE_MODELS["rf"]
# cols = get_tree_columns(model)
# x_row = pd.DataFrame([{k: feats.get(k, 0) for k in cols}])[cols]

# # ✅ new correct call
# explain_tree_sample(model, x_row)
