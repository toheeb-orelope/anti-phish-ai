import pandas as pd
from urllib.parse import urlparse

# Load everything
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
cache = pd.read_csv("data/processed/domain_cache.csv")


# Extract domain from URL
def get_domain(u):
    try:
        return urlparse(str(u)).netloc.split(":")[0].lower()
    except Exception:
        return ""


train["domain"] = train["url"].apply(get_domain)
test["domain"] = test["url"].apply(get_domain)

# Merge with offline trust cache
train = train.merge(cache, on="domain", how="left").fillna(0)
test = test.merge(cache, on="domain", how="left").fillna(0)

# Save back
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("[+] train/test datasets updated with trust features!")
print(f"train shape: {train.shape}, test shape: {test.shape}")
