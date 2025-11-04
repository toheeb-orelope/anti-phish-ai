# ===========================================
# build_domain_cache_offline.py
# Create host-based/contextual cache using offline datasets
# ===========================================
import pandas as pd
import os
from urllib.parse import urlparse

DATA_DIR = "data/raw"
DATA_DIR1 = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR1, exist_ok=True)
OUTPUT = os.path.join(DATA_DIR1, "domain_cache.csv")

# ------------------------------------------------------------
# Load offline datasets
# ------------------------------------------------------------
print("[*] Loading offline datasets ...")
majestic = pd.read_csv("data/raw/majestic_million.csv")
top1m = pd.read_csv("data/raw/top-1m.csv", header=None, names=["rank", "domain"])

# normalise domain column name
majestic.rename(columns={"Domain": "domain"}, inplace=True)
majestic["domain"] = majestic["domain"].str.lower()
top1m["domain"] = top1m["domain"].str.lower()

# ------------------------------------------------------------
# Combine trust metrics
# ------------------------------------------------------------
trust = pd.merge(top1m, majestic, on="domain", how="outer")

# derive simple trust indicators
trust["is_top1m"] = trust["rank"].notna().astype(int)
trust["is_majestic"] = trust["GlobalRank"].notna().astype(int)
trust["popularity_score"] = (
    trust[["rank", "GlobalRank"]].min(axis=1, skipna=True).fillna(2_000_000).astype(int)
)

# Keep relevant columns
trust = trust[["domain", "popularity_score", "is_top1m", "is_majestic"]]
print(f"[INFO] Combined trust dataset shape: {trust.shape}")

# ------------------------------------------------------------
# Gather unique domains from your train/test
# ------------------------------------------------------------
train = pd.read_csv(os.path.join(DATA_DIR1, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR1, "test.csv"))


def extract_domain(url):
    try:
        return urlparse(str(url)).netloc.split(":")[0].lower()
    except Exception:
        return ""


domains = pd.concat([train["url"], test["url"]], ignore_index=True).apply(
    extract_domain
)
unique_domains = pd.DataFrame({"domain": sorted(set(domains) - {""})})

print(f"[INFO] Found {len(unique_domains):,} unique domains in your dataset.")

# ------------------------------------------------------------
# Merge trust info into main cache
# ------------------------------------------------------------
cache = unique_domains.merge(trust, on="domain", how="left")

# Fill missing domains with neutral trust values
cache["popularity_score"].fillna(2_000_000, inplace=True)
cache["is_top1m"].fillna(0, inplace=True)
cache["is_majestic"].fillna(0, inplace=True)

# optional derived trust flag (smaller rank → higher trust)
cache["trust_rank_norm"] = 1 - (cache["popularity_score"] / 2_000_000)
cache.loc[cache["trust_rank_norm"] < 0, "trust_rank_norm"] = 0

# ------------------------------------------------------------
# Save to CSV
# ------------------------------------------------------------
cache.to_csv(OUTPUT, index=False)
print(f"[+] Offline domain cache saved to {OUTPUT} ({len(cache):,} domains).")
print("[DONE]")
