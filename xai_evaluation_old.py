import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm
from run_xai import run_example
from sklearn.metrics import classification_report
from multiprocessing import Pool, cpu_count
import sys
import os

# Old version (for reference)
"""
# ------------------------------
# 1. Load your test dataset
# ------------------------------
test_df = pd.read_csv("data/processed/test.csv")

# Ensure 'label' column exists
if "label" not in test_df.columns:
    raise ValueError(
        "The test dataset must include a 'label' column (0=Benign, 1=Phishing)."
    )

print(f"Loaded {len(test_df)} URLs for evaluation.")

# ------------------------------
# 2. Run inference on all URLs
# ------------------------------


def normalize_url(u: str) -> str:
    # Ensure all URLs start with http(s) and contain only printable ASCII.
    u = str(u).strip()
    # Prefix missing scheme
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    # Remove invisible / non-ASCII characters
    u = re.sub(r"[^\x20-\x7e]", "", u)
    return u


from multiprocessing import Pool, cpu_count


def process_url(url):
    try:
        result = run_example(url)
        return {
            "url": result["url"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "final_prob": result["final_prob"],
            "rf_prob": result["model_breakdown"].get("rf", None),
            "xgb_prob": result["model_breakdown"].get("xgb", None),
            "lgbm_prob": result["model_breakdown"].get("lgbm", None),
            "cnn_prob": result["model_breakdown"].get("cnn", None),
            "ffnn_prob": result["model_breakdown"].get("ffnn", None),
            "lstm_prob": result["model_breakdown"].get("lstm", None),
            "hybrid_prob": result["model_breakdown"].get("hybrid", None),
            "top_reasons": "; ".join(result.get("reasons", [])),
            "error": None,
        }
    except Exception as e:
        return {"url": url, "error": str(e)}


# Multiprocessing setup
n_workers = max(2, cpu_count() - 1)  # use all but one CPU
print(f"⚙️ Running parallel inference with {n_workers} workers...")

urls = test_df["url"].tolist()
with Pool(n_workers) as p:
    records = list(p.map(process_url, urls))


# Apply normalization
test_df["url"] = test_df["url"].apply(normalize_url)
print(f"✅ Normalized {len(test_df)} URLs for evaluation")

records = []

for i, url in enumerate(test_df["url"], 1):  # full dataset
    try:
        result = run_example(url)

        row = {
            "index": i,
            "url": result["url"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "final_prob": result["final_prob"],
            "rf_prob": result["model_breakdown"].get("rf", None),
            "xgb_prob": result["model_breakdown"].get("xgb", None),
            "lgbm_prob": result["model_breakdown"].get("lgbm", None),
            "cnn_prob": result["model_breakdown"].get("cnn", None),
            "ffnn_prob": result["model_breakdown"].get("ffnn", None),
            "lstm_prob": result["model_breakdown"].get("lstm", None),
            "hybrid_prob": result["model_breakdown"].get("hybrid", None),
            "top_reasons": "; ".join(result.get("reasons", [])),
        }

        records.append(row)
        print(f"[{i}] ✅ Processed: {url}")

    except Exception as e:
        print(f"[{i}] ❌ Error processing {url}: {e}")
        records.append({"index": i, "url": url, "error": str(e)})

# ------------------------------
# 3. Save raw results
# ------------------------------
results_df = pd.DataFrame(records)
results_df.to_csv("models/xai_eval_results.csv", index=False, encoding="utf-8")
results_df.to_excel("models/xai_eval_results.xlsx", index=False)
print("\n✅ Raw results saved to:")
print(" - models/xai_eval_results.csv")
print(" - models/xai_eval_results.xlsx")

# ------------------------------
# 4. Merge with ground-truth labels
# ------------------------------
merged = results_df.merge(test_df[["url", "label"]], on="url", how="left")
merged.to_csv("models/xai_with_labels.csv", index=False)
print("✅ Merged results (with labels) saved to models/xai_with_labels.csv")

# ------------------------------
# 5. Compute classification metrics
# ------------------------------
# Convert hybrid_prob → binary prediction (0=Benign, 1=Phishing)
merged = merged.dropna(subset=["hybrid_prob"])
preds = (merged["hybrid_prob"] > 0.5).astype(int)

print("\n=== Classification Report (Threshold = 0.5) ===")
print(
    classification_report(
        merged["label"], preds, target_names=["Benign", "Phishing"], digits=4
    )
)

# Optional summary count
summary = merged["verdict"].value_counts()
print("\n=== Verdict Summary ===")
print(summary)
"""


# ------------------------------
# 1. Load and normalize dataset
# old 2
# ------------------------------
"""
test_df = pd.read_csv("data/processed/test.csv")

if "label" not in test_df.columns:
    raise ValueError("Dataset must include a 'label' column (0=Benign, 1=Phishing).")

print(f"Loaded {len(test_df)} URLs for evaluation...")


def normalize_url(u: str) -> str:
    Ensure all URLs start with http(s) and contain only printable ASCII.
    u = str(u).strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    u = re.sub(r"[^\x20-\x7e]", "", u)  # remove non-printable chars
    return u


test_df["url"] = test_df["url"].apply(normalize_url)
print(f"✅ Normalized {len(test_df)} URLs for evaluation.\n")


# ------------------------------
# 2. Define inference worker
# ------------------------------
def process_url(url):
    try:
        result = run_example(url)
        return {
            "url": result["url"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "final_prob": result["final_prob"],
            "rf_prob": result["model_breakdown"].get("rf", None),
            "xgb_prob": result["model_breakdown"].get("xgb", None),
            "lgbm_prob": result["model_breakdown"].get("lgbm", None),
            "cnn_prob": result["model_breakdown"].get("cnn", None),
            "ffnn_prob": result["model_breakdown"].get("ffnn", None),
            "lstm_prob": result["model_breakdown"].get("lstm", None),
            "hybrid_prob": result["model_breakdown"].get("hybrid", None),
            "top_reasons": "; ".join(result.get("reasons", [])),
            "error": None,
        }
    except Exception as e:
        return {"url": url, "error": str(e)}


# ------------------------------
# 3. Run parallel inference
# ------------------------------
if __name__ == "__main__":
    from tqdm import tqdm
    import time

    # n_workers = max(2, cpu_count() - 1)
    n_workers = 4  # or set to desired number of workers
    print(f"⚙️ Running parallel inference with {n_workers} workers...")

    urls = test_df["url"].tolist()
    with Pool(n_workers) as p:
        # Use imap to stream results progressively
        records = []
        for result in tqdm(
            p.imap(process_url, urls),
            total=len(urls),
            desc="Processing URLs",
            ncols=100,
        ):
            if result is not None:
                records.append(result)
            else:
                # Optional small delay avoids flicker on Windows console
                time.sleep(0.001)

    print("\n✅ Inference complete.\n")

    # ------------------------------
    # 4. Save raw results
    # ------------------------------
    results_df = pd.DataFrame(records)
    results_df.to_csv("models/xai_eval_results.csv", index=False, encoding="utf-8")
    results_df.to_excel("models/xai_eval_results.xlsx", index=False)

    print("✅ Raw results saved:")
    print(" - models/xai_eval_results.csv")
    print(" - models/xai_eval_results.xlsx\n")

    # ------------------------------
    # 5. Merge with labels and evaluate
    # ------------------------------
    merged = results_df.merge(test_df[["url", "label"]], on="url", how="left")
    merged.to_csv("models/xai_with_labels.csv", index=False)
    print("✅ Merged results (with labels) saved to models/xai_with_labels.csv\n")

    merged = merged.dropna(subset=["hybrid_prob"])
    preds = (merged["hybrid_prob"] > 0.5).astype(int)

    print("=== Classification Report (Threshold = 0.5) ===")
    print(
        classification_report(
            merged["label"], preds, target_names=["Benign", "Phishing"], digits=4
        )
    )

    summary = merged["verdict"].value_counts()
    print("\n=== Verdict Summary ===")
    print(summary)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
"""


# ------------------------------
# 1. Load and normalize dataset
# ------------------------------
test_df = pd.read_csv("data/processed/test.csv")

partial_path = "models/xai_eval_partial.csv"
if os.path.exists(partial_path):
    done = pd.read_csv(partial_path)
    processed_urls = set(done["url"])
    test_df = test_df[~test_df["url"].isin(processed_urls)]
    print(
        f"🔄 Resuming: {len(processed_urls)} URLs already done, {len(test_df)} remaining."
    )
else:
    print("🆕 Starting fresh, no checkpoint found.")

if "label" not in test_df.columns:
    raise ValueError("Dataset must include a 'label' column (0=Benign, 1=Phishing).")

print(f"Loaded {len(test_df)} URLs for evaluation...")
print("Original dataset:", len(test_df) + len(done))


def normalize_url(u: str) -> str:
    """Ensure all URLs start with http(s) and contain only printable ASCII."""
    u = str(u).strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    u = re.sub(r"[^\x20-\x7E]", "", u)  # remove non-printable chars
    return u


test_df["url"] = test_df["url"].apply(normalize_url)
print(f"✅ Normalized {len(test_df)} URLs for evaluation.\n")


# ------------------------------
# 2. Define inference worker
# ------------------------------
def process_url(url):
    try:
        result = run_example(url)
        return {
            "url": result["url"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "final_prob": result["final_prob"],
            "rf_prob": result["model_breakdown"].get("rf", None),
            "xgb_prob": result["model_breakdown"].get("xgb", None),
            "lgbm_prob": result["model_breakdown"].get("lgbm", None),
            "cnn_prob": result["model_breakdown"].get("cnn", None),
            "ffnn_prob": result["model_breakdown"].get("ffnn", None),
            "lstm_prob": result["model_breakdown"].get("lstm", None),
            "hybrid_prob": result["model_breakdown"].get("hybrid", None),
            "top_reasons": "; ".join(result.get("reasons", [])),
            "error": None,
        }
    except Exception as e:
        return {"url": url, "error": str(e)}


# ------------------------------
# 3. Run parallel inference
# ------------------------------
if __name__ == "__main__":
    # sys.stdout = open("models/xai_eval_log.txt", "a", encoding="utf-8")
    n_workers = 3  # safer for 2GB MX350 GPU
    print(f"⚙️ Running parallel inference with {n_workers} workers...\n")

    urls = test_df["url"].tolist()

    records = []
    # records = []
    for result in tqdm(
        map(process_url, urls), total=len(urls), desc="Processing URLs", ncols=100
    ):
        if result is not None:
            records.append(result)
        if len(records) % 1000 == 0 and len(records) > 0:
            df_new = pd.DataFrame(records)
            partial_path = "models/xai_eval_partial.csv"

            # If file exists, append only new entries
            if os.path.exists(partial_path):
                df_old = pd.read_csv(partial_path)
                combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["url"])

                # combined.to_csv(partial_path, index=False, encoding="utf-8")
                temp_path = partial_path + ".tmp"
                combined.to_csv(temp_path, index=False, encoding="utf-8")
                os.replace(temp_path, partial_path)
            else:
                df_new.to_csv(partial_path, index=False, encoding="utf-8")

            print(
                f"💾 Checkpoint saved at {len(records)} URLs total (merged).",
                flush=True,
            )

    print("\n✅ Inference complete.\n")
    sys.stdout = open("models/xai_eval_log.txt", "a", encoding="utf-8")
    # with Pool(n_workers) as p:
    #     for result in tqdm(
    #         p.imap(process_url, urls),
    #         total=len(urls),
    #         desc="Processing URLs",
    #         ncols=100,
    #     ):
    #         if result is not None:
    #             records.append(result)
    #         else:
    #             time.sleep(0.001)

    # print("\n✅ Inference complete.\n")

    # ------------------------------
    # 4. Save raw results
    # ------------------------------
    results_df = pd.DataFrame(records)
    results_df.to_csv("models/xai_eval_results.csv", index=False, encoding="utf-8")
    results_df.to_excel("models/xai_eval_results.xlsx", index=False)

    print("✅ Raw results saved:")
    print(" - models/xai_eval_results.csv")
    print(" - models/xai_eval_results.xlsx\n")

    # ------------------------------
    # 5. Merge with labels and evaluate
    # ------------------------------
    merged = results_df.merge(test_df[["url", "label"]], on="url", how="left")
    merged.to_csv("models/xai_with_labels.csv", index=False)
    print("✅ Merged results (with labels) saved to models/xai_with_labels.csv\n")

    merged = merged.dropna(subset=["hybrid_prob"])
    preds = (merged["hybrid_prob"] > 0.5).astype(int)

    print("=== Classification Report (Threshold = 0.5) ===")
    print(
        classification_report(
            merged["label"], preds, target_names=["Benign", "Phishing"], digits=4
        )
    )

    summary = merged["verdict"].value_counts()
    print("\n=== Verdict Summary ===")
    print(summary)
