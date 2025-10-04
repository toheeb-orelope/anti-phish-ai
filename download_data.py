import os
import requests
import zipfile

# Create data/raw directory if not exists
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


# -----------------------------
# PhishTank latest snapshot
# -----------------------------
def download_phishtank():
    url = "http://data.phishtank.com/data/online-valid.csv"
    path = os.path.join(RAW_DIR, "verified_online.csv")

    print("[*] Downloading PhishTank feed...")
    r = requests.get(url, timeout=60)
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"[+] Saved PhishTank dataset to {path}")


# -----------------------------
# Alexa Top 1M (Kaggle mirror)
# -----------------------------
def download_alexa():
    # Kaggle datasets require authentication, so here we use direct Kaggle CLI
    print("[*] Downloading Alexa Top 1M (requires Kaggle API setup)...")
    os.system(f"kaggle datasets download -d cheedcheed/top1m -p {RAW_DIR} --unzip")
    print(f"[+] Saved Alexa Top 1M dataset to {RAW_DIR}/top-1m.csv")


# -----------------------------
# Malicious URLs dataset (Kaggle)
# -----------------------------
def download_malicious_urls():
    print("[*] Downloading Malicious URLs dataset (requires Kaggle API setup)...")
    os.system(
        f"kaggle datasets download -d sid321axn/malicious-urls-dataset -p {RAW_DIR} --unzip"
    )
    print(f"[+] Saved Malicious URLs dataset to {RAW_DIR}")


# -----------------------------
# PhiUSIIL Dataset (UCI)
# -----------------------------
def download_phiusiil():
    # UCI provides direct links; this one may need manual update if URL changes
    url = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
    zip_path = os.path.join(RAW_DIR, "phiusiil.zip")

    print("[*] Downloading PhiUSIIL dataset from UCI...")
    r = requests.get(url, timeout=120)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print(f"[+] Extracted PhiUSIIL dataset to {RAW_DIR}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("=== Downloading phishing datasets ===")
    try:
        download_phishtank()
    except Exception as e:
        print(f"[!] Failed PhishTank: {e}")

    try:
        download_alexa()
    except Exception as e:
        print(f"[!] Failed Alexa: {e}")

    try:
        download_malicious_urls()
    except Exception as e:
        print(f"[!] Failed Malicious URLs: {e}")

    try:
        download_phiusiil()
    except Exception as e:
        print(f"[!] Failed PhiUSIIL: {e}")

    print("=== Done ===")
