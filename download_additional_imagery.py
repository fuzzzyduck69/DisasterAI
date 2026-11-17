"""Download additional official Sen1Floods11 chips.
Downloads additional hand-labeled SAR image and mask pairs from the official GCS repository.
"""
import csv
import os
import time
import urllib.request
from pathlib import Path

BASE_S1 = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/"
BASE_LBL = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/"

S1_DIR = Path("data/imagery/s1_hand")
LBL_DIR = Path("data/imagery/labels_hand")
S1_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# Read all candidates from official splits
split_files = {
    "train": "data/imagery/splits/train.csv",
    "val": "data/imagery/splits/val.csv",
    "test": "data/imagery/splits/test.csv"
}

to_download = []
for split, s_path in split_files.items():
    with open(s_path) as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) >= 2:
                s1_name = r[0].strip()
                lbl_name = r[1].strip()
                chip_id = s1_name.replace("_S1Hand.tif", "")
                country = chip_id.split("_")[0]
                s1_path = S1_DIR / s1_name
                lbl_path = LBL_DIR / lbl_name
                if not (s1_path.exists() and lbl_path.exists()):
                    to_download.append({
                        "chip_id": chip_id,
                        "country": country,
                        "split": split,
                        "s1_name": s1_name,
                        "lbl_name": lbl_name,
                        "s1_path": s1_path,
                        "lbl_path": lbl_path
                    })

print(f"Total chips missing locally: {len(to_download)}")

# Download a balanced quota of up to 40 additional chips across countries & splits
# Priority: 4 chips per country (10 countries = 40 chips)
download_quota = {}
selected = []
for item in to_download:
    c = item["country"]
    if download_quota.get(c, 0) < 4:
        selected.append(item)
        download_quota[c] = download_quota.get(c, 0) + 1

print(f"Selected {len(selected)} chips to download across countries: {download_quota}")

success_count = 0
for idx, item in enumerate(selected, 1):
    c_id = item["chip_id"]
    s1_url = f"{BASE_S1}{item['s1_name']}"
    lbl_url = f"{BASE_LBL}{item['lbl_name']}"
    print(f"[{idx}/{len(selected)}] Downloading {c_id} ({item['split']})...", end=" ", flush=True)
    t0 = time.time()
    try:
        req_s1 = urllib.request.Request(s1_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_s1, timeout=30) as resp:
            data_s1 = resp.read()
        req_lbl = urllib.request.Request(lbl_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_lbl, timeout=30) as resp:
            data_lbl = resp.read()
        
        with open(item["s1_path"], "wb") as f:
            f.write(data_s1)
        with open(item["lbl_path"], "wb") as f:
            f.write(data_lbl)
        
        dur = time.time() - t0
        print(f"OK ({len(data_s1):,} + {len(data_lbl):,} bytes in {dur:.1f}s)")
        success_count += 1
    except Exception as exc:
        print(f"FAILED: {exc}")

print(f"\nDownload summary: {success_count}/{len(selected)} chips successfully downloaded.")
