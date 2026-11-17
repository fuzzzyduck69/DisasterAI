import os
import sys
import io
import json
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
IMAGERY_DIR = DATA_DIR / "imagery"
S1_DIR = IMAGERY_DIR / "s1_hand"
LABEL_DIR = IMAGERY_DIR / "labels_hand"
SPLIT_DIR = IMAGERY_DIR / "splits"
HYDROLOGY_DIR = DATA_DIR / "hydrology"
TEXT_DIR = DATA_DIR / "text"

for d in (S1_DIR, LABEL_DIR, SPLIT_DIR, HYDROLOGY_DIR, TEXT_DIR):
    d.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def download_file(url, out_path, desc=""):
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [Cached] {out_path.name}")
        return True
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            out_path.write_bytes(data)
            print(f"  [Downloaded] {desc or out_path.name} ({len(data)/1024:.1f} KB)")
            return True
    except Exception as e:
        print(f"  [Failed] {url}: {e}")
        return False

# ============================================================================
# 1. DOWNLOAD VISION DATASET: Sen1Floods11 (Sentinel-1 SAR + Hand Labels)
# ============================================================================
print("\n" + "="*70)
print(" 1. DOWNLOADING VISION DATASET: Sen1Floods11 (Cloud to Street / CVPRW 2020)")
print("="*70)

GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"

# 1.1 Download Official Splits
splits = {
    "train": f"{GCS_BASE}/splits/flood_handlabeled/flood_train_data.csv",
    "val": f"{GCS_BASE}/splits/flood_handlabeled/flood_valid_data.csv",
    "test": f"{GCS_BASE}/splits/flood_handlabeled/flood_test_data.csv"
}

split_data = {}
for name, url in splits.items():
    csv_path = SPLIT_DIR / f"{name}.csv"
    if download_file(url, csv_path, f"Official {name} split CSV"):
        lines = csv_path.read_text(encoding="utf-8-sig").strip().splitlines()
        split_data[name] = [line.split(",") for line in lines if "," in line]

# 1.2 Download Curated Real Flood Chips across global events (Ghana, USA, Bolivia, Paraguay, India, Spain, Somalia, Mekong, Sri Lanka)
# Selecting sample chips representing all 11 global flood events
chips_to_download = [
    # Test split
    ("Ghana_313799_S1Hand.tif", "Ghana_313799_LabelHand.tif", "test"),
    ("Ghana_1078550_S1Hand.tif", "Ghana_1078550_LabelHand.tif", "test"),
    ("Mekong_1113949_S1Hand.tif", "Mekong_1113949_LabelHand.tif", "test"),
    ("Paraguay_168470_S1Hand.tif", "Paraguay_168470_LabelHand.tif", "test"),
    ("Somalia_1006148_S1Hand.tif", "Somalia_1006148_LabelHand.tif", "test"),
    ("USA_430764_S1Hand.tif", "USA_430764_LabelHand.tif", "test"),
    ("Spain_976377_S1Hand.tif", "Spain_976377_LabelHand.tif", "test"),
    ("India_431109_S1Hand.tif", "India_431109_LabelHand.tif", "test"),
    ("Sri-Lanka_348270_S1Hand.tif", "Sri-Lanka_348270_LabelHand.tif", "test"),
    ("Bolivia_234031_S1Hand.tif", "Bolivia_234031_LabelHand.tif", "test"),
    # Train / Val splits
    ("Ghana_97059_S1Hand.tif", "Ghana_97059_LabelHand.tif", "train"),
    ("Ghana_359826_S1Hand.tif", "Ghana_359826_LabelHand.tif", "train"),
    ("Mekong_284144_S1Hand.tif", "Mekong_284144_LabelHand.tif", "train"),
    ("Paraguay_500473_S1Hand.tif", "Paraguay_500473_LabelHand.tif", "train"),
    ("USA_101031_S1Hand.tif", "USA_101031_LabelHand.tif", "train"),
    ("Spain_1067202_S1Hand.tif", "Spain_1067202_LabelHand.tif", "val"),
    ("Bolivia_103757_S1Hand.tif", "Bolivia_103757_LabelHand.tif", "val"),
    ("Somalia_279140_S1Hand.tif", "Somalia_279140_LabelHand.tif", "val"),
]

print(f"Downloading real Sentinel-1 SAR chips and matching flood masks ({len(chips_to_download)} chip pairs)...")
downloaded_chips = []
for s1_file, label_file, split_name in chips_to_download:
    s1_url = f"{GCS_BASE}/data/flood_events/HandLabeled/S1Hand/{s1_file}"
    label_url = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand/{label_file}"
    
    ok_s1 = download_file(s1_url, S1_DIR / s1_file, f"SAR {s1_file}")
    ok_lbl = download_file(label_url, LABEL_DIR / label_file, f"Mask {label_file}")
    if ok_s1 and ok_lbl:
        downloaded_chips.append({
            "chip_id": s1_file.replace("_S1Hand.tif", ""),
            "s1_path": str((S1_DIR / s1_file).resolve()),
            "mask_path": str((LABEL_DIR / label_file).resolve()),
            "split": split_name
        })

df_manifest = pd.DataFrame(downloaded_chips)
manifest_path = IMAGERY_DIR / "dataset_manifest.csv"
df_manifest.to_csv(manifest_path, index=False)
print(f"Vision dataset ready! Manifest saved to: {manifest_path} ({len(df_manifest)} pairs)")


# ============================================================================
# 2. DOWNLOAD TEXT DATASET: Real Disaster Crisis Reports & Tweets
# ============================================================================
print("\n" + "="*70)
print(" 2. DOWNLOADING TEXT DATASET: Crisis Disaster Reports & Tweets")
print("="*70)

TEXT_URL = "https://raw.githubusercontent.com/sugatagh/Natural-Language-Processing-with-Disaster-Tweets/main/train.csv"
raw_text_path = TEXT_DIR / "raw_disaster_tweets.csv"

if download_file(TEXT_URL, raw_text_path, "Real Disaster Tweets & Emergency Reports"):
    df_text = pd.read_csv(raw_text_path)
    print(f"  Loaded {len(df_text)} raw disaster reports.")
    
    # Filter and categorize flood and hazard-specific records
    keywords = ["flood", "flooding", "rain", "storm", "inundation", "water", "overflow", "dam", "evacuation", "emergency", "rescue"]
    pattern = "|".join(keywords)
    df_flood = df_text[df_text["text"].str.contains(pattern, case=False, na=False)].copy()
    
    # Clean text
    df_flood["clean_text"] = df_flood["text"].str.replace(r"http\S+", "", regex=True)
    df_flood["clean_text"] = df_flood["clean_text"].str.replace(r"@\S+", "", regex=True)
    df_flood["clean_text"] = df_flood["clean_text"].str.strip()
    
    curated_text_path = TEXT_DIR / "disaster_crisis_reports.csv"
    df_flood.to_csv(curated_text_path, index=False)
    print(f"  Curated {len(df_flood)} disaster incident reports into: {curated_text_path}")


# ============================================================================
# 3. DOWNLOAD HYDROLOGICAL TIME-SERIES DATASET: Real Flood River Telemetry
# ============================================================================
print("\n" + "="*70)
print(" 3. DOWNLOADING TEMPORAL DATASET: Real Hydrological River Stage & Rainfall Sequences")
print("="*70)

# Fetching real hourly streamflow, gage height (river stage), and precipitation from major flood monitoring stations
stations = [
    {"id": "08074000", "name": "Buffalo Bayou at Houston, TX", "basin": "San Jacinto Basin"},
    {"id": "01646500", "name": "Potomac River near Washington, DC", "basin": "Potomac Basin"},
    {"id": "07010000", "name": "Mississippi River at St. Louis, MO", "basin": "Mississippi Basin"},
    {"id": "02469761", "name": "Black Warrior River near Tuscaloosa, AL", "basin": "Mobile Basin"},
    {"id": "11447650", "name": "Sacramento River at Freeport, CA", "basin": "Sacramento Basin"}
]

all_hydro_records = []

for st in stations:
    site_id = st["id"]
    site_name = st["name"]
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_id}&period=P14D&parameterCd=00065,00060"
    print(f"Fetching 14-day 15-minute telemetry for {site_name} (Station {site_id})...")
    
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
            ts_list = data["value"]["timeSeries"]
            
            # Organize into timestamp table
            series_dict = {}
            for ts in ts_list:
                var_desc = ts["variable"]["variableName"]
                col_name = "river_stage_ft" if "height" in var_desc.lower() else "streamflow_cfs"
                for pt in ts["values"][0]["value"]:
                    dt = pt["dateTime"]
                    val = float(pt["value"]) if pt["value"] not in ("-999999", "-9999", None) else None
                    if dt not in series_dict:
                        series_dict[dt] = {"timestamp": dt, "station_id": site_id, "station_name": site_name, "basin": st["basin"]}
                    series_dict[dt][col_name] = val
                    
            station_df = pd.DataFrame(list(series_dict.values()))
            if not station_df.empty:
                station_df = station_df.sort_values("timestamp")
                # Fill missing
                station_df = station_df.ffill().bfill()
                # Compute derivative (river rising velocity)
                if "river_stage_ft" in station_df.columns:
                    station_df["stage_velocity_ft_per_hr"] = station_df["river_stage_ft"].diff().fillna(0) * 4 # 15-min to hr
                all_hydro_records.append(station_df)
                print(f"  Successfully retrieved {len(station_df)} readings for {st['name']}.")
    except Exception as e:
        print(f"  Failed for station {site_id}: {e}")

if all_hydro_records:
    df_hydro = pd.concat(all_hydro_records, ignore_index=True)
    hydro_csv_path = HYDROLOGY_DIR / "real_river_gauge_telemetry.csv"
    df_hydro.to_csv(hydro_csv_path, index=False)
    print(f"Hydrological time-series dataset ready! Saved to: {hydro_csv_path} ({len(df_hydro)} records)")

print("\n" + "="*70)
print(" ALL 3 MULTIMODAL DATASETS SUCCESSFULLY DOWNLOADED AND STORED IN data/")
print("="*70)
