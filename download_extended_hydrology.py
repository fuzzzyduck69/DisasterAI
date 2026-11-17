"""Download extended hydrological telemetry from USGS Water Services.
Expands real gauge data across additional stations and longer observation windows.
"""
import io
import time
import urllib.request
import pandas as pd
from pathlib import Path

OUT_FILE = Path("data/hydrology/real_river_gauge_telemetry.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 10 diverse river basin stations covering coastal, major continental, and urban systems
STATIONS = [
    ("08074000", "Buffalo Bayou at Houston, TX", "Coastal Plain / Urban"),
    ("01646500", "Potomac River near Washington, DC", "Piedmont / Riverine"),
    ("07374000", "Mississippi River at Baton Rouge, LA", "Lowland Mississippi Delta"),
    ("02466030", "Black Warrior River at Northport, AL", "Appalachian Plateau"),
    ("11447650", "Sacramento River at Freeport, CA", "Sacramento Valley / Tidal Delta"),
    ("08073600", "Buffalo Bayou at West Belt Dr, Houston, TX", "Upper Buffalo Bayou"),
    ("08074500", "Whiteoak Bayou at Houston, TX", "High-Velocity Urban Bayou"),
    ("08068000", "San Jacinto River near Conroe, TX", "East Texas Watershed"),
    ("01643000", "Potomac River near Point of Rocks, MD", "Upper Potomac Basin"),
    ("07289000", "Mississippi River at Vicksburg, MS", "Central Mississippi Valley")
]

records = []
print(f"Fetching 60-day 15-minute telemetry for {len(STATIONS)} USGS monitoring stations...")

for site_no, site_name, basin_desc in STATIONS:
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site_no}&period=P60D&parameterCd=00065,00060"
    print(f"  Querying station {site_no} ({site_name})...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        
        lines = [l for l in content.split("\n") if not l.startswith("#") and l.strip()]
        if len(lines) < 3:
            print("No data rows returned.")
            continue
        
        header_cols = lines[0].split("\t")
        data_rows = [l.split("\t") for l in lines[2:] if l.strip()]
        
        # Identify stage and discharge column indices
        stage_col_idx = None
        flow_col_idx = None
        for i, col in enumerate(header_cols):
            if "00065" in col and not col.endswith("_cd"):
                stage_col_idx = i
            elif "00060" in col and not col.endswith("_cd"):
                flow_col_idx = i
        
        site_parsed = 0
        for r in data_rows:
            if len(r) > max(stage_col_idx or 0, flow_col_idx or 0, 2):
                dt = r[2] # datetime column
                stage = r[stage_col_idx] if stage_col_idx is not None and stage_col_idx < len(r) else None
                flow = r[flow_col_idx] if flow_col_idx is not None and flow_col_idx < len(r) else None
                try:
                    s_val = float(stage) if stage and stage.strip() not in ("", "Rat", "Ice", "Eqp") else None
                    f_val = float(flow) if flow and flow.strip() not in ("", "Rat", "Ice", "Eqp") else None
                    if s_val is not None and f_val is not None:
                        records.append({
                            "timestamp": dt,
                            "station_id": site_no,
                            "station_name": site_name,
                            "basin_description": basin_desc,
                            "river_stage_ft": s_val,
                            "streamflow_cfs": f_val
                        })
                        site_parsed += 1
                except ValueError:
                    continue
        print(f"OK ({site_parsed:,} valid readings)")
    except Exception as exc:
        print(f"FAILED ({exc})")

if records:
    df_new = pd.DataFrame(records)
    df_new.to_csv(OUT_FILE, index=False)
    print(f"\nSuccessfully saved {len(df_new):,} hydrological observations across {df_new['station_id'].nunique()} stations to {OUT_FILE}")
else:
    print("\nNo records fetched; keeping existing hydrology file.")
