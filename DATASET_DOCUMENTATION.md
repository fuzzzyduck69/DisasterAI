# DisasterAI: Comprehensive Dataset Documentation

> **Project:** DisasterAI — Multimodal Flood Intelligence and Prediction System  
> **Revision:** Version 3 (Post-Expansion)  
> **Date:** November 2026

---

## 1. Executive Summary & Provenance

To transition DisasterAI from a toy experimental prototype to a scientifically defensible flood intelligence platform, the multimodal dataset was expanded by acquiring real-world data across all three modalities:

| Modality | Primary Dataset Source | Type & Resolution | Original Samples | Expanded Samples | Increase |
|---|---|---|---|---|---|
| **Satellite Imagery** | Cloud to Street / NASA / ESA **Sen1Floods11** | Dual-Pol VV/VH SAR GeoTIFFs (10m res, $128 \times 128$ normalized) | 59 chips | **102 valid chips** | **+72.9%** |
| **Hydrology** | **USGS Water Services NWIS** Telemetry | 15-minute stage (ft), discharge (cfs), rate of rise (ft/hr) across 8 major river basins | 2,058 sequences | **14,216 continuous sequences** | **+590.8%** |
| **Crisis Text** | Real Disaster Crisis Dispatches & Verified Incident Reports | Tokenized crisis reports ($4 \le \text{len} \le 45$), disaster vocab size 800 | 669 reports | **827 cleaned reports** | **+23.6%** |

---

## 2. Satellite SAR Imagery (Sen1Floods11)

### 2.1 Sensor & Radiometric Details
- **Satellite:** Sentinel-1 C-Band Synthetic Aperture Radar (SAR)
- **Polarizations:** Dual-polarization (VV: Vertical transmit / Vertical receive, VH: Vertical transmit / Horizontal receive)
- **Physical Dynamics:** Specular reflection off flat floodwaters causes steep backscatter drops (typically $-20\text{ dB}$ to $-35\text{ dB}$) compared to diffuse scattering over soil and double-bounce scattering in urban/vegetated canopies.
- **Normalization Protocol:** Raw decibels clipped to $[-50.0, 1.0]\text{ dB}$ and mapped linearly:
  $$\text{SAR}_{\text{norm}} = \frac{\text{clip}(\text{dB}, -50.0, 1.0) + 50.0}{51.0} \in [0.0, 1.0]$$
- **Target Size:** Standardized to $128 \times 128$ pixels via bilinear interpolation for SAR backscatter and nearest-neighbor for ground truth masks.

### 2.2 Geographic Distribution
The 102 cleaned chips span 11 countries and diverse geomorphic biomes:
- **USA:** 10 chips
- **Ghana:** 11 chips
- **India:** 10 chips
- **Mekong Basin:** 10 chips
- **Nigeria:** 10 chips
- **Pakistan:** 10 chips
- **Paraguay:** 10 chips
- **Somalia:** 10 chips
- **Spain:** 10 chips
- **Sri Lanka:** 10 chips
- **Bolivia:** 1 chip

### 2.3 Flood Extent & Severity Definition
Flood severity classes are project-defined based on hand-labeled water pixel coverage over valid surveyed areas:
- **Class 0 (Low / Routine Drainage):** Inundated area $\le 5.0\%$ of valid pixels
- **Class 1 (Moderate Flood Risk):** Inundated area $> 5.0\%$ and $\le 25.0\%$
- **Class 2 (Severe Inundation Emergency):** Inundated area $> 25.0\%$

**Class Counts Across All 102 Chips:**
- Class 0: 63 chips (61.8%)
- Class 1: 20 chips (19.6%)
- Class 2: 19 chips (18.6%)

---

## 3. Hydrological River Gauge Telemetry (USGS)

### 3.1 Monitoring Stations
57,560 continuous 15-minute readings were obtained across 8 major river monitoring stations representing diverse hydrological regimes:
1. `08074000` — Buffalo Bayou at Houston, TX (Coastal Plain / Urban Watershed)
2. `01646500` — Potomac River near Washington, DC (Piedmont / Riverine System)
3. `02466030` — Black Warrior River at Northport, AL (Appalachian Plateau)
4. `11447650` — Sacramento River at Freeport, CA (Sacramento Valley / Tidal Delta)
5. `08073600` — Buffalo Bayou at West Belt Dr, Houston, TX (Upper Buffalo Bayou)
6. `08074500` — Whiteoak Bayou at Houston, TX (High-Velocity Urban Drainage)
7. `08068000` — San Jacinto River near Conroe, TX (East Texas Watershed)
8. `01643000` — Potomac River near Point of Rocks, MD (Upper Potomac Basin)

### 3.2 Sequence Construction & Leakage Prevention
- **Window Length:** 24 timesteps (6 hours of continuous 15-min observations).
- **Forecast Horizon:** 8 timesteps ahead (2-hour forward peak stage prediction).
- **Stride:** 4 timesteps between sliding sequence windows.
- **Strict Chronological Splits:**
  - **Train:** First 70% of continuous timeline (10,013 windows)
  - **Validation:** Next 15% of continuous timeline (2,101 windows)
  - **Held-Out Test:** Final 15% of continuous timeline (2,102 windows)
- **Zero Temporal Leakage:** Sequence inputs and forecast targets never cross chronological split boundaries.
- **Normalization:** Robust Median and Interquartile Range (IQR) computed strictly over training windows.

---

## 4. Disaster Crisis Reports & Dispatches

### 4.1 Corpus Compilation
- Combined raw emergency dispatches and tweets into a deduplicated corpus of 827 disaster texts.
- Text cleaning: HTML unescaping, URL removal, mention stripping, emoji stripping, and punctuation token padding.
- Keyword filtering focused on flood, inundation, dam breaches, flash floods, and rescue dispatches.
- Vocabulary: 800 domain tokens fitted strictly on training reports.

### 4.2 Split Integrity
- Split via deterministic SHA-256 hash on unique record IDs (70% train, 15% val, 15% test).
- Prevents cross-split duplication of identical disaster reports.

---

## 5. Cross-Modal Alignment & Scientific Disclosure

**Mandatory Scientific Disclosure:**
While the SAR chips, USGS hydrology, and crisis tweets represent genuine, real-world monitoring data, **they originate from distinct monitoring systems across different geographical regions and dates**. 

Consequently, cross-modal pairings during multimodal training and evaluation are **severity-stratified proxies**, not concurrent physical observations of identical flood events. The system documents this transparently (`pairing_mode: severity_stratified_proxy_unverified_event_match`), preventing unverified physical claims while demonstrating valid intermediate representation fusion.
