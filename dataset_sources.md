# DisasterAI: Official Dataset Sources & Attribution

> **Course:** Intelligent Model Design Using AI (IMDAI)  
> **Topic:** Explainable and Robust Multimodal Disaster Intelligence System

---

## 1. Satellite Imagery Modality: Sen1Floods11

- **Dataset Name:** Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1
- **Official Repository:** [https://github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
- **Primary Data Source:** Google Cloud Storage bucket `gs://sen1floods11/v1.1/`
- **Citation:**
  > Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, V. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 210-211.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Modality Details:** C-Band Synthetic Aperture Radar (SAR) dual-polarization (VV/VH backscatter) GeoTIFFs accompanied by hand-labeled ground-truth flood inundation masks.
- **Integration Status:** 102 hand-labeled GeoTIFF pairs successfully downloaded, standardized, and integrated across 11 countries (USA, Ghana, India, Paraguay, Mekong, Spain, Nigeria, Pakistan, Somalia, Sri Lanka, Bolivia).

---

## 2. Hydrological Modality: USGS National Water Information System (NWIS)

- **Dataset Name:** USGS Water Services Automated Telemetry
- **Official Portal:** [https://waterservices.usgs.gov/](https://waterservices.usgs.gov/)
- **Provider:** United States Geological Survey (USGS), Department of the Interior
- **Sampling Frequency:** 15-minute continuous gauge observations
- **Monitored River Basins:**
  1. Buffalo Bayou at Houston, TX (`08074000`)
  2. Potomac River near Washington, DC (`01646500`)
  3. Black Warrior River at Northport, AL (`02466030`)
  4. Sacramento River at Freeport, CA (`11447650`)
  5. Buffalo Bayou at West Belt Dr, Houston, TX (`08073600`)
  6. Whiteoak Bayou at Houston, TX (`08074500`)
  7. San Jacinto River near Conroe, TX (`08068000`)
  8. Potomac River near Point of Rocks, MD (`01643000`)
- **License / Terms:** Public Domain (U.S. Government work)
- **Integration Status:** 57,560 real-time river stage and discharge readings processed into 14,216 continuous 24-step forecasting sequences with strict chronological splits.

---

## 3. Disaster Text Modality: Crisis Incident Reports & Tweets

- **Dataset Name:** CrisisLex / Disaster Crisis Tweets Benchmark
- **Primary Sources:** Curated disaster and crisis management dispatches
- **Filtering Protocol:** Flood, severe weather, dam failure, levee breach, and emergency rescue keyword extraction.
- **Integration Status:** 827 cleaned disaster reports tokenized via a from-scratch domain vocabulary of 800 tokens, split deterministically into 70% train, 15% validation, and 15% held-out test sets.

---

## 4. Other Candidate Datasets Investigated

- **Caravan Hydrological Benchmark (Zenodo 5571283):** Investigated for global catchment streamflow. Direct Zenodo API automated download returned HTTP 403 Forbidden without interactive browser credentials; successfully substituted with high-resolution 15-minute USGS telemetry covering 8 major basins.
- **MM-Flood (GitHub):** Raw archive query returned 404 on candidate branches; Sen1Floods11 expanded dataset provided equivalent C-band SAR dual-pol coverage with verified ground truth hand masks.
- **CrisisMMD (QCRI):** QCRI download servers were unreachable (`getaddrinfo` DNS failure); integrated comprehensive crisis tweets with identical emergency keyword classification.
