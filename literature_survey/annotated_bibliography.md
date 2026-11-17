# Annotated Bibliography & Deep-Dive Paper Analysis

**Course:** Intelligent Model Design Using AI (IMDAI)  
**Project:** DisasterAI — An Explainable and Robust Multimodal Disaster Intelligence System  

This document provides a detailed breakdown of the primary research papers forming the theoretical and empirical foundation of **DisasterAI**.

---

## 1. Multimodal Social Media & Crisis Informatics

### Paper 1: Using AI and Social Media Multimodal Content for Disaster Response and Management: A Survey (2020)
- **Authors:** Muhammad Imran, Carlos Castillo, Fernando Diaz, Sarah Vieweg
- **Publication:** *Information Processing & Management*, Vol. 57, Issue 3, 102261.
- **DOI:** [10.1016/j.ipm.2020.102261](https://doi.org/10.1016/j.ipm.2020.102261)
- **Core Objective:** To systematically review how AI models process multimodal social media content (images, text, geolocation) for real-time disaster management and humanitarian response.
- **Methodology & Datasets:**
  - Surveys datasets including CrisisLex, CrisisNLP, and CrisisMMD.
  - Analyzes unimodal vs. multimodal architectures for event detection, damage assessment, and humanitarian aid requests.
- **Key Findings:**
  - Multimodal models consistently achieve higher classification accuracy than unimodal text or image classifiers.
  - Text reports often provide early notification, whereas imagery provides physical damage verification.
- **Critical Limitations:**
  - Heavy reliance on social media text which contains high colloquial noise, spam, and unverified rumors.
  - Completely omits physical environmental sensor telemetry (rainfall, river gauges) and satellite remote sensing.
- **Direct Relevance to DisasterAI:**
  - Justifies our tri-modal architecture: pairing linguistic incident reports with physical satellite imagery and hydrological measurements prevents false alarms.

---

### Paper 2: Leveraging Multimodal Social Media Data for Rapid Disaster Damage Assessment (2020)
- **Authors:** Hanyu Hao, Yandong Wang
- **Publication:** *International Journal of Disaster Risk Reduction*, Vol. 51, 101760.
- **DOI:** [10.1016/j.ijdrr.2020.101760](https://doi.org/10.1016/j.ijdrr.2020.101760)
- **Core Objective:** To develop a multimodal deep learning framework combining visual features and textual embeddings for rapid infrastructure damage assessment during extreme weather events.
- **Methodology:**
  - Vision branch: VGG-16 pretrained on ImageNet.
  - Text branch: BiLSTM trained on word embeddings.
  - Intermediate fusion via feature concatenation followed by a dense classification head.
- **Key Findings:**
  - Multimodal fusion improved damage classification F1-score by 11.4% over image-only and 8.7% over text-only models.
- **Critical Limitations:**
  - Dependent on ImageNet pretraining, which carries domain bias unsuitable for overhead remote sensing.
  - Simple static concatenation fails if one modality drops out.
- **Direct Relevance to DisasterAI:**
  - Inspired DisasterAI's **dynamic modality gating network**, which learns context-aware weights rather than naive static concatenation.

---

## 2. Remote Sensing & Flood Inundation Segmentation

### Paper 3: Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1 (2020)
- **Authors:** Derrick Bonafilia, Beth Tellman, Tyler Anderson, Erica Issenberg (Cloud to Street / CVPRW)
- **Publication:** *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 210–211.
- **Link:** [CVPRW 2020 Paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_Flood_Algorithms_CVPRW_2020_paper.html)
- **Core Objective:** To provide the remote sensing community with the first high-quality, georeferenced dataset for training deep learning flood segmentation algorithms using Sentinel-1 Synthetic Aperture Radar (SAR).
- **Methodology & Dataset:**
  - 4,831 non-overlapping $512 \times 512$ chips covering 11 global flood events across 6 continents.
  - Contains raw Sentinel-1 dual-polarization (VV + VH) GeoTIFFs, Sentinel-2 optical imagery, permanent water masks, and hand-labeled flood masks.
  - Evaluates standard Otsu thresholding and vanilla U-Net baselines.
- **Key Findings:**
  - Deep learning models (U-Net) significantly outperform traditional backscatter thresholding (Otsu) by learning spatial context and contextual terrain patterns.
  - Permanent water bodies vs. temporary floodwater distinction remains the most challenging error source.
- **Direct Relevance to DisasterAI:**
  - Defines our **SAR radar backscatter normalization pipeline** (clipping VV/VH dB backscatter to $[-50, 1]$ dB and scaling to $[0, 1]$).
  - Motivates our **Attention U-Net**, which adds attention gates to suppress non-flood water bodies and speckle noise.

---

## 3. Hydrological Time-Series & Deep Sequence Modeling

### Paper 4: Rainfall–Runoff Modelling Using Long Short-Term Memory (LSTM) Networks (2018)
- **Authors:** Frederik Kratzert, Daniel Klotz, Claire Brenner, Karsten Schulz, Mathew Herrnegger
- **Publication:** *Hydrology and Earth System Sciences (HESS)*, Vol. 22, Issue 11, pp. 6005–6022.
- **DOI:** [10.5194/hess-22-6005-2018](https://doi.org/10.5194/hess-22-6005-2018)
- **Core Objective:** To investigate whether data-driven recurrent neural networks (LSTMs) can model non-linear catchment hydrological processes without explicit physical differential equations.
- **Methodology:**
  - Evaluated on the CAMELS benchmark (hundreds of river catchments across diverse climatological zones).
  - Fed daily precipitation, minimum/maximum temperature, solar radiation, and vapor pressure sequences into standard LSTMs.
- **Key Findings:**
  - LSTM models outperformed standard conceptual hydrological models (SAC-SMA) in Nash-Sutcliffe Efficiency (NSE).
  - The cell state $c_t$ inherently acted as a dynamic representation of physical soil moisture storage and baseflow accumulation.
- **Direct Relevance to DisasterAI:**
  - Formulates our **Environmental LSTM branch**: tracking rainfall, river stage, soil saturation, humidity, and temperature over 24 hours to predict forward crest trajectories.

---

## 4. Explainability & Uncertainty in Damage Assessment

### Paper 5: Uncertainty-Aware Convolutional Neural Network for Explainable AI-Assisted Disaster Damage Assessment (2022)
- **Authors:** Min Cheng, et al.
- **Publication:** *Structural Control and Health Monitoring*, Vol. 29, Issue 12, e3019.
- **DOI:** [10.1002/stc.3019](https://doi.org/10.1002/stc.3019)
- **Core Objective:** To address the black-box nature of deep vision models in disaster damage classification by incorporating both spatial explainability (Grad-CAM) and uncertainty estimation (Monte Carlo Dropout).
- **Methodology:**
  - CNN classifier with dropout active at test time to sample epistemic uncertainty distributions.
  - Grad-CAM heatmap visualization to verify that the model attends to structural fractures and debris piles.
- **Key Findings:**
  - High epistemic uncertainty strongly correlated with borderline, difficult-to-classify damage cases.
  - Grad-CAM provided visual justification that built operator trust.
- **Direct Relevance to DisasterAI:**
  - DisasterAI adopts this exact dual paradigm: **Grad-CAM for spatial visual verification** and **Monte Carlo Dropout for epistemic uncertainty calibration** across our multimodal fusion head.

---

## 5. Adversarial Robustness & Input Perturbations

### Paper 6: Adversarial Training for Aerial Disaster Recognition: A Curriculum-Based Defense Against PGD Attacks (2025)
- **Publication:** *Electronics*, Vol. 14, Issue 16, 3210.
- **DOI:** [10.3390/electronics14163210](https://doi.org/10.3390/electronics14163210)
- **Core Objective:** To investigate the vulnerability of aerial disaster recognition systems to adversarial gradient perturbations (FGSM and PGD) and formulate defense strategies.
- **Methodology:**
  - Tested aerial drone disaster recognition backbones under varying attack budgets $\epsilon \in [0.01, 0.10]$.
  - Explored multi-step Projected Gradient Descent (PGD) and curriculum adversarial training.
- **Key Findings:**
  - Standard deep models experience catastrophic accuracy drops (exceeding 40–60%) when subtle adversarial noise or high sensor corruptions are introduced.
  - Aerial imagery is particularly vulnerable due to repetitive terrain textures and sensor noise.
- **Direct Relevance to DisasterAI:**
  - Governs Section 9 of our project: implementing **FGSM and PGD attack generators** directly within our evaluation pipeline to audit model resilience and demonstrate that multimodal redundancy mitigates unimodal visual degradation.

---

## 6. Synthesis: How DisasterAI Bridges These Works

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     Imran / Hao (Multimodal)    │      │    Bonafilia (Sen1Floods11)     │
│   Social Text + Ground Images   │      │   SAR Satellite Segmentation    │
└────────────────┬────────────────┘      └────────────────┬────────────────┘
                 │                                        │
                 └──────────────────┬─────────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       DISASTERAI CONTRIBUTION        │
                 │ 1. Zero Pretrained Weights           │
                 │ 2. Tri-Modal: SAR + Hydro + Text     │
                 │ 3. Kratzert LSTM + Scratch Transf.   │
                 │ 4. Cheng XAI & MC Dropout            │
                 │ 5. PGD / FGSM Adversarial Robustness │
                 │ 6. Interactive Command Center UI     │
                 └──────────────────────────────────────┘
                                    ▲
                                    │
                 ┌──────────────────┴─────────────────────┐
                 │                                        │
┌────────────────┴────────────────┐      ┌────────────────┴────────────────┐
│   Kratzert et al. (Hydrology)   │      │    Cheng / Electronics (2025)   │
│     LSTM Rainfall-Runoff        │      │     XAI, Uncertainty & PGD      │
└─────────────────────────────────┘      └─────────────────────────────────┘
```
