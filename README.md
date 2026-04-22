# Tree Crown Detection from Satellite/Aerial Imagery

**Classical Computer Vision Pipeline for Environmental Monitoring — Forest
Area Segmentation and Individual Tree Detection**

Module: **CT103-3-M-IPCV** (Image Processing and Computer Vision)
Programme: MSc Artificial Intelligence — Asia Pacific University
Intake: **APUMF2512AI**

---

## Overview

An end-to-end classical computer vision pipeline that detects, delineates,
and counts individual tree crowns from satellite and aerial imagery — no
deep learning required. Inspired by Tong et al. (2021), adapted for
RGB + optional NIR input.

The system produces:

- Annotated image with tree crown outlines and centroids
- Per-crown feature table (area, circularity, NDVI, texture)
- Pixel-level metrics: Precision, Recall, F1, IoU
- Object-level metrics: Object Recognition Rate (ORR), merged/split counts

A **Streamlit dashboard** (`dashboard.py`) wraps the pipeline in an
interactive UI with light/dark themes, parameter sliders, and exportable
results.

---

## 🚀 Quick Start

Three commands to get the dashboard running locally:

```bash
# 1. Clone or download this repository, then cd into it
cd tree-crown-detection-ipcv

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Launch the interactive dashboard
streamlit run dashboard.py
```

Your browser opens at **http://localhost:8501**.

On the home page click **🚀 Start Analysis**, then in the sidebar:

1. Drag any image from `data/` (or your own satellite/aerial PNG/JPG) into
   the uploader
2. Click **🚀 Run Detection Pipeline**
3. Explore the 5 result tabs: Result · Pipeline Stages · Feature Analysis ·
   Evaluation · Export

To stop the server: `Ctrl+C` in the terminal.

> **Troubleshooting** — if the port is already used, pick another:
> `streamlit run dashboard.py --server.port 8502`

---

## Repository structure

```
.
├── dashboard.py                       ← interactive Streamlit dashboard
├── requirements.txt                   ← Python dependencies
├── .streamlit/
│   └── config.toml                    ← Streamlit server config
├── src/
│   ├── tree_detection_pipeline.py     ← main 5-stage pipeline
│   └── evaluate_metrics.py            ← pixel + object-level metrics
├── data/                              ← test satellite/aerial images
│   ├── image1.png                     ← dense forest scene
│   ├── image2.png                     ← scattered trees
│   └── image3.png                     ← mixed urban-vegetation
├── output/                            ← pre-generated results (21 PNGs)
│   ├── *_result.png                   ← final annotated detection
│   ├── *_annotated.png                ← overlay with crown numbers
│   ├── *_pipeline_overview.png        ← 9-panel stage visualisation
│   ├── *_tree_mask.png                ← binary tree mask
│   ├── *_ground_truth.png             ← auto-generated GT mask
│   ├── *_feature_analysis.png         ← feature distribution plots
│   └── *_evaluation.png               ← TP/FP/FN overlay
└── reference/
    └── Delineation_of_Individual_Tree_Crowns_Using_High_Spatial_
        Resolution_Multispectral_WorldView-3_Satellite_Imagery.pdf
```

---

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies:

- `numpy`, `scipy`, `opencv-python`
- `scikit-image` (GLCM, watershed, regionprops)
- `scikit-learn` (K-means)
- `matplotlib`, `pandas`
- `streamlit`, `plotly` (dashboard)

---

## Usage

### Option 1 — Interactive dashboard (recommended)

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501`. Upload an any tree image and click
**🚀 Run Detection Pipeline**. Features:

- Theme toggle (☀️ Light / 🌙 Dark)
- Parameter sliders (watershed distance, K-clusters, NDVI threshold, …)
- Auto-downscale for large images (preserves speed)
- 5 tabs: Result, Pipeline Stages, Feature Analysis, Evaluation, Export

### Option 2 — Batch processing via Python

```python
from src.tree_detection_pipeline import run_pipeline

result = run_pipeline(
    image_path="data/image1.png",
    output_dir="output",
    min_distance=15,
    n_clusters=4,
)
print(f"Trees detected: {result['stats']['total_trees']}")
```

### Option 3 — Evaluate metrics

```python
from src.evaluate_metrics import (
    create_ground_truth_from_green,
    pixel_level_metrics,
    object_level_metrics,
)
```

---

## Pipeline stages

| Stage | Operations |
|---|---|
| **1. Preprocessing**   | CLAHE enhancement · bilateral denoise · NDVI/ExG vegetation index · HSV colour space · GLCM texture map · exclusion mask (shadows/water/soil) |
| **2. Segmentation**    | Otsu thresholding on NDVI · K-means clustering in Lab space · marker-controlled watershed |
| **3. Feature Extraction** | Shape (area, circularity, eccentricity, solidity) · colour ratios · NDVI stats · Canny edge density · Haralick GLCM texture |
| **4. Classification**  | Rule-based weighted classifier with multi-criteria thresholds |
| **5. Post-processing** | Small-region cleanup · morphological opening · centroid labelling · annotation overlay · statistics |

---

## Results

Average across the three test scenes (Fredericton-style, Google Earth Pro):

| Metric        | Value  | Tong et al. (2021) |
|---------------|--------|--------------------|
| F1-score      | 0.888  | 0.82               |
| IoU           | 0.800  | —                  |
| Precision     | 0.928  | 0.85               |
| Recall        | 0.856  | 0.79               |
| ORR           | 0.432  | 0.74               |

---

## Author

**Andreas Jeno Figo Topuh - TP103728**
APUMF2512AI · MSc Artificial Intelligence
Asia Pacific University of Technology & Innovation


