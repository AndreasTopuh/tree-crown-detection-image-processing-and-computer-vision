"""
Tree Crown Detection Dashboard
==============================
Interactive Streamlit dashboard for the classical CV tree detection pipeline.

Run with:
    streamlit run dashboard.py
"""

import os
import sys
import io
import time
import tempfile
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tree_detection_pipeline import (
    load_image, preprocess, segment, extract_features,
    classify_trees, postprocess,
)
from evaluate_metrics import (
    create_ground_truth_from_green, pixel_level_metrics,
    object_level_metrics,
)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tree Crown Detection Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state (must be before CSS so theme can be read)
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "theme" not in st.session_state:
    st.session_state.theme = "light"

THEME = st.session_state.theme  # "light" or "dark"

# ---------- Theme colour palettes (WCAG-aware) ------------------------------
if THEME == "dark":
    C = {
        "bg":           "#0f1410",
        "surface":      "#1a211c",
        "surface_2":    "#232d26",
        "surface_3":    "#2a352d",
        "text":         "#f2f4f3",
        "text_sub":     "#b8c2bb",
        "text_muted":   "#8a948d",
        "border":       "#2a352d",
        "accent":       "#4caf50",
        "accent_dark":  "#2e7d32",
        "accent_light": "#81c784",
        "accent_soft":  "#1b2f1f",
        "on_accent":    "#ffffff",   # text on dark-green surface
        "shadow":       "0 2px 12px rgba(0,0,0,0.4)",
        "card_border":  "#2a352d",
        "hero_grad":    "linear-gradient(135deg, #1a211c 0%, #22332a 100%)",
    }
else:  # light
    C = {
        "bg":           "#f5f7f5",
        "surface":      "#ffffff",
        "surface_2":    "#f1f8e9",
        "surface_3":    "#e8f5e9",
        "text":         "#1b1b1b",
        "text_sub":     "#4a4a4a",
        "text_muted":   "#767676",
        "border":       "#e3e8e3",
        "accent":       "#2e7d32",
        "accent_dark":  "#1b5e20",
        "accent_light": "#4caf50",
        "accent_soft":  "#e8f5e9",
        "on_accent":    "#ffffff",   # text on dark-green surface
        "shadow":       "0 2px 12px rgba(0,0,0,0.05)",
        "card_border":  "#e3e8e3",
        "hero_grad":    "linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%)",
    }


# Custom CSS for professional look
_CSS_TEMPLATE = """
<style>
    /* ---------- Base: force theme regardless of browser preference --------- */
    .stApp, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: __BG__ !important;
        color: __TEXT__ !important;
    }
    .main {
        padding-top: 0.5rem;
        background-color: __BG__ !important;
    }
    /* Default text everywhere */
    .stApp p, .stApp li, .stApp span, .stApp div,
    .stApp label, .stMarkdown, .stMarkdown p {
        color: __TEXT__ !important;
    }
    h1 {
        color: __TEXT__ !important;
        font-family: 'Segoe UI', -apple-system, sans-serif;
        font-weight: 700 !important;
    }
    h2, h3, h4, h5, h6 {
        color: __TEXT__ !important;
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }
    .stMetric {
        background-color: __SURFACE__ !important;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid __BORDER__;
    }
    .stMetric label { color: __TEXT_SUB__ !important; font-weight: 600 !important; }
    /* Tabs */
    button[data-baseweb="tab"] {
        color: __TEXT_SUB__ !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: __ACCENT__ !important;
        border-bottom-color: __ACCENT__ !important;
    }
    [data-testid="stCaptionContainer"] { color: __TEXT_MUTED__ !important; }
    [data-testid="stFileUploader"] * { color: __TEXT__ !important; }
    [data-testid="stFileUploader"] section {
        background-color: __SURFACE__ !important;
        border: 2px dashed __BORDER__ !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"] {
        background-color: __SURFACE__ !important;
        border: 1px solid __BORDER__ !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"] * { color: __TEXT__ !important; }

    /* Sidebar — strong overrides */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        background: __SURFACE__ !important;
        border-right: 1px solid __BORDER__ !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown *,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] details summary,
    section[data-testid="stSidebar"] details summary *,
    div[data-testid="stSidebar"] * {
        color: __TEXT__ !important;
    }
    section[data-testid="stSidebar"] details,
    section[data-testid="stSidebar"] .stExpander {
        background-color: __SURFACE_2__ !important;
        border-radius: 10px !important;
        border: 1px solid __BORDER__ !important;
    }
    /* File uploader in sidebar */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: __SURFACE_2__ !important;
        border: 2px dashed __BORDER__ !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: __TEXT__ !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: __SURFACE__ !important;
        color: __TEXT__ !important;
        border: 1px solid __BORDER__ !important;
    }
    /* Slider value text */
    section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
        color: __TEXT_SUB__ !important;
    }
    /* Help icon tooltip */
    section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg {
        fill: __TEXT_SUB__ !important;
    }

    /* Sliders / inputs (main) */
    div[data-testid="stSlider"] label,
    div[data-baseweb="select"] * { color: __TEXT__ !important; }

    /* ---------- Primary metric card (Donezo-style highlighted) ------------ */
    .metric-card {
        background: linear-gradient(135deg, __ACCENT__ 0%, __ACCENT_DARK__ 100%);
        color: __ON_ACCENT__ !important;
        padding: 22px 20px;
        border-radius: 18px;
        text-align: left;
        box-shadow: __SHADOW__;
        position: relative;
        overflow: hidden;
        min-height: 140px;
    }
    .metric-card * { color: __ON_ACCENT__ !important; }
    .metric-card .metric-value {
        font-size: 2.4em;
        font-weight: 800;
        line-height: 1.1;
        margin: 8px 0 4px;
    }
    .metric-card .metric-label {
        font-size: 0.88em;
        opacity: 0.92;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Neutral metric card (white / dark surface depending on theme) */
    .metric-card-neutral {
        background: __SURFACE__;
        color: __TEXT__ !important;
        padding: 22px 20px;
        border-radius: 18px;
        text-align: left;
        border: 1px solid __BORDER__;
        box-shadow: __SHADOW__;
        min-height: 140px;
    }
    .metric-card-neutral .metric-value {
        font-size: 2.4em;
        font-weight: 800;
        color: __TEXT__ !important;
        line-height: 1.1;
        margin: 8px 0 4px;
    }
    .metric-card-neutral .metric-label {
        font-size: 0.88em;
        color: __TEXT_SUB__ !important;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .metric-card-neutral .metric-sub {
        font-size: 0.78em;
        color: __TEXT_MUTED__ !important;
        margin-top: 4px;
    }
    .metric-arrow {
        position: absolute; top: 18px; right: 18px;
        width: 32px; height: 32px; border-radius: 50%;
        background: rgba(255,255,255,0.18);
        display: flex; align-items: center; justify-content: center;
        font-size: 1em;
    }
    .metric-card-neutral .metric-arrow {
        background: __SURFACE_2__;
        color: __ACCENT__ !important;
    }

    /* ---------- Section / stage headers ------------------------------------ */
    .stage-header {
        background: __SURFACE_2__;
        padding: 12px 16px;
        border-radius: 10px;
        border-left: 4px solid __ACCENT__;
        margin: 14px 0;
        color: __TEXT__ !important;
        font-weight: 600;
    }

    /* ---------- Hero / home page ------------------------------------------- */
    .hero {
        background: __HERO_GRAD__;
        padding: 55px 40px;
        border-radius: 20px;
        border: 1px solid __BORDER__;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: __SHADOW__;
    }
    .hero-title {
        font-size: 3.2em;
        font-weight: 800;
        background: linear-gradient(90deg, __ACCENT_DARK__ 0%, __ACCENT_LIGHT__ 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.3em;
        color: __ACCENT__ !important;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .hero-tagline {
        font-size: 1em;
        color: __TEXT_SUB__ !important;
        max-width: 750px;
        margin: 0 auto 20px;
        line-height: 1.6;
    }

    /* Feature cards */
    .feature-card {
        background: __SURFACE__;
        border: 1px solid __BORDER__;
        border-top: 4px solid __ACCENT__;
        padding: 22px;
        border-radius: 14px;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: __SHADOW__;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.18);
    }
    .feature-icon { font-size: 2.2em; margin-bottom: 10px; }
    .feature-title {
        font-size: 1.1em;
        font-weight: 700;
        color: __TEXT__ !important;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.9em;
        color: __TEXT_SUB__ !important;
        line-height: 1.5;
    }

    /* Metric explanation blocks */
    .metric-explain {
        background: __SURFACE_2__;
        border-left: 4px solid __ACCENT__;
        padding: 16px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid __BORDER__;
    }
    .metric-explain-title {
        font-weight: 700;
        color: __TEXT__ !important;
        font-size: 1.05em;
        margin-bottom: 6px;
    }
    .metric-explain-body {
        font-size: 0.9em;
        color: __TEXT_SUB__ !important;
        line-height: 1.55;
    }

    .section-title {
        font-size: 1.8em;
        font-weight: 700;
        color: __TEXT__ !important;
        margin: 40px 0 10px;
        text-align: center;
    }
    .section-sub {
        text-align: center;
        color: __TEXT_MUTED__ !important;
        margin-bottom: 25px;
        font-size: 1em;
    }

    /* Primary action button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, __ACCENT__ 0%, __ACCENT_DARK__ 100%) !important;
        color: __ON_ACCENT__ !important;
        border: none !important;
        padding: 14px 40px;
        font-size: 1.05em;
        font-weight: 600;
        border-radius: 30px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(46, 125, 50, 0.4);
    }
    /* Secondary buttons */
    div[data-testid="stButton"] > button:not([kind="primary"]) {
        background: __SURFACE__ !important;
        color: __TEXT__ !important;
        border: 1px solid __BORDER__ !important;
        border-radius: 24px;
        font-weight: 500;
    }
    div[data-testid="stButton"] > button:not([kind="primary"]):hover {
        background: __SURFACE_2__ !important;
        border-color: __ACCENT__ !important;
    }

    /* Top bar (logo + theme toggle row) */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 4px 10px;
        border-bottom: 1px solid __BORDER__;
        margin-bottom: 18px;
    }
    .brand {
        display: flex; align-items: center; gap: 10px;
        font-weight: 700; font-size: 1.2em; color: __TEXT__ !important;
    }
    .brand-icon {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, __ACCENT__, __ACCENT_DARK__);
        color: __ON_ACCENT__ !important;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.2em;
    }

    /* Dataframe, code, etc. */
    .stDataFrame, .stTable { background-color: __SURFACE__ !important; }
    code { background-color: __SURFACE_2__ !important; color: __ACCENT_DARK__ !important; }

    /* Expander header */
    [data-testid="stExpander"] summary { color: __TEXT__ !important; }
</style>
"""

_repl = {
    "__BG__": C["bg"],
    "__SURFACE__": C["surface"],
    "__SURFACE_2__": C["surface_2"],
    "__SURFACE_3__": C["surface_3"],
    "__TEXT__": C["text"],
    "__TEXT_SUB__": C["text_sub"],
    "__TEXT_MUTED__": C["text_muted"],
    "__BORDER__": C["border"],
    "__ACCENT__": C["accent"],
    "__ACCENT_DARK__": C["accent_dark"],
    "__ACCENT_LIGHT__": C["accent_light"],
    "__ACCENT_SOFT__": C["accent_soft"],
    "__ON_ACCENT__": C["on_accent"],
    "__SHADOW__": C["shadow"],
    "__HERO_GRAD__": C["hero_grad"],
}
_css = _CSS_TEMPLATE
for k, v in _repl.items():
    _css = _css.replace(k, v)
st.markdown(_css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page Routing helpers
# ---------------------------------------------------------------------------
def go_dashboard():
    st.session_state.page = "dashboard"


def go_home():
    st.session_state.page = "home"


def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


def _theme_toggle_row(location="main"):
    """Render a compact top bar with brand + theme toggle."""
    is_dark = THEME == "dark"
    label = "🌙 Dark" if not is_dark else "☀️ Light"
    key = f"theme_toggle_{location}"
    colA, colB = st.columns([5, 1])
    with colA:
        st.markdown(f"""
        <div class='brand'>
            <span class='brand-icon'>🌳</span>
            <span>TreeVision</span>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.button(label, key=key, on_click=toggle_theme,
                  use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar - Input & Parameters (dashboard page only)
# ---------------------------------------------------------------------------
uploaded_file = None
run_button = False
min_distance = 15
n_clusters = 4
max_dim = 1500
min_ndvi = 0.12
min_circularity = 0.10
min_edge_density = 0.005
min_contrast = 0.5
enable_evaluation = True

if st.session_state.page == "dashboard":
    with st.sidebar:
        st.button("← Back to Home", on_click=go_home,
                  use_container_width=True)
        st.markdown("# 🌳 Controls")
        st.markdown("---")

        st.markdown("### 📁 Image Input")
        uploaded_file = st.file_uploader(
            "Upload satellite/aerial image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="Upload an RGB or RGB+NIR image of forest/vegetation area"
        )

        st.markdown("### ⚙️ Pipeline Parameters")

        with st.expander("🔧 Segmentation", expanded=True):
            min_distance = st.slider(
                "Watershed Min Distance", 5, 40, 15,
                help="Minimum distance between tree crown centres. "
                     "Lower = more trees detected but may over-segment."
            )
            n_clusters = st.slider(
                "K-means Clusters", 2, 8, 4,
                help="Number of colour clusters. "
                     "Higher = finer vegetation separation."
            )

        with st.expander("⚡ Performance", expanded=True):
            max_dim = st.select_slider(
                "Max image dimension (px)",
                options=[800, 1000, 1200, 1500, 2000, 2500, 3000, 4000],
                value=1500,
                help="Large images are auto-downscaled before processing. "
                     "Classical CV scales with pixel count — 1500 px is a "
                     "good balance of speed vs detail. 12 MP images take "
                     "30–90 s; 1500 px takes 3–8 s."
            )

        with st.expander("🧠 Classifier"):
            min_ndvi = st.slider("Min NDVI", 0.0, 0.5, 0.12, 0.01)
            min_circularity = st.slider("Min Circularity", 0.0, 0.5, 0.10,
                                         0.01)
            min_edge_density = st.slider(
                "Min Edge Density", 0.0, 0.05, 0.005, 0.001,
                format="%.3f"
            )
            min_contrast = st.slider("Min GLCM Contrast", 0.0, 5.0, 0.5, 0.1)

        with st.expander("📊 Evaluation"):
            enable_evaluation = st.checkbox(
                "Enable quantitative evaluation",
                value=True,
                help="Generate HSV-based ground truth and compute metrics"
            )

        st.markdown("---")
        run_button = st.button("🚀 Run Detection Pipeline",
                                type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
if st.session_state.page == "dashboard":
    _theme_toggle_row("dashboard")
    st.markdown("## 🌳 Tree Crown Detection Dashboard")
    st.markdown(
        f"<span style='color:{C['text_sub']}'>Classical Computer Vision "
        f"Pipeline &nbsp;·&nbsp; Environmental Monitoring &nbsp;·&nbsp; "
        f"Forest Area Segmentation</span>",
        unsafe_allow_html=True,
    )


# Helper: cv2 BGR -> RGB for Streamlit/Plotly
def to_rgb(img_bgr):
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# Pipeline runner with progress
@st.cache_data(show_spinner=False)
def run_pipeline_cached(image_bytes, min_distance, n_clusters, params_tuple,
                         max_dim=1500):
    """Cached pipeline execution — rerun only when inputs change."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = img.shape[:2]
    long_side = max(orig_h, orig_w)
    resized = False
    if long_side > max_dim:
        scale = max_dim / long_side
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized = True

    with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, img)

    try:
        rgb, nir = load_image(tmp_path)
        enhanced, ndvi, hsv, texture_map, exclusion_mask = preprocess(rgb, nir)
        veg_mask, km_labels, combined, ws_labels = segment(
            enhanced, hsv, ndvi, min_distance, n_clusters,
            texture_map, exclusion_mask)
        regions = extract_features(enhanced, hsv, ndvi, ws_labels, combined)

        params = dict(zip(
            ["min_ndvi", "min_circularity", "min_edge_density",
             "min_contrast"], params_tuple))
        full_params = {
            "min_area": 60, "max_area": 80000,
            "max_eccentricity": 0.97, "min_solidity": 0.25,
            "min_green_red_ratio": 0.7,
            "max_mean_brightness": 200,
            **params,
        }
        trees, non_trees, _ = classify_trees(regions, full_params)
        tree_mask, count, annotated, stats = postprocess(
            enhanced, ws_labels, trees, non_trees)

        result = {
            "rgb": rgb, "enhanced": enhanced, "ndvi": ndvi,
            "veg_mask": veg_mask, "km_labels": km_labels,
            "combined": combined, "ws_labels": ws_labels,
            "tree_mask": tree_mask, "annotated": annotated,
            "trees": trees, "non_trees": non_trees,
            "stats": stats, "tmp_path": tmp_path,
            "regions": regions,
            "resized": resized,
            "orig_size": (orig_w, orig_h),
            "proc_size": (img.shape[1], img.shape[0]),
        }
        return result
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e


# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    _theme_toggle_row("home")

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🌳 Tree Crown Detection System</div>
        <div class="hero-subtitle">Classical Computer Vision for Forest Monitoring</div>
        <div class="hero-tagline">
            An end-to-end pipeline that detects, delineates, and counts
            individual tree crowns from satellite and aerial imagery — no
            deep learning required. Built on NDVI, adaptive thresholding,
            K-means clustering, marker-controlled watershed, and GLCM texture
            features.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.button("🚀 Start Analysis", type="primary",
                  use_container_width=True, on_click=go_dashboard)

    # ---- Section: Pipeline Stages ----
    st.markdown('<div class="section-title">🔬 The 5-Stage Pipeline</div>',
                 unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Each image flows through these '
                 'stages from raw pixels to labelled tree crowns</div>',
                 unsafe_allow_html=True)

    stages = [
        ("🎨", "Preprocessing",
         "CLAHE contrast enhancement, bilateral denoise, NDVI/ExG vegetation "
         "index computation, HSV colour space, GLCM texture map, and "
         "exclusion masking for shadows, water, and bare soil."),
        ("🔍", "Segmentation",
         "Otsu auto-thresholding on NDVI, unsupervised K-means clustering "
         "in Lab colour space, and marker-controlled watershed to separate "
         "touching crowns into individual objects."),
        ("📐", "Feature Extraction",
         "For every candidate region: shape descriptors (area, circularity, "
         "eccentricity, solidity), colour ratios, NDVI statistics, Canny "
         "edge density, and Haralick GLCM texture (contrast, homogeneity)."),
        ("🧠", "Classification",
         "Rule-based weighted classifier scores each region against "
         "multi-criteria thresholds (NDVI, shape, texture, colour) and "
         "assigns a confidence score for tree vs non-tree."),
        ("📊", "Post-processing",
         "Small-region cleanup, morphological opening, crown centroid "
         "labelling, colour-coded annotation overlay, and per-image "
         "statistics (count, coverage, mean size)."),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, stages):
        col.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Section: Detection Metrics ----
    st.markdown('<div class="section-title">📊 Detection Metrics Explained</div>',
                 unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What every number on your results '
                 'page actually means</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    detection_metrics = [
        ("🌲 Trees Detected",
         "Total count of individual tree crowns the system identified. "
         "This is the primary output — useful for forest inventory, "
         "biomass estimation, and change detection over time."),
        ("🌍 Coverage (%)",
         "Proportion of the image area classified as tree canopy. "
         "Directly indicates forest density; values above 60% suggest a "
         "dense canopy, below 20% suggest scattered vegetation."),
        ("🌳 Mean Crown Size",
         "Average projected area of detected crowns in pixels (or m² when "
         "a ground-sampling distance is known). Larger mean sizes indicate "
         "mature trees; small sizes indicate saplings or shrubs."),
        ("⚡ Processing Time",
         "Seconds taken by the full pipeline. Classical CV is CPU-bound "
         "and scales with pixel count — a useful baseline for comparing "
         "against deep-learning alternatives."),
    ]
    for i, (title, body) in enumerate(detection_metrics):
        target = left if i % 2 == 0 else right
        target.markdown(f"""
        <div class="metric-explain">
            <div class="metric-explain-title">{title}</div>
            <div class="metric-explain-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Section: Evaluation Metrics ----
    st.markdown('<div class="section-title">🎯 Accuracy Metrics Explained</div>',
                 unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the system knows it is doing '
                 'a good job — benchmarked against auto-generated ground '
                 'truth</div>', unsafe_allow_html=True)

    left2, right2 = st.columns(2)
    accuracy_metrics = [
        ("🎯 Precision",
         "Of all pixels the system labelled as trees, what fraction really "
         "are trees? High precision means few false alarms — the system "
         "does not mistake grass, shadows, or rooftops for trees."),
        ("🔎 Recall",
         "Of all the real tree pixels, what fraction did the system find? "
         "High recall means the system does not miss trees — critical for "
         "complete forest inventories."),
        ("⚖️ F1-Score",
         "The harmonic mean of precision and recall — a single balanced "
         "score that rewards systems good at both finding trees and not "
         "over-reporting. Values above 0.85 are considered strong."),
        ("📐 IoU (Intersection over Union)",
         "Pixel-level overlap between predicted and true tree masks. "
         "Standard metric in segmentation. IoU ≥ 0.7 is excellent; "
         "our pipeline achieves ~0.80 on average."),
        ("✅ ORR (Object Recognition Rate)",
         "Fraction of ground-truth crowns correctly detected as distinct "
         "objects (not just pixels). Matches the paper's object-level "
         "protocol for fair comparison."),
        ("🔀 Merged / Split",
         "Diagnostic counts: merged = two crowns fused into one; split = "
         "one crown broken into pieces. Lower is better — tells you "
         "whether the watershed separation is too aggressive or too loose."),
    ]
    for i, (title, body) in enumerate(accuracy_metrics):
        target = left2 if i % 2 == 0 else right2
        target.markdown(f"""
        <div class="metric-explain">
            <div class="metric-explain-title">{title}</div>
            <div class="metric-explain-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Section: Technologies ----
    st.markdown('<div class="section-title">🛠️ Technology Stack</div>',
                 unsafe_allow_html=True)
    techs = [
        ("OpenCV", "CLAHE, watershed, morphology"),
        ("scikit-image", "GLCM, region properties"),
        ("scikit-learn", "K-means clustering"),
        ("NumPy / SciPy", "Array ops & distance transforms"),
        ("Plotly", "Interactive charts"),
        ("Streamlit", "This dashboard"),
    ]
    tcols = st.columns(6)
    for col, (name, desc) in zip(tcols, techs):
        col.markdown(f"""
        <div class="feature-card" style="text-align:center">
            <div class="feature-title">{name}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Sample Results ----
    st.markdown('<div class="section-title">📸 Sample Results</div>',
                 unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real outputs from three '
                 'Fredericton, Canada test scenes</div>',
                 unsafe_allow_html=True)

    sample_cols = st.columns(3)
    samples = [
        ("image1", "Dense Forest",
         "Closed canopy — pipeline handles heavy occlusion"),
        ("image2", "Scattered Trees",
         "Isolated crowns — clean individual separation"),
        ("image3", "Mixed Urban",
         "Urban-vegetation mix — rejects buildings & roads"),
    ]
    for col, (fname, title, desc) in zip(sample_cols, samples):
        sample_path = os.path.join(os.path.dirname(__file__), "output",
                                    f"{fname}_result.png")
        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(sample_path):
                st.image(sample_path, use_container_width=True)
            st.caption(desc)

    # ---- Final CTA ----
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.button("🚀 Start Analysis Now", type="primary",
                  use_container_width=True, on_click=go_dashboard,
                  key="cta_bottom")

    st.stop()


# ---------------------------------------------------------------------------
# Process Uploaded Image
# ---------------------------------------------------------------------------
if uploaded_file is not None and run_button:
    image_bytes = uploaded_file.read()

    # Show progress
    progress = st.progress(0, text="Initialising pipeline...")
    status = st.empty()

    try:
        status.info("🎨 Running preprocessing, segmentation, and classification...")
        progress.progress(20, text="Preprocessing...")

        start_time = time.time()
        result = run_pipeline_cached(
            image_bytes, min_distance, n_clusters,
            (min_ndvi, min_circularity, min_edge_density, min_contrast),
            max_dim=max_dim,
        )
        elapsed = time.time() - start_time

        if result.get("resized"):
            ow, oh = result["orig_size"]
            pw, ph = result["proc_size"]
            st.info(
                f"ℹ️ Image auto-downscaled from **{ow}×{oh}** "
                f"({ow * oh / 1e6:.1f} MP) to **{pw}×{ph}** "
                f"({pw * ph / 1e6:.1f} MP) for speed. "
                f"Adjust *Max image dimension* in the sidebar if you need "
                f"higher detail."
            )

        progress.progress(80, text="Generating visualisations...")

        # Optional: evaluation
        eval_metrics = None
        if enable_evaluation:
            gt_mask = create_ground_truth_from_green(result["tmp_path"])
            if gt_mask.shape != result["tree_mask"].shape:
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8),
                    (result["tree_mask"].shape[1], result["tree_mask"].shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            pixel_m = pixel_level_metrics(result["tree_mask"], gt_mask)
            object_m = object_level_metrics(result["tree_mask"], gt_mask)
            eval_metrics = {
                "pixel": pixel_m, "object": object_m, "gt_mask": gt_mask,
            }

        progress.progress(100, text="Complete!")
        time.sleep(0.3)
        progress.empty()
        status.success(f"✅ Pipeline completed in **{elapsed:.2f}s**")

    except Exception as e:
        progress.empty()
        status.error(f"❌ Pipeline error: {e}")
        st.stop()

    stats = result["stats"]

    # -------------------------------------------------------------------
    # KEY METRICS CARDS
    # -------------------------------------------------------------------
    st.markdown("### 📊 Detection Summary")
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(col, label, value, subtitle="", highlight=False):
        cls = "metric-card" if highlight else "metric-card-neutral"
        arrow = "↗" if highlight else "→"
        col.markdown(f"""
        <div class='{cls}'>
            <div class='metric-arrow'>{arrow}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub' style='font-size:0.78em;
                 opacity:0.85; margin-top:2px'>{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    metric_card(m1, "Trees Detected",
                 stats["total_trees"],
                 f"Confidence: {stats['mean_conf']:.1%}",
                 highlight=True)
    metric_card(m2, "Coverage",
                 f"{stats['coverage_pct']:.1f}%",
                 "of total area")
    metric_card(m3, "Mean Crown",
                 f"{stats['mean_area']:.0f} px",
                 f"±{stats['std_area']:.0f}")
    metric_card(m4, "Processing Time",
                 f"{elapsed:.2f}s",
                 f"{result['rgb'].shape[1]}×{result['rgb'].shape[0]}")

    st.markdown("---")

    # -------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Result",
        "🔄 Pipeline Stages",
        "📐 Feature Analysis",
        "📊 Evaluation",
        "📥 Export",
    ])

    # --- TAB 1: Main Result ----------------------------------------------
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Original Image")
            st.image(to_rgb(result["rgb"]), use_container_width=True)
        with c2:
            st.markdown("### Detected Tree Crowns")
            st.image(to_rgb(result["annotated"]), use_container_width=True)
            st.caption(
                f"🟢 Green contours: tree crowns | "
                f"🔴 Red dots: crown centroids | "
                f"🟡 Yellow: non-tree regions"
            )

        # Area histogram
        areas = [r["area"] for r in result["trees"]]
        if areas:
            st.markdown("### 📈 Crown Size Distribution")
            fig_hist = px.histogram(
                x=areas, nbins=30,
                labels={"x": "Crown Area (pixels)", "y": "Count"},
                color_discrete_sequence=["#2e7d32"],
            )
            fig_hist.update_layout(
                showlegend=False, height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- TAB 2: Pipeline Stages -----------------------------------------
    with tab2:
        st.markdown("#### Stage 1: Preprocessing")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Original RGB**")
            st.image(to_rgb(result["rgb"]), use_container_width=True)
        with s2:
            st.markdown("**CLAHE Enhanced**")
            st.image(to_rgb(result["enhanced"]), use_container_width=True)
        with s3:
            st.markdown("**NDVI / ExG Map**")
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(result["ndvi"], cmap="RdYlGn", vmin=-0.1, vmax=0.8)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown('<div class="stage-header"><b>Stage 2: Segmentation</b></div>',
                     unsafe_allow_html=True)
        s4, s5, s6 = st.columns(3)
        with s4:
            st.markdown("**Otsu NDVI Threshold**")
            st.image(result["veg_mask"].astype(np.uint8) * 255,
                     use_container_width=True)
        with s5:
            st.markdown("**K-means Clustering**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(result["km_labels"], cmap="tab10")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with s6:
            st.markdown("**Combined Vegetation Mask**")
            st.image(result["combined"].astype(np.uint8) * 255,
                     use_container_width=True)

        st.markdown('<div class="stage-header"><b>Stage 3-5: Watershed & Classification</b></div>',
                     unsafe_allow_html=True)
        s7, s8, s9 = st.columns(3)
        with s7:
            st.markdown("**Watershed Segments**")
            ws_rgb = np.zeros((*result["ws_labels"].shape, 3), dtype=np.uint8)
            np.random.seed(42)
            for lbl in np.unique(result["ws_labels"]):
                if lbl == 0:
                    continue
                ws_rgb[result["ws_labels"] == lbl] = np.random.randint(50, 255, 3)
            st.image(ws_rgb, use_container_width=True)
        with s8:
            st.markdown("**Tree Mask (Classified)**")
            st.image(result["tree_mask"].astype(np.uint8) * 255,
                     use_container_width=True)
        with s9:
            st.markdown("**Final Annotated**")
            st.image(to_rgb(result["annotated"]), use_container_width=True)

    # --- TAB 3: Feature Analysis ----------------------------------------
    with tab3:
        st.markdown("### 🔬 Feature Distributions: Tree vs Non-Tree")

        trees = result["trees"]
        non_trees = result["non_trees"]

        if not trees:
            st.warning("No trees were classified. Try adjusting parameters.")
        else:
            # Build DataFrame
            all_regions = []
            for r in trees:
                all_regions.append({**r, "class": "Tree"})
            for r in non_trees:
                all_regions.append({**r, "class": "Non-Tree"})

            df = pd.DataFrame(all_regions)

            # Feature selection
            features = ["area", "circularity", "mean_ndvi", "eccentricity",
                        "solidity", "edge_density", "contrast",
                        "green_red_ratio"]
            available = [f for f in features if f in df.columns]

            selected_feature = st.selectbox(
                "Select feature to visualise:", available, index=2)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    df, x=selected_feature, color="class",
                    barmode="overlay", nbins=30,
                    color_discrete_map={"Tree": "#2e7d32", "Non-Tree": "#d32f2f"},
                    opacity=0.65,
                )
                fig.update_layout(
                    height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig_box = px.box(
                    df, y=selected_feature, color="class",
                    color_discrete_map={"Tree": "#2e7d32", "Non-Tree": "#d32f2f"},
                    points="all",
                )
                fig_box.update_layout(
                    height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            # Scatter matrix preview
            st.markdown("#### 📊 Multi-Feature Comparison")
            scatter_x = st.selectbox("X axis:", available, index=0, key="sx")
            scatter_y = st.selectbox("Y axis:", available, index=2, key="sy")
            fig_sc = px.scatter(
                df, x=scatter_x, y=scatter_y, color="class",
                color_discrete_map={"Tree": "#2e7d32", "Non-Tree": "#d32f2f"},
                hover_data=["area", "confidence"]
                if "confidence" in df.columns else None,
                opacity=0.7,
            )
            fig_sc.update_layout(
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # Summary statistics table
            st.markdown("#### 📋 Feature Summary Statistics")
            stat_cols = ["area", "circularity", "mean_ndvi",
                         "eccentricity", "edge_density", "contrast"]
            stat_cols = [c for c in stat_cols if c in df.columns]
            summary = df.groupby("class")[stat_cols].agg(["mean", "std"]).round(3)
            st.dataframe(summary, use_container_width=True)

    # --- TAB 4: Evaluation ----------------------------------------------
    with tab4:
        if not enable_evaluation or eval_metrics is None:
            st.info(
                "📊 Enable quantitative evaluation in the sidebar to compute "
                "pixel-level and object-level metrics against HSV-based "
                "approximate ground truth."
            )
        else:
            pm = eval_metrics["pixel"]
            om = eval_metrics["object"]
            gt = eval_metrics["gt_mask"]

            st.markdown("### 📏 Pixel-Level Metrics")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Precision", f"{pm['precision']:.4f}")
            pc2.metric("Recall", f"{pm['recall']:.4f}")
            pc3.metric("F1 Score", f"{pm['f1_score']:.4f}")
            pc4.metric("IoU", f"{pm['iou']:.4f}")

            st.markdown("### 🎯 Object-Level Metrics (ORR-style)")
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("GT Objects", om["gt_objects"])
            oc2.metric("Predicted", om["pred_objects"])
            oc3.metric("ORR", f"{om['ORR']:.4f}")
            oc4.metric("Merged / Split", f"{om['merged']} / {om['split']}")

            st.markdown("### 🖼️ Visual Comparison")
            v1, v2, v3 = st.columns(3)
            with v1:
                st.markdown("**Ground Truth (HSV-based)**")
                st.image(gt.astype(np.uint8) * 255, use_container_width=True)
            with v2:
                st.markdown("**Prediction**")
                st.image(result["tree_mask"].astype(np.uint8) * 255,
                         use_container_width=True)
            with v3:
                st.markdown("**TP(green) / FP(red) / FN(blue)**")
                p = result["tree_mask"].astype(bool)
                overlay = np.zeros((*p.shape, 3), dtype=np.uint8)
                overlay[p & gt] = [0, 255, 0]
                overlay[p & ~gt] = [255, 0, 0]
                overlay[~p & gt] = [0, 0, 255]
                st.image(overlay, use_container_width=True)

            # Confusion breakdown bar chart
            st.markdown("### 📊 Pixel Classification Breakdown")
            conf_df = pd.DataFrame({
                "Class": ["True Positive", "False Positive",
                          "False Negative", "True Negative"],
                "Pixels": [pm["TP"], pm["FP"], pm["FN"], pm["TN"]],
            })
            fig_conf = px.bar(
                conf_df, x="Class", y="Pixels",
                color="Class",
                color_discrete_sequence=["#2e7d32", "#f57c00", "#1976d2", "#9e9e9e"],
            )
            fig_conf.update_layout(
                showlegend=False, height=350,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_conf, use_container_width=True)

    # --- TAB 5: Export --------------------------------------------------
    with tab5:
        st.markdown("### 📥 Export Results")

        # Annotated image
        is_ok, buf = cv2.imencode(".png", result["annotated"])
        st.download_button(
            "📥 Download Annotated Image (PNG)",
            data=buf.tobytes(),
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

        # Tree mask
        is_ok, buf2 = cv2.imencode(
            ".png", result["tree_mask"].astype(np.uint8) * 255)
        st.download_button(
            "📥 Download Tree Mask (PNG)",
            data=buf2.tobytes(),
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_mask.png",
            mime="image/png",
            use_container_width=True,
        )

        # CSV of regions
        if result["trees"]:
            df_export = pd.DataFrame([
                {k: v for k, v in r.items()
                 if not isinstance(v, (tuple, list, np.ndarray))}
                for r in result["trees"]
            ])
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Tree Features (CSV)",
                data=csv_bytes,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Stats report (text)
        report_text = f"""Tree Crown Detection Report
========================================
Image: {uploaded_file.name}
Size: {result['rgb'].shape[1]} x {result['rgb'].shape[0]} px

DETECTION SUMMARY
  Trees detected:     {stats['total_trees']}
  Coverage:           {stats['coverage_pct']:.2f}%
  Mean crown area:    {stats['mean_area']:.1f} px
  Std crown area:     {stats['std_area']:.1f} px
  Mean confidence:    {stats['mean_conf']:.3f}

PARAMETERS
  Min distance:       {min_distance}
  K clusters:         {n_clusters}
  Min NDVI:           {min_ndvi}
  Min circularity:    {min_circularity}
  Min edge density:   {min_edge_density}
  Min GLCM contrast:  {min_contrast}
"""
        if eval_metrics:
            report_text += f"""
EVALUATION METRICS
  Precision:          {eval_metrics['pixel']['precision']:.4f}
  Recall:             {eval_metrics['pixel']['recall']:.4f}
  F1 Score:           {eval_metrics['pixel']['f1_score']:.4f}
  IoU:                {eval_metrics['pixel']['iou']:.4f}
  ORR (object-level): {eval_metrics['object']['ORR']:.4f}
"""

        st.download_button(
            "📥 Download Text Report",
            data=report_text,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 📄 Report Preview")
        st.code(report_text, language="text")

    # Cleanup temp file
    if os.path.exists(result["tmp_path"]):
        try:
            os.unlink(result["tmp_path"])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85em'>"
    "🌳 Tree Crown Detection Pipeline | Classical Computer Vision | "
    "CT103-3-M-IPCV · Asia Pacific University</div>",
    unsafe_allow_html=True,
)
