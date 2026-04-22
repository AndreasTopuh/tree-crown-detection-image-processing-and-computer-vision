"""
Tree Crown Detection and Mapping from Satellite/Aerial Imagery
Using Classical Computer Vision Techniques

Domain: Environmental Monitoring - Forest Area Segmentation and Tree Detection
Course: CT103-3-M-IPCV - Image Processing & Computer Vision
University: Asia Pacific University of Technology & Innovation

This pipeline implements a complete classical CV system for detecting,
segmenting, and recognising individual tree crowns from multispectral
satellite imagery (e.g., Sentinel-2, WorldView-3, or aerial RGB+NIR).

Pipeline Stages:
    1. Preprocessing  - Band selection, NDVI computation, enhancement, denoising
    2. Segmentation   - NDVI thresholding, K-means clustering, Watershed
    3. Feature Extraction - Shape descriptors, GLCM texture, edge features
    4. Recognition    - Rule-based classifier combining colour, texture, shape
    5. Post-processing - Morphological refinement, connected components, counting
"""

import os
import sys
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import morphology, measure, filters, segmentation
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from sklearn.cluster import KMeans
from scipy import ndimage

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# STAGE 1: PREPROCESSING
# ===================================================================

def load_image(path):
    """
    Load a satellite/aerial image.
    Supports standard RGB (.jpg, .png, .tif) and 4-band images
    where the 4th channel is NIR.

    Returns
    -------
    rgb : ndarray (H, W, 3) uint8 BGR
    nir : ndarray (H, W) float32 or None
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        nir = None
    elif img.shape[2] == 4:
        # Check if 4th channel is alpha (all 0 or 255) or real NIR data
        ch4 = img[:, :, 3]
        unique_vals = len(np.unique(ch4))
        if unique_vals <= 3:
            # Likely alpha channel (PNG transparency) - discard it
            rgb = img[:, :, :3].copy()
            nir = None
            print("[Load] 4th channel detected as alpha (transparency) - ignored.")
        else:
            # Real NIR band with varied values
            rgb = img[:, :, :3].copy()
            nir = ch4.astype(np.float32)
    else:
        rgb = img[:, :, :3].copy()
        nir = None

    print(f"[Load] Image shape: {rgb.shape}, NIR available: {nir is not None}")
    return rgb, nir


def compute_ndvi(rgb, nir):
    """
    Compute Normalised Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    When NIR is unavailable a proxy Excess Green Index (ExG) is used:
        ExG = 2*g_norm - r_norm - b_norm
    where g_norm = G / (R+G+B) (Tucker, 1979).
    """
    if nir is not None:
        red = rgb[:, :, 2].astype(np.float32)
        nir_f = nir.astype(np.float32)
        denom = nir_f + red + 1e-10
        ndvi = (nir_f - red) / denom
        print("[NDVI] Computed true NDVI from NIR and Red bands.")
    else:
        b, g, r = (rgb[:, :, i].astype(np.float32) for i in range(3))
        total = b + g + r + 1e-10
        ndvi = 2.0 * (g / total) - (r / total) - (b / total)
        ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-10)
        print("[NDVI] No NIR band - computed Excess Green Index as proxy.")

    return ndvi.astype(np.float32)


def enhance_contrast(image, clip_limit=3.0, tile_size=8):
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Operates on tiles to improve local contrast without
    over-amplifying noise (Zuiderveld, 1994).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced = clahe.apply(image)
    print(f"[Enhance] CLAHE applied (clip={clip_limit}, tile={tile_size}).")
    return enhanced


def denoise(image, ksize=5, sigma=1.0):
    """
    Gaussian blur for speckle / sensor noise suppression
    (Gonzalez & Woods, 2018).
    """
    smoothed = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    print(f"[Denoise] Gaussian blur (kernel={ksize}, sigma={sigma}).")
    return smoothed


def compute_texture_map(gray, ksize=15):
    """
    Compute local texture roughness using standard deviation filter.

    Trees have rough canopy texture (high local std), while grass
    and smooth surfaces have low local std (Haralick et al., 1973).
    """
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sqmean = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = np.maximum(sqmean - mean ** 2, 0)
    std_map = np.sqrt(variance).astype(np.float32)
    print(f"[Texture] Local std map computed (kernel={ksize}), "
          f"range=[{std_map.min():.1f}, {std_map.max():.1f}]")
    return std_map


def create_exclusion_mask(rgb_original):
    """
    Create a combined exclusion mask from the ORIGINAL image
    (before CLAHE) to detect shadows, water, and soil.

    IMPORTANT: This must use the original image, NOT the CLAHE-enhanced
    image, because CLAHE brightens dark pixels (shadows become lighter
    and may pass green filters incorrectly).

    Exclusion categories:
        1. Shadows: very dark pixels (V < 50 in HSV)
        2. Water/lakes: dark blue-black pixels (low V, low saturation
           or blue-shifted hue)
        3. Soil/bare ground: brown/tan pixels (hue 5-25, moderate sat)
    """
    hsv_orig = cv2.cvtColor(rgb_original, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_orig[:, :, 0], hsv_orig[:, :, 1], hsv_orig[:, :, 2]
    b_ch, g_ch, r_ch = (rgb_original[:, :, i].astype(np.float32)
                         for i in range(3))

    # 1. Shadow mask: ONLY very dark pixels (V < 30)
    #    Dark green trees can have V=30-70, so threshold must be low
    shadow = v < 30
    print(f"[Exclude] Shadow (V<30): {np.sum(shadow)/shadow.size*100:.1f}%")

    # 2. Water mask: ONLY clearly blue/black water bodies
    #    Lakes = blue channel significantly dominates green+red, very dark
    #    Must be very strict: dark green trees can have blue close to green
    blue_ratio = b_ch / (g_ch + 1)
    water = ((blue_ratio > 1.4) & (v < 60) & (b_ch > r_ch * 1.2))
    # Extremely dark near-black pixels (true voids, not dark trees)
    void = (v < 15) & (s < 30)
    water = water | void
    print(f"[Exclude] Water/void: {np.sum(water)/water.size*100:.1f}%")

    # 3. Soil/brown mask: hue in brown range AND clearly not green
    #    Brown in HSV: hue ~5-22 (orange-brown), must have saturation
    brown = ((h >= 5) & (h <= 22) & (s > 40) & (s < 200) &
             (v > 50) & (v < 200))
    # Lighter tan/beige bare ground
    tan = ((h >= 15) & (h <= 28) & (s > 20) & (s < 80) &
           (v > 140))
    soil = brown | tan
    print(f"[Exclude] Soil/brown: {np.sum(soil)/soil.size*100:.1f}%")

    exclusion = shadow | water | soil
    total_pct = np.sum(exclusion) / exclusion.size * 100
    print(f"[Exclude] Total excluded: {total_pct:.1f}%")

    return exclusion


def preprocess(rgb, nir):
    """
    Full preprocessing: exclusion mask (from original), CLAHE enhancement,
    Gaussian denoising, NDVI computation, HSV conversion, texture map.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: PREPROCESSING")
    print("=" * 60)

    # Create exclusion mask from ORIGINAL image (before CLAHE)
    exclusion_mask = create_exclusion_mask(rgb)

    # Then apply contrast enhancement
    enhanced = enhance_contrast(rgb, clip_limit=2.0)  # reduced from 3.0
    enhanced = denoise(enhanced)
    ndvi = compute_ndvi(enhanced, nir)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    texture_map = compute_texture_map(gray)
    return enhanced, ndvi, hsv, texture_map, exclusion_mask


# ===================================================================
# STAGE 2: SEGMENTATION
# ===================================================================

def threshold_ndvi(ndvi, method="otsu"):
    """
    Segment vegetation via NDVI thresholding.

    Otsu's method (Otsu, 1979) automatically finds the optimal
    threshold by minimising intra-class variance.
    """
    ndvi_u8 = ((ndvi - ndvi.min()) /
               (ndvi.max() - ndvi.min() + 1e-10) * 255).astype(np.uint8)

    if method == "otsu":
        tval, binary = cv2.threshold(
            ndvi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tval /= 255.0
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            ndvi_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -5)
        tval = "adaptive"
    else:
        tval = float(method)
        binary = (ndvi > tval).astype(np.uint8) * 255

    mask = binary > 0
    pct = np.sum(mask) / mask.size * 100
    print(f"[Threshold] NDVI thresh={tval}, vegetation={pct:.1f}%")
    return mask, tval


def kmeans_clustering(rgb, hsv, ndvi, n_clusters=4):
    """
    K-means clustering on pixel feature vectors [H, S, V, NDVI, Green].

    The cluster with the highest mean NDVI is selected as vegetation
    (Hartigan & Wong, 1979).
    """
    h, w = ndvi.shape
    feat = np.zeros((h * w, 5), dtype=np.float32)
    feat[:, 0] = hsv[:, :, 0].ravel() / 180.0
    feat[:, 1] = hsv[:, :, 1].ravel() / 255.0
    feat[:, 2] = hsv[:, :, 2].ravel() / 255.0
    feat[:, 3] = ((ndvi.ravel() - ndvi.min()) /
                   (ndvi.max() - ndvi.min() + 1e-10))
    feat[:, 4] = rgb[:, :, 1].ravel() / 255.0

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(feat).reshape(h, w)

    means = [ndvi[labels == k].mean() for k in range(n_clusters)]
    sorted_idx = np.argsort(means)[::-1]  # highest NDVI first

    # Select top clusters whose mean NDVI > overall median
    median_ndvi = np.median(ndvi)
    veg_mask = np.zeros((h, w), dtype=bool)
    selected = []
    for k in sorted_idx:
        if means[k] > median_ndvi:
            veg_mask |= (labels == k)
            selected.append(k)
    # Always include at least the top cluster
    if not selected:
        veg_mask = labels == sorted_idx[0]
        selected = [sorted_idx[0]]

    print(f"[K-means] {n_clusters} clusters, veg clusters={selected} "
          f"(NDVI={[round(means[k],3) for k in selected]})")
    return labels, veg_mask


def watershed_segmentation(rgb, veg_mask, min_distance=15):
    """
    Marker-controlled watershed to separate touching tree crowns.

    Steps:
        1. Euclidean distance transform of the vegetation mask
        2. Local maxima detection as crown-centre markers
        3. Watershed flooding to delineate individual crowns
    (Beucher & Meyer, 1993)
    """
    dist = ndimage.distance_transform_edt(veg_mask.astype(np.uint8))
    dist_smooth = cv2.GaussianBlur(dist.astype(np.float32), (7, 7), 2)

    coords = peak_local_max(dist_smooth, min_distance=min_distance,
                             labels=veg_mask.astype(int),
                             exclude_border=False)

    markers = np.zeros_like(veg_mask, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    markers = morphology.dilation(markers, morphology.disk(2))

    ws_labels = segmentation.watershed(
        -dist_smooth, markers=markers, mask=veg_mask.astype(bool))

    n_trees = len(np.unique(ws_labels)) - 1
    print(f"[Watershed] {n_trees} candidate crowns (min_dist={min_distance}).")
    return ws_labels, markers


def segment(rgb, hsv, ndvi, min_distance=15, n_clusters=4,
            texture_map=None, exclusion_mask=None):
    """
    Complete segmentation: NDVI threshold + K-means + exclusion mask
    + texture filtering + morphological cleaning + Watershed.

    Tree detection logic:
        - Green pixels (HSV hue 20-100) are candidate vegetation
        - Excluded pixels (shadow/water/soil from ORIGINAL image) removed
        - High-texture green pixels are trees (rough canopy)
        - Low-texture green pixels are grass/smooth ground (excluded)
    """
    print("\n" + "=" * 60)
    print("STAGE 2: SEGMENTATION")
    print("=" * 60)

    veg_mask, _ = threshold_ndvi(ndvi, method="otsu")
    km_labels, km_veg = kmeans_clustering(rgb, hsv, ndvi, n_clusters)

    # Union of NDVI and K-means for broad vegetation capture
    combined = np.logical_or(veg_mask, km_veg)

    # --- HSV-based green filter (on enhanced image) ---
    hue = hsv[:, :, 0]  # 0-180 in OpenCV
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green_hue = (hue >= 20) & (hue <= 100) & (sat > 12)
    combined = combined & green_hue

    # --- Exclusion mask: shadow/water/soil from ORIGINAL image ---
    # This is critical: we detect shadows from the original image
    # BEFORE CLAHE enhancement, because CLAHE brightens dark pixels
    # and would cause shadows to incorrectly pass green filters.
    if exclusion_mask is not None:
        excluded_count = np.sum(combined & exclusion_mask)
        combined = combined & (~exclusion_mask)
        print(f"[Exclude] Removed {excluded_count} pixels "
              f"(shadow/water/soil from original image).")

    # --- Texture-based tree vs grass discrimination ---
    if texture_map is not None:
        tex_green = texture_map[combined]
        if tex_green.size > 0:
            tex_u8 = ((texture_map - texture_map.min()) /
                       (texture_map.max() - texture_map.min() + 1e-10)
                       * 255).astype(np.uint8)
            green_tex_u8 = tex_u8[combined]
            tex_thresh, _ = cv2.threshold(
                green_tex_u8, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tex_threshold = (tex_thresh / 255.0 *
                              (texture_map.max() - texture_map.min())
                              + texture_map.min())
            # Use 60% of Otsu to keep more tree edge pixels
            tex_threshold *= 0.6

            tree_texture = texture_map >= tex_threshold
            grass_pixels = np.sum(combined & ~tree_texture)
            combined = combined & tree_texture
            print(f"[Texture] Threshold={tex_threshold:.1f}, "
                  f"removed {grass_pixels} smooth/grass pixels.")

    # --- Morphological cleanup ---
    combined = morphology.remove_small_objects(combined, min_size=80)
    combined = morphology.remove_small_holes(combined, area_threshold=300)
    combined = ndimage.binary_fill_holes(combined)

    disk2 = morphology.disk(2)
    combined = morphology.binary_closing(combined, disk2)
    combined = morphology.binary_opening(combined, disk2)

    veg_pct = np.sum(combined) / combined.size * 100
    print(f"[Combined] {np.sum(combined)} tree pixels ({veg_pct:.1f}%).")

    # Adaptive min_distance: smaller for dense vegetation
    veg_ratio = np.sum(combined) / combined.size
    adaptive_dist = max(8, int(min_distance * (1 - veg_ratio * 0.5)))
    print(f"[Adaptive] Vegetation ratio={veg_ratio:.2f}, "
          f"watershed min_dist={adaptive_dist}")

    ws_labels, _ = watershed_segmentation(rgb, combined, adaptive_dist)
    return veg_mask, km_labels, combined, ws_labels


# ===================================================================
# STAGE 3: FEATURE EXTRACTION
# ===================================================================

def extract_shape_features(region):
    """
    Shape descriptors via regionprops: area, perimeter, eccentricity,
    circularity, solidity, extent (Blaschke, 2010).
    """
    area = region.area
    perim = max(region.perimeter, 1e-10)
    circ = min((4.0 * np.pi * area) / (perim ** 2), 1.0)
    return {
        "area": area, "perimeter": perim,
        "eccentricity": region.eccentricity,
        "circularity": circ,
        "solidity": region.solidity,
        "extent": region.extent,
        "major_axis": region.major_axis_length,
        "minor_axis": region.minor_axis_length,
    }


def extract_glcm_texture(gray_patch):
    """
    GLCM texture features: contrast, dissimilarity, homogeneity,
    energy, correlation, ASM (Haralick et al., 1973).
    """
    if gray_patch.size < 16 or min(gray_patch.shape) < 4:
        return {"contrast": 0, "dissimilarity": 0, "homogeneity": 1,
                "energy": 1, "correlation": 0, "ASM": 1}

    levels = 32
    pq = np.clip((gray_patch / 256.0 * levels).astype(np.uint8), 0, levels - 1)
    glcm = graycomatrix(pq, [1], [0], levels=levels,
                         symmetric=True, normed=True)
    return {
        "contrast":      graycoprops(glcm, "contrast")[0, 0],
        "dissimilarity": graycoprops(glcm, "dissimilarity")[0, 0],
        "homogeneity":   graycoprops(glcm, "homogeneity")[0, 0],
        "energy":        graycoprops(glcm, "energy")[0, 0],
        "correlation":   graycoprops(glcm, "correlation")[0, 0],
        "ASM":           graycoprops(glcm, "ASM")[0, 0],
    }


def extract_edge_features(gray_patch):
    """
    Edge density (Canny) and mean Sobel gradient magnitude (Canny, 1986).
    """
    if gray_patch.size < 9:
        return {"edge_density": 0.0, "mean_gradient": 0.0}

    edges = cv2.Canny(gray_patch, 50, 150)
    sx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "edge_density": np.sum(edges > 0) / edges.size,
        "mean_gradient": np.mean(np.sqrt(sx ** 2 + sy ** 2)),
    }


def extract_colour_features(rgb_px, hsv_px, ndvi_px):
    """
    Spectral features: green/red ratio, mean hue/saturation, NDVI
    statistics (Tucker, 1979).
    """
    return {
        "mean_green": np.mean(rgb_px[:, 1]),
        "mean_red": np.mean(rgb_px[:, 2]),
        "green_red_ratio": np.mean(rgb_px[:, 1]) / (np.mean(rgb_px[:, 2]) + 1e-10),
        "mean_hue": np.mean(hsv_px[:, 0]),
        "mean_saturation": np.mean(hsv_px[:, 1]),
        "mean_ndvi": np.mean(ndvi_px),
        "std_ndvi": np.std(ndvi_px),
    }


def extract_features(rgb, hsv, ndvi, ws_labels, combined_mask):
    """
    Extract shape + texture + edge + colour features for every
    candidate region produced by watershed segmentation.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: FEATURE EXTRACTION")
    print("=" * 60)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    props = measure.regionprops(ws_labels, intensity_image=gray)

    regions = []
    for rp in props:
        if rp.area < 50:
            continue

        shape = extract_shape_features(rp)

        minr, minc, maxr, maxc = rp.bbox
        patch = gray[minr:maxr, minc:maxc].copy()
        local_mask = ws_labels[minr:maxr, minc:maxc] == rp.label
        patch[~local_mask] = 0

        texture = extract_glcm_texture(patch)
        edge = extract_edge_features(patch)

        full_mask = ws_labels == rp.label
        colour = extract_colour_features(
            rgb[full_mask], hsv[full_mask], ndvi[full_mask])

        regions.append({
            "label": rp.label, "centroid": rp.centroid, "bbox": rp.bbox,
            **shape, **texture, **edge, **colour,
        })

    print(f"[Features] {len(regions)} candidate regions analysed.")
    return regions


# ===================================================================
# STAGE 4: RECOGNITION  (Rule-Based Classifier)
# ===================================================================

def classify_trees(regions_data, params=None):
    """
    Rule-based classifier to distinguish trees from non-tree vegetation.

    Scoring rules (weighted):
        1. Crown area within plausible range          (weight 1.0)
        2. Mean NDVI above vegetation threshold       (weight 2.0)
        3. Circularity - tree crowns are roughly round (weight 1.5)
        4. Eccentricity not too elongated             (weight 1.0)
        5. Solidity / compactness                     (weight 1.0)
        6. Green-red ratio (chlorophyll response)     (weight 1.5)
        7. Edge density (canopy texture roughness)    (weight 1.0)

    Confidence = score / max_score.  Tree if confidence >= 0.55.
    """
    print("\n" + "=" * 60)
    print("STAGE 4: RECOGNITION")
    print("=" * 60)

    if params is None:
        params = {
            "min_area": 60, "max_area": 80000,
            "min_ndvi": 0.12, "min_circularity": 0.10,
            "max_eccentricity": 0.97, "min_solidity": 0.25,
            "min_green_red_ratio": 0.7, "min_edge_density": 0.005,
            "min_contrast": 0.5, "max_mean_brightness": 200,
        }

    trees, non_trees, results = [], [], []

    for r in regions_data:
        score, total = 0.0, 0.0

        # Rule 1 - area range (weight 1.0)
        total += 1.0
        if params["min_area"] <= r["area"] <= params["max_area"]:
            score += 1.0

        # Rule 2 - NDVI / greenness (weight 2.0)
        total += 2.0
        if r["mean_ndvi"] >= params["min_ndvi"]:
            score += 2.0
        elif r["mean_ndvi"] >= params["min_ndvi"] * 0.6:
            score += 1.0

        # Rule 3 - circularity (weight 1.0)
        total += 1.0
        if r["circularity"] >= params["min_circularity"]:
            score += 1.0

        # Rule 4 - eccentricity (weight 0.5)
        total += 0.5
        if r["eccentricity"] <= params["max_eccentricity"]:
            score += 0.5

        # Rule 5 - solidity (weight 0.5)
        total += 0.5
        if r["solidity"] >= params["min_solidity"]:
            score += 0.5

        # Rule 6 - green-red ratio (weight 1.5)
        total += 1.5
        if r["green_red_ratio"] >= params["min_green_red_ratio"]:
            score += 1.5

        # Rule 7 - edge density / texture roughness (weight 2.0)
        # Trees have rough canopy texture; grass/smooth surfaces don't
        total += 2.0
        if r["edge_density"] >= params["min_edge_density"]:
            score += 2.0

        # Rule 8 - GLCM contrast (weight 2.0)
        # Trees show high contrast from leaf clusters and shadows
        total += 2.0
        if r["contrast"] >= params["min_contrast"]:
            score += 2.0
        elif r["contrast"] >= params["min_contrast"] * 0.3:
            score += 1.0

        # Rule 9 - not too bright (weight 1.0)
        # Very bright green = grass/lawn, trees are darker
        total += 1.0
        if r["mean_green"] <= params["max_mean_brightness"]:
            score += 1.0

        conf = score / total if total > 0 else 0
        is_tree = conf >= 0.50

        r["confidence"] = conf
        r["is_tree"] = is_tree
        (trees if is_tree else non_trees).append(r)
        results.append((r["label"], is_tree, conf))

    print(f"[Classify] Trees: {len(trees)}, Non-trees: {len(non_trees)}")
    if trees:
        print(f"[Classify] Mean confidence: "
              f"{np.mean([t['confidence'] for t in trees]):.3f}")
    return trees, non_trees, results


# ===================================================================
# STAGE 5: POST-PROCESSING
# ===================================================================

def postprocess(rgb, ws_labels, trees, non_trees):
    """
    Morphological refinement, connected-component labelling,
    annotation drawing, and summary statistics.
    """
    print("\n" + "=" * 60)
    print("STAGE 5: POST-PROCESSING")
    print("=" * 60)

    h, w = ws_labels.shape
    tree_mask = np.zeros((h, w), dtype=bool)
    for r in trees:
        tree_mask |= (ws_labels == r["label"])

    # Morphological clean-up
    tree_mask = morphology.binary_closing(tree_mask, morphology.disk(2))
    tree_mask = morphology.remove_small_objects(tree_mask, min_size=50)

    # Final connected-component count
    final_labels = measure.label(tree_mask)
    final_count = final_labels.max()

    # Annotated image - draw green contours for trees
    annotated = rgb.copy()
    mask_u8 = tree_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    # Red dots for tree centroids with confidence labels
    for r in trees:
        cy, cx = int(r["centroid"][0]), int(r["centroid"][1])
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(annotated, f"{r['confidence']:.0%}", (cx + 5, cy - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Yellow contours for non-tree regions
    for r in non_trees:
        nt_mask = (ws_labels == r["label"]).astype(np.uint8) * 255
        cnt, _ = cv2.findContours(nt_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnt, -1, (0, 255, 255), 1)

    areas = [r["area"] for r in trees]
    stats = {
        "total_trees": final_count,
        "coverage_pct": np.sum(tree_mask) / tree_mask.size * 100,
        "mean_area": np.mean(areas) if areas else 0,
        "std_area": np.std(areas) if areas else 0,
        "min_area": np.min(areas) if areas else 0,
        "max_area": np.max(areas) if areas else 0,
        "mean_conf": np.mean([r["confidence"] for r in trees]) if trees else 0,
    }

    print(f"[Post] Trees: {final_count}, coverage: {stats['coverage_pct']:.2f}%")
    return tree_mask, final_count, annotated, stats


# ===================================================================
# VISUALISATION
# ===================================================================

def save_visualisations(rgb, ndvi, veg_mask, km_labels, combined,
                         ws_labels, tree_mask, annotated, stats,
                         out_dir, name):
    """Generate a 3x3 pipeline overview figure and standalone result."""
    rgb_d = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ann_d = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f"Tree Crown Detection Pipeline - {name}",
                  fontsize=16, fontweight="bold")

    ax[0, 0].imshow(rgb_d); ax[0, 0].set_title("(a) Original"); ax[0, 0].axis("off")

    im = ax[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.8)
    ax[0, 1].set_title("(b) NDVI"); ax[0, 1].axis("off")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)

    ax[0, 2].imshow(veg_mask, cmap="gray")
    ax[0, 2].set_title("(c) NDVI Threshold (Otsu)"); ax[0, 2].axis("off")

    ax[1, 0].imshow(km_labels, cmap="tab10")
    ax[1, 0].set_title("(d) K-means Clustering"); ax[1, 0].axis("off")

    ax[1, 1].imshow(combined, cmap="gray")
    ax[1, 1].set_title("(e) Combined Vegetation Mask"); ax[1, 1].axis("off")

    # Watershed colour map
    ws_rgb = np.zeros((*ws_labels.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    for lbl in np.unique(ws_labels):
        if lbl == 0:
            continue
        ws_rgb[ws_labels == lbl] = np.random.randint(50, 255, 3)
    ax[1, 2].imshow(ws_rgb)
    ax[1, 2].set_title("(f) Watershed Segmentation"); ax[1, 2].axis("off")

    ax[2, 0].imshow(tree_mask, cmap="Greens")
    ax[2, 0].set_title("(g) Classified Tree Mask"); ax[2, 0].axis("off")

    ax[2, 1].imshow(ann_d)
    ax[2, 1].set_title("(h) Detected Crowns"); ax[2, 1].axis("off")

    ax[2, 2].axis("off")
    txt = (f"Summary Statistics\n{'='*30}\n"
           f"Trees Detected:  {stats['total_trees']}\n"
           f"Coverage:        {stats['coverage_pct']:.2f}%\n"
           f"Mean Crown Area: {stats['mean_area']:.1f} px\n"
           f"Std Crown Area:  {stats['std_area']:.1f} px\n"
           f"Mean Confidence: {stats['mean_conf']:.3f}")
    ax[2, 2].text(0.1, 0.5, txt, transform=ax[2, 2].transAxes,
                   fontsize=12, va="center", family="monospace",
                   bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.3))
    ax[2, 2].set_title("(i) Statistics")

    plt.tight_layout()
    p1 = os.path.join(out_dir, f"{name}_pipeline_overview.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Viz] Saved: {p1}")

    # Standalone result
    fig2, a2 = plt.subplots(figsize=(14, 12))
    a2.imshow(ann_d)
    a2.set_title(f"Result - {stats['total_trees']} Trees Detected",
                  fontsize=14, fontweight="bold")
    a2.axis("off")
    p2 = os.path.join(out_dir, f"{name}_result.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[Viz] Saved: {p2}")
    return p1, p2


def save_feature_analysis(trees, non_trees, out_dir, name):
    """Histogram comparison of features: tree vs non-tree."""
    if not trees:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Feature Analysis: Tree vs Non-Tree",
                  fontsize=14, fontweight="bold")

    pairs = [("area", "Area (px)"), ("circularity", "Circularity"),
             ("mean_ndvi", "Mean NDVI"), ("eccentricity", "Eccentricity"),
             ("edge_density", "Edge Density"), ("contrast", "GLCM Contrast")]

    for i, (key, label) in enumerate(pairs):
        ax = axes[i // 3, i % 3]
        tv = [r[key] for r in trees if key in r]
        nv = [r[key] for r in non_trees if key in r]
        if tv:
            ax.hist(tv, bins=20, alpha=0.6, color="green", label="Tree")
        if nv:
            ax.hist(nv, bins=20, alpha=0.6, color="red", label="Non-tree")
        ax.set_xlabel(label); ax.set_ylabel("Count"); ax.legend(fontsize=8)

    plt.tight_layout()
    p = os.path.join(out_dir, f"{name}_feature_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Viz] Saved: {p}")
    return p


# ===================================================================
# EVALUATION
# ===================================================================

def evaluate_detection(pred_mask, gt_mask):
    """
    Pixel-level evaluation: precision, recall, F1, IoU, accuracy.
    """
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    tp = np.sum(p & g)
    fp = np.sum(p & ~g)
    fn = np.sum(~p & g)
    tn = np.sum(~p & ~g)

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    acc  = (tp + tn) / (tp + tn + fp + fn)

    m = {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "accuracy": acc}
    print(f"[Eval] P={prec:.4f} R={rec:.4f} F1={f1:.4f} IoU={iou:.4f}")
    return m


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def run_pipeline(image_path, output_dir=OUTPUT_DIR, min_distance=15,
                 n_clusters=4, classifier_params=None):
    """Execute the complete 5-stage tree detection pipeline."""
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'#'*60}\n  TREE CROWN DETECTION PIPELINE\n  Image: {name}\n{'#'*60}")

    rgb, nir = load_image(image_path)
    enhanced, ndvi, hsv, texture_map, exclusion_mask = preprocess(rgb, nir)
    veg_mask, km_labels, combined, ws_labels = segment(
        enhanced, hsv, ndvi, min_distance, n_clusters,
        texture_map, exclusion_mask)
    regions = extract_features(enhanced, hsv, ndvi, ws_labels, combined)
    trees, non_trees, _ = classify_trees(regions, classifier_params)
    tree_mask, count, annotated, stats = postprocess(
        enhanced, ws_labels, trees, non_trees)

    # Save outputs
    cv2.imwrite(os.path.join(output_dir, f"{name}_annotated.png"), annotated)
    cv2.imwrite(os.path.join(output_dir, f"{name}_tree_mask.png"),
                tree_mask.astype(np.uint8) * 255)

    save_visualisations(enhanced, ndvi, veg_mask, km_labels, combined,
                         ws_labels, tree_mask, annotated, stats,
                         output_dir, name)
    save_feature_analysis(trees, non_trees, output_dir, name)

    print(f"\n{'='*60}\nPIPELINE COMPLETE\n{'='*60}")
    print(f"  Trees: {count} | Coverage: {stats['coverage_pct']:.2f}% "
          f"| Confidence: {stats['mean_conf']:.3f}")
    print(f"  Output: {output_dir}\n{'='*60}")

    return {"name": name, "rgb": rgb, "ndvi": ndvi, "veg_mask": veg_mask,
            "km_labels": km_labels, "combined": combined, "ws_labels": ws_labels,
            "tree_mask": tree_mask, "annotated": annotated,
            "trees": trees, "non_trees": non_trees, "stats": stats}


def batch_process(image_dir, output_dir=OUTPUT_DIR, **kw):
    """Process all supported images in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = sorted(f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in exts)
    if not files:
        print(f"No images in {image_dir}"); return []

    results = []
    for f in files:
        try:
            results.append(run_pipeline(
                os.path.join(image_dir, f), output_dir, **kw))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    if results:
        print(f"\n{'='*70}\nBATCH SUMMARY\n{'='*70}")
        print(f"{'Image':<30}{'Trees':>6}{'Coverage':>10}{'Conf':>8}")
        for r in results:
            s = r["stats"]
            print(f"{r['name']:<30}{s['total_trees']:>6}"
                  f"{s['coverage_pct']:>9.2f}%{s['mean_conf']:>7.3f}")
    return results


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Tree Crown Detection - Classical Computer Vision")
    ap.add_argument("input", help="Image file or directory")
    ap.add_argument("--output", "-o", default=OUTPUT_DIR)
    ap.add_argument("--min-distance", "-d", type=int, default=15)
    ap.add_argument("--clusters", "-k", type=int, default=4)
    ap.add_argument("--ground-truth", "-gt", default=None)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        batch_process(args.input, args.output,
                       min_distance=args.min_distance,
                       n_clusters=args.clusters)
    else:
        res = run_pipeline(args.input, args.output,
                            min_distance=args.min_distance,
                            n_clusters=args.clusters)
        if args.ground_truth:
            gt = cv2.imread(args.ground_truth, cv2.IMREAD_GRAYSCALE) > 127
            evaluate_detection(res["tree_mask"], gt)
