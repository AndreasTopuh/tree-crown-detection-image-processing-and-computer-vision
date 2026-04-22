"""
Evaluation Metrics for Tree Crown Detection Pipeline

This script generates quantitative evaluation metrics by:
1. Creating approximate ground truth via manual-assisted thresholding
2. Computing pixel-level metrics (Precision, Recall, F1, IoU)
3. Computing object-level metrics (ORR-like detection rate)
4. Generating comparison visualisations

Paper reference metrics (Tong et al., 2021):
    - ORR (Overall Recognition Rate): percentage of correctly delineated crowns
    - SEI (Segmentation Error Index): measures over/under-segmentation
    - Merged: single delineation covers multiple reference crowns
    - Split: multiple delineations cover a single reference crown
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from tree_detection_pipeline import run_pipeline, OUTPUT_DIR


def create_ground_truth_from_green(rgb_path, output_path=None):
    """
    Generate approximate ground truth for TREE regions (not grass).

    Tree detection criteria:
        - Green hue (HSV hue 20-100)
        - Not too bright/smooth (grass is bright + smooth)
        - Not shadow (very dark)
        - Has texture (trees have rough canopy; grass is smooth)
    """
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {rgb_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Green hue range
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_mask = (hue >= 20) & (hue <= 100) & (sat > 12)

    # Exclude shadows (very dark)
    not_shadow = val >= 35

    # Exclude bright smooth areas (grass/lawn)
    not_bright_smooth = ~((val > 180) & (sat < 80))

    # Texture filter: trees have rough canopy texture
    gray_f = gray.astype(np.float64)
    mean_f = cv2.blur(gray_f, (15, 15))
    sqmean_f = cv2.blur(gray_f ** 2, (15, 15))
    local_std = np.sqrt(np.maximum(sqmean_f - mean_f ** 2, 0))

    # Otsu on texture of green pixels
    green_px = green_mask & not_shadow
    if np.sum(green_px) > 100:
        tex_vals = local_std[green_px]
        tex_u8 = ((tex_vals - tex_vals.min()) /
                   (tex_vals.max() - tex_vals.min() + 1e-10) * 255
                   ).astype(np.uint8)
        t, _ = cv2.threshold(tex_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tex_thresh = (t / 255.0 * (tex_vals.max() - tex_vals.min())
                       + tex_vals.min()) * 0.6
        has_texture = local_std >= tex_thresh
    else:
        has_texture = np.ones_like(green_mask)

    gt_mask = (green_mask & not_shadow & not_bright_smooth &
               has_texture).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gt_mask = cv2.morphologyEx(gt_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    gt_mask = cv2.morphologyEx(gt_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if output_path:
        cv2.imwrite(output_path, gt_mask)
        print(f"[GT] Ground truth saved: {output_path}")

    return gt_mask > 0


def pixel_level_metrics(pred_mask, gt_mask):
    """
    Compute pixel-level evaluation metrics.

    Returns: dict with precision, recall, f1, iou, accuracy
    """
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4),
        "accuracy": round(acc, 4),
    }


def object_level_metrics(pred_mask, gt_mask, iou_threshold=0.3):
    """
    Compute object-level detection metrics similar to paper's ORR.

    For each ground truth crown region, check if it overlaps with
    a predicted region above the IoU threshold.

    Returns: dict with detection_rate (ORR), merged, split counts
    """
    gt_labels = measure.label(gt_mask.astype(int))
    pred_labels = measure.label(pred_mask.astype(int))

    gt_count = gt_labels.max()
    pred_count = pred_labels.max()

    if gt_count == 0:
        return {"gt_objects": 0, "pred_objects": pred_count,
                "ORR": 0, "merged": 0, "split": 0}

    correctly_detected = 0
    merged = 0
    split = 0

    for gt_id in range(1, gt_count + 1):
        gt_region = gt_labels == gt_id
        gt_area = np.sum(gt_region)

        if gt_area < 30:
            continue

        # Find overlapping predicted regions
        overlapping_pred_ids = np.unique(pred_labels[gt_region])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        if len(overlapping_pred_ids) == 0:
            continue  # missed

        # Compute IoU with best-matching prediction
        best_iou = 0
        for pid in overlapping_pred_ids:
            pred_region = pred_labels == pid
            intersection = np.sum(gt_region & pred_region)
            union = np.sum(gt_region | pred_region)
            this_iou = intersection / union if union > 0 else 0
            best_iou = max(best_iou, this_iou)

        if best_iou >= iou_threshold:
            correctly_detected += 1

        # Check for merge: single pred covers >50% of multiple GT
        if len(overlapping_pred_ids) == 1:
            pred_region = pred_labels == overlapping_pred_ids[0]
            other_gts = np.unique(gt_labels[pred_region])
            other_gts = other_gts[other_gts > 0]
            if len(other_gts) > 1:
                merged += 1

        # Check for split: multiple preds cover >50% of single GT
        if len(overlapping_pred_ids) > 1:
            total_overlap = 0
            for pid in overlapping_pred_ids:
                pred_region = pred_labels == pid
                total_overlap += np.sum(gt_region & pred_region)
            if total_overlap / gt_area > 0.5:
                split += 1

    valid_gt = sum(1 for i in range(1, gt_count + 1)
                   if np.sum(gt_labels == i) >= 30)
    orr = correctly_detected / valid_gt if valid_gt > 0 else 0

    return {
        "gt_objects": valid_gt,
        "pred_objects": pred_count,
        "correctly_detected": correctly_detected,
        "ORR": round(orr, 4),
        "merged": merged,
        "split": split,
    }


def save_evaluation_figure(rgb_path, pred_mask, gt_mask,
                            pixel_m, object_m, out_dir, name):
    """Generate side-by-side comparison of prediction vs ground truth."""
    img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Evaluation: {name}", fontsize=16, fontweight="bold")

    # Row 1: visual comparison
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("(a) Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_mask, cmap="Greens")
    axes[0, 1].set_title("(b) Ground Truth (HSV-based)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_mask, cmap="Greens")
    axes[0, 2].set_title("(c) Pipeline Prediction")
    axes[0, 2].axis("off")

    # Row 2: overlap analysis + metrics
    # TP/FP/FN overlay
    h, w = pred_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    overlay[p & g] = [0, 255, 0]    # TP = green
    overlay[p & ~g] = [255, 0, 0]   # FP = red
    overlay[~p & g] = [0, 0, 255]   # FN = blue

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("(d) TP(green) FP(red) FN(blue)")
    axes[1, 0].axis("off")

    # Contour overlay
    contour_img = img.copy()
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, gt_contours, -1, (255, 255, 0), 2)
    cv2.drawContours(contour_img, pred_contours, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(contour_img)
    axes[1, 1].set_title("(e) GT(yellow) vs Pred(green)")
    axes[1, 1].axis("off")

    # Metrics text panel
    axes[1, 2].axis("off")
    metrics_text = (
        f"PIXEL-LEVEL METRICS\n"
        f"{'='*35}\n"
        f"Precision:  {pixel_m['precision']:.4f}\n"
        f"Recall:     {pixel_m['recall']:.4f}\n"
        f"F1 Score:   {pixel_m['f1_score']:.4f}\n"
        f"IoU:        {pixel_m['iou']:.4f}\n"
        f"Accuracy:   {pixel_m['accuracy']:.4f}\n\n"
        f"OBJECT-LEVEL METRICS\n"
        f"{'='*35}\n"
        f"GT Objects:      {object_m['gt_objects']}\n"
        f"Pred Objects:    {object_m['pred_objects']}\n"
        f"Correct:         {object_m['correctly_detected']}\n"
        f"ORR:             {object_m['ORR']:.4f}\n"
        f"Merged:          {object_m['merged']}\n"
        f"Split:           {object_m['split']}\n"
    )
    axes[1, 2].text(0.05, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                     fontsize=11, va="center", family="monospace",
                     bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.5))
    axes[1, 2].set_title("(f) Evaluation Metrics")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name}_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved: {out_path}")
    return out_path


def evaluate_image(image_path, output_dir=OUTPUT_DIR, **pipeline_kwargs):
    """
    Run full evaluation on a single image:
    1. Run detection pipeline
    2. Generate approximate ground truth
    3. Compute pixel-level and object-level metrics
    4. Save evaluation visualisation
    """
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"{'='*60}")

    # Run pipeline
    result = run_pipeline(image_path, output_dir, **pipeline_kwargs)
    pred_mask = result["tree_mask"]

    # Generate ground truth
    gt_path = os.path.join(output_dir, f"{name}_ground_truth.png")
    gt_mask = create_ground_truth_from_green(image_path, gt_path)

    # Resize if needed
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    # Compute metrics
    pixel_m = pixel_level_metrics(pred_mask, gt_mask)
    object_m = object_level_metrics(pred_mask, gt_mask)

    print(f"\n  Pixel: P={pixel_m['precision']:.4f} R={pixel_m['recall']:.4f} "
          f"F1={pixel_m['f1_score']:.4f} IoU={pixel_m['iou']:.4f}")
    print(f"  Object: ORR={object_m['ORR']:.4f} "
          f"Merged={object_m['merged']} Split={object_m['split']}")

    # Save evaluation figure
    save_evaluation_figure(image_path, pred_mask, gt_mask,
                            pixel_m, object_m, output_dir, name)

    return {
        "name": name,
        "pixel_metrics": pixel_m,
        "object_metrics": object_m,
        "pipeline_stats": result["stats"],
    }


def run_full_evaluation(data_dir, output_dir=OUTPUT_DIR):
    """Evaluate all images in data directory and print summary table."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = sorted(f for f in os.listdir(data_dir)
                   if os.path.splitext(f)[1].lower() in exts)

    if not files:
        print(f"No images found in {data_dir}")
        return

    all_results = []
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            r = evaluate_image(path, output_dir)
            all_results.append(r)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    # Summary table
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Image':<20} {'Trees':>6} {'Cover%':>7} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'IoU':>7} "
          f"{'ORR':>7} {'Merge':>6} {'Split':>6}")
    print(f"{'-'*90}")

    for r in all_results:
        pm = r["pixel_metrics"]
        om = r["object_metrics"]
        ps = r["pipeline_stats"]
        print(f"{r['name']:<20} {ps['total_trees']:>6} "
              f"{ps['coverage_pct']:>6.2f}% "
              f"{pm['precision']:>7.4f} {pm['recall']:>7.4f} "
              f"{pm['f1_score']:>7.4f} {pm['iou']:>7.4f} "
              f"{om['ORR']:>7.4f} {om['merged']:>6} {om['split']:>6}")

    # Averages
    if all_results:
        avg_p = np.mean([r["pixel_metrics"]["precision"] for r in all_results])
        avg_r = np.mean([r["pixel_metrics"]["recall"] for r in all_results])
        avg_f1 = np.mean([r["pixel_metrics"]["f1_score"] for r in all_results])
        avg_iou = np.mean([r["pixel_metrics"]["iou"] for r in all_results])
        avg_orr = np.mean([r["object_metrics"]["ORR"] for r in all_results])
        print(f"{'-'*90}")
        print(f"{'AVERAGE':<20} {'':>6} {'':>7} "
              f"{avg_p:>7.4f} {avg_r:>7.4f} "
              f"{avg_f1:>7.4f} {avg_iou:>7.4f} "
              f"{avg_orr:>7.4f}")

    return all_results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    run_full_evaluation(data_dir)
