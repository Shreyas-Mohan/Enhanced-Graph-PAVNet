import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_point_cloud_hd95(coords_a, coords_b):
    """
    Computes bidirectional 95th-percentile Hausdorff Distance (HD95) between two 3D point sets.
    """
    if coords_a is None or coords_b is None or len(coords_a) == 0 or len(coords_b) == 0:
        return float("nan")
    d_a2b, _ = KDTree(coords_b).query(coords_a)
    d_b2a, _ = KDTree(coords_a).query(coords_b)
    hd_a2b = np.percentile(d_a2b, 95)
    hd_b2a = np.percentile(d_b2a, 95)
    return float(max(hd_a2b, hd_b2a))


def evaluate_vascular_predictions(y_true, y_pred, coords=None, pixel_spacing=(1.0, 1.0, 1.0)):
    """
    Computes all 7 canonical standardized metrics across pulmonary artery and vein:
      1. Accuracy (%)
      2. Precision (%)
      3. Recall (%)
      4. Specificity (%)
      5. Dice Similarity Coefficient (DSC)
      6. Mean Intersection over Union (mIoU)
      7. 95th-percentile Hausdorff Distance (HD95, in physical mm)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = accuracy_score(y_true, y_pred) * 100.0
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Class 0: Artery, Class 1: Vein
    art_dice = float(f1[0])
    vein_dice = float(f1[1])
    macro_dice = float(np.mean(f1))

    art_p, vein_p = float(p[0] * 100.0), float(p[1] * 100.0)
    art_r, vein_r = float(r[0] * 100.0), float(r[1] * 100.0)

    # Specificity from confusion matrix: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    c00, c01, c10, c11 = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    # For Artery (class 0), negatives are Vein (class 1): TN=c11, FP=c10
    art_spec = float(c11 / max(1, c11 + c10) * 100.0)
    # For Vein (class 1), negatives are Artery (class 0): TN=c00, FP=c01
    vein_spec = float(c00 / max(1, c00 + c01) * 100.0)
    macro_spec = float((art_spec + vein_spec) / 2.0)

    # Intersection over Union
    iou_art = float(f1[0] / (2.0 - f1[0])) if (2.0 - f1[0]) > 0 else 0.0
    iou_vein = float(f1[1] / (2.0 - f1[1])) if (2.0 - f1[1]) > 0 else 0.0
    miou = float((iou_art + iou_vein) / 2.0)

    # Physical Hausdorff Distance
    hd95 = float("nan")
    if coords is not None:
        coords_art_true = coords[y_true == 0] * np.array(pixel_spacing)
        coords_art_pred = coords[y_pred == 0] * np.array(pixel_spacing)
        coords_vein_true = coords[y_true == 1] * np.array(pixel_spacing)
        coords_vein_pred = coords[y_pred == 1] * np.array(pixel_spacing)

        hd_art = compute_point_cloud_hd95(coords_art_true, coords_art_pred)
        hd_vein = compute_point_cloud_hd95(coords_vein_true, coords_vein_pred)
        hd95 = float(np.nanmean([hd_art, hd_vein]))

    return {
        "accuracy": round(acc, 2),
        "precision": round(float(np.mean(p) * 100.0), 2),
        "recall": round(float(np.mean(r) * 100.0), 2),
        "specificity": round(macro_spec, 2),
        "dice": round(macro_dice, 4),
        "miou": round(miou, 4),
        "hd95": round(hd95, 2) if not np.isnan(hd95) else None,
        "artery_dice": round(art_dice, 4),
        "vein_dice": round(vein_dice, 4),
        "artery_precision": round(art_p, 2),
        "vein_precision": round(vein_p, 2),
        "artery_recall": round(art_r, 2),
        "vein_recall": round(vein_r, 2),
    }

