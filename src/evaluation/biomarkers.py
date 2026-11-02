import numpy as np
import scipy.ndimage as ndi


def compute_copd_imaging_biomarkers(pred_vol, radii=None, spacing=(1.0, 1.0, 1.0)):
    """
    Computes vascular clinical imaging biomarkers:
      - Artery-to-Vein Ratio (AVR)
      - Small Vessel Volume Percentage (BV5%): vessels with radius <= 1.5 physical mm.
    """
    art_count = np.sum(pred_vol == 1)
    vein_count = np.sum(pred_vol == 2)
    avr = float(art_count) / float(max(1, vein_count))

    if radii is not None:
        # Direct graph node caliber calculation
        total_nodes = len(radii)
        small_nodes = np.sum(radii <= 1.5)
        bv5 = float(small_nodes) / float(max(1, total_nodes)) * 100.0
    else:
        # Distance transform based estimation on 3D volume
        vessel_mask = (pred_vol > 0).astype(np.uint8)
        if np.sum(vessel_mask) > 0:
            dist_map = ndi.distance_transform_edt(vessel_mask, sampling=spacing)
            small_voxels = np.sum((vessel_mask > 0) & (dist_map <= 1.5))
            bv5 = float(small_voxels) / float(np.sum(vessel_mask)) * 100.0
        else:
            bv5 = 0.0

    return {
        "avr": round(avr, 4),
        "bv5_pct": round(bv5, 2),
        "artery_volume_voxels": int(art_count),
        "vein_volume_voxels": int(vein_count),
    }

