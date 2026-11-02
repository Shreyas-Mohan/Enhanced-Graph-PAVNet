import matplotlib.pyplot as plt
import numpy as np


def render_prediction_overlay(ct_slice, seg_slice, title="", ax=None, alpha=0.6):
    """
    Renders 2D axial slice with pulmonary artery (Red) and vein (Blue) segmentation overlay.
      - 0: Background
      - 1: Pulmonary Artery (Red)
      - 2: Pulmonary Vein (Cyan/Blue)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize CT for display [-1000, 600] HU
    ct_norm = np.clip((ct_slice + 1000.0) / 1600.0, 0.0, 1.0)
    ax.imshow(ct_norm, cmap="gray", origin="lower")

    if seg_slice is not None:
        rgba = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
        # Artery: Red
        rgba[seg_slice == 1] = [1.0, 0.1, 0.1, alpha]
        # Vein: Blue
        rgba[seg_slice == 2] = [0.1, 0.6, 1.0, alpha]
        ax.imshow(rgba, origin="lower")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    return ax

