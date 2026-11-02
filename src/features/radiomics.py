import torch
import numpy as np
import scipy.ndimage as ndi


def compute_radiomic_features(coords, vol_arr):
    """
    Computes 8 multi-scale radiomic features from the 3D CT volume:
      1. Raw CT attenuation (HU / 1000)
      2. 3D local neighborhood mean attenuation (5x5x5 filter)
      3. 3D local neighborhood intensity standard deviation (texture heterogeneity)
      4. 3D Sobel gradient magnitude (vessel boundary sharpness)
      5. Multi-scale LoG vesselness at sigma=1.0 (fine tubular filter)
      6. Multi-scale LoG vesselness at sigma=2.0 (medium tubular filter)
      7. Multi-scale LoG vesselness at sigma=3.0 (coarse tubular filter)
      8. Center-to-outer shell contrast ratio (5x5x5 vs 11x11x11)
    """
    device = coords.device
    vol_arr = vol_arr.astype(np.float32)

    # Multi-scale 3D Laplacian of Gaussian filters
    log_s1 = ndi.gaussian_laplace(vol_arr, sigma=1.0)
    log_s2 = ndi.gaussian_laplace(vol_arr, sigma=2.0)
    log_s3 = ndi.gaussian_laplace(vol_arr, sigma=3.0)

    # 3D Gradient magnitude
    grad_x = ndi.sobel(vol_arr, axis=0)
    grad_y = ndi.sobel(vol_arr, axis=1)
    grad_z = ndi.sobel(vol_arr, axis=2)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Local uniform window filters
    local_mean = ndi.uniform_filter(vol_arr, size=5)
    local_sq = ndi.uniform_filter(vol_arr**2, size=5)
    local_std = np.sqrt(np.maximum(0, local_sq - local_mean**2))

    # Outer parenchymal shell filter
    outer_mean = ndi.uniform_filter(vol_arr, size=11)

    # Sample at landmark coordinates
    idx_coords = coords.long().cpu().numpy()
    X, Y, Z = vol_arr.shape
    ix = np.clip(idx_coords[:, 0], 0, X - 1)
    iy = np.clip(idx_coords[:, 1], 0, Y - 1)
    iz = np.clip(idx_coords[:, 2], 0, Z - 1)

    raw_hu = vol_arr[ix, iy, iz, None]
    sampled_mean = local_mean[ix, iy, iz, None]
    sampled_std = local_std[ix, iy, iz, None]
    sampled_grad = grad_mag[ix, iy, iz, None]
    sampled_log1 = log_s1[ix, iy, iz, None]
    sampled_log2 = log_s2[ix, iy, iz, None]
    sampled_log3 = log_s3[ix, iy, iz, None]
    contrast = (raw_hu - outer_mean[ix, iy, iz, None])

    radiomic_features = torch.from_numpy(np.hstack([
        raw_hu / 1000.0,
        sampled_mean / 1000.0,
        sampled_std / 500.0,
        sampled_grad / 5000.0,
        sampled_log1 / 500.0,
        sampled_log2 / 200.0,
        sampled_log3 / 100.0,
        contrast / 500.0,
    ])).float().to(device)

    return radiomic_features

