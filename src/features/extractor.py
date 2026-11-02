import torch
from .morphometry import compute_vascular_geometry_features
from .radiomics import compute_radiomic_features
from .topology import compute_topological_features


class FeatureExtractorPipeline:
    """
    Extracts and compiles candidate and optimal feature sets for vascular point clouds.
    """
    def __init__(self):
        pass

    def extract_optimal_18_features(self, coords, radii, edge_index, seed_mask, vol_arr):
        """
        Extracts the 18 optimal descriptors (602d enriched representation):
          - 7 Vascular Geometry Features
          - 8 Multi-Scale Radiomic Features
          - 3 High-Impact Literature Enhancements:
            1. Scale-invariant normalized caliber: r_i / r_trunk
            2. Topological Gaussian radial decay profile
            3. Intramural attenuation Z-contrast gradient
        """
        geom_7 = compute_vascular_geometry_features(coords, radii, edge_index, seed_mask)
        rad_8 = compute_radiomic_features(coords, vol_arr)

        # 1. Scale-invariant normalized caliber
        r_trunk = torch.quantile(radii, 0.99)
        norm_caliber = (radii / (r_trunk + 1e-6)).unsqueeze(1)

        # 2. Gaussian radial decay profile
        gauss_radial = torch.exp(- (radii.unsqueeze(1) - r_trunk)**2 / (2 * (r_trunk / 2 + 1e-6)**2))

        # 3. Intramural attenuation Z-contrast gradient
        raw_hu = rad_8[:, 0:1]
        local_mean = rad_8[:, 1:2]
        local_std = rad_8[:, 2:3]
        z_contrast = torch.clamp((raw_hu - local_mean) / (local_std + 1e-4), -5.0, 5.0)

        optimal_18 = torch.cat([geom_7, rad_8, norm_caliber, gauss_radial, z_contrast], dim=1)
        return torch.nan_to_num(optimal_18, nan=0.0, posinf=1.0, neginf=-1.0)

    def extract_all_candidate_features(self, coords, radii, edge_index, seed_mask, vol_arr):
        """Extracts complete 20-dim candidate pool (7 geometry + 5 topology + 8 radiomics)."""
        geom_7 = compute_vascular_geometry_features(coords, radii, edge_index, seed_mask)
        topo_5 = compute_topological_features(coords, edge_index, seed_mask)
        rad_8 = compute_radiomic_features(coords, vol_arr)

        all_20 = torch.cat([geom_7, topo_5, rad_8], dim=1)
        return torch.nan_to_num(all_20, nan=0.0, posinf=1.0, neginf=-1.0)

