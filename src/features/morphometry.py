import torch


def compute_vascular_geometry_features(coords, radii, edge_index, seed_mask):
    """
    Computes 7 vascular geometry features:
      1. Local vessel caliber radius: r_i = D(v_i, dOmega)
      2. Caliber tapering rate: mean((r_j - r_i) / dist_ij)
      3. Relative neighbor caliber: r_i / mean(r_j)
      4. Craniocaudal orientation: mean(|dz| / dist_ij)
      5. Branching alignment / dispersion: ||mean(u_ij)||_2
      6. Distance to hilum centroid: ||v_i - c_hilum||_2 / 100.0
      7. Murray's law cubic residual: (r_i^3 - mean(r_j^3)) / (r_i^3 + eps)
    """
    N = coords.size(0)
    device = coords.device
    src, dst = edge_index[0], edge_index[1]

    diff = coords[dst] - coords[src]
    dist = torch.norm(diff, dim=1, keepdim=True).clamp_min(1e-6)
    direction = diff / dist

    degree = torch.bincount(src, minlength=N).clamp_min(1).float().unsqueeze(1)

    # 1. Tapering rate along edges
    tapering_edge = (radii[dst] - radii[src]).unsqueeze(1) / dist
    tapering_mean = torch.zeros(N, 1, device=device)
    tapering_mean.index_add_(0, src, tapering_edge)
    tapering_mean = tapering_mean / degree

    # 2. Relative neighbor caliber
    neighbor_radii_sum = torch.zeros(N, 1, device=device)
    neighbor_radii_sum.index_add_(0, src, radii[dst].unsqueeze(1))
    relative_radius = radii.unsqueeze(1) / (neighbor_radii_sum / degree + 1e-6)

    # 3. Craniocaudal orientation (Z-angle)
    z_orient_edge = torch.abs(direction[:, 2:3])
    z_orient_mean = torch.zeros(N, 1, device=device)
    z_orient_mean.index_add_(0, src, z_orient_edge)
    z_orient_mean = z_orient_mean / degree

    # 4. Branching alignment / dispersion
    dir_sum = torch.zeros(N, 3, device=device)
    dir_sum.index_add_(0, src, direction)
    branch_alignment = torch.norm(dir_sum / degree, dim=1, keepdim=True)

    # 5. Distance to hilum centroid
    hilum_nodes = coords[seed_mask] if seed_mask.any() else coords[:10]
    hilum_center = hilum_nodes.mean(dim=0, keepdim=True)
    dist_to_hilum = torch.norm(coords - hilum_center, dim=1, keepdim=True) / 100.0

    # 6. Murray's law cubic residual
    cubed_radii = radii.pow(3).unsqueeze(1)
    cubed_sum = torch.zeros(N, 1, device=device)
    cubed_sum.index_add_(0, src, radii[dst].pow(3).unsqueeze(1))
    murray_residual = (cubed_radii - (cubed_sum / degree)) / (cubed_radii + 1e-6)
    murray_residual = torch.clamp(murray_residual, -5.0, 5.0)

    geom_features = torch.cat([
        radii.unsqueeze(1),
        tapering_mean,
        relative_radius,
        z_orient_mean,
        branch_alignment,
        dist_to_hilum,
        murray_residual,
    ], dim=1)

    return geom_features

