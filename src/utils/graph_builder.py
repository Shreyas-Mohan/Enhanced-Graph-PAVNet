import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


def layerwise_bfs_propagation(edge_index, seed_mask):
    """
    Breadth-First Search (BFS) from cardiac hilum seed nodes to extract hierarchy levels.
    """
    visited = seed_mask.clone()
    current_layer = seed_mask.nonzero(as_tuple=True)[0]
    layers = [current_layer]

    while current_layer.numel() > 0:
        mask = torch.isin(edge_index[0], current_layer)
        neighbors = edge_index[1][mask]

        unvisited_mask = ~visited[neighbors]
        next_layer = neighbors[unvisited_mask].unique()

        if next_layer.numel() == 0:
            break

        visited[next_layer] = True
        layers.append(next_layer)
        current_layer = next_layer

    return layers


def build_skeleton_graph(coords, radii, k=3, max_hops=3):
    """
    Builds a 3D spatial k-NN landmark graph from vascular skeleton landmark nodes.
    
    Robustness against Upstream Segmentation Fractures:
      Unlike traditional 1D topological centerlines that break permanently when a 1-voxel
      partial volume gap or motion artifact occurs, this representation constructs a 3D
      spatial graph over landmark point clouds (e.g., N=8,192).
      - Connecting nodes via spatial Euclidean k-NN (default k=3, bidirectional) bridges
        physical 1-3 mm micro-gaps in 3D Euclidean space.
      - Self-normalizing attention in HGAT seamlessly propagates vascular context across
        local discontinuities.
      - Hilum root discovery integrates caliber radius and cardiac proximity to anchor
        the main pulmonary trunk without requiring unbroken proximal centerlines.

    Args:
        coords (Tensor): 3D node coordinates [N, 3]
        radii (Tensor): Node radius estimates [N]
        k (int): Number of nearest neighbors (default 3)
        max_hops (int): Maximum BFS hierarchy depth (default 3)

    Returns:
        Data: PyTorch Geometric Data object with edge_index, seed_mask, hierarchy_mask, direction_features.
    """
    N = coords.size(0)
    coords_np = coords.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords_np)
    distances, indices = nbrs.kneighbors(coords_np)

    src = torch.tensor(indices[:, 0].repeat(k), dtype=torch.long)
    dst = torch.tensor(indices[:, 1:].flatten(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # Ensure bidirectional connectivity
    rev_edges = edge_index[[1, 0]]
    edge_index = torch.unique(torch.cat([edge_index, rev_edges], dim=1), dim=1)

    # Hilum Root Seed Detection: largest vessels near cardiac hilum center
    center = coords.mean(dim=0)
    dist_to_center = torch.norm(coords - center, dim=1)
    seed_scores = radii - (dist_to_center * 0.1)
    k_seeds = max(10, int(0.02 * N))
    seed_idx = torch.topk(seed_scores, k=min(k_seeds, N))[1]
    seed_mask = torch.zeros(N, dtype=torch.bool)
    seed_mask[seed_idx] = True

    # BFS Hierarchy mask
    layers = layerwise_bfs_propagation(edge_index, seed_mask)
    hierarchy_mask = torch.zeros((N, max_hops + 1), dtype=torch.float32)
    for hop, layer in enumerate(layers):
        if hop > max_hops:
            break
        hierarchy_mask[layer, hop] = 1.0

    # Direction & distance statistics over bidirectional edges
    source, target = edge_index
    diffs = coords[target] - coords[source]
    dist = torch.norm(diffs, dim=1, keepdim=True).clamp_min(1e-6)
    unit_diffs = diffs / dist

    direction_mean = torch.zeros(N, 3)
    dist_mean = torch.zeros(N, 1)
    deg = torch.bincount(source, minlength=N).clamp_min(1).unsqueeze(1).float()

    direction_mean.index_add_(0, source, unit_diffs)
    dist_mean.index_add_(0, source, dist)

    direction_features = torch.cat([direction_mean / deg, dist_mean / deg], dim=1)

    data = Data(edge_index=edge_index)
    data.seed_mask = seed_mask
    data.hierarchy_mask = hierarchy_mask
    data.direction_features = direction_features
    return data

