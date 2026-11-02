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


def build_skeleton_graph(coords, radii, k=10, max_hops=3):
    """
    Builds a 3D spatial k-NN graph from vascular skeleton landmark nodes.
    Extracts seed hilum nodes (top 2% by radius), computes BFS hierarchy mask and direction features.
    """
    N = coords.size(0)
    coords_np = coords.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords_np)
    distances, indices = nbrs.kneighbors(coords_np)

    src = torch.tensor(indices[:, 0].repeat(k), dtype=torch.long)
    dst = torch.tensor(indices[:, 1:].flatten(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # Hilum seeds: largest 2% of vessels by caliber radius
    k_seeds = max(1, int(0.02 * N))
    seed_idx = torch.topk(radii, k=k_seeds)[1]
    seed_mask = torch.zeros(N, dtype=torch.bool)
    seed_mask[seed_idx] = True

    # BFS Hierarchy mask
    layers = layerwise_bfs_propagation(edge_index, seed_mask)
    hierarchy_mask = torch.zeros((N, max_hops + 1), dtype=torch.float32)
    for hop, layer in enumerate(layers):
        if hop > max_hops:
            break
        hierarchy_mask[layer, hop] = 1.0

    # Direction features from k-NN neighbors
    diffs = coords[dst] - coords[src]
    dist = torch.norm(diffs, dim=1, keepdim=True).clamp_min(1e-6)
    unit_diffs = diffs / dist

    direction_mean = torch.zeros(N, 3)
    direction_mean.index_add_(0, src, unit_diffs)
    direction_mean = direction_mean / float(k)

    dist_mean = torch.zeros(N, 1)
    dist_mean.index_add_(0, src, dist)
    dist_mean = dist_mean / float(k)

    direction_features = torch.cat([direction_mean, dist_mean], dim=1)

    data = Data(edge_index=edge_index)
    data.seed_mask = seed_mask
    data.hierarchy_mask = hierarchy_mask
    data.direction_features = direction_features
    return data

