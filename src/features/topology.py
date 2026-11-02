import torch
import numpy as np
import networkx as nx


def compute_topological_features(coords, edge_index, seed_mask):
    """
    Computes 5 topological graph features:
      1. Normalized multi-source geodesic hop distance from hilum seeds
      2. Local clustering coefficient
      3. PageRank flow centrality score
      4. Normalized node degree
      5. Descendant hop depth indicator
    """
    N = coords.size(0)
    device = coords.device

    G = nx.Graph()
    G.add_nodes_from(range(N))
    edges_np = edge_index.t().cpu().numpy()
    G.add_edges_from(edges_np)

    seed_indices = seed_mask.nonzero().squeeze().cpu().tolist()
    if isinstance(seed_indices, int):
        seed_indices = [seed_indices]
    elif len(seed_indices) == 0:
        seed_indices = [0]

    shortest_hops = np.full(N, 20.0, dtype=np.float32)
    for s in seed_indices:
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=20)
        for target, length in lengths.items():
            if length < shortest_hops[target]:
                shortest_hops[target] = length
    shortest_hops_norm = np.clip(shortest_hops, 0, 20) / 20.0

    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict[i] for i in range(N)], dtype=np.float32)[:, None]

    try:
        pagerank_dict = nx.pagerank(G, max_iter=40, tol=1e-4)
        pagerank = np.array([pagerank_dict[i] for i in range(N)], dtype=np.float32)[:, None] * 1000.0
    except Exception:
        pagerank = np.zeros((N, 1), dtype=np.float32)

    degree = np.array([deg for _, deg in G.degree()], dtype=np.float32)[:, None]
    norm_degree = degree / 10.0
    descendant_proxy = (shortest_hops[:, None] > 5).astype(np.float32)

    topo_features = torch.from_numpy(np.hstack([
        shortest_hops_norm[:, None],
        clustering,
        pagerank,
        norm_degree,
        descendant_proxy,
    ])).float().to(device)

    return topo_features

