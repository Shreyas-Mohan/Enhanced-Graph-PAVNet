import torch


def compute_graph_cldice_loss(probs, targets, edge_index, radii):
    """
    Graph Centerline Path Continuity Loss (adapted from clDice, Shit et al., CVPR 2021).
    Enforces cosine probability continuity along skeleton edges, weighted by local caliber:
      w_uv = min(r_u, r_v).
    """
    src, dst = edge_index[0], edge_index[1]
    r_clamped = torch.clamp(radii, min=0.5, max=5.0)

    # Edge caliber weight
    w_uv = torch.min(r_clamped[src], r_clamped[dst])
    w_uv = w_uv / w_uv.mean().clamp_min(1e-6)

    # Cosine distance between neighboring probability vectors
    cos_sim = torch.sum(probs[src] * probs[dst], dim=1)
    cos_dist = 1.0 - cos_sim

    # Supervise primarily along same-class skeleton branches
    same_class_mask = (targets[src] == targets[dst]).float()
    return (w_uv * cos_dist * same_class_mask).sum() / (same_class_mask.sum().clamp_min(1.0))

