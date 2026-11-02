import torch
import numpy as np


def smooth_graph_predictions_majority_vote(pred_labels, edge_index, num_classes=2, rounds=2):
    """
    Vectorized majority voting graph smoothing across k-NN neighbors to reduce isolated misclassifications.
    """
    N = pred_labels.size(0)
    device = pred_labels.device
    smoothed = pred_labels.clone()
    src, dst = edge_index[0], edge_index[1]

    for _ in range(rounds):
        one_hot = torch.zeros((N, num_classes), device=device)
        one_hot.scatter_(1, smoothed.unsqueeze(1), 1.0)

        neighbor_votes = torch.zeros((N, num_classes), device=device)
        neighbor_votes.index_add_(0, src, one_hot[dst])
        smoothed = neighbor_votes.argmax(dim=1)

    return smoothed


def map_skeleton_to_dense_volume(coords, pred_labels, radii, volume_shape):
    """
    Splat skeleton predictions back into a dense 3D volume array (0: BG, 1: Artery, 2: Vein).
    """
    vol = np.zeros(volume_shape, dtype=np.uint8)
    coords_np = coords.long().cpu().numpy()
    labels_np = pred_labels.cpu().numpy() + 1 # 1: Artery, 2: Vein

    X, Y, Z = volume_shape
    ix = np.clip(coords_np[:, 0], 0, X - 1)
    iy = np.clip(coords_np[:, 1], 0, Y - 1)
    iz = np.clip(coords_np[:, 2], 0, Z - 1)

    vol[ix, iy, iz] = labels_np
    return vol

