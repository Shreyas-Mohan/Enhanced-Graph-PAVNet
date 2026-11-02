import torch
import torch.nn.functional as F


def compute_radius_weighted_focal_loss(predictions, targets, radii, class_weights, gamma=2.0):
    """
    Radius-Weighted Focal Loss targeting thin distal vasculature (BV5 clinical focus).
    Applies inverse square-root weighting w_i = 1 / sqrt(r_i) to prioritize small vessels.
    """
    device = predictions.device
    probs = torch.exp(predictions)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0 - 1e-7)
    log_p_t = predictions.gather(1, targets.unsqueeze(1)).squeeze(1)

    r_clamped = torch.clamp(radii, min=0.5, max=5.0)
    w_radius = 1.0 / torch.sqrt(r_clamped)
    w_radius = w_radius / w_radius.mean().clamp_min(1e-6)

    cw = class_weights.to(device)
    alpha_t = cw[targets]
    focal_weight = torch.pow(1.0 - p_t, gamma)

    node_loss = - w_radius * alpha_t * focal_weight * log_p_t
    return node_loss.mean()


def compute_seed_anchoring_loss(predictions, targets, seed_mask, class_weights, gamma=2.0):
    """Anchors topological identity at the cardiac hilum entrance."""
    if not seed_mask.any():
        return torch.tensor(0.0, device=predictions.device)

    probs = torch.exp(predictions)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0 - 1e-7)
    log_p_t = predictions.gather(1, targets.unsqueeze(1)).squeeze(1)

    cw = class_weights.to(predictions.device)
    alpha_t = cw[targets]

    seed_p_t = p_t[seed_mask]
    seed_log_p_t = log_p_t[seed_mask]
    seed_alpha_t = alpha_t[seed_mask]
    seed_focal = torch.pow(1.0 - seed_p_t, gamma)

    return (- seed_alpha_t * seed_focal * seed_log_p_t).mean()

