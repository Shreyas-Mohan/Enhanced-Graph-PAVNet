import torch
import torch.nn.functional as F
from .focal_dice import compute_radius_weighted_focal_loss, compute_seed_anchoring_loss
from .cldice_graph import compute_graph_cldice_loss


def compute_multi_task_enhanced_loss(
    predictions,
    targets,
    seed_mask,
    edge_index,
    radii,
    class_weights,
    pred_radii=None,
    gamma=2.0,
    lambda_cldice=0.10,
    lambda_thick=0.05,
    lambda_fpr=0.05,
    lambda_seed=0.05
):
    """
    Multi-Objective Loss for Enhanced HGAT:
      L_total = (1 - λ_seed) * L_focal + λ_seed * L_seed
                + λ_cldice * L_cldice + λ_thick * L_thick + λ_fpr * L_fpr
    """
    if isinstance(predictions, (tuple, list)):
        if pred_radii is None and len(predictions) > 1:
            pred_radii = predictions[1]
        predictions = predictions[0]

    device = predictions.device
    probs = torch.exp(predictions)

    # 1. Radius-Weighted Focal Loss
    focal_loss = compute_radius_weighted_focal_loss(predictions, targets, radii, class_weights, gamma=gamma)

    # 2. Hilum Seed Anchoring Loss
    seed_loss = compute_seed_anchoring_loss(predictions, targets, seed_mask, class_weights, gamma=gamma)

    # 3. Centerline clDice Graph Continuity Loss
    cldice_loss = compute_graph_cldice_loss(probs, targets, edge_index, radii)

    # 4. Multi-Task Caliber Regression Loss (Smooth L1)
    if pred_radii is not None:
        target_norm_radii = radii / (radii.max().clamp_min(1e-6))
        pred_norm_radii = pred_radii / (pred_radii.max().clamp_min(1e-6) + 1e-6)
        thick_loss = F.smooth_l1_loss(pred_norm_radii, target_norm_radii)
    else:
        thick_loss = torch.tensor(0.0, device=device)

    # 5. Artery-Vein FPR Crossover Penalty
    is_artery = (targets == 0).float()
    is_vein = (targets == 1).float()
    prob_artery = probs[:, 0]
    prob_vein = probs[:, 1]

    fpr_art_on_vein = (prob_artery * is_vein).sum() / (is_vein.sum().clamp_min(1.0))
    fpr_vein_on_art = (prob_vein * is_artery).sum() / (is_artery.sum().clamp_min(1.0))
    fpr_loss = fpr_art_on_vein + fpr_vein_on_art

    total_loss = (
        (1.0 - lambda_seed) * focal_loss
        + lambda_seed * seed_loss
        + lambda_cldice * cldice_loss
        + lambda_thick * thick_loss
        + lambda_fpr * fpr_loss
    )

    return total_loss, focal_loss.item(), cldice_loss.item()

