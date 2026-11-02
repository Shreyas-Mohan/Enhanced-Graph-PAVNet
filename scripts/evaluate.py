#!/usr/bin/env python
"""
Benchmark Evaluation Script for Pulmonary Artery-Vein Separation.
Evaluates trained models on held-out test splits across all 7 standardized metrics:
  - Accuracy (%)
  - Precision (%)
  - Recall (%)
  - Specificity (%)
  - Dice Similarity Coefficient (DSC)
  - Mean Intersection over Union (mIoU)
  - 95th-Percentile Hausdorff Distance (HD95, in physical mm)
Supports per-case patient-wise averaging (standard clinical evaluation protocol)
and post-processing graph smoothing.
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from torch_geometric.loader import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EnhancedHGATNet, ExtendedHGATNet, BaselineHGATNet, StandardGNNBaseline
from src.evaluation import evaluate_vascular_predictions, smooth_graph_predictions_majority_vote
from data.loaders import AirRCGraphDataset, HiPaSGraphDataset, Lung250MGraphDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on held-out test split.")
    parser.add_argument("--config", type=str, default="configs/airrc.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, choices=["airrc", "hipas", "lung250m"], default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument(
        "--model",
        type=str,
        default="enhanced_hgat",
        choices=["enhanced_hgat", "extended_hgat", "baseline_hgat", "gat", "sage"]
    )
    parser.add_argument("--threshold", type=float, default=None, help="Calibrated decision threshold (default: argmax / 0.50)")
    parser.add_argument("--smooth", action="store_true", help="Enable post-processing majority vote graph smoothing")
    parser.add_argument("--no-smooth", action="store_true", help="Disable graph majority voting smoothing (default)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = args.dataset or cfg["dataset"]["name"].lower().replace(" ", "").replace("-", "")
    if "lung" in dataset_name:
        dataset_name = "lung250m"
    elif "hipas" in dataset_name:
        dataset_name = "hipas"
    else:
        dataset_name = "airrc"

    feature_dim = cfg["dataset"].get("feature_dim", 602 if args.model in ["enhanced_hgat", "extended_hgat"] else 584)
    threshold = args.threshold if args.threshold is not None else cfg.get("evaluation", {}).get("decision_threshold", 0.50)

    # Initialize Dataset & DataLoader
    if dataset_name == "airrc":
        test_set = AirRCGraphDataset(split="test", feature_dim=feature_dim)
        spacing = (1.0, 1.0, 1.0)
    elif dataset_name == "hipas":
        test_set = HiPaSGraphDataset(split="test", feature_dim=feature_dim)
        spacing = (0.7, 0.7, 1.0)
    elif dataset_name == "lung250m":
        test_set = Lung250MGraphDataset(split="test", feature_dim=feature_dim)
        spacing = (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(f"[+] Loaded {len(test_set)} held-out test scans from {dataset_name.upper()}.")
    print(f"[+] Model: {args.model.upper()} | Threshold: {threshold:.2f} | Post-smoothing: {not args.no_smooth}")

    # Load Model
    if args.model == "enhanced_hgat":
        model = EnhancedHGATNet(in_channels=feature_dim, out_channels=2, max_hops=3, aux_caliber=True)
    elif args.model == "extended_hgat":
        model = ExtendedHGATNet(in_channels=feature_dim, out_channels=2, max_hops=3)
    elif args.model == "baseline_hgat":
        model = BaselineHGATNet(in_channels=584, out_channels=2, max_hops=3)
    elif args.model in ["gat", "sage"]:
        model = StandardGNNBaseline(in_channels=584, hidden_channels=128, out_channels=2, model_type=args.model)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    case_metrics = []
    pooled_preds, pooled_targets = [], []
    all_coords = []

    print("[+] Running patient-wise evaluation across held-out test split...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]

            if threshold is not None and threshold != 0.5:
                prob_vein = torch.exp(out[:, 1])
                preds = (prob_vein > threshold).long()
            else:
                preds = out.argmax(dim=-1)

            if args.smooth and not args.no_smooth:
                preds = smooth_graph_predictions_majority_vote(preds, batch.edge_index, rounds=2)

            preds_np = preds.cpu().numpy()
            targets_np = batch.y.cpu().numpy()
            coords_np = batch.original_coords.cpu().numpy()

            # Patient-wise metrics
            m = evaluate_vascular_predictions(targets_np, preds_np, coords=coords_np, pixel_spacing=spacing)
            case_metrics.append(m)

            pooled_preds.extend(preds_np)
            pooled_targets.extend(targets_np)
            all_coords.append(coords_np)

    # Compute Patient-Wise Mean & Std across all test scans
    metric_keys = ["accuracy", "precision", "recall", "specificity", "dice", "miou", "hd95"]
    mean_res = {}
    std_res = {}
    for k in metric_keys:
        vals = [c[k] for c in case_metrics if c[k] is not None]
        mean_res[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        std_res[k] = float(np.std(vals)) if len(vals) > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS: {args.model.upper()} on {dataset_name.upper()} ({len(case_metrics)} Test Scans)")
    print("=" * 90)
    print(f"Accuracy:    {mean_res['accuracy']:5.2f}% ± {std_res['accuracy']:4.2f}%")
    print(f"Precision:   {mean_res['precision']:5.2f}% ± {std_res['precision']:4.2f}%")
    print(f"Recall:      {mean_res['recall']:5.2f}% ± {std_res['recall']:4.2f}%")
    print(f"Specificity: {mean_res['specificity']:5.2f}% ± {std_res['specificity']:4.2f}%")
    print(f"Dice (DSC):  {mean_res['dice']:6.4f} ± {std_res['dice']:5.4f}")
    print(f"mIoU:        {mean_res['miou']:6.4f} ± {std_res['miou']:5.4f}")
    if mean_res['hd95'] > 0:
        print(f"HD95:        {mean_res['hd95']:5.2f} mm ± {std_res['hd95']:4.2f} mm")
    print("=" * 90)


if __name__ == "__main__":
    main()
