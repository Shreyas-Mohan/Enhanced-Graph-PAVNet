#!/usr/bin/env python
"""
Automated Feature Ablation Benchmark Pipeline for Pulmonary Artery-Vein Separation.
Reproduces Table 3 ablation studies across feature modalities:
  1. Baseline Graph-PAVNet (584d)
  2. + Radiomics (592d)
  3. + Vascular Geometry (599d)
  4. All Candidate Features (604d)
  5. Extended-HGAT / Best Features (602d)
  6. Enhanced HGAT / Proposed (602d)
Evaluates trained checkpoints on held-out test splits across all 7 standardized metrics.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EnhancedHGATNet, ExtendedHGATNet, BaselineHGATNet
from src.evaluation import evaluate_vascular_predictions, smooth_graph_predictions_majority_vote
from data.loaders import AirRCGraphDataset, HiPaSGraphDataset, Lung250MGraphDataset

ABLATION_CONFIGS = [
    {
        "key": "baseline_584",
        "name": "Graph-PAVNet (Baseline)",
        "dims": 584,
        "model_cls": BaselineHGATNet,
        "ckpt_pattern": "{dataset}_baseline_hgat_584.pth"
    },
    {
        "key": "rad_592",
        "name": "+ Radiomics",
        "dims": 592,
        "model_cls": ExtendedHGATNet,
        "ckpt_pattern": "best_ablation_radiomics_592.pth"
    },
    {
        "key": "vasc_599",
        "name": "+ Vascular Geometry",
        "dims": 599,
        "model_cls": ExtendedHGATNet,
        "ckpt_pattern": "best_ablation_rad_vascular_599.pth"
    },
    {
        "key": "all_604",
        "name": "All Candidate Features",
        "dims": 604,
        "model_cls": ExtendedHGATNet,
        "ckpt_pattern": "best_ablation_all_604.pth"
    },
    {
        "key": "extended_602",
        "name": "Extended-HGAT (Best Features)",
        "dims": 602,
        "model_cls": ExtendedHGATNet,
        "ckpt_pattern": "{dataset}_extended_hgat_602.pth"
    },
    {
        "key": "enhanced_602",
        "name": "Enhanced HGAT (Proposed)",
        "dims": 602,
        "model_cls": EnhancedHGATNet,
        "ckpt_pattern": "{dataset}_enhanced_hgat_602.pth"
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete feature ablation benchmark.")
    parser.add_argument("--dataset", type=str, default="airrc", choices=["airrc", "hipas", "lung250m"])
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory containing .pth checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_dataset(dataset_name, feature_dim):
    if dataset_name == "airrc":
        return AirRCGraphDataset(split="test", feature_dim=feature_dim), (1.0, 1.0, 1.0)
    elif dataset_name == "hipas":
        return HiPaSGraphDataset(split="test", feature_dim=feature_dim), (0.7, 0.7, 1.0)
    elif dataset_name == "lung250m":
        return Lung250MGraphDataset(split="test", feature_dim=feature_dim), (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoints_dir)

    print("=" * 105)
    print(f"FEATURE ABLATION STUDY BENCHMARK: {args.dataset.upper()} (Held-Out Test Split)")
    print("=" * 105)
    print(f"{'Ablation Stage':<30} | {'Dims':<5} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'Dice (DSC)':<10} | {'HD95 (mm)':<9}")
    print("-" * 105)

    for cfg in ABLATION_CONFIGS:
        fname = cfg["ckpt_pattern"].format(dataset=args.dataset)
        ckpt_path = ckpt_dir / fname

        # If checkpoint exists, evaluate it directly
        if ckpt_path.exists():
            test_set, spacing = get_dataset(args.dataset, cfg["dims"])
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

            st = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in st:
                st = st["model_state_dict"]

            has_adapter = any(k.startswith("adapter.") for k in st.keys())
            has_caliber = any(k.startswith("caliber_head.") for k in st.keys())
            has_cross_gated = any("proj_vis" in k for k in st.keys())

            if has_adapter:
                if has_caliber or has_cross_gated:
                    model = EnhancedHGATNet(in_channels=cfg["dims"], out_channels=2, max_hops=3, aux_caliber=has_caliber)
                else:
                    model = ExtendedHGATNet(in_channels=cfg["dims"], out_channels=2, max_hops=3)
            else:
                model = BaselineHGATNet(in_channels=cfg["dims"], out_channels=2, max_hops=3)

            model.load_state_dict(st, strict=True)
            model.to(device).eval()

            case_metrics = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    preds = out.argmax(dim=-1)
                    m = evaluate_vascular_predictions(
                        batch.y.cpu().numpy(),
                        preds.cpu().numpy(),
                        coords=batch.original_coords.cpu().numpy(),
                        pixel_spacing=spacing
                    )
                    case_metrics.append(m)

            acc = np.mean([c["accuracy"] for c in case_metrics])
            prec = np.mean([c["precision"] for c in case_metrics])
            rec = np.mean([c["recall"] for c in case_metrics])
            dice = np.mean([c["dice"] for c in case_metrics])
            hd95 = np.mean([c["hd95"] for c in case_metrics if c["hd95"] is not None])

            print(f"{cfg['name']:<30} | {cfg['dims']:<5} | {acc:7.2f}%  | {prec:7.2f}%  | {rec:7.2f}%  | {dice:8.4f}   | {hd95:7.2f} mm")
        else:
            # Display checkpoint requirement
            print(f"{cfg['name']:<30} | {cfg['dims']:<5} | (Awaiting checkpoint: {fname})")

    print("=" * 105)


if __name__ == "__main__":
    main()
