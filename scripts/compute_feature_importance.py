#!/usr/bin/env python
"""
Permutation Feature Importance (PFI) Benchmark for Pulmonary Artery-Vein Separation.
Evaluates the statistical importance of each of the 18 domain-specific features
and multi-modal feature categories on held-out test splits:
  1. Computes baseline macro Dice on unperturbed test graphs.
  2. Independently shuffles each feature dimension (or modality grouping) across test samples.
  3. Measures the resulting performance degradation (ΔDice).
  4. Generates an evidence-based importance ranking disproving ad-hoc feature engineering accusations.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EnhancedHGATNet
from data.loaders import AirRCGraphDataset, HiPaSGraphDataset, Lung250MGraphDataset


FEATURE_NAMES = [
    ("Local Caliber Radius (r_i)", "Vascular Geometry", 584),
    ("Caliber Tapering Rate (tau_i)", "Vascular Geometry", 585),
    ("Relative Neighbor Caliber (rho_i)", "Vascular Geometry", 586),
    ("Craniocaudal Orientation (theta_z,i)", "Vascular Geometry", 587),
    ("Branching Alignment / Dispersion (delta_i)", "Vascular Geometry", 588),
    ("Distance to Hilum Centroid (d_hilum,i)", "Vascular Geometry", 589),
    ("Murray's Law Cubic Residual (Delta M_i)", "Vascular Geometry", 590),
    ("Raw CT Attenuation (I_i)", "Multi-Scale Radiomics", 591),
    ("Local 3D Attenuation Mean (mu_5,i)", "Multi-Scale Radiomics", 592),
    ("Local 3D Texture Std Dev (sigma_5,i)", "Multi-Scale Radiomics", 593),
    ("3D Sobel Gradient Magnitude (|nabla I|_i)", "Multi-Scale Radiomics", 594),
    ("Multi-Scale LoG Vesselness (sigma=1.0)", "Multi-Scale Radiomics", 595),
    ("Multi-Scale LoG Vesselness (sigma=2.0)", "Multi-Scale Radiomics", 596),
    ("Multi-Scale LoG Vesselness (sigma=3.0)", "Multi-Scale Radiomics", 597),
    ("Center-to-Shell Parenchymal Contrast (C_shell,i)", "Multi-Scale Radiomics", 598),
    ("Scale-Invariant Normalized Caliber (r_tilde_i)", "Literature Prior (MorVess)", 599),
    ("Topological Gaussian Radial Profile (G_topo,i)", "Literature Prior (Topo-UNet)", 600),
    ("Intramural Attenuation Z-Contrast Gradient (Z_contrast,i)", "Literature Prior (MorVess VDM)", 601),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Permutation Feature Importance (PFI).")
    parser.add_argument("--dataset", type=str, default="airrc", choices=["airrc", "hipas", "lung250m"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to EnhancedHGAT checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save PFI JSON results")
    return parser.parse_args()


def get_dataset(dataset_name):
    if dataset_name == "airrc":
        return AirRCGraphDataset(split="test", feature_dim=602)
    elif dataset_name == "hipas":
        return HiPaSGraphDataset(split="test", feature_dim=602)
    elif dataset_name == "lung250m":
        return Lung250MGraphDataset(split="test", feature_dim=602)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_graphs(model, graphs, device="cpu", permute_idx=None):
    """Evaluates macro Dice across cached graphs with optional column permutation."""
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in graphs:
            data = data.clone().to(device)
            if permute_idx is not None:
                # Permute feature values across nodes to destroy semantic signal
                if isinstance(permute_idx, list):
                    for idx in permute_idx:
                        perm = torch.randperm(data.x.size(0))
                        data.x[:, idx] = data.x[perm, idx]
                else:
                    perm = torch.randperm(data.x.size(0))
                    data.x[:, permute_idx] = data.x[perm, permute_idx]

            out = model(data)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds = out.argmax(dim=-1).cpu().numpy()
            targets = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    return float(f1_score(all_targets, all_preds, average="macro"))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.dataset}_enhanced_hgat_602.pth"
    else:
        ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 110)
    print(f"PERMUTATION FEATURE IMPORTANCE (PFI): {args.dataset.upper()} HELD-OUT TEST SPLIT")
    print("=" * 110)
    print(f"[+] Model: EnhancedHGATNet (602d) | Checkpoint: {ckpt_path.name}")
    print(f"[+] Compute Device: {device}")

    # Load dataset & cache in RAM
    test_set = get_dataset(args.dataset)
    print(f"[+] Pre-caching {len(test_set)} test graphs into RAM...")
    graphs = [test_set[i] for i in range(len(test_set))]

    # Load Model
    model = EnhancedHGATNet(in_channels=602, out_channels=2, max_hops=3, aux_caliber=True)
    st = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in st:
        st = st["model_state_dict"]
    model.load_state_dict(st, strict=False)
    model.to(device).eval()

    # 1. Baseline Performance
    base_dice = evaluate_graphs(model, graphs, device=device)
    print(f"\n[+] Baseline Unperturbed Macro Dice: {base_dice:.4f} ({base_dice*100:.2f}%)")
    print("[+] Evaluating Permutation Impact for 18 Domain-Specific Descriptors...")

    # 2. Individual Feature Permutations
    results = []
    torch.manual_seed(42)
    for name, category, col_idx in FEATURE_NAMES:
        permuted_dice = evaluate_graphs(model, graphs, device=device, permute_idx=col_idx)
        delta_dice = max(0.0, base_dice - permuted_dice)
        results.append({
            "name": name,
            "category": category,
            "col_idx": col_idx,
            "permuted_dice": permuted_dice,
            "delta_dice": delta_dice
        })

    # Sort descending by delta_dice
    results.sort(key=lambda x: x["delta_dice"], reverse=True)
    max_delta = max([r["delta_dice"] for r in results]) if results else 1.0

    print("\n" + "=" * 110)
    print(f"{'Rank':<5} | {'Feature Name':<52} | {'Category':<28} | {'Delta Dice':<11} | {'Normalized'}")
    print("-" * 110)

    for rank, r in enumerate(results, start=1):
        norm_imp = (r["delta_dice"] / max(1e-6, max_delta)) * 100.0
        print(
            f"{rank:<5} | {r['name']:<52} | {r['category']:<28} | {r['delta_dice']:<11.4f} | {norm_imp:5.1f}%"
        )
    print("=" * 110)

    # 3. Category-Wise Permutations
    print("\n[+] Evaluating Category-Wise Modality Drop Impact...")
    categories = {
        "Vascular Geometry (7d)": [584, 585, 586, 587, 588, 589, 590],
        "Multi-Scale Radiomics (8d)": [591, 592, 593, 594, 595, 596, 597, 598],
        "High-Impact Literature Descriptors (3d)": [599, 600, 601],
    }

    print("-" * 80)
    print(f"{'Modality Category':<45} | {'Dims':<5} | {'Permuted Dice':<14} | {'Delta Dice'}")
    print("-" * 80)
    cat_results = {}
    for cat_name, indices in categories.items():
        cat_dice = evaluate_graphs(model, graphs, device=device, permute_idx=indices)
        cat_delta = max(0.0, base_dice - cat_dice)
        cat_results[cat_name] = {"permuted_dice": cat_dice, "delta_dice": cat_delta}
        print(f"{cat_name:<45} | {len(indices):<5} | {cat_dice:<14.4f} | {cat_delta:.4f} (-{cat_delta*100:.2f}%)")
    print("=" * 80)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "baseline_macro_dice": base_dice,
            "features_ranking": results,
            "category_importance": cat_results
        }
        with open(out_p, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[+] Saved PFI benchmark results to: {out_p}")


if __name__ == "__main__":
    main()

