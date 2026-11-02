#!/usr/bin/env python
"""
Unified GPU Training Script for Pulmonary Artery-Vein Separation.
Supports Enhanced HGAT, Baseline HGAT, PyG GNNs (GAT, GraphSAGE), and MONAI volumetric networks.
Default: 50 epochs on NVIDIA GPU with AdamW optimizer and Cosine Annealing learning rate schedule.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EnhancedHGATNet, ExtendedHGATNet, BaselineHGATNet, StandardGNNBaseline, build_volumetric_model
from src.losses import compute_multi_task_enhanced_loss
from data.loaders import AirRCGraphDataset, HiPaSGraphDataset, Lung250MGraphDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train models for Pulmonary Artery-Vein Separation.")
    parser.add_argument("--config", type=str, default="configs/airrc.yaml", help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, choices=["airrc", "hipas", "lung250m"], default=None, help="Dataset name override")
    parser.add_argument("--model", type=str, default="enhanced_hgat", choices=["enhanced_hgat", "extended_hgat", "baseline_hgat", "gat", "sage", "unet3d", "swin_unetr"])
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device (e.g., cuda:0)")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pth", help="Checkpoint save destination")
    return parser.parse_args()


def get_dataset(dataset_name, split, feature_dim=602):
    name = dataset_name.lower()
    if name == "airrc":
        return AirRCGraphDataset(split=split, feature_dim=feature_dim)
    elif name == "hipas":
        return HiPaSGraphDataset(split=split, feature_dim=feature_dim)
    elif name == "lung250m":
        return Lung250MGraphDataset(split=split, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[+] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = args.dataset or cfg["dataset"]["name"].lower().replace(" ", "").replace("-", "")
    if "lung" in dataset_name:
        dataset_name = "lung250m"
    elif "hipas" in dataset_name:
        dataset_name = "hipas"
    else:
        dataset_name = "airrc"

    epochs = args.epochs or cfg.get("training", {}).get("epochs", 50)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 4)
    lr = args.lr or cfg.get("training", {}).get("lr", 1e-3)
    feature_dim = cfg["dataset"].get("feature_dim", 602)

    print(f"[+] Benchmark Dataset: {dataset_name.upper()} | Model: {args.model.upper()} | Epochs: {epochs}")

    # Build DataLoaders
    train_set = get_dataset(dataset_name, split="train", feature_dim=feature_dim)
    val_set = get_dataset(dataset_name, split="val", feature_dim=feature_dim)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[+] Loaded {len(train_set)} training graphs, {len(val_set)} validation graphs.")

    # Initialize Model
    if args.model == "enhanced_hgat":
        model = EnhancedHGATNet(in_channels=feature_dim, out_channels=2, max_hops=3, aux_caliber=True)
    elif args.model == "extended_hgat":
        model = ExtendedHGATNet(in_channels=feature_dim, out_channels=2, max_hops=3)
    elif args.model == "baseline_hgat":
        model = BaselineHGATNet(in_channels=584, out_channels=2, max_hops=3)
    elif args.model in ["gat", "sage"]:
        model = StandardGNNBaseline(in_channels=584, hidden_channels=128, out_channels=2, model_type=args.model)
    elif args.model in ["unet3d", "swin_unetr"]:
        raise NotImplementedError(
            f"Training volumetric models ({args.model}) operates on dense 3D CT patch volumes. "
            "Please use the volumetric dataset loader with configs/baselines/unet3d.yaml or swin_unetr.yaml."
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    class_weights = torch.tensor([1.0, 1.0], device=device)
    best_val_dice = 0.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Starting Training: {epochs} Epochs on {device}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.model == "enhanced_hgat":
                pred_logits, pred_radii = model(batch, return_aux=True)
                loss, _, _ = compute_multi_task_enhanced_loss(
                    predictions=pred_logits,
                    targets=batch.y,
                    seed_mask=batch.seed_mask,
                    edge_index=batch.edge_index,
                    radii=batch.radii,
                    class_weights=class_weights,
                    pred_radii=pred_radii,
                    lambda_thick=0.05,
                    lambda_fpr=0.05
                )
            elif args.model == "extended_hgat":
                pred_logits = model(batch)
                loss, _, _ = compute_multi_task_enhanced_loss(
                    predictions=pred_logits,
                    targets=batch.y,
                    seed_mask=batch.seed_mask,
                    edge_index=batch.edge_index,
                    radii=batch.radii,
                    class_weights=class_weights,
                    pred_radii=None,
                    lambda_thick=0.0,
                    lambda_fpr=0.0
                )
            else:
                pred_logits = model(batch)
                loss = torch.nn.functional.nll_loss(pred_logits, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        scheduler.step()
        epoch_loss = total_loss / max(1, total_nodes)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                val_preds.extend(out.argmax(dim=-1).cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())

        p, r, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average=None, zero_division=0)
        macro_dice = float((f1[0] + f1[1]) / 2.0)
        elapsed = time.time() - t0

        saved = ""
        if macro_dice > best_val_dice:
            best_val_dice = macro_dice
            torch.save(model.state_dict(), str(out_path))
            saved = f"--> Best Checkpoint Saved ({best_val_dice:.4f})"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Loss: {epoch_loss:.4f} | Val Dice: {macro_dice:.4f} (PA: {f1[0]:.4f}, PV: {f1[1]:.4f}) {saved}")

    print("=" * 80)
    print(f"Training Complete. Best Validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoint stored at: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

