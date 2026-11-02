# Baseline Architectures & Benchmark Configurations

This directory contains configuration files for the canonical deep learning and graph neural network baselines benchmarked against **Enhanced HGAT** across the three pulmonary vascular datasets (AirRC, HiPaS 2025, and Lung250M-4B).

---

## 1. Dense Volumetric 3D Baselines (Project MONAI)

All dense volumetric architectures are instantiated via **Project MONAI** (`monai.networks.nets`), adhering strictly to standard benchmark implementations in the medical image segmentation literature:

| Model | Config File | MONAI Class | Citation | Key Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **3D U-Net** | [`unet3d.yaml`](unet3d.yaml) | `monai.networks.nets.UNet` | Çiçek et al. (*MICCAI 2016*) | 5 stages, channels: [16, 32, 64, 128, 256], strides: 2, InstanceNorm |
| **V-Net** | [`vnet3d.yaml`](vnet3d.yaml) | `monai.networks.nets.VNet` | Milletari et al. (*3DV 2016*) | Residual volumetric blocks, ELU activation, dropout: 0.1 |
| **nnU-Net** | [`nnunet3d.yaml`](nnunet3d.yaml) | `monai.networks.nets.DynUNet` | Isensee et al. (*Nature Methods 2021*) | Self-configuring dynamic U-Net, LeakyReLU, deep supervision ready |
| **UNETR** | [`unetr3d.yaml`](unetr3d.yaml) | `monai.networks.nets.UNETR` | Hatamizadeh et al. (*WACV 2022*) | ViT transformer backbone (12 heads, hidden: 768, MLP: 3072), deconv decoders |
| **Swin UNETR** | [`swin_unetr.yaml`](swin_unetr.yaml) | `monai.networks.nets.SwinUNETR` | Tang et al. (*MICCAI BrainLes 2021*) | Hierarchical shifted-window transformer, feature size: 48, checkpointing |

### Volumetric Training Details
- **Loss Function**: Combined Dice and Cross-Entropy Loss (`DiceCELoss(to_onehot_y=True, softmax=True)`).
- **Patch Extraction**: $64 \times 64 \times 64$ voxel sliding windows sampled centered around pulmonary vessel segments.
- **Optimization**: AdamW optimizer with initial learning rate $\eta \in [10^{-4}, 3 \times 10^{-4}]$, cosine annealing scheduler, 50 epochs on NVIDIA GPU (`--device cuda:0`).
- **Factory Implementation**: See [`src/models/volumetric_baselines.py`](../../src/models/volumetric_baselines.py).

---

## 2. Graph Neural Network (GNN) Baselines

All graph neural network baselines operate on the extracted pulmonary vascular skeleton graphs:

| Model | Config File | Implementation Source | Citation | Key Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **GAT** | [`gnn_baselines.yaml`](gnn_baselines.yaml) | PyTorch Geometric (`torch_geometric.nn.GATConv`) | Veličković et al. (*ICLR 2018*) | 3 layers, 4 attention heads, hidden dim: 128, dropout: 0.1 |
| **GraphSAGE** | [`gnn_baselines.yaml`](gnn_baselines.yaml) | PyTorch Geometric (`torch_geometric.nn.SAGEConv`) | Hamilton et al. (*NeurIPS 2017*) | 3 layers, mean neighborhood aggregator, hidden dim: 128 |
| **Graph-PAVNet HGAT** | [`gnn_baselines.yaml`](gnn_baselines.yaml) | Custom reimplementation of Li et al. | Li et al. (*GitHub: Lqy1018/Graph-PAVNet*) | 3-hop multi-hop neighborhood aggregation, 584d visual-patch inputs |

### Graph Training Details
- **Loss Function**: Focal Dice + Cross-Entropy loss.
- **Optimization**: AdamW, learning rate $\eta = 4 \times 10^{-4}$, weight decay $10^{-4}$, 50 epochs on NVIDIA GPU (`--device cuda:0`).
- **Factory Implementation**: See [`src/models/gnn_baselines.py`](../../src/models/gnn_baselines.py) and [`src/models/baseline_hgat.py`](../../src/models/baseline_hgat.py).

---

## 3. Graph-PAVNet Baseline: Origin, Architecture & Setup Guide

### 3.1 Origin & Literature Reference
- **Authors & Affiliation**: Qingya Li, Ye Yuan, Lu Liu, Nan Bao, Lisheng Xu, Wenjun Tan. Key Research Laboratory of Intelligent Computing of Medical Images, Northeastern University, Shenyang, China.
- **Paper**: *Graph-PAVNet: A Graph-Based Learning Framework for Pulmonary Artery and Vein Separation Using Multimodal Feature Sampling*.
- **Upstream Repository**: [https://github.com/Lqy1018/Graph-PAVNet](https://github.com/Lqy1018/Graph-PAVNet)

### 3.2 Canonical Architecture (584-dim Input)
Graph-PAVNet constructs a pulmonary vascular skeleton graph and represents each centerline node using 584 features:
1. **Visual ViT Tokens ($576\text{d}$)**: Extracted from a $16 \times 16 \times 16$ voxel CT patch centered at each node using a frozen 3D Vision Transformer encoder.
2. **Spatial & Geometric Features ($8\text{d}$)**: Normalized Cartesian coordinates $(x, y, z)$, vessel radius $r$, parent branch geodesic distance, and 3D directional vectors $(\Delta x, \Delta y, \Delta z)$.
3. **Hierarchical Graph Attention Network (HGAT)**: Aggregates information across 3 hierarchy hops from the hilum seed nodes using attention coefficients and residual gating.

### 3.3 How to Set Up & Run Graph-PAVNet

#### Option A: Unified PyG Implementation (Included in this Repository)
We provide a unified PyTorch Geometric reimplementation in [`src/models/baseline_hgat.py`](../../src/models/baseline_hgat.py) that integrates seamlessly with all data loaders and evaluation scripts in this codebase.

**Train on NVIDIA GPU (50 epochs):**
```bash
# AirRC
python scripts/train.py --config configs/airrc.yaml --model baseline_hgat --epochs 50 --device cuda:0

# HiPaS 2025
python scripts/train.py --config configs/hipas.yaml --model baseline_hgat --epochs 50 --device cuda:0

# Lung250M-4B
python scripts/train.py --config configs/lung250m.yaml --model baseline_hgat --epochs 50 --device cuda:0
```

**Evaluate on held-out test split:**
```bash
python scripts/evaluate.py --config configs/airrc.yaml --dataset airrc --model baseline_hgat --checkpoint checkpoints/airrc_baseline_hgat_584.pth
```

#### Option B: Standalone Upstream Repository
If you wish to benchmark against the original upstream codebase directly:
```bash
git clone https://github.com/Lqy1018/Graph-PAVNet.git
cd Graph-PAVNet
pip install -r requirements.txt
```
The data formatting scripts in `data/loaders/` produce skeleton graphs fully compatible with both our unified reimplementation and the upstream repository format.
