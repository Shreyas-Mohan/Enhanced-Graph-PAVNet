# Comprehensive Benchmark Results Across All Three Clinical Datasets

All models were evaluated under identical zero-patient-leakage protocols across all 7 standardized metrics:
- **Accuracy (%)**
- **Precision (%)**
- **Recall (%)**
- **Specificity (%)**
- **Dice Similarity Coefficient (DSC)**
- **Mean Intersection over Union (mIoU)**
- **95th-Percentile Hausdorff Distance (HD95, in physical mm)**

All models were trained for 50 epochs.

> Complete mentor report document (`.docx`) containing these benchmarks alongside the canonical baselines table and qualitative figures is located at [`docs/reports/Pulmonary_Artery_Vein_Separation_Report.docx`](reports/Pulmonary_Artery_Vein_Separation_Report.docx).

---

## 1. AirRC Thoracic CT Benchmark (Held-Out Test Split: 39 Scans)

- **Dataset Citation**: Liu, J., Zhang, Z., Niu, B. et al. *A Custom Annotated Dataset for Segmentation of Pulmonary Veins, Arteries, and Airways.* [Nature Scientific Data 12, 1806 (2025)](https://doi.org/10.1038/s41597-025-06074-6).
- **Cohort**: 254 thoracic CT scans (177 train / 38 val / 39 test), resampled to isotropic 1.0 mm grid (1 voxel = 1.0 physical mm).

| Model Architecture | Paradigm / Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D V-Net** | MONAI Residual CNN | 55.23% | 55.10% | 54.22% | 54.22% | 0.5263 | 0.3652 | 15.45 |
| **3D UNETR** | MONAI ViT | 53.94% | 53.85% | 53.80% | 53.80% | 0.5358 | 0.3672 | 13.80 |
| **3D U-Net** | MONAI Dense CNN | 56.85% | 56.69% | 56.41% | 56.41% | 0.5610 | 0.3923 | 13.38 |
| **3D Swin UNETR** | MONAI Swin ViT | 61.51% | 61.78% | 60.95% | 60.95% | 0.6051 | 0.4374 | 13.63 |
| **3D nnU-Net** | Self-Configuring DynUNet | 64.98% | 66.22% | 64.15% | 64.15% | 0.6335 | 0.4697 | 14.53 |
| **Standard GAT** | PyG GNN (584d) | 65.23% | 66.02% | 64.94% | 64.94% | 0.6443 | 0.4787 | 14.52 |
| **Standard GraphSAGE** | PyG GNN (584d) | 65.78% | 66.64% | 65.59% | 65.59% | 0.6506 | 0.4853 | 14.00 |
| **Graph-PAVNet (Baseline)** | Baseline HGAT (584d) | 74.17% | 74.65% | 74.02% | 74.02% | 0.7388 | 0.5881 | 12.80 |
| **Extended-HGAT (Best Features)** | Multimodal (602d) | 77.23% | 77.71% | 76.99% | 76.99% | 0.7693 | 0.6283 | 12.57 |
| **Enhanced HGAT (Proposed)** | Decoupled GNN (602d) | **77.99%** | **78.24%** | **77.85%** | **77.85%** | **0.7780** | **0.6400** | **12.05** |

*Class Breakdown for Enhanced HGAT*: Artery Dice = 0.7937, Vein Dice = 0.7641 | Patient-wise mean HD95 = 12.05 mm (Pooled point-cloud HD95 = 3.87 mm).

---

## 2. HiPaS 2025 Thoracic CT Benchmark (Held-Out Test Split: 37 Scans)

- **Dataset Citation**: Chu, Y., Luo, G., Zhou, L. et al. *Deep learning-driven pulmonary artery and vein segmentation reveals demography-associated vasculature anatomical differences.* [Nature Communications 16, 2262 (2025)](https://doi.org/10.1038/s41467-025-56505-6).
- **Cohort**: 250 scans (175 train / 38 val / 37 test), compressed NPZ arrays with dynamic physical millimeter spacing.

### 2.1 Dense Volumetric Baselines (Voxel Space, Full Lung CT)
*Evaluated across all CT voxels via 3D sliding window (64x64x64). Voxel accuracy reflects background non-vascular lung parenchyma (>98% of thoracic volume).*

| Model Architecture | Category / Representation | Voxel Acc | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D Swin UNETR** | Shifted-Window ViT (64x64x64) | 98.39% | 20.85% | 75.56% | 98.50% | 0.3229 | 0.1940 | 98.06 |
| **3D UNETR** | Vision Transformer (64x64x64) | 98.79% | 25.00% | 69.03% | 98.94% | 0.3623 | 0.2230 | 90.89 |
| **3D U-Net** | MONAI CNN (64x64x64) | 99.46% | 47.07% | 69.34% | 99.62% | 0.5551 | 0.3861 | 94.34 |
| **3D V-Net** | Residual CNN (64x64x64) | 99.64% | 60.88% | 79.50% | 99.75% | 0.6838 | 0.5214 | 73.22 |
| **3D nnU-Net** | DynUNet (64x64x64) | 99.74% | 70.88% | 82.14% | 99.83% | 0.7573 | 0.6114 | 58.98 |

*Note on 3D Swin UNETR*: At vascular skeleton landmark points, Swin UNETR achieves Point Acc: 55.15%, Point Dice: 0.4132, and Point HD95: 38.30 mm.

### 2.2 Graph Neural Network Baselines & Proposed (Landmark Skeleton Graph, 8,192 Nodes)
*Evaluated directly on the 8,192 landmark skeleton nodes (binary pulmonary artery vs. pulmonary vein classification).*

| Model Architecture | Category / Representation | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | HGAT Baseline (584d) | 77.39% | 77.74% | 76.21% | 76.21% | 0.7600 | 0.6243 | 16.48 |
| **Standard GraphSAGE** | PyG Inductive (584d) | 78.13% | 78.32% | 77.05% | 77.05% | 0.7692 | 0.6372 | 15.89 |
| **Standard GAT** | PyG Attention (584d) | 78.43% | 79.13% | 77.15% | 77.15% | 0.7707 | 0.6397 | 16.20 |
| **Extended-HGAT (Best Features)**| Multimodal (602d) | 82.35% | 82.13% | 81.84% | 81.84% | 0.8184 | 0.7045 | 11.96 |
| **Enhanced HGAT (Proposed)** | Decoupled GNN (602d) | **82.60%** | **82.46%** | **82.10%** | **82.10%** | **0.8212** | **0.7089** | **11.74** |

*HiPaS 2025 Class-Specific Breakdown*:
- Extended-HGAT: Artery DSC: 0.8219, Vein DSC: 0.8149 | Artery HD95: 11.47 mm, Vein HD95: 12.45 mm
- Enhanced HGAT: Artery DSC: 0.8244, Vein DSC: 0.8180 | Artery HD95: 11.32 mm, Vein HD95: 12.17 mm

---

## 3. Lung250M-4B Thoracic CT Benchmark (Held-Out Test Split: 30 Paired Scans)

- **Dataset Citation**: Falta, F., Großbröhmer, C., Hering, A., Bigalke, A., Heinrich, M. *Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration.* [NeurIPS 2023 Datasets and Benchmarks Track](https://doi.org/10.52202/075280-2392) ([OpenReview](https://openreview.net/forum?id=j3E3mSsm8w)).
- **Cohort**: 248 scans from 124 patients, paired inspiratory (`_1`) and expiratory (`_2`) acquisitions. Group split: 138 train (69 pairs) / 36 val (18 pairs) / 30 test (15 pairs).
- **Dual Evaluation Regimes**: Models are benchmarked under two standardized evaluation protocols:
  1. **Table 3.1: True Dense 3D Volumetric Master Benchmark**: Evaluated across every voxel in the complete 3D CT volume against dense multi-class ground truth (`dataset/segTs/`).
  2. **Table 3.2: Landmark Centerline Point-Cloud Benchmark**: Evaluated directly on the 8,192 medial-axis vascular skeleton landmark nodes per scan.

### 3.1 True Dense 3D Volumetric Master Benchmark (Held-Out Test Split: 30 Scans)
*Evaluation over all 30 unseen test scans in full volumetric 3D voxel space against dense expert ground truth volumes (`dataset/segTs/`). Captures complete vascular arborization, companion vessel boundaries, and non-vessel lung parenchymal background.*

| Model Architecture | Paradigm / Representation | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D Swin UNETR** | Shifted-Window ViT (64x64x64) | 95.77% | 4.28% | 29.68% | 96.08% | 0.0730 | 0.0380 | 67.34 |
| **3D U-Net** | Dense Volumetric CNN (64x64x64) | 99.52% | 41.86% | 34.48% | 99.79% | 0.3702 | 0.2288 | 47.25 |
| **3D UNETR** | Vision Transformer (64x64x64) | 99.53% | 45.88% | 48.36% | 99.74% | 0.4629 | 0.3042 | 56.62 |
| **3D V-Net** | Residual Volumetric CNN (64x64x64) | 99.53% | 47.78% | 57.19% | 99.71% | 0.5116 | 0.3464 | 32.86 |
| **3D nnU-Net** | Self-Configuring DynUNet | 99.65% | 59.63% | 64.68% | 99.80% | 0.6130 | 0.4456 | 19.09 |
| **Standard GAT** | Graph Attention (584d) | 99.68% | 67.00% | 65.47% | 99.81% | 0.6537 | 0.4885 | 12.88 |
| **Standard GraphSAGE** | Inductive GNN (584d) | 99.69% | 68.21% | 66.68% | 99.82% | 0.6659 | 0.5019 | 12.47 |
| **Graph-PAVNet (Baseline)** | Paper HGAT Baseline (584d) | 99.71% | 70.04% | 68.42% | 99.84% | 0.6844 | 0.5228 | 12.95 |
| **Extended-HGAT (Best Features)**| Multimodal GNN (602d) | **99.78%** | **74.47%** | **74.20%** | **99.89%** | **0.7397** | **0.5889** | **11.39** |
| **Enhanced HGAT (Proposed)** | Decoupled Cross-Gating (602d) | **99.78%** | 74.21% | 73.91% | **99.89%** | 0.7368 | 0.5851 | 11.45 |

*Enhanced HGAT Class-Specific Foreground Breakdown*:
- Artery DSC: 0.7538 | Artery Precision: 74.26% | Artery Recall: 76.54% | Artery mIoU: 0.6049 | Artery HD95: 11.21 mm
- Vein DSC: 0.7233 | Vein Precision: 73.59% | Vein Recall: 71.12% | Vein mIoU: 0.5666 | Vein HD95: 11.69 mm
- Volumetric Lead over nnU-Net: +12.38% to +12.67% higher Dice score (0.7368 / 0.7397 vs. 0.6130), +14.58% higher precision (74.21% vs. 59.63%), and a 40.0% reduction in surface boundary distance error (11.45 mm vs. 19.09 mm).

### 3.2 Landmark Centerline Point-Cloud Benchmark (8,192 Nodes / Scan, 30 Test Scans)
*Evaluated directly on the 8,192 landmark skeleton nodes.*

| Model Architecture | Model Category / Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D Swin UNETR** | MONAI Swin ViT | 37.86% | 37.60% | 37.68% | 37.68% | 0.3757 | 0.2335 | 15.14 |
| **3D U-Net** | MONAI Dense CNN | 55.33% | 55.12% | 55.10% | 55.10% | 0.5492 | 0.3802 | 14.19 |
| **3D UNETR** | MONAI Vision Transformer | 61.67% | 61.90% | 61.56% | 61.56% | 0.6103 | 0.4418 | 13.93 |
| **3D V-Net** | MONAI Residual CNN | 64.08% | 64.53% | 63.74% | 63.74% | 0.6305 | 0.4647 | 13.83 |
| **Standard GAT** | Official PyG (584d) | 65.25% | 65.46% | 65.04% | 65.04% | 0.6472 | 0.4807 | 13.96 |
| **Standard GraphSAGE** | Official PyG (584d) | 66.27% | 66.62% | 66.08% | 66.08% | 0.6571 | 0.4917 | 13.80 |
| **Graph-PAVNet (Baseline)** | HGAT Baseline (584d) | 68.73% | 69.13% | 68.96% | 68.96% | 0.6851 | 0.5227 | 14.19 |
| **nnU-Net (DynUNet)** | MONAI DynUNet | 74.22% | 74.54% | 73.84% | 73.84% | 0.7356 | 0.5867 | 12.91 |
| **Enhanced HGAT (Proposed)** | Decoupled GNN (602d) | 73.95% | 74.07% | 74.03% | 74.03% | 0.7373 | 0.5863 | **13.05** |
| **Extended-HGAT (Best Features)**| Multimodal (602d) | **74.33%** | **74.42%** | **74.45%** | **74.45%** | **0.7414** | **0.5915** | 13.11 |

---

## 4. Multi-Objective Loss Sensitivity & Optimization Stability Analysis

The objective function for Enhanced HGAT balances five complementary loss components:
1. **Focal Loss**: Core class-balanced loss focusing on hard peripheral vessel nodes.
2. **Seed Supervision**: High-confidence root supervision near cardiac hilum entry points (weight: 0.05).
3. **clDice Graph Continuity**: Centerline topological smoothness constraint along vascular branches (weight: 0.10).
4. **Caliber Regularization**: Continuous lumen diameter radius regression maintaining proximal-to-distal vessel tapering (weight: 0.05).
5. **Crossover Penalty**: False-positive penalty preventing label switching across parallel companion artery-vein pairs (weight: 0.05).

### Table 4.1: Hyperparameter Perturbation Matrix (AirRC Validation Split)

To evaluate optimization stability, weights were perturbed by ±50% around the default configuration:

| Configuration / Perturbation | Seed Weight | clDice Weight | Caliber Weight | Crossover Weight | Val Accuracy | Val Dice (DSC) | Change in Dice | Optimization Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Operating Point** | **0.05** | **0.10** | **0.05** | **0.05** | **78.12%** | **0.7794** | **Baseline** | **Stable (Optimal)** |
| Low clDice (0.5x) | 0.05 | 0.05 | 0.05 | 0.05 | 77.89% | 0.7768 | -0.26% | Stable |
| High clDice (1.5x) | 0.05 | 0.15 | 0.05 | 0.05 | 78.06% | 0.7788 | -0.06% | Stable |
| Extreme clDice (2.0x) | 0.05 | 0.20 | 0.05 | 0.05 | 77.94% | 0.7772 | -0.22% | Stable |
| Low Caliber Reg (0.5x) | 0.05 | 0.10 | 0.025 | 0.05 | 78.01% | 0.7781 | -0.13% | Stable |
| High Caliber Reg (1.5x) | 0.05 | 0.10 | 0.075 | 0.05 | 77.98% | 0.7780 | -0.14% | Stable |
| Low Crossover Penalty (0.5x) | 0.05 | 0.10 | 0.05 | 0.025 | 77.95% | 0.7775 | -0.19% | Stable |
| High Crossover Penalty (1.5x) | 0.05 | 0.10 | 0.05 | 0.075 | 78.08% | 0.7790 | -0.04% | Stable |
| Simultaneous Perturbation (±50% mixed) | 0.025 | 0.15 | 0.075 | 0.025 | 77.85% | 0.7762 | -0.32% | Stable |
| Ablate clDice (0.0) | 0.05 | 0.00 | 0.05 | 0.05 | 76.92% | 0.7671 | -1.23% | Sub-optimal |
| Ablate Caliber Reg (0.0) | 0.05 | 0.10 | 0.00 | 0.05 | 77.15% | 0.7698 | -0.96% | Sub-optimal |

*Stability Summary*: Fluctuations of ±50% across any individual loss weight yield < 0.35% change in validation Dice score. The same fixed weight vector (Seed: 0.05, clDice: 0.10, Caliber: 0.05, Crossover: 0.05) generalizes stably across AirRC, HiPaS, and Lung250M without per-dataset tuning.
