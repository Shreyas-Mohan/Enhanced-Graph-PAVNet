# Feature Descriptions & Dataset-Wise Ablation Studies

This document details the 18 optimal domain-specific features, their clinical motivation, category-wise ablation benchmarks, and permutation feature importance rankings.

> **Overview**: To complement patch-level ViT visual embeddings where peripheral microvascular contrast is low, the framework incorporates 18 domain-specific descriptors spanning vascular geometry (e.g. tapering rate, Murray's law), multi-scale radiomics, and literature priors. Out of 23 initially evaluated candidates, 5 global graph metrics sensitive to peripheral mask boundaries were pruned to maximize cross-dataset generalization.

---

## 1. Feature Inventory by Modality

### Category A: Vascular Morphometry (7 Dimensions)
Geometric descriptors capturing lumen cross-section, tree hierarchy, and physical orientation.

1. **Local Vessel Caliber Radius (`r_i`)**:
   Physical Euclidean distance from the skeleton node to the nearest outer vessel wall. Distinguishes high-caliber main trunks from thin peripheral capillaries.

2. **Caliber Tapering Rate (`tau_i`)**:
   Directional rate of change in radius across connected neighboring nodes. Veins maintain relatively uniform caliber over long distances, whereas arteries taper rapidly toward the lung periphery.

3. **Relative Local Caliber Ratio (`rho_i`)**:
   Ratio of a node's radius to the mean radius of its immediate 1-ring neighbors. Identifies local stenosis, bifurcations, and focal caliber changes.

4. **Craniocaudal Angle Orientation (`theta_z`)**:
   Vertical z-component of unit edge vectors. Distinguishes vertically descending basal veins from diagonally fanning arterial trees.

5. **Branching Alignment / Dispersion (`delta_i`)**:
   Euclidean norm of average outward direction unit vectors. Values near 1.0 indicate straight vessel segments; lower values indicate high-curvature bifurcations.

6. **Distance to Cardiac Hilum Centroid (`d_hilum`)**:
   Normalized physical Euclidean distance from the node to the cardiac hilum seed centroid, providing proximal-to-distal positional tree depth.

7. **Murray's Law Cubic Flow Residual (`Delta_M`)**:
   Normalized deviation from Murray's physiological cubic branching law (parent radius cubed equals the sum of daughter radii cubed), capturing physiological flow splits.

---

### Category B: Multi-Scale 3D Radiomics (8 Dimensions)
Intensity distributions and structural second-order texture derivatives computed directly from raw CT volumes.

1. **Raw CT Attenuation (`HU_raw`)**:
   Normalized CT Hounsfield Unit (HU) attenuation scaled to standard range.

2. **Local 3D Mean Attenuation (`mean_5x5`)**:
   Mean attenuation across a 5x5x5 voxel neighborhood surrounding the skeleton node, capturing local blood-pool density.

3. **Local 3D Intensity Standard Deviation (`std_5x5`)**:
   Standard deviation of voxel intensities within a 5x5x5 patch, quantifying local tissue heterogeneity and blood-pool lumen uniformity.

4. **3D Sobel Gradient Magnitude (`grad_sobel`)**:
   Three-dimensional intensity gradient magnitude, capturing sharp contrast transitions at vessel wall boundaries.

5. **Multi-Scale Laplacian of Gaussian (`LoG_fine`, `LoG_med`, `LoG_coarse`)**:
   Three distinct scale-space tubular vesselness responses for fine (sigma = 1.0 mm), medium (sigma = 2.0 mm), and coarse (sigma = 3.0 mm) vessel branches.

6. **Center-to-Parenchymal Shell Contrast (`contrast_shell`)**:
   Difference between intramural core attenuation and surrounding background lung parenchyma, highlighting vessel-to-tissue contrast.

---

### Category C: High-Impact Literature Descriptors (3 Dimensions)
Priors adapted from top-performing vascular segmentation literature.

1. **Scale-Invariant Normalized Caliber (`r_norm`)** *(from MorVess)*:
   Lumen diameter normalized by the 99th-percentile main pulmonary trunk radius, ensuring patient-independent caliber scale invariance.

2. **Topological Gaussian Radial Decay Profile (`G_topo`)** *(from Topo-UNet)*:
   Continuous Gaussian decay potential centered at the main pulmonary trunk, encoding structural distance from cardiac inflow/outflow roots.

3. **Intramural Attenuation Z-Contrast Gradient (`Z_contrast`)** *(from MorVess VDM)*:
   Standardized local attenuation contrast score distinguishing high-pressure contrast-enhanced arteries from venous return.

---

## 2. Dataset-Wise Feature Ablation Tables

### Table 2.1: AirRC Dataset Ablations (39 Test Scans)

| Ablation Stage | Input Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | 584d | 74.17% | 74.65% | 74.02% | 74.02% | 0.7388 | 0.5881 | 12.80 |
| **+ Radiomics** | 592d | 74.48% | 74.84% | 74.49% | 74.49% | 0.7429 | 0.5938 | 12.82 |
| **+ Vascular Geometry** | 599d | 74.81% | 75.15% | 74.62% | 74.62% | 0.7453 | 0.5973 | 12.92 |
| **All Candidate Features** | 604d | 74.38% | 74.64% | 74.39% | 74.39% | 0.7422 | 0.5927 | 12.91 |
| **Extended-HGAT (Best Features)**| 602d | 77.23% | 77.71% | 76.99% | 76.99% | 0.7693 | 0.6283 | 12.57 |
| **Enhanced HGAT (Proposed)** | 602d | **77.99%** | **78.24%** | **77.85%** | **77.85%** | **0.7780** | **0.6400** | **12.05** |

---

### Table 2.2: HiPaS 2025 Dataset Ablations (37 Test Scans)

| Ablation Stage | Input Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | 584d | 77.39% | 77.74% | 76.21% | 76.21% | 0.7600 | 0.6243 | 16.48 |
| **+ Radiomics** | 592d | 79.84% | 79.62% | 79.15% | 79.15% | 0.7938 | 0.6698 | 14.82 |
| **+ Vascular Geometry** | 599d | 81.65% | 81.42% | 81.08% | 81.08% | 0.8124 | 0.6952 | 13.25 |
| **All Candidate Features** | 604d | 81.49% | 81.18% | 81.01% | 81.01% | 0.8096 | 0.6925 | 12.33 |
| **Extended-HGAT (Best Features)**| 602d | 82.35% | 82.13% | 81.84% | 81.84% | 0.8184 | 0.7045 | 11.96 |
| **Enhanced HGAT (Proposed)** | 602d | **82.60%** | **82.46%** | **82.10%** | **82.10%** | **0.8212** | **0.7089** | **11.74** |

---

### Table 2.3: Lung250M-4B Dataset Ablations (30 Paired Test Scans)

| Ablation Stage | Input Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | 584d | 68.73% | 69.13% | 68.96% | 68.96% | 0.6851 | 0.5227 | 14.19 |
| **+ Radiomics** | 592d | 69.14% | 69.51% | 69.00% | 69.00% | 0.6867 | 0.5250 | 14.11 |
| **+ Vascular Geometry** | 599d | 70.34% | 70.44% | 70.19% | 70.19% | 0.6996 | 0.5404 | 14.95 |
| **All Candidate Features (with Graph Topo)**| 604d | 69.28% | 70.23% | 69.88% | 69.88% | 0.6910 | 0.5294 | 15.44 |
| **Extended-HGAT (Best Features)**| 602d | **74.33%** | **74.42%** | **74.45%** | **74.45%** | **0.7414** | **0.5915** | 13.11 |
| **Enhanced HGAT (Proposed)** | 602d | 73.95% | 74.07% | 74.03% | 74.03% | 0.7373 | 0.5863 | **13.05** |

*Note on Topological Features (604d)*: Adding all candidate features (+5d global topological graph metrics, 604d) resulted in a performance regression (0.6996 -> 0.6910 Dice) due to k-NN shortcut message passing between closely traveling parallel companion artery-vein pairs. Pruning these noisy topological descriptors while retaining the 18 optimal descriptors (7 Geometry + 8 Radiomics + 3 Literature = 602d) achieved peak performance.

---

## 3. The 18 Optimal Features: Permutation Feature Importance (PFI)

The 18 optimal features ranked by mean test Dice degradation when individual features are permuted:

| Rank | Feature Identifier | Modality Category | Dice Drop on Permutation | Normalized Importance |
| :---: | :--- | :--- | :---: | :---: |
| **1** | Local Caliber Radius (`r_i`) | Vascular Geometry | 0.0842 | 100.0% |
| **2** | Normalized Caliber Ratio (`r_norm`) | Literature Prior (MorVess) | 0.0765 | 90.9% |
| **3** | Caliber Tapering Rate (`tau_i`) | Vascular Geometry | 0.0691 | 82.1% |
| **4** | Multi-Scale LoG Vesselness (Fine, sigma=1.0) | Multi-Scale Radiomics | 0.0624 | 74.1% |
| **5** | Craniocaudal Orientation (Z-angle) | Vascular Geometry | 0.0583 | 69.2% |
| **6** | Intramural Z-Contrast Gradient | Literature Prior (MorVess) | 0.0541 | 64.3% |
| **7** | Topological Gaussian Radial Profile | Literature Prior (Topo-UNet) | 0.0498 | 59.1% |
| **8** | Relative Neighbor Caliber (`rho_i`) | Vascular Geometry | 0.0462 | 54.9% |
| **9** | 3D Sobel Gradient Magnitude | Multi-Scale Radiomics | 0.0425 | 50.5% |
| **10** | Multi-Scale LoG Vesselness (Medium, sigma=2.0)| Multi-Scale Radiomics | 0.0389 | 46.2% |
| **11** | Local 3D Attenuation Mean | Multi-Scale Radiomics | 0.0354 | 42.0% |
| **12** | Local 3D Texture Std Dev | Multi-Scale Radiomics | 0.0321 | 38.1% |
| **13** | Distance to Hilum Centroid | Vascular Geometry | 0.0294 | 34.9% |
| **14** | Murray's Law Cubic Residual | Vascular Geometry | 0.0268 | 31.8% |
| **15** | Multi-Scale LoG Vesselness (Coarse, sigma=3.0)| Multi-Scale Radiomics | 0.0241 | 28.6% |
| **16** | Branching Alignment / Dispersion | Vascular Geometry | 0.0215 | 25.5% |
| **17** | Raw CT Attenuation | Multi-Scale Radiomics | 0.0189 | 22.4% |
| **18** | Center-to-Shell Parenchymal Contrast | Multi-Scale Radiomics | 0.0163 | 19.4% |

---

### Category-Wise Modality Ablation Impact

Permuting entire feature blocks simultaneously demonstrates the necessity of multimodal fusion:
- **Dropping All Vascular Geometry (7d)**: Macro Dice drops by **-5.84%** (Dice drop: 0.0584). Distal subsegmental branches lose tapering consistency and flip labels.
- **Dropping All Radiomics (8d)**: Macro Dice drops by **-4.32%** (Dice drop: 0.0432). Loss of intramural attenuation and tubular vesselness sharply degrades precision in small vessels.
- **Dropping All Literature Descriptors (3d)**: Macro Dice drops by **-5.18%** (Dice drop: 0.0518). Loss of MorVess normalized caliber and Topo-UNet Gaussian potential destabilizes trunk-to-periphery scaling.

### Reproducing PFI via CLI
Any user or reviewer can independently verify the empirical ranking on held-out test splits using:
```bash
# Evaluate Permutation Feature Importance on AirRC test set
python scripts/compute_feature_importance.py --dataset airrc --checkpoint checkpoints/airrc_enhanced_hgat_602.pth

# Evaluate on HiPaS test set
python scripts/compute_feature_importance.py --dataset hipas --checkpoint checkpoints/hipas_enhanced_hgat_602.pth
```
