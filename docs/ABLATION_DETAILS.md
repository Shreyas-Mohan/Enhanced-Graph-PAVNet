# Feature Descriptions, Formulations & Dataset-Wise Ablations

This document details the candidate feature inventory, their mathematical formulations, category-wise ablation benchmarks, and the 18 optimal feature importance rankings.

> **Overview**: To complement patch-level ViT visual embeddings where peripheral microvascular contrast is low, the framework incorporates 18 domain-specific descriptors spanning vascular geometry (e.g. tapering rate, Murray's law), multi-scale radiomics, and literature priors. Out of 23 initially evaluated candidates, 5 global graph metrics sensitive to peripheral mask boundaries were pruned to optimize generalization.

---

## 1. Candidate Features Inventory by Modality

### Category A: Vascular Morphometry (7 Dimensions)
Geometric descriptors capturing lumen cross-section, tree hierarchy, and physical orientation.

1. **Local Vessel Caliber Radius ($r_i$)**:
   $$r_i = D(v_i, \partial\Omega) = \min_{x \in \partial\Omega} \|v_i - x\|_2$$
   *Physical Euclidean distance from skeleton node $v_i$ to nearest outer vessel wall $\partial\Omega$. Distinguishes main trunks from capillaries.*

2. **Caliber Tapering Rate ($\tau_i$)**:
   $$\tau_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \frac{r_j - r_i}{\|x_j - x_i\|_2}$$
   *Mean directional rate of change in radius across connected neighboring nodes. Veins maintain more uniform caliber, whereas arteries taper rapidly.*

3. **Relative Local Caliber Ratio ($\rho_i$)**:
   $$\rho_i = \frac{r_i}{\frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} r_j + \epsilon}$$
   *Ratio of node radius to mean radius of its immediate 1-ring neighbors. Identifies local stenosis or bifurcation enlargements.*

4. **Craniocaudal Angle Orientation ($\theta_{z,i}$)**:
   $$\theta_{z,i} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \frac{|z_j - z_i|}{\|x_j - x_i\|_2}$$
   *Mean absolute vertical z-component of unit edge vectors. Distinguishes vertically descending basal veins from diagonally fanning arteries.*

5. **Branching Alignment / Dispersion ($\delta_i$)**:
   $$\delta_i = \left\| \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \frac{x_j - x_i}{\|x_j - x_i\|_2} \right\|_2$$
   *Euclidean norm of average outward direction unit vectors. Values near 1.0 indicate straight segments; values near 0.0 indicate high-curvature bifurcations.*

6. **Distance to Cardiac Hilum Centroid ($d_{\text{hilum},i}$)**:
   $$d_{\text{hilum},i} = \frac{\|x_i - c_{\text{hilum}}\|_2}{100.0}, \quad c_{\text{hilum}} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} x_s$$
   *Normalized physical Euclidean distance from node $v_i$ to cardiac hilum seed centroid $c_{\text{hilum}}$. Encodes proximal-to-distal tree depth.*

7. **Murray's Law Cubic Flow Residual ($\Delta M_i$)**:
   $$\Delta M_i = \frac{r_i^3 - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} r_j^3}{r_i^3 + \epsilon}$$
   *Normalized residual deviation from Murray's physiological cubic branching law ($r_{\text{parent}}^3 = \sum r_{\text{child}}^3$).*

---

### Category B: Multi-Scale 3D Radiomics (8 Dimensions)
Intensity distributions and structural second-order texture derivatives from raw CT volumes.

1. **Raw CT Attenuation ($I_i$)**:
   $$\hat{I}_i = \frac{I(x_i, y_i, z_i)}{1000.0}$$
   *Raw CT Hounsfield Unit (HU) attenuation scaled by 1,000.*

2. **Local 3D Mean Attenuation ($\mu_{5,i}$)**:
   $$\mu_{5,i} = \frac{1}{125} \sum_{u,v,w \in [-2,2]} I(x_i+u, y_i+v, z_i+w)$$
   *Mean attenuation across a $5 \times 5 \times 5$ voxel neighborhood.*

3. **Local 3D Intensity Standard Deviation ($\sigma_{5,i}$)**:
   $$\sigma_{5,i} = \sqrt{\frac{1}{125} \sum (I - \mu_5)^2}$$
   *Quantifies local tissue heterogeneity and blood-pool lumen uniformity.*

4. **3D Sobel Gradient Magnitude ($|\nabla I|_i$)**:
   $$|\nabla I|_i = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2 + \left(\frac{\partial I}{\partial z}\right)^2}$$
   *Quantifies sharp contrast transitions at the vessel wall boundary.*

5. **Multi-Scale Laplacian of Gaussian ($\text{LoG}_{\sigma=1,2,3}$)**:
   $$\text{LoG}_\sigma(x) = \sigma^2 \nabla^2 (G_\sigma * I)(x)$$
   *Three distinct scale-space tubular vesselness responses for fine ($\sigma=1.0$), medium ($\sigma=2.0$), and coarse ($\sigma=3.0$) branches.*

6. **Center-to-Parenchymal Shell Contrast ($C_{\text{shell},i}$)**:
   $$C_{\text{shell},i} = I(x_i) - \mu_{11}(x_i)$$
   *Difference between intramural core attenuation and surrounding parenchyma.*

---

### Category C: High-Impact Literature Descriptors (3 Dimensions)

1. **Scale-Invariant Normalized Caliber ($\tilde{r}_i$)** (MorVess):
   $$\tilde{r}_i = \frac{r_i}{\text{Quantile}_{0.99}(R)}$$
   *Normalizes lumen diameter by 99th-percentile main pulmonary trunk radius.*

2. **Topological Gaussian Radial Decay Profile ($G_{\text{topo},i}$)** (Topo-UNet):
   $$G_{\text{topo},i} = \exp\left( -\frac{(r_i - r_{\text{trunk}})^2}{2 \sigma_r^2} \right)$$
   *Encodes distance from main pulmonary artery trunk as continuous Gaussian decay.*

3. **Intramural Attenuation Z-Contrast Gradient ($Z_{\text{contrast},i}$)** (MorVess VDM):
   $$Z_{\text{contrast},i} = \text{clip}\left( \frac{I_i - \mu_{5,i}}{\sigma_{5,i} + \epsilon}, -5.0, 5.0 \right)$$
   *Normalized local contrast gradient distinguishing high-pressure contrast in arteries from venous return.*

---

## 2. Dataset-Wise Feature Ablation Tables

### Table 3.1: AirRC Dataset Ablations (39 Test Scans)

| Ablation Stage | Input Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | 584d | 74.17% | 74.65% | 74.02% | 74.02% | 0.7388 | 0.5881 | 12.80 |
| **+ Radiomics** | 592d | 74.48% | 74.84% | 74.49% | 74.49% | 0.7429 | 0.5938 | 12.82 |
| **+ Vascular Geometry** | 599d | 74.81% | 75.15% | 74.62% | 74.62% | 0.7453 | 0.5973 | 12.92 |
| **All Candidate Features** | 604d | 74.38% | 74.64% | 74.39% | 74.39% | 0.7422 | 0.5927 | 12.91 |
| **Extended-HGAT (Best Features)**| 602d | 77.23% | 77.71% | 76.99% | 76.99% | 0.7693 | 0.6283 | 12.57 |
| **Enhanced HGAT (Proposed)** | 602d | **77.99%** | **78.24%** | **77.85%** | **77.85%** | **0.7780** | **0.6400** | **12.05** |

---

### Table 3.2: HiPaS 2025 Dataset Ablations (37 Test Scans)

| Ablation Stage | Input Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | 584d | 77.39% | 77.74% | 76.21% | 76.21% | 0.7600 | 0.6243 | 16.48 |
| **+ Radiomics** | 592d | 79.84% | 79.62% | 79.15% | 79.15% | 0.7938 | 0.6698 | 14.82 |
| **+ Vascular Geometry** | 599d | 81.65% | 81.42% | 81.08% | 81.08% | 0.8124 | 0.6952 | 13.25 |
| **All Candidate Features** | 604d | 81.49% | 81.18% | 81.01% | 81.01% | 0.8096 | 0.6925 | 12.33 |
| **Extended-HGAT (Best Features)**| 602d | 82.35% | 82.13% | 81.84% | 81.84% | 0.8184 | 0.7045 | 11.96 |
| **Enhanced HGAT (Proposed)** | 602d | **82.60%** | **82.46%** | **82.10%** | **82.10%** | **0.8212** | **0.7089** | **11.74** |

---

### Table 3.3: Lung250M-4B Dataset Ablations (30 Paired Test Scans)

| Ablation Stage | Input Dims | Accuracy | Precision | Recall | Specificity | Dice (DSC) | mIoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-PAVNet (Baseline)** | 584d | 68.73% | 69.13% | 68.96% | 68.96% | 0.6851 | 0.5227 | 14.19 |
| **+ Radiomics (Cat C)** | 592d | 69.14% | 69.51% | 69.00% | 69.00% | 0.6867 | 0.5250 | 14.11 |
| **+ Vascular Geometry (Cat A+C)** | 599d | 70.34% | 70.44% | 70.19% | 70.19% | 0.6996 | 0.5404 | 14.95 |
| **+ Topological (Cat A+B+C)** | 604d | 69.28% | 70.23% | 69.88% | 69.88% | 0.6910 | 0.5294 | 15.44 |
| **Extended-HGAT (Best Features)**| 602d | **74.33%** | **74.42%** | **74.45%** | **74.45%** | **0.7414** | **0.5915** | 13.11 |
| **Enhanced HGAT (Proposed)** | 602d | 73.95% | 74.07% | 74.03% | 74.03% | 0.7373 | 0.5863 | **13.05** |

*Note on Topological Features (604d)*: Adding all 20 candidate features (+5d topological graph features, 604d) resulted in a performance regression (0.6996 -> 0.6910 Dice) due to k-NN shortcut message passing between closely traveling parallel companion artery-vein pairs. Pruning these noisy topological descriptors while retaining the 18 optimal descriptors (7 Geometry + 8 Radiomics + 3 Literature = 602d) achieved the peak performance.

---

## 3. The 18 Optimal Features: Permutation Feature Importance (PFI)

The 18 optimal features ranked by mean test Dice degradation ($\Delta \text{Dice}$) when permuted:

| Rank | Feature Identifier | Mathematical Symbol | Modality Category | $\Delta \text{Dice}$ | Normalized Importance |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **1** | Local Caliber Radius | $r_i$ | Vascular Geometry | 0.0842 | 100.0% |
| **2** | Normalized Caliber Ratio | $\tilde{r}_i$ | Literature Prior (MorVess) | 0.0765 | 90.9% |
| **3** | Caliber Tapering Rate | $\tau_i$ | Vascular Geometry | 0.0691 | 82.1% |
| **4** | Multi-Scale LoG Vesselness ($\sigma=1.0$) | $\text{LoG}_1$ | Multi-Scale Radiomics | 0.0624 | 74.1% |
| **5** | Craniocaudal Orientation ($Z$-angle) | $\theta_{z,i}$ | Vascular Geometry | 0.0583 | 69.2% |
| **6** | Intramural Z-Contrast Gradient | $Z_{\text{contrast},i}$ | Literature Prior (MorVess) | 0.0541 | 64.3% |
| **7** | Topological Gaussian Radial Profile | $G_{\text{topo},i}$ | Literature Prior (Topo-UNet) | 0.0498 | 59.1% |
| **8** | Relative Neighbor Caliber | $\rho_i$ | Vascular Geometry | 0.0462 | 54.9% |
| **9** | 3D Sobel Gradient Magnitude | $\|\nabla I\|_i$ | Multi-Scale Radiomics | 0.0425 | 50.5% |
| **10** | Multi-Scale LoG Vesselness ($\sigma=2.0$) | $\text{LoG}_2$ | Multi-Scale Radiomics | 0.0389 | 46.2% |
| **11** | Local 3D Attenuation Mean | $\mu_{5,i}$ | Multi-Scale Radiomics | 0.0354 | 42.0% |
| **12** | Local 3D Texture Std Dev | $\sigma_{5,i}$ | Multi-Scale Radiomics | 0.0321 | 38.1% |
| **13** | Distance to Hilum Centroid | $d_{\text{hilum},i}$ | Vascular Geometry | 0.0294 | 34.9% |
| **14** | Murray's Law Cubic Residual | $\Delta M_i$ | Vascular Geometry | 0.0268 | 31.8% |
| **15** | Multi-Scale LoG Vesselness ($\sigma=3.0$) | $\text{LoG}_3$ | Multi-Scale Radiomics | 0.0241 | 28.6% |
| **16** | Branching Alignment / Dispersion | $\delta_i$ | Vascular Geometry | 0.0215 | 25.5% |
| **17** | Raw CT Attenuation | $I_i$ | Multi-Scale Radiomics | 0.0189 | 22.4% |
| **18** | Center-to-Shell Parenchymal Contrast | $C_{\text{shell},i}$ | Multi-Scale Radiomics | 0.0163 | 19.4% |

---

### Category-Wise Modality Ablation Impact

Permuting entire feature blocks simultaneously demonstrates the necessity of multimodal fusion:
- **Dropping All Vascular Geometry (7d)**: Macro Dice drops by **-5.84%** ($\Delta \text{Dice} = 0.0584$). Distal subsegmental branches lose tapering consistency and flip labels.
- **Dropping All Radiomics (8d)**: Macro Dice drops by **-4.32%** ($\Delta \text{Dice} = 0.0432$). Loss of intramural attenuation and tubular vesselness sharply degrades precision in small vessels.
- **Dropping All Literature Descriptors (3d)**: Macro Dice drops by **-5.18%** ($\Delta \text{Dice} = 0.0518$). Loss of MorVess normalized caliber and Topo-UNet Gaussian potential destabilizes trunk-to-periphery scaling.

### Reproducing PFI via CLI
Any user or reviewer can independently verify the empirical ranking on held-out test splits using:
```bash
# Evaluate Permutation Feature Importance on AirRC test set
python scripts/compute_feature_importance.py --dataset airrc --checkpoint checkpoints/airrc_enhanced_hgat_602.pth

# Evaluate on HiPaS test set
python scripts/compute_feature_importance.py --dataset hipas --checkpoint checkpoints/hipas_enhanced_hgat_602.pth
```


