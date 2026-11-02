# Dataset Specifications & Setup

This repository evaluates models across three distinct clinical thoracic CT benchmark datasets under strict, zero-patient-leakage protocols.

---

## 1. Dataset Overview & Citations

### 1.1 AirRC Dataset
- **Citation**: Liu, J., Zhang, Z., Niu, B. et al. *A Custom Annotated Dataset for Segmentation of Pulmonary Veins, Arteries, and Airways.* Scientific Data 12, 1806 (2025). [https://doi.org/10.1038/s41597-025-06074-6](https://doi.org/10.1038/s41597-025-06074-6)
- **Origin**: Curated from the open-source LUNA16 thoracic CT repository with expert radiologist voxel-level annotations.
- **Cohort**: 254 thoracic CT scans stored as 3D NIfTI (`.nii.gz`) volumes.
- **Resolution**: Resampled to an isotropic 1.0 mm³ grid (1 voxel = 1.0 physical mm).
- **Split**: 177 train / 38 val / 39 held-out test (`data/splits/airrc_splits.json`).

### 1.2 HiPaS 2025 Dataset
- **Citation**: Chu, Y., Luo, G., Zhou, L. et al. *Deep learning-driven pulmonary artery and vein segmentation reveals demography-associated vasculature anatomical differences.* Nature Communications 16, 2262 (2025). [https://doi.org/10.1038/s41467-025-56505-6](https://doi.org/10.1038/s41467-025-56505-6)
- **Origin**: HiPaS 2025 Pulmonary Artery-Vein Challenge, Shanghai Chest Hospital.
- **Cohort**: 250 thoracic CT scans stored as compressed NumPy arrays (`.npz`, 512×512×Z) with per-patient dynamic millimeter spacing from `metadata.xlsx`.
- **Pre-Processing**: Dual-path normalization guard ([-1000, 600] HU) and targeted hard-negative soft-tissue patch mining (60% foreground, 25% soft-tissue non-vessel, 15% parenchyma).
- **Split**: 175 train / 38 val / 37 held-out test (`data/splits/hipas_splits.json`).

### 1.3 Lung250M-4B Dataset
- **Citation**: Falta, F., Großbröhmer, C., Hering, A., Bigalke, A., Heinrich, M. *Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration.* NeurIPS 2023 Datasets and Benchmarks Track. [https://doi.org/10.52202/075280-2392](https://doi.org/10.52202/075280-2392)
- **Origin**: Institute of Medical Informatics, University of Lübeck ([GitHub](https://github.com/multimodallearning/Lung250M-4B)).
- **Cohort**: 248 paired CT scans from 124 patients, each with full inspiration (`_1`) and forced expiration (`_2`) breath-hold scans.
- **Splitting**: Patient-wise group splitting prevents leakage across paired breath-holds.
- **Split**: 138 train (69 pairs) / 36 val (18 pairs) / 30 held-out test (15 pairs) (`data/splits/lung250m_splits.json`).

---

## 2. Directory Structure

### 2.1 Raw Volume Layout
To run preprocessing from raw files, place downloaded volumes in `data/<dataset_name>/`:

```text
data/
├── airrc/
│   ├── imagesTr/          # Raw CT NIfTI scans (.nii.gz)
│   └── segTr/             # Ground truth label volumes (.nii.gz)
├── hipas/
│   ├── ct_scan/           # Raw CT scans (.npz, 001.npz ... 250.npz)
│   ├── annotation/        # Raw artery & vein segmentations (.npz)
│   │   ├── artery/
│   │   └── vein/
│   └── metadata.xlsx      # Patient voxel spacing (dx, dy, dz)
├── lung250m/
│   ├── imagesTr/, imagesTs/       # CT scans (.nii.gz, case_XXX_1/2)
│   ├── segTr/, segTs/             # Dense multi-class ground truth (.nii.gz)
│   ├── cloudsTr/, cloudsTs/       # Skeleton coordinates & distance radii (.pth)
│   └── features_optimal_602Tr/    # Precomputed optimal feature descriptors (.pth)
└── splits/
    ├── airrc_splits.json
    ├── hipas_splits.json
    └── lung250m_splits.json
```

### 2.2 Landmark Graph Structure (PyG Datasets)
For landmark graph-based training and evaluation (`EnhancedHGATNet`, `BaselineHGATNet`, `StandardGNNBaseline`), each dataset directory contains pre-extracted graph representations:

```text
data/<dataset_name>/
├── cloudsTr/
│   ├── coordinates/       # [N, 3] skeleton landmark coordinates (.pth)
│   ├── distance/          # [N, 1] distance transform radius values (.pth)
│   └── artery_vein/       # [N] ground-truth vessel class labels (.pth)
├── featuresTr/            # [N, 576] multi-planar ViT patch visual embeddings (.pth)
└── features_optimal_602/  # [N, 18] or [N, 602] domain-specific optimal feature tensors (.pth)
```

---

## 3. Landmark Graph Sampling & Preprocessing

Vascular landmark graphs ($N = 8,192$ nodes per scan) are extracted from binary vessel masks using distance-transform-weighted sampling. Graph edges are constructed via 3D spatial Euclidean $k$-nearest neighbors ($k=3$, bidirectional), connecting adjacent vascular segments and naturally bridging small $1\text{--}3\text{ mm}$ partial-volume discontinuities without requiring unbroken 1D medial centerlines.


