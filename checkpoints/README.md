# Model Checkpoints & Pre-Trained Weights

To comply with GitHub repository size best practices (< 50 MB total) and ensure open scientific access, pre-trained `.pth` model checkpoints are hosted externally on open repositories (e.g. Zenodo and Hugging Face Hub).

---

## 1. Automated Download Script

You can download all pre-trained weights with a single command:

```bash
python checkpoints/download_checkpoints.py --all
```

Or download weights for a specific dataset:

```bash
python checkpoints/download_checkpoints.py --dataset airrc
python checkpoints/download_checkpoints.py --dataset hipas
python checkpoints/download_checkpoints.py --dataset lung250m
```

---

## 2. Checkpoint Files & Direct Links

| Dataset | Model Architecture | Input Dims | File Name | Size |
| :--- | :--- | :--- | :--- | :--- |
| **AirRC** | Enhanced HGAT (Proposed) | 602d | `airrc_enhanced_hgat_602.pth` | ~4.2 MB |
| **AirRC** | Extended-HGAT (Best Features) | 602d | `airrc_extended_hgat_602.pth` | ~4.1 MB |
| **AirRC** | Graph-PAVNet (Baseline) | 584d | `airrc_baseline_hgat_584.pth` | ~4.0 MB |
| **HiPaS 2025** | Enhanced HGAT (Proposed) | 602d | `hipas_enhanced_hgat_602.pth` | ~4.2 MB |
| **HiPaS 2025** | Extended-HGAT (Best Features) | 602d | `hipas_extended_hgat_602.pth` | ~4.1 MB |
| **Lung250M-4B**| Enhanced HGAT (Proposed) | 602d | `lung250m_enhanced_hgat_602.pth` | ~4.2 MB |
| **Lung250M-4B**| Extended-HGAT (Best Features) | 602d | `lung250m_extended_hgat_602.pth` | ~4.1 MB |
| **Common** | Frozen ViT Patch Extractor | 576d | `patch_extractor.pth` | ~85 MB |

