# Model Checkpoints & Pre-Trained Weights

To comply with GitHub repository size best practices (< 50 MB total) and ensure open scientific access, pre-trained `.pth` model checkpoints will be hosted externally on the Hugging Face Hub.

> [!NOTE]
> **Checkpoints Release Notice**: The remote URLs configured in `download_checkpoints.py` currently serve as placeholder endpoints. Model weights are being packaged and will soon be uploaded to Hugging Face upon paper publication.
> In the meantime, all models can be trained directly from scratch using `scripts/train.py`, or locally trained `.pth` weight files can be manually placed into this `checkpoints/` directory.

---

## 1. Automated Download Script

Once public checkpoints are released on Hugging Face, you can download all pre-trained weights with a single command:

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

## 2. Checkpoint Files & Model Architecture

| Dataset | Model Architecture | Input Dims | Checkpoint File Name | Size |
| :--- | :--- | :--- | :--- | :--- |
| **AirRC** | Enhanced HGAT (Proposed) | 602d | `airrc_enhanced_hgat_602.pth` | ~4.2 MB |
| **AirRC** | Extended-HGAT (Best Features) | 602d | `airrc_extended_hgat_602.pth` | ~4.1 MB |
| **AirRC** | Graph-PAVNet (Baseline) | 584d | `airrc_baseline_hgat_584.pth` | ~4.0 MB |
| **HiPaS 2025** | Enhanced HGAT (Proposed) | 602d | `hipas_enhanced_hgat_602.pth` | ~4.2 MB |
| **HiPaS 2025** | Extended-HGAT (Best Features) | 602d | `hipas_extended_hgat_602.pth` | ~4.1 MB |
| **Lung250M-4B**| Enhanced HGAT (Proposed) | 602d | `lung250m_enhanced_hgat_602.pth` | ~4.2 MB |
| **Lung250M-4B**| Extended-HGAT (Best Features) | 602d | `lung250m_extended_hgat_602.pth` | ~4.1 MB |
| **Common** | Frozen ViT Patch Extractor | 576d | `patch_extractor.pth` | ~85 MB |
