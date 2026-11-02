#!/usr/bin/env python
"""
Automated Downloader for Pre-Trained Checkpoints and Model Weights.

NOTE:
The remote URLs configured below currently serve as placeholder endpoints.
Trained model checkpoints are being prepared and will soon be uploaded to the Hugging Face
Hub upon paper publication.
In the meantime, you can:
  1. Train models from scratch using `scripts/train.py`
  2. Place locally trained `.pth` weight files directly in this directory
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request

# Placeholder Hugging Face repository endpoints.
# These will be updated with the live Hugging Face repository upon publication release.
CHECKPOINT_URLS = {
    # AirRC
    "airrc_enhanced_hgat_602.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/airrc_enhanced_hgat_602.pth",
    "airrc_extended_hgat_602.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/airrc_extended_hgat_602.pth",
    "airrc_baseline_hgat_584.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/airrc_baseline_hgat_584.pth",

    # HiPaS 2025
    "hipas_enhanced_hgat_602.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/hipas_enhanced_hgat_602.pth",
    "hipas_extended_hgat_602.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/hipas_extended_hgat_602.pth",

    # Lung250M-4B
    "lung250m_enhanced_hgat_602.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/lung250m_enhanced_hgat_602.pth",
    "lung250m_extended_hgat_602.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/lung250m_extended_hgat_602.pth",

    # ViT Patch Extractor
    "patch_extractor.pth": "https://huggingface.co/pulmonary-vascular-ai/enhanced-hgat/resolve/main/patch_extractor.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-trained model weights.")
    parser.add_argument("--all", action="store_true", help="Download all checkpoints")
    parser.add_argument("--dataset", type=str, choices=["airrc", "hipas", "lung250m", "common"], default="airrc")
    return parser.parse_args()


def main():
    args = parse_args()
    dest_dir = Path(__file__).resolve().parent
    print(f"[+] Target Checkpoints Directory: {dest_dir}")

    if args.all:
        targets = list(CHECKPOINT_URLS.keys())
    elif args.dataset == "common":
        targets = ["patch_extractor.pth"]
    else:
        targets = [k for k in CHECKPOINT_URLS if args.dataset in k]

    for fname in targets:
        target_path = dest_dir / fname
        if target_path.exists():
            print(f"[OK] Already exists: {fname}")
            continue

        url = CHECKPOINT_URLS[fname]
        print(f"[+] Downloading {fname} from {url}...")
        try:
            urllib.request.urlretrieve(url, str(target_path))
            print(f"[SUCCESS] Downloaded: {fname}")
        except Exception as e:
            print(f"[!] Note: Remote checkpoints are currently placeholders ({e}).")
            print(f"    Checkpoints will soon be uploaded to Hugging Face upon publication.")
            print(f"    In the meantime, you can manually place '{fname}' into {dest_dir} or run scripts/train.py.")


if __name__ == "__main__":
    main()
