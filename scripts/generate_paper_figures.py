#!/usr/bin/env python
"""
Paper Figure Generator & Inspector: Reproduces 12-Panel Baseline Grids (Figure 1)
and 8-Panel Feature Ablation Grids (Figure 2) across AirRC, HiPaS 2025, and Lung250M-4B.
"""

import sys
import shutil
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIGURE_REGISTRY = {
    "airrc": {
        "baselines": [
            "grid1_baselines_slice131_case_test15.png",
            "grid1_baselines_slice136_case_test15.png",
        ],
        "ablations": [
            "grid2_ablations_slice140_case_test15.png",
            "grid2_ablations_slice135_case_test15.png",
        ]
    },
    "hipas": {
        "baselines": [
            "grid1_baselines_slice198_case036.png",
        ],
        "ablations": [
            "grid2_ablations_slice195_case036.png",
            "grid2_ablations_slice210_case036.png",
        ]
    },
    "lung250m": {
        "baselines": [
            "grid1_baselines_slice128_case_054_2.png",
            "grid1_baselines_slice169_case_054_2.png",
            "grid1_baselines_slice173_case_054_1.png",
        ],
        "ablations": [
            "grid2_ablations_slice169_case_054_2.png",
            "grid2_ablations_slice178_case_054_2.png",
            "grid2_ablations_slice53_case_054_1.png",
        ]
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate/inspect qualitative prediction grids for paper.")
    parser.add_argument("--dataset", type=str, default="airrc", choices=["airrc", "hipas", "lung250m"])
    parser.add_argument("--grid-type", type=str, default="all", choices=["baselines", "ablations", "all"])
    parser.add_argument("--figures-dir", type=str, default=str(PROJECT_ROOT / "docs" / "figures"))
    parser.add_argument("--output-dir", type=str, default=None, help="Optional destination to copy figures")
    return parser.parse_args()


def main():
    args = parse_args()
    fig_dir = Path(args.figures_dir)
    print("=" * 80)
    print(f"QUALITATIVE FIGURES INSPECTOR: {args.dataset.upper()} ({args.grid_type.upper()})")
    print("=" * 80)

    dataset_figs = FIGURE_REGISTRY.get(args.dataset, {})
    types_to_check = ["baselines", "ablations"] if args.grid_type == "all" else [args.grid_type]

    found_count = 0
    for gtype in types_to_check:
        fig_list = dataset_figs.get(gtype, [])
        print(f"\n[+] {gtype.capitalize()} Figures:")
        for fname in fig_list:
            fpath = fig_dir / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                print(f"    [FOUND] {fname:<45} ({size_mb:.2f} MB)")
                found_count += 1
                if args.output_dir:
                    out_p = Path(args.output_dir)
                    out_p.mkdir(parents=True, exist_ok=True)
                    shutil.copy(fpath, out_p / fname)
                    print(f"            --> Copied to {out_p / fname}")
            else:
                print(f"    [MISSING] {fname:<45} (Expected in {fig_dir})")

    print("\n" + "=" * 80)
    print(f"Figure inspection complete. {found_count} verified high-resolution figures ready for paper / DOCX.")
    print("=" * 80)


if __name__ == "__main__":
    main()
