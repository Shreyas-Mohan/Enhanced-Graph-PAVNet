from pathlib import Path
from .base_loader import BaseVascularGraphDataset, load_dataset_splits


class HiPaSGraphDataset(BaseVascularGraphDataset):
    """
    HiPaS 2025 Thoracic CT graph dataset (Chu et al., Nat Commun 2025).
    Incorporates dynamic millimeter spacing and hard-negative mining metadata.
    """
    def __init__(
        self,
        root_dir="data/hipas",
        split="train",
        splits_file="data/splits/hipas_splits.json",
        feature_dim=602,
        max_hops=3
    ):
        root = Path(root_dir)
        train_files, val_files, test_files = load_dataset_splits(splits_file)

        split_map = {
            "train": train_files,
            "val": val_files,
            "validation": val_files,
            "test": test_files
        }
        file_list = split_map.get(split.lower(), train_files)

        # Support both 'features_optimal_602' and 'features_optimal_602Tr'
        optimal_dir = root / "features_optimal_602"
        if not optimal_dir.exists() and (root / "features_optimal_602Tr").exists():
            optimal_dir = root / "features_optimal_602Tr"

        super().__init__(
            clouds_dir=root / "cloudsTr",
            features_dir=root / "featuresTr",
            optimal_features_dir=optimal_dir,
            file_list=file_list,
            feature_dim=feature_dim,
            max_hops=max_hops
        )

