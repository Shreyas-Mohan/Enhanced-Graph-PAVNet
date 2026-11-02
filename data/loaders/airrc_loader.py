from pathlib import Path
from .base_loader import BaseVascularGraphDataset, load_dataset_splits


class AirRCGraphDataset(BaseVascularGraphDataset):
    """
    AirRC Thoracic CT graph dataset (Liu et al., Sci Data 2025).
    Volumes resampled to isotropic 1.0 mm³ grid with 8,192 landmark skeleton nodes.
    """
    def __init__(
        self,
        root_dir="data/airrc",
        split="train",
        splits_file="data/splits/airrc_splits.json",
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

        # Support both 'features_optimal_602Tr' and 'features_optimal_602'
        optimal_dir = root / "features_optimal_602Tr"
        if not optimal_dir.exists() and (root / "features_optimal_602").exists():
            optimal_dir = root / "features_optimal_602"

        super().__init__(
            clouds_dir=root / "cloudsTr",
            features_dir=root / "featuresTr",
            optimal_features_dir=optimal_dir,
            file_list=file_list,
            feature_dim=feature_dim,
            max_hops=max_hops
        )

