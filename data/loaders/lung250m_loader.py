from pathlib import Path
from .base_loader import BaseVascularGraphDataset, load_dataset_splits


class Lung250MGraphDataset(BaseVascularGraphDataset):
    """
    Lung250M-4B paired inspiratory/expiratory CT graph dataset (Falta et al., NeurIPS 2023).
    Patient-grouped splits strictly preserve pairs (e.g. case_054_1 and case_054_2).
    """
    def __init__(
        self,
        root_dir="data/lung250m",
        split="train",
        splits_file="data/splits/lung250m_splits.json",
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

        is_test = split.lower() == "test"
        clouds_dir = root / ("cloudsTs" if is_test else "cloudsTr")
        features_dir = root / ("featuresTs" if is_test else "featuresTr")
        optimal_dir = root / ("features_optimal_602Ts" if is_test else "features_optimal_602Tr")

        super().__init__(
            clouds_dir=clouds_dir,
            features_dir=features_dir,
            optimal_features_dir=optimal_dir,
            file_list=file_list,
            feature_dim=feature_dim,
            max_hops=max_hops
        )

