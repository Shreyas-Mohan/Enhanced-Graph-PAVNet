import sys
import json
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import Data, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BaseVascularGraphDataset(Dataset):
    """
    Base PyTorch Geometric Dataset for vascular landmark point-cloud graphs.
    Standard representation per scan: 8,192 landmark skeleton nodes with k-NN edges (k=10).
    """
    def __init__(
        self,
        clouds_dir,
        features_dir,
        optimal_features_dir=None,
        extended_features_dir=None,
        file_list=None,
        feature_dim=602,
        max_hops=3
    ):
        super().__init__()
        self.clouds_dir = Path(clouds_dir)
        self.features_dir = Path(features_dir)
        self.feature_dim = feature_dim
        self.max_hops = max_hops
        self.data_dir = self.clouds_dir.parent

        self.optimal_features_dir = Path(optimal_features_dir) if optimal_features_dir else None
        if self.optimal_features_dir is None:
            for cand in [self.data_dir / "features_optimal_602Tr", self.data_dir / "features_optimal_602"]:
                if cand.exists():
                    self.optimal_features_dir = cand
                    break

        self.extended_features_dir = Path(extended_features_dir) if extended_features_dir else None
        if self.extended_features_dir is None:
            for cand in [self.data_dir / "extended_featuresTr", self.data_dir / "extended_features"]:
                if cand.exists():
                    self.extended_features_dir = cand
                    break

        coord_dir = self.clouds_dir / "coordinates"
        if not coord_dir.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if (repo_root / self.clouds_dir / "coordinates").exists():
                self.clouds_dir = repo_root / self.clouds_dir
                self.features_dir = repo_root / self.features_dir
                if self.optimal_features_dir:
                    self.optimal_features_dir = repo_root / self.optimal_features_dir
                coord_dir = self.clouds_dir / "coordinates"
        if file_list is not None:
            def _clean_stem(fname):
                name = Path(fname).name
                for ext in [".nii.gz", ".npz", ".mhd", ".pth", ".nii"]:
                    if name.endswith(ext):
                        return name[:-len(ext)]
                return Path(name).stem

            self.file_names = [f if f.endswith(".pth") else f"{_clean_stem(f)}.pth" for f in file_list]
            def _file_ready(f):
                if not (coord_dir / f).exists():
                    return False
                if not (self.features_dir / f).exists():
                    return False
                if self.feature_dim == 602 and self.optimal_features_dir and not (self.optimal_features_dir / f).exists():
                    return False
                return True
            self.file_names = [f for f in self.file_names if _file_ready(f)]
        else:
            self.file_names = sorted([f.name for f in coord_dir.glob("*.pth") if (self.features_dir / f.name).exists()])

        self._graph_cache = {}
        for candidate in [self.data_dir / "skeleton_graph_cache.pt", self.clouds_dir.parent / "skeleton_graph_cache.pt"]:
            if candidate.exists():
                try:
                    self._graph_cache = torch.load(candidate, map_location="cpu", weights_only=False)
                    break
                except Exception:
                    pass

    def __len__(self):
        return len(self.file_names)

    def _normalize_features(self, coords, radii, direction_features):
        coords_min = coords.min(dim=0, keepdim=True)[0]
        coords_max = coords.max(dim=0, keepdim=True)[0]
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-6)

        radii_mean = radii.mean()
        radii_std = radii.std() + 1e-6
        radii_norm = (radii - radii_mean) / radii_std

        dir_mean = direction_features.mean(dim=0, keepdim=True)
        dir_std = direction_features.std(dim=0, keepdim=True) + 1e-6
        dir_norm = (direction_features - dir_mean) / dir_std

        return coords_norm, radii_norm.unsqueeze(1), dir_norm

    def get_sample_data(self, idx):
        file_name = self.file_names[idx]
        coords = torch.load(self.clouds_dir / "coordinates" / file_name, map_location="cpu", weights_only=False)[0].float()
        radii = torch.load(self.clouds_dir / "distance" / file_name, map_location="cpu", weights_only=False)[0].float()
        raw_labels = torch.load(self.clouds_dir / "artery_vein" / file_name, map_location="cpu", weights_only=False)[0]

        # Target mapping: 0 for Artery, 1 for Vein
        if (raw_labels >= 3).any():
            labels = (raw_labels - 3).long()
        elif (raw_labels >= 1).all():
            labels = (raw_labels - 1).long()
        else:
            labels = raw_labels.long()

        # Load 576-dim ViT visual embeddings
        vit_feats = torch.load(self.features_dir / file_name, map_location="cpu", weights_only=False).float()

        # Cached or computed graph topology
        if file_name in self._graph_cache:
            edge_index, seed_mask, hierarchy_mask, dir_feats = self._graph_cache[file_name]
        else:
            from src.utils.graph_builder import build_skeleton_graph
            graph_data = build_skeleton_graph(coords, radii, k=10, max_hops=self.max_hops)
            edge_index = graph_data.edge_index
            seed_mask = graph_data.seed_mask
            hierarchy_mask = graph_data.hierarchy_mask
            dir_feats = graph_data.direction_features
            self._graph_cache[file_name] = (edge_index, seed_mask, hierarchy_mask, dir_feats)

        coords_n, radii_n, dir_n = self._normalize_features(coords, radii, dir_feats)
        base_x = torch.cat([coords_n, radii_n, dir_n, vit_feats], dim=1) # 584 dims

        x = base_x
        if self.feature_dim == 602 and self.optimal_features_dir and (self.optimal_features_dir / file_name).exists():
            opt_feats = torch.load(self.optimal_features_dir / file_name, map_location="cpu", weights_only=False).float()
            if opt_feats.shape[1] == 18:
                opt_mean = opt_feats.mean(dim=0, keepdim=True)
                opt_std = opt_feats.std(dim=0, keepdim=True) + 1e-6
                opt_norm = torch.nan_to_num((opt_feats - opt_mean) / opt_std, nan=0.0)
                x = torch.cat([base_x, opt_norm], dim=1) # 602 dims
            elif opt_feats.shape[1] == 602:
                x = opt_feats
        elif self.feature_dim in [592, 599, 604] and self.extended_features_dir and (self.extended_features_dir / file_name).exists():
            ext_feats = torch.load(self.extended_features_dir / file_name, map_location="cpu", weights_only=False).float()
            ext_mean = ext_feats.mean(dim=0, keepdim=True)
            ext_std = ext_feats.std(dim=0, keepdim=True) + 1e-6
            ext_norm = torch.nan_to_num((ext_feats - ext_mean) / ext_std, nan=0.0)
            if self.feature_dim == 592:
                x = torch.cat([base_x, ext_norm[:, 12:20]], dim=1) # 584 + 8 = 592
            elif self.feature_dim == 599:
                x = torch.cat([base_x, ext_norm[:, 0:7], ext_norm[:, 12:20]], dim=1) # 584 + 15 = 599
            elif self.feature_dim == 604:
                x = torch.cat([base_x, ext_norm], dim=1) # 584 + 20 = 604

        data = Data(x=x, edge_index=edge_index, y=labels)
        data.seed_mask = seed_mask
        data.hierarchy_mask = hierarchy_mask
        data.radii = radii
        data.original_coords = coords
        data.file_name = file_name
        return data

    def __getitem__(self, idx):
        return self.get_sample_data(idx)


def load_dataset_splits(splits_json_path):
    """Loads standardized patient-disjoint train/val/test split file lists."""
    p = Path(splits_json_path)
    if not p.exists():
        alt = Path(__file__).resolve().parent.parent / "splits" / p.name
        if alt.exists():
            p = alt
    with open(p, "r") as f:
        splits = json.load(f)
    return splits.get("train", []), splits.get("val", []), splits.get("test", [])

