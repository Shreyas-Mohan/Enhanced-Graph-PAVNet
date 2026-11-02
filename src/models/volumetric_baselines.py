import torch
import torch.nn as nn
from monai.networks.nets import UNet, VNet, DynUNet, UNETR, SwinUNETR


def build_volumetric_model(
    model_name="unet3d",
    in_channels=1,
    out_channels=3,
    spatial_size=(64, 64, 64)
):
    """
    Instantiates canonical dense volumetric architectures via Project MONAI.
    Classes: 0: Background/Parenchyma, 1: Pulmonary Artery, 2: Pulmonary Vein.
    """
    name = model_name.lower().replace("-", "").replace("_", "")

    if name in ["unet3d", "unet"]:
        # Çiçek et al. (MICCAI 2016)
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance"
        )
    elif name in ["vnet3d", "vnet"]:
        # Milletari et al. (3DV 2016)
        return VNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            act=("elu", {"inplace": True}),
            dropout_prob=0.1
        )
    elif name in ["nnunet", "nnunet3d", "dynunet"]:
        # Isensee et al. (Nature Methods 2021)
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        return DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=[16, 32, 64, 128, 256],
            dropout=0.1,
            norm_name="instance",
            act_name="leakyrelu"
        )
    elif name in ["unetr", "unetr3d"]:
        # Hatamizadeh et al. (WACV 2022)
        return UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=spatial_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="conv",
            norm_name="instance",
            dropout_rate=0.0
        )
    elif name in ["swinunetr", "swinunetr3d"]:
        # Tang et al. (MICCAI BrainLes 2021)
        return SwinUNETR(
            img_size=spatial_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
            drop_rate=0.0,
            attn_drop_rate=0.0
        )
    else:
        raise ValueError(f"Unknown volumetric model name: {model_name}")

