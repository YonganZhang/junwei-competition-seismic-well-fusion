"""Thin scratch-only MONAI 3D U-Net adapter for approved spatial labels."""
from __future__ import annotations

from typing import Any

from .p5_common import TorchModuleAdapter, output_dim, require_single_target

model_id = "monai_unet3d"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression"], "input_modalities": ["volume"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "volume", "input_shape": "[B,C,D,H,W]", "pretrained_weights": "forbidden"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    if config.get("pretrained", False):
        raise ValueError("pretrained MONAI/medical weights are forbidden for sweetspot Stage 1")
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("monai") from exc
    in_channels = int(config.get("in_channels", 1)); out_channels = output_dim(task_spec)
    channels = tuple(config.get("channels", (4, 8, 16))); strides = tuple(config.get("strides", (2, 2)))
    device = str(config.get("device", "cpu"))
    factory = lambda: UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, channels=channels, strides=strides, num_res_units=1)
    return TorchModuleAdapter(task_spec, factory, input_key="volume", device=device)
