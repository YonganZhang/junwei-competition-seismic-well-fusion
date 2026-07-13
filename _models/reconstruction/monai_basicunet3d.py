"""Thin 3-D regression adapter for MONAI BasicUNet."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import require_dependency, validate_n_features


model_id = "monai_basicunet3d"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["seismic", "coordinates", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "batch_representation": "volume",
        "trainable": True,
        "dependency_group": "monai-3d",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    torch = require_dependency("torch", model_id=model_id, distribution="torch")
    monai = require_dependency("monai", model_id=model_id, distribution="monai")
    from _models.reconstruction._p5_torch import TorchRegressionAdapter

    n_features = int(config["n_features"])
    validate_n_features(task_spec, n_features)
    seed = int(config.get("seed", 2693))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base = int(config.get("base_channels", 8))
    network = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=n_features,
        out_channels=1,
        features=(base, base, base * 2, base * 4, base * 8, base),
        upsample="deconv",
    )
    return TorchRegressionAdapter(
        torch, network, task_spec, model_id=model_id, n_features=n_features,
        representation="volume", learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        # Four down-sampling stages turn 16^3 into a 1^3 bottleneck, which is
        # invalid for BasicUNet's training-mode InstanceNorm.  Padding to 32^3
        # preserves the upstream normalization and yields a 2^3 bottleneck.
        device=str(config.get("device", "cpu")), minimum_spatial_size=32,
    )
