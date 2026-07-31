"""Thin 3-D Fourier Neural Operator regression adapter."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import require_dependency, validate_n_features


model_id = "neuralop_fno3d"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["seismic", "coordinates", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "batch_representation": "volume",
        "trainable": True,
        "dependency_group": "operator-inr",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    torch = require_dependency("torch", model_id=model_id, distribution="torch")
    require_dependency("neuralop", model_id=model_id, distribution="neuraloperator")
    from neuralop.models import FNO
    from _models.reconstruction._p5_torch import TorchRegressionAdapter

    n_features = int(config["n_features"])
    validate_n_features(task_spec, n_features)
    seed = int(config.get("seed", 2693))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    modes = tuple(int(value) for value in config.get("n_modes", (4, 8, 8)))
    network = FNO(
        n_modes=modes,
        in_channels=n_features,
        out_channels=1,
        hidden_channels=int(config.get("hidden_channels", 16)),
        n_layers=int(config.get("n_layers", 2)),
    )
    return TorchRegressionAdapter(
        torch, network, task_spec, model_id=model_id, n_features=n_features,
        representation="volume", learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        device=str(config.get("device", "cpu")), minimum_spatial_size=1,
    )
