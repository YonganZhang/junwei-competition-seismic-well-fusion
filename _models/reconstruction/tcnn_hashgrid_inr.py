"""Thin tiny-cuda-nn HashGrid implicit-regression adapter."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import require_dependency, validate_n_features


model_id = "tcnn_hashgrid_inr"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["coordinates", "seismic", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "batch_representation": "point",
        "trainable": True,
        "dependency_group": "operator-inr-compiled",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    torch = require_dependency("torch", model_id=model_id, distribution="torch")
    tcnn = require_dependency("tinycudann", model_id=model_id, distribution="tiny-cuda-nn")
    from _models.reconstruction._p5_torch import TorchRegressionAdapter

    n_features = int(config["n_features"])
    validate_n_features(task_spec, n_features)
    device = str(config.get("device", "cuda"))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        from _models.reconstruction._p5_adapter import AdapterSkip
        raise AdapterSkip(
            "cuda_required", "tiny-cuda-nn requires a CUDA device", model_id=model_id, device=device
        )
    seed = int(config.get("seed", 2693))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=n_features,
        n_output_dims=1,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": int(config.get("n_levels", 4)),
            "n_features_per_level": int(config.get("n_features_per_level", 2)),
            "log2_hashmap_size": int(config.get("log2_hashmap_size", 12)),
            "base_resolution": int(config.get("base_resolution", 4)),
            "per_level_scale": float(config.get("per_level_scale", 2.0)),
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": int(config.get("hidden_features", 32)),
            "n_hidden_layers": int(config.get("hidden_layers", 2)),
        },
    )
    return TorchRegressionAdapter(
        torch, network, task_spec, model_id=model_id, n_features=n_features,
        representation="point", learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-6)), device=device,
    )
