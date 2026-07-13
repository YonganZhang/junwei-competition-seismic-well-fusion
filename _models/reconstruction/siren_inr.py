"""Thin modern-PyTorch implementation of the locked MIT SIREN architecture."""
from __future__ import annotations

import math
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import require_dependency, validate_n_features


model_id = "siren_inr"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["coordinates", "seismic", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "batch_representation": "point",
        "trainable": True,
        "dependency_group": "torch-common",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    torch = require_dependency("torch", model_id=model_id, distribution="torch")
    from _models.reconstruction._p5_torch import TorchRegressionAdapter

    n_features = int(config["n_features"])
    validate_n_features(task_spec, n_features)
    hidden_features = int(config.get("hidden_features", 32))
    hidden_layers = int(config.get("hidden_layers", 2))
    omega_0 = float(config.get("omega_0", 30.0))
    seed = int(config.get("seed", 2693))
    if hidden_features <= 0 or hidden_layers <= 0 or omega_0 <= 0:
        raise ValueError("invalid SIREN width/depth/frequency")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class SineLayer(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, *, first: bool) -> None:
            super().__init__()
            self.in_features = in_features
            self.first = first
            self.linear = torch.nn.Linear(in_features, out_features)
            with torch.no_grad():
                bound = (1.0 / in_features) if first else (math.sqrt(6.0 / in_features) / omega_0)
                self.linear.weight.uniform_(-bound, bound)

        def forward(self, values: Any) -> Any:
            return torch.sin(omega_0 * self.linear(values))

    layers: list[Any] = [SineLayer(n_features, hidden_features, first=True)]
    layers.extend(
        SineLayer(hidden_features, hidden_features, first=False) for _ in range(hidden_layers - 1)
    )
    final = torch.nn.Linear(hidden_features, 1)
    with torch.no_grad():
        bound = math.sqrt(6.0 / hidden_features) / omega_0
        final.weight.uniform_(-bound, bound)
    layers.append(final)
    network = torch.nn.Sequential(*layers)
    return TorchRegressionAdapter(
        torch, network, task_spec, model_id=model_id, n_features=n_features,
        representation="point", learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        device=str(config.get("device", "cpu")),
    )
