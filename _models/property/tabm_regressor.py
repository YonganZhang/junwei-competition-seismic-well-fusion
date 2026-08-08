"""Thin official TabM adapter; no upstream source is vendored."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import ModelBatch, TaskSpec
from _models.property._p5_common import (
    TorchMultiTargetAdapter,
    feature_matrix,
    require_model_dependencies,
    seed_torch_runtime,
    validate_property_task_spec,
)


model_id = "tabm_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "supported_losses": ["mse", "mae", "huber"],
        "supported_output_activations": ["identity", "bounded"],
        "target_strategy": "shared backbone with independently masked target losses",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> TorchMultiTargetAdapter:
    validate_property_task_spec(task_spec)
    modules = require_model_dependencies(model_id)
    torch = modules["torch"]
    tabm = modules["tabm"]
    seed = int(config.get("seed", 2693))
    seed_torch_runtime(torch, seed)
    n_features = int(config.get("n_features", 153))
    module = tabm.TabM.make(
        n_num_features=n_features,
        d_out=len(task_spec.targets),
        arch_type=str(config.get("arch_type", "tabm-mini")),
        k=int(config.get("k", 4)),
        n_blocks=int(config.get("n_blocks", 2)),
        d_block=int(config.get("d_block", 64)),
    )

    def inputs(batch: ModelBatch) -> tuple[Any, ...]:
        return (feature_matrix(batch, "tabular").reshape(len(batch.sample_ids), -1),)

    return TorchMultiTargetAdapter(
        model_id=model_id,
        task_spec=task_spec,
        torch_module=module,
        input_builder=inputs,
        torch=torch,
        learning_rate=float(config.get("learning_rate", 0.002)),
        weight_decay=float(config.get("weight_decay", 3e-4)),
        device=str(config.get("device", "cpu")),
        config={"seed": seed, "n_features": n_features, **config},
    )
