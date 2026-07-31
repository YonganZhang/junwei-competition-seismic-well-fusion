"""Thin official RTDL FT-Transformer adapter."""
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


model_id = "ft_transformer_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "target_strategy": "shared backbone with independently masked target losses",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> TorchMultiTargetAdapter:
    validate_property_task_spec(task_spec)
    modules = require_model_dependencies(model_id)
    torch = modules["torch"]
    rtdl = modules["rtdl_revisiting_models"]
    seed = int(config.get("seed", 2693))
    seed_torch_runtime(torch, seed)
    n_features = int(config.get("n_features", 153))
    module = rtdl.FTTransformer(
        n_cont_features=n_features,
        cat_cardinalities=[],
        d_out=len(task_spec.targets),
        n_blocks=int(config.get("n_blocks", 2)),
        d_block=int(config.get("d_block", 32)),
        attention_n_heads=int(config.get("attention_n_heads", 4)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=float(config.get("ffn_d_hidden_multiplier", 4 / 3)),
        ffn_dropout=float(config.get("ffn_dropout", 0.1)),
        residual_dropout=float(config.get("residual_dropout", 0.0)),
    )

    def inputs(batch: ModelBatch) -> tuple[Any, ...]:
        return (feature_matrix(batch, "tabular").reshape(len(batch.sample_ids), -1), None)

    return TorchMultiTargetAdapter(
        model_id=model_id,
        task_spec=task_spec,
        torch_module=module,
        input_builder=inputs,
        torch=torch,
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        device=str(config.get("device", "cpu")),
        config={"seed": seed, "n_features": n_features, **config},
    )
