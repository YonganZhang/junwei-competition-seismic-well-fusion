"""Thin source-locked architecture adapter for nnU-Net v2 3d_fullres."""
from typing import Any
from _code.ml_framework.contracts import TaskSpec
from _models.fault._p5_adapter import build_locked_fault_model, locked_capabilities

model_id = "nnunet_v2_3d_fullres"


def capabilities() -> dict[str, Any]:
    return locked_capabilities(model_id)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_locked_fault_model(model_id, task_spec, **config)
