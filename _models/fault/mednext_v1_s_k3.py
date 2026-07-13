"""Thin fail-closed declaration for the exact MedNeXt S/K3 checkout."""
from typing import Any
from _code.ml_framework.contracts import TaskSpec
from _models.fault._p5_adapter import build_locked_fault_model, locked_capabilities

model_id = "mednext_v1_s_k3"


def capabilities() -> dict[str, Any]:
    return locked_capabilities(model_id)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_locked_fault_model(model_id, task_spec, **config)
