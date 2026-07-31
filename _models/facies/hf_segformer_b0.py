"""P5 thin adapter: scratch Hugging Face SegFormer MiT-B0."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.facies._p5_common import build_segformer_b0, capabilities_for


model_id = "hf_segformer_b0"


def capabilities() -> Mapping[str, Any]:
    return capabilities_for(model_id)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_segformer_b0(task_spec, **config)
