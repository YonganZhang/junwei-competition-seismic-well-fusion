"""P5 gated adapter for the pinned Seismic Foundation Model source."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.facies._p5_common import build_legacy_source_model, capabilities_for


model_id = "sfm_base_facies"


def capabilities() -> Mapping[str, Any]:
    return capabilities_for(model_id)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_legacy_source_model(model_id, task_spec, **config)
