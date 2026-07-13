"""P5 thin adapter: SMP U-Net with a scratch ResNet18 encoder."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.facies._p5_common import build_smp_model, capabilities_for


model_id = "smp_unet_r18"


def capabilities() -> Mapping[str, Any]:
    return capabilities_for(model_id)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_smp_model(model_id, "Unet", task_spec, **config)
