"""P5 thin adapter: scratch MONAI 3-D U-Net with a strict block gate."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.facies._p5_common import build_monai_unet3d, capabilities_for


model_id = "monai_unet3d"


def capabilities() -> Mapping[str, Any]:
    return capabilities_for(model_id, volume=True)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_monai_unet3d(task_spec, **config)
