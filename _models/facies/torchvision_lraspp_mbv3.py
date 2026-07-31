"""P5 thin adapter: scratch TorchVision LR-ASPP MobileNetV3."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.facies._p5_common import build_torchvision_lraspp, capabilities_for


model_id = "torchvision_lraspp_mbv3"


def capabilities() -> Mapping[str, Any]:
    return capabilities_for(model_id)


def build_model(task_spec: TaskSpec, **config: Any):
    return build_torchvision_lraspp(task_spec, **config)
