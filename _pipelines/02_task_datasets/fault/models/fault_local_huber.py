"""Compatibility shim for ``_models.fault.fault_local_huber``."""
from __future__ import annotations

from typing import Any

from ml_framework.model_registry import register_model
from _models.fault.fault_local_huber import FaultLocalHuber as _Canonical
from p4_contract import fault_task_spec


class FaultLocalHuber(_Canonical):
    def __init__(self, **config: Any) -> None:
        super().__init__(fault_task_spec(), **config)


@register_model("fault_local_huber")
def build_model(**config: Any) -> FaultLocalHuber:
    return FaultLocalHuber(**config)
