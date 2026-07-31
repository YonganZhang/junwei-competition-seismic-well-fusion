"""Compatibility shim for ``_models.fault.fault_raw_logistic``."""
from __future__ import annotations

from typing import Any

from ml_framework.model_registry import register_model
from _models.fault.fault_raw_logistic import FaultRawLogistic as _Canonical
from p4_contract import fault_task_spec


class FaultRawLogistic(_Canonical):
    def __init__(self, **config: Any) -> None:
        super().__init__(fault_task_spec(), **config)


@register_model("fault_raw_logistic")
def build_model(**config: Any) -> FaultRawLogistic:
    return FaultRawLogistic(**config)
