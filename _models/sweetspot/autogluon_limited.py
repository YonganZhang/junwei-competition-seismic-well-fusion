"""Fail-closed AutoGluon adapter; Stage 1 requires an explicit scratch output root."""
from __future__ import annotations

from typing import Any

from .p5_common import AdapterSkip, require_single_target

model_id = "autogluon_limited"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression"], "input_modalities": ["tabular", "time_series"], "supports_missing_mask": True, "supports_uncertainty": True, "stage1_input_key": "tabular", "external_weights": "forbidden"}


class AutoGluonLimitedAdapter:
    def __init__(self, task_spec, output_root=None):
        self.task_spec = task_spec
        self.output_root = output_root

    def stage1_smoke(self, inputs, target, target_mask, *, seed):
        del inputs, target, target_mask, seed
        if self.output_root is None:
            raise AdapterSkip("explicit_scratch_root_required", "AutoGluon writes model artifacts; Stage 1 requires an explicit disposable output root")
        raise AdapterSkip("autogluon_stage1_not_enabled", "AutoGluon smoke remains disabled until an approved development manifest and time budget are supplied")


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        import autogluon.tabular  # noqa: F401
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("autogluon.tabular") from exc
    return AutoGluonLimitedAdapter(task_spec, output_root=config.get("output_root"))
