"""Source-locked boundary for the SEG joint spatial-context TCN."""
from __future__ import annotations

from typing import Any

from .p5_common import AdapterSkip, require_single_target

model_id = "seg_spatial_tcn"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "regression"], "input_modalities": ["spatial_trace_sequence"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "spatial_trace", "input_shape": "[B,1,T,7]"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    if config.get("source_root") is None:
        raise AdapterSkip("locked_source_checkout_unavailable", "SEG spatial TCN needs the author source at the locked commit; the old hard-coded CUDA training loop is not copied")
    raise AdapterSkip("locked_source_adapter_pending", "the locked SEG checkout was not authorized for download in this run")
