"""Source-locked PatchTST adapter boundary (author checkout required)."""
from __future__ import annotations

from typing import Any

from .p5_common import AdapterSkip, require_single_target

model_id = "patchtst"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "regression"], "input_modalities": ["sequence"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "sequence", "input_shape": "[B,T,C]"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    source_root = config.get("source_root")
    if source_root is None:
        raise AdapterSkip("locked_source_checkout_unavailable", "PatchTST requires the audited author checkout at the locked commit; Transformers or another implementation cannot substitute")
    raise AdapterSkip("locked_source_adapter_pending", "the locked PatchTST checkout was not authorized for download in this run")
