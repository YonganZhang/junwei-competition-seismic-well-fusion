"""Thin tsai InceptionTime adapter for approved well/production sequences."""
from __future__ import annotations

from typing import Any

from .p5_common import TorchModuleAdapter, output_dim, require_single_target

model_id = "inceptiontime"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression"], "input_modalities": ["sequence"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "sequence", "input_shape": "[B,C,T]"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        from tsai.models.InceptionTimePlus import InceptionTimePlus
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("tsai") from exc
    c_in = int(config.get("c_in", 1)); c_out = output_dim(task_spec)
    nf = int(config.get("nf", 8)); seq_len = config.get("seq_len", 64)
    device = str(config.get("device", "cpu"))
    factory = lambda: InceptionTimePlus(c_in=c_in, c_out=c_out, seq_len=seq_len, nf=nf, ks=16)
    return TorchModuleAdapter(task_spec, factory, input_key="sequence", device=device)
