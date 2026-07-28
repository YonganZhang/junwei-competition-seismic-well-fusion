"""MOMENT foundation adapter for measured-depth lithofacies windows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import (
    consume_config,
    route_model,
    verify_checkpoint,
)


model_id = "moment_depth"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["measured_depth_multivariate_window"],
        "input_shape": "[B,35,33]",
        "output_shape": "[B,9]",
        "foundation_model": "AutonLab/MOMENT-1-base",
        "conditioning": "depth_window",
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "requires_pretrained_weight": True,
        "auto_download": False,
    }


def _make_network(torch: Any, pipeline: Any) -> Any:
    nn = torch.nn
    functional = torch.nn.functional

    class MOMENTDepthClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pipeline = pipeline

        def forward(self, inputs: Any, input_mask: Any | None = None) -> Any:
            if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (35, 33):
                raise ValueError("MOMENT lithofacies input must be [B,35,33]")
            if not bool(torch.isfinite(inputs).all()):
                raise ValueError("MOMENT lithofacies input contains non-finite values")
            values = functional.interpolate(inputs, size=512, mode="linear", align_corners=False)
            if input_mask is None:
                mask = torch.ones(
                    (inputs.shape[0], 33), dtype=torch.float32, device=inputs.device
                )
            else:
                if input_mask.shape != (inputs.shape[0], 33):
                    raise ValueError("MOMENT input_mask must be [B,33]")
                mask = input_mask.to(dtype=torch.float32)
            mask = functional.interpolate(
                mask[:, None, :], size=512, mode="nearest"
            )[:, 0, :].to(dtype=torch.long)
            output = self.pipeline(x_enc=values, input_mask=mask)
            logits = getattr(output, "logits", None)
            if logits is None or logits.shape != (inputs.shape[0], 9):
                shape = None if logits is None else tuple(logits.shape)
                raise ValueError(f"unexpected MOMENT classification logits: {shape}")
            return logits

    return MOMENTDepthClassifier()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    if task_spec.track_id != "lithofacies":
        raise ValueError("MOMENT depth adapter is restricted to the lithofacies track")
    values = consume_config(
        config,
        required=("snapshot_path",),
        optional=("device", "freeze_encoder", "freeze_embedder"),
    )
    snapshot = Path(values["snapshot_path"]).resolve()
    verify_checkpoint("lithofacies", snapshot / "model.safetensors")
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError("MOMENT local snapshot is missing config.json")
    import torch
    from momentfm import MOMENTPipeline

    pipeline = MOMENTPipeline.from_pretrained(
        str(snapshot),
        local_files_only=True,
        model_kwargs={
            "task_name": "classification",
            "n_channels": 35,
            "num_class": 9,
            "freeze_encoder": bool(values.get("freeze_encoder", True)),
            "freeze_embedder": bool(values.get("freeze_embedder", True)),
            "freeze_head": False,
        },
    )
    pipeline.init()
    device = str(values.get("device", "cpu"))
    return _make_network(torch, pipeline).to(device)
