"""SAM-Med3D encoder adapter for automatic fault-volume segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import (
    consume_config,
    insert_import_root,
    route_model,
    verify_checkpoint,
    verify_git_source,
)


model_id = "sam_med3d_semantic"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["binary"],
        "input_modalities": ["seismic_volume_3d"],
        "input_shape": "[B,1,D,H,W]",
        "output_shape": "[B,1,D,H,W]",
        "foundation_model": "SAM-Med3D-turbo",
        "conditioning": "spatial_prompt:none",
        "unsupported_prompts": ["box_3d", "validation_or_test_gt_points"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
        "requires_pretrained_weight": True,
        "auto_download": False,
    }


def _make_network(torch: Any, backbone: Any, *, freeze_encoder: bool) -> Any:
    nn = torch.nn
    functional = torch.nn.functional

    class SAMMed3DFault(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = backbone.image_encoder
            self.freeze_encoder = bool(freeze_encoder)
            if self.freeze_encoder:
                for parameter in self.image_encoder.parameters():
                    parameter.requires_grad = False
            self.semantic_head = nn.Sequential(
                nn.Conv3d(384, 96, kernel_size=3, padding=1),
                nn.GroupNorm(12, 96),
                nn.GELU(),
                nn.Conv3d(96, 1, kernel_size=1),
            )

        def forward(self, inputs: Any, *, prompt_kind: str = "none") -> Any:
            if prompt_kind != "none":
                raise ValueError(
                    "automatic fault segmentation accepts prompt_kind='none' only; "
                    "interactive point completion must be a separate task"
                )
            if inputs.ndim != 5 or inputs.shape[1] != 1:
                raise ValueError("SAM-Med3D fault input must be [B,1,D,H,W]")
            if not bool(torch.isfinite(inputs).all()):
                raise ValueError("SAM-Med3D fault input contains non-finite values")
            original_size = tuple(int(value) for value in inputs.shape[-3:])
            volume = functional.interpolate(
                torch.clamp(inputs, -5.0, 5.0),
                size=(128, 128, 128),
                mode="trilinear",
                align_corners=False,
            )
            context = torch.no_grad() if self.freeze_encoder else _NullContext()
            with context:
                embedding = self.image_encoder(volume)
            if embedding.ndim != 5 or embedding.shape[1] != 384:
                raise ValueError(
                    f"unexpected SAM-Med3D embedding shape: {tuple(embedding.shape)}"
                )
            logits = self.semantic_head(embedding)
            return functional.interpolate(
                logits, size=original_size, mode="trilinear", align_corners=False
            )

    class _NullContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    return SAMMed3DFault()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    if task_spec.track_id != "fault":
        raise ValueError("SAM-Med3D semantic adapter is restricted to the fault track")
    values = consume_config(
        config,
        required=("source_root", "checkpoint_path"),
        optional=("device", "freeze_encoder"),
    )
    model_ref = route_model("fault")
    source_root = verify_git_source(Path(values["source_root"]), model_ref.source_revision)
    checkpoint = verify_checkpoint("fault", Path(values["checkpoint_path"]))
    insert_import_root(source_root, "segment_anything")
    import torch
    from segment_anything.build_sam3D import sam_model_registry3D

    backbone = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    backbone.load_state_dict(state, strict=True)
    device = str(values.get("device", "cpu"))
    return _make_network(
        torch,
        backbone,
        freeze_encoder=bool(values.get("freeze_encoder", True)),
    ).to(device)
