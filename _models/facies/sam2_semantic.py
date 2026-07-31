"""SAM 2.1 encoder with a trainable closed-set facies segmentation head."""
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


model_id = "sam2_semantic"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_image_2d"],
        "input_shape": "[B,1,H,W]",
        "output_shape": "[B,K,H,W]",
        "foundation_model": "facebook/sam2.1-hiera-base-plus",
        "conditioning": "spatial_prompt:none",
        "supports_missing_mask": False,
        "supports_uncertainty": False,
        "requires_pretrained_weight": True,
        "auto_download": False,
    }


def _make_network(torch: Any, backbone: Any, *, num_classes: int, freeze_encoder: bool) -> Any:
    nn = torch.nn
    functional = torch.nn.functional

    class SAM2SemanticFacies(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.freeze_encoder = bool(freeze_encoder)
            if self.freeze_encoder:
                for parameter in self.backbone.image_encoder.parameters():
                    parameter.requires_grad = False
            self.projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(256, 128, kernel_size=1),
                        nn.GroupNorm(16, 128),
                        nn.GELU(),
                    )
                    for _ in range(3)
                ]
            )
            self.semantic_head = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
                nn.Conv2d(128, num_classes, kernel_size=1),
            )

        def _encode(self, images: Any) -> list[Any]:
            context = torch.no_grad() if self.freeze_encoder else _NullContext()
            with context:
                output = self.backbone.image_encoder(images)
            maps = list(output["backbone_fpn"])[-3:]
            if len(maps) != 3 or any(item.ndim != 4 or item.shape[1] != 256 for item in maps):
                shapes = [tuple(item.shape) for item in maps]
                raise ValueError(f"unexpected SAM2 backbone feature shapes: {shapes}")
            return maps

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 4 or inputs.shape[1] != 1:
                raise ValueError("SAM2 facies input must be [B,1,H,W]")
            if not bool(torch.isfinite(inputs).all()):
                raise ValueError("SAM2 facies input contains non-finite values")
            original_size = tuple(int(value) for value in inputs.shape[-2:])
            # Contract: train-only z-score, fixed clip, then official SAM normalization.
            normalized = torch.clamp(inputs, -5.0, 5.0).add(5.0).div(10.0)
            normalized = functional.interpolate(
                normalized, size=(1024, 1024), mode="bilinear", align_corners=False
            ).repeat(1, 3, 1, 1)
            mean = normalized.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = normalized.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            maps = self._encode((normalized - mean) / std)
            target_size = tuple(int(value) for value in maps[0].shape[-2:])
            fused = None
            for projection, feature in zip(self.projections, maps):
                value = projection(feature)
                value = functional.interpolate(
                    value, size=target_size, mode="bilinear", align_corners=False
                )
                fused = value if fused is None else fused + value
            logits = self.semantic_head(fused / len(maps))
            return functional.interpolate(
                logits, size=original_size, mode="bilinear", align_corners=False
            )

    class _NullContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    return SAM2SemanticFacies()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    if task_spec.track_id != "facies":
        raise ValueError("SAM2 semantic adapter is restricted to the facies track")
    values = consume_config(
        config,
        required=("source_root", "checkpoint_path", "num_classes"),
        optional=("device", "freeze_encoder"),
    )
    num_classes = int(values["num_classes"])
    if num_classes not in {8, 10}:
        raise ValueError("facies SAM2 head must use the frozen 8- or 10-class label space")
    model_ref = route_model("facies")
    source_root = verify_git_source(Path(values["source_root"]), model_ref.source_revision)
    checkpoint = verify_checkpoint("facies", Path(values["checkpoint_path"]))
    insert_import_root(source_root, "sam2")
    import torch
    from sam2.build_sam import build_sam2

    device = str(values.get("device", "cpu"))
    backbone = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        ckpt_path=str(checkpoint),
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    return _make_network(
        torch,
        backbone,
        num_classes=num_classes,
        freeze_encoder=bool(values.get("freeze_encoder", True)),
    ).to(device)
