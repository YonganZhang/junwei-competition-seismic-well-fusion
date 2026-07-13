"""Shared safety gates and small wrappers for P5 facies model adapters."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from _code.ml_framework.contracts import TaskSpec


TASK_HEADS = {
    "facies_f3": ("f3-zenodo-1471548-ids-0-9-v1", 10),
    "facies_penobscot": ("penobscot-dataset-log-v3-ids-0-7-v1", 8),
}
SOURCE_LOCK_PATH = Path(__file__).with_name("p5_sources.json")


class P5AdapterSkip(RuntimeError):
    """A deliberate, machine-readable Stage-0/1 skip rather than a crash."""

    def __init__(self, code: str, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "reason": self.reason, "details": self.details}


@lru_cache(maxsize=1)
def source_locks() -> Mapping[str, Mapping[str, Any]]:
    payload = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "p5-source-lock-v1":
        raise ValueError("unsupported facies P5 source-lock schema")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("facies P5 source lock has no models")
    return models


def source_lock(model_id: str) -> Mapping[str, Any]:
    try:
        return source_locks()[model_id]
    except KeyError as exc:
        raise ValueError(f"missing source lock for {model_id!r}") from exc


def capabilities_for(model_id: str, *, volume: bool = False) -> dict[str, Any]:
    lock = source_lock(model_id)
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_volume" if volume else "seismic_amplitude_2d"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
        "output_contract": "raw_logits_BCDHW" if volume else "raw_logits_BCHW",
        "source_lock": SOURCE_LOCK_PATH.name,
        "dependency_group": lock["dependency_group"],
        "allowed_lanes": list(lock["allowed_lanes"]),
        "requires_contiguous_3d_blocks": volume,
    }


def _validate_task_and_head(task_spec: TaskSpec, config: dict[str, Any]) -> int:
    if task_spec.track_id != "facies" or task_spec.task_type != "multiclass":
        raise ValueError("P5 facies adapters require the facies multiclass TaskSpec")
    try:
        expected_label_version, expected_classes = TASK_HEADS[task_spec.task_id]
    except KeyError as exc:
        raise ValueError(f"unsupported facies task {task_spec.task_id!r}") from exc
    if task_spec.label_version != expected_label_version:
        raise ValueError(
            f"{task_spec.task_id} label version {task_spec.label_version!r} does not match "
            f"the locked version {expected_label_version!r}"
        )
    metadata_classes = int(task_spec.metadata.get("num_classes", 0))
    requested_classes = int(config.pop("num_classes", metadata_classes))
    if metadata_classes != expected_classes or requested_classes != expected_classes:
        raise ValueError(
            f"{task_spec.task_id} requires an independent {expected_classes}-class head; "
            f"metadata={metadata_classes}, requested={requested_classes}"
        )
    return expected_classes


def _validate_scratch_lane(model_id: str, config: dict[str, Any]) -> None:
    lock = source_lock(model_id)
    lane = str(config.pop("lane", "scratch"))
    weight_keys = {
        "weights",
        "weights_backbone",
        "encoder_weights",
        "pretrained",
        "checkpoint",
        "checkpoint_path",
    }
    supplied_weights = sorted(key for key in weight_keys if config.get(key) not in (None, False))
    if lane != "scratch":
        weights = dict(lock["weights"])
        raise P5AdapterSkip(
            "weight_lane_not_approved",
            f"{model_id} pretrained lane is blocked until license, URL and SHA-256 are frozen",
            requested_lane=lane,
            allowed_lanes=list(lock["allowed_lanes"]),
            weight_status=weights.get("status"),
            weight_license=weights.get("license"),
            weight_url=weights.get("url"),
            weight_sha256=weights.get("sha256"),
        )
    if supplied_weights:
        raise P5AdapterSkip(
            "scratch_lane_received_weights",
            f"scratch lane cannot accept weight-bearing arguments: {supplied_weights}",
            supplied_weight_keys=supplied_weights,
        )
    for key in weight_keys:
        config.pop(key, None)


def _require_locked_runtime(model_id: str) -> Any:
    lock = source_lock(model_id)
    if lock["adapter_mode"] == "legacy_source_port_required":
        raise P5AdapterSkip(
            "legacy_source_port_not_available",
            f"{model_id} needs its pinned legacy upstream architecture port; the shared environment "
            "does not provide an importable package and substituting a same-name local network is forbidden",
            dependency_group=lock["dependency_group"],
            source_url=lock["source_url"],
            source_commit=lock["source_commit"],
        )
    distribution = lock.get("distribution")
    module_name = lock.get("import_module")
    if not distribution or not module_name:
        raise P5AdapterSkip(
            "dependency_unavailable",
            f"{model_id} has no importable locked runtime in the shared environment",
            dependency_group=lock["dependency_group"],
        )
    try:
        observed = importlib.metadata.version(str(distribution))
    except importlib.metadata.PackageNotFoundError as exc:
        raise P5AdapterSkip(
            "dependency_unavailable",
            f"required distribution {distribution!r} is not installed",
            distribution=distribution,
            allowed_versions=list(lock["allowed_versions"]),
        ) from exc
    if observed not in lock["allowed_versions"]:
        raise P5AdapterSkip(
            "runtime_version_mismatch",
            f"{distribution}=={observed} is outside the locked versions {lock['allowed_versions']}",
            distribution=distribution,
            observed_version=observed,
            allowed_versions=list(lock["allowed_versions"]),
        )
    try:
        return importlib.import_module(str(module_name))
    except Exception as exc:
        raise P5AdapterSkip(
            "dependency_import_failed",
            f"{module_name} import failed: {type(exc).__name__}: {exc}",
            distribution=distribution,
            observed_version=observed,
        ) from exc


def _prepare_build(model_id: str, task_spec: TaskSpec, config: Mapping[str, Any]) -> tuple[int, dict[str, Any], Any]:
    mutable = dict(config)
    classes = _validate_task_and_head(task_spec, mutable)
    _validate_scratch_lane(model_id, mutable)
    runtime = _require_locked_runtime(model_id)
    return classes, mutable, runtime


def build_smp_model(
    model_id: str,
    architecture: str,
    task_spec: TaskSpec,
    **config: Any,
) -> nn.Module:
    classes, mutable, smp = _prepare_build(model_id, task_spec, config)
    encoder_name = str(mutable.pop("encoder_name", "resnet18"))
    if encoder_name != "resnet18":
        raise ValueError(f"{model_id} source lock requires encoder_name='resnet18'")
    if mutable:
        raise ValueError(f"unsupported {model_id} config: {sorted(mutable)}")
    constructor = getattr(smp, architecture)
    return constructor(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=classes,
        activation=None,
        aux_params=None,
    )


class TorchvisionOutputAdapter(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_2d_inputs(inputs)
        output = self.network(inputs.repeat(1, 3, 1, 1))
        logits = output["out"]
        if logits.shape[-2:] != inputs.shape[-2:]:
            logits = F.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def build_torchvision_lraspp(task_spec: TaskSpec, **config: Any) -> nn.Module:
    model_id = "torchvision_lraspp_mbv3"
    classes, mutable, _ = _prepare_build(model_id, task_spec, config)
    if mutable:
        raise ValueError(f"unsupported {model_id} config: {sorted(mutable)}")
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    network = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=classes,
    )
    return TorchvisionOutputAdapter(network)


class SegformerOutputAdapter(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_2d_inputs(inputs)
        output = self.network(pixel_values=inputs.repeat(1, 3, 1, 1)).logits
        return F.interpolate(output, size=inputs.shape[-2:], mode="bilinear", align_corners=False)


def build_segformer_b0(task_spec: TaskSpec, **config: Any) -> nn.Module:
    model_id = "hf_segformer_b0"
    classes, mutable, _ = _prepare_build(model_id, task_spec, config)
    if mutable:
        raise ValueError(f"unsupported {model_id} config: {sorted(mutable)}")
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    configuration = SegformerConfig(
        num_channels=3,
        num_labels=classes,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 160, 256],
        decoder_hidden_size=256,
        num_attention_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        semantic_loss_ignore_index=255,
    )
    return SegformerOutputAdapter(SegformerForSemanticSegmentation(configuration))


def build_monai_unet3d(task_spec: TaskSpec, **config: Any) -> nn.Module:
    model_id = "monai_unet3d"
    classes, mutable, _ = _prepare_build(model_id, task_spec, config)
    allow_3d_contract = bool(mutable.pop("allow_3d_contract", False))
    if not allow_3d_contract:
        raise P5AdapterSkip(
            "contiguous_3d_development_blocks_unavailable",
            "the frozen facies ModelBatch is 2-D; a legal contiguous same-core 3-D block adapter "
            "must be frozen before MONAI U-Net can consume real development data",
            current_input_modality=list(task_spec.input_modalities),
            required_input="B,1,D,H,W with every inline inside one development core",
        )
    channels = tuple(int(value) for value in mutable.pop("channels", (8, 16, 32)))
    strides = tuple(int(value) for value in mutable.pop("strides", (2, 2)))
    num_res_units = int(mutable.pop("num_res_units", 1))
    if mutable:
        raise ValueError(f"unsupported {model_id} config: {sorted(mutable)}")
    if len(channels) != len(strides) + 1:
        raise ValueError("MONAI channels must contain exactly one more item than strides")
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=classes,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )


def build_legacy_source_model(model_id: str, task_spec: TaskSpec, **config: Any) -> nn.Module:
    _prepare_build(model_id, task_spec, config)
    raise AssertionError("legacy source gate should always produce a structured skip")


def _validate_2d_inputs(inputs: torch.Tensor) -> None:
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError(f"expected seismic input [B,1,H,W], got {tuple(inputs.shape)}")
