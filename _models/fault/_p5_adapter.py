"""Shared mechanics for thin, source-locked P5 fault model adapters."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec
from _models.fault.p5_lock import require_runtime_ready, source_lock


TARGET_NAME = "fault"


def _validate_fault_batch(batch: ModelBatch) -> None:
    contract = importlib.import_module("_pipelines.02_task_datasets.fault.p4_contract")
    contract.validate_fault_batch(batch)


def locked_capabilities(model_id: str) -> dict[str, Any]:
    record = source_lock(model_id)
    return {
        "task_types": ["binary"],
        "input_modalities": ["seismic_amplitude"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "supports_target_mask": True,
        "trainable": bool(record["adapter"]["trainable"]),
        "source_revision": record["source"]["revision"],
        "code_license": record["source"]["license_spdx"],
    }


def _validate_task_spec(task_spec: TaskSpec) -> None:
    if task_spec.track_id != "fault" or task_spec.task_type != "binary":
        raise ValueError("P5 fault adapters require the frozen binary fault TaskSpec")
    if tuple(task_spec.targets) != (TARGET_NAME,):
        raise ValueError("P5 fault adapters require exactly the fault target")
    if tuple(task_spec.input_modalities) != ("seismic_amplitude",):
        raise ValueError("P5 fault adapters accept only seismic_amplitude")
    if task_spec.inference_transform.get(TARGET_NAME) != "sigmoid":
        raise ValueError("fault inference transform must remain sigmoid")


def _round_up(value: int, minimum: int, multiple: int) -> int:
    target = max(value, minimum)
    return int(math.ceil(target / multiple) * multiple)


class TorchFaultAdapter(nn.Module):
    """Map frozen NumPy ``ModelBatch`` volumes to one-channel raw Torch logits."""

    def __init__(self, model_id: str, task_spec: TaskSpec, network: nn.Module) -> None:
        super().__init__()
        _validate_task_spec(task_spec)
        record = source_lock(model_id)
        self.model_id = model_id
        self.task_spec = task_spec
        self.network = network
        self.minimum_shape = tuple(int(v) for v in record["adapter"]["minimum_shape"])
        self.shape_multiple = tuple(int(v) for v in record["adapter"]["shape_multiple"])

    def _network_input(self, batch: ModelBatch) -> tuple[torch.Tensor, tuple[int, int, int]]:
        _validate_fault_batch(batch)
        amplitude = np.asarray(batch.inputs["seismic_amplitude"], dtype=np.float32)
        if amplitude.ndim != 4 or not np.isfinite(amplitude).all():
            raise ValueError("fault amplitude must be finite [B,D,H,W]")
        device = next(self.network.parameters()).device
        values = torch.as_tensor(amplitude, dtype=torch.float32, device=device).unsqueeze(1)
        original = tuple(int(v) for v in values.shape[-3:])
        padded = tuple(
            _round_up(value, minimum, multiple)
            for value, minimum, multiple in zip(original, self.minimum_shape, self.shape_multiple)
        )
        pad_d, pad_h, pad_w = (padded[i] - original[i] for i in range(3))
        if pad_d or pad_h or pad_w:
            values = functional.pad(values, (0, pad_w, 0, pad_h, 0, pad_d))
        return values, original

    def forward(self, batch: ModelBatch) -> ModelOutput:
        values, original = self._network_input(batch)
        logits = self.network(values)
        if isinstance(logits, (tuple, list)):
            if len(logits) != 1:
                raise RuntimeError(f"{self.model_id} returned ambiguous multi-output logits")
            logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 5 or logits.shape[1] != 1:
            raise RuntimeError(f"{self.model_id} must return [B,1,D,H,W] raw logits")
        if tuple(logits.shape[-3:]) != tuple(values.shape[-3:]):
            logits = functional.interpolate(
                logits,
                size=values.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        logits = logits[:, 0, : original[0], : original[1], : original[2]]
        expected = np.asarray(batch.inputs["seismic_amplitude"]).shape
        if tuple(logits.shape) != tuple(expected) or not torch.isfinite(logits).all():
            raise RuntimeError(f"{self.model_id} produced invalid fault logits")
        return ModelOutput(
            raw={TARGET_NAME: logits},
            transformed={TARGET_NAME: torch.sigmoid(logits)},
            aux={
                "model_id": self.model_id,
                "source_revision": source_lock(self.model_id)["source"]["revision"],
                "test_accessed": False,
            },
        )

    def masked_loss(self, batch: ModelBatch, output: ModelOutput | None = None) -> torch.Tensor:
        _validate_fault_batch(batch)
        if output is None:
            output = self(batch)
        logits = output.raw[TARGET_NAME]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("fault raw output must be a Torch tensor")
        target = torch.as_tensor(
            np.asarray(batch.targets[TARGET_NAME], dtype=np.float32),
            dtype=logits.dtype,
            device=logits.device,
        )
        valid = torch.as_tensor(
            np.asarray(batch.target_masks[TARGET_NAME], dtype=bool),
            dtype=torch.bool,
            device=logits.device,
        )
        truth = target[valid]
        if truth.numel() == 0 or not torch.any(truth == 1) or not torch.any(truth == 0):
            raise RuntimeError(
                "official masked fault loss requires audited positive and verified-negative labels"
            )
        loss = functional.binary_cross_entropy_with_logits(logits[valid], truth)
        if not torch.isfinite(loss):
            raise RuntimeError("masked fault loss is non-finite")
        return loss


class FaultNetTorchScriptAdapter(TorchFaultAdapter):
    """Inference-only adapter for an approved, hash-locked FaultNet TorchScript file."""

    def masked_loss(self, batch: ModelBatch, output: ModelOutput | None = None) -> torch.Tensor:
        del batch, output
        raise RuntimeError("faultnet_md is an inference-only anchor; training/backward is unavailable")


def _build_monai(builder: str, config: dict[str, Any]) -> nn.Module:
    from monai.networks.nets import DynUNet, SegResNet, SwinUNETR, VNet

    if builder == "segresnet":
        return SegResNet(
            spatial_dims=3,
            init_filters=int(config.pop("init_filters", 8)),
            in_channels=1,
            out_channels=1,
            blocks_down=tuple(config.pop("blocks_down", (1, 1, 1))),
            blocks_up=tuple(config.pop("blocks_up", (1, 1))),
            dropout_prob=None,
            **config,
        )
    if builder == "dynunet":
        return DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=config.pop("kernel_size", [3, 3, 3]),
            strides=config.pop("strides", [1, 2, 2]),
            upsample_kernel_size=config.pop("upsample_kernel_size", [2, 2]),
            filters=config.pop("filters", [8, 16, 32]),
            norm_name=config.pop("norm_name", ("GROUP", {"num_groups": 4})),
            deep_supervision=False,
            **config,
        )
    if builder == "vnet":
        return VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            dropout_prob_down=float(config.pop("dropout_prob_down", 0.0)),
            dropout_prob_up=tuple(config.pop("dropout_prob_up", (0.0, 0.0))),
            **config,
        )
    if builder == "swinunetr":
        return SwinUNETR(
            in_channels=1,
            out_channels=1,
            depths=tuple(config.pop("depths", (1, 1, 1, 1))),
            num_heads=tuple(config.pop("num_heads", (3, 6, 12, 24))),
            feature_size=int(config.pop("feature_size", 12)),
            norm_name=config.pop("norm_name", ("GROUP", {"num_groups": 3})),
            use_checkpoint=bool(config.pop("use_checkpoint", False)),
            spatial_dims=3,
            **config,
        )
    raise ValueError(f"unsupported MONAI builder {builder!r}")


def _build_nnunet() -> nn.Module:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=1,
        n_stages=4,
        features_per_stage=[8, 16, 32, 64],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2],
        num_classes=1,
        n_conv_per_stage_decoder=[2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )


def _build_pytorch3dunet() -> nn.Module:
    from pytorch3dunet.unet3d.model import UNet3D

    return UNet3D(
        in_channels=1,
        out_channels=1,
        final_sigmoid=False,
        f_maps=16,
        num_levels=4,
        is_segmentation=False,
    )


def build_locked_fault_model(model_id: str, task_spec: TaskSpec, **config: Any) -> nn.Module:
    """Build only the exact locked source; never substitute a same-name implementation."""

    source_root_raw = config.pop("source_root", None)
    weight_path_raw = config.pop("weight_path", None)
    source_root = Path(source_root_raw) if source_root_raw is not None else None
    weight_path = Path(weight_path_raw) if weight_path_raw is not None else None
    require_runtime_ready(model_id, source_root=source_root, weight_path=weight_path)
    _validate_task_spec(task_spec)
    record = source_lock(model_id)
    adapter = dict(record["adapter"])
    backend = adapter["backend"]
    builder = adapter["builder"]

    torch.manual_seed(int(config.pop("seed", 2693)))
    if backend == "monai":
        network = _build_monai(builder, dict(config))
        return TorchFaultAdapter(model_id, task_spec, network)
    if backend == "nnunetv2":
        if config:
            raise ValueError(f"unsupported nnU-Net smoke config keys: {sorted(config)}")
        return TorchFaultAdapter(model_id, task_spec, _build_nnunet())
    if backend == "pytorch3dunet":
        if config:
            raise ValueError(f"unsupported pytorch3dunet smoke config keys: {sorted(config)}")
        return TorchFaultAdapter(model_id, task_spec, _build_pytorch3dunet())
    if backend == "torchscript_probability":
        if weight_path is None:
            raise AssertionError("runtime gate allowed FaultNet without its locked weight")
        network = torch.jit.load(str(weight_path), map_location="cpu")
        return FaultNetTorchScriptAdapter(model_id, task_spec, network)
    if backend in {"tensorflow_keras", "pytorch_locked_checkout"}:
        raise RuntimeError(
            f"{model_id} exact checkout passed static gates but requires an isolated upstream adapter; "
            "the shared torch-common execution intentionally does not vendor or rewrite it"
        )
    raise ValueError(f"unsupported locked backend {backend!r}")
