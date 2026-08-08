"""Scratch-only MONAI DenseNet3D adapter for real seismic patches."""
from __future__ import annotations

from typing import Any

import numpy as np

from _code.ml_framework.contracts import ModelBatch, TaskSpec
from _models.property._p5_common import (
    TorchMultiTargetAdapter,
    feature_matrix,
    require_model_dependencies,
    seed_torch_runtime,
    validate_property_task_spec,
)


model_id = "monai_densenet3d_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["seismic_patch"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
        "supported_losses": ["mse", "mae", "huber"],
        "supported_output_activations": ["identity", "bounded"],
        "pretrained_lane": "scratch_only",
        "cuda_determinism": "strict_with_fixed_pooling_replacements",
        "target_strategy": "shared seismic backbone with independently masked target losses",
    }


def _padded_seismic(batch: ModelBatch) -> tuple[np.ndarray]:
    seismic = feature_matrix(batch, "seismic_patch")
    if seismic.shape[1:] != (3, 3, 9):
        raise ValueError(f"expected [N,3,3,9] seismic patches, got {seismic.shape}")
    volume = seismic[:, None, :, :, :]
    # DenseNet's fixed stem downsamples twice.  Pad deterministically to the
    # smallest conservative 16^3 smoke volume; the manifest records this fact.
    padded = np.pad(volume, ((0, 0), (0, 0), (6, 7), (6, 7), (3, 4)), mode="constant")
    return (padded,)


def _replace_nondeterministic_pool3d(module: Any, torch: Any) -> dict[str, int]:
    """Replace CUDA-nondeterministic 3D pooling operations.

    PyTorch 2.12 has no deterministic CUDA backward for ``avg_pool3d`` or
    ``max_pool3d``.  DenseNet's transition average pooling is replaced by an
    exactly equivalent fixed depthwise convolution.  Its stem max pooling is
    replaced by a documented fixed depthwise averaging downsampler.  The
    final output-size-one adaptive average is a direct spatial mean.
    """

    class DeterministicFixedPool3d(torch.nn.Module):
        def __init__(self, original: Any, *, exact_average: bool) -> None:
            super().__init__()
            kernel = original.kernel_size
            stride = original.stride if original.stride is not None else kernel
            padding = original.padding
            self.kernel_size = (kernel,) * 3 if isinstance(kernel, int) else tuple(kernel)
            self.stride = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
            self.padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
            if original.ceil_mode:
                raise ValueError("ceil-mode 3D pooling has no deterministic Stage-1 replacement")
            if isinstance(original, torch.nn.AvgPool3d):
                if (
                    not original.count_include_pad
                    or original.divisor_override is not None
                    or any(self.padding)
                    or self.stride != self.kernel_size
                ):
                    raise ValueError("unsupported MONAI AvgPool3d deterministic replacement")
            elif original.dilation != 1 or original.return_indices:
                raise ValueError("unsupported MONAI MaxPool3d deterministic replacement")
            self.exact_average = exact_average
            self.divisor = float(np.prod(self.kernel_size))

        def forward(self, values: Any) -> Any:
            channels = int(values.shape[1])
            weight = values.new_ones((channels, 1, *self.kernel_size)) / self.divisor
            return torch.nn.functional.conv3d(
                values,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=channels,
            )

    class DeterministicAdaptiveAverage3d(torch.nn.Module):
        def forward(self, values: Any) -> Any:
            return values.mean(dim=(2, 3, 4), keepdim=True)

    replaced = {"avg_pool3d": 0, "max_pool3d": 0, "adaptive_avg_pool3d": 0}
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.AvgPool3d):
            setattr(module, name, DeterministicFixedPool3d(child, exact_average=True))
            replaced["avg_pool3d"] += 1
        elif isinstance(child, torch.nn.MaxPool3d):
            setattr(module, name, DeterministicFixedPool3d(child, exact_average=False))
            replaced["max_pool3d"] += 1
        elif isinstance(child, torch.nn.AdaptiveAvgPool3d):
            if child.output_size not in {1, (1, 1, 1)}:
                raise ValueError("only output-size-one adaptive pooling is supported")
            setattr(module, name, DeterministicAdaptiveAverage3d())
            replaced["adaptive_avg_pool3d"] += 1
        else:
            nested = _replace_nondeterministic_pool3d(child, torch)
            for key, count in nested.items():
                replaced[key] += count
    return replaced


def build_model(task_spec: TaskSpec, **config: Any) -> TorchMultiTargetAdapter:
    validate_property_task_spec(task_spec)
    modules = require_model_dependencies(model_id)
    torch = modules["torch"]
    monai = modules["monai"]
    seed = int(config.get("seed", 2693))
    seed_torch_runtime(torch, seed)
    module = monai.networks.nets.DenseNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(task_spec.targets),
        init_features=int(config.get("init_features", 8)),
        growth_rate=int(config.get("growth_rate", 8)),
        block_config=tuple(config.get("block_config", (2, 2))),
        bn_size=int(config.get("bn_size", 2)),
        dropout_prob=float(config.get("dropout_prob", 0.0)),
    )
    deterministic_pool_replacements = _replace_nondeterministic_pool3d(module, torch)
    if deterministic_pool_replacements["avg_pool3d"] <= 0:
        raise RuntimeError("MONAI DenseNet topology did not expose an AvgPool3d transition")
    if deterministic_pool_replacements["max_pool3d"] <= 0:
        raise RuntimeError("MONAI DenseNet topology did not expose its MaxPool3d stem")
    return TorchMultiTargetAdapter(
        model_id=model_id,
        task_spec=task_spec,
        torch_module=module,
        input_builder=_padded_seismic,
        torch=torch,
        learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        device=str(config.get("device", "cpu")),
        config={
            "seed": seed,
            "padding": "3x3x9_to_16x16x16_zero",
            "deterministic_pool3d_replacements": deterministic_pool_replacements,
            **config,
        },
    )
