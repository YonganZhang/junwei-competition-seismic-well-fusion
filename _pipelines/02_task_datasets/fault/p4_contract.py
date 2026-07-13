"""Strict P4 task and fixed array-to-envelope adapter for fault segmentation.

The Volve interpretation contains sparse fault-stick positives.  Absence of a
stick is not evidence of background.  This module therefore keeps unknown and
weak-proxy voxels outside ``valid_label_mask`` at the envelope boundary, before
any model or metric can see the data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _code.ml_framework.contracts import ModelBatch, TaskSpec  # noqa: E402


TARGET_NAME = "fault"
PROXY_SAMPLE_KINDS = frozenset({"non_fault", "weak_negative", "proxy_negative"})


def fault_task_spec() -> TaskSpec:
    """Return the frozen track-specific semantics on the shared outer contract."""

    return TaskSpec(
        track_id="fault",
        task_id="fault_stick_segmentation",
        task_type="binary",
        input_modalities=("seismic_amplitude",),
        targets=(TARGET_NAME,),
        units={TARGET_NAME: "binary_interpretation"},
        label_version="volve-fault-stick-sparse-v1",
        target_masks={TARGET_NAME: "valid_label_mask"},
        group_keys=("spatial_block_id",),
        target_transform={TARGET_NAME: "identity"},
        inverse_transform={TARGET_NAME: "identity"},
        train_loss={
            TARGET_NAME: {
                "default": "bce_with_logits",
                "candidates": (
                    "bce_with_logits",
                    "bce_with_logits_plus_dice",
                    "focal",
                    "tversky",
                ),
                "reduction": "valid_label_count_weighted",
            }
        },
        inference_transform={TARGET_NAME: "sigmoid"},
        threshold_policy={
            TARGET_NAME: "pooled_oof_only",
            "test_selection_forbidden": True,
        },
        calibration_policy={TARGET_NAME: "development_oof_only"},
        primary_metrics=("average_precision",),
        metric_directions={
            "average_precision": "maximize",
            "precision": "maximize",
            "recall": "maximize",
            "dice": "maximize",
            "iou": "maximize",
            "boundary_f1": "maximize",
            "component_fragmentation": "minimize",
        },
        secondary_metrics=("precision", "recall", "dice", "iou"),
        guardrail_metrics=("boundary_f1", "component_fragmentation"),
        spatial_buffer={
            "axes": ("inline", "crossline", "time_index"),
            "default_inline_radius": 8,
            "rule": "global_gap_between_contiguous_cv_blocks",
        },
        hpo={
            "optional_backend": "optuna",
            "direction": "maximize",
            "sanity_trials": (8, 12),
            "pilot_trials": (20, 30),
            "test_loader_allowed": False,
        },
        visualizer_id="fault_archived_volume",
        required_figures=(
            "input_gt_probability_confusion",
            "orthogonal_views",
            "pr_threshold",
            "boundary_components",
        ),
        input_whitelist=("seismic_amplitude",),
        forbidden_inputs=("fault", "valid_label_mask", "proxy_mask"),
        metadata={
            "io_shape": "[batch,depth, crossline,time]",
            "positive_semantics": "rasterized interpreted fault-stick voxel",
            "unknown_semantics": "unlabelled voxel; never an automatic negative",
            "negative_semantics": "allowed only inside completely audited annotation coverage",
            "proxy_semantics": "weak negative; regression/proxy metrics only",
            "legacy_audited_v2_role": "regression_evidence_only",
        },
    )


@dataclass(frozen=True)
class FaultMasks:
    """Disjoint scientific mask roles for one fixed target tensor."""

    positive: np.ndarray
    verified_negative: np.ndarray
    valid_label: np.ndarray
    unknown: np.ndarray
    proxy: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "positive": self.positive,
            "verified_negative": self.verified_negative,
            "valid_label": self.valid_label,
            "unknown": self.unknown,
            "proxy": self.proxy,
        }
        shapes = {np.asarray(value).shape for value in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"fault masks must have one shape, received {sorted(shapes)}")
        if any(np.asarray(value).dtype != np.bool_ for value in arrays.values()):
            raise TypeError("fault masks must be boolean")
        if np.any(self.positive & self.verified_negative):
            raise ValueError("positive and verified-negative masks overlap")
        if not np.array_equal(self.valid_label, self.positive | self.verified_negative):
            raise ValueError("valid_label_mask must equal positive | verified_negative")
        if not np.array_equal(self.unknown, ~self.valid_label):
            raise ValueError("unknown_mask must be the complement of valid_label_mask")
        if np.any(self.proxy & self.valid_label):
            raise ValueError("proxy_mask must remain outside valid_label_mask")


def _as_volume_batch(values: np.ndarray, *, name: str, batch_size: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, None, :, :]
    if array.ndim != 4:
        raise ValueError(f"{name} must have [B,D,H,W] or legacy [B,H,W] shape, got {array.shape}")
    if batch_size is not None and array.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {array.shape[0]} does not match {batch_size}")
    return array


def build_fault_masks(
    stick_labels: np.ndarray,
    sample_kinds: Sequence[str],
    *,
    verified_negative_mask: np.ndarray | None = None,
) -> FaultMasks:
    """Build masks without promoting an unlabelled voxel to a real negative."""

    positive = _as_volume_batch(stick_labels, name="stick_labels").astype(bool, copy=False)
    if len(sample_kinds) != positive.shape[0]:
        raise ValueError("sample_kinds length must match the label batch")
    if verified_negative_mask is None:
        verified_negative = np.zeros_like(positive, dtype=bool)
    else:
        verified_negative = _as_volume_batch(
            verified_negative_mask,
            name="verified_negative_mask",
            batch_size=positive.shape[0],
        ).astype(bool, copy=False)
    if np.any(positive & verified_negative):
        raise ValueError("a fault-stick positive cannot also be a verified negative")

    valid_label = positive | verified_negative
    unknown = ~valid_label
    proxy = np.zeros_like(positive, dtype=bool)
    for index, kind in enumerate(sample_kinds):
        if kind in PROXY_SAMPLE_KINDS:
            proxy[index] = unknown[index]
    return FaultMasks(
        positive=positive,
        verified_negative=verified_negative,
        valid_label=valid_label,
        unknown=unknown,
        proxy=proxy,
    )


def adapt_fault_arrays(
    amplitudes: np.ndarray,
    stick_labels: np.ndarray,
    positions: Sequence[Mapping[str, int | float]],
    sample_kinds: Sequence[str],
    *,
    verified_negative_mask: np.ndarray | None = None,
    spatial_block_ids: Sequence[str] | None = None,
) -> ModelBatch:
    """Adapt arrays to the fixed P4 ``ModelBatch`` envelope.

    Target zeros outside ``valid_label_mask`` are storage placeholders only.
    Consumers must use the target mask; the proxy/unknown masks are also made
    explicit in ``input_masks`` and metadata.
    """

    amplitude = _as_volume_batch(amplitudes, name="amplitudes")
    if not np.issubdtype(amplitude.dtype, np.number) or not np.isfinite(amplitude).all():
        raise ValueError("seismic amplitudes must be finite numeric values")
    amplitude = amplitude.astype(np.float32, copy=False)
    labels = _as_volume_batch(stick_labels, name="stick_labels", batch_size=amplitude.shape[0])
    if labels.shape != amplitude.shape:
        raise ValueError(f"amplitude/label shape mismatch: {amplitude.shape} vs {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("fault-stick labels must be binary")
    if len(positions) != amplitude.shape[0]:
        raise ValueError("positions length must match the array batch")
    masks = build_fault_masks(
        labels,
        sample_kinds,
        verified_negative_mask=verified_negative_mask,
    )

    required_coordinates = ("inline", "crossline", "time_index")
    coordinate_values: dict[str, np.ndarray] = {}
    sample_ids: list[str] = []
    for key in required_coordinates:
        if any(key not in position for position in positions):
            raise ValueError(f"every fault position must contain {key!r}")
        coordinate_values[key] = np.asarray([int(position[key]) for position in positions], dtype=np.int32)
    for index, position in enumerate(positions):
        sample_ids.append(
            "fault:"
            f"il{int(position['inline'])}:xl{int(position['crossline'])}:"
            f"t{int(position['time_index'])}:n{index}"
        )
    if spatial_block_ids is None:
        block_ids = [f"inline-{int(position['inline'])}" for position in positions]
    else:
        block_ids = list(spatial_block_ids)
        if len(block_ids) != amplitude.shape[0] or any(not block_id for block_id in block_ids):
            raise ValueError("spatial_block_ids must provide one non-empty ID per sample")

    targets = masks.positive.astype(np.float32)
    return ModelBatch(
        inputs={"seismic_amplitude": amplitude},
        targets={TARGET_NAME: targets},
        input_masks={
            "unknown_mask": masks.unknown,
            "proxy_mask": masks.proxy,
            "verified_negative_mask": masks.verified_negative,
        },
        target_masks={TARGET_NAME: masks.valid_label},
        sample_ids=sample_ids,
        groups={"spatial_block_id": block_ids},
        coordinates=coordinate_values,
        metadata={
            "sample_kinds": tuple(sample_kinds),
            "mask_semantics": {
                "positive": "fault-stick voxel",
                "valid_label_mask": "positive or audited verified negative only",
                "unknown_mask": "not valid; never an automatic negative",
                "proxy_mask": "weak negative; proxy/regression use only",
            },
        },
    )


def validate_fault_batch(batch: ModelBatch) -> None:
    """Fail loudly if a caller weakened the frozen fault mask contract."""

    if set(batch.inputs) != {"seismic_amplitude"}:
        raise ValueError("fault inputs must contain only seismic_amplitude")
    if batch.targets is None or set(batch.targets) != {TARGET_NAME}:
        raise ValueError("fault training batches must contain exactly the fault target")
    if set(batch.target_masks) != {TARGET_NAME}:
        raise ValueError("fault target_masks must contain exactly valid_label_mask for fault")
    required_masks = {"unknown_mask", "proxy_mask", "verified_negative_mask"}
    if not required_masks.issubset(batch.input_masks):
        raise ValueError(f"fault batch is missing masks: {sorted(required_masks - set(batch.input_masks))}")
    amplitude = np.asarray(batch.inputs["seismic_amplitude"])
    target = np.asarray(batch.targets[TARGET_NAME])
    valid = np.asarray(batch.target_masks[TARGET_NAME], dtype=bool)
    unknown = np.asarray(batch.input_masks["unknown_mask"], dtype=bool)
    proxy = np.asarray(batch.input_masks["proxy_mask"], dtype=bool)
    verified_negative = np.asarray(batch.input_masks["verified_negative_mask"], dtype=bool)
    if not (amplitude.shape == target.shape == valid.shape == unknown.shape == proxy.shape == verified_negative.shape):
        raise ValueError("fault input, target, and all masks must have identical [B,D,H,W] shape")
    positive = target.astype(bool)
    FaultMasks(positive, verified_negative, valid, unknown, proxy)
