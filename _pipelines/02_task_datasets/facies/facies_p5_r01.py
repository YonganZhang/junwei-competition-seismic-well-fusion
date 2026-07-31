#!/usr/bin/env python3
"""P5.1 R0/R1 development-only full-section leakage protocol for facies.

This runner deliberately has no processed-root, test, holdout, or lifecycle
argument.  It reads only explicitly allowed development inlines from the raw
F3/Penobscot containers, reconstructs complete validation sections with a
deterministic sliding window, and compares one fixed linear-pixel model under:

* a diagnostic random split made after overlapping windows are extracted; and
* a legal block split whose guard is applied before any window is created.

The diagnostic lane is intentionally contaminated and is never rankable.  The
outer reusable holdout is neither indexed as a sample set nor read as arrays.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import matplotlib
import numpy as np
import torch
from PIL import Image
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.preprocess import (  # noqa: E402
    NormStats,
    denoise_identity,
    denormalize,
    fit_zscore,
    normalize,
)
from _code.ml_framework.seeding import derive_seed, seed_everything  # noqa: E402
from p4_data import inverse_sqrt_class_weights  # noqa: E402
from p4_metrics import confusion_matrix  # noqa: E402
from p4_tasks import (  # noqa: E402
    EXPECTED_OUTER_SPLITS,
    INTERNAL_BUFFER_GROUPS,
    LABEL_VERSIONS,
    TASK_IDS,
    get_task_spec,
)
from pipeline_contract import (  # noqa: E402
    get_task_schema,
    ordered_spatial_split,
    segmentation_metrics_from_confusion,
    validate_label_array,
)


ROOT_SEED = 2693
MODEL_ID = "facies_linear_pixel"
SCHEMA_VERSION = "facies-p5.1-r01-v1"
DEFAULT_OUTPUT_ROOT = TRACK_ROOT / "_outputs" / "p5_r01"


class ProtocolBlocked(RuntimeError):
    """A scientific/data contract prevents a legal numeric result."""


@dataclass(frozen=True)
class R01Config:
    window_size: int = 128
    stride: int = 64
    train_sections: int = 2
    validation_sections: int = 2
    batch_size: int = 4
    updates: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    root_seed: int = ROOT_SEED

    def validate(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 < self.stride < self.window_size:
            raise ValueError("R1 requires 0 < stride < window_size to expose overlap")
        if self.train_sections <= 0 or self.validation_sections <= 0:
            raise ValueError("bounded train/validation section counts must be positive")
        if self.batch_size <= 0 or self.updates <= 0:
            raise ValueError("batch_size and updates must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer configuration is invalid")
        if self.root_seed != ROOT_SEED:
            raise ValueError(f"root_seed is frozen to {ROOT_SEED}")


@dataclass(frozen=True)
class TaskR01Spec:
    task_id: str
    label_version: str
    num_classes: int
    development_range: tuple[int, int]
    legal_train_range: tuple[int, int]
    legal_guard_range: tuple[int, int]
    legal_validation_range: tuple[int, int]
    buffer_groups: int
    source_files: tuple[str, ...]

    @property
    def development_inlines(self) -> tuple[int, ...]:
        return tuple(range(self.development_range[0], self.development_range[1] + 1))

    @property
    def legal_train_inlines(self) -> tuple[int, ...]:
        return tuple(range(self.legal_train_range[0], self.legal_train_range[1] + 1))

    @property
    def legal_validation_inlines(self) -> tuple[int, ...]:
        return tuple(range(self.legal_validation_range[0], self.legal_validation_range[1] + 1))


@dataclass(frozen=True)
class FullSection:
    task_id: str
    inline: int
    seismic: np.ndarray
    label: np.ndarray


@dataclass(frozen=True, order=True)
class WindowRef:
    task_id: str
    inline: int
    row: int
    col: int
    size: int

    @property
    def sample_id(self) -> str:
        return (
            f"{self.task_id}:development:inline={self.inline}:"
            f"row={self.row}:col={self.col}:size={self.size}"
        )


def task_r01_spec(task_id: str) -> TaskR01Spec:
    """Derive the internal legal split without enumerating an outer holdout."""
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown facies task {task_id!r}")
    outer = EXPECTED_OUTER_SPLITS[task_id]
    development_range = tuple(int(value) for value in outer["development_inline_range"])
    development = range(development_range[0], development_range[1] + 1)
    train, guard, validation = ordered_spatial_split(
        development, holdout_fraction=0.20, guard_fraction=0.05
    )
    if task_id == "facies_f3":
        sources = ("f3demo/inlines.zip", "f3demo/masks.tar.gz")
    else:
        sources = ("penobscot/dataset.h5",)
    schema = get_task_schema(task_id)
    return TaskR01Spec(
        task_id=task_id,
        label_version=LABEL_VERSIONS[task_id],
        num_classes=schema.num_classes,
        development_range=development_range,
        legal_train_range=(min(train), max(train)),
        legal_guard_range=(min(guard), max(guard)),
        legal_validation_range=(min(validation), max(validation)),
        buffer_groups=INTERNAL_BUFFER_GROUPS[task_id],
        source_files=sources,
    )


class DevelopmentOnlyFullSectionReader:
    """Read exact development payloads and reject non-development IDs first."""

    def __init__(self, data_root: Path, task_id: str) -> None:
        self.data_root = Path(data_root)
        self.spec = task_r01_spec(task_id)
        self.requested_inlines: list[int] = []
        self.outer_payloads_read = 0
        for relative in self.spec.source_files:
            if not (self.data_root / relative).is_file():
                raise ProtocolBlocked(f"missing registered source file: {relative}")

    def _assert_development(self, inline: int) -> None:
        lower, upper = self.spec.development_range
        if not lower <= int(inline) <= upper:
            raise PermissionError(
                f"{self.spec.task_id} inline {inline} is outside development "
                f"{lower}..{upper}; outer holdout access is forbidden"
            )

    def read_section(self, inline: int) -> FullSection:
        inline = int(inline)
        self._assert_development(inline)
        if self.spec.task_id == "facies_f3":
            seismic, label = self._read_f3(inline)
        else:
            seismic, label = self._read_penobscot(inline)
        seismic = np.asarray(denoise_identity(seismic), dtype=np.float32)
        label = np.asarray(label, dtype=np.uint8)
        if seismic.ndim == 3 and seismic.shape[-1] == 1:
            seismic = seismic[..., 0]
        if seismic.ndim != 2 or label.ndim != 2 or seismic.shape != label.shape:
            raise ProtocolBlocked(
                f"{self.spec.task_id} inline {inline} has unaligned full sections "
                f"{seismic.shape}/{label.shape}"
            )
        if not np.isfinite(seismic).all():
            raise ProtocolBlocked(f"{self.spec.task_id} inline {inline} contains NaN/Inf")
        validate_label_array(label, get_task_schema(self.spec.task_id))
        self.requested_inlines.append(inline)
        return FullSection(self.spec.task_id, inline, seismic, label)

    def _read_f3(self, inline: int) -> tuple[np.ndarray, np.ndarray]:
        seismic_path = self.data_root / "f3demo" / "inlines.zip"
        masks_path = self.data_root / "f3demo" / "masks.tar.gz"
        seismic_name = f"inlines/inline_{inline}.tiff"
        mask_name = f"masks/inline_{inline}_mask.png"
        try:
            # Only exact allowed payload names are requested.  The ZIP/TAR
            # container may read its directory metadata, but no outer member
            # payload is opened or exposed as a sample index.
            with zipfile.ZipFile(seismic_path) as seismic_zip:
                seismic_bytes = seismic_zip.read(seismic_name)
            with tarfile.open(masks_path, "r:gz") as masks:
                mask_stream = masks.extractfile(mask_name)
                if mask_stream is None:
                    raise KeyError(mask_name)
                mask_bytes = mask_stream.read()
        except (KeyError, OSError) as exc:
            raise ProtocolBlocked(f"missing F3 development payload for inline {inline}") from exc
        with Image.open(BytesIO(seismic_bytes)) as image:
            seismic = np.asarray(image).copy()
        with Image.open(BytesIO(mask_bytes)) as image:
            label = np.asarray(image).copy()
        return seismic, label

    def _read_penobscot(self, inline: int) -> tuple[np.ndarray, np.ndarray]:
        dataset_path = self.data_root / "penobscot" / "dataset.h5"
        # Published line numbers are contiguous 1000..1600.  Resolve one scalar
        # and one exact development slice; never read line_number[:] or an
        # outer volume range.
        index = inline - self.spec.development_range[0]
        if index < 0:
            raise PermissionError("negative Penobscot development index")
        with h5py.File(dataset_path, "r") as dataset:
            required = {"features", "label", "line_number"}
            if not required.issubset(dataset.keys()):
                raise ProtocolBlocked(f"Penobscot source lacks {sorted(required - set(dataset.keys()))}")
            observed_inline = int(dataset["line_number"][index])
            if observed_inline != inline:
                raise ProtocolBlocked(
                    f"Penobscot development index mismatch: requested {inline}, found {observed_inline}"
                )
            seismic = np.asarray(dataset["features"][index, ..., 0])
            label = np.asarray(dataset["label"][index, ...])
        return seismic, label

    def firewall_evidence(self) -> dict[str, Any]:
        lower, upper = self.spec.development_range
        if any(not lower <= inline <= upper for inline in self.requested_inlines):
            raise AssertionError("non-development payload escaped the reader firewall")
        return {
            "allowed_scope": "raw_development_inlines_only",
            "requested_inline_ids": sorted(set(self.requested_inlines)),
            "requested_payload_count": len(self.requested_inlines),
            "outer_sample_index_built": False,
            "outer_payloads_read": self.outer_payloads_read,
            "physical_test_h5_opened": False,
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_predictions_read": False,
            "test_metrics_read": False,
            "fresh_blind": False,
            "evidence_class": "development_protocol_mechanism_only",
        }

    def source_fingerprints(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for relative in self.spec.source_files:
            path = self.data_root / relative
            result.append(
                {
                    "logical_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": hash_file(path),
                }
            )
        return result


def bounded_section_ids(spec: TaskR01Spec, config: R01Config) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select adjacent-to-guard sections using IDs only, never labels."""
    train = spec.legal_train_inlines
    validation = spec.legal_validation_inlines
    if config.train_sections > len(train) or config.validation_sections > len(validation):
        raise ProtocolBlocked("bounded section request exceeds the legal split")
    selected_train = train[-config.train_sections :]
    selected_validation = validation[: config.validation_sections]
    nearest = min(abs(a - b) for a in selected_train for b in selected_validation)
    if nearest <= spec.buffer_groups:
        raise ProtocolBlocked(
            f"nearest train/validation inline distance {nearest} does not exceed buffer {spec.buffer_groups}"
        )
    return tuple(selected_train), tuple(selected_validation)


def axis_origins(length: int, window_size: int, stride: int) -> tuple[int, ...]:
    """Return deterministic origins including the last edge-aligned window."""
    if length < window_size:
        raise ProtocolBlocked(
            f"full-section axis length {length} is smaller than window {window_size}"
        )
    origins = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if origins[-1] != last:
        origins.append(last)
    return tuple(origins)


def plan_windows(section: FullSection, config: R01Config) -> tuple[WindowRef, ...]:
    rows = axis_origins(section.seismic.shape[0], config.window_size, config.stride)
    cols = axis_origins(section.seismic.shape[1], config.window_size, config.stride)
    return tuple(
        WindowRef(section.task_id, section.inline, row, col, config.window_size)
        for row in rows
        for col in cols
    )


def coverage_counts(shape: tuple[int, int], windows: Sequence[WindowRef]) -> np.ndarray:
    counts = np.zeros(shape, dtype=np.int32)
    for ref in windows:
        counts[ref.row : ref.row + ref.size, ref.col : ref.col + ref.size] += 1
    return counts


def coverage_evidence(section: FullSection, windows: Sequence[WindowRef]) -> dict[str, Any]:
    counts = coverage_counts(section.seismic.shape, windows)
    total = int(counts.size)
    covered = int(np.count_nonzero(counts))
    duplicate = int(np.maximum(counts - 1, 0).sum())
    if covered != total or int(counts.min()) < 1:
        raise ProtocolBlocked(
            f"{section.task_id} inline {section.inline} full-section coverage is {covered}/{total}"
        )
    return {
        "inline": section.inline,
        "section_shape": list(section.seismic.shape),
        "window_count": len(windows),
        "valid_voxels": total,
        "covered_unique_voxels": covered,
        "coverage_fraction": covered / total,
        "min_predictions_per_voxel": int(counts.min()),
        "max_predictions_per_voxel": int(counts.max()),
        "duplicate_prediction_assignments": duplicate,
        "scoring_rule": "uniform_mean_logits_then_one_global_confusion_entry_per_voxel",
    }


def window_patch(section: FullSection, ref: WindowRef) -> tuple[np.ndarray, np.ndarray]:
    if section.task_id != ref.task_id or section.inline != ref.inline:
        raise ValueError("window and full section identity differ")
    row_end, col_end = ref.row + ref.size, ref.col + ref.size
    seismic = section.seismic[ref.row:row_end, ref.col:col_end]
    label = section.label[ref.row:row_end, ref.col:col_end]
    if seismic.shape != (ref.size, ref.size) or label.shape != seismic.shape:
        raise ValueError("window escaped its full section")
    return seismic, label


def _rectangles_intersect(left: WindowRef, right: WindowRef) -> bool:
    return (
        left.inline == right.inline
        and left.row < right.row + right.size
        and right.row < left.row + left.size
        and left.col < right.col + right.size
        and right.col < left.col + left.size
    )


def overlap_audit(
    train: Sequence[WindowRef],
    validation: Sequence[WindowRef],
    section_shapes: Mapping[int, tuple[int, int]],
) -> dict[str, Any]:
    train_ids = {ref.sample_id for ref in train}
    validation_ids = {ref.sample_id for ref in validation}
    train_sections = {ref.inline for ref in train}
    validation_sections = {ref.inline for ref in validation}
    shared_sections = sorted(train_sections & validation_sections)
    rectangle_pairs = sum(
        1 for left in train for right in validation if _rectangles_intersect(left, right)
    )
    unique_shared_voxels = 0
    train_assignments_on_validation = 0
    for inline in shared_sections:
        shape = section_shapes[inline]
        train_mask = np.zeros(shape, dtype=bool)
        validation_mask = np.zeros(shape, dtype=bool)
        train_counts = np.zeros(shape, dtype=np.int16)
        for ref in train:
            if ref.inline == inline:
                region = np.s_[ref.row : ref.row + ref.size, ref.col : ref.col + ref.size]
                train_mask[region] = True
                train_counts[region] += 1
        for ref in validation:
            if ref.inline == inline:
                validation_mask[
                    ref.row : ref.row + ref.size,
                    ref.col : ref.col + ref.size,
                ] = True
        shared = train_mask & validation_mask
        unique_shared_voxels += int(shared.sum())
        train_assignments_on_validation += int(train_counts[shared].sum())
    return {
        "train_window_count": len(train),
        "validation_window_count": len(validation),
        "exact_sample_id_overlap": len(train_ids & validation_ids),
        "section_overlap_count": len(shared_sections),
        "shared_inline_ids": shared_sections,
        "intersecting_rectangle_pairs": rectangle_pairs,
        "unique_shared_voxels": unique_shared_voxels,
        "training_assignments_on_shared_voxels": train_assignments_on_validation,
    }


def diagnostic_random_split(
    legal_train: Sequence[WindowRef],
    evaluation: Sequence[WindowRef],
    *,
    seed: int,
) -> tuple[tuple[WindowRef, ...], tuple[WindowRef, ...]]:
    """Split overlapping windows by ID, intentionally before spatial isolation."""
    pool = tuple(legal_train) + tuple(evaluation)
    if len(pool) < 4 or not legal_train or not evaluation:
        raise ProtocolBlocked("diagnostic random split needs both legal train and evaluation windows")
    order = np.random.default_rng(seed).permutation(len(pool))
    train_count = len(legal_train)
    selected = set(int(value) for value in order[:train_count])
    random_train = [ref for index, ref in enumerate(pool) if index in selected]
    random_validation = [ref for index, ref in enumerate(pool) if index not in selected]

    evaluation_ids = {ref.sample_id for ref in evaluation}
    if not any(ref.sample_id in evaluation_ids for ref in random_train):
        # Deterministically force the intended negative control without labels.
        incoming = next(ref for ref in random_validation if ref.sample_id in evaluation_ids)
        outgoing = next(ref for ref in random_train if ref.sample_id not in evaluation_ids)
        random_train[random_train.index(outgoing)] = incoming
        random_validation[random_validation.index(incoming)] = outgoing
    if len(random_train) != len(legal_train):
        raise AssertionError("diagnostic and legal lanes differ in training window count")
    return tuple(sorted(random_train)), tuple(sorted(random_validation))


def fit_window_preprocessor(
    sections: Mapping[int, FullSection],
    windows: Sequence[WindowRef],
    *,
    num_classes: int,
) -> tuple[NormStats, np.ndarray, dict[str, Any]]:
    if not windows:
        raise ProtocolBlocked("cannot fit preprocessing on zero training windows")
    amplitudes: list[np.ndarray] = []
    histogram = np.zeros(num_classes, dtype=np.int64)
    for ref in windows:
        seismic, label = window_patch(sections[ref.inline], ref)
        amplitudes.append(seismic.reshape(-1))
        histogram += np.bincount(label.reshape(-1), minlength=num_classes)[:num_classes]
    missing = np.flatnonzero(histogram <= 0).tolist()
    if missing:
        raise ProtocolBlocked(f"fold-train window support misses configured classes {missing}")
    fit_values = np.concatenate(amplitudes).astype(np.float32, copy=False)
    stats = fit_zscore(fit_values)
    restored = denormalize(normalize(amplitudes[0], stats), stats)
    roundtrip = float(np.max(np.abs(restored - amplitudes[0])))
    if not np.isfinite(roundtrip) or roundtrip > 1e-2:
        raise ProtocolBlocked(f"fold-train normalization roundtrip failed: {roundtrip}")
    weights = inverse_sqrt_class_weights(histogram)
    evidence = {
        "fit_scope": "lane_train_windows_only",
        "fit_window_count": len(windows),
        "fit_sample_ids_hash": hash_payload(sorted(ref.sample_id for ref in windows)),
        "normalization": stats.to_dict(),
        "normalization_roundtrip_max_abs_error": roundtrip,
        "class_histogram": histogram.tolist(),
        "class_weights": weights.tolist(),
        "class_weight_rule": "inverse_sqrt_train_only_clipped_0.2_5.0",
        "target_transform": "identity_integer_ids",
        "threshold_or_calibration_fit": "none",
    }
    return stats, weights, evidence


def materialize_training_windows(
    sections: Mapping[int, FullSection],
    windows: Sequence[WindowRef],
    stats: NormStats,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for ref in windows:
        seismic, label = window_patch(sections[ref.inline], ref)
        images.append(np.asarray(normalize(seismic, stats), dtype=np.float32))
        labels.append(label.astype(np.int64, copy=False))
    return np.stack(images)[:, None], np.stack(labels)


def train_fixed_model(
    task_id: str,
    images: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
    config: R01Config,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train the fixed simple model for a fixed number of updates; no selection."""
    if images.shape[0] == 0 or labels.shape[0] != images.shape[0]:
        raise ProtocolBlocked("empty or unaligned training windows")
    torch.set_num_threads(1)
    seed_report = seed_everything(config.root_seed, strict=False).to_dict()
    model = discover_model("facies", MODEL_ID).build(
        get_task_spec(task_id), num_classes=get_task_schema(task_id).num_classes
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"{MODEL_ID} did not build a torch module")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32))
    schedule_seed = derive_seed(config.root_seed, "facies-r01-update-schedule", task_id)
    schedule = np.random.default_rng(schedule_seed).integers(
        0, images.shape[0], size=(config.updates, config.batch_size)
    )
    history: list[float] = []
    started = time.monotonic()
    for row in schedule:
        batch_images = torch.from_numpy(images[row]).float()
        batch_labels = torch.from_numpy(labels[row]).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        expected = (
            len(row),
            get_task_schema(task_id).num_classes,
            config.window_size,
            config.window_size,
        )
        if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
            raise RuntimeError(f"invalid {MODEL_ID} logits {tuple(logits.shape)}, expected {expected}")
        loss = criterion(logits, batch_labels)
        if not torch.isfinite(loss):
            raise RuntimeError("R1 training loss is NaN/Inf")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, {
        "model_id": MODEL_ID,
        "model_source": "_models/facies/facies_linear_pixel.py",
        "root_seed": config.root_seed,
        "seed_report": seed_report,
        "schedule_seed": schedule_seed,
        "schedule_hash": hash_payload(schedule.tolist()),
        "optimizer": "AdamW",
        "loss": "weighted_cross_entropy_on_raw_logits",
        "activation": "none_training_argmax_of_mean_logits_inference",
        "updates": config.updates,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_history": history,
        "wall_seconds": time.monotonic() - started,
        "hpo": False,
        "checkpoint_written": False,
    }


def predict_full_section(
    model: nn.Module,
    section: FullSection,
    windows: Sequence[WindowRef],
    stats: NormStats,
    *,
    num_classes: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend overlapping raw logits and return one prediction per voxel."""
    model.eval()
    logit_sum = np.zeros((num_classes, *section.seismic.shape), dtype=np.float32)
    counts = np.zeros(section.seismic.shape, dtype=np.int32)
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            refs = windows[start : start + batch_size]
            images = []
            for ref in refs:
                seismic, _ = window_patch(section, ref)
                images.append(np.asarray(normalize(seismic, stats), dtype=np.float32))
            logits = model(torch.from_numpy(np.stack(images)[:, None]).float()).cpu().numpy()
            if logits.shape[1] != num_classes or not np.isfinite(logits).all():
                raise RuntimeError("full-section inference produced invalid logits")
            for ref, patch_logits in zip(refs, logits, strict=True):
                region = np.s_[ref.row : ref.row + ref.size, ref.col : ref.col + ref.size]
                logit_sum[(slice(None), *region)] += patch_logits
                counts[region] += 1
    if int(np.count_nonzero(counts)) != counts.size or int(counts.min()) < 1:
        raise ProtocolBlocked("sliding-window inference failed full-section coverage")
    mean_logits = logit_sum / counts[None]
    prediction = mean_logits.argmax(axis=0).astype(np.uint8)
    evidence = {
        "inline": section.inline,
        "valid_voxels": int(counts.size),
        "covered_unique_voxels": int(np.count_nonzero(counts)),
        "unique_scored_voxels": int(prediction.size),
        "duplicate_prediction_assignments": int(np.maximum(counts - 1, 0).sum()),
        "min_predictions_per_voxel": int(counts.min()),
        "max_predictions_per_voxel": int(counts.max()),
        "blend": "uniform_mean_raw_logits",
        "metric_entry_rule": "one_argmax_prediction_per_unique_voxel",
    }
    return prediction, evidence


def evaluate_full_volume(
    model: nn.Module,
    validation_sections: Mapping[int, FullSection],
    windows_by_inline: Mapping[int, Sequence[WindowRef]],
    stats: NormStats,
    *,
    num_classes: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions: dict[int, np.ndarray] = {}
    coverage: list[dict[str, Any]] = []
    for inline in sorted(validation_sections):
        section = validation_sections[inline]
        prediction, evidence = predict_full_section(
            model,
            section,
            windows_by_inline[inline],
            stats,
            num_classes=num_classes,
            batch_size=batch_size,
        )
        matrix += confusion_matrix(section.label, prediction, num_classes)
        predictions[inline] = prediction
        coverage.append(evidence)
    try:
        metrics = segmentation_metrics_from_confusion(matrix)
    except ValueError as exc:
        raise ProtocolBlocked(f"full-volume fixed-class metrics are not rankable: {exc}") from exc
    total_valid = sum(item["valid_voxels"] for item in coverage)
    total_unique = sum(item["unique_scored_voxels"] for item in coverage)
    if total_valid != total_unique or int(metrics["evaluated_pixels"]) != total_valid:
        raise AssertionError("full-volume metric counted a voxel zero or multiple times")
    metrics.update(
        {
            "fixed_macro_f1": metrics["macro_f1"],
            "confusion_matrix": matrix.tolist(),
            "full_volume_inline_count": len(validation_sections),
            "full_volume_valid_voxels": total_valid,
            "unique_scored_voxels": total_unique,
            "duplicate_prediction_assignments_before_blend": sum(
                item["duplicate_prediction_assignments"] for item in coverage
            ),
            "coverage_fraction": 1.0,
            "coverage_by_inline": coverage,
            "averaging": "global_confusion_all_configured_classes_one_entry_per_unique_voxel",
        }
    )
    return metrics, predictions


def _load_sections(
    reader: DevelopmentOnlyFullSectionReader, inlines: Iterable[int]
) -> dict[int, FullSection]:
    return {inline: reader.read_section(inline) for inline in inlines}


def _flatten_windows(
    sections: Mapping[int, FullSection], config: R01Config
) -> tuple[WindowRef, ...]:
    return tuple(ref for inline in sorted(sections) for ref in plan_windows(sections[inline], config))


def _render_protocol_figure(
    task_id: str,
    section: FullSection,
    legal_prediction: np.ndarray,
    leaky_prediction: np.ndarray,
    legal_metrics: Mapping[str, Any],
    leaky_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    classes = get_task_schema(task_id).num_classes
    cmap = plt.get_cmap("tab10", classes)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    lo, hi = np.percentile(section.seismic, [1, 99])
    axes[0, 0].imshow(section.seismic, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    axes[0, 0].set_title(f"seismic | inline {section.inline}")
    axes[0, 1].imshow(section.label, cmap=cmap, vmin=-0.5, vmax=classes - 0.5, aspect="auto")
    axes[0, 1].set_title("ground truth")
    axes[0, 2].imshow(legal_prediction, cmap=cmap, vmin=-0.5, vmax=classes - 0.5, aspect="auto")
    axes[0, 2].set_title(
        f"legal block+guard\nmIoU={legal_metrics['miou']:.4f}, F1={legal_metrics['macro_f1']:.4f}"
    )
    axes[1, 0].imshow(leaky_prediction, cmap=cmap, vmin=-0.5, vmax=classes - 0.5, aspect="auto")
    axes[1, 0].set_title(
        f"diagnostic random-overlap\nmIoU={leaky_metrics['miou']:.4f}, F1={leaky_metrics['macro_f1']:.4f}"
    )
    axes[1, 1].imshow(legal_prediction != section.label, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].set_title("legal error")
    axes[1, 2].imshow(leaky_prediction != section.label, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[1, 2].set_title("leaky diagnostic error")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{task_id} R1 protocol mechanism only | development full section | "
        "fresh_blind=false",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_task(
    data_root: Path,
    task_id: str,
    config: R01Config,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reader = DevelopmentOnlyFullSectionReader(data_root, task_id)
    spec = reader.spec
    train_ids, validation_ids = bounded_section_ids(spec, config)
    train_sections = _load_sections(reader, train_ids)
    validation_sections = _load_sections(reader, validation_ids)
    all_sections = {**train_sections, **validation_sections}
    shapes = {inline: section.seismic.shape for inline, section in all_sections.items()}

    legal_train_windows = _flatten_windows(train_sections, config)
    evaluation_windows = _flatten_windows(validation_sections, config)
    windows_by_inline = {
        inline: plan_windows(section, config)
        for inline, section in validation_sections.items()
    }
    r0_coverage = [
        coverage_evidence(validation_sections[inline], windows_by_inline[inline])
        for inline in sorted(validation_sections)
    ]
    legal_overlap = overlap_audit(legal_train_windows, evaluation_windows, shapes)
    if any(
        legal_overlap[key] != 0
        for key in (
            "exact_sample_id_overlap",
            "section_overlap_count",
            "intersecting_rectangle_pairs",
            "unique_shared_voxels",
        )
    ):
        raise AssertionError("legal block+guard lane contains spatial overlap")
    nearest = min(abs(a - b) for a in train_ids for b in validation_ids)
    if nearest <= spec.buffer_groups:
        raise AssertionError("legal bounded split violates its inline buffer")

    leaky_train_windows, leaky_native_validation = diagnostic_random_split(
        legal_train_windows,
        evaluation_windows,
        seed=derive_seed(config.root_seed, "facies-r01-random-patch-split", task_id),
    )
    leaky_native_overlap = overlap_audit(leaky_train_windows, leaky_native_validation, shapes)
    leaky_evaluation_overlap = overlap_audit(leaky_train_windows, evaluation_windows, shapes)
    if leaky_native_overlap["unique_shared_voxels"] <= 0:
        raise AssertionError("diagnostic random-overlap lane did not expose native patch leakage")
    if leaky_evaluation_overlap["unique_shared_voxels"] <= 0:
        raise AssertionError("diagnostic lane did not contaminate the common evaluation block")
    if len(leaky_train_windows) != len(legal_train_windows):
        raise AssertionError("R1 lanes have unequal training-window counts")

    r0 = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R0_zero_training_contract",
        "task_id": task_id,
        "label_version": spec.label_version,
        "num_classes": spec.num_classes,
        "valid_label_ids": list(range(spec.num_classes)),
        "ignore_index": None,
        "development_range": list(spec.development_range),
        "legal_full_ranges": {
            "train": list(spec.legal_train_range),
            "guard": list(spec.legal_guard_range),
            "validation": list(spec.legal_validation_range),
        },
        "bounded_real_sections": {
            "train": list(train_ids),
            "validation": list(validation_ids),
            "nearest_inline_distance": nearest,
            "required_buffer_groups": spec.buffer_groups,
        },
        "split_hash": hash_payload(
            {
                "task_id": task_id,
                "label_version": spec.label_version,
                "ranges": [spec.legal_train_range, spec.legal_guard_range, spec.legal_validation_range],
                "bounded_train": train_ids,
                "bounded_validation": validation_ids,
            }
        ),
        "window_contract": {
            "window_size": config.window_size,
            "stride": config.stride,
            "edge_policy": "append_last_edge_aligned_origin",
            "blend": "uniform_mean_raw_logits",
            "metric_rule": "one_global_confusion_entry_per_unique_voxel",
        },
        "full_section_coverage": r0_coverage,
        "legal_overlap_audit": legal_overlap,
        "source_files": reader.source_fingerprints(),
        "data_root": "provided_at_runtime_not_serialized",
        "firewall": reader.firewall_evidence(),
        "status": "completed",
    }

    lanes: dict[str, dict[str, Any]] = {}
    predictions_by_lane: dict[str, dict[int, np.ndarray]] = {}
    lane_definitions = {
        "legal_block_guard": {
            "windows": legal_train_windows,
            "rankable": False,
            "diagnostic_only": True,
            "split": "block_then_guard_then_window",
            "overlap": legal_overlap,
        },
        "leaky_random_overlap": {
            "windows": leaky_train_windows,
            "rankable": False,
            "diagnostic_only": True,
            "split": "overlapping_windows_then_random_patch_split_intentionally_invalid",
            "overlap": leaky_evaluation_overlap,
            "native_random_patch_overlap": leaky_native_overlap,
        },
    }
    for lane_id, definition in lane_definitions.items():
        stats, weights, preprocessing = fit_window_preprocessor(
            all_sections, definition["windows"], num_classes=spec.num_classes
        )
        images, labels = materialize_training_windows(
            all_sections, definition["windows"], stats
        )
        model, training = train_fixed_model(
            task_id, images, labels, weights, config
        )
        metrics, predictions = evaluate_full_volume(
            model,
            validation_sections,
            windows_by_inline,
            stats,
            num_classes=spec.num_classes,
            batch_size=config.batch_size,
        )
        lanes[lane_id] = {
            "lane_id": lane_id,
            "status": "completed",
            "rankable": definition["rankable"],
            "diagnostic_only": definition["diagnostic_only"],
            "split_protocol": definition["split"],
            "overlap_audit": definition["overlap"],
            "native_random_patch_overlap": definition.get("native_random_patch_overlap"),
            "preprocessing": preprocessing,
            "training": training,
            "metrics": metrics,
            "test_archive_opened": False,
            "test_labels_read": False,
            "fresh_blind": False,
            "evidence_class": "development_protocol_mechanism_only",
        }
        predictions_by_lane[lane_id] = predictions

    first_inline = min(validation_sections)
    figure_name = f"{task_id}_r1_protocol_comparison.png"
    _render_protocol_figure(
        task_id,
        validation_sections[first_inline],
        predictions_by_lane["legal_block_guard"][first_inline],
        predictions_by_lane["leaky_random_overlap"][first_inline],
        lanes["legal_block_guard"]["metrics"],
        lanes["leaky_random_overlap"]["metrics"],
        output_root / figure_name,
    )
    r1 = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1_leakage_semantics_minimal_reproduction",
        "task_id": task_id,
        "label_version": spec.label_version,
        "num_classes": spec.num_classes,
        "model_id": MODEL_ID,
        "root_seed": config.root_seed,
        "config": asdict(config),
        "config_hash": hash_payload(asdict(config)),
        "split_hash": r0["split_hash"],
        "common_full_development_validation_inlines": list(validation_ids),
        "training_window_count_per_lane": len(legal_train_windows),
        "lanes": lanes,
        "comparison": {
            "purpose": "protocol_mechanism_only_not_model_ranking",
            "primary_observation": "quantify_metric_change_when_random_overlapping_windows_contaminate_a_common_development_evaluation_block",
            "legal_minus_leaky": {
                key: lanes["legal_block_guard"]["metrics"][key]
                - lanes["leaky_random_overlap"]["metrics"][key]
                for key in ("accuracy", "miou", "macro_f1")
            },
        },
        "figure": figure_name,
        "firewall": reader.firewall_evidence(),
        "test_archive_opened": False,
        "test_labels_read": False,
        "fresh_blind": False,
        "evidence_class": "development_protocol_mechanism_only",
        "ranking_status": "not_rankable",
        "status": "completed",
    }
    return r0, r1


def _blocked_record(task_id: str, stage: str, reason: str) -> dict[str, Any]:
    spec = task_r01_spec(task_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "task_id": task_id,
        "label_version": spec.label_version,
        "num_classes": spec.num_classes,
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason": reason,
        "test_archive_opened": False,
        "test_labels_read": False,
        "fresh_blind": False,
        "evidence_class": "development_protocol_mechanism_only",
    }


def run_r01(
    *,
    data_root: Path,
    output_root: Path,
    config: R01Config,
    task_ids: Sequence[str] = TASK_IDS,
) -> dict[str, Any]:
    config.validate()
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty R0/R1 output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    r0_tasks: dict[str, Any] = {}
    r1_tasks: dict[str, Any] = {}
    for task_id in task_ids:
        try:
            r0, r1 = run_task(Path(data_root), task_id, config, output_root)
        except ProtocolBlocked as exc:
            r0 = _blocked_record(task_id, "R0_zero_training_contract", str(exc))
            r1 = _blocked_record(task_id, "R1_leakage_semantics_minimal_reproduction", str(exc))
        r0_tasks[task_id] = r0
        r1_tasks[task_id] = r1

    common = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "root_seed": config.root_seed,
        "tasks_are_independent": True,
        "task_ids": list(task_ids),
        "data_root": "provided_at_runtime_not_serialized",
        "physical_test_h5_opened": False,
        "test_archive_opened": False,
        "test_labels_read": False,
        "known_holdout_predictions_or_metrics_read": False,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": "development_protocol_mechanism_only",
        "model_ranking_published": False,
        "hpo": False,
    }
    r0_payload = {**common, "stage": "R0", "tasks": r0_tasks}
    r1_payload = {**common, "stage": "R1", "config": asdict(config), "tasks": r1_tasks}
    atomic_write_json(output_root / "r0_contract.json", r0_payload)
    atomic_write_json(output_root / "r1_results.json", r1_payload)
    summary = {
        **common,
        "status": (
            "completed"
            if all(record["status"] == "completed" for record in r1_tasks.values())
            else "blocked_or_partial"
        ),
        "tasks": {
            task_id: {
                "r0_status": r0_tasks[task_id]["status"],
                "r1_status": r1_tasks[task_id]["status"],
                "ranking_status": r1_tasks[task_id].get("ranking_status", "not_rankable"),
                "legal_metrics": (
                    r1_tasks[task_id]["lanes"]["legal_block_guard"]["metrics"]
                    if r1_tasks[task_id]["status"] == "completed"
                    else None
                ),
                "leaky_metrics": (
                    r1_tasks[task_id]["lanes"]["leaky_random_overlap"]["metrics"]
                    if r1_tasks[task_id]["status"] == "completed"
                    else None
                ),
                "reason": r1_tasks[task_id].get("reason"),
            }
            for task_id in task_ids
        },
    }
    atomic_write_json(output_root / "r1_summary.json", summary)

    manifest = ArtifactManifest(run_id="facies-p5-r01", root=output_root)
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            role = "protocol_figure" if path.suffix == ".png" else "portable_protocol_evidence"
            manifest.register(path.name, role=role)
    manifest.write("artifact_manifest.json")
    manifest.verify()
    return summary


def verify_artifacts(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    manifest_path = output_root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for relative, record in payload["artifacts"].items():
        path = output_root / relative
        if not path.is_file() or hash_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {relative}")
        checked += 1
    for name in ("r0_contract.json", "r1_results.json", "r1_summary.json"):
        evidence = json.loads((output_root / name).read_text(encoding="utf-8"))
        if evidence.get("test_archive_opened") is not False:
            raise RuntimeError(f"{name} does not preserve the test firewall")
        if evidence.get("test_labels_read") is not False or evidence.get("fresh_blind") is not False:
            raise RuntimeError(f"{name} has invalid test/fresh-blind evidence")
    return {
        "status": "verified",
        "artifact_count": checked,
        "test_archive_opened": False,
        "test_labels_read": False,
        "fresh_blind": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run bounded development-only R0/R1")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--task", choices=("all", *TASK_IDS), default="all")
    run.add_argument("--window-size", type=int, default=128)
    run.add_argument("--stride", type=int, default=64)
    run.add_argument("--train-sections", type=int, default=2)
    run.add_argument("--validation-sections", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--updates", type=int, default=20)
    verify = commands.add_parser("verify", help="verify portable hashes without data/model")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify":
        result = verify_artifacts(args.output_root)
    else:
        config = R01Config(
            window_size=args.window_size,
            stride=args.stride,
            train_sections=args.train_sections,
            validation_sections=args.validation_sections,
            batch_size=args.batch_size,
            updates=args.updates,
        )
        selected = TASK_IDS if args.task == "all" else (args.task,)
        result = run_r01(
            data_root=args.data_root,
            output_root=args.output_root,
            config=config,
            task_ids=selected,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
