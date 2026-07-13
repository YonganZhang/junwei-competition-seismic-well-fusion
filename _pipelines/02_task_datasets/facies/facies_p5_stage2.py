#!/usr/bin/env python3
"""Fixed-budget, development-only P5 Stage-2 pilots for seismic facies.

F3 and Penobscot are separate tasks throughout: each has its own locked P4
manifest, label version, class head, metrics and leaderboard.  This module has
no frozen-test archive argument or loader.  It opens only ``train.h5`` and
selects a label-independent subset of the first locked development fold using
stable seeds before reading any selected labels.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import h5py
import numpy as np
import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.preprocess import denormalize, fit_zscore, normalize  # noqa: E402
from _code.ml_framework.seeding import derive_seed, seed_everything  # noqa: E402
from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest  # noqa: E402
from _models.facies._p5_common import P5AdapterSkip, source_lock  # noqa: E402

from p4_data import (  # noqa: E402
    ArchiveRecord,
    FaciesArchive,
    FoldPreprocessor,
    inverse_sqrt_class_weights,
)
from p4_experiment import _manifest_from_dict  # noqa: E402
from p4_losses import build_loss, softmax_probabilities  # noqa: E402
from p4_metrics import calibration_metrics, confusion_matrix  # noqa: E402
from p4_tasks import LABEL_VERSIONS, TASK_IDS, get_task_spec  # noqa: E402
from pipeline_contract import validate_label_array  # noqa: E402


MODEL_IDS = (
    "smp_unet_r18",
    "smp_deeplabv3plus_r18",
    "smp_unetpp_r18",
    "smp_fpn_r18",
    "torchvision_lraspp_mbv3",
    "deepseismic_patch_skip",
    "deepseismic_seresnet_unet",
    "hf_segformer_b0",
    "sfm_base_facies",
    "monai_unet3d",
)
STAGE1_ELIGIBLE = frozenset(
    {
        "smp_unet_r18",
        "smp_deeplabv3plus_r18",
        "smp_unetpp_r18",
        "smp_fpn_r18",
        "torchvision_lraspp_mbv3",
        "hf_segformer_b0",
    }
)
STAGE1_SKIPS: Mapping[str, Mapping[str, Any]] = {
    "deepseismic_patch_skip": {
        "code": "legacy_source_port_not_available",
        "reason": "pinned DeepSeismic patch source is not an importable locked runtime; substitution is forbidden",
    },
    "deepseismic_seresnet_unet": {
        "code": "legacy_source_port_not_available",
        "reason": "pinned DeepSeismic SE-ResNet U-Net source is not an importable locked runtime; substitution is forbidden",
    },
    "sfm_base_facies": {
        "code": "legacy_source_port_not_available",
        "reason": "pinned SFM source is not an importable locked runtime; substitution is forbidden",
    },
    "monai_unet3d": {
        "code": "contiguous_3d_development_blocks_unavailable",
        "reason": "the locked ModelBatch contains independent 2-D patches, not verified contiguous development-only 3-D blocks",
    },
}

ROOT_SEED = 2693
RESULT_SCHEMA = "facies-p5-stage2-v1"
LOCKED_MANIFEST_STABLE_HASHES: Mapping[str, str] = {
    "facies_f3": "76a501385482bedce4e48dc44dfe5e9854b1b7aa0674fc223b3a6b42760f4614",
    "facies_penobscot": "3440bbbd415158e41c6f0ed14513208fc77075c108716f9f6593ff0847382c81",
}
DEFAULT_GPU_LOCK = Path("/mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock")
DEFAULT_PORTABLE_OUTPUT = TRACK_DIR / "_outputs" / "p5_stage2"
DEFAULT_RUNTIME_OUTPUT = TRACK_DIR / "_outputs" / "p5_stage2_runtime"


@dataclass(frozen=True)
class PilotBudget:
    """One frozen scratch-lane budget shared by every task/model cell."""

    profile_id: str = "facies-p5-stage2-fixed-v1"
    fold_id: int = 0
    max_updates: int = 40
    max_wall_seconds: float = 180.0
    max_train_samples: int = 32
    max_validation_samples: int = 16
    batch_size: int = 2
    validation_interval: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    loss_id: str = "cross_entropy"

    def __post_init__(self) -> None:
        if self.fold_id != 0:
            raise ValueError("Stage-2 pilot is frozen to the first valid development fold (fold 0)")
        if not 1 <= self.max_updates <= 200:
            raise ValueError("2-D Stage-2 max_updates must be in [1, 200]")
        if not 0.0 < self.max_wall_seconds <= 600.0:
            raise ValueError("2-D Stage-2 max_wall_seconds must be in (0, 600]")
        if self.max_train_samples <= 0 or self.max_validation_samples <= 0:
            raise ValueError("sample caps must be positive")
        if self.batch_size <= 0 or self.batch_size > self.max_train_samples:
            raise ValueError("batch_size must be positive and no larger than train cap")
        if self.validation_interval <= 0 or self.validation_interval > self.max_updates:
            raise ValueError("validation_interval must be in [1, max_updates]")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer budget values are invalid")
        if self.loss_id != "cross_entropy":
            raise ValueError("Stage-2 comparison freezes weighted CrossEntropy")


@dataclass(frozen=True)
class PreparedPilot:
    task_id: str
    label_version: str
    num_classes: int
    manifest_stable_hash: str
    manifest_file_sha256: str
    fold_id: int
    train_images: np.ndarray
    train_labels: np.ndarray
    validation_images: np.ndarray
    validation_labels: np.ndarray
    class_weights: tuple[float, ...]
    preprocessor: FoldPreprocessor
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_support: tuple[int, ...]
    validation_support: tuple[int, ...]
    nearest_inline_distance: int
    buffer_groups: int
    pilot_split_hash: str
    update_schedule: np.ndarray


class Stage2DevelopmentArchive(FaciesArchive):
    """Reject test resolution before any filesystem call."""

    def split_path(self, split: str) -> Path:
        if split != "train":
            raise RuntimeError(
                "P5 Stage-2 is development-only; resolving a frozen-test path is forbidden"
            )
        return super().split_path(split)


class GpuFlock:
    """Exclusive POSIX flock required for every CUDA Stage-2 cell."""

    def __init__(self, path: Path = DEFAULT_GPU_LOCK) -> None:
        self.path = Path(path)
        self._handle: Any | None = None
        self.wait_seconds = 0.0

    def __enter__(self) -> "GpuFlock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self._handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        self.wait_seconds = time.perf_counter() - started
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "download_bytes": 0,
        "offline_weights": True,
    }
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(device)
        payload["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_locked_manifest(task_id: str, path: Path) -> tuple[SplitManifest, str]:
    """Open exactly one caller-supplied manifest file; never inspect its directory."""
    manifest_path = Path(path)
    manifest = _manifest_from_dict(_read_json_object(manifest_path))
    validate_manifest(manifest)
    metadata = manifest.metadata
    spec = get_task_spec(task_id)
    if metadata.get("track_id") != "facies" or metadata.get("task_id") != task_id:
        raise ValueError("locked manifest track/task identity mismatch")
    if metadata.get("label_version") != spec.label_version:
        raise ValueError("locked manifest label version differs from TaskSpec")
    if int(metadata.get("num_classes", -1)) != int(spec.metadata["num_classes"]):
        raise ValueError("locked manifest class count differs from TaskSpec")
    if not manifest.folds or manifest.folds[0].fold_id != 0:
        raise ValueError("locked manifest has no valid fold 0")
    observed_hash = manifest.stable_hash()
    expected_hash = LOCKED_MANIFEST_STABLE_HASHES[task_id]
    if observed_hash != expected_hash:
        raise ValueError(
            f"locked manifest stable hash mismatch for {task_id}: "
            f"expected {expected_hash}, observed {observed_hash}; re-splitting is forbidden"
        )
    return manifest, hash_file(manifest_path)


def deterministic_subset(
    sample_ids: Sequence[str], *, count: int, seed: int
) -> tuple[str, ...]:
    """Select IDs without consulting labels, then canonicalize their read order."""
    values = tuple(sample_ids)
    if not values:
        raise ValueError("cannot sample an empty locked fold partition")
    chosen_count = min(int(count), len(values))
    indices = np.random.default_rng(seed).choice(len(values), size=chosen_count, replace=False)
    return tuple(sorted(values[int(index)] for index in indices))


def fixed_update_schedule(
    sample_count: int, budget: PilotBudget, *, seed: int
) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("training schedule requires samples")
    generator = np.random.default_rng(seed)
    return generator.integers(
        0,
        sample_count,
        size=(budget.max_updates, budget.batch_size),
        endpoint=False,
        dtype=np.int64,
    )


def _storage_key(task_id: str, sample_id: str) -> str:
    prefix = f"{task_id}:train:"
    if not sample_id.startswith(prefix):
        raise ValueError(f"manifest sample ID is not locked development data: {sample_id!r}")
    key = sample_id[len(prefix) :]
    if not key or "/" in key:
        raise ValueError(f"unsafe development storage key in {sample_id!r}")
    return key


def _materialize_selected(
    archive: Stage2DevelopmentArchive,
    sample_ids: Sequence[str],
) -> tuple[tuple[ArchiveRecord, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Read only the already-selected development IDs from ``train.h5``."""
    records: list[ArchiveRecord] = []
    raw_images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    path = archive.split_path("train")
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("task", archive.task_id)) != archive.task_id:
            raise ValueError("development archive task identity mismatch")
        if str(handle.attrs.get("split", "train")) != "train":
            raise ValueError("Stage-2 may only open a train.h5 development archive")
        for sample_id in sample_ids:
            key = _storage_key(archive.task_id, sample_id)
            if key not in handle:
                raise ValueError(f"locked development sample is absent from train.h5: {sample_id}")
            group = handle[key]
            position = json.loads(group.attrs["position"])
            metadata = json.loads(group.attrs["meta"])
            label = np.asarray(group["label"][()], dtype=np.int64)
            validate_label_array(label, archive.schema)
            record = ArchiveRecord(
                task_id=archive.task_id,
                split="train",
                sample_id=sample_id,
                storage_key=key,
                inline=int(position["inline"]),
                crossline=(
                    None if position.get("crossline") is None else int(position["crossline"])
                ),
                time_ms=None if position.get("time_ms") is None else float(position["time_ms"]),
                patch_shape=tuple(int(value) for value in label.shape),
                label_support=tuple(
                    int(value)
                    for value in np.bincount(
                        label.reshape(-1), minlength=archive.schema.num_classes
                    )[: archive.schema.num_classes]
                ),
                source=str(metadata.get("source", "unknown")),
            )
            raw, checked_label, _ = archive._read_group(  # locked P4 adapter, train handle only
                handle, record, include_target=True
            )
            assert checked_label is not None
            records.append(record)
            raw_images.append(np.asarray(raw, dtype=np.float32))
            labels.append(np.asarray(checked_label, dtype=np.int64))
    if len({image.shape for image in raw_images}) != 1:
        raise ValueError("fixed pilot samples do not share one 2-D patch shape")
    return tuple(records), tuple(raw_images), tuple(labels)


def _support(labels: Sequence[np.ndarray], num_classes: int) -> tuple[int, ...]:
    histogram = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        histogram += np.bincount(label.reshape(-1), minlength=num_classes)[:num_classes]
    return tuple(int(value) for value in histogram)


def prepare_pilot(
    *,
    task_id: str,
    manifest_path: Path,
    processed_root: Path,
    budget: PilotBudget,
    root_seed: int = ROOT_SEED,
) -> PreparedPilot:
    """Prepare one task from locked fold 0 without reading any test artifact."""
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown facies task {task_id!r}")
    manifest, manifest_file_sha256 = load_locked_manifest(task_id, manifest_path)
    fold = manifest.folds[budget.fold_id]
    train_seed = derive_seed(root_seed, "sampler", task_id, "fold0", "train_subset")
    validation_seed = derive_seed(root_seed, "sampler", task_id, "fold0", "validation_subset")
    selected_train_ids = deterministic_subset(
        fold.train_sample_ids, count=budget.max_train_samples, seed=train_seed
    )
    selected_validation_ids = deterministic_subset(
        fold.validation_sample_ids, count=budget.max_validation_samples, seed=validation_seed
    )
    if set(selected_train_ids) & set(selected_validation_ids):
        raise ValueError("fixed pilot train and validation samples overlap")

    archive = Stage2DevelopmentArchive(task_id, processed_root)
    train_records, train_raw, train_labels = _materialize_selected(archive, selected_train_ids)
    validation_records, validation_raw, validation_labels = _materialize_selected(
        archive, selected_validation_ids
    )
    train_groups = tuple(str(record.inline) for record in train_records)
    validation_groups = tuple(str(record.inline) for record in validation_records)
    if set(train_groups) & set(validation_groups):
        raise ValueError("fixed pilot train and validation inline groups overlap")
    declared_train = set(fold.train_groups)
    declared_validation = set(fold.validation_groups)
    if not set(train_groups) <= declared_train or not set(validation_groups) <= declared_validation:
        raise ValueError("selected pilot group escapes locked fold 0")

    classes = int(get_task_spec(task_id).metadata["num_classes"])
    fold_support = np.asarray(fold.support.get("train_per_class_pixels", ()), dtype=np.int64)
    if fold_support.shape != (classes,):
        raise ValueError("locked fold lacks complete fold-train class-support evidence")
    class_weights = inverse_sqrt_class_weights(fold_support)
    fit_values = np.concatenate([image.reshape(-1) for image in train_raw]).astype(
        np.float32, copy=False
    )
    normalization = fit_zscore(fit_values)
    probe = train_raw[0]
    recovered = denormalize(normalize(probe, normalization), normalization)
    roundtrip_error = float(np.max(np.abs(recovered - probe)))
    if not math.isfinite(roundtrip_error) or roundtrip_error > 1e-2:
        raise ValueError(f"fold-train normalization round-trip failed: {roundtrip_error}")
    preprocessor = FoldPreprocessor(
        task_id=task_id,
        label_version=LABEL_VERSIONS[task_id],
        normalization=normalization,
        class_weights=tuple(float(value) for value in class_weights),
        class_histogram=tuple(int(value) for value in fold_support),
        fit_sample_count=len(selected_train_ids),
        fit_sample_ids_hash=hash_payload(list(selected_train_ids)),
        roundtrip_max_abs_error=roundtrip_error,
    )

    train_images = np.stack(
        [normalize(image, normalization).astype(np.float32) for image in train_raw]
    )[:, None]
    validation_images = np.stack(
        [normalize(image, normalization).astype(np.float32) for image in validation_raw]
    )[:, None]
    train_label_array = np.stack(train_labels).astype(np.int64)
    validation_label_array = np.stack(validation_labels).astype(np.int64)
    if not np.isfinite(train_images).all() or not np.isfinite(validation_images).all():
        raise ValueError("fold-train normalization produced NaN/Inf")

    nearest = int(fold.purge.get("nearest_train_validation_inline_distance", 0))
    buffer_groups = int(fold.purge.get("buffer_groups", 0))
    if nearest <= buffer_groups or buffer_groups <= 0:
        raise ValueError("locked fold 0 does not prove a positive spatial buffer")
    split_evidence = {
        "task_id": task_id,
        "label_version": LABEL_VERSIONS[task_id],
        "manifest_stable_hash": manifest.stable_hash(),
        "fold_id": fold.fold_id,
        "train_sample_ids": list(selected_train_ids),
        "validation_sample_ids": list(selected_validation_ids),
        "train_groups": list(train_groups),
        "validation_groups": list(validation_groups),
        "nearest_inline_distance": nearest,
        "buffer_groups": buffer_groups,
        "selection": "stable_seed_before_label_read",
    }
    schedule_seed = derive_seed(root_seed, "sampler", task_id, "fold0", budget.profile_id)
    return PreparedPilot(
        task_id=task_id,
        label_version=LABEL_VERSIONS[task_id],
        num_classes=classes,
        manifest_stable_hash=manifest.stable_hash(),
        manifest_file_sha256=manifest_file_sha256,
        fold_id=fold.fold_id,
        train_images=train_images,
        train_labels=train_label_array,
        validation_images=validation_images,
        validation_labels=validation_label_array,
        class_weights=tuple(float(value) for value in class_weights),
        preprocessor=preprocessor,
        train_sample_ids=selected_train_ids,
        validation_sample_ids=selected_validation_ids,
        train_groups=train_groups,
        validation_groups=validation_groups,
        train_support=_support(train_labels, classes),
        validation_support=_support(validation_labels, classes),
        nearest_inline_distance=nearest,
        buffer_groups=buffer_groups,
        pilot_split_hash=hash_payload(split_evidence),
        update_schedule=fixed_update_schedule(
            len(selected_train_ids), budget, seed=schedule_seed
        ),
    )


def _assert_logits(logits: torch.Tensor, images: torch.Tensor, classes: int) -> None:
    expected = (images.shape[0], classes, images.shape[2], images.shape[3])
    if tuple(logits.shape) != expected:
        raise ValueError(f"expected finite raw logits {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("model produced NaN/Inf raw logits")


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _evaluate(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    *,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = torch.as_tensor(
                images[start : start + batch_size], dtype=torch.float32, device=device
            )
            batch_labels = torch.as_tensor(
                labels[start : start + batch_size], dtype=torch.long, device=device
            )
            logits = model(batch_images)
            _assert_logits(logits, batch_images, num_classes)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise ValueError("validation loss is NaN/Inf")
            losses.append(float(loss.detach()))
            probabilities.append(softmax_probabilities(logits).cpu().numpy())
    merged = np.concatenate(probabilities, axis=0)
    matrix = confusion_matrix(labels, merged.argmax(axis=1), num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    false_positive = matrix.sum(axis=0).astype(np.float64) - true_positive
    false_negative = support.astype(np.float64) - true_positive
    union = true_positive + false_positive + false_negative
    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    per_class_iou = np.divide(
        true_positive, union, out=np.zeros(num_classes, dtype=np.float64), where=union > 0
    )
    per_class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=f1_denominator > 0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "miou": float(per_class_iou.mean()),
        "macro_f1": float(per_class_f1.mean()),
        "per_class_support": support.tolist(),
        "per_class_iou": per_class_iou.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": matrix.tolist(),
        "evaluated_pixels": int(matrix.sum()),
        "observed_class_ids": np.flatnonzero(support > 0).tolist(),
        "all_classes_supported": bool(np.all(support > 0)),
        "averaging": "all_configured_classes_missing_support_scores_zero",
        "finite_logits": True,
    }
    metrics.update(calibration_metrics(merged, labels, n_bins=15))
    numeric = [
        metrics["accuracy"],
        metrics["miou"],
        metrics["macro_f1"],
        metrics["nll"],
        metrics["brier"],
        metrics["ece"],
        *metrics["per_class_iou"],
        *metrics["per_class_f1"],
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("Stage-2 validation metrics contain NaN/Inf")
    return float(np.mean(losses)), metrics


def _trainer_state(step: int, best_step: int, best_loss: float, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "next_epoch": step + 1,
        "global_step": step,
        "best_epoch": best_step,
        "best_val_loss": best_loss,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "history": history,
    }


def _structured_skip(
    task_id: str,
    model_id: str,
    skip: Mapping[str, Any],
    *,
    budget: PilotBudget,
    prepared: PreparedPilot | None,
) -> dict[str, Any]:
    spec = get_task_spec(task_id)
    return {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "task_id": task_id,
        "label_version": spec.label_version,
        "head_num_classes": int(spec.metadata["num_classes"]),
        "model_id": model_id,
        "lane": "scratch",
        "status": "skipped",
        "skip": dict(skip),
        "source_lock": dict(source_lock(model_id)),
        "budget": asdict(budget),
        "split": None if prepared is None else _split_result(prepared),
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
    }


def _split_result(prepared: PreparedPilot) -> dict[str, Any]:
    return {
        "manifest_stable_hash": prepared.manifest_stable_hash,
        "manifest_file_sha256": prepared.manifest_file_sha256,
        "fold_id": prepared.fold_id,
        "pilot_split_hash": prepared.pilot_split_hash,
        "selection": "stable_seed_before_label_read",
        "train_sample_count": len(prepared.train_sample_ids),
        "validation_sample_count": len(prepared.validation_sample_ids),
        "train_sample_ids_hash": hash_payload(list(prepared.train_sample_ids)),
        "validation_sample_ids_hash": hash_payload(list(prepared.validation_sample_ids)),
        "train_groups_hash": hash_payload(list(prepared.train_groups)),
        "validation_groups_hash": hash_payload(list(prepared.validation_groups)),
        "train_validation_sample_overlap": 0,
        "train_validation_group_overlap": 0,
        "nearest_inline_distance": prepared.nearest_inline_distance,
        "buffer_groups": prepared.buffer_groups,
        "train_per_class_support": list(prepared.train_support),
        "validation_per_class_support": list(prepared.validation_support),
    }


def run_model_pilot(
    *,
    prepared: PreparedPilot,
    model_id: str,
    budget: PilotBudget,
    runtime_root: Path,
    device: torch.device,
    gpu_lock_wait_seconds: float,
    root_seed: int = ROOT_SEED,
) -> dict[str, Any]:
    """Train and validate one eligible model with the shared fixed schedule."""
    if model_id not in STAGE1_ELIGIBLE:
        return _structured_skip(
            prepared.task_id,
            model_id,
            STAGE1_SKIPS[model_id],
            budget=budget,
            prepared=prepared,
        )
    task_id = prepared.task_id
    spec = get_task_spec(task_id)
    model_seed = derive_seed(root_seed, "model", task_id, model_id, "scratch", "stage2")
    seed_report = seed_everything(model_seed, strict=False).to_dict()
    started = time.perf_counter()
    model: nn.Module | None = None
    try:
        discovered = discover_model("facies", model_id)
        model = discovered.build(
            spec, num_classes=prepared.num_classes, lane="scratch"
        ).to(device)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable_count = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        criterion = build_loss(
            budget.loss_id,
            num_classes=prepared.num_classes,
            class_weights=torch.tensor(
                prepared.class_weights, dtype=torch.float32, device=device
            ),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay
        )
        checkpoint_path = (
            runtime_root / task_id / model_id / "scratch" / "best.ckpt"
        )
        configuration = {
            "task_id": task_id,
            "label_version": prepared.label_version,
            "model_id": model_id,
            "lane": "scratch",
            "num_classes": prepared.num_classes,
            "root_seed": root_seed,
            "model_seed": model_seed,
            "budget": asdict(budget),
            "loss_activation": "weighted_cross_entropy_raw_logits; softmax_inference_only",
        }
        history: list[dict[str, Any]] = []
        best_loss = math.inf
        best_step = -1
        updates = 0
        timeout = False
        gradients_finite = False
        peak_before = 0
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            peak_before = int(torch.cuda.memory_allocated(device))

        for step_index, indices in enumerate(prepared.update_schedule, start=1):
            if time.perf_counter() - started >= budget.max_wall_seconds:
                timeout = True
                break
            images = torch.as_tensor(
                prepared.train_images[indices], dtype=torch.float32, device=device
            )
            labels = torch.as_tensor(
                prepared.train_labels[indices], dtype=torch.long, device=device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            _assert_logits(logits, images, prepared.num_classes)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("training loss is NaN/Inf")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients_finite = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            if not gradients_finite:
                raise ValueError("backward produced missing or non-finite gradients")
            optimizer.step()
            updates = step_index
            train_loss = float(loss.detach())

            if step_index % budget.validation_interval == 0 or step_index == budget.max_updates:
                val_loss, _ = _evaluate(
                    model,
                    prepared.validation_images,
                    prepared.validation_labels,
                    criterion,
                    batch_size=budget.batch_size,
                    num_classes=prepared.num_classes,
                    device=device,
                )
                history.append(
                    {
                        "update": step_index,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                    }
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_step = step_index
                    save_checkpoint(
                        checkpoint_path,
                        epoch=step_index,
                        model_state=_cpu_state_dict(model),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=None,
                        scaler_state=None,
                        config_hash=hash_payload(configuration),
                        split_hash=prepared.pilot_split_hash,
                        trainer_state=_trainer_state(
                            step_index, best_step, best_loss, history
                        ),
                        seed_report=seed_report,
                        environment=_environment(device),
                        extra={
                            "stage": "p5_stage2_fixed_budget_pilot",
                            "preprocessor_hash": hash_payload(prepared.preprocessor.to_dict()),
                            "manifest_stable_hash": prepared.manifest_stable_hash,
                            "test_access": False,
                        },
                    )
        wall_seconds = time.perf_counter() - started
        if timeout:
            return {
                "schema_version": RESULT_SCHEMA,
                "track_id": "facies",
                "task_id": task_id,
                "label_version": prepared.label_version,
                "head_num_classes": prepared.num_classes,
                "model_id": model_id,
                "lane": "scratch",
                "status": "timeout",
                "reason": "fixed per-cell model wall-clock budget exhausted",
                "budget": asdict(budget),
                "updates_completed": updates,
                "wall_seconds": wall_seconds,
                "split": _split_result(prepared),
                "test_archive_opened": False,
                "test_labels_read": False,
                "test_metrics_computed": False,
            }
        if updates != budget.max_updates or best_step < 0 or not checkpoint_path.is_file():
            raise RuntimeError("pilot did not complete its fixed update/checkpoint contract")

        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["split_hash"] != prepared.pilot_split_hash:
            raise ValueError("best checkpoint split hash mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        probe = torch.as_tensor(
            prepared.validation_images[:1], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            expected = model(probe).detach().cpu()
        restored = discovered.build(
            spec, num_classes=prepared.num_classes, lane="scratch"
        ).to(device)
        restored.load_state_dict(checkpoint["model_state"])
        restored.eval()
        with torch.no_grad():
            actual = restored(probe).detach().cpu()
        checkpoint_difference = float(torch.max(torch.abs(actual - expected)))
        if checkpoint_difference > 1e-6:
            raise ValueError(
                f"checkpoint round-trip changed logits by {checkpoint_difference}"
            )
        validation_loss, metrics = _evaluate(
            restored,
            prepared.validation_images,
            prepared.validation_labels,
            criterion,
            batch_size=budget.batch_size,
            num_classes=prepared.num_classes,
            device=device,
        )
        peak_vram = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        checkpoint_relative = checkpoint_path.relative_to(runtime_root).as_posix()
        return {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            "task_id": task_id,
            "label_version": prepared.label_version,
            "head_num_classes": prepared.num_classes,
            "model_id": model_id,
            "lane": "scratch",
            "status": "development_piloted",
            "source_lock": dict(source_lock(model_id)),
            "budget": asdict(budget),
            "root_seed": root_seed,
            "model_seed": model_seed,
            "split": _split_result(prepared),
            "preprocessing": {
                "fit_scope": "selected_locked_fold0_train_only",
                "normalization": prepared.preprocessor.normalization.to_dict(),
                "class_weight_scope": "locked_full_fold0_train_support_only",
                "class_weights": list(prepared.class_weights),
                "roundtrip_max_abs_error": prepared.preprocessor.roundtrip_max_abs_error,
                "denoise": "identity",
            },
            "training": {
                "updates_completed": updates,
                "history": history,
                "best_update": best_step,
                "best_validation_loss": best_loss,
                "reloaded_validation_loss": validation_loss,
                "optimizer": "AdamW",
                "loss": "weighted_cross_entropy_on_raw_logits",
                "activation": "none_during_training_softmax_inference_only",
                "backward_finite": gradients_finite,
            },
            "validation_metrics": metrics,
            "checkpoint": {
                "runtime_relative_path": checkpoint_relative,
                "sha256": hash_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "prediction_max_abs_difference": checkpoint_difference,
            },
            "resources": {
                "parameters": parameter_count,
                "trainable_parameters": trainable_count,
                "wall_seconds": wall_seconds,
                "gpu_lock_name": DEFAULT_GPU_LOCK.name,
                "gpu_lock_held": device.type == "cuda",
                "gpu_lock_wait_seconds_excluded": gpu_lock_wait_seconds,
                "cuda_memory_before_bytes": peak_before,
                "cuda_peak_allocated_bytes": peak_vram,
                "download_bytes": 0,
            },
            "environment": _environment(device),
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        }
    except P5AdapterSkip as skip:
        return _structured_skip(
            task_id, model_id, skip.to_dict(), budget=budget, prepared=prepared
        )
    except Exception as exc:
        return {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            "task_id": task_id,
            "label_version": prepared.label_version,
            "head_num_classes": prepared.num_classes,
            "model_id": model_id,
            "lane": "scratch",
            "status": "failed",
            "failure": {
                "type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            },
            "budget": asdict(budget),
            "split": _split_result(prepared),
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        }
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _blocked_results(
    task_id: str,
    reason: Exception,
    budget: PilotBudget,
) -> list[dict[str, Any]]:
    spec = get_task_spec(task_id)
    return [
        {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            "task_id": task_id,
            "label_version": spec.label_version,
            "head_num_classes": int(spec.metadata["num_classes"]),
            "model_id": model_id,
            "lane": "scratch",
            "status": "blocked",
            "blocker": {
                "code": "locked_development_fold_unavailable",
                "type": type(reason).__name__,
                "reason": str(reason),
            },
            "budget": asdict(budget),
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        }
        for model_id in MODEL_IDS
    ]


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _leaderboard(task_id: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record["task_id"] == task_id and record["status"] == "development_piloted"
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["validation_metrics"]["miou"]),
            -float(item["validation_metrics"]["macro_f1"]),
            str(item["model_id"]),
        ),
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "task_id": task_id,
        "label_version": get_task_spec(task_id).label_version,
        "lane": "scratch",
        "ranking_scope": "locked_development_fold0_validation_only",
        "primary_metric": "miou",
        "secondary_metric": "macro_f1",
        "frozen_test_consumed": False,
        "rows": [
            {
                "rank": rank,
                "model_id": record["model_id"],
                "miou": record["validation_metrics"]["miou"],
                "macro_f1": record["validation_metrics"]["macro_f1"],
                "accuracy": record["validation_metrics"]["accuracy"],
                "all_classes_supported": record["validation_metrics"]["all_classes_supported"],
                "updates": record["training"]["updates_completed"],
                "wall_seconds": record["resources"]["wall_seconds"],
            }
            for rank, record in enumerate(ranked, start=1)
        ],
    }


def write_portable_results(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    budget: PilotBudget,
    device: torch.device,
) -> dict[str, Any]:
    output_root = Path(output_root)
    results_path = _atomic_write_jsonl(output_root / "p5_stage2_results.jsonl", records)
    task_summaries: dict[str, Any] = {}
    for task_id in TASK_IDS:
        task_records = [record for record in records if record["task_id"] == task_id]
        leaderboard = _leaderboard(task_id, records)
        leaderboard_name = f"{task_id}_scratch_leaderboard.json"
        leaderboard_path = atomic_write_json(output_root / leaderboard_name, leaderboard)
        task_summaries[task_id] = {
            "label_version": get_task_spec(task_id).label_version,
            "head_num_classes": int(get_task_spec(task_id).metadata["num_classes"]),
            "status_counts": {
                status: sum(record["status"] == status for record in task_records)
                for status in ("development_piloted", "skipped", "blocked", "failed", "timeout")
            },
            "leaderboard": leaderboard_name,
            "leaderboard_sha256": hash_file(leaderboard_path),
            "ranked_models": len(leaderboard["rows"]),
        }
    summary = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "lane": "scratch",
        "root_seed": ROOT_SEED,
        "budget": asdict(budget),
        "model_ids": list(MODEL_IDS),
        "stage1_eligible_model_ids": sorted(STAGE1_ELIGIBLE),
        "tasks_are_independent": True,
        "cross_task_ranking_forbidden": True,
        "result_count": len(records),
        "results_file": results_path.name,
        "results_sha256": hash_file(results_path),
        "tasks": task_summaries,
        "environment": _environment(device),
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
    }
    atomic_write_json(output_root / "p5_stage2_summary.json", summary)
    return summary


def run_stage2(
    *,
    manifest_paths: Mapping[str, Path],
    processed_root: Path,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
    budget: PilotBudget | None = None,
    gpu_lock_path: Path = DEFAULT_GPU_LOCK,
) -> dict[str, Any]:
    """Run all ten candidates for both independent development tasks."""
    active_budget = PilotBudget() if budget is None else budget
    if set(manifest_paths) != set(TASK_IDS):
        raise ValueError(f"manifest paths must be supplied for exactly {TASK_IDS}")
    records: list[dict[str, Any]] = []
    prepared_by_task: dict[str, PreparedPilot] = {}
    for task_id in TASK_IDS:
        try:
            prepared_by_task[task_id] = prepare_pilot(
                task_id=task_id,
                manifest_path=manifest_paths[task_id],
                processed_root=processed_root,
                budget=active_budget,
            )
        except Exception as exc:
            records.extend(_blocked_results(task_id, exc, active_budget))

    def execute(lock_wait: float) -> None:
        for task_id in TASK_IDS:
            prepared = prepared_by_task.get(task_id)
            if prepared is None:
                continue
            for model_id in MODEL_IDS:
                records.append(
                    run_model_pilot(
                        prepared=prepared,
                        model_id=model_id,
                        budget=active_budget,
                        runtime_root=runtime_root,
                        device=device,
                        gpu_lock_wait_seconds=lock_wait,
                    )
                )

    if device.type == "cuda":
        if Path(gpu_lock_path) != DEFAULT_GPU_LOCK:
            raise ValueError(f"CUDA Stage-2 must flock the frozen lock {DEFAULT_GPU_LOCK}")
        with GpuFlock(DEFAULT_GPU_LOCK) as lock:
            execute(lock.wait_seconds)
    else:
        execute(0.0)
    order = {(task, model): index for index, (task, model) in enumerate(
        (task, model) for task in TASK_IDS for model in MODEL_IDS
    )}
    records.sort(key=lambda record: order[(record["task_id"], record["model_id"])])
    return write_portable_results(
        output_root, records, budget=active_budget, device=device
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--f3-manifest", type=Path, required=True)
    parser.add_argument("--penobscot-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PORTABLE_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage2(
        manifest_paths={
            "facies_f3": args.f3_manifest,
            "facies_penobscot": args.penobscot_manifest,
        },
        processed_root=args.processed_root,
        output_root=args.output_root,
        runtime_root=args.runtime_root,
        device=resolve_device(args.device),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
