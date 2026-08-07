#!/usr/bin/env python3
"""P5.2 R2 facies learning-curve and single-factor ablation runner.

This runner is development-only. It keeps F3 and Penobscot fully separate,
uses the frozen Stage-3 representative models as controls, and never exposes
any frozen-test or known-holdout input surface. The one training loop per run
uses the shared ``ml_framework.train.train_loop()`` with repeatable factories
so train/val loss are recorded per epoch and the best checkpoint is selected by
minimum validation loss.

R2 expands the raw-section R01 contract into:

* 2-D weighted CE learning curves at 40/400/1000-update prefixes;
* 1000-update CE+Dice and CE→Lovasz single-factor ablations; and
* 1000-update 2.5-D weighted CE context ablations using previous/center/next
  inline channels with edge replication inside the frozen split only.

The validation loss seen by ``train_loop`` is a deterministic proxy sampled
from the frozen validation block.  Final scientific metrics are always
computed on the full validation block with 100% voxel coverage and one global
confusion entry per unique voxel.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import ArtifactManifest, atomic_write_json, hash_file, hash_payload  # noqa: E402
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.preprocess import NormStats, denormalize, normalize  # noqa: E402
from _code.ml_framework.seeding import derive_seed, seed_everything  # noqa: E402
from _code.ml_framework.train import TrainHistory, train_loop  # noqa: E402
from _code.ml_framework.visualize import plot_loss_curve  # noqa: E402
from _models.facies._p5_common import SOURCE_LOCK_PATH, source_lock  # noqa: E402

import facies_p5_r01 as r01  # noqa: E402
from p4_data import inverse_sqrt_class_weights  # noqa: E402
from p4_losses import build_loss, softmax_probabilities  # noqa: E402
from p4_metrics import confidence_entropy_error, confusion_matrix  # noqa: E402
from p4_tasks import TASK_IDS, get_task_spec  # noqa: E402


ROOT_SEED = 2693
RESULT_SCHEMA = "facies-p5-r2-v1"
EXPECTED_GPU_LOCK = Path("/mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock")
DEFAULT_PORTABLE_OUTPUT = TRACK_DIR / "_outputs" / "p5_r2"
DEFAULT_RUNTIME_OUTPUT = TRACK_DIR / "_outputs" / "p5_r2_runtime"
EXPECTED_R01_SOURCE_SHA256 = "db2d3cab1f26814c279bebaf8c72e80ea79f6a3b383944ba0ed7a21535b8a13b"
EXPECTED_SOURCE_LOCK_SHA256 = "1a301d6e5764e73bf55719c54eb384de9efb14f7932beb4574cfd537896a8b97"
EXPECTED_CONTROL_ADAPTER_SHA256: Mapping[str, str] = {
    "facies_f3": "3eb40bae400d9eaf2ebb1d6e4109f8ba03607a527cbd09254f08346eb14b4178",
    "facies_penobscot": "1f16e9d8c221eeecff613d45ab901d2798d3567611117e51407c70a087179bdd",
}
CONTROL_MODELS: Mapping[str, str] = {
    "facies_f3": "smp_fpn_r18",
    "facies_penobscot": "smp_deeplabv3plus_r18",
}
MODEL_SEEDS = (1867973658, 2137841944, 3902865753)
LOSS_RECIPES = ("cross_entropy", "cross_entropy_plus_dice", "cross_entropy_to_lovasz")
ENDPOINTS = (40, 400, 1000)
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 64
DEFAULT_BATCH_SIZE = 2
DEFAULT_VAL_PROXY_WINDOWS = 128
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_MAX_WALL_SECONDS = 7200.0
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
TRAIN_CORE_RANGES: Mapping[str, tuple[int, int]] = {
    "facies_f3": (100, 463),
    "facies_penobscot": (1000, 1335),
}
VALIDATION_RANGES: Mapping[str, tuple[int, int]] = {
    "facies_f3": (489, 586),
    "facies_penobscot": (1359, 1448),
}
BUFFER_RANGES: Mapping[str, tuple[int, int]] = {
    "facies_f3": (464, 488),
    "facies_penobscot": (1336, 1358),
}


class ProtocolBlocked(RuntimeError):
    """A scientific/data contract prevents a legal numeric result."""


@dataclass(frozen=True)
class R2Budget:
    profile_id: str = "facies-p5-r2-fixed-v1"
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    batch_size: int = DEFAULT_BATCH_SIZE
    val_proxy_windows: int = DEFAULT_VAL_PROXY_WINDOWS
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    max_updates: int = 1000
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    root_seed: int = ROOT_SEED

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.stride <= 0 or self.stride >= self.window_size:
            raise ValueError("invalid sliding-window contract")
        if self.batch_size <= 0 or self.val_proxy_windows <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch/window counts must be positive")
        if self.max_updates != 1000:
            raise ValueError("R2 freezes a single 1000-update schedule")
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer budget values are invalid")
        if self.root_seed != ROOT_SEED:
            raise ValueError(f"root_seed is frozen to {ROOT_SEED}")


@dataclass(frozen=True)
class TaskMaterial:
    task_id: str
    label_version: str
    num_classes: int
    development_range: tuple[int, int]
    train_range: tuple[int, int]
    guard_range: tuple[int, int]
    validation_range: tuple[int, int]
    source_fingerprints: tuple[dict[str, Any], ...]
    source_lock_sha256: str
    r01_source_sha256: str
    adapter_sha256: str
    train_sections: dict[int, r01.FullSection]
    validation_sections: dict[int, r01.FullSection]
    train_window_refs: tuple[r01.WindowRef, ...]
    validation_window_refs: tuple[r01.WindowRef, ...]
    validation_proxy_refs: tuple[r01.WindowRef, ...]
    normalization: NormStats
    class_weights: tuple[float, ...]
    class_histogram: tuple[int, ...]
    roundtrip_max_abs_error: float
    split_hash: str
    firewall: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    task_id: str
    model_id: str
    model_seed: int
    repeat_id: int
    recipe_id: str
    context_mode: str
    loss_id: str
    epochs: int
    endpoint_epochs: tuple[int, ...]

    @property
    def key(self) -> str:
        return (
            f"{self.task_id}/{self.recipe_id}/seed-{self.repeat_id}/"
            f"{self.model_id}/{self.context_mode}"
        )


@dataclass(frozen=True)
class SectionPrediction:
    inline: int
    sample_id: str
    gt_class_count: int
    correct_pixels: int
    error_pixels: int
    boundary_pixels: int
    boundary_error_pixels: int
    total_pixels: int
    error_fraction: float
    coverage: dict[str, Any]
    has_informative_signal: bool


@dataclass(frozen=True)
class RunOutcome:
    spec: RunSpec
    status: str
    history: TrainHistory | None
    history_path: Path | None
    loss_curve_path: Path | None
    checkpoint_paths: dict[int, Path]
    best_checkpoint_path: Path | None
    best_epoch: int | None
    best_val_loss: float | None
    endpoint_results: list[dict[str, Any]]
    validation_records: list[SectionPrediction]
    selected_diagnostic: SectionPrediction | None
    selected_checkpoint_epoch: int | None
    selected_checkpoint_path: Path | None
    train_wall_seconds: float
    eval_wall_seconds: float
    gpu_peak_bytes: int | None
    gpu_wait_seconds: float
    error: dict[str, Any] | None = None


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


class GpuFlock:
    """Exclusive POSIX flock required for every CUDA R2 trajectory."""

    def __init__(self, path: Path = EXPECTED_GPU_LOCK) -> None:
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


def _task_bounds(task_id: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown facies task {task_id!r}")
    return TRAIN_CORE_RANGES[task_id], BUFFER_RANGES[task_id], VALIDATION_RANGES[task_id]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _train_window_refs(sections: Mapping[int, r01.FullSection], config: R2Budget) -> tuple[r01.WindowRef, ...]:
    return tuple(
        ref
        for inline in sorted(sections)
        for ref in r01.plan_windows(sections[inline], r01.R01Config(window_size=config.window_size, stride=config.stride))
    )


def _stream_train_statistics(
    sections: Mapping[int, r01.FullSection],
    train_refs: Sequence[r01.WindowRef],
    *,
    num_classes: int,
) -> tuple[NormStats, tuple[float, ...], tuple[int, ...], float]:
    pixel_count = 0
    pixel_sum = 0.0
    pixel_sumsq = 0.0
    histogram = np.zeros(num_classes, dtype=np.int64)
    probe: np.ndarray | None = None
    for ref in train_refs:
        seismic, label = r01.window_patch(sections[ref.inline], ref)
        if probe is None:
            probe = np.asarray(seismic, dtype=np.float32)
        values = np.asarray(seismic, dtype=np.float64).reshape(-1)
        pixel_sum += float(values.sum())
        pixel_sumsq += float(np.square(values).sum())
        pixel_count += int(values.size)
        histogram += np.bincount(label.reshape(-1), minlength=num_classes)[:num_classes]
    if pixel_count <= 0 or probe is None:
        raise ProtocolBlocked("cannot fit train-core normalization on zero windows")
    mean = pixel_sum / pixel_count
    variance = max(pixel_sumsq / pixel_count - mean * mean, 1e-12)
    stats = NormStats(method="zscore", mean=float(mean), std=float(math.sqrt(variance)))
    restored = denormalize(normalize(probe, stats), stats)
    roundtrip = float(np.max(np.abs(restored - probe)))
    if not np.isfinite(roundtrip) or roundtrip > 1e-2:
        raise ProtocolBlocked(f"fold-train normalization roundtrip failed: {roundtrip}")
    weights = inverse_sqrt_class_weights(histogram)
    return stats, tuple(float(value) for value in weights), tuple(int(value) for value in histogram), roundtrip


def _select_proxy_window_refs(
    validation_refs: Sequence[r01.WindowRef],
    *,
    count: int,
    seed: int,
) -> tuple[r01.WindowRef, ...]:
    refs = tuple(sorted(validation_refs, key=lambda ref: ref.sample_id))
    if not refs:
        raise ProtocolBlocked("validation window pool is empty")
    chosen_count = min(count, len(refs))
    indices = np.random.default_rng(seed).choice(len(refs), size=chosen_count, replace=False)
    return tuple(sorted(refs[int(index)] for index in indices))


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return Path(path).name


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


def _validate_contract_hashes(task_id: str) -> None:
    expected_range = {
        "facies_f3": ((100, 463), (464, 488), (489, 586)),
        "facies_penobscot": ((1000, 1335), (1336, 1358), (1359, 1448)),
    }[task_id]
    if _task_bounds(task_id)[0] != expected_range[0]:
        raise ValueError(f"{task_id} train core range drifted")
    if _task_bounds(task_id)[1] != expected_range[1]:
        raise ValueError(f"{task_id} guard range drifted")
    if _task_bounds(task_id)[2] != expected_range[2]:
        raise ValueError(f"{task_id} validation range drifted")
    if hash_file(SOURCE_LOCK_PATH) != EXPECTED_SOURCE_LOCK_SHA256:
        raise ValueError("facies source lock changed; R2 is fail-closed")
    if hash_file(TRACK_DIR / "facies_p5_r01.py") != EXPECTED_R01_SOURCE_SHA256:
        raise ValueError("R01 source changed; R2 is fail-closed")
    adapter_path = PROJECT_ROOT / "_models" / "facies" / f"{CONTROL_MODELS[task_id]}.py"
    if hash_file(adapter_path) != EXPECTED_CONTROL_ADAPTER_SHA256[task_id]:
        raise ValueError(f"{task_id} control adapter changed; R2 is fail-closed")


def load_task_material(task_id: str, data_root: Path, budget: R2Budget) -> TaskMaterial:
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown facies task {task_id!r}")
    _validate_contract_hashes(task_id)
    reader = r01.DevelopmentOnlyFullSectionReader(Path(data_root), task_id)
    spec = r01.task_r01_spec(task_id)
    train_range, guard_range, validation_range = _task_bounds(task_id)
    train_sections = {
        inline: reader.read_section(inline)
        for inline in range(train_range[0], train_range[1] + 1)
    }
    validation_sections = {
        inline: reader.read_section(inline)
        for inline in range(validation_range[0], validation_range[1] + 1)
    }
    train_refs = _train_window_refs(train_sections, budget)
    validation_refs = _train_window_refs(validation_sections, budget)
    proxy_refs = _select_proxy_window_refs(
        validation_refs,
        count=budget.val_proxy_windows,
        seed=derive_seed(ROOT_SEED, "r2", task_id, "validation_proxy"),
    )
    normalization, class_weights, class_histogram, roundtrip = _stream_train_statistics(
        train_sections,
        train_refs,
        num_classes=spec.num_classes,
    )
    split_hash = hash_payload(
        {
            "task_id": task_id,
            "label_version": spec.label_version,
            "train_range": list(train_range),
            "guard_range": list(guard_range),
            "validation_range": list(validation_range),
            "train_window_count": len(train_refs),
            "validation_window_count": len(validation_refs),
            "validation_proxy_window_ids": [ref.sample_id for ref in proxy_refs],
            "normalization": normalization.to_dict(),
        }
    )
    reader_fence = reader.firewall_evidence()
    reader_fence["test_archive_opened"] = False
    reader_fence["test_labels_read"] = False
    reader_fence["known_holdout_predictions_or_metrics_read"] = False
    reader_fence["fresh_blind"] = False
    return TaskMaterial(
        task_id=task_id,
        label_version=spec.label_version,
        num_classes=spec.num_classes,
        development_range=spec.development_range,
        train_range=train_range,
        guard_range=guard_range,
        validation_range=validation_range,
        source_fingerprints=tuple(reader.source_fingerprints()),
        source_lock_sha256=EXPECTED_SOURCE_LOCK_SHA256,
        r01_source_sha256=EXPECTED_R01_SOURCE_SHA256,
        adapter_sha256=EXPECTED_CONTROL_ADAPTER_SHA256[task_id],
        train_sections=train_sections,
        validation_sections=validation_sections,
        train_window_refs=train_refs,
        validation_window_refs=validation_refs,
        validation_proxy_refs=proxy_refs,
        normalization=normalization,
        class_weights=class_weights,
        class_histogram=class_histogram,
        roundtrip_max_abs_error=roundtrip,
        split_hash=split_hash,
        firewall=reader_fence,
    )


def build_run_specs() -> tuple[RunSpec, ...]:
    specs: list[RunSpec] = []
    for task_id in TASK_IDS:
        model_id = CONTROL_MODELS[task_id]
        for repeat_id, model_seed in enumerate(MODEL_SEEDS):
            specs.append(
                RunSpec(
                    task_id=task_id,
                    model_id=model_id,
                    model_seed=model_seed,
                    repeat_id=repeat_id,
                    recipe_id="ce_2d",
                    context_mode="2d",
                    loss_id="cross_entropy",
                    epochs=1000,
                    endpoint_epochs=ENDPOINTS,
                )
            )
            specs.append(
                RunSpec(
                    task_id=task_id,
                    model_id=model_id,
                    model_seed=model_seed,
                    repeat_id=repeat_id,
                    recipe_id="ce_plus_dice_2d",
                    context_mode="2d",
                    loss_id="cross_entropy_plus_dice",
                    epochs=1000,
                    endpoint_epochs=(1000,),
                )
            )
            specs.append(
                RunSpec(
                    task_id=task_id,
                    model_id=model_id,
                    model_seed=model_seed,
                    repeat_id=repeat_id,
                    recipe_id="ce_to_lovasz_2d",
                    context_mode="2d",
                    loss_id="cross_entropy_to_lovasz",
                    epochs=1000,
                    endpoint_epochs=(1000,),
                )
            )
            specs.append(
                RunSpec(
                    task_id=task_id,
                    model_id=model_id,
                    model_seed=model_seed,
                    repeat_id=repeat_id,
                    recipe_id="ce_2p5d",
                    context_mode="2p5d",
                    loss_id="cross_entropy",
                    epochs=1000,
                    endpoint_epochs=(1000,),
                )
            )
    return tuple(specs)


def _batch_from_refs(
    sections: Mapping[int, r01.FullSection],
    refs: Sequence[r01.WindowRef],
    stats: NormStats,
    *,
    context_mode: str,
    split_bounds: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    lower, upper = split_bounds
    for ref in refs:
        section = sections[ref.inline]
        if context_mode == "2d":
            seismic, label = r01.window_patch(section, ref)
            image = np.asarray(normalize(seismic, stats), dtype=np.float32)[None]
        elif context_mode == "2p5d":
            channels: list[np.ndarray] = []
            for offset in (-1, 0, 1):
                neighbor_inline = min(max(ref.inline + offset, lower), upper)
                neighbor = sections[neighbor_inline]
                seismic, _ = r01.window_patch(neighbor, r01.WindowRef(ref.task_id, neighbor_inline, ref.row, ref.col, ref.size))
                channels.append(np.asarray(normalize(seismic, stats), dtype=np.float32))
            label = r01.window_patch(section, ref)[1]
            image = np.stack(channels, axis=0)
        else:
            raise ValueError(f"unknown context_mode {context_mode!r}")
        images.append(image)
        labels.append(label.astype(np.int64, copy=False))
    return np.stack(images), np.stack(labels)


class _EpochBatchFactory:
    def __init__(
        self,
        *,
        sections: Mapping[int, r01.FullSection],
        refs: Sequence[r01.WindowRef],
        stats: NormStats,
        context_mode: str,
        schedule: np.ndarray,
        split_bounds: tuple[int, int],
    ) -> None:
        self.sections = sections
        self.refs = tuple(refs)
        self.stats = stats
        self.context_mode = context_mode
        self.schedule = np.asarray(schedule, dtype=np.int64)
        self.split_bounds = split_bounds
        self.cursor = 0

    def __call__(self) -> Iterable[dict[str, Any]]:
        if self.cursor >= len(self.schedule):
            raise RuntimeError("train schedule exhausted before epochs completed")
        epoch = self.cursor + 1
        row = self.schedule[self.cursor]
        self.cursor += 1
        batch_refs = [self.refs[int(index)] for index in row]
        images, labels = _batch_from_refs(
            self.sections,
            batch_refs,
            self.stats,
            context_mode=self.context_mode,
            split_bounds=self.split_bounds,
        )
        return [
            {
                "epoch": epoch,
                "images": images,
                "labels": labels,
                "sample_ids": [ref.sample_id for ref in batch_refs],
                "inline_ids": [int(ref.inline) for ref in batch_refs],
            }
        ]


def _proxy_validation_batch(
    *,
    sections: Mapping[int, r01.FullSection],
    refs: Sequence[r01.WindowRef],
    stats: NormStats,
    context_mode: str,
    split_bounds: tuple[int, int],
    epoch: int,
) -> dict[str, Any]:
    images, labels = _batch_from_refs(
        sections,
        refs,
        stats,
        context_mode=context_mode,
        split_bounds=split_bounds,
    )
    return {
        "epoch": epoch,
        "images": images,
        "labels": labels,
        "sample_ids": [ref.sample_id for ref in refs],
        "inline_ids": [int(ref.inline) for ref in refs],
    }


class CalibrationAccumulator:
    def __init__(self, num_classes: int, n_bins: int = 15) -> None:
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.pixel_count = 0
        self.nll_sum = 0.0
        self.brier_sum = 0.0
        self.top_counts = np.zeros(n_bins, dtype=np.int64)
        self.top_conf_sum = np.zeros(n_bins, dtype=np.float64)
        self.top_correct_sum = np.zeros(n_bins, dtype=np.float64)
        self.class_counts = np.zeros((num_classes, n_bins), dtype=np.int64)
        self.class_prob_sum = np.zeros((num_classes, n_bins), dtype=np.float64)
        self.class_truth_sum = np.zeros((num_classes, n_bins), dtype=np.float64)
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def update(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        probs = np.asarray(probabilities, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.int64)
        if probs.ndim != 4 or probs.shape[1] != self.num_classes:
            raise ValueError("probability tensor is malformed")
        if probs.shape[0] != truth.shape[0] or probs.shape[2:] != truth.shape[1:]:
            raise ValueError("probabilities and labels are unaligned")
        flat_probs = np.moveaxis(probs, 1, -1).reshape(-1, self.num_classes)
        flat_truth = truth.reshape(-1)
        if flat_truth.size == 0:
            raise ValueError("calibration update received zero pixels")
        row = np.arange(flat_truth.size)
        true_probability = flat_probs[row, flat_truth]
        self.nll_sum += float(-np.log(np.clip(true_probability, 1e-12, 1.0)).sum())
        self.brier_sum += float(
            (np.square(flat_probs).sum(axis=1) + 1.0 - 2.0 * true_probability).sum()
        )
        predicted = flat_probs.argmax(axis=1)
        confidence = flat_probs.max(axis=1)
        correct = (predicted == flat_truth).astype(np.float64)
        encoded = flat_truth * self.num_classes + predicted
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        bin_ids = np.clip(
            np.searchsorted(self.bin_edges, confidence, side="right") - 1,
            0,
            self.n_bins - 1,
        )
        for bin_id in range(self.n_bins):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue
            self.top_counts[bin_id] += int(mask.sum())
            self.top_conf_sum[bin_id] += float(confidence[mask].sum())
            self.top_correct_sum[bin_id] += float(correct[mask].sum())
        for class_id in range(self.num_classes):
            class_probability = flat_probs[:, class_id]
            class_truth = (flat_truth == class_id).astype(np.float64)
            class_bin_ids = np.clip(
                np.searchsorted(self.bin_edges, class_probability, side="right") - 1,
                0,
                self.n_bins - 1,
            )
            for bin_id in range(self.n_bins):
                mask = class_bin_ids == bin_id
                if not np.any(mask):
                    continue
                self.class_counts[class_id, bin_id] += int(mask.sum())
                self.class_prob_sum[class_id, bin_id] += float(class_probability[mask].sum())
                self.class_truth_sum[class_id, bin_id] += float(class_truth[mask].sum())
        self.pixel_count += int(flat_truth.size)

    def finalize(self) -> dict[str, Any]:
        if self.pixel_count <= 0:
            raise ValueError("no pixels were accumulated for metrics")
        matrix = self.confusion
        tp = np.diag(matrix).astype(np.float64)
        support = matrix.sum(axis=1).astype(np.int64)
        fp = matrix.sum(axis=0).astype(np.float64) - tp
        fn = support.astype(np.float64) - tp
        union = tp + fp + fn
        denom = 2.0 * tp + fp + fn
        if np.any(support <= 0):
            raise ValueError("full validation support does not cover every configured class")
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        top_ece = 0.0
        reliability: list[dict[str, Any]] = []
        for bin_id in range(self.n_bins):
            count = int(self.top_counts[bin_id])
            lower, upper = float(self.bin_edges[bin_id]), float(self.bin_edges[bin_id + 1])
            mean_conf = float(self.top_conf_sum[bin_id] / count) if count else None
            mean_acc = float(self.top_correct_sum[bin_id] / count) if count else None
            if count:
                top_ece += (count / self.pixel_count) * abs(mean_acc - mean_conf)
            reliability.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "mean_confidence": mean_conf,
                    "accuracy": mean_acc,
                }
            )
        classwise_ece: list[float] = []
        for class_id in range(self.num_classes):
            class_ece = 0.0
            for bin_id in range(self.n_bins):
                count = int(self.class_counts[class_id, bin_id])
                if not count:
                    continue
                mean_prob = float(self.class_prob_sum[class_id, bin_id] / count)
                mean_truth = float(self.class_truth_sum[class_id, bin_id] / count)
                class_ece += (count / self.pixel_count) * abs(mean_truth - mean_prob)
            classwise_ece.append(float(class_ece))
        return {
            "accuracy": float(tp.sum() / matrix.sum()),
            "miou": float(iou.mean()),
            "macro_f1": float(f1.mean()),
            "fixed_macro_f1": float(f1.mean()),
            "per_class_support": support.tolist(),
            "per_class_iou": iou.tolist(),
            "per_class_f1": f1.tolist(),
            "confusion_matrix": matrix.tolist(),
            "evaluated_pixels": int(matrix.sum()),
            "ignored_pixels": 0,
            "nll": float(self.nll_sum / self.pixel_count),
            "brier": float(self.brier_sum / self.pixel_count),
            "ece": float(top_ece),
            "classwise_ece": classwise_ece,
            "macro_classwise_ece": float(np.mean(classwise_ece)),
            "reliability_bins": reliability,
            "calibration_pixels": int(self.pixel_count),
            "n_bins": self.n_bins,
        }


def _boundary_mask(label: np.ndarray) -> np.ndarray:
    array = np.asarray(label, dtype=np.int64)
    mask = np.zeros_like(array, dtype=bool)
    mask[1:, :] |= array[1:, :] != array[:-1, :]
    mask[:-1, :] |= array[:-1, :] != array[1:, :]
    mask[:, 1:] |= array[:, 1:] != array[:, :-1]
    mask[:, :-1] |= array[:, :-1] != array[:, 1:]
    return mask


def _section_forward(
    model: nn.Module,
    sections: Mapping[int, r01.FullSection],
    section: r01.FullSection,
    refs: Sequence[r01.WindowRef],
    stats: NormStats,
    *,
    context_mode: str,
    split_bounds: tuple[int, int],
    batch_size: int,
    num_classes: int,
    device: torch.device,
    arrays: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray] | None]:
    logit_sum = np.zeros((num_classes, *section.seismic.shape), dtype=np.float32)
    counts = np.zeros(section.seismic.shape, dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(refs), batch_size):
            chunk = refs[start : start + batch_size]
            images, _ = _batch_from_refs(
                sections,
                chunk,
                stats,
                context_mode=context_mode,
                split_bounds=split_bounds,
            )
            logits = model(torch.as_tensor(images, dtype=torch.float32, device=device))
            expected_shape = (len(chunk), num_classes, images.shape[-2], images.shape[-1])
            if tuple(logits.shape) != expected_shape:
                raise ValueError("validation inference produced malformed logits")
            logits_np = logits.detach().cpu().numpy()
            for ref, patch_logits in zip(chunk, logits_np, strict=True):
                region = np.s_[ref.row : ref.row + ref.size, ref.col : ref.col + ref.size]
                logit_sum[(slice(None), *region)] += patch_logits
                counts[region] += 1
    if int(counts.min()) < 1 or int(np.count_nonzero(counts)) != counts.size:
        raise ProtocolBlocked("sliding-window inference failed full-section coverage")
    mean_logits = logit_sum / counts[None]
    probabilities = softmax_probabilities(torch.as_tensor(mean_logits[None], dtype=torch.float32)).numpy()[0]
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0).astype(np.float32)
    entropy = (
        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=0)
        / math.log(num_classes)
    ).astype(np.float32)
    error = (prediction != section.label).astype(np.uint8)
    boundary = _boundary_mask(section.label)
    coverage = {
        "inline": section.inline,
        "valid_voxels": int(counts.size),
        "covered_unique_voxels": int(np.count_nonzero(counts)),
        "unique_scored_voxels": int(prediction.size),
        "duplicate_prediction_assignments_before_blend": int(np.maximum(counts - 1, 0).sum()),
        "min_predictions_per_voxel": int(counts.min()),
        "max_predictions_per_voxel": int(counts.max()),
        "coverage_fraction": 1.0,
        "blend": "uniform_mean_raw_logits",
        "metric_entry_rule": "one_argmax_prediction_per_unique_voxel",
    }
    arrays_payload: dict[str, np.ndarray] | None = None
    if arrays:
        arrays_payload = {
            "seismic": section.seismic.astype(np.float32),
            "labels": section.label.astype(np.uint8),
            "prediction": prediction.astype(np.uint8),
            "confidence": confidence.astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "error": error.astype(np.uint8),
            "boundary": boundary.astype(np.uint8),
            "probabilities": probabilities.astype(np.float32),
            "logits": mean_logits.astype(np.float32),
        }
    return prediction, probabilities, coverage, arrays_payload


def _select_diagnostic_section(section_stats: Sequence[SectionPrediction]) -> SectionPrediction:
    eligible = [item for item in section_stats if item.has_informative_signal]
    key_source = eligible if eligible else list(section_stats)
    if not key_source:
        raise ProtocolBlocked("no validation section statistics were produced")
    return min(
        key_source,
        key=lambda item: (
            -item.gt_class_count,
            -min(item.correct_pixels, item.error_pixels),
            -item.boundary_error_pixels,
            item.inline,
        ),
    )


def _section_statistics(
    section: r01.FullSection,
    prediction: np.ndarray,
    coverage: Mapping[str, Any],
) -> SectionPrediction:
    label = section.label
    if label.shape != prediction.shape:
        raise ValueError("diagnostic label/prediction shape mismatch")
    error = prediction != label
    boundary = _boundary_mask(label)
    boundary_error = np.logical_and(boundary, error)
    total_pixels = int(label.size)
    error_pixels = int(error.sum())
    correct_pixels = total_pixels - error_pixels
    gt_class_count = int(np.unique(label).size)
    return SectionPrediction(
        inline=section.inline,
        sample_id=f"{section.task_id}:validation:inline={section.inline}",
        gt_class_count=gt_class_count,
        correct_pixels=correct_pixels,
        error_pixels=error_pixels,
        boundary_pixels=int(boundary.sum()),
        boundary_error_pixels=int(boundary_error.sum()),
        total_pixels=total_pixels,
        error_fraction=error_pixels / total_pixels,
        coverage=dict(coverage),
        has_informative_signal=gt_class_count >= 2 and correct_pixels > 0 and error_pixels > 0,
    )


def _evaluate_full_validation(
    model: nn.Module,
    material: TaskMaterial,
    *,
    context_mode: str,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[SectionPrediction], dict[int, np.ndarray]]:
    accumulator = CalibrationAccumulator(num_classes)
    section_stats: list[SectionPrediction] = []
    predictions: dict[int, np.ndarray] = {}
    for inline in sorted(material.validation_sections):
        section = material.validation_sections[inline]
        refs = tuple(
            ref for ref in material.validation_window_refs if ref.inline == inline
        )
        prediction, probabilities, coverage, arrays_payload = _section_forward(
            model,
            material.validation_sections,
            section,
            refs,
            material.normalization,
            context_mode=context_mode,
            split_bounds=material.validation_range,
            batch_size=batch_size,
            num_classes=num_classes,
            device=device,
            arrays=True,
        )
        accumulator.update(probabilities[None], section.label[None])
        section_stats.append(_section_statistics(section, prediction, coverage))
        predictions[inline] = prediction
    metrics = accumulator.finalize()
    metrics.update(
        {
            "full_validation_inline_count": len(material.validation_sections),
            "full_validation_valid_voxels": int(
                sum(item.total_pixels for item in section_stats)
            ),
            "unique_scored_voxels": int(sum(item.total_pixels for item in section_stats)),
            "duplicate_prediction_assignments_before_blend": int(
                sum(item.coverage["duplicate_prediction_assignments_before_blend"] for item in section_stats)
            ),
            "coverage_fraction": 1.0,
            "coverage_by_inline": [dict(item.coverage) for item in section_stats],
            "boundary_error_rate": float(
                sum(item.boundary_error_pixels for item in section_stats)
                / max(1, sum(item.boundary_pixels for item in section_stats))
            ),
        }
    )
    return metrics, section_stats, predictions


def _run_one_training(
    *,
    material: TaskMaterial,
    run_spec: RunSpec,
    budget: R2Budget,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
    gpu_wait_seconds: float,
) -> RunOutcome:
    run_root = runtime_root / run_spec.task_id / run_spec.recipe_id / f"seed_{run_spec.repeat_id}"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    task_spec = get_task_spec(run_spec.task_id)
    model_config: dict[str, Any] = {"lane": "scratch"}
    if run_spec.context_mode == "2p5d":
        model_config["in_channels"] = 3
    model = discover_model("facies", run_spec.model_id).build(
        task_spec,
        num_classes=material.num_classes,
        **model_config,
    )
    if not isinstance(model, nn.Module):
        raise TypeError("facies model registry did not return nn.Module")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=budget.learning_rate,
        weight_decay=budget.weight_decay,
    )
    ce_loss = build_loss(
        "cross_entropy",
        num_classes=material.num_classes,
        class_weights=torch.tensor(material.class_weights, dtype=torch.float32, device=device),
    ).to(device)
    dice_loss = build_loss(
        "cross_entropy_plus_dice",
        num_classes=material.num_classes,
        class_weights=torch.tensor(material.class_weights, dtype=torch.float32, device=device),
    ).to(device)
    lovasz_loss = build_loss(
        "cross_entropy_plus_lovasz",
        num_classes=material.num_classes,
        class_weights=torch.tensor(material.class_weights, dtype=torch.float32, device=device),
        compound_weight=1.0,
    ).to(device)
    current_state: dict[str, Any] = {
        "epoch": 0,
        "current_train_loss": None,
        "history": [],
        "best_epoch": -1,
        "best_val_loss": float("inf"),
        "best_path": None,
        "epoch_to_checkpoint": {},
    }
    seed_report = seed_everything(
        derive_seed(ROOT_SEED, "r2", run_spec.task_id, run_spec.model_id, run_spec.repeat_id),
        strict=False,
    ).to_dict()
    schedule_seed = derive_seed(
        ROOT_SEED,
        "r2",
        run_spec.task_id,
        run_spec.model_id,
        run_spec.context_mode,
        run_spec.loss_id,
        run_spec.repeat_id,
    )
    schedule = np.random.default_rng(schedule_seed).integers(
        0,
        len(material.train_window_refs),
        size=(run_spec.epochs, budget.batch_size),
        endpoint=False,
        dtype=np.int64,
    )
    train_batches_factory = _EpochBatchFactory(
        sections=material.train_sections,
        refs=material.train_window_refs,
        stats=material.normalization,
        context_mode=run_spec.context_mode,
        schedule=schedule,
        split_bounds=material.train_range,
    )

    def val_batches_factory() -> Iterable[dict[str, Any]]:
        return [
            _proxy_validation_batch(
                sections=material.validation_sections,
                refs=material.validation_proxy_refs,
                stats=material.normalization,
                context_mode=run_spec.context_mode,
                split_bounds=material.validation_range,
                epoch=current_state["epoch"] + 1,
            )
        ]

    def _train_loss_fn(epoch: int) -> nn.Module:
        if run_spec.loss_id == "cross_entropy":
            return ce_loss
        if run_spec.loss_id == "cross_entropy_plus_dice":
            return dice_loss
        if run_spec.loss_id == "cross_entropy_to_lovasz":
            return ce_loss if epoch <= 500 else lovasz_loss
        raise ValueError(f"unsupported loss recipe {run_spec.loss_id!r}")

    def train_step(batch: Mapping[str, Any]) -> float:
        epoch = int(batch["epoch"])
        images = torch.as_tensor(batch["images"], dtype=torch.float32, device=device)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if tuple(logits.shape) != (images.shape[0], material.num_classes, images.shape[2], images.shape[3]):
            raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")
        loss_fn = _train_loss_fn(epoch)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("training loss is NaN/Inf")
        loss.backward()
        optimizer.step()
        current_state["epoch"] = epoch
        current_state["current_train_loss"] = float(loss.detach())
        return current_state["current_train_loss"]

    @torch.no_grad()
    def val_step(batch: Mapping[str, Any]) -> float:
        epoch = int(batch["epoch"])
        images = torch.as_tensor(batch["images"], dtype=torch.float32, device=device)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
        model.eval()
        logits = model(images)
        loss_fn = _train_loss_fn(epoch)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("validation loss is NaN/Inf")
        val_loss = float(loss.detach())
        row = {
            "epoch": epoch,
            "train_loss": current_state["current_train_loss"],
            "val_loss": val_loss,
        }
        current_state["history"].append(row)
        if val_loss < current_state["best_val_loss"]:
            current_state["best_val_loss"] = val_loss
            current_state["best_epoch"] = epoch - 1
        return val_loss

    selected_epochs = set(run_spec.endpoint_epochs)

    def save_checkpoint_fn(model_obj: nn.Module, path: Path) -> None:
        epoch = int(current_state["epoch"])
        if epoch not in selected_epochs and Path(path).name not in {"best.ckpt", "last.ckpt"}:
            return
        payload = {
            "epoch": epoch,
            "model_state": {name: value.detach().cpu() for name, value in model_obj.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "scaler_state": None,
            "config_hash": hash_payload(
                {
                    "run_spec": asdict(run_spec),
                    "budget": asdict(budget),
                    "source_lock": dict(source_lock(run_spec.model_id)),
                    "model_config": model_config,
                    "split_hash": material.split_hash,
                }
            ),
            "split_hash": material.split_hash,
            "trainer_state": {
                "next_epoch": epoch + 1,
                "global_step": epoch,
                "best_epoch": current_state["best_epoch"],
                "best_val_loss": current_state["best_val_loss"],
                "epochs_without_improvement": 0,
                "stopped_early": False,
                "history": list(current_state["history"]),
            },
            "seed_report": seed_report,
            "environment": _environment(device),
            "extra": {
                "schema_version": RESULT_SCHEMA,
                "task_id": run_spec.task_id,
                "model_id": run_spec.model_id,
                "recipe_id": run_spec.recipe_id,
                "context_mode": run_spec.context_mode,
                "endpoint_epochs": list(run_spec.endpoint_epochs),
                "selected_epoch": epoch,
                "test_archive_opened": False,
                "test_labels_read": False,
                "known_holdout_predictions_or_metrics_read": False,
                "fresh_blind": False,
            },
        }
        save_checkpoint(
            path,
            epoch=epoch,
            model_state=payload["model_state"],
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload["scheduler_state"],
            scaler_state=payload["scaler_state"],
            config_hash=payload["config_hash"],
            split_hash=payload["split_hash"],
            trainer_state=payload["trainer_state"],
            seed_report=payload["seed_report"],
            environment=payload["environment"],
            extra=payload["extra"],
        )
        if epoch in selected_epochs:
            endpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.ckpt"
            if endpoint_path != path:
                save_checkpoint(
                    endpoint_path,
                    epoch=epoch,
                    model_state=payload["model_state"],
                    optimizer_state=payload["optimizer_state"],
                    scheduler_state=payload["scheduler_state"],
                    scaler_state=payload["scaler_state"],
                    config_hash=payload["config_hash"],
                    split_hash=payload["split_hash"],
                    trainer_state=payload["trainer_state"],
                    seed_report=payload["seed_report"],
                    environment=payload["environment"],
                    extra=payload["extra"],
                )
            current_state["epoch_to_checkpoint"][epoch] = endpoint_path

    checkpoint_paths = {epoch: checkpoint_dir / f"epoch_{epoch:04d}.ckpt" for epoch in selected_epochs}
    existing_history_path = run_root / "history.json"
    can_resume_evaluation = (
        existing_history_path.is_file()
        and (checkpoint_dir / "best.ckpt").is_file()
        and (checkpoint_dir / "last.ckpt").is_file()
        and all(path.is_file() for path in checkpoint_paths.values())
    )
    if can_resume_evaluation:
        history_payload = json.loads(existing_history_path.read_text(encoding="utf-8"))
        history = TrainHistory(
            train_loss=[float(value) for value in history_payload["train_loss"]],
            val_loss=[float(value) for value in history_payload["val_loss"]],
            best_epoch=int(history_payload["best_epoch"]),
            best_val_loss=float(history_payload["best_val_loss"]),
        )
        if len(history.train_loss) != run_spec.epochs or len(history.val_loss) != run_spec.epochs:
            raise ProtocolBlocked(f"incomplete persisted history for evaluation resume: {run_spec.key}")
        train_wall_seconds = 0.0
        history_path = existing_history_path
        loss_curve_path = run_root / "loss_curve.png"
        if not loss_curve_path.is_file():
            loss_curve_path = plot_loss_curve(history, loss_curve_path)
    else:
        started = time.perf_counter()
        history = train_loop(
            model,
            train_step,
            val_step,
            train_batches_factory,
            val_batches_factory,
            run_spec.epochs,
            save_checkpoint_fn,
            checkpoint_dir,
            min_epochs_before_early_check=10,
        )
        train_wall_seconds = time.perf_counter() - started
        history_path = atomic_write_json(run_root / "history.json", history.to_dict())
        loss_curve_path = plot_loss_curve(history, run_root / "loss_curve.png")
    best_path = checkpoint_dir / "best.ckpt"
    last_path = checkpoint_dir / "last.ckpt"
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("train_loop did not persist best/last checkpoints")
    best = load_checkpoint(best_path)
    best_epoch = int(best["trainer_state"]["best_epoch"])
    best_val_loss = float(best["trainer_state"]["best_val_loss"])
    for epoch, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

    endpoint_results: list[dict[str, Any]] = []
    validation_records: list[SectionPrediction] = []
    selected_diagnostic: SectionPrediction | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_path: Path | None = None
    eval_started = time.perf_counter()
    for epoch in run_spec.endpoint_epochs:
        path = checkpoint_paths[epoch]
        checkpoint = load_checkpoint(path)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        metrics, section_stats, _ = _evaluate_full_validation(
            model,
            material,
            context_mode=run_spec.context_mode,
            batch_size=budget.eval_batch_size,
            num_classes=material.num_classes,
            device=device,
        )
        if not validation_records:
            validation_records = section_stats
        endpoint_results.append(
            {
                "epoch": epoch,
                "checkpoint_path": _portable_path(path, runtime_root),
                "checkpoint_sha256": hash_file(path),
                "checkpoint_bytes": path.stat().st_size,
                "metrics": metrics,
            }
        )
    eval_wall_seconds = time.perf_counter() - eval_started
    diagnostic_checkpoint = best_path
    checkpoint = load_checkpoint(diagnostic_checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _, section_stats, _ = _evaluate_full_validation(
        model,
        material,
        context_mode=run_spec.context_mode,
        batch_size=budget.eval_batch_size,
        num_classes=material.num_classes,
        device=device,
    )
    selected_diagnostic = _select_diagnostic_section(section_stats)
    selected_checkpoint_epoch = best_epoch + 1
    selected_checkpoint_path = diagnostic_checkpoint
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return RunOutcome(
        spec=run_spec,
        status="completed",
        history=history,
        history_path=history_path,
        loss_curve_path=loss_curve_path,
        checkpoint_paths=checkpoint_paths,
        best_checkpoint_path=best_path,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        endpoint_results=endpoint_results,
        validation_records=validation_records,
        selected_diagnostic=selected_diagnostic,
        selected_checkpoint_epoch=selected_checkpoint_epoch,
        selected_checkpoint_path=selected_checkpoint_path,
        train_wall_seconds=train_wall_seconds,
        eval_wall_seconds=eval_wall_seconds,
        gpu_peak_bytes=peak_vram,
        gpu_wait_seconds=gpu_wait_seconds,
    )


def _prediction_arrays_for_selection(
    *,
    model: nn.Module,
    material: TaskMaterial,
    diagnostic: SectionPrediction,
    context_mode: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    section = material.validation_sections[diagnostic.inline]
    refs = tuple(ref for ref in material.validation_window_refs if ref.inline == diagnostic.inline)
    prediction, probabilities, coverage, arrays = _section_forward(
        model,
        material.validation_sections,
        section,
        refs,
        material.normalization,
        context_mode=context_mode,
        split_bounds=material.validation_range,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_classes=material.num_classes,
        device=device,
        arrays=True,
    )
    assert arrays is not None
    arrays["prediction"] = prediction.astype(np.uint8)
    arrays["probabilities"] = probabilities.astype(np.float32)
    arrays["coverage"] = np.array([coverage["duplicate_prediction_assignments_before_blend"]], dtype=np.int64)
    return arrays, coverage


def _render_learning_curve(
    *,
    task_id: str,
    records: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    task_records = [
        row
        for row in records
        if row["task_id"] == task_id
        and row["recipe_id"] == "ce_2d"
        and row["status"] == "completed"
        and int(row["endpoint_epoch"]) == int(row["endpoint_epochs"][-1])
    ]
    if not task_records:
        raise ValueError(f"no 1000-update CE records found for {task_id}")
    fig, ax = plt.subplots(figsize=(8, 5))
    epoch_40_values: list[float] = []
    epoch_400_values: list[float] = []
    epoch_1000_values: list[float] = []
    for row in sorted(task_records, key=lambda item: item["repeat_id"]):
        history = row["history"]
        train_loss = [float(value) for value in history["train_loss"]]
        val_loss = [float(value) for value in history["val_loss"]]
        if not train_loss or len(train_loss) != len(val_loss):
            raise ValueError(f"malformed persisted loss history for {row['run_key']}")
        epochs = list(range(1, len(train_loss) + 1))
        ax.plot(
            epochs,
            train_loss,
            color="#5B8FF9",
            alpha=0.35,
        )
        ax.plot(
            epochs,
            val_loss,
            color="#F6BD16",
            alpha=0.35,
        )
        epoch_40_values.append(next(item for item in row["endpoint_results"] if item["epoch"] == 40)["metrics"]["miou"])
        epoch_400_values.append(next(item for item in row["endpoint_results"] if item["epoch"] == 400)["metrics"]["miou"])
        epoch_1000_values.append(next(item for item in row["endpoint_results"] if item["epoch"] == 1000)["metrics"]["miou"])
    ax.axvline(40, color="#999999", linestyle="--", linewidth=1)
    ax.axvline(400, color="#999999", linestyle="--", linewidth=1)
    ax.axvline(1000, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("epoch / update")
    ax.set_ylabel("loss")
    ax.set_title(
        f"{task_id} 2-D CE learning curves | mIoU@40={np.mean(epoch_40_values):.4f}, "
        f"@400={np.mean(epoch_400_values):.4f}, @1000={np.mean(epoch_1000_values):.4f}"
    )
    ax.legend(["train loss", "val loss"], loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _render_loss_ablation(
    *,
    task_id: str,
    records: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    recipes = ("ce_2d", "ce_plus_dice_2d", "ce_to_lovasz_2d", "ce_2p5d")
    labels = {
        "ce_2d": "2D CE",
        "ce_plus_dice_2d": "2D CE+Dice",
        "ce_to_lovasz_2d": "2D CE→Lovasz",
        "ce_2p5d": "2.5D CE",
    }
    means: list[float] = []
    stds: list[float] = []
    for recipe in recipes:
        values = [
            next(item for item in row["endpoint_results"] if item["epoch"] == row["endpoint_epochs"][-1])["metrics"]["miou"]
            for row in records
            if row["task_id"] == task_id
            and row["recipe_id"] == recipe
            and row["status"] == "completed"
            and int(row["endpoint_epoch"]) == int(row["endpoint_epochs"][-1])
        ]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=0)))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(recipes))
    ax.bar(x, means, yerr=stds, color=["#5B8FF9", "#5AD8A6", "#F6BD16", "#E8684A"], capsize=4)
    ax.set_xticks(x, [labels[recipe] for recipe in recipes], rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mIoU @ endpoint")
    ax.set_title(f"{task_id} loss/context ablation")
    for index, value in enumerate(means):
        ax.text(index, min(0.98, value + 0.02), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _render_context_ablation(
    *,
    task_id: str,
    records: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    recipes = ("ce_2d", "ce_2p5d")
    labels = {"ce_2d": "2D CE", "ce_2p5d": "2.5D CE"}
    means: list[float] = []
    stds: list[float] = []
    for recipe in recipes:
        values = [
            next(item for item in row["endpoint_results"] if item["epoch"] == row["endpoint_epochs"][-1])["metrics"]["miou"]
            for row in records
            if row["task_id"] == task_id
            and row["recipe_id"] == recipe
            and row["status"] == "completed"
            and int(row["endpoint_epoch"]) == int(row["endpoint_epochs"][-1])
        ]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=0)))
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(recipes))
    ax.bar(x, means, yerr=stds, color=["#5B8FF9", "#E8684A"], capsize=4)
    ax.set_xticks(x, [labels[recipe] for recipe in recipes])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mIoU @ endpoint")
    ax.set_title(f"{task_id} context ablation")
    for index, value in enumerate(means):
        ax.text(index, min(0.98, value + 0.02), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _render_diagnostics(
    *,
    task_id: str,
    checkpoint_path: Path,
    material: TaskMaterial,
    diagnostic: SectionPrediction,
    context_mode: str,
    runtime_root: Path,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    task_spec = get_task_spec(task_id)
    model_id = CONTROL_MODELS[task_id]
    model_config: dict[str, Any] = {"lane": "scratch"}
    if context_mode == "2p5d":
        model_config["in_channels"] = 3
    model = discover_model("facies", model_id).build(
        task_spec,
        num_classes=material.num_classes,
        **model_config,
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    section = material.validation_sections[diagnostic.inline]
    refs = tuple(ref for ref in material.validation_window_refs if ref.inline == diagnostic.inline)
    arrays, coverage = _prediction_arrays_for_selection(
        model=model,
        material=material,
        diagnostic=diagnostic,
        context_mode=context_mode,
        device=device,
    )
    gt = arrays["labels"]
    pred = arrays["prediction"]
    confidence = arrays["confidence"]
    error = arrays["error"]
    boundary = arrays["boundary"]
    probability = arrays["probabilities"]
    class_count = task_spec.metadata["num_classes"]
    confusion = confusion_matrix(gt, pred, class_count)
    support = confusion.sum(axis=1)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = support - tp
    iou = np.divide(tp, tp + fp + fn, out=np.zeros(class_count), where=(tp + fp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(class_count), where=(2 * tp + fp + fn) > 0)

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.9))
    cmap = plt.get_cmap("tab10", class_count)
    seismic = arrays["seismic"]
    amplitude = float(np.percentile(np.abs(seismic), 99.0)) or 1.0
    panels = [
        (seismic, "gray", "seismic", -amplitude, amplitude),
        (gt, cmap, "ground truth", -0.5, class_count - 0.5),
        (pred, cmap, "prediction", -0.5, class_count - 0.5),
        (confidence, "viridis", "confidence", 0.0, 1.0),
        (error, "magma", "error", 0.0, 1.0),
        (boundary, "cividis", "boundary", 0.0, 1.0),
    ]
    for index, (values, panel_cmap, title, vmin, vmax) in enumerate(panels):
        axis = fig.add_subplot(grid[index // 3, index % 3])
        axis.imshow(values, cmap=panel_cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    confusion_axis = fig.add_subplot(grid[1, 3])
    confusion_image = confusion_axis.imshow(
        confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1),
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    confusion_axis.set_title("confusion")
    confusion_axis.set_xlabel("pred")
    confusion_axis.set_ylabel("gt")
    confusion_axis.set_xticks(range(class_count))
    confusion_axis.set_yticks(range(class_count))
    fig.colorbar(confusion_image, ax=confusion_axis, fraction=0.046)

    class_axis = fig.add_subplot(grid[2, 0:2])
    pos = np.arange(class_count)
    class_axis.bar(pos - 0.18, iou, width=0.36, label="IoU", color="#5B8FF9")
    class_axis.bar(pos + 0.18, f1, width=0.36, label="F1", color="#F6BD16")
    class_axis.set_ylim(0.0, 1.0)
    class_axis.set_xticks(pos)
    class_axis.set_xlabel("class")
    class_axis.set_title("per-class IoU / F1")
    class_axis.legend()

    reliability_axis = fig.add_subplot(grid[2, 2:4])
    top_conf = probability.max(axis=0).reshape(-1)
    top_pred = probability.argmax(axis=0).reshape(-1)
    top_truth = gt.reshape(-1)
    bins = np.linspace(0.0, 1.0, 16)
    mean_conf = []
    mean_acc = []
    centers = []
    for index in range(len(bins) - 1):
        lower, upper = bins[index], bins[index + 1]
        mask = (top_conf >= lower) & ((top_conf < upper) if index < len(bins) - 2 else (top_conf <= upper))
        centers.append((lower + upper) / 2.0)
        if np.any(mask):
            mean_conf.append(float(top_conf[mask].mean()))
            mean_acc.append(float((top_pred[mask] == top_truth[mask]).mean()))
        else:
            mean_conf.append(np.nan)
            mean_acc.append(np.nan)
    reliability_axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    reliability_axis.plot(centers, mean_acc, marker="o", label="accuracy")
    reliability_axis.plot(centers, mean_conf, marker="s", label="confidence")
    reliability_axis.set_ylim(0.0, 1.0)
    reliability_axis.set_xlim(0.0, 1.0)
    reliability_axis.set_xlabel("confidence")
    reliability_axis.set_ylabel("value")
    reliability_axis.set_title("reliability")
    reliability_axis.legend()

    fig.suptitle(
        f"{task_id} R2 diagnostics | selection={diagnostic.sample_id} | "
        f"boundary_error_rate={diagnostic.boundary_error_pixels / max(1, diagnostic.boundary_pixels):.4f}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    manifest = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "task_id": task_id,
        "checkpoint_runtime_relative_path": _portable_path(checkpoint_path, runtime_root),
        "checkpoint_sha256": hash_file(checkpoint_path),
        "figure": output_path.name,
        "figure_sha256": hash_file(output_path),
        "selection_rule": "prefer_validation_section_with_>=2_gt_classes_and_correct_plus_error_pixels",
        "selection_sample_id": diagnostic.sample_id,
        "selection_inline": diagnostic.inline,
        "selection_statistics": asdict(diagnostic),
        "firewall": {
            "test_archive_opened": False,
            "test_labels_read": False,
            "known_holdout_predictions_or_metrics_read": False,
            "fresh_blind": False,
        },
        "storage_boundary": "track_private_portable_evidence",
    }
    return manifest


def _flatten_rows(task_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in task_results:
        rows.extend(result["records"])
    return rows


def _record_for_failure(
    *,
    task_id: str,
    run_spec: RunSpec,
    endpoint: int,
    material: TaskMaterial,
    budget: R2Budget,
    status: str,
    reason: str,
) -> dict[str, Any]:
    spec = get_task_spec(task_id)
    return {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "task_id": task_id,
        "label_version": spec.label_version,
        "num_classes": material.num_classes,
        "model_id": run_spec.model_id,
        "model_seed": run_spec.model_seed,
        "repeat_id": run_spec.repeat_id,
        "recipe_id": run_spec.recipe_id,
        "context_mode": run_spec.context_mode,
        "loss_id": run_spec.loss_id,
        "endpoint_epoch": endpoint,
        "run_key": run_spec.key,
        "status": status,
        "failure": {"type": "RuntimeError", "reason": reason},
        "budget": asdict(budget),
        "split": {
            "development_range": list(material.development_range),
            "train_range": list(material.train_range),
            "guard_range": list(material.guard_range),
            "validation_range": list(material.validation_range),
            "train_window_count": len(material.train_window_refs),
            "validation_window_count": len(material.validation_window_refs),
            "validation_proxy_window_count": len(material.validation_proxy_refs),
            "split_hash": material.split_hash,
        },
        "source_lock": dict(source_lock(run_spec.model_id)),
        "source_hashes": {
            "source_lock_sha256": material.source_lock_sha256,
            "r01_source_sha256": material.r01_source_sha256,
            "adapter_sha256": material.adapter_sha256,
        },
        "test_archive_opened": False,
        "test_labels_read": False,
        "known_holdout_predictions_or_metrics_read": False,
        "fresh_blind": False,
    }


def _task_rows(records: Sequence[Mapping[str, Any]], task_id: str) -> list[Mapping[str, Any]]:
    return [row for row in records if row["task_id"] == task_id]


def _endpoint_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    recipe_id: str | None = None,
    endpoint_epoch: int | None = None,
) -> list[Mapping[str, Any]]:
    filtered = [row for row in records if row["task_id"] == task_id and row["status"] == "completed"]
    if recipe_id is not None:
        filtered = [row for row in filtered if row["recipe_id"] == recipe_id]
    if endpoint_epoch is not None:
        filtered = [row for row in filtered if int(row["endpoint_epoch"]) == int(endpoint_epoch)]
    return filtered


def _run_level_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    recipe_id: str | None = None,
) -> list[Mapping[str, Any]]:
    rows = _endpoint_rows(records, task_id=task_id, recipe_id=recipe_id, endpoint_epoch=1000)
    return [
        row
        for row in rows
        if row.get("history") and row.get("endpoint_results") and row.get("endpoint_epochs")
    ]


def _task_result_rows(records: Sequence[Mapping[str, Any]], task_id: str) -> list[Mapping[str, Any]]:
    return [row for row in records if row["task_id"] == task_id and row["status"] == "completed"]


def _plateau_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = {
        row["repeat_id"]
        for row in rows
        if row["recipe_id"] == "ce_2d" and int(row["endpoint_epoch"]) in (400, 1000)
    }
    rows_400 = {
        row["repeat_id"]: row
        for row in rows
        if row["recipe_id"] == "ce_2d" and int(row["endpoint_epoch"]) == 400
    }
    rows_1000 = {
        row["repeat_id"]: row
        for row in rows
        if row["recipe_id"] == "ce_2d" and int(row["endpoint_epoch"]) == 1000
    }
    gains: list[float] = []
    for repeat_id in sorted(set(rows_400) & set(rows_1000)):
        gains.append(
            float(rows_1000[repeat_id]["validation_metrics"]["miou"])
            - float(rows_400[repeat_id]["validation_metrics"]["miou"])
        )
    median_gain = float(np.median(gains)) if gains else float("nan")
    max_gain = float(np.max(gains)) if gains else float("nan")
    ready = len(rows_400) == 3 and len(rows_1000) == 3 and median_gain <= 0.005 and max_gain <= 0.01
    return {
        "all_three_seeds_completed": len(rows_400) == 3 and len(rows_1000) == 3,
        "seed_gains": gains,
        "median_gain": median_gain,
        "max_gain": max_gain,
        "not_ready_for_r3": not ready,
        "ready_for_r3": ready,
    }


def _load_resume_records(results_path: Path, budget: R2Budget) -> list[dict[str, Any]]:
    results_path = Path(results_path)
    config_path = results_path.parent / "p5_r2_config.json"
    if not results_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("resume requires both p5_r2_results.jsonl and p5_r2_config.json")
    config = _read_json(config_path)
    if config.get("budget") != asdict(budget):
        raise ProtocolBlocked("resume results use a different frozen R2 budget")
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 36 or any(row.get("status") != "completed" for row in rows):
        raise ProtocolBlocked("resume requires exactly 36 completed endpoint rows")
    keys = {
        (
            row["task_id"],
            row["recipe_id"],
            int(row["repeat_id"]),
            int(row["endpoint_epoch"]),
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise ProtocolBlocked("resume results contain duplicate endpoint keys")
    for row in rows:
        row["training"]["loss_curve_path"] = (
            f"{row['task_id']}/{row['recipe_id']}/seed_{int(row['repeat_id'])}/loss_curve.png"
        )
    return rows


def _rebuild_task_diagnostic(
    *,
    row: Mapping[str, Any],
    material: TaskMaterial,
    budget: R2Budget,
    runtime_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    context_mode = str(row["context_mode"])
    model_config: dict[str, Any] = {"lane": "scratch"}
    if context_mode == "2p5d":
        model_config["in_channels"] = 3
    model = discover_model("facies", str(row["model_id"])).build(
        get_task_spec(material.task_id),
        num_classes=material.num_classes,
        **model_config,
    ).to(device)
    best_path = (
        runtime_root
        / material.task_id
        / str(row["recipe_id"])
        / f"seed_{int(row['repeat_id'])}"
        / "checkpoints"
        / "best.ckpt"
    )
    if not best_path.is_file():
        raise FileNotFoundError(f"resume diagnostic checkpoint missing: {best_path}")
    checkpoint = load_checkpoint(best_path)
    if checkpoint["split_hash"] != material.split_hash:
        raise ProtocolBlocked(f"resume checkpoint split hash mismatch: {material.task_id}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _, section_stats, _ = _evaluate_full_validation(
        model,
        material,
        context_mode=context_mode,
        batch_size=budget.eval_batch_size,
        num_classes=material.num_classes,
        device=device,
    )
    return {
        "diagnostic": _select_diagnostic_section(section_stats),
        "checkpoint_path": best_path,
        "context_mode": context_mode,
        "run_key": row["run_key"],
    }


def run_r2(
    *,
    data_root: Path,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
    budget: R2Budget | None = None,
    gpu_lock_path: Path = EXPECTED_GPU_LOCK,
    resume_results_path: Path | None = None,
) -> dict[str, Any]:
    active_budget = R2Budget() if budget is None else budget
    if set(TASK_IDS) != {"facies_f3", "facies_penobscot"}:
        raise ValueError("R2 expects the two independent facies tasks only")
    if device.type == "cuda" and Path(gpu_lock_path) != EXPECTED_GPU_LOCK:
        raise ValueError(f"CUDA R2 must flock the frozen lock {EXPECTED_GPU_LOCK}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty R2 output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    materials = {task_id: load_task_material(task_id, data_root, active_budget) for task_id in TASK_IDS}
    records: list[dict[str, Any]] = []
    task_summaries: dict[str, dict[str, Any]] = {}
    task_diagnostics: dict[str, dict[str, Any]] = {}
    task_runtime_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if resume_results_path is not None:
        records = _load_resume_records(Path(resume_results_path), active_budget)
    gpu_wait_seconds = 0.0
    with GpuFlock(gpu_lock_path) as lock:
        gpu_wait_seconds = lock.wait_seconds
        if resume_results_path is not None:
            for task_id in TASK_IDS:
                candidates = [
                    row
                    for row in records
                    if row["task_id"] == task_id
                    and row["recipe_id"] == "ce_2d"
                    and int(row["repeat_id"]) == 1
                    and int(row["endpoint_epoch"]) == 1000
                ]
                if len(candidates) != 1:
                    raise ProtocolBlocked(f"resume diagnostic row missing or ambiguous: {task_id}")
                task_diagnostics[task_id] = _rebuild_task_diagnostic(
                    row=candidates[0],
                    material=materials[task_id],
                    budget=active_budget,
                    runtime_root=runtime_root,
                    device=device,
                )
        for run_spec in (() if resume_results_path is not None else build_run_specs()):
            material = materials[run_spec.task_id]
            try:
                outcome = _run_one_training(
                    material=material,
                    run_spec=run_spec,
                    budget=active_budget,
                    output_root=output_root,
                    runtime_root=runtime_root,
                    device=device,
                    gpu_wait_seconds=lock.wait_seconds,
                )
                if outcome.status != "completed":
                    raise RuntimeError(outcome.status)
                if outcome.history is None or outcome.history_path is None or outcome.loss_curve_path is None:
                    raise RuntimeError("completed run missing run-level history artifacts")
                for endpoint_record in outcome.endpoint_results:
                    row = {
                        "schema_version": RESULT_SCHEMA,
                        "track_id": "facies",
                        "task_id": run_spec.task_id,
                        "label_version": material.label_version,
                        "num_classes": material.num_classes,
                        "model_id": run_spec.model_id,
                        "model_seed": run_spec.model_seed,
                        "repeat_id": run_spec.repeat_id,
                        "recipe_id": run_spec.recipe_id,
                        "context_mode": run_spec.context_mode,
                        "loss_id": run_spec.loss_id,
                        "run_key": run_spec.key,
                        "endpoint_epoch": endpoint_record["epoch"],
                        "status": "completed",
                        "budget": asdict(active_budget),
                        "split": {
                            "development_range": list(material.development_range),
                            "train_range": list(material.train_range),
                            "guard_range": list(material.guard_range),
                            "validation_range": list(material.validation_range),
                            "train_window_count": len(material.train_window_refs),
                            "validation_window_count": len(material.validation_window_refs),
                            "validation_proxy_window_count": len(material.validation_proxy_refs),
                            "split_hash": material.split_hash,
                        },
                        "source_lock": dict(source_lock(run_spec.model_id)),
                        "source_hashes": {
                            "source_lock_sha256": material.source_lock_sha256,
                            "r01_source_sha256": material.r01_source_sha256,
                            "adapter_sha256": material.adapter_sha256,
                        },
                        "checkpoint": {
                            "runtime_relative_path": endpoint_record["checkpoint_path"],
                            "sha256": endpoint_record["checkpoint_sha256"],
                            "bytes": endpoint_record["checkpoint_bytes"],
                            "epoch": endpoint_record["epoch"],
                        },
                        "training": {
                            "history_path": _portable_path(outcome.history_path, runtime_root) if outcome.history_path else None,
                            "loss_curve_path": _portable_path(outcome.loss_curve_path, output_root) if outcome.loss_curve_path else None,
                            "best_epoch": outcome.best_epoch,
                            "best_val_loss": outcome.best_val_loss,
                            "train_wall_seconds": outcome.train_wall_seconds,
                            "eval_wall_seconds": outcome.eval_wall_seconds,
                        },
                        "endpoint_epochs": list(run_spec.endpoint_epochs),
                        "history": outcome.history.to_dict(),
                        "endpoint_results": outcome.endpoint_results,
                        "validation_metrics": endpoint_record["metrics"],
                        "resources": {
                            "device": str(device),
                            "gpu_lock_wait_seconds": lock.wait_seconds,
                            "wall_seconds": outcome.train_wall_seconds + outcome.eval_wall_seconds,
                            "cuda_peak_allocated_bytes": outcome.gpu_peak_bytes,
                        },
                        "test_archive_opened": False,
                        "test_labels_read": False,
                        "known_holdout_predictions_or_metrics_read": False,
                        "fresh_blind": False,
                    }
                    records.append(row)
                task_runtime_rows[run_spec.task_id].append(
                    {
                        "run_key": run_spec.key,
                        "task_id": run_spec.task_id,
                        "recipe_id": run_spec.recipe_id,
                        "repeat_id": run_spec.repeat_id,
                        "model_seed": run_spec.model_seed,
                        "history": outcome.history.to_dict(),
                        "endpoint_results": outcome.endpoint_results,
                        "endpoint_epochs": list(run_spec.endpoint_epochs),
                        "best_epoch": outcome.best_epoch,
                        "best_val_loss": outcome.best_val_loss,
                        "loss_curve_path": _portable_path(outcome.loss_curve_path, output_root),
                        "history_path": _portable_path(outcome.history_path, runtime_root),
                    }
                )
                if run_spec.recipe_id == "ce_2d" and run_spec.repeat_id == 1 and outcome.selected_diagnostic is not None and outcome.selected_checkpoint_path is not None:
                    task_diagnostics[run_spec.task_id] = {
                        "diagnostic": outcome.selected_diagnostic,
                        "checkpoint_path": outcome.selected_checkpoint_path,
                        "context_mode": run_spec.context_mode,
                        "run_key": run_spec.key,
                    }
            except Exception as exc:
                for endpoint in run_spec.endpoint_epochs:
                    records.append(
                        _record_for_failure(
                            task_id=run_spec.task_id,
                            run_spec=run_spec,
                            endpoint=endpoint,
                            material=material,
                            budget=active_budget,
                            status="failed",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
    order = {
        (task_id, recipe_id): index
        for index, (task_id, recipe_id) in enumerate(
            (task_id, recipe_id)
            for task_id in TASK_IDS
            for recipe_id in ("ce_2d", "ce_plus_dice_2d", "ce_to_lovasz_2d", "ce_2p5d")
        )
    }
    records.sort(
        key=lambda row: (
            TASK_IDS.index(row["task_id"]),
            order[(row["task_id"], row["recipe_id"])],
            int(row["repeat_id"]),
            int(row["endpoint_epoch"]),
        )
    )
    results_path = _atomic_write_jsonl(output_root / "p5_r2_results.jsonl", records)
    config_path = atomic_write_json(
        output_root / "p5_r2_config.json",
        {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            "root_seed": ROOT_SEED,
            "budget": asdict(active_budget),
            "task_ids": list(TASK_IDS),
            "control_models": dict(CONTROL_MODELS),
            "model_seeds": list(MODEL_SEEDS),
            "endpoints": list(ENDPOINTS),
            "gpu_lock_name": EXPECTED_GPU_LOCK.name,
            "source_hashes": {
                "source_lock_sha256": EXPECTED_SOURCE_LOCK_SHA256,
                "r01_source_sha256": EXPECTED_R01_SOURCE_SHA256,
                "control_adapter_sha256": dict(EXPECTED_CONTROL_ADAPTER_SHA256),
            },
            "test_archive_opened": False,
            "test_labels_read": False,
            "known_holdout_predictions_or_metrics_read": False,
            "fresh_blind": False,
        },
    )
    for task_id in TASK_IDS:
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        task_records = _task_rows(records, task_id)
        curve_rows = _run_level_rows(records, task_id=task_id, recipe_id="ce_2d")
        if len(curve_rows) != 3:
            raise ProtocolBlocked(f"{task_id} missing 3 CE endpoint rows for learning curve")
        learning_curve_path = _render_learning_curve(
            task_id=task_id,
            records=curve_rows,
            output_path=task_root / "p5_r2_learning_curve.png",
        )
        loss_ablation_path = _render_loss_ablation(
            task_id=task_id,
            records=task_records,
            output_path=task_root / "p5_r2_loss_ablation.png",
        )
        context_ablation_path = _render_context_ablation(
            task_id=task_id,
            records=task_records,
            output_path=task_root / "p5_r2_context_ablation.png",
        )
        diagnostic_record = task_diagnostics.get(task_id)
        if diagnostic_record is None:
            raise ProtocolBlocked(f"{task_id} diagnostic checkpoint was not selected")
        diagnostic_manifest = _render_diagnostics(
            task_id=task_id,
            checkpoint_path=Path(diagnostic_record["checkpoint_path"]),
            material=materials[task_id],
            diagnostic=diagnostic_record["diagnostic"],
            context_mode=diagnostic_record["context_mode"],
            runtime_root=runtime_root,
            device=device,
            output_path=task_root / "p5_r2_prediction_diagnostics.png",
        )
        diagnostic_manifest_path = atomic_write_json(
            task_root / "p5_r2_prediction_manifest.json",
            diagnostic_manifest,
        )
        ce2d_rows = _endpoint_rows(records, task_id=task_id, recipe_id="ce_2d")
        task_1000_rows = _endpoint_rows(records, task_id=task_id, endpoint_epoch=1000)
        if len(task_1000_rows) != 12:
            raise ProtocolBlocked(f"{task_id} expected 12 completed 1000-endpoint rows")
        endpoint_means = {
            endpoint: {
                metric: float(
                    np.mean(
                        [
                            row["validation_metrics"][metric]
                            for row in ce2d_rows
                            if int(row["endpoint_epoch"]) == endpoint
                        ]
                    )
                )
                for metric in ("accuracy", "miou", "macro_f1")
            }
            for endpoint in ENDPOINTS
        }
        loss_ablation_rows = {
            recipe_id: _endpoint_rows(records, task_id=task_id, recipe_id=recipe_id, endpoint_epoch=1000)
            for recipe_id in ("ce_2d", "ce_plus_dice_2d", "ce_to_lovasz_2d", "ce_2p5d")
        }
        lane_table = [
            {
                "recipe_id": recipe_id,
                "completed_runs": len(rows),
                "miou_mean": float(np.mean([row["validation_metrics"]["miou"] for row in rows])),
                "miou_std": float(np.std([row["validation_metrics"]["miou"] for row in rows], ddof=0)),
                "macro_f1_mean": float(np.mean([row["validation_metrics"]["macro_f1"] for row in rows])),
                "accuracy_mean": float(np.mean([row["validation_metrics"]["accuracy"] for row in rows])),
            }
            for recipe_id, rows in loss_ablation_rows.items()
        ]
        plateau = _plateau_decision(task_records)
        status_counts = {
            status: sum(row["status"] == status for row in task_records)
            for status in ("completed", "blocked", "failed", "timeout")
        }
        task_summary = {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            "task_id": task_id,
            "label_version": materials[task_id].label_version,
            "head_num_classes": materials[task_id].num_classes,
            "label_space_independent": True,
            "lane": "scratch",
            "control_model_id": CONTROL_MODELS[task_id],
            "status_counts": status_counts,
            "result_rows": len(task_records),
            "completed_endpoint_rows": len(_task_result_rows(records, task_id)),
            "completed_1000_endpoint_rows": len(task_1000_rows),
            "endpoint_means": endpoint_means,
            "lane_table": lane_table,
            "plateau_rule": plateau,
            "learning_curve": {
                "path": learning_curve_path.name,
                "sha256": hash_file(learning_curve_path),
            },
            "loss_ablation": {
                "path": loss_ablation_path.name,
                "sha256": hash_file(loss_ablation_path),
            },
            "context_ablation": {
                "path": context_ablation_path.name,
                "sha256": hash_file(context_ablation_path),
            },
            "prediction_diagnostics": {
                "path": Path(diagnostic_manifest["figure"]).name,
                "figure_sha256": diagnostic_manifest["figure_sha256"],
                "manifest_path": diagnostic_manifest_path.name,
                "manifest_sha256": hash_file(diagnostic_manifest_path),
                "selection_sample_id": diagnostic_manifest["selection_sample_id"],
                "selection_rule": diagnostic_manifest["selection_rule"],
            },
            "runtime_artifacts": {
                "history_path": "runtime_only",
                "checkpoint_path": "runtime_only",
                "loss_curve_path": "runtime_only",
            },
            "test_archive_opened": False,
            "test_labels_read": False,
            "known_holdout_predictions_or_metrics_read": False,
            "fresh_blind": False,
        }
        task_summary_path = atomic_write_json(task_root / "p5_r2_summary.json", task_summary)
        task_summaries[task_id] = {
            **task_summary,
            "summary_path": task_summary_path.name,
            "summary_sha256": hash_file(task_summary_path),
        }
    artifact_manifest = ArtifactManifest(run_id=RESULT_SCHEMA, root=output_root)
    artifact_manifest.register("p5_r2_results.jsonl", role="results")
    artifact_manifest.register("p5_r2_config.json", role="config")
    for task_id in TASK_IDS:
        task_root = output_root / task_id
        artifact_manifest.register(f"{task_id}/p5_r2_summary.json", role="task-summary")
        artifact_manifest.register(f"{task_id}/p5_r2_learning_curve.png", role="figure")
        artifact_manifest.register(f"{task_id}/p5_r2_loss_ablation.png", role="figure")
        artifact_manifest.register(f"{task_id}/p5_r2_context_ablation.png", role="figure")
        artifact_manifest.register(f"{task_id}/p5_r2_prediction_diagnostics.png", role="figure")
        artifact_manifest.register(f"{task_id}/p5_r2_prediction_manifest.json", role="manifest")
    summary = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "root_seed": ROOT_SEED,
        "budget": asdict(active_budget),
        "task_ids": list(TASK_IDS),
        "control_models": dict(CONTROL_MODELS),
        "model_seeds": list(MODEL_SEEDS),
        "endpoint_epochs": list(ENDPOINTS),
        "result_count": len(records),
        "status_counts": {
            status: sum(row["status"] == status for row in records)
            for status in ("completed", "blocked", "failed", "timeout")
        },
        "gpu_lock_wait_seconds": gpu_wait_seconds,
        "results_file": results_path.name,
        "results_sha256": hash_file(results_path),
        "config_file": config_path.name,
        "config_sha256": hash_file(config_path),
        "artifact_manifest": {
            "path": "p5_r2_artifact_manifest.json",
        },
        "tasks": task_summaries,
        "tasks_are_independent": True,
        "cross_task_ranking_forbidden": True,
        "test_archive_opened": False,
        "test_labels_read": False,
        "known_holdout_predictions_or_metrics_read": False,
        "fresh_blind": False,
    }
    summary_path = atomic_write_json(output_root / "p5_r2_summary.json", summary)
    artifact_manifest.register("p5_r2_summary.json", role="summary")
    artifact_manifest_path = artifact_manifest.write("p5_r2_artifact_manifest.json")
    artifact_manifest.verify()
    return {
        **summary,
        "summary_file": summary_path.name,
        "summary_sha256": hash_file(summary_path),
        "artifact_manifest_sha256": hash_file(artifact_manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PORTABLE_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-results", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_r2(
        data_root=args.data_root,
        output_root=args.output_root,
        runtime_root=args.runtime_root,
        device=resolve_device(args.device),
        resume_results_path=args.resume_results,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
