#!/usr/bin/env python3
"""P5 Stage-3 fixed top-3 × five-fold × three-seed facies confirmation.

The runner is development-only.  It accepts exactly the two locked P4 split
manifests and opens only each task's ``train.h5``.  F3 and Penobscot retain
independent TaskSpecs, class heads, OOF archives, figures and leaderboards.
Stage-2's sample caps, optimizer, loss, preprocessing policy and 40-update
budget are frozen; Stage-3 changes only the manifest fold and repeat model
seed.  No HPO or frozen-test entry point exists here.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
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

from _code.ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.preprocess import denormalize, fit_zscore, normalize  # noqa: E402
from _code.ml_framework.seeding import derive_seed  # noqa: E402
from _models.facies._p5_common import P5AdapterSkip, source_lock  # noqa: E402

import facies_p5_stage2 as stage2  # noqa: E402
from p4_data import FoldPreprocessor, inverse_sqrt_class_weights  # noqa: E402
from p4_losses import build_loss, softmax_probabilities  # noqa: E402
from p4_metrics import confidence_entropy_error, confusion_matrix  # noqa: E402
from p4_tasks import LABEL_VERSIONS, TASK_IDS, get_task_spec  # noqa: E402


ROOT_SEED = 2693
RESULT_SCHEMA = "facies-p5-stage3-v1"
EXPECTED_GPU_LOCK = Path("/mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock")
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
FOLD_IDS = (0, 1, 2, 3, 4)
LANE = "scratch"
TOP_MODELS: Mapping[str, tuple[str, ...]] = {
    "facies_f3": (
        "smp_fpn_r18",
        "smp_deeplabv3plus_r18",
        "hf_segformer_b0",
    ),
    "facies_penobscot": (
        "smp_deeplabv3plus_r18",
        "smp_fpn_r18",
        "smp_unet_r18",
    ),
}
EXPECTED_CELLS = 90
RANKABLE_COMPLETION_THRESHOLD = 0.80
DEFAULT_PORTABLE_OUTPUT = TRACK_DIR / "_outputs" / "p5_stage3"
DEFAULT_RUNTIME_OUTPUT = TRACK_DIR / "_outputs" / "p5_stage3_runtime"
SOURCE_LOCK_PATH = PROJECT_ROOT / "_models" / "facies" / "p5_sources.json"
STAGE2_RESULTS_PATH = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_results.jsonl"
STAGE2_SUMMARY_PATH = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_summary.json"


@dataclass(frozen=True)
class Stage3Budget:
    """An immutable copy of the accepted Stage-2 neural pilot budget."""

    profile_id: str = "facies-p5-stage2-fixed-v1"
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
        accepted = stage2.PilotBudget()
        for field_name in (
            "profile_id",
            "max_updates",
            "max_wall_seconds",
            "max_train_samples",
            "max_validation_samples",
            "batch_size",
            "validation_interval",
            "learning_rate",
            "weight_decay",
            "loss_id",
        ):
            if getattr(self, field_name) != getattr(accepted, field_name):
                raise ValueError(
                    f"Stage-3 must reuse Stage-2 {field_name}={getattr(accepted, field_name)!r}"
                )
        if self.max_updates > 200 or self.max_wall_seconds > 600:
            raise ValueError("Stage-3 exceeds the frozen 2-D neural resource ceiling")


@dataclass(frozen=True)
class CellSpec:
    task_id: str
    lane: str
    model_id: str
    fold_id: int
    repeat_id: int
    model_seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.task_id}/{self.lane}/{self.model_id}/"
            f"fold-{self.fold_id}/repeat-{self.repeat_id}"
        )


@dataclass(frozen=True)
class PreparedFold:
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
    fold_split_hash: str
    update_schedule: np.ndarray


def expected_cells() -> tuple[CellSpec, ...]:
    cells = tuple(
        CellSpec(task_id, LANE, model_id, fold_id, repeat_id, model_seed)
        for task_id in TASK_IDS
        for model_id in TOP_MODELS[task_id]
        for fold_id in FOLD_IDS
        for repeat_id, model_seed in enumerate(REPEAT_SEEDS)
    )
    validate_cell_specs(cells)
    return cells


def validate_cell_specs(cells: Sequence[CellSpec]) -> None:
    values = tuple(cells)
    if len(values) != EXPECTED_CELLS:
        raise ValueError(f"facies Stage-3 requires exactly {EXPECTED_CELLS} cells")
    if len({cell.key for cell in values}) != len(values):
        raise ValueError("duplicate Stage-3 cell identity")
    for cell in values:
        if cell.task_id not in TASK_IDS:
            raise ValueError(f"unknown task in Stage-3 cell: {cell.task_id}")
        if cell.lane != LANE:
            raise ValueError("cross-lane pollution: facies Stage-3 is scratch-only")
        if cell.model_id not in TOP_MODELS[cell.task_id]:
            raise ValueError("Stage-2 leaderboard outsider entered Stage-3")
        if cell.fold_id not in FOLD_IDS:
            raise ValueError("Stage-3 cell uses a non-frozen fold")
        if cell.repeat_id not in range(len(REPEAT_SEEDS)):
            raise ValueError("Stage-3 cell uses a non-frozen repeat")
        if cell.model_seed != REPEAT_SEEDS[cell.repeat_id]:
            raise ValueError("Stage-3 cell model seed differs from frozen repeat seed")


def _cell_contract_hash(cell: CellSpec, prepared: PreparedFold, budget: Stage3Budget) -> str:
    return hash_payload(
        {
            "schema_version": RESULT_SCHEMA,
            "cell": asdict(cell),
            "fold_split_hash": prepared.fold_split_hash,
            "manifest_stable_hash": prepared.manifest_stable_hash,
            "budget": asdict(budget),
            "source_lock": dict(source_lock(cell.model_id)),
            "preprocessor_hash": hash_payload(prepared.preprocessor.to_dict()),
            "hpo": False,
        }
    )


def validate_gpu_contract(device: torch.device, lock_value: str | None) -> Path:
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("Stage-3 neural cells must run on cuda:0; CPU is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-3 requires a real CUDA runtime")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("cuda:0 is absent")
    if not lock_value:
        raise RuntimeError("VOLVE_P5_GPU_LOCK must be set for Stage-3")
    lock_path = Path(lock_value)
    if lock_path != EXPECTED_GPU_LOCK:
        raise RuntimeError(
            f"VOLVE_P5_GPU_LOCK must equal the frozen shared lock {EXPECTED_GPU_LOCK}"
        )
    return lock_path


def seed_repeat_model(model_seed: int) -> dict[str, Any]:
    """Use the frozen repeat value itself as every model-initialization RNG seed."""
    if model_seed not in REPEAT_SEEDS:
        raise ValueError(f"unregistered Stage-3 model seed {model_seed}")
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    torch.use_deterministic_algorithms(False, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    return {
        "root_seed": ROOT_SEED,
        "seed_tree": {
            "repeat_model_seed": model_seed,
            "derivation": "derive_seed(2693,'model','p5-stage3',repeat_id)",
        },
        "strict_requested": False,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "torch_initial_seed": int(torch.initial_seed()),
        "python_hash_seed_effective": False,
        "warnings": ["PYTHONHASHSEED is fixed only at interpreter startup"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def _support(labels: Sequence[np.ndarray], num_classes: int) -> tuple[int, ...]:
    histogram = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        histogram += np.bincount(label.reshape(-1), minlength=num_classes)[:num_classes]
    return tuple(int(value) for value in histogram)


def prepare_fold(
    *,
    task_id: str,
    fold_id: int,
    manifest_path: Path,
    processed_root: Path,
    budget: Stage3Budget,
) -> PreparedFold:
    """Fit one fold using only its locked fold-train and capped validation IDs."""
    manifest, manifest_file_sha256 = stage2.load_locked_manifest(task_id, manifest_path)
    if manifest.effective_n_splits != 5 or tuple(f.fold_id for f in manifest.folds) != FOLD_IDS:
        raise ValueError("Stage-3 requires the exact five effective P4 folds")
    fold = manifest.folds[fold_id]
    train_seed = derive_seed(ROOT_SEED, "sampler", task_id, f"fold{fold_id}", "train_subset")
    validation_seed = derive_seed(
        ROOT_SEED, "sampler", task_id, f"fold{fold_id}", "validation_subset"
    )
    train_ids = stage2.deterministic_subset(
        fold.train_sample_ids, count=budget.max_train_samples, seed=train_seed
    )
    validation_ids = stage2.deterministic_subset(
        fold.validation_sample_ids,
        count=budget.max_validation_samples,
        seed=validation_seed,
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("locked fold train/validation sample overlap")
    archive = stage2.Stage2DevelopmentArchive(task_id, processed_root)
    train_records, train_raw, train_labels = stage2._materialize_selected(archive, train_ids)
    validation_records, validation_raw, validation_labels = stage2._materialize_selected(
        archive, validation_ids
    )
    train_groups = tuple(str(record.inline) for record in train_records)
    validation_groups = tuple(str(record.inline) for record in validation_records)
    if set(train_groups) & set(validation_groups):
        raise ValueError("locked fold train/validation inline overlap")
    if not set(train_groups) <= set(fold.train_groups):
        raise ValueError("selected train group escapes frozen fold")
    if not set(validation_groups) <= set(fold.validation_groups):
        raise ValueError("selected validation group escapes frozen fold")

    spec = get_task_spec(task_id)
    classes = int(spec.metadata["num_classes"])
    fold_train_support = np.asarray(
        fold.support.get("train_per_class_pixels", ()), dtype=np.int64
    )
    if fold_train_support.shape != (classes,):
        raise ValueError("locked manifest lacks full fold-train class support")
    class_weights = inverse_sqrt_class_weights(fold_train_support)
    fit_values = np.concatenate([raw.reshape(-1) for raw in train_raw]).astype(
        np.float32, copy=False
    )
    normalization = fit_zscore(fit_values)
    recovered = denormalize(normalize(train_raw[0], normalization), normalization)
    roundtrip_error = float(np.max(np.abs(recovered - train_raw[0])))
    if not math.isfinite(roundtrip_error) or roundtrip_error > 1e-2:
        raise ValueError(f"fold-train normalization round-trip failed: {roundtrip_error}")
    preprocessor = FoldPreprocessor(
        task_id=task_id,
        label_version=LABEL_VERSIONS[task_id],
        normalization=normalization,
        class_weights=tuple(float(value) for value in class_weights),
        class_histogram=tuple(int(value) for value in fold_train_support),
        fit_sample_count=len(train_ids),
        fit_sample_ids_hash=hash_payload(list(train_ids)),
        roundtrip_max_abs_error=roundtrip_error,
    )
    train_images = np.stack(
        [normalize(raw, normalization).astype(np.float32) for raw in train_raw]
    )[:, None]
    validation_images = np.stack(
        [normalize(raw, normalization).astype(np.float32) for raw in validation_raw]
    )[:, None]
    train_label_array = np.stack(train_labels).astype(np.int64)
    validation_label_array = np.stack(validation_labels).astype(np.int64)
    if not np.isfinite(train_images).all() or not np.isfinite(validation_images).all():
        raise ValueError("fold normalization produced NaN/Inf")
    nearest = int(fold.purge.get("nearest_train_validation_inline_distance", 0))
    buffer_groups = int(fold.purge.get("buffer_groups", 0))
    if buffer_groups <= 0 or nearest <= buffer_groups:
        raise ValueError("locked fold does not prove spatial train/validation isolation")
    split_evidence = {
        "task_id": task_id,
        "label_version": spec.label_version,
        "manifest_stable_hash": manifest.stable_hash(),
        "fold_id": fold_id,
        "train_sample_ids": list(train_ids),
        "validation_sample_ids": list(validation_ids),
        "train_groups": list(train_groups),
        "validation_groups": list(validation_groups),
        "buffer_groups": buffer_groups,
        "nearest_inline_distance": nearest,
        "selection": "stable_seed_before_label_read",
    }
    schedule_seed = derive_seed(
        ROOT_SEED, "sampler", task_id, f"fold{fold_id}", budget.profile_id
    )
    return PreparedFold(
        task_id=task_id,
        label_version=spec.label_version,
        num_classes=classes,
        manifest_stable_hash=manifest.stable_hash(),
        manifest_file_sha256=manifest_file_sha256,
        fold_id=fold_id,
        train_images=train_images,
        train_labels=train_label_array,
        validation_images=validation_images,
        validation_labels=validation_label_array,
        class_weights=tuple(float(value) for value in class_weights),
        preprocessor=preprocessor,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        train_groups=train_groups,
        validation_groups=validation_groups,
        train_support=_support(train_labels, classes),
        validation_support=_support(validation_labels, classes),
        nearest_inline_distance=nearest,
        buffer_groups=buffer_groups,
        fold_split_hash=hash_payload(split_evidence),
        update_schedule=stage2.fixed_update_schedule(
            len(train_ids), budget, seed=schedule_seed
        ),
    )


def _split_result(prepared: PreparedFold) -> dict[str, Any]:
    return {
        "manifest_stable_hash": prepared.manifest_stable_hash,
        "manifest_file_sha256": prepared.manifest_file_sha256,
        "fold_id": prepared.fold_id,
        "fold_split_hash": prepared.fold_split_hash,
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


def _runtime_cell_dir(runtime_root: Path, cell: CellSpec) -> Path:
    return (
        Path(runtime_root)
        / cell.task_id
        / cell.lane
        / cell.model_id
        / f"fold_{cell.fold_id}"
        / f"repeat_{cell.repeat_id}"
    )


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _predict_probabilities(
    model: nn.Module,
    images: np.ndarray,
    *,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(
                images[start : start + batch_size], dtype=torch.float32, device=device
            )
            logits = model(batch)
            stage2._assert_logits(logits, batch, num_classes)
            chunks.append(softmax_probabilities(logits).cpu().numpy())
    probabilities = np.concatenate(chunks, axis=0)
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities contain NaN/Inf")
    return probabilities


def _checkpoint_trainer_state(
    step: int,
    best_step: int,
    best_loss: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "next_epoch": step + 1,
        "global_step": step,
        "best_epoch": best_step,
        "best_val_loss": best_loss,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "history": history,
    }


def run_cell(
    *,
    cell: CellSpec,
    prepared: PreparedFold,
    budget: Stage3Budget,
    runtime_root: Path,
    device: torch.device,
    gpu_lock_wait_seconds: float,
) -> dict[str, Any]:
    """Run one legal CUDA cell; preserve every failure as a structured row."""
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("run_cell refuses non-cuda:0 execution")
    if cell.task_id != prepared.task_id or cell.fold_id != prepared.fold_id:
        raise ValueError("cell and prepared fold identity mismatch")
    if cell.lane != LANE or cell.model_id not in TOP_MODELS[cell.task_id]:
        raise ValueError("cross-lane or candidate pollution in Stage-3 cell")
    contract_hash = _cell_contract_hash(cell, prepared, budget)
    cell_dir = _runtime_cell_dir(runtime_root, cell)
    checkpoint_path = cell_dir / "best.ckpt"
    prediction_path = cell_dir / "oof_predictions.npz"
    started = time.perf_counter()
    model: nn.Module | None = None
    try:
        seed_report = seed_repeat_model(cell.model_seed)
        discovered = discover_model("facies", cell.model_id)
        spec = get_task_spec(cell.task_id)
        model = discovered.build(
            spec, num_classes=prepared.num_classes, lane=cell.lane
        ).to(device)
        parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable_parameters = int(
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
        configuration = {
            "cell": asdict(cell),
            "budget": asdict(budget),
            "num_classes": prepared.num_classes,
            "label_version": prepared.label_version,
            "loss_activation": "weighted_cross_entropy_raw_logits; softmax_inference_only",
            "hpo": False,
        }
        history: list[dict[str, Any]] = []
        best_loss = math.inf
        best_step = -1
        updates = 0
        gradients_finite = False
        timed_out = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = int(torch.cuda.memory_allocated(device))

        for update, indices in enumerate(prepared.update_schedule, start=1):
            if time.perf_counter() - started >= budget.max_wall_seconds:
                timed_out = True
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
            stage2._assert_logits(logits, images, prepared.num_classes)
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
            updates = update
            train_loss = float(loss.detach())
            if update % budget.validation_interval == 0 or update == budget.max_updates:
                validation_loss, _ = stage2._evaluate(
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
                        "update": update,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    }
                )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_step = update
                    save_checkpoint(
                        checkpoint_path,
                        epoch=update,
                        model_state=stage2._cpu_state_dict(model),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=None,
                        scaler_state=None,
                        config_hash=hash_payload(configuration),
                        split_hash=prepared.fold_split_hash,
                        trainer_state=_checkpoint_trainer_state(
                            update, best_step, best_loss, history
                        ),
                        seed_report=seed_report,
                        environment=stage2._environment(device),
                        extra={
                            "stage": "p5_stage3_multiseed_cv",
                            "cell_key": cell.key,
                            "cell_contract_hash": contract_hash,
                            "manifest_stable_hash": prepared.manifest_stable_hash,
                            "preprocessor_hash": hash_payload(prepared.preprocessor.to_dict()),
                            "test_access": False,
                        },
                    )
        model_wall_seconds = time.perf_counter() - started
        common = {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            **asdict(cell),
            "cell_key": cell.key,
            "cell_contract_hash": contract_hash,
            "label_version": prepared.label_version,
            "head_num_classes": prepared.num_classes,
            "budget": asdict(budget),
            "split": _split_result(prepared),
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        }
        if timed_out:
            return {
                **common,
                "status": "timeout",
                "reason": "fixed Stage-2 per-cell model wall-clock budget exhausted",
                "updates_completed": updates,
                "resources": {
                    "device": "cuda:0",
                    "gpu_lock_mechanism": "fcntl.flock(LOCK_EX)",
                    "gpu_lock_name": EXPECTED_GPU_LOCK.name,
                    "gpu_lock_held": True,
                    "gpu_lock_wait_seconds_excluded": gpu_lock_wait_seconds,
                    "wall_seconds": model_wall_seconds,
                    "cuda_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                },
            }
        if updates != budget.max_updates or best_step < 0 or not checkpoint_path.is_file():
            raise RuntimeError("cell did not complete its fixed update/checkpoint contract")

        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["split_hash"] != prepared.fold_split_hash:
            raise ValueError("best checkpoint split hash mismatch")
        if checkpoint["extra"].get("cell_contract_hash") != contract_hash:
            raise ValueError("best checkpoint cell contract mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        probe = torch.as_tensor(
            prepared.validation_images[:1], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            expected_logits = model(probe).detach().cpu()
        restored = discovered.build(
            spec, num_classes=prepared.num_classes, lane=cell.lane
        ).to(device)
        restored.load_state_dict(checkpoint["model_state"])
        restored.eval()
        with torch.no_grad():
            restored_logits = restored(probe).detach().cpu()
        checkpoint_difference = float(
            torch.max(torch.abs(expected_logits - restored_logits))
        )
        if checkpoint_difference > 1e-6:
            raise ValueError(
                f"checkpoint round-trip changed logits by {checkpoint_difference}"
            )
        validation_loss, metrics = stage2._evaluate(
            restored,
            prepared.validation_images,
            prepared.validation_labels,
            criterion,
            batch_size=budget.batch_size,
            num_classes=prepared.num_classes,
            device=device,
        )
        probabilities = _predict_probabilities(
            restored,
            prepared.validation_images,
            batch_size=budget.batch_size,
            num_classes=prepared.num_classes,
            device=device,
        )
        prediction, confidence, entropy, error = confidence_entropy_error(
            probabilities, prepared.validation_labels
        )
        observed_matrix = confusion_matrix(
            prepared.validation_labels, prediction, prepared.num_classes
        )
        if observed_matrix.tolist() != metrics["confusion_matrix"]:
            raise ValueError("archived OOF prediction differs from validation metrics")
        _atomic_save_npz(
            prediction_path,
            sample_ids=np.asarray(prepared.validation_sample_ids, dtype=str),
            inline=np.asarray([int(value) for value in prepared.validation_groups]),
            seismic=prepared.validation_images[:, 0].astype(np.float16),
            labels=prepared.validation_labels.astype(np.uint8),
            probabilities=probabilities.astype(np.float16),
            prediction=prediction.astype(np.uint8),
            confidence=confidence.astype(np.float16),
            entropy=entropy.astype(np.float16),
            error=error.astype(np.uint8),
        )
        peak_vram = int(torch.cuda.max_memory_allocated(device))
        return {
            **common,
            "status": "completed",
            "source_lock": dict(source_lock(cell.model_id)),
            "preprocessing": {
                "normalization_fit_scope": "selected_locked_fold_train_only",
                "normalization": prepared.preprocessor.normalization.to_dict(),
                "class_weight_fit_scope": "locked_full_fold_train_support_only",
                "class_weights": list(prepared.class_weights),
                "target_transform": "identity_integer_ids",
                "denoise": "identity",
                "roundtrip_max_abs_error": prepared.preprocessor.roundtrip_max_abs_error,
                "preprocessor_hash": hash_payload(prepared.preprocessor.to_dict()),
            },
            "calibration": {
                "method": "identity",
                "fit_scope": "none_no_calibrator_fit",
                "evaluation": "raw_softmax_probabilities",
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
                "hpo": False,
            },
            "validation_metrics": metrics,
            "checkpoint": {
                "runtime_relative_path": checkpoint_path.relative_to(runtime_root).as_posix(),
                "sha256": hash_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "prediction_max_abs_difference": checkpoint_difference,
            },
            "oof_prediction": {
                "runtime_relative_path": prediction_path.relative_to(runtime_root).as_posix(),
                "sha256": hash_file(prediction_path),
                "bytes": prediction_path.stat().st_size,
                "scope": "budget_capped_locked_fold_validation_subset",
            },
            "resources": {
                "device": "cuda:0",
                "parameters": parameters,
                "trainable_parameters": trainable_parameters,
                "wall_seconds": model_wall_seconds,
                "gpu_lock_mechanism": "fcntl.flock(LOCK_EX)",
                "gpu_lock_name": EXPECTED_GPU_LOCK.name,
                "gpu_lock_held": True,
                "gpu_lock_wait_seconds_excluded": gpu_lock_wait_seconds,
                "cuda_memory_before_bytes": memory_before,
                "cuda_peak_allocated_bytes": peak_vram,
                "download_bytes": 0,
            },
            "environment": stage2._environment(device),
        }
    except P5AdapterSkip as skip:
        return {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            **asdict(cell),
            "cell_key": cell.key,
            "cell_contract_hash": contract_hash,
            "label_version": prepared.label_version,
            "head_num_classes": prepared.num_classes,
            "status": "skipped",
            "skip": skip.to_dict(),
            "budget": asdict(budget),
            "split": _split_result(prepared),
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        }
    except Exception as exc:
        return {
            "schema_version": RESULT_SCHEMA,
            "track_id": "facies",
            **asdict(cell),
            "cell_key": cell.key,
            "cell_contract_hash": contract_hash,
            "label_version": prepared.label_version,
            "head_num_classes": prepared.num_classes,
            "status": "failed",
            "failure": {"type": type(exc).__name__, "reason": str(exc)},
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


def _blocked_result(cell: CellSpec, exc: Exception, budget: Stage3Budget) -> dict[str, Any]:
    spec = get_task_spec(cell.task_id)
    return {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        **asdict(cell),
        "cell_key": cell.key,
        "label_version": spec.label_version,
        "head_num_classes": int(spec.metadata["num_classes"]),
        "status": "blocked",
        "blocker": {
            "code": "locked_development_fold_unavailable",
            "type": type(exc).__name__,
            "reason": str(exc),
        },
        "budget": asdict(budget),
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
    }


def _write_runtime_result(runtime_root: Path, cell: CellSpec, result: Mapping[str, Any]) -> Path:
    return atomic_write_json(_runtime_cell_dir(runtime_root, cell) / "cell_result.json", result)


def _load_resumable_result(
    runtime_root: Path,
    cell: CellSpec,
    prepared: PreparedFold,
    budget: Stage3Budget,
) -> dict[str, Any] | None:
    path = _runtime_cell_dir(runtime_root, cell) / "cell_result.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cell_key") != cell.key:
        raise ValueError(f"runtime cell identity collision at {path}")
    expected_hash = _cell_contract_hash(cell, prepared, budget)
    if payload.get("cell_contract_hash") != expected_hash:
        raise ValueError(f"runtime cell contract changed at {path}; refusing contaminated resume")
    if payload.get("status") != "completed":
        return payload
    for artifact_name in ("checkpoint", "oof_prediction"):
        record = payload[artifact_name]
        artifact = Path(runtime_root) / record["runtime_relative_path"]
        if not artifact.is_file() or hash_file(artifact) != record["sha256"]:
            raise ValueError(f"resumable {artifact_name} is absent or corrupt for {cell.key}")
    return payload


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    rows = tuple(records)
    expected = expected_cells()
    expected_by_key = {cell.key: cell for cell in expected}
    if len(rows) != EXPECTED_CELLS:
        raise ValueError(f"Stage-3 results require exactly {EXPECTED_CELLS} rows")
    keys = [str(row.get("cell_key", "")) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate Stage-3 result cell")
    if set(keys) != set(expected_by_key):
        raise ValueError("Stage-3 results omit or add a frozen cell")
    for row in rows:
        cell = expected_by_key[row["cell_key"]]
        for field_name, expected_value in asdict(cell).items():
            if row.get(field_name) != expected_value:
                raise ValueError(f"result {row['cell_key']} changed frozen {field_name}")
        if row.get("lane") != LANE:
            raise ValueError("cross-lane result contamination")
        if row.get("status") not in {"completed", "skipped", "blocked", "failed", "timeout"}:
            raise ValueError("unknown Stage-3 result status")
        if any(
            bool(row.get(flag))
            for flag in ("test_archive_opened", "test_labels_read", "test_metrics_computed")
        ):
            raise ValueError("frozen-test firewall violation in Stage-3 result")


def _bootstrap_ci(values: Sequence[float], *, seed: int, resamples: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires finite completed-cell metrics")
    generator = np.random.default_rng(seed)
    selections = generator.integers(0, array.size, size=(resamples, array.size))
    means = array[selections].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(values).astype(np.float64)
    support = values.sum(axis=1).astype(np.int64)
    false_positive = values.sum(axis=0).astype(np.float64) - true_positive
    false_negative = support.astype(np.float64) - true_positive
    union = true_positive + false_positive + false_negative
    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / values.sum()),
        "miou": float(iou.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_support": support.tolist(),
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": values.tolist(),
        "all_classes_supported": bool(np.all(support > 0)),
        "averaging": "all_configured_classes_missing_support_scores_zero",
    }


def build_leaderboard(task_id: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown facies task {task_id}")
    task_rows = [row for row in records if row["task_id"] == task_id]
    if any(row["lane"] != LANE for row in task_rows):
        raise ValueError("cross-lane pollution in task leaderboard")
    expected_task_cells = len(TOP_MODELS[task_id]) * len(FOLD_IDS) * len(REPEAT_SEEDS)
    completed_task = sum(row["status"] == "completed" for row in task_rows)
    completion_rate = completed_task / expected_task_cells
    task_rankable = completion_rate >= RANKABLE_COMPLETION_THRESHOLD
    entries: list[dict[str, Any]] = []
    for model_id in TOP_MODELS[task_id]:
        model_rows = [row for row in task_rows if row["model_id"] == model_id]
        completed = [row for row in model_rows if row["status"] == "completed"]
        expected_model_cells = len(FOLD_IDS) * len(REPEAT_SEEDS)
        model_completion = len(completed) / expected_model_cells
        entry: dict[str, Any] = {
            "model_id": model_id,
            "expected_cells": expected_model_cells,
            "completed_cells": len(completed),
            "completion_rate": model_completion,
            "status_counts": {
                status: sum(row["status"] == status for row in model_rows)
                for status in ("completed", "skipped", "blocked", "failed", "timeout")
            },
            "rankable": bool(task_rankable and model_completion >= RANKABLE_COMPLETION_THRESHOLD),
        }
        if completed:
            scores = [float(row["validation_metrics"]["miou"]) for row in completed]
            fold_means = {
                str(fold_id): float(
                    np.mean(
                        [
                            row["validation_metrics"]["miou"]
                            for row in completed
                            if row["fold_id"] == fold_id
                        ]
                    )
                )
                for fold_id in FOLD_IDS
                if any(row["fold_id"] == fold_id for row in completed)
            }
            seed_means = {
                str(repeat_id): float(
                    np.mean(
                        [
                            row["validation_metrics"]["miou"]
                            for row in completed
                            if row["repeat_id"] == repeat_id
                        ]
                    )
                )
                for repeat_id in range(len(REPEAT_SEEDS))
                if any(row["repeat_id"] == repeat_id for row in completed)
            }
            confusion = np.sum(
                [np.asarray(row["validation_metrics"]["confusion_matrix"]) for row in completed],
                axis=0,
            )
            entry.update(
                {
                    "mean_miou": float(np.mean(scores)),
                    "miou_95_bootstrap_ci": _bootstrap_ci(
                        scores,
                        seed=derive_seed(ROOT_SEED, "diagnostic", task_id, model_id, "bootstrap"),
                    ),
                    "worst_fold_miou": float(min(fold_means.values())),
                    "fold_mean_miou": fold_means,
                    "seed_mean_miou": seed_means,
                    "seed_std_miou": float(np.std(list(seed_means.values()), ddof=0)),
                    "mean_macro_f1": float(
                        np.mean([row["validation_metrics"]["macro_f1"] for row in completed])
                    ),
                    "mean_accuracy": float(
                        np.mean([row["validation_metrics"]["accuracy"] for row in completed])
                    ),
                    "aggregate_oof_metrics": _metrics_from_confusion(confusion),
                    "resources": {
                        "mean_wall_seconds": float(
                            np.mean([row["resources"]["wall_seconds"] for row in completed])
                        ),
                        "total_wall_seconds": float(
                            np.sum([row["resources"]["wall_seconds"] for row in completed])
                        ),
                        "max_cuda_peak_allocated_bytes": int(
                            max(row["resources"]["cuda_peak_allocated_bytes"] for row in completed)
                        ),
                    },
                }
            )
        entries.append(entry)
    rankable_entries = [entry for entry in entries if entry["rankable"]]
    ordered = sorted(
        rankable_entries,
        key=lambda entry: (
            -entry["mean_miou"],
            -entry["worst_fold_miou"],
            entry["seed_std_miou"],
            entry["resources"]["mean_wall_seconds"],
            entry["model_id"],
        ),
    )
    rank_by_model = {entry["model_id"]: rank for rank, entry in enumerate(ordered, start=1)}
    for entry in entries:
        entry["rank"] = rank_by_model.get(entry["model_id"])
    entries.sort(key=lambda entry: (entry["rank"] is None, entry["rank"] or 999, entry["model_id"]))
    return {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "task_id": task_id,
        "label_version": get_task_spec(task_id).label_version,
        "lane": LANE,
        "status": "ranked" if task_rankable else "not_rankable",
        "not_rankable_reason": (
            None
            if task_rankable
            else f"legal completion {completion_rate:.3f} is below {RANKABLE_COMPLETION_THRESHOLD:.2f}"
        ),
        "expected_cells": expected_task_cells,
        "completed_cells": completed_task,
        "completion_rate": completion_rate,
        "primary_metric": "mean_miou",
        "tie_break_order": (
            "mean_miou_desc",
            "worst_fold_miou_desc",
            "seed_std_miou_asc",
            "mean_wall_seconds_asc",
            "model_id_asc",
        ),
        "bootstrap_resamples": 5000,
        "frozen_test_consumed": False,
        "entries": entries,
    }


def _write_oof_manifest(
    output_root: Path,
    runtime_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    entries: list[dict[str, Any]] = []
    for row in records:
        if row["status"] != "completed":
            continue
        prediction = row["oof_prediction"]
        path = Path(runtime_root) / prediction["runtime_relative_path"]
        if not path.is_file() or hash_file(path) != prediction["sha256"]:
            raise ValueError(f"OOF archive missing/corrupt for {row['cell_key']}")
        entries.append(
            {
                "cell_key": row["cell_key"],
                "task_id": row["task_id"],
                "lane": row["lane"],
                "model_id": row["model_id"],
                "fold_id": row["fold_id"],
                "repeat_id": row["repeat_id"],
                "model_seed": row["model_seed"],
                "runtime_relative_path": prediction["runtime_relative_path"],
                "sha256": prediction["sha256"],
                "bytes": prediction["bytes"],
                "sample_count": row["split"]["validation_sample_count"],
                "sample_ids_hash": row["split"]["validation_sample_ids_hash"],
                "manifest_stable_hash": row["split"]["manifest_stable_hash"],
                "test_access": False,
            }
        )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "scope": "budget_capped_locked_development_oof",
        "entry_count": len(entries),
        "runtime_root": "provided_at_rebuild_time_not_serialized",
        "entries": entries,
        "frozen_test_consumed": False,
    }
    return atomic_write_json(Path(output_root) / "p5_stage3_oof_manifest.json", payload)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


_DIAGNOSTIC_ARRAYS = frozenset(
    {
        "sample_ids",
        "inline",
        "seismic",
        "labels",
        "prediction",
        "confidence",
        "entropy",
        "error",
    }
)


def _diagnostic_sample_statistics(
    sample_id: str, labels: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    if labels.shape != prediction.shape:
        raise ValueError("diagnostic label/prediction shape mismatch")
    incorrect = np.asarray(labels != prediction)
    error_pixels = int(incorrect.sum())
    total_pixels = int(incorrect.size)
    correct_pixels = total_pixels - error_pixels
    gt_class_count = int(np.unique(labels).size)
    return {
        "sample_id": str(sample_id),
        "gt_class_count": gt_class_count,
        "correct_pixels": correct_pixels,
        "error_pixels": error_pixels,
        "total_pixels": total_pixels,
        "error_fraction": error_pixels / total_pixels,
        "has_at_least_two_gt_classes": gt_class_count >= 2,
        "has_correct_and_error_pixels": correct_pixels > 0 and error_pixels > 0,
        "eligible": gt_class_count >= 2 and correct_pixels > 0 and error_pixels > 0,
    }


def _select_diagnostic_sample(
    *,
    task_id: str,
    winner_id: str,
    winner_rows: Sequence[Mapping[str, Any]],
    oof_manifest: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Choose an informative development OOF sample without touching test data."""
    ordered_rows = sorted(
        winner_rows, key=lambda row: (row["fold_id"], row["repeat_id"], row["cell_key"])
    )
    entries_by_key = {entry["cell_key"]: entry for entry in oof_manifest["entries"]}
    verified_paths: set[Path] = set()

    def source_for(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
        try:
            entry = entries_by_key[row["cell_key"]]
        except KeyError as exc:
            raise ValueError(f"OOF manifest omits {row['cell_key']}") from exc
        path = Path(runtime_root) / entry["runtime_relative_path"]
        if path not in verified_paths:
            if hash_file(path) != entry["sha256"]:
                raise ValueError("visualization source OOF archive hash mismatch")
            verified_paths.add(path)
        return entry, path

    def load_sample(
        row: Mapping[str, Any], sample_index: int
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], Mapping[str, Any]]:
        entry, path = source_for(row)
        with np.load(path, allow_pickle=False) as archive:
            if not _DIAGNOSTIC_ARRAYS <= set(archive.files):
                raise ValueError("visualization OOF archive is incomplete")
            if not 0 <= sample_index < len(archive["sample_ids"]):
                raise ValueError("diagnostic sample index is out of range")
            arrays = {
                name: np.asarray(archive[name][sample_index])
                for name in _DIAGNOSTIC_ARRAYS
            }
        expected_error = np.asarray(arrays["labels"] != arrays["prediction"])
        if not np.array_equal(expected_error, arrays["error"].astype(bool)):
            raise ValueError("archived OOF error map is inconsistent")
        statistics = _diagnostic_sample_statistics(
            str(arrays["sample_ids"]), arrays["labels"], arrays["prediction"]
        )
        return arrays, statistics, entry

    representative = ordered_rows[0]
    _, representative_path = source_for(representative)
    with np.load(representative_path, allow_pickle=False) as archive:
        if not _DIAGNOSTIC_ARRAYS <= set(archive.files):
            raise ValueError("visualization OOF archive is incomplete")
        preferred_index = derive_seed(
            ROOT_SEED, "diagnostic", task_id, winner_id
        ) % len(archive["sample_ids"])
    arrays, statistics, entry = load_sample(representative, preferred_index)
    rule = (
        "seeded_candidate_if_eligible_else_global_max_gt_diversity_"
        "then_correct_error_balance_then_sample_and_cell_id"
    )
    if statistics["eligible"]:
        return {
            "arrays": arrays,
            "statistics": statistics,
            "row": representative,
            "manifest_entry": entry,
            "selection_rule": rule,
            "selection_outcome": "seeded_candidate_eligible",
        }

    eligible: list[tuple[tuple[Any, ...], Mapping[str, Any], int]] = []
    for row in ordered_rows:
        _, path = source_for(row)
        with np.load(path, allow_pickle=False) as archive:
            if not _DIAGNOSTIC_ARRAYS <= set(archive.files):
                raise ValueError("visualization OOF archive is incomplete")
            sample_ids = np.asarray(archive["sample_ids"]).astype(str)
            labels = np.asarray(archive["labels"])
            predictions = np.asarray(archive["prediction"])
            if len(sample_ids) != len(labels) or labels.shape != predictions.shape:
                raise ValueError("visualization OOF sample arrays are misaligned")
            for sample_index, sample_id in enumerate(sample_ids):
                candidate = _diagnostic_sample_statistics(
                    sample_id, labels[sample_index], predictions[sample_index]
                )
                if not candidate["eligible"]:
                    continue
                balance = min(candidate["correct_pixels"], candidate["error_pixels"])
                key = (
                    -candidate["gt_class_count"],
                    -balance,
                    candidate["sample_id"],
                    row["cell_key"],
                    sample_index,
                )
                eligible.append((key, row, sample_index))
    if not eligible:
        return {
            "arrays": arrays,
            "statistics": statistics,
            "row": representative,
            "manifest_entry": entry,
            "selection_rule": rule,
            "selection_outcome": "fallback_no_eligible_development_oof_sample",
        }
    _, selected_row, selected_index = min(eligible, key=lambda item: item[0])
    arrays, statistics, entry = load_sample(selected_row, selected_index)
    return {
        "arrays": arrays,
        "statistics": statistics,
        "row": selected_row,
        "manifest_entry": entry,
        "selection_rule": rule,
        "selection_outcome": "global_informative_candidate",
    }


def _render_task_figure(
    *,
    task_id: str,
    records: Sequence[Mapping[str, Any]],
    leaderboard: Mapping[str, Any],
    oof_manifest: Mapping[str, Any],
    runtime_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if leaderboard["status"] != "ranked":
        return {"task_id": task_id, "status": "not_generated", "reason": "task not rankable"}
    winner = next(entry for entry in leaderboard["entries"] if entry["rank"] == 1)
    winner_id = winner["model_id"]
    winner_rows = [
        row
        for row in records
        if row["task_id"] == task_id
        and row["model_id"] == winner_id
        and row["status"] == "completed"
    ]
    if not winner_rows:
        raise ValueError(f"ranked winner {winner_id} has no completed OOF rows")
    selection = _select_diagnostic_sample(
        task_id=task_id,
        winner_id=winner_id,
        winner_rows=winner_rows,
        oof_manifest=oof_manifest,
        runtime_root=runtime_root,
    )
    representative = selection["row"]
    manifest_entry = selection["manifest_entry"]
    arrays = selection["arrays"]
    classes = int(get_task_spec(task_id).metadata["num_classes"])
    aggregate = winner["aggregate_oof_metrics"]
    matrix = np.asarray(aggregate["confusion_matrix"], dtype=np.int64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sum > 0,
    )
    grid_values = np.full((len(FOLD_IDS), len(REPEAT_SEEDS)), np.nan)
    for row in winner_rows:
        grid_values[row["fold_id"], row["repeat_id"]] = row["validation_metrics"]["miou"]

    figure = plt.figure(figsize=(18, 12))
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.9))
    class_cmap = plt.get_cmap("tab10", classes)
    seismic = arrays["seismic"].astype(np.float32)
    amplitude = float(np.percentile(np.abs(seismic), 99.0)) or 1.0
    panels = [
        (seismic, "gray", "OOF seismic profile", -amplitude, amplitude),
        (arrays["labels"], class_cmap, "Ground truth", -0.5, classes - 0.5),
        (arrays["prediction"], class_cmap, "OOF prediction", -0.5, classes - 0.5),
        (arrays["error"], "Reds", "Error", 0.0, 1.0),
        (arrays["entropy"], "magma", "Normalized entropy", 0.0, 1.0),
        (arrays["confidence"], "viridis", "Confidence", 0.0, 1.0),
    ]
    for index, (values, cmap, title, lower, upper) in enumerate(panels):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        axis.imshow(values, cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    confusion_axis = figure.add_subplot(grid[1, 2])
    confusion_image = confusion_axis.imshow(
        normalized_matrix, cmap="Blues", vmin=0.0, vmax=1.0
    )
    confusion_axis.set_title("Aggregate OOF confusion")
    confusion_axis.set_xlabel("Predicted")
    confusion_axis.set_ylabel("Ground truth")
    confusion_axis.set_xticks(range(classes))
    confusion_axis.set_yticks(range(classes))
    figure.colorbar(confusion_image, ax=confusion_axis, fraction=0.046)

    class_axis = figure.add_subplot(grid[1, 3])
    positions = np.arange(classes)
    class_axis.bar(
        positions - 0.18,
        aggregate["per_class_iou"],
        width=0.36,
        label="IoU",
        color="#4C72B0",
    )
    class_axis.bar(
        positions + 0.18,
        aggregate["per_class_f1"],
        width=0.36,
        label="F1",
        color="#DD8452",
    )
    class_axis.set_ylim(0.0, 1.0)
    class_axis.set_xticks(positions)
    class_axis.set_xlabel("Facies class ID")
    class_axis.set_title("Aggregate per-class IoU / F1")
    class_axis.legend()

    heatmap_axis = figure.add_subplot(grid[2, 0:2])
    heatmap = heatmap_axis.imshow(grid_values, cmap="viridis", aspect="auto")
    heatmap_axis.set_xticks(range(len(REPEAT_SEEDS)), [f"seed {i}" for i in range(3)])
    heatmap_axis.set_yticks(range(len(FOLD_IDS)), [f"fold {i}" for i in FOLD_IDS])
    heatmap_axis.set_title("Fold × seed validation mIoU")
    for fold_id in FOLD_IDS:
        for repeat_id in range(len(REPEAT_SEEDS)):
            value = grid_values[fold_id, repeat_id]
            if np.isfinite(value):
                heatmap_axis.text(
                    repeat_id, fold_id, f"{value:.3f}", ha="center", va="center", color="white"
                )
    figure.colorbar(heatmap, ax=heatmap_axis, fraction=0.025)

    distribution_axis = figure.add_subplot(grid[2, 2:4])
    seed_values = [
        [
            row["validation_metrics"]["miou"]
            for row in winner_rows
            if row["repeat_id"] == repeat_id
        ]
        for repeat_id in range(len(REPEAT_SEEDS))
    ]
    distribution_axis.boxplot(seed_values, tick_labels=[f"seed {index}" for index in range(3)])
    distribution_axis.set_ylabel("mIoU across folds")
    distribution_axis.set_title(
        f"Repeat distribution | mean={winner['mean_miou']:.4f}, "
        f"worst-fold={winner['worst_fold_miou']:.4f}"
    )
    figure.suptitle(
        f"{task_id} Stage-3 development OOF | winner={winner_id}\n"
        f"sample={arrays['sample_ids']} inline={int(arrays['inline'])} | frozen test not consumed",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return {
        "task_id": task_id,
        "status": "generated",
        "model_id": winner_id,
        "figure": output_path.name,
        "figure_sha256": hash_file(output_path),
        "figure_bytes": output_path.stat().st_size,
        "representative_cell_key": representative["cell_key"],
        "representative_oof_runtime_relative_path": manifest_entry["runtime_relative_path"],
        "representative_oof_sha256": manifest_entry["sha256"],
        "selection": selection["selection_rule"],
        "selection_rule": selection["selection_rule"],
        "selection_outcome": selection["selection_outcome"],
        "selection_sample_id": selection["statistics"]["sample_id"],
        "selection_statistics": selection["statistics"],
        "contains": (
            "oof_seismic_gt_prediction",
            "error_entropy_confidence",
            "aggregate_confusion",
            "per_class_iou_f1",
            "fold_seed_distribution",
        ),
        "no_model_or_dataset_loaded": True,
        "frozen_test_consumed": False,
    }


def render_visualizations(*, output_root: Path, runtime_root: Path) -> Path:
    """Rebuild track figures from archived OOF predictions and portable JSON only."""
    output_root = Path(output_root)
    results_path = output_root / "p5_stage3_results.jsonl"
    oof_path = output_root / "p5_stage3_oof_manifest.json"
    records = _load_jsonl(results_path)
    validate_records(records)
    oof_manifest = _load_json(oof_path)
    figures: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        leaderboard_path = output_root / f"{task_id}_scratch_leaderboard.json"
        leaderboard = _load_json(leaderboard_path)
        figures.append(
            _render_task_figure(
                task_id=task_id,
                records=records,
                leaderboard=leaderboard,
                oof_manifest=oof_manifest,
                runtime_root=runtime_root,
                output_path=output_root / f"{task_id}_stage3_oof_diagnostics.png",
            )
        )
    manifest = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "visualizer": "facies_p5_stage3.render_visualizations",
        "source_results_sha256": hash_file(results_path),
        "source_oof_manifest_sha256": hash_file(oof_path),
        "runtime_root": "provided_at_rebuild_time_not_serialized",
        "figures": figures,
        "rebuild_requires": "portable JSON plus archived private OOF NPZ; no model/dataset/test",
        "frozen_test_consumed": False,
    }
    return atomic_write_json(output_root / "p5_stage3_visualization_manifest.json", manifest)


def collate_results(
    *,
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    runtime_root: Path,
    budget: Stage3Budget,
    device: torch.device,
) -> dict[str, Any]:
    validate_records(records)
    output_root = Path(output_root)
    results_path = stage2._atomic_write_jsonl(
        output_root / "p5_stage3_results.jsonl", records
    )
    leaderboards: dict[str, dict[str, Any]] = {}
    leaderboard_files: dict[str, dict[str, Any]] = {}
    for task_id in TASK_IDS:
        leaderboard = build_leaderboard(task_id, records)
        path = atomic_write_json(
            output_root / f"{task_id}_scratch_leaderboard.json", leaderboard
        )
        leaderboards[task_id] = leaderboard
        leaderboard_files[task_id] = {
            "path": path.name,
            "sha256": hash_file(path),
            "status": leaderboard["status"],
        }
    oof_path = _write_oof_manifest(output_root, runtime_root, records)
    visualization_path = render_visualizations(
        output_root=output_root, runtime_root=runtime_root
    )
    status_counts = {
        status: sum(row["status"] == status for row in records)
        for status in ("completed", "skipped", "blocked", "failed", "timeout")
    }
    summary = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "baseline_commit": "16bebd18a0bc722afcbc4b841610bf76ce9503e4",
        "root_seed": ROOT_SEED,
        "repeat_model_seeds": list(REPEAT_SEEDS),
        "fold_ids": list(FOLD_IDS),
        "lane": LANE,
        "top_models": {task: list(models) for task, models in TOP_MODELS.items()},
        "budget": asdict(budget),
        "hpo_performed": False,
        "expected_cells": EXPECTED_CELLS,
        "result_count": len(records),
        "status_counts": status_counts,
        "legal_completion_rate": status_counts["completed"] / EXPECTED_CELLS,
        "results_file": results_path.name,
        "results_sha256": hash_file(results_path),
        "leaderboards": leaderboard_files,
        "oof_manifest": {"path": oof_path.name, "sha256": hash_file(oof_path)},
        "visualization_manifest": {
            "path": visualization_path.name,
            "sha256": hash_file(visualization_path),
        },
        "source_hashes": {
            "stage2_runner_sha256": hash_file(TRACK_DIR / "facies_p5_stage2.py"),
            "stage2_results_sha256": hash_file(STAGE2_RESULTS_PATH),
            "stage2_summary_sha256": hash_file(STAGE2_SUMMARY_PATH),
            "source_lock_sha256": hash_file(SOURCE_LOCK_PATH),
            "stage3_runner_sha256": hash_file(Path(__file__)),
        },
        "tasks": {
            task_id: {
                "label_version": get_task_spec(task_id).label_version,
                "head_num_classes": int(get_task_spec(task_id).metadata["num_classes"]),
                "manifest_stable_hash": stage2.LOCKED_MANIFEST_STABLE_HASHES[task_id],
                "expected_cells": 45,
                "completed_cells": leaderboards[task_id]["completed_cells"],
                "completion_rate": leaderboards[task_id]["completion_rate"],
                "ranking_status": leaderboards[task_id]["status"],
            }
            for task_id in TASK_IDS
        },
        "environment": stage2._environment(device),
        "gpu_contract": {
            "device": "cuda:0",
            "lock_env": "VOLVE_P5_GPU_LOCK",
            "lock_name": EXPECTED_GPU_LOCK.name,
            "mechanism": "fcntl.flock(LOCK_EX)",
        },
        "tasks_are_independent": True,
        "cross_task_and_cross_lane_ranking_forbidden": True,
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
    }
    atomic_write_json(output_root / "p5_stage3_summary.json", summary)
    return summary


def run_stage3(
    *,
    manifest_paths: Mapping[str, Path],
    processed_root: Path,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
    budget: Stage3Budget | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    active_budget = Stage3Budget() if budget is None else budget
    if set(manifest_paths) != set(TASK_IDS):
        raise ValueError(f"manifest paths must be supplied for exactly {TASK_IDS}")
    lock_path = validate_gpu_contract(device, os.environ.get("VOLVE_P5_GPU_LOCK"))
    cells = expected_cells()
    prepared: dict[tuple[str, int], PreparedFold] = {}
    preparation_errors: dict[tuple[str, int], Exception] = {}
    for task_id in TASK_IDS:
        for fold_id in FOLD_IDS:
            try:
                prepared[(task_id, fold_id)] = prepare_fold(
                    task_id=task_id,
                    fold_id=fold_id,
                    manifest_path=manifest_paths[task_id],
                    processed_root=processed_root,
                    budget=active_budget,
                )
            except Exception as exc:
                preparation_errors[(task_id, fold_id)] = exc

    records: list[dict[str, Any]] = []
    with stage2.GpuFlock(lock_path) as gpu_lock:
        for index, cell in enumerate(cells, start=1):
            fold_key = (cell.task_id, cell.fold_id)
            if fold_key in preparation_errors:
                result = _blocked_result(cell, preparation_errors[fold_key], active_budget)
            else:
                fold = prepared[fold_key]
                resumed = (
                    _load_resumable_result(runtime_root, cell, fold, active_budget)
                    if resume
                    else None
                )
                result = (
                    resumed
                    if resumed is not None
                    else run_cell(
                        cell=cell,
                        prepared=fold,
                        budget=active_budget,
                        runtime_root=runtime_root,
                        device=device,
                        gpu_lock_wait_seconds=gpu_lock.wait_seconds,
                    )
                )
            _write_runtime_result(runtime_root, cell, result)
            records.append(dict(result))
            metric = result.get("validation_metrics", {}).get("miou")
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{EXPECTED_CELLS}",
                        "cell_key": cell.key,
                        "status": result["status"],
                        "miou": metric,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return collate_results(
        records=records,
        output_root=output_root,
        runtime_root=runtime_root,
        budget=active_budget,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run/resume all 90 locked development cells")
    run.add_argument("--processed-root", type=Path, required=True)
    run.add_argument("--f3-manifest", type=Path, required=True)
    run.add_argument("--penobscot-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_PORTABLE_OUTPUT)
    run.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_OUTPUT)
    run.add_argument("--no-resume", action="store_true")
    visualize = subparsers.add_parser(
        "visualize", help="rebuild figures from archived OOF predictions only"
    )
    visualize.add_argument("--output-root", type=Path, default=DEFAULT_PORTABLE_OUTPUT)
    visualize.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "visualize":
        path = render_visualizations(
            output_root=args.output_root, runtime_root=args.runtime_root
        )
        print(path)
        return
    summary = run_stage3(
        manifest_paths={
            "facies_f3": args.f3_manifest,
            "facies_penobscot": args.penobscot_manifest,
        },
        processed_root=args.processed_root,
        output_root=args.output_root,
        runtime_root=args.runtime_root,
        device=torch.device("cuda:0"),
        resume=not args.no_resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
