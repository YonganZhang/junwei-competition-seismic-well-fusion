#!/usr/bin/env python3
"""P5 Stage-4 known-holdout confirmation for the two facies tasks.

This runner is deliberately fail-closed.  It first verifies the committed
Stage-3 leaderboards, the locked P4 spatial split, and the historical P4
``TEST_CONSUMED`` evidence.  It then refits each frozen winner from scratch
using the full legal development *population* and the unchanged Stage-3
40-update recipe.  Only after both refit checkpoints and their evidence have
been persisted does it mark a new, track-private single-use access record and
read the already-seen holdout labels.

F3 and Penobscot never share a head, preprocessor, checkpoint, prediction
archive, metric object, or figure.  The P4 lifecycle files are read-only and
their hashes are checked again after confirmation.
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
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from _models.facies._p5_common import SOURCE_LOCK_PATH, source_lock  # noqa: E402

import facies_p5_stage2 as stage2  # noqa: E402
import facies_p5_stage3 as stage3  # noqa: E402
from p4_data import FaciesArchive, FoldPreprocessor, inverse_sqrt_class_weights  # noqa: E402
from p4_losses import build_loss, softmax_probabilities  # noqa: E402
from p4_metrics import confidence_entropy_error, evaluate_probabilities  # noqa: E402
from p4_tasks import LABEL_VERSIONS, TASK_IDS, get_task_spec  # noqa: E402


ROOT_SEED = 2693
SCHEMA_VERSION = "facies-p5-stage4-known-holdout-v1"
EVIDENCE_CLASS = "previously_seen_reusable_holdout"
EXPECTED_GPU_LOCK = Path("/mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock")
DEFAULT_STAGE3_ROOT = TRACK_DIR / "_outputs" / "p5_stage3"
DEFAULT_OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p5_stage4_confirmation"
DEFAULT_RUNTIME_ROOT = TRACK_DIR / "_outputs" / "p5_stage4_confirmation_runtime"

FROZEN_WINNERS: Mapping[str, str] = {
    "facies_f3": "smp_fpn_r18",
    "facies_penobscot": "smp_deeplabv3plus_r18",
}
EXPECTED_STAGE3_SUMMARY_SHA256 = (
    "129cc31db4aec34c9e13789c0dcd1f4f6f1402e97f2dd020f91c5f880744c911"
)
EXPECTED_STAGE3_LEADERBOARD_SHA256: Mapping[str, str] = {
    "facies_f3": "f79d1027a903eba4ba93ad8ec517bd7827098f9599ac526c5991184d56562242",
    "facies_penobscot": "8904769dc0b4b806785ce304d449ecf18a132b2ca8f320b61475c3f9c2f74b21",
}
EXPECTED_MANIFEST_STABLE_HASH: Mapping[str, str] = dict(
    stage2.LOCKED_MANIFEST_STABLE_HASHES
)


@dataclass(frozen=True)
class Stage4Budget:
    """Exact immutable copy of the accepted Stage-3 neural recipe."""

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
        accepted = stage3.Stage3Budget()
        if asdict(self) != asdict(accepted):
            raise ValueError("Stage-4 must preserve the complete Stage-3 budget")


@dataclass(frozen=True)
class LockedTask:
    task_id: str
    winner_model_id: str
    label_version: str
    num_classes: int
    leaderboard_sha256: str
    manifest_stable_hash: str
    manifest_file_sha256: str
    development_sample_ids: tuple[str, ...]
    development_groups: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    test_groups: tuple[str, ...]
    prior_lifecycle_sha256: str
    prior_test_consumed_at: str


@dataclass(frozen=True)
class PreparedRefit:
    locked: LockedTask
    images: np.ndarray
    labels: np.ndarray
    preprocessor: FoldPreprocessor
    update_schedule: np.ndarray
    development_support: tuple[int, ...]
    sampled_sample_ids: tuple[str, ...]
    sampled_groups: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _seed_model(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    return {
        "root_seed": ROOT_SEED,
        "model_seed": seed,
        "seed_tree": {
            "stage4_refit_model": seed,
            "sampler_root": ROOT_SEED,
        },
        "derivation": "derive_seed(2693,'model',task_id,model_id,'scratch','p5-stage4')",
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
    }


def _validate_new_output(output_root: Path, runtime_root: Path) -> None:
    for path, role in ((Path(output_root), "portable"), (Path(runtime_root), "runtime")):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Stage-4 {role} output is nonempty: {path}; known-holdout access is single-use"
            )


def validate_gpu_contract(device: torch.device, lock_value: str | None) -> Path:
    if device != torch.device("cuda:0"):
        raise RuntimeError("facies Stage-4 must execute on cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("facies Stage-4 requires a real cuda:0 device")
    if not lock_value or Path(lock_value) != EXPECTED_GPU_LOCK:
        raise RuntimeError(
            f"VOLVE_P5_GPU_LOCK must equal the frozen lock {EXPECTED_GPU_LOCK}"
        )
    return EXPECTED_GPU_LOCK


def validate_stage3(stage3_root: Path) -> dict[str, Any]:
    """Verify committed ranking evidence before any refit or test access."""
    root = Path(stage3_root)
    summary_path = root / "p5_stage3_summary.json"
    if hash_file(summary_path) != EXPECTED_STAGE3_SUMMARY_SHA256:
        raise ValueError("Stage-3 summary hash changed; winner freeze is invalid")
    summary = _read_json(summary_path)
    if summary.get("schema_version") != stage3.RESULT_SCHEMA:
        raise ValueError("unexpected Stage-3 summary schema")
    if summary.get("root_seed") != ROOT_SEED or summary.get("lane") != "scratch":
        raise ValueError("Stage-3 seed/lane differs from frozen confirmation contract")
    if summary.get("status_counts", {}).get("completed") != 90:
        raise ValueError("Stage-3 did not preserve all 90 completed cells")
    if summary.get("legal_completion_rate") != 1.0:
        raise ValueError("Stage-3 completion is no longer rankable")
    if not summary.get("tasks_are_independent"):
        raise ValueError("Stage-3 no longer proves independent task ranking")
    if any(
        bool(summary.get(flag))
        for flag in ("test_archive_opened", "test_labels_read", "test_metrics_computed")
    ):
        raise ValueError("Stage-3 development-only firewall evidence changed")
    if summary.get("budget") != asdict(Stage4Budget()):
        raise ValueError("Stage-3 summary budget differs from Stage-4 frozen recipe")
    if hash_file(SOURCE_LOCK_PATH) != summary["source_hashes"]["source_lock_sha256"]:
        raise ValueError("facies source lock changed after Stage-3")

    tasks: dict[str, Any] = {}
    for task_id in TASK_IDS:
        board_record = summary["leaderboards"][task_id]
        board_path = root / str(board_record["path"])
        observed_sha = hash_file(board_path)
        expected_sha = EXPECTED_STAGE3_LEADERBOARD_SHA256[task_id]
        if observed_sha != expected_sha or board_record.get("sha256") != expected_sha:
            raise ValueError(f"{task_id} Stage-3 leaderboard hash changed")
        board = _read_json(board_path)
        if board.get("task_id") != task_id or board.get("lane") != "scratch":
            raise ValueError(f"{task_id} leaderboard identity/lane mismatch")
        if board.get("status") != "ranked" or board.get("completion_rate") != 1.0:
            raise ValueError(f"{task_id} Stage-3 leaderboard is not fully rankable")
        if board.get("frozen_test_consumed") is not False:
            raise ValueError(f"{task_id} Stage-3 leaderboard test firewall changed")
        winners = [entry for entry in board.get("entries", ()) if entry.get("rank") == 1]
        expected_winner = FROZEN_WINNERS[task_id]
        if len(winners) != 1 or winners[0].get("model_id") != expected_winner:
            raise ValueError(f"{task_id} unique Stage-3 winner changed")
        if not winners[0].get("rankable") or winners[0].get("completion_rate") != 1.0:
            raise ValueError(f"{task_id} winner is no longer rankable")
        lock = dict(source_lock(expected_winner))
        if lock.get("allowed_lanes") != ["scratch"]:
            raise ValueError(f"{task_id} winner scratch lane is not frozen")
        if lock.get("weights", {}).get("status") == "approved":
            raise ValueError(f"{task_id} winner unexpectedly entered a weight lane")
        task_summary = summary["tasks"][task_id]
        if task_summary.get("manifest_stable_hash") != EXPECTED_MANIFEST_STABLE_HASH[task_id]:
            raise ValueError(f"{task_id} Stage-3 split hash changed")
        tasks[task_id] = {
            "winner_model_id": expected_winner,
            "leaderboard_sha256": expected_sha,
            "mean_stage3_miou": float(winners[0]["mean_miou"]),
            "source_lock": lock,
        }
    return {
        "summary_sha256": EXPECTED_STAGE3_SUMMARY_SHA256,
        "source_lock_sha256": hash_file(SOURCE_LOCK_PATH),
        "tasks": tasks,
    }


def _validate_prior_lifecycle(
    task_id: str, lifecycle_path: Path, manifest_stable_hash: str
) -> dict[str, Any]:
    lifecycle = _read_json(lifecycle_path)
    if lifecycle.get("state") != "TEST_CONSUMED":
        raise ValueError(f"{task_id} P4 lifecycle is not TEST_CONSUMED")
    consumed_at = lifecycle.get("test_consumed_at")
    if not isinstance(consumed_at, str) or not consumed_at:
        raise ValueError(f"{task_id} P4 lifecycle lacks test_consumed_at")
    split = lifecycle.get("evidence", {}).get("SPLIT_LOCKED", {})
    consumed = lifecycle.get("evidence", {}).get("TEST_CONSUMED", {})
    if split.get("task_id") != task_id:
        raise ValueError(f"{task_id} P4 lifecycle task identity mismatch")
    if split.get("split_hash") != manifest_stable_hash:
        raise ValueError(f"{task_id} P4 SPLIT_LOCKED hash mismatch")
    if consumed.get("split_hash") != manifest_stable_hash:
        raise ValueError(f"{task_id} P4 TEST_CONSUMED split hash mismatch")
    return {
        "sha256": hash_file(lifecycle_path),
        "state": "TEST_CONSUMED",
        "test_consumed_at": consumed_at,
        "experiment_id": lifecycle.get("experiment_id"),
    }


def lock_task(
    *,
    task_id: str,
    manifest_path: Path,
    lifecycle_path: Path,
    stage3_evidence: Mapping[str, Any],
) -> LockedTask:
    manifest, manifest_file_sha256 = stage2.load_locked_manifest(task_id, manifest_path)
    if manifest.stable_hash() != EXPECTED_MANIFEST_STABLE_HASH[task_id]:
        raise ValueError(f"{task_id} split stable hash changed")
    development_ids = tuple(manifest.development_sample_ids)
    test_ids = tuple(manifest.test_sample_ids)
    development_groups = tuple(manifest.development_groups)
    test_groups = tuple(manifest.test_groups)
    if not development_ids or not test_ids:
        raise ValueError(f"{task_id} manifest has an empty outer partition")
    if set(development_ids) & set(test_ids) or set(development_groups) & set(test_groups):
        raise ValueError(f"{task_id} development/test overlap")
    validation_counts = Counter(
        sample_id for fold in manifest.folds for sample_id in fold.validation_sample_ids
    )
    if set(validation_counts) != set(development_ids) or set(validation_counts.values()) != {1}:
        raise ValueError(f"{task_id} OOF folds do not cover development exactly once")
    metadata = manifest.metadata
    outer = metadata.get("outer_split", {})
    development_range = tuple(outer.get("development_inline_range", ()))
    guard_range = tuple(outer.get("external_guard_inline_range", ()))
    test_range = tuple(outer.get("test_inline_range", ()))
    if not all(len(values) == 2 for values in (development_range, guard_range, test_range)):
        raise ValueError(f"{task_id} manifest lacks frozen outer spatial ranges")
    if not (
        int(development_range[1]) < int(guard_range[0])
        <= int(guard_range[1]) < int(test_range[0])
    ):
        raise ValueError(f"{task_id} outer spatial guard is invalid")
    if any(not int(development_range[0]) <= int(group) <= int(development_range[1]) for group in development_groups):
        raise ValueError(f"{task_id} development group escapes outer range")
    if any(not int(test_range[0]) <= int(group) <= int(test_range[1]) for group in test_groups):
        raise ValueError(f"{task_id} test group escapes outer range")
    prior = _validate_prior_lifecycle(task_id, lifecycle_path, manifest.stable_hash())
    spec = get_task_spec(task_id)
    return LockedTask(
        task_id=task_id,
        winner_model_id=str(stage3_evidence["winner_model_id"]),
        label_version=spec.label_version,
        num_classes=int(spec.metadata["num_classes"]),
        leaderboard_sha256=str(stage3_evidence["leaderboard_sha256"]),
        manifest_stable_hash=manifest.stable_hash(),
        manifest_file_sha256=manifest_file_sha256,
        development_sample_ids=development_ids,
        development_groups=development_groups,
        test_sample_ids=test_ids,
        test_groups=test_groups,
        prior_lifecycle_sha256=str(prior["sha256"]),
        prior_test_consumed_at=str(prior["test_consumed_at"]),
    )


def prepare_refit(
    locked: LockedTask, processed_root: Path, budget: Stage4Budget
) -> PreparedRefit:
    """Fit preprocessing on every legal development sample; never open test.h5."""
    archive = stage2.Stage2DevelopmentArchive(locked.task_id, processed_root)
    records, raw_images, labels = stage2._materialize_selected(
        archive, locked.development_sample_ids
    )
    if tuple(record.sample_id for record in records) != locked.development_sample_ids:
        raise ValueError("development archive order/identity differs from locked manifest")
    observed_groups = {str(record.inline) for record in records}
    if observed_groups != set(locked.development_groups):
        raise ValueError("development archive groups differ from locked manifest")
    histogram = np.zeros(locked.num_classes, dtype=np.int64)
    for label in labels:
        histogram += np.bincount(
            label.reshape(-1), minlength=locked.num_classes
        )[: locked.num_classes]
    class_weights = inverse_sqrt_class_weights(histogram)
    fit_values = np.concatenate([image.reshape(-1) for image in raw_images]).astype(
        np.float32, copy=False
    )
    normalization = fit_zscore(fit_values)
    recovered = denormalize(normalize(raw_images[0], normalization), normalization)
    roundtrip_error = float(np.max(np.abs(recovered - raw_images[0])))
    if not math.isfinite(roundtrip_error) or roundtrip_error > 1e-2:
        raise ValueError(f"full-development normalization round-trip failed: {roundtrip_error}")
    preprocessor = FoldPreprocessor(
        task_id=locked.task_id,
        label_version=locked.label_version,
        normalization=normalization,
        class_weights=tuple(float(value) for value in class_weights),
        class_histogram=tuple(int(value) for value in histogram),
        fit_sample_count=len(records),
        fit_sample_ids_hash=hash_payload(list(locked.development_sample_ids)),
        roundtrip_max_abs_error=roundtrip_error,
    )
    images = np.stack(
        [normalize(image, normalization).astype(np.float32) for image in raw_images]
    )[:, None]
    label_array = np.stack(labels).astype(np.int64)
    if not np.isfinite(images).all():
        raise ValueError("full-development normalization produced NaN/Inf")
    schedule_seed = derive_seed(
        ROOT_SEED, "sampler", locked.task_id, "full-development", budget.profile_id
    )
    schedule = stage2.fixed_update_schedule(len(records), budget, seed=schedule_seed)
    sampled_indices = schedule.reshape(-1)
    sampled_ids = tuple(records[int(index)].sample_id for index in sampled_indices)
    sampled_groups = tuple(str(records[int(index)].inline) for index in sampled_indices)
    return PreparedRefit(
        locked=locked,
        images=images,
        labels=label_array,
        preprocessor=preprocessor,
        update_schedule=schedule,
        development_support=tuple(int(value) for value in histogram),
        sampled_sample_ids=sampled_ids,
        sampled_groups=sampled_groups,
    )


def _configuration(locked: LockedTask, budget: Stage4Budget) -> dict[str, Any]:
    model_seed = derive_seed(
        ROOT_SEED,
        "model",
        locked.task_id,
        locked.winner_model_id,
        "scratch",
        "p5-stage4",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "task_id": locked.task_id,
        "label_version": locked.label_version,
        "num_classes": locked.num_classes,
        "winner_model_id": locked.winner_model_id,
        "winner_selection": "unique_rank_1_from_frozen_stage3_leaderboard",
        "leaderboard_sha256": locked.leaderboard_sha256,
        "lane": "scratch",
        "weights_loaded": False,
        "root_seed": ROOT_SEED,
        "model_seed": model_seed,
        "budget": asdict(budget),
        "refit_population": "all_locked_manifest_development_sample_ids",
        "optimizer_sampling": "frozen_40x2_with_replacement_from_full_development_population",
        "normalization": "zscore_fit_all_legal_development_only",
        "denoise": "identity",
        "target_transform": "identity_integer_class_ids",
        "loss": "inverse_sqrt_weighted_cross_entropy_on_raw_logits",
        "activation": "none_training_softmax_inference_only",
        "calibration": "identity_no_test_fit",
        "manifest_stable_hash": locked.manifest_stable_hash,
        "manifest_file_sha256": locked.manifest_file_sha256,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "hpo": False,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def run_refit(
    *,
    prepared: PreparedRefit,
    budget: Stage4Budget,
    runtime_root: Path,
    device: torch.device,
    gpu_lock_wait_seconds: float,
) -> dict[str, Any]:
    locked = prepared.locked
    configuration = _configuration(locked, budget)
    configuration_hash = hash_payload(configuration)
    model_seed = int(configuration["model_seed"])
    seed_report = _seed_model(model_seed)
    discovered = discover_model("facies", locked.winner_model_id)
    task_spec = get_task_spec(locked.task_id)
    model = discovered.build(
        task_spec, num_classes=locked.num_classes, lane="scratch"
    ).to(device)
    criterion = build_loss(
        budget.loss_id,
        num_classes=locked.num_classes,
        class_weights=torch.tensor(
            prepared.preprocessor.class_weights, dtype=torch.float32, device=device
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay
    )
    checkpoint_path = Path(runtime_root) / locked.task_id / "refit_final.ckpt"
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gradients_finite = True
    model.train()
    for update, indices in enumerate(prepared.update_schedule, start=1):
        images = torch.as_tensor(
            prepared.images[indices], dtype=torch.float32, device=device
        )
        labels = torch.as_tensor(
            prepared.labels[indices], dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        stage2._assert_logits(logits, images, locked.num_classes)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("refit loss is NaN/Inf")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                gradients_finite = False
                raise ValueError("refit gradient is NaN/Inf")
        optimizer.step()
        history.append({"update": update, "train_loss": float(loss.detach())})
        if time.perf_counter() - started > budget.max_wall_seconds:
            raise TimeoutError("Stage-4 refit exceeded the frozen Stage-3 wall-clock budget")
    wall_seconds = time.perf_counter() - started
    save_checkpoint(
        checkpoint_path,
        epoch=budget.max_updates,
        model_state=_cpu_state_dict(model),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        scaler_state=None,
        config_hash=configuration_hash,
        split_hash=locked.manifest_stable_hash,
        trainer_state={
            "next_epoch": budget.max_updates + 1,
            "global_step": budget.max_updates,
            # The shared checkpoint schema requires these resume fields.  A
            # final fixed-budget refit has no validation selection: the final
            # update and its train loss are stored only as schema-compatible
            # resume anchors, never presented as a best validation epoch.
            "best_epoch": budget.max_updates,
            "best_val_loss": history[-1]["train_loss"],
            "epochs_without_improvement": 0,
            "stopped_early": False,
            "history": history,
        },
        seed_report=seed_report,
        environment=stage2._environment(device),
        extra={
            "stage": "p5_stage4_full_development_refit",
            "task_id": locked.task_id,
            "winner_model_id": locked.winner_model_id,
            "leaderboard_sha256": locked.leaderboard_sha256,
            "manifest_stable_hash": locked.manifest_stable_hash,
            "preprocessor_hash": hash_payload(prepared.preprocessor.to_dict()),
            "test_access": False,
            "evidence_class": EVIDENCE_CLASS,
        },
    )
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config_hash") != configuration_hash:
        raise ValueError("refit checkpoint configuration hash mismatch")
    if checkpoint.get("split_hash") != locked.manifest_stable_hash:
        raise ValueError("refit checkpoint split hash mismatch")
    restored = discovered.build(
        task_spec, num_classes=locked.num_classes, lane="scratch"
    ).to(device)
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval()
    probe = torch.as_tensor(prepared.images[:1], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        difference = float(torch.max(torch.abs(model(probe) - restored(probe))).cpu())
    if difference > 1e-6:
        raise ValueError(f"refit checkpoint round-trip changed logits by {difference}")
    del restored, model, optimizer, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": locked.task_id,
        "label_version": locked.label_version,
        "num_classes": locked.num_classes,
        "winner_model_id": locked.winner_model_id,
        "lane": "scratch",
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "source_lock": dict(source_lock(locked.winner_model_id)),
        "source_lock_sha256": hash_file(SOURCE_LOCK_PATH),
        "split": {
            "manifest_stable_hash": locked.manifest_stable_hash,
            "manifest_file_sha256": locked.manifest_file_sha256,
            "development_sample_count": len(locked.development_sample_ids),
            "development_group_count": len(locked.development_groups),
            "development_sample_ids_hash": hash_payload(list(locked.development_sample_ids)),
            "development_groups_hash": hash_payload(list(locked.development_groups)),
            "test_sample_overlap": 0,
            "test_group_overlap": 0,
        },
        "preprocessing": {
            **prepared.preprocessor.to_dict(),
            "fit_scope": "all_locked_legal_development_samples_only",
            "preprocessor_hash": hash_payload(prepared.preprocessor.to_dict()),
        },
        "training": {
            "optimizer": "AdamW",
            "updates_completed": budget.max_updates,
            "batch_size": budget.batch_size,
            "history": history,
            "development_population_samples": len(locked.development_sample_ids),
            "development_population_groups": len(locked.development_groups),
            "sample_draws": len(prepared.sampled_sample_ids),
            "unique_samples_drawn": len(set(prepared.sampled_sample_ids)),
            "unique_groups_drawn": len(set(prepared.sampled_groups)),
            "sampled_sample_ids_hash": hash_payload(list(prepared.sampled_sample_ids)),
            "sampled_groups_hash": hash_payload(list(prepared.sampled_groups)),
            "full_population_eligible_for_every_draw": True,
            "validation_used": False,
            "early_stopping": False,
            "hpo": False,
            "gradients_finite": gradients_finite,
        },
        "checkpoint": {
            "runtime_relative_path": checkpoint_path.relative_to(runtime_root).as_posix(),
            "sha256": hash_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "kind": "final_fixed_update_refit_checkpoint_no_test_selection",
            "prediction_max_abs_difference": difference,
        },
        "resources": {
            "device": "cuda:0",
            "parameters": parameters,
            "trainable_parameters": trainable,
            "refit_wall_seconds": wall_seconds,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_lock_wait_seconds_excluded": gpu_lock_wait_seconds,
            "gpu_lock_name": EXPECTED_GPU_LOCK.name,
            "gpu_lock_held": True,
            "download_bytes": 0,
        },
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
    }


def _write_access_started(
    output_root: Path,
    locked_tasks: Mapping[str, LockedTask],
    refit_evidence: Mapping[str, Mapping[str, Any]],
) -> Path:
    state_path = Path(output_root) / "stage4_state.json"
    if state_path.exists():
        raise RuntimeError("Stage-4 state already exists; holdout confirmation cannot be replayed")
    for task_id in TASK_IDS:
        evidence = refit_evidence[task_id]
        checkpoint = evidence["checkpoint"]
        if evidence.get("test_archive_opened") or evidence.get("test_labels_read"):
            raise ValueError("refit evidence was contaminated by test access")
        if not checkpoint.get("sha256"):
            raise ValueError("refit checkpoint hash missing before test authorization")
    return atomic_write_json(
        state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "state": "TEST_ACCESS_STARTED",
            "started_at": _utc_now(),
            "evidence_class": EVIDENCE_CLASS,
            "prior_test_consumed": True,
            "fresh_blind": False,
            "single_use": True,
            "p4_lifecycle_mutation_forbidden": True,
            "tasks": {
                task_id: {
                    "winner_model_id": locked_tasks[task_id].winner_model_id,
                    "leaderboard_sha256": locked_tasks[task_id].leaderboard_sha256,
                    "manifest_stable_hash": locked_tasks[task_id].manifest_stable_hash,
                    "prior_lifecycle_sha256": locked_tasks[task_id].prior_lifecycle_sha256,
                    "prior_test_consumed_at": locked_tasks[task_id].prior_test_consumed_at,
                    "configuration_hash": refit_evidence[task_id]["configuration_hash"],
                    "checkpoint_sha256": refit_evidence[task_id]["checkpoint"]["sha256"],
                    "confirmation_completed": False,
                }
                for task_id in TASK_IDS
            },
        },
    )


def _assert_access_started(
    state_path: Path, locked: LockedTask, refit_evidence: Mapping[str, Any]
) -> None:
    if not Path(state_path).is_file():
        raise RuntimeError(
            "known holdout labels require the single-use TEST_ACCESS_STARTED state"
        )
    state = _read_json(state_path)
    if state.get("state") != "TEST_ACCESS_STARTED" or not state.get("single_use"):
        raise RuntimeError("known holdout labels require the single-use TEST_ACCESS_STARTED state")
    if state.get("evidence_class") != EVIDENCE_CLASS:
        raise RuntimeError("known holdout evidence class changed")
    task = state.get("tasks", {}).get(locked.task_id, {})
    expected = {
        "winner_model_id": locked.winner_model_id,
        "leaderboard_sha256": locked.leaderboard_sha256,
        "manifest_stable_hash": locked.manifest_stable_hash,
        "prior_lifecycle_sha256": locked.prior_lifecycle_sha256,
        "configuration_hash": refit_evidence["configuration_hash"],
        "checkpoint_sha256": refit_evidence["checkpoint"]["sha256"],
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise RuntimeError(f"holdout firewall binding mismatch for {locked.task_id}/{key}")


def consume_known_holdout(
    *,
    locked: LockedTask,
    refit_evidence: Mapping[str, Any],
    processed_root: Path,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read labels only after the track-private access state is durably bound."""
    state_path = Path(output_root) / "stage4_state.json"
    _assert_access_started(state_path, locked, refit_evidence)
    # Validate every refit-side dependency before the irreversible label read.
    preprocessor_payload = dict(refit_evidence["preprocessing"])
    preprocessor_payload.pop("fit_scope", None)
    preprocessor_payload.pop("preprocessor_hash", None)
    preprocessor = FoldPreprocessor.from_dict(preprocessor_payload)
    discovered = discover_model("facies", locked.winner_model_id)
    model = discovered.build(
        get_task_spec(locked.task_id), num_classes=locked.num_classes, lane="scratch"
    ).to(device)
    checkpoint_path = Path(runtime_root) / refit_evidence["checkpoint"]["runtime_relative_path"]
    if hash_file(checkpoint_path) != refit_evidence["checkpoint"]["sha256"]:
        raise ValueError("refit checkpoint missing/corrupt immediately before test inference")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config_hash") != refit_evidence["configuration_hash"]:
        raise ValueError("refit checkpoint/config binding changed before test inference")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    archive = FaciesArchive(locked.task_id, processed_root)
    records = archive.frozen_test_index(labels_consumed=True)
    if tuple(record.sample_id for record in records) != locked.test_sample_ids:
        raise ValueError(f"{locked.task_id} test archive differs from locked test IDs")
    if {str(record.inline) for record in records} != set(locked.test_groups):
        raise ValueError(f"{locked.task_id} test archive differs from locked test groups")
    sample_ids: list[str] = []
    inline_values: list[int] = []
    seismic_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        batches = archive.iter_model_batches(
            records,
            preprocessor,
            batch_size=batch_size,
            shuffle=False,
            seed=ROOT_SEED,
            include_targets=True,
        )
        for batch in batches:
            images_np = np.asarray(batch.inputs["seismic"], dtype=np.float32)
            labels_np = np.asarray(batch.targets["facies"], dtype=np.int64)
            images = torch.as_tensor(images_np, dtype=torch.float32, device=device)
            logits = model(images)
            stage2._assert_logits(logits, images, locked.num_classes)
            probabilities = softmax_probabilities(logits).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("known-holdout probabilities contain NaN/Inf")
            sample_ids.extend(batch.sample_ids)
            inline_values.extend(int(value) for value in batch.coordinates["inline"])
            seismic_chunks.append(images_np[:, 0])
            label_chunks.append(labels_np)
            probability_chunks.append(probabilities)
    inference_seconds = time.perf_counter() - started
    labels = np.concatenate(label_chunks, axis=0)
    probabilities = np.concatenate(probability_chunks, axis=0)
    seismic = np.concatenate(seismic_chunks, axis=0)
    metrics, _ = evaluate_probabilities(
        probabilities,
        labels,
        num_classes=locked.num_classes,
        n_bins=15,
        require_all_classes=True,
    )
    prediction, confidence, entropy, error = confidence_entropy_error(probabilities, labels)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_context": {
                "track_id": "facies",
                "task_id": locked.task_id,
                "label_version": locked.label_version,
                "winner_model_id": locked.winner_model_id,
                "lane": "scratch",
                "configuration_hash": refit_evidence["configuration_hash"],
                "checkpoint_sha256": refit_evidence["checkpoint"]["sha256"],
                "leaderboard_sha256": locked.leaderboard_sha256,
                "manifest_stable_hash": locked.manifest_stable_hash,
            },
            "test_sample_count": len(records),
            "test_group_count": len(set(inline_values)),
            "valid_label_ids": list(range(locked.num_classes)),
            "ignore_index": None,
            "ignored_pixels": 0,
            "prediction_rule": "argmax_of_raw_logit_softmax",
            "calibration_fit": "none_identity_no_test_fit",
            "evidence_class": EVIDENCE_CLASS,
            "prior_test_consumed": True,
            "fresh_blind": False,
            "test_archive_opened": True,
            "test_labels_read": True,
            "test_metrics_computed": True,
            "inference_wall_seconds": inference_seconds,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    )
    numeric = [
        metrics["accuracy"], metrics["miou"], metrics["macro_f1"],
        metrics["nll"], metrics["brier"], metrics["ece"],
        *metrics["per_class_iou"], *metrics["per_class_f1"],
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("formal known-holdout metrics contain NaN/Inf")
    task_runtime = Path(runtime_root) / locked.task_id
    prediction_path = _atomic_save_npz(
        task_runtime / "known_holdout_predictions.npz",
        sample_ids=np.asarray(sample_ids, dtype=str),
        inline=np.asarray(inline_values, dtype=np.int64),
        seismic=seismic.astype(np.float16),
        labels=labels.astype(np.uint8),
        probabilities=probabilities.astype(np.float16),
        prediction=prediction.astype(np.uint8),
        confidence=confidence.astype(np.float16),
        entropy=entropy.astype(np.float16),
        error=error.astype(np.uint8),
    )
    arrays = {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "inline": np.asarray(inline_values, dtype=np.int64),
        "seismic": seismic.astype(np.float16),
        "labels": labels.astype(np.uint8),
        "probabilities": probabilities.astype(np.float16),
        "prediction": prediction.astype(np.uint8),
        "confidence": confidence.astype(np.float16),
        "entropy": entropy.astype(np.float16),
        "error": error.astype(np.uint8),
    }
    prediction_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": locked.task_id,
        "runtime_relative_path": prediction_path.relative_to(runtime_root).as_posix(),
        "sha256": hash_file(prediction_path),
        "bytes": prediction_path.stat().st_size,
        "storage_boundary": "ignored_track_private_runtime_artifact",
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
        "sample_count": len(sample_ids),
        "sample_ids_hash": hash_payload(sample_ids),
        "test_groups_hash": hash_payload(sorted(set(inline_values))),
        "configuration_hash": refit_evidence["configuration_hash"],
        "checkpoint_sha256": refit_evidence["checkpoint"]["sha256"],
        "leaderboard_sha256": locked.leaderboard_sha256,
        "manifest_stable_hash": locked.manifest_stable_hash,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
    }
    del model, checkpoint, probabilities, labels, seismic
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, prediction_manifest


def _select_representative(archive: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(archive["labels"])
    error = np.asarray(archive["error"])
    sample_ids = np.asarray(archive["sample_ids"]).astype(str)
    candidates: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        error_pixels = int(error[index].sum())
        pixels = int(error[index].size)
        item = {
            "index": index,
            "sample_id": str(sample_id),
            "gt_class_count": int(np.unique(labels[index]).size),
            "correct_pixels": pixels - error_pixels,
            "error_pixels": error_pixels,
        }
        item["eligible"] = bool(
            item["gt_class_count"] >= 2
            and item["correct_pixels"] > 0
            and item["error_pixels"] > 0
        )
        candidates.append(item)
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: item["sample_id"],
    )
    if eligible:
        selected = eligible[ROOT_SEED % len(eligible)]
        outcome = "informative_gate_then_seed_modulo_sorted_ids"
    else:
        selected = sorted(candidates, key=lambda item: item["sample_id"])[0]
        outcome = "deterministic_fallback_no_informative_sample_exists"
    return {
        **selected,
        "selection_rule": (
            "prefer GT>=2 classes and both correct/error pixels; sort by sample_id; "
            "select seed2693 modulo eligible count; never alter predictions or metrics"
        ),
        "selection_outcome": outcome,
        "eligible_candidate_count": len(eligible),
    }


def render_task_figure(
    *,
    task_id: str,
    prediction_path: Path,
    metrics_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    metrics = _read_json(metrics_path)
    with np.load(prediction_path, allow_pickle=False) as handle:
        required = {
            "sample_ids", "inline", "seismic", "labels", "prediction",
            "confidence", "entropy", "error",
        }
        missing = required - set(handle.files)
        if missing:
            raise ValueError(f"prediction archive missing {sorted(missing)}")
        arrays = {name: handle[name] for name in required}
    selected = _select_representative(arrays)
    index = int(selected["index"])
    num_classes = len(metrics["per_class_support"])
    seismic = np.asarray(arrays["seismic"][index], dtype=np.float32)
    amplitude = float(np.percentile(np.abs(seismic), 99.0))
    if not math.isfinite(amplitude) or amplitude <= 0:
        amplitude = 1.0
    fig = plt.figure(figsize=(17, 8.5))
    grid = fig.add_gridspec(2, 12, height_ratios=(1.0, 1.15))
    top = [fig.add_subplot(grid[0, 2 * item : 2 * item + 2]) for item in range(6)]
    class_cmap = plt.get_cmap("tab10", num_classes)
    top[0].imshow(seismic, cmap="gray", vmin=-amplitude, vmax=amplitude)
    top[1].imshow(arrays["labels"][index], cmap=class_cmap, vmin=-0.5, vmax=num_classes - 0.5)
    top[2].imshow(arrays["prediction"][index], cmap=class_cmap, vmin=-0.5, vmax=num_classes - 0.5)
    top[3].imshow(arrays["confidence"][index], cmap="viridis", vmin=0.0, vmax=1.0)
    top[4].imshow(arrays["entropy"][index], cmap="magma", vmin=0.0, vmax=1.0)
    top[5].imshow(arrays["error"][index], cmap="Reds", vmin=0, vmax=1)
    for axis, title in zip(top, ("Seismic", "Ground truth", "Prediction", "Confidence", "Entropy", "Error")):
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    confusion_axis = fig.add_subplot(grid[1, 0:4])
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    image = confusion_axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    confusion_axis.set_title("Row-normalized confusion")
    confusion_axis.set_xlabel("Predicted class")
    confusion_axis.set_ylabel("Ground-truth class")
    confusion_axis.set_xticks(range(num_classes))
    confusion_axis.set_yticks(range(num_classes))
    fig.colorbar(image, ax=confusion_axis, fraction=0.046)

    class_axis = fig.add_subplot(grid[1, 4:8])
    x = np.arange(num_classes)
    class_axis.bar(x - 0.2, metrics["per_class_iou"], width=0.4, label="IoU", color="#4C72B0")
    class_axis.bar(x + 0.2, metrics["per_class_f1"], width=0.4, label="F1", color="#DD8452")
    class_axis.set_ylim(0, 1)
    class_axis.set_xticks(x)
    class_axis.set_xlabel("Facies class ID")
    class_axis.set_ylabel("Score")
    class_axis.set_title("Per-class IoU/F1 (support annotated)")
    class_axis.legend(loc="upper right")
    for class_id, support in enumerate(metrics["per_class_support"]):
        class_axis.text(class_id, 0.02, f"n={support}", rotation=90, fontsize=6, ha="center", va="bottom")

    reliability_axis = fig.add_subplot(grid[1, 8:12])
    bins = [entry for entry in metrics["reliability_bins"] if entry["count"] > 0]
    reliability_axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    reliability_axis.plot(
        [entry["mean_confidence"] for entry in bins],
        [entry["accuracy"] for entry in bins],
        marker="o", color="#55A868",
    )
    reliability_axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
    reliability_axis.set_title(f"Reliability | ECE={metrics['ece']:.4f}, NLL={metrics['nll']:.4f}")

    fig.suptitle(
        f"{task_id} | known reused holdout | {selected['sample_id']} | inline={int(arrays['inline'][index])}\n"
        f"Accuracy={metrics['accuracy']:.4f} | mIoU={metrics['miou']:.4f} | Macro-F1={metrics['macro_f1']:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "figure": output_path.name,
        "figure_sha256": hash_file(output_path),
        "figure_bytes": output_path.stat().st_size,
        "prediction_sha256": hash_file(prediction_path),
        "metrics_sha256": hash_file(metrics_path),
        "selection": selected,
        "rendered_from_archived_predictions_only": True,
        "model_or_dataset_loaded": False,
        "threshold_or_calibration_tuned": False,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _relative_artifact(path: Path, output_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "sha256": hash_file(path),
        "bytes": path.stat().st_size,
    }


def verify_portable_artifacts(output_root: Path, runtime_root: Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = _read_json(root / "p5_stage4_artifact_manifest.json")
    for record in manifest.get("portable_artifacts", ()):
        path = root / record["path"]
        if not path.is_file() or hash_file(path) != record["sha256"]:
            raise ValueError(f"portable Stage-4 artifact missing/corrupt: {record['path']}")
    for task_id, record in manifest.get("runtime_artifacts", {}).items():
        for artifact in record.values():
            path = Path(runtime_root) / artifact["runtime_relative_path"]
            if not path.is_file() or hash_file(path) != artifact["sha256"]:
                raise ValueError(f"runtime Stage-4 artifact missing/corrupt: {task_id}")
    for path in root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if "/mnt/" in text or ".claude/worktrees" in text:
            raise ValueError(f"nonportable machine path serialized in {path}")
    return {
        "portable_artifact_count": len(manifest.get("portable_artifacts", ())),
        "runtime_task_count": len(manifest.get("runtime_artifacts", {})),
        "all_hashes_verified": True,
        "machine_paths_absent": True,
    }


def run_stage4(
    *,
    processed_root: Path,
    manifest_paths: Mapping[str, Path],
    lifecycle_paths: Mapping[str, Path],
    stage3_root: Path,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
    budget: Stage4Budget | None = None,
) -> dict[str, Any]:
    active_budget = Stage4Budget() if budget is None else budget
    if set(manifest_paths) != set(TASK_IDS) or set(lifecycle_paths) != set(TASK_IDS):
        raise ValueError(f"Stage-4 requires manifest/lifecycle paths for exactly {TASK_IDS}")
    _validate_new_output(output_root, runtime_root)
    lock_path = validate_gpu_contract(device, os.environ.get("VOLVE_P5_GPU_LOCK"))
    stage3_evidence = validate_stage3(stage3_root)
    locked_tasks = {
        task_id: lock_task(
            task_id=task_id,
            manifest_path=manifest_paths[task_id],
            lifecycle_path=lifecycle_paths[task_id],
            stage3_evidence=stage3_evidence["tasks"][task_id],
        )
        for task_id in TASK_IDS
    }
    output_root = Path(output_root)
    runtime_root = Path(runtime_root)
    output_root.mkdir(parents=True, exist_ok=False)
    runtime_root.mkdir(parents=True, exist_ok=False)
    global_config = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "root_seed": ROOT_SEED,
        "budget": asdict(active_budget),
        "stage3_summary_sha256": stage3_evidence["summary_sha256"],
        "source_lock_sha256": stage3_evidence["source_lock_sha256"],
        "tasks_are_independent": True,
        "cross_task_ranking_forbidden": True,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "p4_lifecycle_mutation_forbidden": True,
        "paths": "data/manifests/lifecycles supplied at runtime and not serialized",
        "tasks": {
            task_id: _configuration(locked_tasks[task_id], active_budget)
            for task_id in TASK_IDS
        },
    }
    atomic_write_json(output_root / "p5_stage4_config.json", global_config)
    refit_evidence: dict[str, dict[str, Any]] = {}
    metrics_by_task: dict[str, dict[str, Any]] = {}
    prediction_manifests: dict[str, dict[str, Any]] = {}
    lifecycle_before = {
        task_id: hash_file(lifecycle_paths[task_id]) for task_id in TASK_IDS
    }
    run_started = time.perf_counter()
    with stage2.GpuFlock(lock_path) as gpu_lock:
        for task_id in TASK_IDS:
            prepared = prepare_refit(locked_tasks[task_id], processed_root, active_budget)
            evidence = run_refit(
                prepared=prepared,
                budget=active_budget,
                runtime_root=runtime_root,
                device=device,
                gpu_lock_wait_seconds=gpu_lock.wait_seconds,
            )
            task_root = output_root / task_id
            task_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(task_root / "refit_evidence.json", evidence)
            refit_evidence[task_id] = evidence
            print(json.dumps({"task_id": task_id, "stage": "refit", "status": "completed"}), flush=True)
            del prepared
            gc.collect()
        _write_access_started(output_root, locked_tasks, refit_evidence)
        for task_id in TASK_IDS:
            task_root = output_root / task_id
            metrics, prediction_manifest = consume_known_holdout(
                locked=locked_tasks[task_id],
                refit_evidence=refit_evidence[task_id],
                processed_root=processed_root,
                output_root=output_root,
                runtime_root=runtime_root,
                device=device,
                batch_size=active_budget.batch_size,
            )
            metrics_path = atomic_write_json(task_root / "metrics.json", metrics)
            prediction_path = runtime_root / prediction_manifest["runtime_relative_path"]
            atomic_write_json(task_root / "prediction_manifest.json", prediction_manifest)
            render_task_figure(
                task_id=task_id,
                prediction_path=prediction_path,
                metrics_path=metrics_path,
                output_path=task_root / "known_holdout_diagnostics.png",
                manifest_path=task_root / "visualization_manifest.json",
            )
            metrics_by_task[task_id] = metrics
            prediction_manifests[task_id] = prediction_manifest
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "stage": "known_holdout_confirmation",
                        "accuracy": metrics["accuracy"],
                        "miou": metrics["miou"],
                        "macro_f1": metrics["macro_f1"],
                    }
                ),
                flush=True,
            )
    for task_id in TASK_IDS:
        if hash_file(lifecycle_paths[task_id]) != lifecycle_before[task_id]:
            raise RuntimeError(f"{task_id} P4 TEST_CONSUMED lifecycle was mutated")
        _validate_prior_lifecycle(
            task_id, lifecycle_paths[task_id], locked_tasks[task_id].manifest_stable_hash
        )
    state = _read_json(output_root / "stage4_state.json")
    state["state"] = "CONFIRMATION_COMPLETE"
    state["completed_at"] = _utc_now()
    for task_id in TASK_IDS:
        state["tasks"][task_id]["confirmation_completed"] = True
        state["tasks"][task_id]["metrics_sha256"] = hash_file(
            output_root / task_id / "metrics.json"
        )
        state["tasks"][task_id]["prediction_sha256"] = prediction_manifests[task_id]["sha256"]
    atomic_write_json(output_root / "stage4_state.json", state)
    total_seconds = time.perf_counter() - run_started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "status": "confirmation_complete",
        "root_seed": ROOT_SEED,
        "stage3_summary_sha256": stage3_evidence["summary_sha256"],
        "budget": asdict(active_budget),
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "scientific_claim_boundary": (
            "reusable previously seen spatial holdout; not fresh blind, external validation, "
            "or hidden competition test evidence"
        ),
        "tasks_are_independent": True,
        "cross_task_ranking_forbidden": True,
        "p4_test_consumed_preserved": True,
        "total_wall_seconds": total_seconds,
        "environment": stage2._environment(device),
        "gpu_contract": {
            "device": "cuda:0",
            "lock_name": EXPECTED_GPU_LOCK.name,
            "mechanism": "fcntl.flock(LOCK_EX)",
        },
        "tasks": {
            task_id: {
                "winner_model_id": locked_tasks[task_id].winner_model_id,
                "label_version": locked_tasks[task_id].label_version,
                "num_classes": locked_tasks[task_id].num_classes,
                "leaderboard_sha256": locked_tasks[task_id].leaderboard_sha256,
                "manifest_stable_hash": locked_tasks[task_id].manifest_stable_hash,
                "prior_lifecycle_sha256": locked_tasks[task_id].prior_lifecycle_sha256,
                "development_sample_count": len(locked_tasks[task_id].development_sample_ids),
                "development_group_count": len(locked_tasks[task_id].development_groups),
                "test_sample_count": metrics_by_task[task_id]["test_sample_count"],
                "test_group_count": metrics_by_task[task_id]["test_group_count"],
                "evaluated_pixels": metrics_by_task[task_id]["evaluated_pixels"],
                "accuracy": metrics_by_task[task_id]["accuracy"],
                "miou": metrics_by_task[task_id]["miou"],
                "macro_f1": metrics_by_task[task_id]["macro_f1"],
                "ece": metrics_by_task[task_id]["ece"],
                "refit_wall_seconds": refit_evidence[task_id]["resources"]["refit_wall_seconds"],
                "inference_wall_seconds": metrics_by_task[task_id]["inference_wall_seconds"],
                "metrics_path": f"{task_id}/metrics.json",
                "refit_evidence_path": f"{task_id}/refit_evidence.json",
                "prediction_manifest_path": f"{task_id}/prediction_manifest.json",
                "visualization_manifest_path": f"{task_id}/visualization_manifest.json",
                "figure_path": f"{task_id}/known_holdout_diagnostics.png",
            }
            for task_id in TASK_IDS
        },
    }
    summary_path = atomic_write_json(output_root / "p5_stage4_summary.json", summary)
    portable_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "p5_stage4_artifact_manifest.json"
    )
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "portable_artifacts": [
            _relative_artifact(path, output_root) for path in portable_paths
        ],
        "runtime_artifacts": {
            task_id: {
                "checkpoint": refit_evidence[task_id]["checkpoint"],
                "predictions": prediction_manifests[task_id],
            }
            for task_id in TASK_IDS
        },
        "summary_sha256": hash_file(summary_path),
        "dense_arrays_committed": False,
        "p4_artifacts_copied_or_modified": False,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
    }
    atomic_write_json(output_root / "p5_stage4_artifact_manifest.json", artifact_manifest)
    verification = verify_portable_artifacts(output_root, runtime_root)
    return {**summary, "artifact_verification": verification}


def resume_incomplete_confirmation(
    *,
    processed_root: Path,
    manifest_paths: Mapping[str, Path],
    lifecycle_paths: Mapping[str, Path],
    stage3_root: Path,
    output_root: Path,
    runtime_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Finish an authorized but incomplete inference session without refitting.

    This path cannot create or reset state.  It accepts only the exact
    ``TEST_ACCESS_STARTED`` bindings emitted by :func:`run_stage4`, verifies
    the existing refit checkpoints, and records the recovery conservatively.
    It exists so an infrastructure/setup failure after authorization is not
    hidden by deleting the single-use evidence.
    """
    active_budget = Stage4Budget()
    output_root = Path(output_root)
    runtime_root = Path(runtime_root)
    state_path = output_root / "stage4_state.json"
    state = _read_json(state_path)
    if state.get("state") != "TEST_ACCESS_STARTED" or not state.get("single_use"):
        raise RuntimeError("recovery requires the exact incomplete TEST_ACCESS_STARTED state")
    if (output_root / "p5_stage4_summary.json").exists():
        raise RuntimeError("completed Stage-4 confirmation cannot be recovered/replayed")
    lock_path = validate_gpu_contract(device, os.environ.get("VOLVE_P5_GPU_LOCK"))
    stage3_evidence = validate_stage3(stage3_root)
    locked_tasks = {
        task_id: lock_task(
            task_id=task_id,
            manifest_path=manifest_paths[task_id],
            lifecycle_path=lifecycle_paths[task_id],
            stage3_evidence=stage3_evidence["tasks"][task_id],
        )
        for task_id in TASK_IDS
    }
    config = _read_json(output_root / "p5_stage4_config.json")
    if config.get("budget") != asdict(active_budget):
        raise ValueError("recovery budget differs from frozen Stage-4 config")
    refit_evidence: dict[str, dict[str, Any]] = {}
    for task_id in TASK_IDS:
        evidence = _read_json(output_root / task_id / "refit_evidence.json")
        expected_config = _configuration(locked_tasks[task_id], active_budget)
        if evidence.get("configuration") != expected_config:
            raise ValueError(f"{task_id} recovery refit configuration changed")
        if evidence.get("configuration_hash") != hash_payload(expected_config):
            raise ValueError(f"{task_id} recovery configuration hash changed")
        checkpoint = runtime_root / evidence["checkpoint"]["runtime_relative_path"]
        if hash_file(checkpoint) != evidence["checkpoint"]["sha256"]:
            raise ValueError(f"{task_id} recovery checkpoint missing/corrupt")
        _assert_access_started(state_path, locked_tasks[task_id], evidence)
        refit_evidence[task_id] = evidence
    lifecycle_before = {
        task_id: hash_file(lifecycle_paths[task_id]) for task_id in TASK_IDS
    }
    for task_id in TASK_IDS:
        if lifecycle_before[task_id] != locked_tasks[task_id].prior_lifecycle_sha256:
            raise ValueError(f"{task_id} P4 lifecycle changed before recovery")
    recovery = {
        "count": int(state.get("recovery", {}).get("count", 0)) + 1,
        "resumed_at": _utc_now(),
        "reason": "inference_setup_failure_after_access_authorization",
        "refit_reused_without_retraining": True,
        "labels_may_have_been_read_in_incomplete_attempt": True,
        "metrics_or_predictions_from_failed_attempt_reused": False,
        "model_or_configuration_changed": False,
    }
    if recovery["count"] != 1:
        raise RuntimeError("Stage-4 permits at most one transparent recovery attempt")
    state["recovery"] = recovery
    atomic_write_json(state_path, state)

    metrics_by_task: dict[str, dict[str, Any]] = {}
    prediction_manifests: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with stage2.GpuFlock(lock_path) as gpu_lock:
        recovery["gpu_lock_wait_seconds"] = gpu_lock.wait_seconds
        for task_id in TASK_IDS:
            task_root = output_root / task_id
            metrics, prediction_manifest = consume_known_holdout(
                locked=locked_tasks[task_id],
                refit_evidence=refit_evidence[task_id],
                processed_root=processed_root,
                output_root=output_root,
                runtime_root=runtime_root,
                device=device,
                batch_size=active_budget.batch_size,
            )
            metrics_path = atomic_write_json(task_root / "metrics.json", metrics)
            prediction_path = runtime_root / prediction_manifest["runtime_relative_path"]
            atomic_write_json(task_root / "prediction_manifest.json", prediction_manifest)
            render_task_figure(
                task_id=task_id,
                prediction_path=prediction_path,
                metrics_path=metrics_path,
                output_path=task_root / "known_holdout_diagnostics.png",
                manifest_path=task_root / "visualization_manifest.json",
            )
            metrics_by_task[task_id] = metrics
            prediction_manifests[task_id] = prediction_manifest
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "stage": "known_holdout_confirmation_recovery",
                        "accuracy": metrics["accuracy"],
                        "miou": metrics["miou"],
                        "macro_f1": metrics["macro_f1"],
                    }
                ),
                flush=True,
            )
    for task_id in TASK_IDS:
        if hash_file(lifecycle_paths[task_id]) != lifecycle_before[task_id]:
            raise RuntimeError(f"{task_id} P4 TEST_CONSUMED lifecycle was mutated")
        _validate_prior_lifecycle(
            task_id, lifecycle_paths[task_id], locked_tasks[task_id].manifest_stable_hash
        )
    state = _read_json(state_path)
    state["state"] = "CONFIRMATION_COMPLETE"
    state["completed_at"] = _utc_now()
    state["recovery"]["completed_at"] = state["completed_at"]
    for task_id in TASK_IDS:
        state["tasks"][task_id]["confirmation_completed"] = True
        state["tasks"][task_id]["metrics_sha256"] = hash_file(
            output_root / task_id / "metrics.json"
        )
        state["tasks"][task_id]["prediction_sha256"] = prediction_manifests[task_id]["sha256"]
    atomic_write_json(state_path, state)
    completion_seconds = time.perf_counter() - started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "status": "confirmation_complete",
        "root_seed": ROOT_SEED,
        "stage3_summary_sha256": stage3_evidence["summary_sha256"],
        "budget": asdict(active_budget),
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "scientific_claim_boundary": (
            "reusable previously seen spatial holdout; not fresh blind, external validation, "
            "or hidden competition test evidence"
        ),
        "tasks_are_independent": True,
        "cross_task_ranking_forbidden": True,
        "p4_test_consumed_preserved": True,
        "incomplete_access_recovery": state["recovery"],
        "completion_wall_seconds": completion_seconds,
        "environment": stage2._environment(device),
        "gpu_contract": {
            "device": "cuda:0",
            "lock_name": EXPECTED_GPU_LOCK.name,
            "mechanism": "fcntl.flock(LOCK_EX)",
        },
        "tasks": {
            task_id: {
                "winner_model_id": locked_tasks[task_id].winner_model_id,
                "label_version": locked_tasks[task_id].label_version,
                "num_classes": locked_tasks[task_id].num_classes,
                "leaderboard_sha256": locked_tasks[task_id].leaderboard_sha256,
                "manifest_stable_hash": locked_tasks[task_id].manifest_stable_hash,
                "prior_lifecycle_sha256": locked_tasks[task_id].prior_lifecycle_sha256,
                "development_sample_count": len(locked_tasks[task_id].development_sample_ids),
                "development_group_count": len(locked_tasks[task_id].development_groups),
                "test_sample_count": metrics_by_task[task_id]["test_sample_count"],
                "test_group_count": metrics_by_task[task_id]["test_group_count"],
                "evaluated_pixels": metrics_by_task[task_id]["evaluated_pixels"],
                "accuracy": metrics_by_task[task_id]["accuracy"],
                "miou": metrics_by_task[task_id]["miou"],
                "macro_f1": metrics_by_task[task_id]["macro_f1"],
                "ece": metrics_by_task[task_id]["ece"],
                "refit_wall_seconds": refit_evidence[task_id]["resources"]["refit_wall_seconds"],
                "inference_wall_seconds": metrics_by_task[task_id]["inference_wall_seconds"],
                "metrics_path": f"{task_id}/metrics.json",
                "refit_evidence_path": f"{task_id}/refit_evidence.json",
                "prediction_manifest_path": f"{task_id}/prediction_manifest.json",
                "visualization_manifest_path": f"{task_id}/visualization_manifest.json",
                "figure_path": f"{task_id}/known_holdout_diagnostics.png",
            }
            for task_id in TASK_IDS
        },
    }
    summary_path = atomic_write_json(output_root / "p5_stage4_summary.json", summary)
    portable_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "p5_stage4_artifact_manifest.json"
    )
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "portable_artifacts": [_relative_artifact(path, output_root) for path in portable_paths],
        "runtime_artifacts": {
            task_id: {
                "checkpoint": refit_evidence[task_id]["checkpoint"],
                "predictions": prediction_manifests[task_id],
            }
            for task_id in TASK_IDS
        },
        "summary_sha256": hash_file(summary_path),
        "dense_arrays_committed": False,
        "p4_artifacts_copied_or_modified": False,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "incomplete_access_recovery_recorded": True,
    }
    atomic_write_json(output_root / "p5_stage4_artifact_manifest.json", artifact_manifest)
    verification = verify_portable_artifacts(output_root, runtime_root)
    return {**summary, "artifact_verification": verification}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="single-use refit and known-holdout confirmation")
    resume = subparsers.add_parser(
        "resume-incomplete",
        help="finish one exact TEST_ACCESS_STARTED session without refit/reset",
    )
    for command in (run, resume):
        command.add_argument("--processed-root", type=Path, required=True)
        command.add_argument("--f3-manifest", type=Path, required=True)
        command.add_argument("--penobscot-manifest", type=Path, required=True)
        command.add_argument("--f3-prior-lifecycle", type=Path, required=True)
        command.add_argument("--penobscot-prior-lifecycle", type=Path, required=True)
        command.add_argument("--stage3-root", type=Path, default=DEFAULT_STAGE3_ROOT)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    verify = subparsers.add_parser("verify", help="read-only hash verification of saved Stage-4 artifacts")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify":
        print(json.dumps(verify_portable_artifacts(args.output_root, args.runtime_root), indent=2))
        return
    common = {
        "processed_root": args.processed_root,
        "manifest_paths": {
            "facies_f3": args.f3_manifest,
            "facies_penobscot": args.penobscot_manifest,
        },
        "lifecycle_paths": {
            "facies_f3": args.f3_prior_lifecycle,
            "facies_penobscot": args.penobscot_prior_lifecycle,
        },
        "stage3_root": args.stage3_root,
        "output_root": args.output_root,
        "runtime_root": args.runtime_root,
        "device": torch.device("cuda:0"),
    }
    if args.command == "resume-incomplete":
        result = resume_incomplete_confirmation(**common)
    else:
        result = run_stage4(**common)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
