#!/usr/bin/env python3
"""Development-only P5 contract smoke for the ten frozen facies candidates.

This runner has no frozen-test argument or code path.  It builds a fresh model,
head and checkpoint for each of the two incompatible TaskSpecs.  Models that
cannot satisfy the source, license, dependency or fixed-I/O contract emit a
structured ``skipped`` result instead of silently changing implementation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.seeding import derive_seed, seed_everything  # noqa: E402
from _models.facies._p5_common import P5AdapterSkip, source_lock  # noqa: E402

from p4_data import (  # noqa: E402
    FaciesArchive,
    select_records_with_all_classes,
)
from p4_losses import build_loss  # noqa: E402
from p4_tasks import (  # noqa: E402
    EXPECTED_OUTER_SPLITS,
    INTERNAL_BUFFER_GROUPS,
    TASK_IDS,
    get_task_spec,
)


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
ROOT_SEED = 2693
RESULT_SCHEMA = "facies-p5-stage1-v1"


class DevelopmentOnlyArchive(FaciesArchive):
    """Fail closed if any Stage-1 code tries to resolve a non-train archive."""

    def split_path(self, split: str) -> Path:
        if split != "train":
            raise RuntimeError(
                "P5 Stage-1 is development-only; resolving frozen-test paths is forbidden"
            )
        return super().split_path(split)


@dataclass(frozen=True)
class PreparedDevelopmentBatch:
    task_id: str
    images: np.ndarray
    labels: np.ndarray
    target_mask: np.ndarray
    class_weights: tuple[float, ...]
    sample_ids: tuple[str, ...]
    inline_groups: tuple[str, ...]
    fold_train_inline_range: tuple[int, int]
    guard_inline_range: tuple[int, int]
    validation_inline_range: tuple[int, int]
    preprocessor_hash: str
    development_batch_hash: str


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _environment(device: torch.device) -> dict[str, Any]:
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "download_bytes": 0,
        "offline_weight_guards": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    if device.type == "cuda":
        environment["gpu_name"] = torch.cuda.get_device_name(device)
        environment["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return environment


def prepare_development_batch(
    task_id: str,
    processed_root: Path,
    *,
    batch_size: int = 2,
    max_candidates: int = 64,
    max_fit_records: int = 16,
) -> PreparedDevelopmentBatch:
    """Fit preprocessing and materialize one real batch from train.h5 only."""
    spec = get_task_spec(task_id)
    classes = int(spec.metadata["num_classes"])
    archive = DevelopmentOnlyArchive(task_id, processed_root)
    candidates = archive.sampled_development_index(max_candidates=max_candidates)
    development_start, development_end = EXPECTED_OUTER_SPLITS[task_id][
        "development_inline_range"
    ]
    development_count = development_end - development_start + 1
    validation_count = max(1, math.ceil(development_count * 0.20))
    validation_start = development_end - validation_count + 1
    guard_end = validation_start - 1
    guard_start = guard_end - INTERNAL_BUFFER_GROUPS[task_id] + 1
    train_end = guard_start - 1
    if train_end < development_start:
        raise ValueError("Stage-1 development fold leaves no fold-train inline region")
    fold_train_candidates = tuple(
        record
        for record in candidates
        if development_start <= record.inline <= train_end
    )
    if not fold_train_candidates:
        raise ValueError("sampled development records contain no Stage-1 fold-train candidates")
    fit_records = select_records_with_all_classes(
        fold_train_candidates,
        num_classes=classes,
        max_records=max_fit_records,
    )
    preprocessor = archive.fit_preprocessor(fit_records, method="zscore")
    batch = next(
        archive.iter_model_batches(
            fit_records,
            preprocessor,
            batch_size=batch_size,
            shuffle=False,
            seed=ROOT_SEED,
            include_targets=True,
        )
    )
    if batch.targets is None:
        raise ValueError("real development batch unexpectedly lacks facies targets")
    images = np.asarray(batch.inputs["seismic"], dtype=np.float32)
    labels = np.asarray(batch.targets["facies"], dtype=np.int64)
    target_mask = np.asarray(batch.target_masks["facies"], dtype=bool)
    if images.shape != (len(batch.sample_ids), 1, *labels.shape[-2:]):
        raise ValueError(f"unaligned real development shapes: {images.shape}, {labels.shape}")
    if not target_mask.all() or target_mask.shape != labels.shape:
        raise ValueError("facies Stage-1 requires an all-valid target mask aligned to labels")
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError("real development labels violate the independent TaskSpec head")
    if not np.isfinite(images).all():
        raise ValueError("real development normalization produced NaN/Inf")
    preprocessor_hash = hash_payload(preprocessor.to_dict())
    batch_hash = hash_payload(
        {
            "scope": "development_train_h5_only",
            "task_id": task_id,
            "label_version": spec.label_version,
            "sample_ids": list(batch.sample_ids),
            "inline_groups": list(batch.groups["inline"]),
            "fold_train_inline_range": [development_start, train_end],
            "guard_inline_range": [guard_start, guard_end],
            "validation_inline_range": [validation_start, development_end],
            "preprocessor_hash": preprocessor_hash,
        }
    )
    return PreparedDevelopmentBatch(
        task_id=task_id,
        images=images,
        labels=labels,
        target_mask=target_mask,
        class_weights=tuple(preprocessor.class_weights),
        sample_ids=tuple(batch.sample_ids),
        inline_groups=tuple(batch.groups["inline"]),
        fold_train_inline_range=(development_start, train_end),
        guard_inline_range=(guard_start, guard_end),
        validation_inline_range=(validation_start, development_end),
        preprocessor_hash=preprocessor_hash,
        development_batch_hash=batch_hash,
    )


def _assert_2d_logits(logits: torch.Tensor, inputs: torch.Tensor, classes: int) -> None:
    expected = (inputs.shape[0], classes, inputs.shape[2], inputs.shape[3])
    if tuple(logits.shape) != expected:
        raise ValueError(f"expected raw logits {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("model produced NaN/Inf raw logits")


def _parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _checkpoint_state(loss_value: float) -> dict[str, Any]:
    return {
        "next_epoch": 1,
        "global_step": 1,
        "best_epoch": 0,
        "best_val_loss": loss_value,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "history": [
            {
                "stage": "p5_contract_smoke",
                "development_loss": loss_value,
                "test_access": False,
            }
        ],
    }


def _skip_payload(
    task_id: str,
    model_id: str,
    skip: P5AdapterSkip,
    *,
    device: torch.device,
) -> dict[str, Any]:
    spec = get_task_spec(task_id)
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "skipped",
        "track_id": "facies",
        "task_id": task_id,
        "label_version": spec.label_version,
        "model_id": model_id,
        "lane": "scratch",
        "head_num_classes": int(spec.metadata["num_classes"]),
        "skip": skip.to_dict(),
        "source_lock": dict(source_lock(model_id)),
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "environment": _environment(device),
    }


def run_task_model(
    *,
    task_id: str,
    model_id: str,
    batch: PreparedDevelopmentBatch,
    output_root: Path,
    device: torch.device,
    root_seed: int = ROOT_SEED,
) -> dict[str, Any]:
    """Run one real development contract smoke and archive its own result."""
    if batch.task_id != task_id:
        raise ValueError("prepared batch cannot cross facies tasks")
    spec = get_task_spec(task_id)
    classes = int(spec.metadata["num_classes"])
    result_dir = output_root / task_id / model_id / "scratch"
    result_path = result_dir / "stage1.json"
    model_seed = derive_seed(root_seed, "model", task_id, model_id, "scratch")
    started = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model: nn.Module | None = None
    try:
        discovered = discover_model("facies", model_id)
        if bool(discovered.capabilities.get("requires_contiguous_3d_blocks")):
            raise P5AdapterSkip(
                "contiguous_3d_development_blocks_unavailable",
                "the P4 archive exposes independent 2-D patches, not verified contiguous same-core "
                "3-D blocks; constructing pseudo-volumes would violate the spatial leakage contract",
                current_model_batch="B,1,H,W",
                required_model_batch="B,1,D,H,W",
            )

        seed_report = seed_everything(model_seed, strict=False).to_dict()
        model = discovered.build(spec, num_classes=classes, lane="scratch")
        if not isinstance(model, nn.Module):
            raise TypeError("discovered facies adapter did not return torch.nn.Module")
        model = model.to(device)
        total_parameters, trainable_parameters = _parameter_count(model)

        synthetic = torch.linspace(
            -1.0,
            1.0,
            steps=64 * 96,
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, 64, 96)
        model.eval()
        with torch.no_grad():
            initial_synthetic = model(synthetic).detach().cpu()
        _assert_2d_logits(initial_synthetic, synthetic.cpu(), classes)

        images = torch.as_tensor(batch.images, dtype=torch.float32, device=device)
        labels = torch.as_tensor(batch.labels, dtype=torch.long, device=device)
        target_mask = torch.as_tensor(batch.target_mask, dtype=torch.bool, device=device)
        if not target_mask.all():
            raise ValueError("target mask changed after device transfer")
        criterion = build_loss(
            "cross_entropy",
            num_classes=classes,
            class_weights=torch.tensor(batch.class_weights, dtype=torch.float32, device=device),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        _assert_2d_logits(logits, images, classes)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("development loss is NaN/Inf")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("backward produced missing or non-finite gradients")
        optimizer.step()
        loss_value = float(loss.detach())
        del logits, loss, gradients
        model.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            expected_after_step = model(images).detach().cpu()
        _assert_2d_logits(expected_after_step, images.cpu(), classes)

        checkpoint_path = result_dir / "checkpoint.pkl"
        configuration = {
            "track_id": "facies",
            "task_id": task_id,
            "label_version": spec.label_version,
            "model_id": model_id,
            "lane": "scratch",
            "num_classes": classes,
            "root_seed": root_seed,
            "model_seed": model_seed,
            "loss": "cross_entropy_raw_logits",
        }
        save_checkpoint(
            checkpoint_path,
            epoch=0,
            model_state=_cpu_state_dict(model),
            optimizer_state={"kind": "one_step_stage1_not_resume_training"},
            scheduler_state=None,
            scaler_state=None,
            config_hash=hash_payload(configuration),
            split_hash=batch.development_batch_hash,
            trainer_state=_checkpoint_state(loss_value),
            seed_report=seed_report,
            environment=_environment(device),
            extra={
                "stage": "p5_contract_smoke",
                "source_lock": dict(source_lock(model_id)),
                "preprocessor_hash": batch.preprocessor_hash,
                "test_access": False,
            },
        )
        checkpoint = load_checkpoint(checkpoint_path)
        restored = discovered.build(spec, num_classes=classes, lane="scratch").to(device)
        restored.load_state_dict(checkpoint["model_state"])
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(images).detach().cpu()
        checkpoint_difference = float(
            torch.max(torch.abs(restored_prediction - expected_after_step))
        )
        if checkpoint_difference > 1e-6:
            raise ValueError(
                f"checkpoint round-trip changed prediction by {checkpoint_difference}"
            )
        del restored, restored_prediction, checkpoint

        seed_everything(model_seed, strict=False)
        repeated = discovered.build(spec, num_classes=classes, lane="scratch").to(device)
        repeated.eval()
        with torch.no_grad():
            repeated_synthetic = repeated(synthetic).detach().cpu()
        deterministic_difference = float(
            torch.max(torch.abs(repeated_synthetic - initial_synthetic))
        )
        if deterministic_difference > 1e-6:
            raise ValueError(
                f"same-seed model build changed synthetic prediction by {deterministic_difference}"
            )
        del repeated, repeated_synthetic

        wall_seconds = time.perf_counter() - started
        peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_vram = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "contract_smoked",
            "track_id": "facies",
            "task_id": task_id,
            "label_version": spec.label_version,
            "model_id": model_id,
            "lane": "scratch",
            "head_num_classes": classes,
            "source_lock": dict(source_lock(model_id)),
            "capabilities": dict(discovered.capabilities),
            "synthetic": {
                "input_shape": list(synthetic.shape),
                "output_shape": list(initial_synthetic.shape),
                "finite_raw_logits": True,
            },
            "real_development": {
                "input_shape": list(images.shape),
                "target_shape": list(labels.shape),
                "output_shape": list(expected_after_step.shape),
                "sample_ids": list(batch.sample_ids),
                "inline_groups": list(batch.inline_groups),
                "fold_train_inline_range": list(batch.fold_train_inline_range),
                "guard_inline_range": list(batch.guard_inline_range),
                "validation_inline_range": list(batch.validation_inline_range),
                "target_mask_all_valid": True,
                "loss": loss_value,
                "loss_contract": "weighted_cross_entropy_on_raw_logits",
                "backward_finite": True,
                "optimizer_step": "AdamW(lr=1e-4,weight_decay=0)",
                "preprocessor_fit_scope": "bounded_stage1_fold_train_only",
                "preprocessor_hash": batch.preprocessor_hash,
                "development_batch_hash": batch.development_batch_hash,
            },
            "checkpoint": {
                "path": checkpoint_path.relative_to(output_root).as_posix(),
                "sha256": hash_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "prediction_max_abs_difference": checkpoint_difference,
            },
            "determinism": {
                "model_seed": model_seed,
                "same_seed_synthetic_max_abs_difference": deterministic_difference,
                "tolerance": 1e-6,
            },
            "resources": {
                "parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "wall_seconds": wall_seconds,
                "process_max_rss_kib_before": int(peak_before),
                "process_max_rss_kib_after": int(peak_after),
                "cuda_peak_allocated_bytes": peak_vram,
                "download_bytes": 0,
            },
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
            "environment": _environment(device),
        }
    except P5AdapterSkip as skip:
        result = _skip_payload(task_id, model_id, skip, device=device)
    except Exception as exc:
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "failed",
            "track_id": "facies",
            "task_id": task_id,
            "label_version": spec.label_version,
            "model_id": model_id,
            "lane": "scratch",
            "head_num_classes": classes,
            "failure": {
                "type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            },
            "source_lock": dict(source_lock(model_id)),
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
            "environment": _environment(device),
        }
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_write_json(result_path, result)
    return result


def run_stage1(
    *,
    tasks: Sequence[str],
    models: Sequence[str],
    processed_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    unknown_tasks = sorted(set(tasks) - set(TASK_IDS))
    unknown_models = sorted(set(models) - set(MODEL_IDS))
    if unknown_tasks or unknown_models:
        raise ValueError(f"unknown tasks={unknown_tasks}, models={unknown_models}")
    per_task: dict[str, Any] = {}
    for task_id in tasks:
        try:
            batch = prepare_development_batch(task_id, processed_root)
            preparation_error: P5AdapterSkip | None = None
        except Exception as exc:
            batch = None
            preparation_error = P5AdapterSkip(
                "development_data_unavailable",
                f"real development batch preparation failed: {type(exc).__name__}: {exc}",
                processed_root=str(processed_root),
            )
        task_results: dict[str, Any] = {}
        for model_id in models:
            if preparation_error is not None:
                result = _skip_payload(task_id, model_id, preparation_error, device=device)
                atomic_write_json(
                    output_root / task_id / model_id / "scratch" / "stage1.json",
                    result,
                )
            else:
                assert batch is not None
                result = run_task_model(
                    task_id=task_id,
                    model_id=model_id,
                    batch=batch,
                    output_root=output_root,
                    device=device,
                )
            task_results[model_id] = {
                "status": result["status"],
                "result": (Path(task_id) / model_id / "scratch" / "stage1.json").as_posix(),
                "skip_code": result.get("skip", {}).get("code"),
                "failure_type": result.get("failure", {}).get("type"),
            }
        spec = get_task_spec(task_id)
        per_task[task_id] = {
            "label_version": spec.label_version,
            "head_num_classes": int(spec.metadata["num_classes"]),
            "results": task_results,
        }
    summary = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "facies",
        "lane": "scratch",
        "root_seed": ROOT_SEED,
        "tasks_are_independent": True,
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "tasks": per_task,
        "environment": _environment(device),
    }
    atomic_write_json(output_root / "summary.json", summary)
    manifest = ArtifactManifest(
        run_id=f"facies-p5-stage1-scratch-{hash_payload({'tasks': list(tasks), 'models': list(models)})[:12]}",
        root=output_root,
    )
    manifest.register("summary.json", role="stage1_summary")
    for task_id in tasks:
        for model_id in models:
            relative_root = Path(task_id) / model_id / "scratch"
            manifest.register(
                (relative_root / "stage1.json").as_posix(),
                role="stage1_result",
                metadata={"task_id": task_id, "model_id": model_id, "lane": "scratch"},
            )
            checkpoint_path = output_root / relative_root / "checkpoint.pkl"
            if checkpoint_path.is_file():
                manifest.register(
                    (relative_root / "checkpoint.pkl").as_posix(),
                    role="stage1_checkpoint",
                    metadata={"task_id": task_id, "model_id": model_id, "lane": "scratch"},
                )
    manifest.write()
    manifest.verify()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TRACK_DIR / "_outputs" / "p5_stage1",
    )
    parser.add_argument("--task", action="append", choices=TASK_IDS)
    parser.add_argument("--model", action="append", choices=MODEL_IDS)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage1(
        tasks=tuple(args.task or TASK_IDS),
        models=tuple(args.model or MODEL_IDS),
        processed_root=args.processed_root,
        output_root=args.output_root,
        device=resolve_device(args.device),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
