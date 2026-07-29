#!/usr/bin/env python3
"""P12 diagnostic repair of the facies P11 SAM2 residual harness.

This experiment preserves the P11 main branch, gate, residual bound, fixed
development folds, and five-way comparison. It changes only two diagnosed
SAM2-path choices:

1. the native 128x128 facies slice is normalized in place instead of being
   interpolated to 1024x1024; and
2. the last two Hiera image-encoder blocks are trained online with a
   conservative learning rate instead of caching a fully frozen encoder.

No frozen holdout or test archive is addressable by this CLI.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p11_residual_fusion as p11  # noqa: E402


OUTPUT_ROOT = HERE / "_outputs" / "p12_repair_v1"
P11_SUMMARY_PATH = (
    HERE
    / "_outputs"
    / "p11_residual_fusion"
    / "p11_residual_fusion_summary.json"
)
P11_ARTIFACT_MANIFEST = (
    HERE / "_outputs" / "p11_residual_fusion" / "artifact_manifest.csv"
)
SCHEMA_VERSION = "facies-p12-repair-v1/v1"
ROOT_SEED = p11.ROOT_SEED
FOLDS = p11.FOLDS
MAX_UPDATES = p11.MAX_UPDATES
BATCH_SIZE = p11.BATCH_SIZE
HEAD_LR = p11.LR
ENCODER_LR = 1e-5
WEIGHT_DECAY = p11.WEIGHT_DECAY
ENCODER_GRAD_CLIP = 1.0
TRAINABLE_HIERA_BLOCKS = 2
NATIVE_INPUT_SIZE = (128, 128)
MIN_PROMOTION_DELTA = p11.MIN_PROMOTION_DELTA
MAX_RESIDUAL_CORRECTION = p11.MAX_RESIDUAL_CORRECTION
VARIANTS = p11.VARIANTS
TASK_MANIFEST_HASHES = p11.TASK_MANIFEST_HASHES


def _validate_output_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(
            f"P12 output must stay inside the facies track: {resolved}"
        ) from exc
    if resolved == p11.OUTPUT_ROOT.resolve():
        raise ValueError("P12 refuses to overwrite the committed P11 evidence")
    return resolved


def _sam2_normalize_native(inputs: torch.Tensor) -> torch.Tensor:
    """Normalize a native facies slice without spatial interpolation."""
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError(f"SAM2 expects [B,1,H,W], got {tuple(inputs.shape)}")
    spatial = tuple(int(value) for value in inputs.shape[-2:])
    if spatial != NATIVE_INPUT_SIZE:
        raise ValueError(
            "P12 native SAM2 adaptation is locked to 128x128 development "
            f"slices, got {spatial}"
        )
    normalized = torch.clamp(inputs, -5.0, 5.0).add(5.0).div(10.0)
    normalized = normalized.repeat(1, 3, 1, 1)
    mean = normalized.new_tensor((0.485, 0.456, 0.406))[
        None, :, None, None
    ]
    std = normalized.new_tensor((0.229, 0.224, 0.225))[
        None, :, None, None
    ]
    return (normalized - mean) / std


def _set_last_hiera_blocks_trainable(
    image_encoder: torch.nn.Module,
    *,
    block_count: int = TRAINABLE_HIERA_BLOCKS,
) -> tuple[int, ...]:
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False
    trunk = image_encoder.trunk
    blocks = trunk.blocks
    if block_count < 1 or block_count > len(blocks):
        raise ValueError(
            f"invalid trainable Hiera block count {block_count} for "
            f"{len(blocks)} blocks"
        )
    start = len(blocks) - block_count
    indices = tuple(range(start, len(blocks)))
    for index in indices:
        for parameter in blocks[index].parameters():
            parameter.requires_grad = True
    return indices


class TrainableSam2Encoder(torch.nn.Module):
    """SAM2 encoder with only the final Hiera blocks trainable."""

    def __init__(
        self,
        *,
        task_id: str,
        num_classes: int,
        device: str,
        randomize: bool = False,
        random_seed: int = ROOT_SEED + 17,
    ) -> None:
        super().__init__()
        p11._prepare_sam2_dependency_path()
        spec = p11._task_spec(task_id)
        model = p11.build_sam2_direct(
            spec,
            source_root=p11.SAM2_SOURCE_ROOT,
            checkpoint_path=p11.SAM2_CHECKPOINT,
            num_classes=num_classes,
            device=device,
            freeze_encoder=True,
        )
        self.image_encoder = model.backbone.image_encoder
        if randomize:
            generator = torch.Generator(device="cpu").manual_seed(random_seed)
            with torch.no_grad():
                for name, parameter in self.image_encoder.named_parameters():
                    values = torch.empty(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device="cpu",
                    )
                    if parameter.ndim >= 2:
                        values.normal_(0.0, 0.02, generator=generator)
                    elif "norm" in name.lower():
                        values.fill_(1.0)
                    else:
                        values.zero_()
                    parameter.copy_(values.to(parameter.device))
        self.trainable_block_indices = _set_last_hiera_blocks_trainable(
            self.image_encoder
        )

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for parameter in self.image_encoder.parameters()
            if parameter.requires_grad
        ]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.trainable_parameters()
        )

    @property
    def total_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.image_encoder.parameters()
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        encoded = self.image_encoder(_sam2_normalize_native(inputs))
        maps = list(encoded["backbone_fpn"])[-3:]
        if len(maps) != 3 or any(
            feature.ndim != 4 or feature.shape[1] != 256
            for feature in maps
        ):
            raise ValueError(
                "unexpected native-resolution SAM2 feature pyramid: "
                + repr([tuple(feature.shape) for feature in maps])
            )
        return maps


def _make_sam2_encoder(
    task_id: str,
    num_classes: int,
    device: str,
    *,
    randomize: bool = False,
    random_seed: int = ROOT_SEED + 17,
) -> TrainableSam2Encoder:
    return TrainableSam2Encoder(
        task_id=task_id,
        num_classes=num_classes,
        device=device,
        randomize=randomize,
        random_seed=random_seed,
    )


def _snapshot_trainable(
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor]:
    return [
        parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for parameter in parameters
    ]


def _parameter_delta_l2(
    parameters: Sequence[torch.nn.Parameter],
    snapshots: Sequence[torch.Tensor],
) -> float:
    squared = 0.0
    for parameter, snapshot in zip(parameters, snapshots):
        current = parameter.detach().to(device="cpu", dtype=torch.float32)
        squared += float((current - snapshot).square().sum())
    return math.sqrt(squared)


def _optimizer(
    head_parameters: Iterable[torch.nn.Parameter],
    encoder_parameters: Sequence[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": list(head_parameters), "lr": HEAD_LR},
            {"params": list(encoder_parameters), "lr": ENCODER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _encoder_metadata(
    encoder: TrainableSam2Encoder,
    *,
    update_l2: float,
    last_grad_norm: float,
    feature_shapes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    return {
        "encoder_trainable_blocks": list(
            encoder.trainable_block_indices
        ),
        "encoder_trainable_parameters": (
            encoder.trainable_parameter_count
        ),
        "encoder_total_parameters": encoder.total_parameter_count,
        "encoder_update_l2": float(update_l2),
        "encoder_last_grad_norm": float(last_grad_norm),
        "sam2_input_shape": [3, *NATIVE_INPUT_SIZE],
        "feature_shapes": [list(shape) for shape in feature_shapes],
    }


def _predict_direct_online(
    encoder: TrainableSam2Encoder,
    head: p11.DirectSam2Head,
    images: np.ndarray,
    *,
    output_size: tuple[int, int],
    device: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.as_tensor(
                images[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            logits = head(encoder(batch), output_size=output_size)
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _train_direct_online(
    prepared: Any,
    *,
    task_id: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    p11._seed_all(seed)
    encoder = _make_sam2_encoder(
        task_id,
        prepared.num_classes,
        device,
        randomize=False,
    ).to(device)
    head = p11.DirectSam2Head(prepared.num_classes).to(device)
    encoder_parameters = encoder.trainable_parameters()
    snapshots = _snapshot_trainable(encoder_parameters)
    optimizer = _optimizer(head.parameters(), encoder_parameters)
    weights = torch.as_tensor(
        prepared.class_weights,
        dtype=torch.float32,
        device=device,
    )
    output_size = tuple(
        int(value) for value in prepared.train_labels.shape[-2:]
    )
    history: list[float] = []
    last_grad_norm = 0.0
    feature_shapes: list[list[int]] = []
    encoder.train()
    head.train()
    for update_index, indices in enumerate(
        p11._make_index_batches(len(prepared.train_labels), seed=seed)
    ):
        images = torch.as_tensor(
            prepared.train_images[indices],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            prepared.train_labels[indices],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        maps = encoder(images)
        if not feature_shapes:
            feature_shapes = [
                list(feature.shape[1:]) for feature in maps
            ]
        logits = head(maps, output_size=output_size)
        loss = F.cross_entropy(logits, labels, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{task_id}/direct_sam2 loss became non-finite"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder_parameters,
            max_norm=ENCODER_GRAD_CLIP,
        )
        if update_index == MAX_UPDATES - 1:
            last_grad_norm = float(grad_norm.detach().cpu())
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    validation_logits = _predict_direct_online(
        encoder,
        head,
        prepared.validation_images,
        output_size=tuple(
            int(value)
            for value in prepared.validation_labels.shape[-2:]
        ),
        device=device,
    )
    metrics, confusion = p11._metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    update_l2 = _parameter_delta_l2(encoder_parameters, snapshots)
    metadata = _encoder_metadata(
        encoder,
        update_l2=update_l2,
        last_grad_norm=last_grad_norm,
        feature_shapes=feature_shapes,
    )
    encoder.cpu()
    head.cpu()
    return {
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "metrics": metrics,
        "confusion": confusion,
        **metadata,
    }


def _predict_residual_online(
    encoder: TrainableSam2Encoder,
    core: p11.ResidualFusionCore,
    images: np.ndarray,
    small_logits: torch.Tensor,
    *,
    device: str,
    gate_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    correction_parts: list[np.ndarray] = []
    encoder.eval()
    core.eval()
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(images))
            batch = torch.as_tensor(
                images[start:stop],
                dtype=torch.float32,
                device=device,
            )
            logits, gates, corrections = core(
                small_logits[start:stop].to(
                    device=device,
                    dtype=torch.float32,
                ),
                encoder(batch),
                gate_override=gate_override,
            )
            logits_parts.append(logits.detach().cpu().numpy())
            gate_parts.append(gates.detach().cpu().numpy())
            correction_parts.append(
                corrections.detach().cpu().numpy()
            )
    return (
        np.concatenate(logits_parts, axis=0),
        np.concatenate(gate_parts, axis=0),
        np.concatenate(correction_parts, axis=0),
    )


def _train_residual_online(
    prepared: Any,
    train_small_logits: torch.Tensor,
    validation_small_logits: torch.Tensor,
    *,
    task_id: str,
    device: str,
    seed: int,
    randomize: bool,
    label: str,
) -> dict[str, Any]:
    p11._seed_all(seed)
    encoder = _make_sam2_encoder(
        task_id,
        prepared.num_classes,
        device,
        randomize=randomize,
        random_seed=seed,
    ).to(device)
    core = p11.ResidualFusionCore(prepared.num_classes).to(device)
    encoder_parameters = encoder.trainable_parameters()
    snapshots = _snapshot_trainable(encoder_parameters)
    optimizer = _optimizer(core.parameters(), encoder_parameters)
    weights = torch.as_tensor(
        prepared.class_weights,
        dtype=torch.float32,
        device=device,
    )
    history: list[float] = []
    train_gates: list[float] = []
    train_corrections: list[float] = []
    last_grad_norm = 0.0
    feature_shapes: list[list[int]] = []
    encoder.train()
    core.train()
    for update_index, indices in enumerate(
        p11._make_index_batches(len(prepared.train_labels), seed=seed)
    ):
        images = torch.as_tensor(
            prepared.train_images[indices],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            prepared.train_labels[indices],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        maps = encoder(images)
        if not feature_shapes:
            feature_shapes = [
                list(feature.shape[1:]) for feature in maps
            ]
        logits, gate, correction = core(
            train_small_logits[indices].to(
                device=device,
                dtype=torch.float32,
            ),
            maps,
        )
        loss = F.cross_entropy(logits, labels, weight=weights)
        loss = (
            loss
            + p11.GATE_REG * gate.mean()
            + p11.RESIDUAL_REG * correction.abs().mean()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{label} loss became non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder_parameters,
            max_norm=ENCODER_GRAD_CLIP,
        )
        if update_index == MAX_UPDATES - 1:
            last_grad_norm = float(grad_norm.detach().cpu())
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        train_gates.extend(
            float(value) for value in gate.detach().cpu().flatten()
        )
        train_corrections.append(
            float(correction.detach().abs().mean().cpu())
        )

    validation_logits, gates, corrections = _predict_residual_online(
        encoder,
        core,
        prepared.validation_images,
        validation_small_logits,
        device=device,
    )
    metrics, confusion = p11._metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    (
        gate_zero_logits,
        gate_zero_gates,
        gate_zero_corrections,
    ) = _predict_residual_online(
        encoder,
        core,
        prepared.validation_images,
        validation_small_logits,
        device=device,
        gate_override=0.0,
    )
    gate_zero_metrics, gate_zero_confusion = p11._metrics_payload(
        gate_zero_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    small_numpy = validation_small_logits.numpy()
    gate_zero_diff = float(
        np.max(np.abs(gate_zero_logits - small_numpy))
    )
    if gate_zero_diff > 1e-7:
        raise RuntimeError(
            "gate=0 failed to reproduce the small-model logits: "
            f"{gate_zero_diff}"
        )
    if float(np.max(np.abs(gate_zero_gates))) != 0.0:
        raise RuntimeError("gate=0 produced a nonzero gate")
    if float(np.max(np.abs(gate_zero_corrections))) != 0.0:
        raise RuntimeError("gate=0 produced a nonzero correction")

    correction_abs = np.abs(corrections)
    update_l2 = _parameter_delta_l2(encoder_parameters, snapshots)
    metadata = _encoder_metadata(
        encoder,
        update_l2=update_l2,
        last_grad_norm=last_grad_norm,
        feature_shapes=feature_shapes,
    )
    payload = {
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "train_gate_mean": float(np.mean(train_gates)),
        "train_correction_mean_abs": float(
            np.mean(train_corrections)
        ),
        "metrics": metrics,
        "confusion": confusion,
        "gate_stats": p11._distribution_stats(gates),
        "residual_stats": {
            "mean_abs": float(np.mean(correction_abs)),
            "max_abs": float(np.max(correction_abs)),
            "fixed_bound": MAX_RESIDUAL_CORRECTION,
            "ratio_to_small_logits": float(
                np.mean(correction_abs)
                / max(np.mean(np.abs(small_numpy)), 1e-12)
            ),
        },
        "gate_zero": {
            "metrics": gate_zero_metrics,
            "confusion": gate_zero_confusion,
            "gate_stats": p11._distribution_stats(
                gate_zero_gates
            ),
            "residual_stats": {
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "fixed_bound": MAX_RESIDUAL_CORRECTION,
                "ratio_to_small_logits": 0.0,
            },
            "max_abs_logit_diff": gate_zero_diff,
        },
        **metadata,
    }
    encoder.cpu()
    core.cpu()
    return payload


def _fold_seed(fold_id: int) -> int:
    return ROOT_SEED + fold_id


def _run_task(
    task_id: str,
    *,
    manifest_path: Path,
    processed_root: Path,
    device: str,
    run_command: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    budget = p11.stage3.Stage3Budget()
    results: list[dict[str, Any]] = []
    stable_hash: str | None = None
    runner_sha256 = p11._sha256(Path(__file__))
    git_head = p11._git_head()
    checkpoint_sha256 = p11._sha256(p11.SAM2_CHECKPOINT)
    for fold_id in FOLDS:
        fold_started = time.perf_counter()
        prepared = p11.stage3.prepare_fold(
            task_id=task_id,
            fold_id=fold_id,
            manifest_path=manifest_path,
            processed_root=processed_root,
            budget=budget,
        )
        spatial = tuple(
            int(value) for value in prepared.train_images.shape[-2:]
        )
        if spatial != NATIVE_INPUT_SIZE:
            raise RuntimeError(
                f"{task_id} fold {fold_id} input shape drifted: "
                f"{spatial}"
            )
        if stable_hash is None:
            stable_hash = prepared.manifest_stable_hash
        elif stable_hash != prepared.manifest_stable_hash:
            raise RuntimeError("fold identity drift detected")
        if (
            prepared.manifest_stable_hash
            != TASK_MANIFEST_HASHES[task_id]
        ):
            raise RuntimeError(
                f"{task_id} manifest identity changed: "
                f"{prepared.manifest_stable_hash}"
            )
        seed = _fold_seed(fold_id)

        variant_started = time.perf_counter()
        baseline_model = p11._build_seeded_small_model(
            task_id,
            prepared.num_classes,
            device,
            seed=seed,
        )
        baseline_result = p11._train_ce_model(
            baseline_model,
            prepared,
            device=device,
            seed=seed,
            label=f"{task_id}/baseline",
        )
        baseline_seconds = time.perf_counter() - variant_started
        trained_small_model = baseline_result.pop("model")
        train_small_logits = torch.from_numpy(
            p11._predict_model(
                trained_small_model,
                prepared.train_images,
                device=device,
            )
        )
        validation_small_logits = torch.from_numpy(
            baseline_result["validation_logits"]
        )
        trained_small_model.cpu()
        del trained_small_model, baseline_model
        gc.collect()
        torch.cuda.empty_cache()

        variant_started = time.perf_counter()
        direct_sam2 = _train_direct_online(
            prepared,
            task_id=task_id,
            device=device,
            seed=seed,
        )
        direct_seconds = time.perf_counter() - variant_started
        gc.collect()
        torch.cuda.empty_cache()

        variant_started = time.perf_counter()
        pretrained_residual = _train_residual_online(
            prepared,
            train_small_logits,
            validation_small_logits,
            task_id=task_id,
            device=device,
            seed=seed,
            randomize=False,
            label=f"{task_id}/pretrained_residual",
        )
        pretrained_residual_seconds = (
            time.perf_counter() - variant_started
        )
        gate_zero = pretrained_residual.pop("gate_zero")
        gc.collect()
        torch.cuda.empty_cache()

        variant_started = time.perf_counter()
        random_residual = _train_residual_online(
            prepared,
            train_small_logits,
            validation_small_logits,
            task_id=task_id,
            device=device,
            seed=seed,
            randomize=True,
            label=f"{task_id}/random_sam2_residual",
        )
        random_residual.pop("gate_zero")
        random_residual_seconds = (
            time.perf_counter() - variant_started
        )
        gc.collect()
        torch.cuda.empty_cache()

        baseline_metric = float(
            baseline_result["metrics"]["miou"]
        )
        variant_payloads = [
            (
                "strong_small_baseline",
                baseline_result,
                "trained",
                "same-run strong small-model baseline",
                baseline_seconds,
            ),
            (
                "direct_sam2",
                direct_sam2,
                "trained",
                "native-128 SAM2 direct head with last two "
                "Hiera blocks fine-tuned",
                direct_seconds,
            ),
            (
                "pretrained_residual",
                pretrained_residual,
                "trained",
                "bounded residual over native-128 pretrained SAM2 "
                "with last two Hiera blocks fine-tuned",
                pretrained_residual_seconds,
            ),
            (
                "random_sam2_residual",
                random_residual,
                "trained",
                "same repaired residual structure with deterministic "
                "random SAM2 weights",
                random_residual_seconds,
            ),
            (
                "gate_zero",
                gate_zero,
                "control",
                "exact gate=0 degeneration of the repaired "
                "pretrained residual route",
                0.0,
            ),
        ]
        feature_shapes = direct_sam2["feature_shapes"]
        for variant, payload, status, notes, duration_seconds in (
            variant_payloads
        ):
            metrics = payload["metrics"]
            row = {
                "track": "facies",
                "dataset": (
                    "F3"
                    if task_id == "facies_f3"
                    else "Penobscot"
                ),
                "task_id": task_id,
                "fold_id": fold_id,
                "seed": seed,
                "variant": variant,
                "status": status,
                "command": run_command,
                "metric_name": "miou",
                "metric_value": float(metrics["miou"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "nll": float(metrics["nll"]),
                "brier": float(metrics["brier"]),
                "ece": float(metrics["ece"]),
                "baseline_metric": baseline_metric,
                "delta_abs": float(
                    metrics["miou"] - baseline_metric
                ),
                "gate_mean": float(
                    payload.get("gate_stats", {}).get("mean", 0.0)
                ),
                "gate_std": float(
                    payload.get("gate_stats", {}).get("std", 0.0)
                ),
                "gate_min": float(
                    payload.get("gate_stats", {}).get("min", 0.0)
                ),
                "gate_max": float(
                    payload.get("gate_stats", {}).get("max", 0.0)
                ),
                "gate_p05": float(
                    payload.get("gate_stats", {}).get("p05", 0.0)
                ),
                "gate_p50": float(
                    payload.get("gate_stats", {}).get("p50", 0.0)
                ),
                "gate_p95": float(
                    payload.get("gate_stats", {}).get("p95", 0.0)
                ),
                "residual_mean_abs": float(
                    payload.get("residual_stats", {}).get(
                        "mean_abs",
                        0.0,
                    )
                ),
                "residual_max_abs": float(
                    payload.get("residual_stats", {}).get(
                        "max_abs",
                        0.0,
                    )
                ),
                "residual_fixed_bound": MAX_RESIDUAL_CORRECTION,
                "residual_ratio_to_small_logits": float(
                    payload.get("residual_stats", {}).get(
                        "ratio_to_small_logits",
                        0.0,
                    )
                ),
                "train_loss_last": payload.get("train_loss_last"),
                "train_loss_mean": payload.get("train_loss_mean"),
                "gate_zero_max_abs_logit_diff": payload.get(
                    "max_abs_logit_diff"
                ),
                "duration_seconds": float(duration_seconds),
                "evidence_path": str(
                    (
                        output_root / "evidence.md"
                    ).relative_to(PROJECT_ROOT)
                ),
                "checkpoint_path": (
                    str(p11.SAM2_CHECKPOINT)
                    if variant != "strong_small_baseline"
                    else ""
                ),
                "checkpoint_sha256": (
                    checkpoint_sha256
                    if variant != "strong_small_baseline"
                    else ""
                ),
                "git_head_at_run": git_head,
                "runner_sha256": runner_sha256,
                "notes": notes,
                "fold_split_hash": prepared.fold_split_hash,
                "manifest_stable_hash": (
                    prepared.manifest_stable_hash
                ),
                "train_samples": len(prepared.train_images),
                "validation_samples": len(
                    prepared.validation_images
                ),
                "sam2_input_policy": (
                    "native_128_no_spatial_interpolation"
                ),
                "sam2_input_shape": payload.get(
                    "sam2_input_shape",
                    [3, *NATIVE_INPUT_SIZE],
                ),
                "feature_shapes": payload.get(
                    "feature_shapes",
                    feature_shapes,
                ),
                "encoder_trainable_blocks": payload.get(
                    "encoder_trainable_blocks",
                    [],
                ),
                "encoder_trainable_parameters": int(
                    payload.get(
                        "encoder_trainable_parameters",
                        0,
                    )
                ),
                "encoder_total_parameters": int(
                    payload.get(
                        "encoder_total_parameters",
                        0,
                    )
                ),
                "encoder_update_l2": float(
                    payload.get("encoder_update_l2", 0.0)
                ),
                "encoder_last_grad_norm": float(
                    payload.get("encoder_last_grad_norm", 0.0)
                ),
                "frozen_test_accessed": False,
            }
            results.append(row)

        del train_small_logits, validation_small_logits
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "fold_id": fold_id,
                    "baseline_miou": baseline_metric,
                    "direct_sam2_miou": float(
                        direct_sam2["metrics"]["miou"]
                    ),
                    "pretrained_residual_miou": float(
                        pretrained_residual["metrics"]["miou"]
                    ),
                    "random_sam2_residual_miou": float(
                        random_residual["metrics"]["miou"]
                    ),
                    "gate_mean": float(
                        pretrained_residual["gate_stats"]["mean"]
                    ),
                    "encoder_update_l2": float(
                        pretrained_residual["encoder_update_l2"]
                    ),
                    "fold_seconds": (
                        time.perf_counter() - fold_started
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def _variant_means(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        means[variant] = {
            "miou": float(
                np.mean([row["metric_value"] for row in subset])
            ),
            "accuracy": float(
                np.mean([row["accuracy"] for row in subset])
            ),
            "macro_f1": float(
                np.mean([row["macro_f1"] for row in subset])
            ),
            "gate_mean": float(
                np.mean([row["gate_mean"] for row in subset])
            ),
            "residual_mean_abs": float(
                np.mean(
                    [row["residual_mean_abs"] for row in subset]
                )
            ),
            "encoder_update_l2": float(
                np.mean(
                    [row["encoder_update_l2"] for row in subset]
                )
            ),
        }
    return means


def _historical_comparison(
    task_name: str,
    repaired_means: Mapping[str, Mapping[str, float]],
    p11_summary: Mapping[str, Any],
) -> dict[str, Any]:
    historical = p11_summary["tasks"][task_name]["variant_means"]
    p11_baseline = float(
        historical["strong_small_baseline"]["miou"]
    )
    p12_baseline = float(
        repaired_means["strong_small_baseline"]["miou"]
    )
    comparisons: dict[str, Any] = {
        "p11_baseline_miou": p11_baseline,
        "p12_baseline_miou": p12_baseline,
        "baseline_reproduction_delta": p12_baseline - p11_baseline,
        "variants": {},
    }
    for variant in ("direct_sam2", "pretrained_residual"):
        p11_miou = float(historical[variant]["miou"])
        p12_miou = float(repaired_means[variant]["miou"])
        p11_delta = p11_miou - p11_baseline
        p12_delta = p12_miou - p12_baseline
        comparisons["variants"][variant] = {
            "p11_miou": p11_miou,
            "p12_miou": p12_miou,
            "p12_minus_p11_same_variant": p12_miou - p11_miou,
            "p11_delta_vs_baseline": p11_delta,
            "p12_delta_vs_baseline": p12_delta,
            "delta_vs_baseline_change": p12_delta - p11_delta,
        }
    return comparisons


def _write_evidence(
    task_summaries: Mapping[str, Any],
    output_root: Path,
    *,
    run_command: str,
) -> Path:
    lines = [
        "# P12 repair-v1 evidence",
        "",
        "## Diagnostic question",
        "",
        "P11 was non-beneficial while expanding every 128×128 facies "
        "slice to 1024×1024 and freezing the complete SAM2 image "
        "encoder. P12 tests whether native-resolution input plus "
        "conservative top-block fine-tuning changes that result.",
        "",
        "## Frozen repair contract",
        "",
        "- SAM2 receives normalized `[B,3,128,128]` tensors with no "
        "spatial interpolation.",
        "- A real Hiera-B+ forward produced finite feature maps at "
        "`32×32`, `16×16`, and `8×8`; the local Hiera positional "
        "encoding and window partitioning therefore accept this shape.",
        "- Only Hiera blocks 22 and 23 are trainable; the encoder "
        f"learning rate is `{ENCODER_LR:g}` and the head/core rate "
        f"is `{HEAD_LR:g}`.",
        "- Folds, seeds, samples, 40-update budget, metrics, strong "
        "main route, residual formula, regularizers, and five variants "
        "are unchanged from P11.",
        "",
        "## Results",
        "",
        "| Task | Variant | P11 mIoU | P12 mIoU | P12 Δ vs baseline | "
        "Change in Δ vs P11 | Gate mean | Encoder update L2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_name in ("F3", "Penobscot"):
        task = task_summaries[task_name]
        repaired = task["variant_means"]
        historical = task["historical_comparison"]
        baseline = repaired["strong_small_baseline"]["miou"]
        p11_means = task["p11_variant_means"]
        for variant in VARIANTS:
            p12_miou = repaired[variant]["miou"]
            p11_miou = p11_means[variant]["miou"]
            if variant in historical["variants"]:
                change = historical["variants"][variant][
                    "delta_vs_baseline_change"
                ]
            else:
                change = (
                    p12_miou
                    - baseline
                    - (
                        p11_miou
                        - historical["p11_baseline_miou"]
                    )
                )
            lines.append(
                f"| {task_name} | {variant} | {p11_miou:.6f} | "
                f"{p12_miou:.6f} | {p12_miou - baseline:+.6f} | "
                f"{change:+.6f} | "
                f"{repaired[variant]['gate_mean']:.6f} | "
                f"{repaired[variant]['encoder_update_l2']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Cross-run comparison caveat",
            "",
            "The same seeded small-model baseline was not bitwise "
            "stable across the separately executed P11 and P12 GPU "
            "runs. P12 minus P11 baseline mIoU was "
            f"{task_summaries['F3']['historical_comparison']['baseline_reproduction_delta']:+.6f} "
            "for F3 and "
            f"{task_summaries['Penobscot']['historical_comparison']['baseline_reproduction_delta']:+.6f} "
            "for Penobscot. Therefore same-run deltas versus the P12 "
            "baseline are the primary causal comparison; P12-minus-P11 "
            "same-variant values are descriptive only.",
            "",
            "## Honest conclusion",
            "",
        ]
    )
    for task_name in ("F3", "Penobscot"):
        task = task_summaries[task_name]
        direct = task["historical_comparison"]["variants"][
            "direct_sam2"
        ]
        residual = task["historical_comparison"]["variants"][
            "pretrained_residual"
        ]
        lines.append(
            f"- {task_name}: decision **"
            f"{task['decision']['state']}**. Repaired direct SAM2 "
            f"is {direct['p12_delta_vs_baseline']:+.6f} mIoU versus "
            "the same-run baseline; repaired pretrained residual is "
            f"{residual['p12_delta_vs_baseline']:+.6f}. Their "
            "descriptive same-variant P12-minus-P11 changes are "
            f"{direct['p12_minus_p11_same_variant']:+.6f} and "
            f"{residual['p12_minus_p11_same_variant']:+.6f}; their "
            "baseline-relative deltas changed by "
            f"{direct['delta_vs_baseline_change']:+.6f} and "
            f"{residual['delta_vs_baseline_change']:+.6f}, "
            "respectively."
        )
    lines.extend(
        [
            "",
            "Promotion still requires pretrained residual to beat both "
            "the same-run baseline and the repaired random-SAM2 "
            f"control by at least {MIN_PROMOTION_DELTA:.3f} mIoU.",
            "",
            "## Encoder-update evidence",
            "",
            "Every trainable SAM2 variant records its final gradient "
            "norm and the L2 displacement of blocks 22–23 from their "
            "initial values. The verifier rejects zero-gradient or "
            "zero-update evidence.",
            "",
            "## Data boundary",
            "",
            "- Only the locked F3 and Penobscot development manifests "
            "and each task's `train.h5` were used.",
            "- Folds are exactly 0 and 4 with seed `2693 + fold_id`.",
            "- No frozen holdout, `test.h5`, dense prediction, feature "
            "cache, or checkpoint copy was read or persisted.",
            "- The committed P11 artifact manifest was hash-checked "
            "before and after this independent output was written.",
            "",
            "## Reproduction command",
            "",
            "```text",
            run_command,
            "```",
        ]
    )
    path = output_root / "evidence.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
    device: str = "cuda:0",
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    started = time.perf_counter()
    p11._validate_cuda_device(device)
    manifests = p11.validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    output_root = _validate_output_root(output_root)
    if not P11_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"committed P11 summary is missing: {P11_SUMMARY_PATH}"
        )
    p11.verify(p11.OUTPUT_ROOT)
    p11_manifest_sha = p11._sha256(P11_ARTIFACT_MANIFEST)
    p11_summary = json.loads(
        P11_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    dependency_site = p11._prepare_sam2_dependency_path()
    source_root = p11.verify_git_source(
        p11.SAM2_SOURCE_ROOT,
        p11.SAM2_SOURCE_REVISION,
    )
    checkpoint = p11.verify_checkpoint(
        "facies",
        p11.SAM2_CHECKPOINT,
    )
    p11.insert_import_root(source_root, "sam2")
    output_root.mkdir(parents=True, exist_ok=True)
    run_command = " ".join(
        [
            sys.executable,
            str(Path(__file__).relative_to(PROJECT_ROOT)),
            "run",
            "--f3-manifest",
            str(manifests["facies_f3"]),
            "--penobscot-manifest",
            str(manifests["facies_penobscot"]),
            "--processed-root",
            str(Path(processed_root).resolve()),
            "--device",
            device,
        ]
    )
    all_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    for task_id in ("facies_f3", "facies_penobscot"):
        rows = _run_task(
            task_id,
            manifest_path=manifests[task_id],
            processed_root=Path(processed_root).resolve(),
            device=device,
            run_command=run_command,
            output_root=output_root,
        )
        all_rows.extend(rows)
        task_name = (
            "F3" if task_id == "facies_f3" else "Penobscot"
        )
        means = _variant_means(rows)
        baseline = means["strong_small_baseline"]["miou"]
        pretrained = means["pretrained_residual"]["miou"]
        random_control = means["random_sam2_residual"]["miou"]
        promoted = (
            pretrained - baseline >= MIN_PROMOTION_DELTA
            and pretrained - random_control >= MIN_PROMOTION_DELTA
        )
        historical = _historical_comparison(
            task_name,
            means,
            p11_summary,
        )
        task_summaries[task_name] = {
            "task_id": task_id,
            "variant_means": means,
            "p11_variant_means": p11_summary["tasks"][
                task_name
            ]["variant_means"],
            "historical_comparison": historical,
            "comparison": {
                "pretrained_residual_minus_baseline": (
                    pretrained - baseline
                ),
                "random_sam2_residual_minus_baseline": (
                    random_control - baseline
                ),
                "pretrained_minus_random_sam2_residual": (
                    pretrained - random_control
                ),
                "gate_zero_minus_baseline": (
                    means["gate_zero"]["miou"] - baseline
                ),
            },
            "decision": {
                "state": (
                    "PROMOTED" if promoted else "NON_BENEFICIAL"
                ),
                "default_enabled": promoted,
                "minimum_promotion_delta": MIN_PROMOTION_DELTA,
                "promotion_rule": (
                    "pretrained_residual_minus_baseline>=0.005_and_"
                    "pretrained_minus_random_sam2_residual>=0.005"
                ),
            },
        }

    if p11._sha256(P11_ARTIFACT_MANIFEST) != p11_manifest_sha:
        raise RuntimeError("committed P11 evidence changed during P12")
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "parent_experiment": "p11_residual_fusion",
            "foundation_model": (
                "facebook/sam2.1-hiera-base-plus"
            ),
            "sam2_source_revision": p11.SAM2_SOURCE_REVISION,
            "sam2_checkpoint_sha256": p11._sha256(checkpoint),
            "sam2_dependency_site": str(dependency_site),
            "input_policy": (
                "native_128_no_spatial_interpolation"
            ),
            "input_shape_constraint": [3, *NATIVE_INPUT_SIZE],
            "observed_feature_shapes": [
                [256, 32, 32],
                [256, 16, 16],
                [256, 8, 8],
            ],
            "trainable_hiera_blocks": [22, 23],
            "trainable_block_count": TRAINABLE_HIERA_BLOCKS,
            "encoder_learning_rate": ENCODER_LR,
            "head_learning_rate": HEAD_LR,
            "encoder_grad_clip": ENCODER_GRAD_CLIP,
            "feature_policy": (
                "online_encoder_forward_with_top_two_blocks_trainable"
            ),
            "residual_formula": (
                "small_logits + sigmoid(gate) * 0.05 * "
                "tanh(residual_logits)"
            ),
            "residual_max_abs_logit_correction": (
                MAX_RESIDUAL_CORRECTION
            ),
            "gate_inputs": (
                "small_logits_summary_and_sam2_feature_norms_only"
            ),
        },
        "evaluation": {
            "evidence_class": (
                "fixed_development_diagnostic_repair"
            ),
            "folds": list(FOLDS),
            "root_seed": ROOT_SEED,
            "budget": asdict(p11.stage3.Stage3Budget()),
            "f3_manifest_stable_hash": (
                TASK_MANIFEST_HASHES["facies_f3"]
            ),
            "penobscot_manifest_stable_hash": (
                TASK_MANIFEST_HASHES["facies_penobscot"]
            ),
            "frozen_test_accessed": False,
            "holdout_paths_accepted": False,
            "raw_predictions_persisted": False,
            "feature_cache_persisted": False,
            "cross_run_comparison_policy": (
                "same_run_delta_is_primary_due_to_observed_"
                "p11_p12_baseline_drift"
            ),
        },
        "p11_reference": {
            "summary_path": str(
                P11_SUMMARY_PATH.relative_to(PROJECT_ROOT)
            ),
            "artifact_manifest_sha256": p11_manifest_sha,
            "schema_version": p11_summary["schema_version"],
        },
        "tasks": task_summaries,
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "command": run_command,
        },
    }
    results_path = p11._write_jsonl(
        output_root / "p12_repair_v1_results.jsonl",
        all_rows,
    )
    summary_path = p11._write_json(
        output_root / "p12_repair_v1_summary.json",
        summary_payload,
    )
    evidence_path = _write_evidence(
        task_summaries,
        output_root,
        run_command=run_command,
    )
    manifest_rows = [
        {
            "kind": "jsonl",
            "name": results_path.name,
            "path": str(results_path.relative_to(PROJECT_ROOT)),
            "sha256": p11._sha256(results_path),
        },
        {
            "kind": "json",
            "name": summary_path.name,
            "path": str(summary_path.relative_to(PROJECT_ROOT)),
            "sha256": p11._sha256(summary_path),
        },
        {
            "kind": "md",
            "name": evidence_path.name,
            "path": str(evidence_path.relative_to(PROJECT_ROOT)),
            "sha256": p11._sha256(evidence_path),
        },
    ]
    manifest_path = p11._write_csv(
        output_root / "artifact_manifest.csv",
        manifest_rows,
        ["kind", "name", "path", "sha256"],
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "evidence": evidence_path,
        "artifact_manifest": manifest_path,
    }


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    manifest_path = output_root / "artifact_manifest.csv"
    summary_path = output_root / "p12_repair_v1_summary.json"
    results_path = output_root / "p12_repair_v1_results.jsonl"
    evidence_path = output_root / "evidence.md"
    for path in (
        manifest_path,
        summary_path,
        results_path,
        evidence_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"missing P12 evidence artifact: {path}"
            )

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 3:
        raise ValueError(
            "P12 artifact manifest must contain three artifacts, "
            f"got {len(manifest_rows)}"
        )
    for row in manifest_rows:
        path = PROJECT_ROOT / row["path"]
        try:
            path.resolve().relative_to(HERE)
        except ValueError as exc:
            raise ValueError(
                f"non-facies artifact escaped P12 manifest: {path}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"P12 manifest artifact is missing: {path}"
            )
        if p11._sha256(path) != row["sha256"]:
            raise ValueError(
                f"P12 artifact hash mismatch: {path}"
            )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported P12 summary schema")
    evaluation = summary["evaluation"]
    if (
        tuple(evaluation["folds"]) != FOLDS
        or evaluation["frozen_test_accessed"]
        or evaluation["holdout_paths_accepted"]
    ):
        raise ValueError(
            "P12 evidence violates the fixed-development boundary"
        )
    experiment = summary["experiment"]
    if (
        experiment["input_policy"]
        != "native_128_no_spatial_interpolation"
        or experiment["input_shape_constraint"]
        != [3, 128, 128]
        or experiment["observed_feature_shapes"]
        != [[256, 32, 32], [256, 16, 16], [256, 8, 8]]
    ):
        raise ValueError(
            "P12 native-resolution SAM2 contract drifted"
        )
    if experiment["trainable_hiera_blocks"] != [22, 23]:
        raise ValueError("P12 trainable Hiera block contract drifted")
    if (
        summary["p11_reference"]["artifact_manifest_sha256"]
        != p11._sha256(P11_ARTIFACT_MANIFEST)
    ):
        raise ValueError("P11 evidence changed after P12 run")

    rows = [
        json.loads(line)
        for line in results_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    expected_cells = {
        (task_id, fold_id, variant)
        for task_id in ("facies_f3", "facies_penobscot")
        for fold_id in FOLDS
        for variant in VARIANTS
    }
    observed_cells = {
        (row["task_id"], int(row["fold_id"]), row["variant"])
        for row in rows
    }
    if observed_cells != expected_cells or len(rows) != 20:
        raise ValueError(
            "P12 results do not contain the exact 20-cell grid"
        )
    current_runner_sha = p11._sha256(Path(__file__))
    trainable_variants = {
        "direct_sam2",
        "pretrained_residual",
        "random_sam2_residual",
    }
    for row in rows:
        if row["frozen_test_accessed"]:
            raise ValueError("a P12 row claims frozen-test access")
        if row["runner_sha256"] != current_runner_sha:
            raise ValueError(
                "P12 row is not bound to the current runner"
            )
        if row["sam2_input_shape"] != [3, 128, 128]:
            raise ValueError("P12 row contains a non-native input shape")
        if row["feature_shapes"] != [
            [256, 32, 32],
            [256, 16, 16],
            [256, 8, 8],
        ]:
            raise ValueError(
                "P12 row contains an unexpected native feature pyramid"
            )
        if float(row["residual_max_abs"]) > (
            MAX_RESIDUAL_CORRECTION + 1e-6
        ):
            raise ValueError(
                "P12 result exceeded the residual correction bound"
            )
        if row["variant"] in trainable_variants:
            if row["encoder_trainable_blocks"] != [22, 23]:
                raise ValueError(
                    "P12 trainable variant used the wrong blocks"
                )
            if (
                int(row["encoder_trainable_parameters"]) <= 0
                or float(row["encoder_update_l2"]) <= 0.0
                or float(row["encoder_last_grad_norm"]) <= 0.0
            ):
                raise ValueError(
                    "P12 trainable encoder lacks update evidence"
                )
    for task_id in ("facies_f3", "facies_penobscot"):
        for fold_id in FOLDS:
            subset = [
                row
                for row in rows
                if row["task_id"] == task_id
                and int(row["fold_id"]) == fold_id
            ]
            by_variant = {
                row["variant"]: row for row in subset
            }
            baseline = float(
                by_variant["strong_small_baseline"][
                    "metric_value"
                ]
            )
            gate_zero = by_variant["gate_zero"]
            if abs(
                float(gate_zero["metric_value"]) - baseline
            ) > 1e-12:
                raise ValueError(
                    f"{task_id} fold {fold_id} gate=0 metric drift"
                )
            if (
                float(
                    gate_zero["gate_zero_max_abs_logit_diff"]
                )
                > 1e-7
            ):
                raise ValueError(
                    f"{task_id} fold {fold_id} gate=0 logit drift"
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "artifacts": len(manifest_rows),
        "folds": list(FOLDS),
        "decisions": {
            task: summary["tasks"][task]["decision"]["state"]
            for task in ("F3", "Penobscot")
        },
        "native_input_shape": [3, 128, 128],
        "trainable_hiera_blocks": [22, 23],
        "frozen_test_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run_parser = commands.add_parser(
        "run",
        help="run both fixed-development P12 repair ablations",
    )
    run_parser.add_argument(
        "--f3-manifest",
        type=Path,
        required=True,
    )
    run_parser.add_argument(
        "--penobscot-manifest",
        type=Path,
        required=True,
    )
    run_parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
    )
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    verify_parser = commands.add_parser(
        "verify",
        help="verify portable P12 repair evidence",
    )
    verify_parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        outputs = build(
            f3_manifest=args.f3_manifest,
            penobscot_manifest=args.penobscot_manifest,
            processed_root=args.processed_root,
            device=args.device,
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    name: str(path)
                    for name, path in outputs.items()
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                verify(args.output_root),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
