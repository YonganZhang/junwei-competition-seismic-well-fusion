#!/usr/bin/env python3
"""P13 cross-attention fusion experiment for the facies track.

The accepted small CNN supplies the query feature map and SAM2 supplies
key/value features. A low-resolution multi-head cross-attention module writes
the fused representation back into the CNN decoder path. The SAM2 weight mode
is a single explicit parameter (``pretrained`` or ``random``); this run uses
pretrained weights, while the random mode is reserved for a later ablation.

The CLI can only address the locked F3/Penobscot development manifests and
their train.h5 archives. It has no frozen-holdout or test-archive argument.
"""
from __future__ import annotations

import argparse
import copy
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
import p12_repair_v1 as p12  # noqa: E402


OUTPUT_ROOT = HERE / "_outputs" / "p13_cross_attention"
SCHEMA_VERSION = "facies-p13-cross-attention/v1"
ROOT_SEED = p11.ROOT_SEED
FOLDS = p11.FOLDS
BASELINE_UPDATES = p11.MAX_UPDATES
BATCH_SIZE = p11.BATCH_SIZE
CANDIDATE_UPDATES = 160
CNN_LR = 5e-5
FUSION_LR = 2e-4
SAM2_LR = 1e-5
WEIGHT_DECAY = 1e-4
DICE_WEIGHT = 0.25
NOISE_STD = 0.03
HORIZONTAL_FLIP_PROBABILITY = 0.5
ATTENTION_DIM = 128
ATTENTION_HEADS = 4
GRAD_CLIP = 1.0
SAM2_WEIGHT_MODES = ("pretrained", "random")
DEFAULT_SAM2_WEIGHT_MODE = "pretrained"
VARIANTS = (
    "strong_small_baseline",
    "continued_cnn_control",
    "cross_attention_fusion",
)
TASK_MANIFEST_HASHES = p11.TASK_MANIFEST_HASHES


def _validate_output_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(
            f"P13 output must stay inside the facies track: {resolved}"
        ) from exc
    protected = {
        p11.OUTPUT_ROOT.resolve(),
        p12.OUTPUT_ROOT.resolve(),
    }
    if resolved in protected:
        raise ValueError("P13 refuses to overwrite P11/P12 evidence")
    return resolved


def _validate_sam2_weight_mode(weight_mode: str) -> str:
    value = str(weight_mode).strip().lower()
    if value not in SAM2_WEIGHT_MODES:
        raise ValueError(
            f"SAM2 weight mode must be one of {SAM2_WEIGHT_MODES}, "
            f"got {weight_mode!r}"
        )
    return value


def build_sam2_encoder(
    task_id: str,
    num_classes: int,
    device: str,
    *,
    weight_mode: str = DEFAULT_SAM2_WEIGHT_MODE,
    seed: int = ROOT_SEED,
) -> p12.TrainableSam2Encoder:
    """Build SAM2 with a one-argument pretrained/random switch."""
    mode = _validate_sam2_weight_mode(weight_mode)
    return p12._make_sam2_encoder(
        task_id,
        num_classes,
        device,
        randomize=mode == "random",
        random_seed=seed,
    )


def _coordinate_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=0)[None]
    return coordinates.expand(batch_size, -1, -1, -1)


class CrossAttentionFusion(torch.nn.Module):
    """CNN queries attend to SAM2 key/value tokens at low resolution."""

    def __init__(
        self,
        *,
        cnn_channels: int = 512,
        sam_channels: int = 256,
        attention_dim: int = ATTENTION_DIM,
        attention_heads: int = ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        if attention_dim % attention_heads != 0:
            raise ValueError("attention_dim must divide attention_heads")
        self.cnn_channels = cnn_channels
        self.sam_channels = sam_channels
        self.attention_dim = attention_dim
        self.query_projection = torch.nn.Sequential(
            torch.nn.Conv2d(cnn_channels, attention_dim, kernel_size=1),
            torch.nn.GroupNorm(8, attention_dim),
        )
        self.key_projection = torch.nn.Sequential(
            torch.nn.Conv2d(sam_channels, attention_dim, kernel_size=1),
            torch.nn.GroupNorm(8, attention_dim),
        )
        self.value_projection = torch.nn.Conv2d(
            sam_channels,
            attention_dim,
            kernel_size=1,
        )
        self.position_projection = torch.nn.Conv2d(
            2,
            attention_dim,
            kernel_size=1,
        )
        self.attention = torch.nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.output_projection = torch.nn.Conv2d(
            attention_dim,
            cnn_channels,
            kernel_size=1,
        )
        self.output_norm = torch.nn.GroupNorm(16, cnn_channels)
        self.fusion_scale_logit = torch.nn.Parameter(
            torch.tensor(-1.3862944)
        )
        torch.nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=1e-3,
        )
        torch.nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sam_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            cnn_feature.ndim != 4
            or sam_feature.ndim != 4
            or cnn_feature.shape[0] != sam_feature.shape[0]
        ):
            raise ValueError(
                "cross attention expects aligned four-dimensional "
                "CNN/SAM2 feature maps"
            )
        if cnn_feature.shape[1] != self.cnn_channels:
            raise ValueError(
                f"expected {self.cnn_channels} CNN channels, got "
                f"{cnn_feature.shape[1]}"
            )
        if sam_feature.shape[1] != self.sam_channels:
            raise ValueError(
                f"expected {self.sam_channels} SAM2 channels, got "
                f"{sam_feature.shape[1]}"
            )
        batch_size, _, query_height, query_width = cnn_feature.shape
        _, _, key_height, key_width = sam_feature.shape
        query_map = self.query_projection(cnn_feature)
        key_map = self.key_projection(sam_feature)
        value_map = self.value_projection(sam_feature)
        query_map = query_map + self.position_projection(
            _coordinate_grid(
                batch_size,
                query_height,
                query_width,
                device=query_map.device,
                dtype=query_map.dtype,
            )
        )
        key_map = key_map + self.position_projection(
            _coordinate_grid(
                batch_size,
                key_height,
                key_width,
                device=key_map.device,
                dtype=key_map.dtype,
            )
        )
        query_tokens = query_map.flatten(2).transpose(1, 2)
        key_tokens = key_map.flatten(2).transpose(1, 2)
        value_tokens = value_map.flatten(2).transpose(1, 2)
        attended, weights = self.attention(
            query_tokens,
            key_tokens,
            value_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        attended_map = attended.transpose(1, 2).reshape(
            batch_size,
            self.attention_dim,
            query_height,
            query_width,
        )
        delta = self.output_norm(
            self.output_projection(attended_map)
        )
        scale = torch.sigmoid(self.fusion_scale_logit)
        return cnn_feature + scale * delta, weights

    @property
    def fusion_scale(self) -> float:
        return float(
            torch.sigmoid(self.fusion_scale_logit).detach().cpu()
        )


class CrossAttentionSegmentationModel(torch.nn.Module):
    def __init__(
        self,
        small_model: torch.nn.Module,
        sam2_encoder: p12.TrainableSam2Encoder,
    ) -> None:
        super().__init__()
        self.small_model = small_model
        self.sam2_encoder = sam2_encoder
        for parameter in self.small_model.parameters():
            parameter.requires_grad = False
        for parameter in self.small_model.decoder.parameters():
            parameter.requires_grad = True
        for parameter in self.small_model.segmentation_head.parameters():
            parameter.requires_grad = True
        self.fusion = CrossAttentionFusion(
            cnn_channels=int(
                self.small_model.encoder.out_channels[-1]
            )
        )

    def train(
        self,
        mode: bool = True,
    ) -> "CrossAttentionSegmentationModel":
        super().train(mode)
        self.small_model.encoder.eval()
        self.small_model.decoder.train(mode)
        self.small_model.segmentation_head.train(mode)
        self.sam2_encoder.train(mode)
        self.fusion.train(mode)
        return self

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cnn_features = list(self.small_model.encoder(inputs))
        sam_features = self.sam2_encoder(inputs)
        fused, attention_weights = self.fusion(
            cnn_features[-1],
            sam_features[-1],
        )
        cnn_features[-1] = fused
        decoded = self.small_model.decoder(cnn_features)
        logits = self.small_model.segmentation_head(decoded)
        return logits, attention_weights


class ContinuedCnnModel(torch.nn.Module):
    """Decoder/head continuation control with no SAM2 path."""

    def __init__(self, small_model: torch.nn.Module) -> None:
        super().__init__()
        self.small_model = small_model
        for parameter in self.small_model.parameters():
            parameter.requires_grad = False
        for parameter in self.small_model.decoder.parameters():
            parameter.requires_grad = True
        for parameter in self.small_model.segmentation_head.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "ContinuedCnnModel":
        super().train(mode)
        self.small_model.encoder.eval()
        self.small_model.decoder.train(mode)
        self.small_model.segmentation_head.train(mode)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.small_model.encoder(inputs)
        decoded = self.small_model.decoder(features)
        return self.small_model.segmentation_head(decoded)


def _candidate_batches(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    updates: int = CANDIDATE_UPDATES,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed + 1300)
    indices = np.arange(len(images))
    for _ in range(updates):
        choice = rng.choice(
            indices,
            size=min(BATCH_SIZE, len(indices)),
            replace=False,
        )
        batch_images = np.asarray(
            images[choice],
            dtype=np.float32,
        ).copy()
        batch_labels = np.asarray(
            labels[choice],
            dtype=np.int64,
        ).copy()
        flips = rng.random(len(choice)) < HORIZONTAL_FLIP_PROBABILITY
        for item, flipped in enumerate(flips):
            if flipped:
                batch_images[item] = np.flip(
                    batch_images[item],
                    axis=-1,
                )
                batch_labels[item] = np.flip(
                    batch_labels[item],
                    axis=-1,
                )
        scales = rng.uniform(
            0.95,
            1.05,
            size=(len(choice), 1, 1, 1),
        ).astype(np.float32)
        noise = rng.normal(
            0.0,
            NOISE_STD,
            size=batch_images.shape,
        ).astype(np.float32)
        batch_images = batch_images * scales + noise
        yield (
            torch.from_numpy(np.ascontiguousarray(batch_images)),
            torch.from_numpy(np.ascontiguousarray(batch_labels)),
        )


def _soft_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).permute(0, 3, 1, 2)
    targets = one_hot.to(dtype=probabilities.dtype)
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = (
        probabilities.sum(dim=(0, 2, 3))
        + targets.sum(dim=(0, 2, 3))
    )
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
    ) + DICE_WEIGHT * _soft_dice_loss(logits, labels)


def _predict_logits(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    logits_parts: list[np.ndarray] = []
    attention_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.as_tensor(
                images[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            output = model(batch)
            if isinstance(output, tuple):
                logits, attention = output
                attention_parts.append(attention.detach().cpu())
            else:
                logits = output
            logits_parts.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_parts, axis=0), attention_parts


def _attention_entropy(
    attention_parts: Sequence[torch.Tensor],
) -> float:
    if not attention_parts:
        return 0.0
    weights = torch.cat(list(attention_parts), dim=0).float()
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=-1)
    normalization = math.log(max(int(weights.shape[-1]), 2))
    return float((entropy / normalization).mean())


def _train_continued_cnn(
    model: ContinuedCnnModel,
    prepared: Any,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    p11._seed_all(seed)
    model = model.to(device)
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=CNN_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CANDIDATE_UPDATES,
    )
    class_weights = torch.as_tensor(
        prepared.class_weights,
        dtype=torch.float32,
        device=device,
    )
    history: list[float] = []
    model.train()
    for images, labels in _candidate_batches(
        prepared.train_images,
        prepared.train_labels,
        seed=seed,
    ):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = _combined_loss(
            logits,
            labels,
            class_weights=class_weights,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "continued CNN loss became non-finite"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        history.append(float(loss.detach().cpu()))
    validation_logits, _ = _predict_logits(
        model,
        prepared.validation_images,
        device=device,
    )
    metrics, confusion = p11._metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    model.cpu()
    return {
        "metrics": metrics,
        "confusion": confusion,
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
    }


def _train_cross_attention(
    model: CrossAttentionSegmentationModel,
    prepared: Any,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    p11._seed_all(seed)
    model = model.to(device)
    sam_parameters = model.sam2_encoder.trainable_parameters()
    sam_snapshots = p12._snapshot_trainable(sam_parameters)
    cnn_parameters = [
        parameter
        for module in (
            model.small_model.decoder,
            model.small_model.segmentation_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    fusion_parameters = list(model.fusion.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": cnn_parameters, "lr": CNN_LR},
            {"params": fusion_parameters, "lr": FUSION_LR},
            {"params": sam_parameters, "lr": SAM2_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CANDIDATE_UPDATES,
    )
    all_trainable = [
        *cnn_parameters,
        *fusion_parameters,
        *sam_parameters,
    ]
    class_weights = torch.as_tensor(
        prepared.class_weights,
        dtype=torch.float32,
        device=device,
    )
    history: list[float] = []
    last_grad_norm = 0.0
    model.train()
    for update_index, (images, labels) in enumerate(
        _candidate_batches(
            prepared.train_images,
            prepared.train_labels,
            seed=seed,
        )
    ):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = _combined_loss(
            logits,
            labels,
            class_weights=class_weights,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "cross-attention loss became non-finite"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            all_trainable,
            GRAD_CLIP,
        )
        if update_index == CANDIDATE_UPDATES - 1:
            last_grad_norm = float(grad_norm.detach().cpu())
        optimizer.step()
        scheduler.step()
        history.append(float(loss.detach().cpu()))
    validation_logits, attention_parts = _predict_logits(
        model,
        prepared.validation_images,
        device=device,
    )
    metrics, confusion = p11._metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    sam_update_l2 = p12._parameter_delta_l2(
        sam_parameters,
        sam_snapshots,
    )
    payload = {
        "metrics": metrics,
        "confusion": confusion,
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "trainable_parameters": sum(
            parameter.numel() for parameter in all_trainable
        ),
        "sam2_trainable_parameters": (
            model.sam2_encoder.trainable_parameter_count
        ),
        "sam2_update_l2": sam_update_l2,
        "last_grad_norm": last_grad_norm,
        "attention_entropy": _attention_entropy(attention_parts),
        "fusion_scale": model.fusion.fusion_scale,
    }
    model.cpu()
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
    sam2_weight_mode: str,
) -> list[dict[str, Any]]:
    budget = p11.stage3.Stage3Budget()
    runner_sha256 = p11._sha256(Path(__file__))
    git_head = p11._git_head()
    checkpoint_sha256 = p11._sha256(p11.SAM2_CHECKPOINT)
    results: list[dict[str, Any]] = []
    stable_hash: str | None = None
    for fold_id in FOLDS:
        fold_started = time.perf_counter()
        prepared = p11.stage3.prepare_fold(
            task_id=task_id,
            fold_id=fold_id,
            manifest_path=manifest_path,
            processed_root=processed_root,
            budget=budget,
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
        baseline_started = time.perf_counter()
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
        baseline_seconds = time.perf_counter() - baseline_started
        trained_baseline = baseline_result.pop("model")
        trained_baseline.cpu()

        continued_model = ContinuedCnnModel(
            copy.deepcopy(trained_baseline)
        )
        continued_started = time.perf_counter()
        continued_result = _train_continued_cnn(
            continued_model,
            prepared,
            device=device,
            seed=seed,
        )
        continued_seconds = (
            time.perf_counter() - continued_started
        )
        del continued_model
        gc.collect()
        torch.cuda.empty_cache()

        p11._seed_all(seed)
        cross_small_model = copy.deepcopy(trained_baseline)
        sam2_encoder = build_sam2_encoder(
            task_id,
            prepared.num_classes,
            device,
            weight_mode=sam2_weight_mode,
            seed=seed,
        )
        cross_model = CrossAttentionSegmentationModel(
            cross_small_model,
            sam2_encoder,
        )
        cross_started = time.perf_counter()
        cross_result = _train_cross_attention(
            cross_model,
            prepared,
            device=device,
            seed=seed,
        )
        cross_seconds = time.perf_counter() - cross_started
        del cross_model, cross_small_model, sam2_encoder
        del trained_baseline, baseline_model
        gc.collect()
        torch.cuda.empty_cache()

        baseline_metric = float(
            baseline_result["metrics"]["miou"]
        )
        payloads = [
            (
                "strong_small_baseline",
                baseline_result,
                "original accepted 40-update small CNN baseline",
                baseline_seconds,
            ),
            (
                "continued_cnn_control",
                continued_result,
                "same extra optimization and augmentation without "
                "the cross-attention/SAM2 branch",
                continued_seconds,
            ),
            (
                "cross_attention_fusion",
                cross_result,
                "CNN feature queries cross-attend to SAM2 key/value "
                "features before the small-model decoder",
                cross_seconds,
            ),
        ]
        for variant, payload, notes, duration_seconds in payloads:
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
                "train_loss_last": payload.get("train_loss_last"),
                "train_loss_mean": payload.get("train_loss_mean"),
                "baseline_updates": BASELINE_UPDATES,
                "candidate_updates": (
                    CANDIDATE_UPDATES
                    if variant != "strong_small_baseline"
                    else 0
                ),
                "augmentation": (
                    "horizontal_flip_intensity_scale_gaussian_noise"
                    if variant != "strong_small_baseline"
                    else "none"
                ),
                "dice_weight": (
                    DICE_WEIGHT
                    if variant != "strong_small_baseline"
                    else 0.0
                ),
                "sam2_weight_mode": (
                    sam2_weight_mode
                    if variant == "cross_attention_fusion"
                    else "not_applicable"
                ),
                "sam2_trainable_blocks": (
                    [22, 23]
                    if variant == "cross_attention_fusion"
                    else []
                ),
                "sam2_trainable_parameters": int(
                    payload.get(
                        "sam2_trainable_parameters",
                        0,
                    )
                ),
                "sam2_update_l2": float(
                    payload.get("sam2_update_l2", 0.0)
                ),
                "attention_entropy": float(
                    payload.get("attention_entropy", 0.0)
                ),
                "fusion_scale": float(
                    payload.get("fusion_scale", 0.0)
                ),
                "last_grad_norm": float(
                    payload.get("last_grad_norm", 0.0)
                ),
                "trainable_parameters": int(
                    payload.get("trainable_parameters", 0)
                ),
                "duration_seconds": float(duration_seconds),
                "command": run_command,
                "notes": notes,
                "evidence_path": str(
                    (output_root / "evidence.md").relative_to(
                        PROJECT_ROOT
                    )
                ),
                "runner_sha256": runner_sha256,
                "git_head_at_run": git_head,
                "manifest_stable_hash": (
                    prepared.manifest_stable_hash
                ),
                "fold_split_hash": prepared.fold_split_hash,
                "train_samples": len(prepared.train_images),
                "validation_samples": len(
                    prepared.validation_images
                ),
                "checkpoint_path": (
                    str(p11.SAM2_CHECKPOINT)
                    if variant == "cross_attention_fusion"
                    and sam2_weight_mode == "pretrained"
                    else ""
                ),
                "checkpoint_sha256": (
                    checkpoint_sha256
                    if variant == "cross_attention_fusion"
                    and sam2_weight_mode == "pretrained"
                    else ""
                ),
                "frozen_test_accessed": False,
            }
            results.append(row)
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "fold_id": fold_id,
                    "baseline_miou": baseline_metric,
                    "continued_cnn_miou": float(
                        continued_result["metrics"]["miou"]
                    ),
                    "cross_attention_miou": float(
                        cross_result["metrics"]["miou"]
                    ),
                    "cross_delta": float(
                        cross_result["metrics"]["miou"]
                        - baseline_metric
                    ),
                    "fusion_scale": float(
                        cross_result["fusion_scale"]
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
            "nll": float(
                np.mean([row["nll"] for row in subset])
            ),
            "attention_entropy": float(
                np.mean(
                    [row["attention_entropy"] for row in subset]
                )
            ),
            "fusion_scale": float(
                np.mean([row["fusion_scale"] for row in subset])
            ),
            "sam2_update_l2": float(
                np.mean([row["sam2_update_l2"] for row in subset])
            ),
        }
    return means


def _write_evidence(
    summary: Mapping[str, Any],
    output_root: Path,
) -> Path:
    weight_mode = summary["experiment"]["sam2_weight_mode"]
    lines = [
        "# P13 cross-attention evidence",
        "",
        "## Experiment",
        "",
        "The accepted CNN deepest feature map supplies query tokens. "
        "The native-128 SAM2 feature map supplies key/value tokens. "
        "Four-head cross-attention writes a gated feature residual "
        "back before the original CNN decoder.",
        "",
        "SAM2 loading is controlled by one `sam2_weight_mode` "
        "parameter with accepted values `pretrained` and `random`. "
        f"This evidence was generated with `{weight_mode}`; the "
        "opposite mode is not included in this package.",
        "",
        "The candidate also uses 160 continuation updates, horizontal "
        "flips, mild intensity/noise augmentation, AdamW, cosine "
        "decay, and CE+Dice loss. A continued-CNN control receives "
        "the same optimization without the cross-attention/SAM2 path.",
        "",
        "## Fixed-development results",
        "",
        "| Task | Variant | mIoU | Δ vs baseline | Accuracy | "
        "Macro F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task_name in ("F3", "Penobscot"):
        task = summary["tasks"][task_name]
        baseline = task["variant_means"][
            "strong_small_baseline"
        ]["miou"]
        for variant in VARIANTS:
            values = task["variant_means"][variant]
            lines.append(
                f"| {task_name} | {variant} | "
                f"{values['miou']:.6f} | "
                f"{values['miou'] - baseline:+.6f} | "
                f"{values['accuracy']:.6f} | "
                f"{values['macro_f1']:.6f} |"
            )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            "## Objective conclusion",
            "",
            f"- F3 overall mIoU changed from "
            f"{summary['tasks']['F3']['variant_means']['strong_small_baseline']['miou']:.6f} "
            f"to "
            f"{summary['tasks']['F3']['variant_means']['cross_attention_fusion']['miou']:.6f} "
            f"({summary['tasks']['F3']['comparison']['cross_attention_minus_baseline']:+.6f}).",
            f"- Penobscot overall mIoU changed from "
            f"{summary['tasks']['Penobscot']['variant_means']['strong_small_baseline']['miou']:.6f} "
            f"to "
            f"{summary['tasks']['Penobscot']['variant_means']['cross_attention_fusion']['miou']:.6f} "
            f"({summary['tasks']['Penobscot']['comparison']['cross_attention_minus_baseline']:+.6f}).",
            f"- Equal-task mean mIoU changed from "
            f"{overall['baseline_mean_miou']:.6f} to "
            f"{overall['cross_attention_mean_miou']:.6f} "
            f"({overall['cross_attention_minus_baseline']:+.6f}).",
            "",
            "**大模型贡献占比待下一轮消融确认。** These numbers "
            "objectively describe the complete P13 package; they do "
            "not attribute the change to SAM2 because the random-weight "
            "ablation has not yet been run.",
            "",
            "## Data and evaluation boundary",
            "",
            "- mIoU uses the unchanged P11/P12 probability evaluator.",
            "- Validation samples and locked folds 0/4 are unchanged.",
            "- Augmentation and fitting use training samples/labels only.",
            "- No frozen holdout or `test.h5` path was accepted or read.",
            "- No dense prediction, checkpoint copy, or feature cache "
            "is persisted.",
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
    sam2_weight_mode: str = DEFAULT_SAM2_WEIGHT_MODE,
) -> dict[str, Path]:
    started = time.perf_counter()
    mode = _validate_sam2_weight_mode(sam2_weight_mode)
    p11._validate_cuda_device(device)
    manifests = p11.validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    output_root = _validate_output_root(output_root)
    p11.verify(p11.OUTPUT_ROOT)
    p12.verify(p12.OUTPUT_ROOT)
    protected_hashes = {
        "p11_manifest": p11._sha256(
            p11.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p12_manifest": p11._sha256(
            p12.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
    }
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
            "--sam2-weight-mode",
            mode,
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
            sam2_weight_mode=mode,
        )
        all_rows.extend(rows)
        task_name = (
            "F3" if task_id == "facies_f3" else "Penobscot"
        )
        means = _variant_means(rows)
        baseline = means["strong_small_baseline"]["miou"]
        task_summaries[task_name] = {
            "task_id": task_id,
            "variant_means": means,
            "comparison": {
                "continued_cnn_minus_baseline": (
                    means["continued_cnn_control"]["miou"]
                    - baseline
                ),
                "cross_attention_minus_baseline": (
                    means["cross_attention_fusion"]["miou"]
                    - baseline
                ),
                "cross_attention_minus_continued_cnn": (
                    means["cross_attention_fusion"]["miou"]
                    - means["continued_cnn_control"]["miou"]
                ),
            },
        }
    baseline_mean = float(
        np.mean(
            [
                task_summaries[task]["variant_means"][
                    "strong_small_baseline"
                ]["miou"]
                for task in ("F3", "Penobscot")
            ]
        )
    )
    cross_mean = float(
        np.mean(
            [
                task_summaries[task]["variant_means"][
                    "cross_attention_fusion"
                ]["miou"]
                for task in ("F3", "Penobscot")
            ]
        )
    )
    current_hashes = {
        "p11_manifest": p11._sha256(
            p11.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p12_manifest": p11._sha256(
            p12.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
    }
    if current_hashes != protected_hashes:
        raise RuntimeError("P11/P12 evidence changed during P13")
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "architecture": (
                "deepest_cnn_query_to_sam2_key_value_"
                "multihead_cross_attention"
            ),
            "sam2_weight_mode": mode,
            "sam2_weight_mode_interface": list(SAM2_WEIGHT_MODES),
            "sam2_source_revision": p11.SAM2_SOURCE_REVISION,
            "sam2_checkpoint_sha256": p11._sha256(checkpoint),
            "sam2_dependency_site": str(dependency_site),
            "sam2_input_policy": (
                "native_128_no_spatial_interpolation"
            ),
            "sam2_trainable_hiera_blocks": [22, 23],
            "attention_dim": ATTENTION_DIM,
            "attention_heads": ATTENTION_HEADS,
            "baseline_updates": BASELINE_UPDATES,
            "candidate_updates": CANDIDATE_UPDATES,
            "optimization": {
                "cnn_lr": CNN_LR,
                "fusion_lr": FUSION_LR,
                "sam2_lr": SAM2_LR,
                "weight_decay": WEIGHT_DECAY,
                "dice_weight": DICE_WEIGHT,
                "horizontal_flip_probability": (
                    HORIZONTAL_FLIP_PROBABILITY
                ),
                "noise_std": NOISE_STD,
                "scheduler": "cosine",
            },
            "attribution_status": (
                "large_model_contribution_pending_random_weight_"
                "ablation"
            ),
        },
        "evaluation": {
            "evidence_class": (
                "fixed_development_cross_attention_experiment"
            ),
            "folds": list(FOLDS),
            "root_seed": ROOT_SEED,
            "budget": asdict(p11.stage3.Stage3Budget()),
            "candidate_updates": CANDIDATE_UPDATES,
            "f3_manifest_stable_hash": (
                TASK_MANIFEST_HASHES["facies_f3"]
            ),
            "penobscot_manifest_stable_hash": (
                TASK_MANIFEST_HASHES["facies_penobscot"]
            ),
            "metric_implementation": (
                "_pipelines.02_task_datasets.facies."
                "p4_metrics.evaluate_probabilities"
            ),
            "frozen_test_accessed": False,
            "holdout_paths_accepted": False,
            "raw_predictions_persisted": False,
        },
        "protected_evidence": protected_hashes,
        "tasks": task_summaries,
        "overall": {
            "aggregation": "equal_mean_of_F3_and_Penobscot_mIoU",
            "baseline_mean_miou": baseline_mean,
            "cross_attention_mean_miou": cross_mean,
            "cross_attention_minus_baseline": (
                cross_mean - baseline_mean
            ),
        },
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
        output_root / "p13_cross_attention_results.jsonl",
        all_rows,
    )
    summary_path = p11._write_json(
        output_root / "p13_cross_attention_summary.json",
        summary_payload,
    )
    evidence_path = _write_evidence(
        summary_payload,
        output_root,
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
    summary_path = output_root / "p13_cross_attention_summary.json"
    results_path = output_root / "p13_cross_attention_results.jsonl"
    evidence_path = output_root / "evidence.md"
    for path in (
        manifest_path,
        summary_path,
        results_path,
        evidence_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"missing P13 evidence artifact: {path}"
            )
    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 3:
        raise ValueError(
            "P13 artifact manifest must contain three artifacts"
        )
    for row in manifest_rows:
        path = PROJECT_ROOT / row["path"]
        try:
            path.resolve().relative_to(HERE)
        except ValueError as exc:
            raise ValueError(
                f"non-facies artifact escaped P13 manifest: {path}"
            ) from exc
        if not path.is_file() or p11._sha256(path) != row["sha256"]:
            raise ValueError(
                f"P13 artifact missing or hash-mismatched: {path}"
            )
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported P13 summary schema")
    if (
        tuple(summary["evaluation"]["folds"]) != FOLDS
        or summary["evaluation"]["frozen_test_accessed"]
        or summary["evaluation"]["holdout_paths_accepted"]
    ):
        raise ValueError(
            "P13 evidence violates the development-only boundary"
        )
    experiment = summary["experiment"]
    weight_mode = _validate_sam2_weight_mode(
        experiment["sam2_weight_mode"]
    )
    if experiment["sam2_weight_mode_interface"] != [
        "pretrained",
        "random",
    ]:
        raise ValueError("P13 SAM2 weight-mode contract drifted")
    current_protected = {
        "p11_manifest": p11._sha256(
            p11.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
        "p12_manifest": p11._sha256(
            p12.OUTPUT_ROOT / "artifact_manifest.csv"
        ),
    }
    if summary["protected_evidence"] != current_protected:
        raise ValueError("P11/P12 evidence changed after P13")
    rows = [
        json.loads(line)
        for line in results_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    expected = {
        (task_id, fold_id, variant)
        for task_id in ("facies_f3", "facies_penobscot")
        for fold_id in FOLDS
        for variant in VARIANTS
    }
    observed = {
        (row["task_id"], int(row["fold_id"]), row["variant"])
        for row in rows
    }
    if len(rows) != 12 or observed != expected:
        raise ValueError(
            "P13 results do not contain the exact 12-cell grid"
        )
    runner_sha = p11._sha256(Path(__file__))
    for row in rows:
        if row["runner_sha256"] != runner_sha:
            raise ValueError(
                "P13 row is not bound to the current runner"
            )
        if row["frozen_test_accessed"]:
            raise ValueError("a P13 row claims frozen-test access")
        if row["variant"] == "cross_attention_fusion":
            if (
                row["sam2_weight_mode"] != weight_mode
                or row["sam2_trainable_blocks"] != [22, 23]
                or int(row["sam2_trainable_parameters"]) <= 0
                or float(row["sam2_update_l2"]) <= 0.0
                or float(row["last_grad_norm"]) <= 0.0
                or not 0.0 < float(row["fusion_scale"]) < 1.0
                or not 0.0 < float(row["attention_entropy"]) <= 1.0
            ):
                raise ValueError(
                    "P13 cross-attention row lacks live training evidence"
                )
    evidence_text = evidence_path.read_text(encoding="utf-8")
    if "大模型贡献占比待下一轮消融确认" not in evidence_text:
        raise ValueError(
            "P13 evidence omitted the required attribution caveat"
        )
    forbidden_claim = "提升来自SAM2"
    if forbidden_claim in evidence_text:
        raise ValueError(
            "P13 evidence contains a forbidden attribution claim"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "artifacts": len(manifest_rows),
        "folds": list(FOLDS),
        "sam2_weight_mode": weight_mode,
        "overall_delta_miou": summary["overall"][
            "cross_attention_minus_baseline"
        ],
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
        help="run both fixed-development P13 comparisons",
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
    run_parser.add_argument(
        "--sam2-weight-mode",
        choices=SAM2_WEIGHT_MODES,
        default=DEFAULT_SAM2_WEIGHT_MODE,
    )
    verify_parser = commands.add_parser(
        "verify",
        help="verify portable P13 evidence",
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
            sam2_weight_mode=args.sam2_weight_mode,
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
