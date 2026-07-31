#!/usr/bin/env python3
"""Train registered facies models through the shared ml_framework skeleton."""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.dataset_io import load_dataset  # noqa: E402
from _code.ml_framework.model_registry import get_model  # noqa: E402
from _code.ml_framework.train import train_loop  # noqa: E402
from _code.ml_framework.visualize import plot_loss_curve  # noqa: E402
from pipeline_contract import (  # noqa: E402
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_VALIDATION_GUARD_FRACTION,
    PIPELINE_VERSION,
    TASK_SCHEMAS,
    get_task_schema,
    ordered_spatial_split,
    segmentation_metrics_from_confusion,
    validate_label_array,
)

TASKS = tuple(TASK_SCHEMAS)
DEFAULT_EPOCHS = {"facies_f3": 40, "facies_penobscot": 120}
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / PIPELINE_VERSION


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class FaciesSamples(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Thin PyTorch wrapper around samples loaded through dataset_io."""

    def __init__(self, samples: list[dict[str, Any]], augment: bool) -> None:
        if not samples:
            raise ValueError("dataset split contains no samples")
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = torch.from_numpy(
            np.asarray(sample["seismic_patch"], dtype=np.float32)
        ).unsqueeze(0)
        label = torch.from_numpy(np.asarray(sample["label"], dtype=np.int64))
        if image.shape[1:] != label.shape:
            raise ValueError(
                f"unaligned sample {index}: image={image.shape}, label={label.shape}"
            )
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
                label = torch.flip(label, dims=(1,))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(1,))
                label = torch.flip(label, dims=(0,))
        return image, label


def split_train_validation(
    samples: list[dict[str, Any]],
    validation_fraction: float,
    validation_guard_fraction: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[int],
    set[int],
    set[int],
]:
    """Reserve ordered inline guard and validation ranges inside saved train."""
    inline_numbers = sorted({int(sample["position"]["inline"]) for sample in samples})
    model_train_lines, guard_lines, validation_lines = ordered_spatial_split(
        inline_numbers, validation_fraction, validation_guard_fraction
    )
    model_train = [
        sample
        for sample in samples
        if int(sample["position"]["inline"]) in model_train_lines
    ]
    validation = [
        sample
        for sample in samples
        if int(sample["position"]["inline"]) in validation_lines
    ]
    guard = [
        sample
        for sample in samples
        if int(sample["position"]["inline"]) in guard_lines
    ]
    if not model_train or not guard or not validation:
        raise ValueError("invalid train/validation inline partition")
    return (
        model_train,
        guard,
        validation,
        model_train_lines,
        guard_lines,
        validation_lines,
    )


def label_histogram(samples: list[dict[str, Any]], task: str) -> np.ndarray:
    schema = get_task_schema(task)
    histogram = np.zeros(schema.num_classes, dtype=np.int64)
    for sample in samples:
        label = np.asarray(sample["label"])
        validate_label_array(label, schema)
        histogram += np.bincount(
            label.reshape(-1), minlength=schema.num_classes
        )[: schema.num_classes]
    return histogram


def validate_normalization_contract(
    samples: list[dict[str, Any]], expected_stats: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot validate normalization on an empty split")
    reference = samples[0]["meta"].get("normalization_stats")
    if reference is None:
        raise ValueError("sample is missing train-fitted normalization_stats")
    if expected_stats is not None and reference != expected_stats:
        raise ValueError("split normalization stats differ from model-train stats")
    for sample in samples:
        meta = sample["meta"]
        if meta.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("processed sample does not use the leakage-fixed pipeline version")
        if meta.get("normalization_fit_scope") != "model_train_inline_only":
            raise ValueError("normalization was not fitted exclusively on model-train inlines")
        if meta.get("normalization_stats") != reference:
            raise ValueError("normalization stats vary between samples")
    return reference


def class_weights(histogram: np.ndarray) -> np.ndarray:
    """Inverse-square-root frequency weights derived from model-train pixels only."""
    if np.any(histogram == 0):
        missing = np.flatnonzero(histogram == 0).tolist()
        raise ValueError(f"model-train partition has no pixels for classes {missing}")
    frequencies = histogram / histogram.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights /= weights.mean()
    return np.clip(weights, 0.2, 5.0).astype(np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        encoded = labels.reshape(-1) * num_classes + predictions.reshape(-1)
        confusion += torch.bincount(
            encoded, minlength=num_classes**2
        ).reshape(num_classes, num_classes)

    matrix = confusion.cpu().numpy()
    metrics = segmentation_metrics_from_confusion(matrix)
    return metrics, matrix


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def assert_nonempty_loader(loader: DataLoader[Any], split_name: str) -> None:
    if len(loader) <= 0:
        raise RuntimeError(f"{split_name} DataLoader has zero batches")


def train_task(
    task: str,
    *,
    model_name: str,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    base_channels: int,
    validation_fraction: float,
    validation_guard_fraction: float,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    schema = get_task_schema(task)
    num_classes = schema.num_classes
    print(f"[{task}] loading saved train through dataset_io", flush=True)
    full_train_samples = list(load_dataset(task, "train"))
    (
        model_train_samples,
        validation_guard_samples,
        validation_samples,
        model_train_lines,
        validation_guard_lines,
        validation_lines,
    ) = split_train_validation(
        full_train_samples, validation_fraction, validation_guard_fraction
    )
    normalization_stats = validate_normalization_contract(full_train_samples)
    train_hist = label_histogram(model_train_samples, task)
    validation_guard_hist = label_histogram(validation_guard_samples, task)
    validation_hist = label_histogram(validation_samples, task)
    weights_np = class_weights(train_hist)

    train_loader = DataLoader(
        FaciesSamples(model_train_samples, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        FaciesSamples(validation_samples, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    assert_nonempty_loader(train_loader, "train")
    assert_nonempty_loader(validation_loader, "validation")

    model = get_model(
        model_name,
        models_package="models",
        num_classes=num_classes,
        base_channels=base_channels,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"registered model {model_name!r} did not return torch.nn.Module")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_np).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    progress = {
        "epoch": 0,
        "train_sum": 0.0,
        "train_count": 0,
        "val_sum": 0.0,
        "val_count": 0,
    }

    def train_step(batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        model.train()
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach())
        progress["train_sum"] += loss_value
        progress["train_count"] += 1
        return loss_value

    @torch.no_grad()
    def validation_step(batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        model.eval()
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss_value = float(criterion(model(images), labels))
        progress["val_sum"] += loss_value
        progress["val_count"] += 1
        if progress["val_count"] == len(validation_loader):
            progress["epoch"] += 1
            print(
                f"[{task}/{model_name}] epoch {progress['epoch']}/{epochs} "
                f"train={progress['train_sum'] / progress['train_count']:.6f} "
                f"val={progress['val_sum'] / progress['val_count']:.6f}",
                flush=True,
            )
            progress["train_sum"] = 0.0
            progress["train_count"] = 0
            progress["val_sum"] = 0.0
            progress["val_count"] = 0
        return loss_value

    task_output = output_dir / task / model_name
    checkpoint_metadata = {
        "task": task,
        "pipeline_version": PIPELINE_VERSION,
        "model_name": model_name,
        "num_classes": num_classes,
        "model_kwargs": {"base_channels": base_channels},
    }

    def save_checkpoint(model_to_save: nn.Module, path: Path) -> None:
        torch.save(
            {**checkpoint_metadata, "state_dict": model_to_save.state_dict()},
            path,
        )

    start_time = time.monotonic()
    history = train_loop(
        model=model,
        train_step_fn=train_step,
        val_step_fn=validation_step,
        train_batches_fn=lambda: train_loader,
        val_batches_fn=lambda: validation_loader,
        epochs=epochs,
        save_checkpoint_fn=save_checkpoint,
        checkpoint_dir=task_output,
        min_epochs_before_early_check=10,
    )
    loss_curve_path = plot_loss_curve(history, task_output / "loss_curve.png")

    best_checkpoint_path = task_output / "best.ckpt"
    best_checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["state_dict"])

    # Test labels are intentionally not loaded until model selection is complete.
    print(f"[{task}] loading held-out test after best checkpoint selection", flush=True)
    test_samples = list(load_dataset(task, "test"))
    validate_normalization_contract(test_samples, normalization_stats)
    test_hist = label_histogram(test_samples, task)
    test_loader = DataLoader(
        FaciesSamples(test_samples, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    assert_nonempty_loader(test_loader, "test")
    metrics, confusion = evaluate(model, test_loader, device, num_classes)

    best_epoch_number = history.best_epoch + 1
    post_best_val = history.val_loss[best_epoch_number:]
    overfit_turn_observed = bool(
        post_best_val
        and max(post_best_val) > history.best_val_loss * 1.01
    )
    result: dict[str, Any] = {
        "task": task,
        "pipeline_version": PIPELINE_VERSION,
        "model_name": model_name,
        "architecture": type(model).__name__,
        "device": str(device),
        "num_classes": num_classes,
        "full_train_samples": len(full_train_samples),
        "model_train_samples": len(model_train_samples),
        "validation_guard_samples": len(validation_guard_samples),
        "validation_samples": len(validation_samples),
        "test_samples": len(test_samples),
        "model_train_inline_range": [min(model_train_lines), max(model_train_lines)],
        "validation_guard_inline_range": [
            min(validation_guard_lines),
            max(validation_guard_lines),
        ],
        "validation_inline_range": [min(validation_lines), max(validation_lines)],
        "model_train_label_histogram": train_hist.tolist(),
        "validation_guard_label_histogram": validation_guard_hist.tolist(),
        "validation_label_histogram": validation_hist.tolist(),
        "test_label_histogram": test_hist.tolist(),
        "class_weights": weights_np.tolist(),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "base_channels": base_channels,
            "validation_fraction": validation_fraction,
            "validation_guard_fraction": validation_guard_fraction,
            "seed": seed,
            "augmentation": "random horizontal and vertical flips",
        },
        "history": history.to_dict(),
        "best_epoch_number": best_epoch_number,
        "overfit_turn_observed": overfit_turn_observed,
        "test_metrics_from_best_checkpoint": metrics,
        "confusion_matrix": confusion.tolist(),
        "elapsed_seconds": time.monotonic() - start_time,
        "best_checkpoint": project_relative(best_checkpoint_path),
        "last_checkpoint": project_relative(task_output / "last.ckpt"),
        "loss_curve": project_relative(loss_curve_path),
        "label_schema": {
            "valid_label_ids": list(schema.valid_label_ids),
            "ignore_index": schema.ignore_index,
            "source": schema.source,
        },
        "normalization_stats": normalization_stats,
        "reproducibility": {
            "git_head": git_head(),
            "command": [
                "python3",
                project_relative(Path(sys.argv[0])),
                *sys.argv[1:],
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }
    metrics_path = task_output / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[{task}/{model_name}] best_epoch={best_epoch_number} "
        f"test accuracy={metrics['accuracy']:.6f} "
        f"mIoU={metrics['miou']:.6f} macro-F1={metrics['macro_f1']:.6f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument(
        "--model",
        default="small_unet",
        help="model name dynamically discovered from models/<name>.py",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override task defaults (F3=40, Penobscot=120)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument(
        "--validation-guard-fraction",
        type=float,
        default=DEFAULT_VALIDATION_GUARD_FRACTION,
    )
    parser.add_argument("--seed", type=int, default=2693)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0 or args.base_channels <= 0:
        raise ValueError("batch-size and base-channels must be positive")
    device = resolve_device(args.device)
    selected_tasks = TASKS if args.task == "all" else (args.task,)
    results = [
        train_task(
            task,
            model_name=args.model,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs if args.epochs is not None else DEFAULT_EPOCHS[task],
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            validation_fraction=args.validation_fraction,
            validation_guard_fraction=args.validation_guard_fraction,
            seed=args.seed,
        )
        for task in selected_tasks
    ]
    summary = {
        result["task"]: result["test_metrics_from_best_checkpoint"]
        for result in results
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
