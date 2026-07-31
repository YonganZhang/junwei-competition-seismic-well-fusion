#!/usr/bin/env python3
"""Train and evaluate the registered small real-multimodal lithofacies baseline."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.dataset_io import load_dataset  # noqa: E402
from _code.ml_framework.model_registry import get_model  # noqa: E402
from _code.ml_framework.preprocess import NormStats, denormalize  # noqa: E402
from _code.ml_framework.train import train_loop  # noqa: E402
from _code.ml_framework.visualize import plot_loss_curve  # noqa: E402
from pipeline_contract import (  # noqa: E402
    CLASS_NAMES,
    LOG_CHANNELS,
    PIPELINE_VERSION,
    assert_family_isolation,
    classification_metrics_from_confusion,
    validate_label_ids,
)

OUTPUT_DIR = TRACK_DIR / "_outputs" / "multimodal_mlp"


def project_relative(path: Path) -> str:
    """Serialize project-owned artifacts without a host/worktree prefix."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class LithofaciesSamples(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        if not samples:
            raise ValueError("数据集partition为空")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        well_log = torch.from_numpy(np.asarray(sample["well_log_seq"], dtype=np.float32))
        seismic = torch.from_numpy(np.asarray(sample["seismic_patch"], dtype=np.float32))
        label = torch.tensor(int(sample["label"]), dtype=torch.long)
        if not torch.isfinite(well_log).all() or not torch.isfinite(seismic).all():
            raise ValueError(f"样本{index}含NaN/Inf")
        return well_log, seismic, label


def assert_nonempty_loader(loader: DataLoader[Any], name: str) -> None:
    if len(loader) <= 0:
        raise RuntimeError(f"{name} DataLoader没有batch")


def _partition_saved_train(
    saved_train: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_train = [sample for sample in saved_train if sample["meta"]["partition"] == "train"]
    guard = [sample for sample in saved_train if sample["meta"]["partition"] == "guard"]
    if not model_train or not guard:
        raise ValueError("lithofacies/train必须同时含meta.partition=train和guard")
    return model_train, guard


def _validate_samples(samples: list[dict[str, Any]]) -> None:
    if not samples:
        raise ValueError("样本列表为空")
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    validate_label_ids(labels)
    reference_stats = samples[0]["meta"].get("normalization_stats")
    if reference_stats is None:
        raise ValueError("样本缺少训练井拟合的normalization_stats")
    for sample in samples:
        meta = sample["meta"]
        trace = meta.get("label_trace", {})
        if meta.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("样本pipeline_version不是冻结版本")
        if meta.get("normalization_fit_scope") != "train_mother_well_families_only":
            raise ValueError("归一化不是只在训练母井族fit")
        if meta.get("normalization_stats") != reference_stats:
            raise ValueError("不同样本携带了不同归一化统计")
        if trace.get("source") != "GM09" or trace.get("curve_type") != "GENETIC FACIES":
            raise ValueError("标签溯源不是GM09/GENETIC FACIES")
        if trace.get("class_name") in ("UNKNOWN", "UNDEFINED"):
            raise ValueError("UNKNOWN/UNDEFINED混入数据集")


def _histogram(samples: list[dict[str, Any]]) -> np.ndarray:
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    return np.bincount(labels, minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]


def _class_weights(histogram: np.ndarray) -> np.ndarray:
    if np.any(histogram == 0):
        raise ValueError(f"训练母井族未覆盖固定9类: missing={np.flatnonzero(histogram == 0).tolist()}")
    frequency = histogram / histogram.sum()
    weights = 1.0 / np.sqrt(frequency)
    return (weights / weights.mean()).astype(np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, list[int]]:
    model.eval()
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64, device=device)
    predictions: list[int] = []
    for well_log, seismic, labels in loader:
        logits = model(well_log.to(device), seismic.to(device))
        predicted = logits.argmax(dim=1)
        encoded = labels.to(device) * len(CLASS_NAMES) + predicted
        confusion += torch.bincount(
            encoded, minlength=len(CLASS_NAMES) ** 2
        ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        predictions.extend(int(value) for value in predicted.cpu())
    matrix = confusion.cpu().numpy()
    return classification_metrics_from_confusion(matrix), matrix, predictions


def _plot_confusion(matrix: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title("Held-out mother-family test confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(int(matrix[row, col])),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_real_predictions(
    samples: list[dict[str, Any]], predictions: list[int], path: Path
) -> Path:
    count = min(4, len(samples))
    indices = np.linspace(0, len(samples) - 1, count, dtype=int)
    fig, axes = plt.subplots(count, 4, figsize=(15, 3.2 * count), squeeze=False)
    for row, index in enumerate(indices):
        sample = samples[int(index)]
        prediction = predictions[int(index)]
        log_array = np.asarray(sample["well_log_seq"], dtype=np.float32)
        n_channels = len(LOG_CHANNELS)
        normalized_values = log_array[:n_channels]
        observed_mask = log_array[n_channels:]
        stats = sample["meta"]["normalization_stats"]
        seismic_stats = NormStats.from_dict(stats["seismic"])
        seismic_raw = denormalize(
            np.asarray(sample["seismic_patch"], dtype=np.float64), seismic_stats
        )

        axes[row, 0].imshow(normalized_values, aspect="auto", cmap="coolwarm")
        axes[row, 0].set_yticks(range(n_channels), LOG_CHANNELS, fontsize=6)
        axes[row, 0].set_title("Real log values (train-normalized)")
        axes[row, 0].set_xlabel("depth window")
        axes[row, 1].imshow(observed_mask, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
        axes[row, 1].set_yticks(range(n_channels), LOG_CHANNELS, fontsize=6)
        axes[row, 1].set_title("Observed-channel mask")
        axes[row, 1].set_xlabel("depth window")
        axes[row, 2].imshow(seismic_raw.reshape(-1, seismic_raw.shape[-1]), aspect="auto", cmap="gray")
        axes[row, 2].set_title("Real ST0202 patch (9 traces)")
        axes[row, 2].set_xlabel("time sample")
        axes[row, 2].set_ylabel("spatial trace")

        truth = int(sample["label"])
        trace = sample["meta"]["label_trace"]
        axes[row, 3].axis("off")
        axes[row, 3].text(
            0.02,
            0.75,
            f"well: {sample['position']['well_name']}\n"
            f"family: {sample['meta']['family_id']}\n"
            f"MD: {trace['top_md_m']:.1f}–{trace['base_md_m']:.1f} m",
            va="top",
            fontsize=10,
        )
        axes[row, 3].text(0.02, 0.34, f"GT: {CLASS_NAMES[truth]}", fontsize=11, color="green")
        axes[row, 3].text(
            0.02,
            0.18,
            f"Pred: {CLASS_NAMES[prediction]}",
            fontsize=11,
            color="green" if prediction == truth else "crimson",
        )
    fig.suptitle("Real best-checkpoint inputs, ground truth, and predictions", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但当前解释器没有可用CUDA")
    return device


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = _resolve_device(args.device)
    saved_train = list(load_dataset("lithofacies", "train"))
    model_train, guard = _partition_saved_train(saved_train)
    _validate_samples(model_train + guard)
    isolation_records = [
        {"partition": sample["meta"]["partition"], "family_id": sample["meta"]["family_id"]}
        for sample in model_train + guard
    ]

    train_loader = DataLoader(
        LithofaciesSamples(model_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    guard_loader = DataLoader(
        LithofaciesSamples(guard),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    assert_nonempty_loader(train_loader, "train")
    assert_nonempty_loader(guard_loader, "guard")
    first = model_train[0]
    well_log_shape = tuple(int(value) for value in first["well_log_seq"].shape)
    seismic_shape = tuple(int(value) for value in first["seismic_patch"].shape)
    model = get_model(
        args.model,
        models_package="models",
        num_classes=len(CLASS_NAMES),
        well_log_shape=well_log_shape,
        seismic_shape=seismic_shape,
        hidden_size=args.hidden_size,
    )
    if not isinstance(model, nn.Module):
        raise TypeError("动态注册模型没有返回torch.nn.Module")
    model = model.to(device)
    weights = torch.from_numpy(_class_weights(_histogram(model_train))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_progress = {"number": 0, "train": [], "guard": []}

    def train_step(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        model.train()
        well_log, seismic, label = batch
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(well_log.to(device), seismic.to(device)), label.to(device))
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        epoch_progress["train"].append(value)
        return value

    @torch.no_grad()
    def guard_step(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        model.eval()
        well_log, seismic, label = batch
        value = float(criterion(model(well_log.to(device), seismic.to(device)), label.to(device)))
        epoch_progress["guard"].append(value)
        if len(epoch_progress["guard"]) == len(guard_loader):
            epoch_progress["number"] += 1
            print(
                f"epoch {epoch_progress['number']:03d}/{args.epochs} "
                f"train={np.mean(epoch_progress['train']):.6f} "
                f"guard={np.mean(epoch_progress['guard']):.6f}",
                flush=True,
            )
            epoch_progress["train"].clear()
            epoch_progress["guard"].clear()
        return value

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT_DIR / "checkpoints"

    def save_checkpoint(model_to_save: nn.Module, path: Path) -> None:
        torch.save(
            {
                "pipeline_version": PIPELINE_VERSION,
                "model_name": args.model,
                "state_dict": model_to_save.state_dict(),
                "num_classes": len(CLASS_NAMES),
                "class_names": CLASS_NAMES,
                "well_log_shape": well_log_shape,
                "seismic_shape": seismic_shape,
                "hidden_size": args.hidden_size,
            },
            path,
        )

    start = time.monotonic()
    history = train_loop(
        model=model,
        train_step_fn=train_step,
        val_step_fn=guard_step,
        train_batches_fn=lambda: train_loader,
        val_batches_fn=lambda: guard_loader,
        epochs=args.epochs,
        save_checkpoint_fn=save_checkpoint,
        checkpoint_dir=checkpoint_dir,
        min_epochs_before_early_check=10,
    )
    plot_loss_curve(history, OUTPUT_DIR / "loss_curve.png")

    best = torch.load(checkpoint_dir / "best.ckpt", map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    test_samples = list(load_dataset("lithofacies", "test"))
    _validate_samples(test_samples)
    isolation_records.extend(
        {"partition": "test", "family_id": sample["meta"]["family_id"]}
        for sample in test_samples
    )
    isolation = assert_family_isolation(isolation_records)
    test_loader = DataLoader(
        LithofaciesSamples(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    assert_nonempty_loader(test_loader, "test")
    metrics, confusion, predictions = evaluate(model, test_loader, device)
    per_well: dict[str, Any] = {}
    for well_id in sorted({sample["position"]["well_name"] for sample in test_samples}):
        indices = [
            index
            for index, sample in enumerate(test_samples)
            if sample["position"]["well_name"] == well_id
        ]
        matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        for index in indices:
            matrix[int(test_samples[index]["label"]), predictions[index]] += 1
        per_well[well_id] = classification_metrics_from_confusion(matrix)

    _plot_confusion(confusion, OUTPUT_DIR / "confusion_matrix.png")
    _plot_real_predictions(test_samples, predictions, OUTPUT_DIR / "best_checkpoint_predictions.png")
    result: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "model_name": args.model,
        "architecture": type(model).__name__,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": history.best_epoch + 1,
        "best_guard_loss": history.best_val_loss,
        "elapsed_seconds": time.monotonic() - start,
        "sample_counts": {
            "train": len(model_train),
            "guard": len(guard),
            "test": len(test_samples),
        },
        "class_support": {
            "train": _histogram(model_train).tolist(),
            "guard": _histogram(guard).tolist(),
            "test": _histogram(test_samples).tolist(),
        },
        "family_isolation": isolation,
        "metrics": metrics,
        "per_well_test_metrics": per_well,
        "artifacts": {
            "best_checkpoint": project_relative(checkpoint_dir / "best.ckpt"),
            "last_checkpoint": project_relative(checkpoint_dir / "last.ckpt"),
            "history": project_relative(checkpoint_dir / "history.json"),
            "loss_curve": project_relative(OUTPUT_DIR / "loss_curve.png"),
            "confusion_matrix": project_relative(OUTPUT_DIR / "confusion_matrix.png"),
            "predictions": project_relative(OUTPUT_DIR / "best_checkpoint_predictions.png"),
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_manifest = {
        "command": ["python3", project_relative(Path(__file__)), *sys.argv[1:]],
        "test_loaded_after_best_checkpoint_selection": True,
        "zero_argument_batch_factories": True,
        "shared_train_loop": "_code.ml_framework.train.train_loop",
        "result": result,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="multimodal_mlp")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2693)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, learning-rate must be positive")
    return args


if __name__ == "__main__":
    output = train(parse_args())
    print(json.dumps({
        "best_epoch": output["best_epoch"],
        "best_guard_loss": output["best_guard_loss"],
        "metrics": output["metrics"],
    }, ensure_ascii=False, indent=2))
