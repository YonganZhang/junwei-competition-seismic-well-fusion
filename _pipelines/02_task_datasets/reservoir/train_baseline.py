#!/usr/bin/env python3
"""Train/evaluate the registered small real-data multimodal baseline."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from data_pipeline import (  # noqa: E402
    INPUT_CHANNELS,
    TARGET_NAMES,
    inverse_targets,
    load_guard,
)
from dataset_io import load_dataset  # noqa: E402
from ml_framework.model_registry import get_model  # noqa: E402
from ml_framework.preprocess import (  # noqa: E402
    NormStats,
    denoise_identity,
    denormalize,
    fit_zscore,
    normalize,
)
from ml_framework.train import train_loop  # noqa: E402
from ml_framework.visualize import plot_loss_curve  # noqa: E402


OUTPUT_DIR = HERE / "_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
NORMALIZATION_PATH = OUTPUT_DIR / "normalization.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"


def _stack(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise RuntimeError("split为空，不能组成batch")
    seismic = denoise_identity(np.stack([sample["seismic_patch"] for sample in samples]).astype(np.float64))
    logs = denoise_identity(np.stack([sample["well_log_seq"] for sample in samples]).astype(np.float64))
    targets = np.stack([sample["label"] for sample in samples]).astype(np.float64)
    if not np.isfinite(seismic).all() or not np.isfinite(logs).all() or not np.isfinite(targets).all():
        raise RuntimeError("输入或标签含NaN/Inf")
    if logs.shape[2] != len(INPUT_CHANNELS) * 2 or targets.shape[1] != len(TARGET_NAMES):
        raise RuntimeError(f"schema异常: logs={logs.shape}, targets={targets.shape}")
    return seismic, logs, targets


def fit_train_statistics(
    seismic: np.ndarray, logs: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    """The only stats-fitting entrypoint; callers pass train arrays only."""
    seismic_stats = fit_zscore(seismic.reshape(-1))
    n_channels = len(INPUT_CHANNELS)
    log_stats: list[NormStats] = []
    values = logs[..., :n_channels]
    masks = logs[..., n_channels:]
    for channel in range(n_channels):
        observed = values[..., channel][masks[..., channel] > 0.5]
        if observed.size == 0:
            raise RuntimeError(f"训练井中{INPUT_CHANNELS[channel]}完全缺失")
        log_stats.append(fit_zscore(observed))
    target_stats = [fit_zscore(targets[:, i]) for i in range(targets.shape[1])]
    return {
        "seismic": seismic_stats,
        "logs": log_stats,
        "targets": target_stats,
    }


def apply_statistics(
    seismic: np.ndarray,
    logs: np.ndarray,
    targets: np.ndarray,
    stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    seismic_norm = normalize(seismic, stats["seismic"])
    n_channels = len(INPUT_CHANNELS)
    values = logs[..., :n_channels].copy()
    masks = logs[..., n_channels:].copy()
    for channel, channel_stats in enumerate(stats["logs"]):
        observed = masks[..., channel] > 0.5
        values[..., channel][observed] = normalize(values[..., channel][observed], channel_stats)
        values[..., channel][~observed] = 0.0
    target_norm = np.column_stack([
        normalize(targets[:, i], stats["targets"][i]) for i in range(targets.shape[1])
    ])
    features = np.concatenate([
        seismic_norm.reshape(len(seismic_norm), -1),
        values.reshape(len(values), -1),
        masks.reshape(len(masks), -1),
    ], axis=1)
    if not np.isfinite(features).all() or not np.isfinite(target_norm).all():
        raise RuntimeError("归一化产生NaN/Inf")
    return features, target_norm


def inverse_normalized_targets(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return np.column_stack([
        denormalize(values[:, i], stats["targets"][i]) for i in range(values.shape[1])
    ])


def _stats_to_json(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "fit_source": "train families only",
        "seismic": stats["seismic"].to_dict(),
        "logs": {name: value.to_dict() for name, value in zip(INPUT_CHANNELS, stats["logs"])},
        "targets": {name: value.to_dict() for name, value in zip(TARGET_NAMES, stats["targets"])},
    }


def make_batches_factory(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Callable[[], list[tuple[np.ndarray, np.ndarray]]]:
    if len(features) == 0 or len(features) != len(targets):
        raise ValueError("非空且等长的features/targets才可生成batch")
    call_count = 0

    def factory() -> list[tuple[np.ndarray, np.ndarray]]:
        nonlocal call_count
        indices = np.arange(len(features))
        if shuffle:
            np.random.default_rng(seed + call_count).shuffle(indices)
        call_count += 1
        return [
            (features[batch_indices], targets[batch_indices])
            for batch_indices in (
                indices[start:start + batch_size] for start in range(0, len(indices), batch_size)
            )
        ]

    return factory


def _metric_vector(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    if len(actual) < 2 or np.var(actual) == 0:
        r2 = None
        r2_reason = "undefined: fewer than 2 values or zero target variance"
        pearson = None
        pearson_reason = r2_reason
    else:
        r2 = float(1.0 - np.sum(error ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        if np.std(predicted) == 0:
            pearson = None
            pearson_reason = "undefined: prediction variance is zero"
        else:
            pearson = float(np.corrcoef(actual, predicted)[0, 1])
            pearson_reason = None
        r2_reason = None
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "R2_reason": r2_reason,
        "Pearson": pearson,
        "Pearson_reason": pearson_reason,
    }


def compute_metrics(
    actual: np.ndarray, predicted: np.ndarray, stats: dict[str, Any]
) -> dict[str, Any]:
    per_target = {
        name: _metric_vector(actual[:, i], predicted[:, i])
        for i, name in enumerate(TARGET_NAMES)
    }
    normalized_rmse = [
        per_target[name]["RMSE"] / float(stats["targets"][i].std)
        for i, name in enumerate(TARGET_NAMES)
    ]
    result = {
        "space": "PHIF, log1p(KLOGH), SW",
        "per_target": per_target,
        "composite_mean_train_std_normalized_RMSE": float(np.mean(normalized_rmse)),
    }
    numeric_values = [
        value
        for metrics in per_target.values()
        for key, value in metrics.items()
        if key in {"RMSE", "MAE", "R2", "Pearson"} and value is not None
    ] + [result["composite_mean_train_std_normalized_RMSE"]]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError(f"评估指标出现非有限值: {result}")
    return result


def _plot_predictions(
    samples: list[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray, out_path: Path
) -> None:
    depths = np.asarray([sample["meta"]["depth_m"] for sample in samples])
    order = np.argsort(depths)
    actual_physical = inverse_targets(actual)
    predicted_physical = inverse_targets(predicted)
    names = ("PHIF", "KLOGH (mD)", "SW")
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True)
    for i, axis in enumerate(axes):
        # Horizontal wells can cross the Hugin interval several times.  Points
        # are intentionally not connected across those disjoint intervals.
        axis.scatter(actual_physical[order, i], depths[order], label="GT", s=10, alpha=0.75)
        axis.scatter(
            predicted_physical[order, i], depths[order], label="prediction", s=10, alpha=0.65
        )
        axis.set_xlabel(names[i])
        axis.grid(alpha=0.25)
        axis.invert_yaxis()
    axes[0].set_ylabel("Measured depth (m)")
    axes[0].legend()
    fig.suptitle(f"Held-out family: {samples[0]['meta']['family_id']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_seismic_input(sample: dict[str, Any], out_path: Path) -> None:
    patch = np.asarray(sample["seismic_patch"])
    center_time = patch.shape[2] // 2
    center_space = patch.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    image = axes[0].imshow(patch[:, :, center_time], cmap="seismic", aspect="equal")
    axes[0].set_title("Real ST0202 spatial slice")
    fig.colorbar(image, ax=axes[0], shrink=0.8)
    axes[1].plot(np.arange(patch.shape[2]) - center_time, patch[center_space, center_space])
    axes[1].set_title("Central real seismic trace window")
    axes[1].set_xlabel("Sample offset (4 ms)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_training(epochs: int = 60, batch_size: int = 64, model_name: str = "tiny_mlp") -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_samples = list(load_dataset("reservoir", "train"))
    guard_samples = load_guard()
    train_families = {sample["meta"]["family_id"] for sample in train_samples}
    guard_families = {sample["meta"]["family_id"] for sample in guard_samples}
    if train_families & guard_families or not guard_families:
        raise RuntimeError("train/guard井族隔离失败")

    train_seismic, train_logs, train_targets = _stack(train_samples)
    guard_seismic, guard_logs, guard_targets = _stack(guard_samples)
    stats = fit_train_statistics(train_seismic, train_logs, train_targets)
    train_features, train_target_norm = apply_statistics(train_seismic, train_logs, train_targets, stats)
    guard_features, guard_target_norm = apply_statistics(guard_seismic, guard_logs, guard_targets, stats)

    roundtrip = inverse_normalized_targets(train_target_norm, stats)
    normalization_error = float(np.max(np.abs(roundtrip - train_targets)))
    physical_roundtrip = inverse_targets(train_targets)
    transform_error = float(np.max(np.abs(np.log1p(physical_roundtrip[:, 1]) - train_targets[:, 1])))
    if normalization_error > 1e-8 or transform_error > 1e-8:
        raise RuntimeError(
            f"变换不可逆: normalization={normalization_error}, log1p={transform_error}"
        )
    normalization_json = _stats_to_json(stats)
    normalization_json["max_normalization_roundtrip_error"] = normalization_error
    normalization_json["max_log1p_roundtrip_error"] = transform_error
    NORMALIZATION_PATH.write_text(json.dumps(normalization_json, indent=2, ensure_ascii=False))

    model = get_model(
        model_name,
        models_package="models",
        n_features=train_features.shape[1],
        n_outputs=3,
        hidden_dim=24,
        learning_rate=0.002,
        seed=2693,
    )
    history = train_loop(
        model=model,
        train_step_fn=model.train_batch,
        val_step_fn=model.validation_loss,
        train_batches_fn=make_batches_factory(train_features, train_target_norm, batch_size, True, 2693),
        val_batches_fn=make_batches_factory(guard_features, guard_target_norm, batch_size, False, 2693),
        epochs=epochs,
        save_checkpoint_fn=lambda current, path: current.save_checkpoint(path),
        checkpoint_dir=CHECKPOINT_DIR,
    )
    plot_loss_curve(history, OUTPUT_DIR / "loss_curve.png")
    model.load_checkpoint(CHECKPOINT_DIR / "best.ckpt")

    # Test is intentionally loaded only after training and best-checkpoint selection.
    test_samples = list(load_dataset("reservoir", "test"))
    test_families = {sample["meta"]["family_id"] for sample in test_samples}
    if train_families & test_families or guard_families & test_families:
        raise RuntimeError("train/guard/test井族隔离失败")
    test_seismic, test_logs, test_targets = _stack(test_samples)
    test_features, _test_target_norm_unused = apply_statistics(test_seismic, test_logs, test_targets, stats)
    prediction_norm = model.predict(test_features)
    prediction = inverse_normalized_targets(prediction_norm, stats)
    metrics = compute_metrics(test_targets, prediction, stats)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "well_id", "family_id", "depth_m",
            "PHIF_gt", "PHIF_pred", "log1p_KLOGH_gt", "log1p_KLOGH_pred", "SW_gt", "SW_pred",
        ])
        for sample, actual, pred in zip(test_samples, test_targets, prediction):
            writer.writerow([
                sample["meta"]["well_id"], sample["meta"]["family_id"], sample["meta"]["depth_m"],
                actual[0], pred[0], actual[1], pred[1], actual[2], pred[2],
            ])

    _plot_predictions(test_samples, test_targets, prediction, OUTPUT_DIR / "test_depth_gt_vs_pred.png")
    _plot_seismic_input(test_samples[len(test_samples) // 2], OUTPUT_DIR / "real_seismic_input.png")
    manifest = {
        "model": model_name,
        "framework": "NumPy small one-hidden-layer MLP via shared ml_framework.train_loop",
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "sample_counts": {
            "train": len(train_samples), "guard": len(guard_samples), "test": len(test_samples)
        },
        "families": {
            "train": sorted(train_families), "guard": sorted(guard_families), "test": sorted(test_families)
        },
        "family_zero_overlap": True,
        "normalization_fit_sources": ["train"],
        "guard_used_for_val_loss_only": True,
        "test_loaded_after_best_checkpoint": True,
        "test_used_in_training_or_statistics": False,
        "denoise": "denoise_identity",
        "target_transform": "[PHIF, log1p(KLOGH), SW] with expm1 inverse",
    }
    RUN_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"manifest": manifest, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="tiny_mlp")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    print(json.dumps(run_training(args.epochs, args.batch_size, args.model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
