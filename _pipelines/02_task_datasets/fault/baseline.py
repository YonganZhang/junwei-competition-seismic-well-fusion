#!/usr/bin/env python3
"""Train a registered fault model through the shared ml_framework skeleton."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_DIR = Path(__file__).resolve().parent
BUILD_SUMMARY_NAME = "build_summary.json"

sys.path.insert(0, str(PROJECT_ROOT / "_code"))
from dataset_io import load_dataset  # noqa: E402
from ml_framework.model_registry import get_model  # noqa: E402
from ml_framework.preprocess import NormStats, denormalize, normalize  # noqa: E402
from ml_framework.train import train_loop  # noqa: E402
from ml_framework.visualize import plot_loss_curve  # noqa: E402
from audit_utils import (  # noqa: E402
    sha256_file,
    software_versions,
    validated_run_dir,
    verify_historical_artifacts_if_present,
)
from split_utils import validation_masks  # noqa: E402

@dataclass(frozen=True)
class SampleSet:
    patches: np.ndarray
    labels: np.ndarray
    positions: list[dict]
    roundtrip_max_abs_error: float
    normalization_stats: dict | None = None


class ShuffledBatches:
    """Reusable iterable: ml_framework.train_loop asks for it once per epoch."""

    def __init__(
        self,
        samples: SampleSet,
        weights: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.weights = weights
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        order = self.rng.permutation(len(self.samples.patches))
        for start in range(0, len(order), self.batch_size):
            selected = order[start : start + self.batch_size]
            yield (
                self.samples.patches[selected],
                self.samples.labels[selected],
                self.weights[selected],
            )


def fixed_batches(
    samples: SampleSet, weights: np.ndarray, batch_size: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            samples.patches[start : start + batch_size],
            samples.labels[start : start + batch_size],
            weights[start : start + batch_size],
        )
        for start in range(0, len(samples.patches), batch_size)
    ]


def _verify_persisted_roundtrip(patch: np.ndarray, meta: dict) -> float:
    stats_raw = meta.get("normalization")
    if not isinstance(stats_raw, dict):
        raise ValueError(
            "fault dataset predates the shared normalization contract; rerun build_dataset.py"
        )
    stats = NormStats.from_dict(stats_raw)
    physical = denormalize(patch, stats)
    renormalized = normalize(physical, stats)
    error = float(np.max(np.abs(renormalized - patch)))
    if error > 2e-6:
        raise AssertionError(f"persisted normalization round-trip error is {error}, expected <= 2e-6")
    return error


def load_samples(split: str) -> SampleSet:
    patches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positions: list[dict] = []
    errors: list[float] = []
    expected_patch_shape: tuple[int, ...] | None = None
    expected_label_shape: tuple[int, ...] | None = None
    expected_stats: dict | None = None
    for sample in load_dataset("fault", split):
        patch = np.asarray(sample["seismic_patch"], dtype=np.float32)
        label = np.asarray(sample["label"], dtype=np.uint8)
        if patch.ndim != 3 or patch.shape[0] != 1:
            raise ValueError(f"expected seismic patch [1,H,W], received {patch.shape}")
        if patch.shape[1:] != label.shape:
            raise ValueError(f"patch/label shape mismatch in {split}: {patch.shape} vs {label.shape}")
        if expected_patch_shape is None:
            expected_patch_shape, expected_label_shape = patch.shape, label.shape
        elif patch.shape != expected_patch_shape or label.shape != expected_label_shape:
            raise ValueError(f"inconsistent sample shapes in fault/{split}")
        meta = sample["meta"]
        if meta.get("normalization_fit_split") != "train_fit":
            raise ValueError(
                f"fault/{split} normalization was not explicitly fitted on train-fit; "
                "rerun the audited build_dataset.py"
            )
        stats_raw = meta.get("normalization")
        if expected_stats is None:
            expected_stats = stats_raw
        elif stats_raw != expected_stats:
            raise ValueError(f"fault/{split} contains inconsistent normalization statistics")
        if "time_index" not in sample["position"]:
            raise ValueError(f"fault/{split} sample is missing position.time_index")
        errors.append(_verify_persisted_roundtrip(patch, meta))
        patches.append(patch)
        labels.append(label)
        positions.append(sample["position"])
    if not patches:
        raise ValueError(f"fault/{split} contains no samples")
    return SampleSet(
        patches=np.stack(patches),
        labels=np.stack(labels),
        positions=positions,
        roundtrip_max_abs_error=max(errors),
        normalization_stats=expected_stats,
    )


def subset(samples: SampleSet, selected: np.ndarray) -> SampleSet:
    indices = np.flatnonzero(selected)
    return SampleSet(
        patches=samples.patches[indices],
        labels=samples.labels[indices],
        positions=[samples.positions[int(i)] for i in indices],
        roundtrip_max_abs_error=samples.roundtrip_max_abs_error,
        normalization_stats=samples.normalization_stats,
    )


def split_train_validation(
    samples: SampleSet, val_fraction: float, guard_inlines: int
) -> tuple[SampleSet, SampleSet, int, list[int]]:
    inlines = np.asarray([int(position["inline"]) for position in samples.positions], dtype=np.int32)
    fit_mask, _, validation_mask, plan = validation_masks(inlines, val_fraction, guard_inlines)
    fit = subset(samples, fit_mask)
    validation = subset(samples, validation_mask)
    if fit.labels.sum() == 0 or validation.labels.sum() == 0:
        raise ValueError("fit and validation splits must both contain positive fault voxels")
    guard = list(plan.guard_sampled_inlines)
    if set(int(position["inline"]) for position in fit.positions) & set(guard):
        raise AssertionError("fit/validation guard leakage detected")
    return fit, validation, plan.val_start_inline, guard


def class_weights(labels: np.ndarray) -> dict[int, float]:
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"both classes are required, observed counts={counts.tolist()}")
    total = float(len(flat))
    return {0: total / (2.0 * float(counts[0])), 1: total / (2.0 * float(counts[1]))}


def voxel_weights(labels: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    return np.where(labels == 1, weights[1], weights[0]).astype(np.float32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "f1": f1,
    }


def aggregate_physical_voxels(
    samples: SampleSet, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Average repeated patch predictions at one physical voxel before scoring."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != samples.labels.shape:
        raise ValueError(f"probability/label shape mismatch: {probs.shape} vs {samples.labels.shape}")
    accumulated: dict[tuple[int, int, int], list[float | int]] = {}
    for sample_index, position in enumerate(samples.positions):
        inline = int(position["inline"])
        centre_crossline = int(position["crossline"])
        centre_time = int(position["time_index"])
        height, width = samples.labels[sample_index].shape
        xl0 = centre_crossline - height // 2
        t0 = centre_time - width // 2
        for row in range(height):
            for col in range(width):
                key = (inline, xl0 + row, t0 + col)
                label = int(samples.labels[sample_index, row, col])
                probability = float(probs[sample_index, row, col])
                if key not in accumulated:
                    accumulated[key] = [label, probability, 1]
                    continue
                existing = accumulated[key]
                if int(existing[0]) != label:
                    raise AssertionError(f"inconsistent labels for repeated physical voxel {key}")
                existing[1] = float(existing[1]) + probability
                existing[2] = int(existing[2]) + 1
    ordered = sorted(accumulated)
    truth = np.asarray([int(accumulated[key][0]) for key in ordered], dtype=np.uint8)
    mean_probabilities = np.asarray(
        [float(accumulated[key][1]) / int(accumulated[key][2]) for key in ordered],
        dtype=np.float64,
    )
    counts = np.asarray([int(accumulated[key][2]) for key in ordered], dtype=np.int32)
    audit = {
        "patch_voxel_observations": int(samples.labels.size),
        "unique_physical_voxels": int(len(ordered)),
        "repeated_physical_voxels": int(np.sum(counts > 1)),
        "max_coverage_multiplicity": int(counts.max()),
    }
    return truth, mean_probabilities, audit


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.uint8)
    probs = np.asarray(probabilities, dtype=np.float64)
    precision, recall, _ = precision_recall_curve(truth, probs)
    result = {
        "average_precision": float(average_precision_score(truth, probs)),
        "pr_auc": float(auc(recall[::-1], precision[::-1])),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"non-finite probability metric: {result}")
    return result


def select_validation_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, requested: str
) -> tuple[float, str, float]:
    if requested != "auto":
        threshold = float(requested)
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be 'auto' or a float strictly between 0 and 1")
        metrics = binary_metrics(y_true, probabilities >= threshold)
        return threshold, "fixed_cli", float(metrics["f1"])
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        raise ValueError("validation probabilities do not provide a usable threshold")
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), "validation_max_f1", float(f1[best])


def assert_finite_metrics(metrics: dict[str, float | int]) -> None:
    bad = {key: value for key, value in metrics.items() if isinstance(value, float) and not math.isfinite(value)}
    if bad:
        raise RuntimeError(f"non-finite metrics: {bad}")


def save_checkpoint(model: object, path: Path) -> None:
    joblib.dump(model, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fault_local_logistic")
    parser.add_argument("--seed", type=int, default=2693)
    parser.add_argument("--threshold", default="auto")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--val-guard-inlines", type=int, default=2)
    parser.add_argument("--run-name", default="audited_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_artifacts = verify_historical_artifacts_if_present()
    run_dir = validated_run_dir(args.run_name)
    build_summary_path = run_dir / BUILD_SUMMARY_NAME
    if not build_summary_path.exists():
        raise FileNotFoundError(
            f"audited build summary is missing: {build_summary_path}; run build_dataset.py first"
        )
    checkpoint_dir = run_dir / "checkpoints"
    model_path = run_dir / "baseline_model.joblib"
    metrics_path = run_dir / "baseline_metrics.json"
    loss_curve_path = run_dir / "loss_curve.png"
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    if args.epochs < 30:
        raise ValueError("epochs must be >= 30 so the train/val curve is observable")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not 0.05 <= args.val_fraction <= 0.4:
        raise ValueError("val-fraction must be between 0.05 and 0.4")
    if args.val_guard_inlines < 1:
        raise ValueError("val-guard-inlines must be >= 1")
    build_args = build_summary["arguments"]
    if (
        float(build_args["val_fraction"]) != args.val_fraction
        or int(build_args["val_guard_inlines"]) != args.val_guard_inlines
    ):
        raise ValueError(
            "training validation split differs from the split used to fit normalization; "
            "rerun build_dataset.py with matching --val-fraction/--val-guard-inlines"
        )

    model = get_model(args.model, models_package="models", seed=args.seed)
    required_methods = ("train_batch", "loss_batch", "predict_batch")
    missing = [name for name in required_methods if not hasattr(model, name)]
    if missing:
        raise TypeError(f"registered model {args.model!r} is missing adapter methods: {missing}")

    full_train = load_samples("train")
    test = load_samples("test")
    if full_train.normalization_stats != test.normalization_stats:
        raise ValueError("train and test do not reuse the exact same train-fitted normalization stats")
    fit, validation, val_start_inline, val_guard_inlines = split_train_validation(
        full_train, args.val_fraction, args.val_guard_inlines
    )
    fit_inlines = {int(position["inline"]) for position in fit.positions}
    val_inlines = {int(position["inline"]) for position in validation.positions}
    test_inlines = {int(position["inline"]) for position in test.positions}
    if fit_inlines & val_inlines or (fit_inlines | val_inlines) & test_inlines:
        raise AssertionError("fit/validation/test inline leakage detected")

    weights = class_weights(fit.labels)
    fit_voxel_weights = voxel_weights(fit.labels, weights)
    val_voxel_weights = voxel_weights(validation.labels, weights)
    train_batches = ShuffledBatches(fit, fit_voxel_weights, args.batch_size, args.seed)
    val_batches = fixed_batches(validation, val_voxel_weights, args.batch_size)

    history = train_loop(
        model=model,
        train_step_fn=lambda batch: model.train_batch(*batch),
        val_step_fn=lambda batch: model.loss_batch(*batch),
        # Factories return fresh iterators on every epoch. ShuffledBatches was
        # already reusable under the old API, while the validation source was
        # a reusable list; making both factories explicit now satisfies the
        # fixed fail-loud ml_framework contract.
        train_batches_fn=lambda: iter(train_batches),
        val_batches_fn=lambda: iter(val_batches),
        epochs=args.epochs,
        save_checkpoint_fn=save_checkpoint,
        checkpoint_dir=checkpoint_dir,
        min_epochs_before_early_check=30,
    )
    plot_loss_curve(history, loss_curve_path)

    best_model = joblib.load(checkpoint_dir / "best.ckpt")
    joblib.dump(best_model, model_path)
    validation_probabilities_by_patch = np.concatenate(
        [
            best_model.predict_batch(validation.patches[start : start + args.batch_size])
            for start in range(0, len(validation.patches), args.batch_size)
        ]
    )
    validation_labels, validation_probabilities, validation_coverage = aggregate_physical_voxels(
        validation, validation_probabilities_by_patch
    )
    threshold, threshold_source, validation_threshold_f1 = select_validation_threshold(
        validation_labels, validation_probabilities, args.threshold
    )
    test_probabilities = np.concatenate(
        [
            best_model.predict_batch(test.patches[start : start + args.batch_size])
            for start in range(0, len(test.patches), args.batch_size)
        ],
        axis=0,
    )
    test_labels, test_probabilities_deduplicated, test_coverage = aggregate_physical_voxels(
        test, test_probabilities
    )
    predictions = test_probabilities_deduplicated >= threshold
    metrics = binary_metrics(test_labels, predictions)
    assert_finite_metrics(metrics)
    probability_scores = probability_metrics(test_labels, test_probabilities_deduplicated)
    all_negative_metrics = binary_metrics(test_labels, np.zeros_like(test_labels, dtype=bool))
    best_epoch_one_based = history.best_epoch + 1
    result = {
        "model": args.model,
        "models_package": "models",
        "batch_factory_contract": {
            "train": "lambda: iter(ShuffledBatches)",
            "validation": "lambda: iter(list)",
        },
        "model_description": getattr(best_model, "description", type(best_model).__name__),
        "seed": args.seed,
        "run_name": args.run_name,
        "arguments": vars(args),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "validation_f1_at_selected_threshold": validation_threshold_f1,
        "epochs_completed": len(history.train_loss),
        "best_epoch": best_epoch_one_based,
        "best_val_loss": history.best_val_loss,
        "final_train_loss": history.train_loss[-1],
        "final_val_loss": history.val_loss[-1],
        "epochs_after_best": len(history.val_loss) - best_epoch_one_based,
        "validation_turn_visible": bool(
            best_epoch_one_based < len(history.val_loss)
            and history.val_loss[-1] > history.best_val_loss
        ),
        "validation_start_inline": val_start_inline,
        "validation_guard_inlines": val_guard_inlines,
        "fit_samples": len(fit.patches),
        "validation_samples": len(validation.patches),
        "test_samples": len(test.patches),
        "fit_unique_inlines": len(fit_inlines),
        "validation_unique_inlines": len(val_inlines),
        "test_unique_inlines": len(test_inlines),
        "inline_overlap": [],
        "train_voxels": int(fit.labels.size),
        "train_positive_voxels": int(fit.labels.sum()),
        "validation_voxels": int(validation.labels.size),
        "validation_positive_voxels": int(validation.labels.sum()),
        "test_voxels": int(test_labels.size),
        "test_positive_voxels": int(test_labels.sum()),
        "class_weights": {str(key): value for key, value in weights.items()},
        "normalization": "ml_framework.fit_zscore/normalize/denormalize",
        "normalization_roundtrip_max_abs_error": max(
            full_train.roundtrip_max_abs_error, test.roundtrip_max_abs_error
        ),
        "normalization_stats": full_train.normalization_stats,
        "best_checkpoint": str((checkpoint_dir / "best.ckpt").relative_to(TRACK_DIR)),
        "last_checkpoint": str((checkpoint_dir / "last.ckpt").relative_to(TRACK_DIR)),
        "loss_curve": str(loss_curve_path.relative_to(TRACK_DIR)),
        "validation_physical_voxel_coverage": validation_coverage,
        "test_physical_voxel_coverage": test_coverage,
        "test_metrics": metrics,
        "test_probability_metrics": probability_scores,
        "all_negative_reference": all_negative_metrics,
        "software_versions": software_versions(),
        "dataset_sha256": {
            "train": sha256_file(PROJECT_ROOT / "_data" / "processed" / "fault" / "train.h5"),
            "test": sha256_file(PROJECT_ROOT / "_data" / "processed" / "fault" / "test.h5"),
        },
        "source_sha256": {
            "baseline.py": sha256_file(Path(__file__)),
            "model": sha256_file(TRACK_DIR / "models" / "fault_local_logistic.py"),
        },
        "historical_artifacts_verified": historical_artifacts,
    }
    if result["dataset_sha256"] != build_summary["dataset_sha256"]:
        raise RuntimeError("fault HDF5 files changed after the audited build summary was written")
    result["artifacts_sha256"] = {
        "best_checkpoint": sha256_file(checkpoint_dir / "best.ckpt"),
        "last_checkpoint": sha256_file(checkpoint_dir / "last.ckpt"),
        "baseline_model": sha256_file(model_path),
        "loss_curve": sha256_file(loss_curve_path),
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_historical_artifacts_if_present()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
