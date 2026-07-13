"""Facies segmentation, confidence, and calibration diagnostics."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pipeline_contract import segmentation_metrics_from_confusion


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("labels and predictions must be nonempty and aligned")
    if truth.min() < 0 or predicted.min() < 0 or truth.max() >= num_classes or predicted.max() >= num_classes:
        raise ValueError("labels/predictions violate configured class IDs")
    encoded = truth.reshape(-1) * num_classes + predicted.reshape(-1)
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute multiclass NLL/Brier and top-label/classwise ECE."""
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.ndim < 2:
        raise ValueError("probabilities must contain a class dimension")
    if probs.shape[0] != truth.shape[0] or probs.shape[2:] != truth.shape[1:]:
        raise ValueError(f"unaligned probabilities {probs.shape} and labels {truth.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError("probabilities do not sum to one")
    if n_bins < 2:
        raise ValueError("n_bins must be >=2")
    num_classes = probs.shape[1]
    flat_probs = np.moveaxis(probs, 1, -1).reshape(-1, num_classes)
    flat_truth = truth.reshape(-1)
    if flat_truth.min() < 0 or flat_truth.max() >= num_classes:
        raise ValueError("calibration labels violate probability width")
    row = np.arange(flat_truth.size)
    true_probability = flat_probs[row, flat_truth]
    nll = float(-np.log(np.clip(true_probability, epsilon, 1.0)).mean())
    one_hot = np.eye(num_classes, dtype=np.float64)[flat_truth]
    brier = float(np.square(flat_probs - one_hot).sum(axis=1).mean())
    predicted = flat_probs.argmax(axis=1)
    confidence = flat_probs.max(axis=1)
    correct = predicted == flat_truth
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        count = int(selected.sum())
        bin_confidence = float(confidence[selected].mean()) if count else None
        bin_accuracy = float(correct[selected].mean()) if count else None
        if count:
            ece += count / flat_truth.size * abs(bin_accuracy - bin_confidence)
        reliability.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )
    classwise: list[float] = []
    for class_id in range(num_classes):
        class_probability = flat_probs[:, class_id]
        class_truth = flat_truth == class_id
        class_ece = 0.0
        for index in range(n_bins):
            lower, upper = edges[index], edges[index + 1]
            selected = (class_probability >= lower) & (
                class_probability <= upper if index == n_bins - 1 else class_probability < upper
            )
            count = int(selected.sum())
            if count:
                class_ece += count / flat_truth.size * abs(
                    float(class_truth[selected].mean()) - float(class_probability[selected].mean())
                )
        classwise.append(float(class_ece))
    result = {
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "classwise_ece": classwise,
        "macro_classwise_ece": float(np.mean(classwise)),
        "reliability_bins": reliability,
        "calibration_pixels": int(flat_truth.size),
        "n_bins": n_bins,
    }
    numeric = [nll, brier, ece, *classwise]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("calibration metric is NaN/Inf")
    return result


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    n_bins: int = 15,
    require_all_classes: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = np.asarray(probabilities).argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, num_classes)
    if require_all_classes:
        metrics = segmentation_metrics_from_confusion(matrix)
        averaging = "all_configured_classes_and_valid_pixels"
    else:
        support = matrix.sum(axis=1).astype(np.int64)
        observed = support > 0
        if not observed.any():
            raise ValueError("smoke metric has no observed facies classes")
        true_positive = np.diag(matrix).astype(np.float64)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = support - true_positive
        union = true_positive + false_positive + false_negative
        denominator = 2.0 * true_positive + false_positive + false_negative
        iou = np.divide(true_positive, union, out=np.zeros(num_classes), where=union > 0)
        f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros(num_classes), where=denominator > 0)
        metrics = {
            "accuracy": float(true_positive.sum() / matrix.sum()),
            "miou": float(iou[observed].mean()),
            "macro_f1": float(f1[observed].mean()),
            "per_class_support": support.tolist(),
            "per_class_iou": iou.tolist(),
            "per_class_f1": f1.tolist(),
            "evaluated_pixels": int(matrix.sum()),
            "ignored_pixels": 0,
            "observed_class_ids": np.flatnonzero(observed).tolist(),
        }
        averaging = "observed_support_smoke_only_not_formal_metric"
    metrics.update(calibration_metrics(probabilities, labels, n_bins=n_bins))
    metrics.update(
        {
            "confusion_matrix": matrix.tolist(),
            "all_classes_supported": bool(np.all(np.asarray(metrics["per_class_support"]) > 0)),
            "finite_logits": True,
            "averaging": averaging,
        }
    )
    return metrics, matrix


def confidence_entropy_error(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.int64)
    prediction = probs.argmax(axis=1).astype(np.uint8)
    confidence = probs.max(axis=1).astype(np.float32)
    entropy = (
        -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        / math.log(probs.shape[1])
    ).astype(np.float32)
    error = (prediction != truth).astype(np.uint8)
    return prediction, confidence, entropy, error


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    nll_before: float
    nll_after: float
    fit_pixels: int
    fit_scope: str = "pooled_oof_only"
    method: str = "temperature_scaling_lbfgs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 50,
) -> TemperatureCalibration:
    """Fit one positive temperature from pooled OOF logits only."""
    values = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    truth = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if values.ndim != 2 or truth.ndim != 1 or values.shape[0] != truth.shape[0]:
        raise ValueError("temperature fit expects logits [N,C] and labels [N]")
    if values.shape[0] == 0 or not torch.isfinite(values).all():
        raise ValueError("temperature fit logits are empty or non-finite")
    if int(truth.min()) < 0 or int(truth.max()) >= values.shape[1]:
        raise ValueError("temperature fit labels violate logits width")
    before = float(F.cross_entropy(values, truth))
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(values / temperature, truth)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    after = float(F.cross_entropy(values / temperature, truth))
    if not all(math.isfinite(value) for value in (temperature, before, after)):
        raise ValueError("temperature scaling produced NaN/Inf")
    return TemperatureCalibration(
        temperature=temperature,
        nll_before=before,
        nll_after=after,
        fit_pixels=int(values.shape[0]),
    )


def sample_calibration_pixels(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap OOF calibration storage without using test."""
    values = np.asarray(logits)
    truth = np.asarray(labels)
    if values.ndim != 4 or truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("unaligned calibration logits/labels")
    flat_logits = np.moveaxis(values, 1, -1).reshape(-1, values.shape[1])
    flat_labels = truth.reshape(-1)
    if max_pixels <= 0:
        raise ValueError("max_pixels must be >0")
    if flat_labels.size > max_pixels:
        selected = np.random.default_rng(seed).choice(
            flat_labels.size, size=max_pixels, replace=False
        )
        selected.sort()
        flat_logits = flat_logits[selected]
        flat_labels = flat_labels[selected]
    return flat_logits.astype(np.float16), flat_labels.astype(np.uint8)
