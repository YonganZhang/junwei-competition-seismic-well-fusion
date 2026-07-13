"""Track-private P4 contracts for the nine-class GM09 lithofacies task.

The shared framework owns the outer contracts and lifecycle.  This module
owns only lithofacies semantics: the frozen label schema, mother-family split,
fold-local preprocessing, logits/probability adapters, and classification
diagnostics.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec
from _code.ml_framework.hpo import HPOPlan
from _code.ml_framework.preprocess import NormStats, denormalize, fit_zscore, normalize
from _code.ml_framework.splits import Fold, SplitManifest, build_group_folds, validate_manifest

from pipeline_contract import (
    CLASS_NAMES,
    LOG_CHANNELS,
    PIPELINE_VERSION,
    TARGET_CURVE_TYPE,
    TARGET_SOURCE,
    classification_metrics_from_confusion,
    validate_label_ids,
)


TARGET_NAME = "genetic_facies"
TEST_FAMILY = "15/9-F-5"
DEVELOPMENT_FAMILIES = (
    "15/9-19",
    "15/9-F-14",
    "15/9-F-15",
    "15/9-F-4",
)
REQUESTED_N_SPLITS = 5
EFFECTIVE_N_SPLITS = 4


def lithofacies_task_spec() -> TaskSpec:
    """Return the strict task declaration consumed by the shared P4 shell."""
    return TaskSpec(
        track_id="lithofacies",
        task_id="gm09_genetic_facies_9class",
        task_type="multiclass",
        input_modalities=("well_log_sequence", "st0202_seismic_patch"),
        targets=(TARGET_NAME,),
        units={TARGET_NAME: "class_id"},
        label_version=f"{TARGET_SOURCE}-{TARGET_CURVE_TYPE}-9-v1",
        target_masks={TARGET_NAME: "explicit_gm09_interval_only"},
        group_keys=("mother_family",),
        target_transform={TARGET_NAME: "identity_class_id"},
        inverse_transform={TARGET_NAME: "class_name_lookup"},
        train_loss={
            TARGET_NAME: {
                "default": "cross_entropy_sqrt_inverse_frequency",
                "candidates": (
                    "cross_entropy",
                    "cross_entropy_sqrt_inverse_frequency",
                    "focal",
                    "class_balanced_cross_entropy",
                ),
                "weights_fit_scope": "fold_train_only",
            }
        },
        inference_transform={TARGET_NAME: "softmax_then_argmax"},
        threshold_policy={TARGET_NAME: "none_for_closed_set_multiclass"},
        calibration_policy={
            TARGET_NAME: {
                "method": "temperature_scaling",
                "fit_scope": "pooled_development_oof_only",
                "test_refit_forbidden": True,
            }
        },
        primary_metrics=("supported_class_macro_f1",),
        secondary_metrics=(
            "fixed_schema_macro_f1",
            "balanced_accuracy",
            "negative_log_likelihood",
            "expected_calibration_error",
        ),
        guardrail_metrics=("worst_family_supported_class_macro_f1",),
        metric_directions={
            "supported_class_macro_f1": "maximize",
            "fixed_schema_macro_f1": "maximize",
            "balanced_accuracy": "maximize",
            "negative_log_likelihood": "minimize",
            "expected_calibration_error": "minimize",
            "worst_family_supported_class_macro_f1": "maximize",
        },
        hpo={
            "direction": "maximize",
            "primary_metric": "supported_class_macro_f1",
            "sanity_trials": "8-12",
            "pilot_trials": "20-30",
            "sampler": "single_process_tpe_after_random_startup",
            "default_pruner": "nop",
            "conditional_pruner": "conservative_median_after_comparable_trials",
            "confirmation": "top_3_configs_x_3_preregistered_seeds_x_4_folds",
        },
        visualizer_id="lithofacies_p4_archived_predictions",
        required_figures=(
            "depth_facies_track",
            "confusion_count_and_row_normalized",
            "per_class_precision_recall_f1_support",
            "calibration_reliability",
        ),
        input_whitelist=tuple(LOG_CHANNELS) + (
            "well_log_observed_mask",
            "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME",
        ),
        forbidden_inputs=(
            "Litho Class",
            "GENETIC FACIES",
            "LITH",
            "UNKNOWN",
            "UNDEFINED",
            "VSH",
            "PHIE",
            "SAND",
            "formation_name",
            "RMS_realization",
        ),
        metadata={
            "pipeline_version": PIPELINE_VERSION,
            "class_names": CLASS_NAMES,
            "class_count": len(CLASS_NAMES),
            "frozen_test_family": TEST_FAMILY,
            "requested_n_splits": REQUESTED_N_SPLITS,
            "effective_n_splits": EFFECTIVE_N_SPLITS,
            "splitter": "leave_one_mother_family_out",
        },
    )


def lithofacies_hpo_plan() -> dict[str, Any]:
    """Archive the direction and bounded plan without launching a study."""
    plan = HPOPlan(
        sanity_trials=8,
        pilot_trials=20,
        top_configs=3,
        confirm_seeds=3,
        sampler="random_then_tpe_single_process",
        pruner="nop",
        direction="maximize",
    )
    return {
        **asdict(plan),
        "primary_metric": "supported_class_macro_f1",
        "test_access": "forbidden",
        "effective_folds": EFFECTIVE_N_SPLITS,
        "search_space": {
            "model_id": (
                "lithofacies_concat_linear",
                "lithofacies_late_fusion",
                "multimodal_mlp",
            ),
            "loss": (
                "cross_entropy",
                "cross_entropy_sqrt_inverse_frequency",
                "focal",
                "class_balanced_cross_entropy",
            ),
            "learning_rate": {"low": 1e-4, "high": 3e-3, "log": True},
            "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
            "batch_size": (16, 32, 64),
            "hidden_size": (16, 32, 64, 128),
        },
        "pruner_upgrade_gate": (
            "median pruning is allowed only after comparable complete trials show that "
            "fold-level pruning is stable; small/noisy runs stay on NopPruner"
        ),
        "selection": {
            "first": "maximize fold-mean supported_class_macro_f1",
            "tie_breakers": (
                "maximize worst fold",
                "minimize fold std",
                "minimize negative_log_likelihood",
                "prefer simpler model",
            ),
        },
    }


def _sample_family(sample: Mapping[str, Any]) -> str:
    family = str(sample.get("meta", {}).get("family_id", ""))
    if not family:
        raise ValueError("sample is missing meta.family_id")
    return family


def sample_id(sample: Mapping[str, Any]) -> str:
    """Construct a stable ID from stored provenance, never row order."""
    position = sample.get("position", {})
    trace = sample.get("meta", {}).get("label_trace", {})
    values = (
        position.get("well_name"),
        position.get("center_md_m"),
        position.get("time_ms"),
        trace.get("member"),
        trace.get("excel_row"),
    )
    if not values[0]:
        raise ValueError("sample is missing position.well_name")
    return "|".join("NA" if value is None else str(value) for value in values)


def validate_p4_sample(sample: Mapping[str, Any]) -> None:
    label = np.asarray([int(sample["label"])], dtype=np.int64)
    validate_label_ids(label)
    meta = sample.get("meta", {})
    trace = meta.get("label_trace", {})
    if trace.get("source") != TARGET_SOURCE or trace.get("curve_type") != TARGET_CURVE_TYPE:
        raise ValueError("label provenance is not GM09/GENETIC FACIES")
    if trace.get("class_name") != CLASS_NAMES[int(label[0])]:
        raise ValueError("label ID and traced class name disagree")
    _sample_family(sample)
    well_log = np.asarray(sample["well_log_seq"])
    seismic = np.asarray(sample["seismic_patch"])
    if well_log.ndim != 2 or well_log.shape[0] != 2 * len(LOG_CHANNELS):
        raise ValueError(f"well_log_seq must have 2C rows, got {well_log.shape}")
    if seismic.ndim != 3:
        raise ValueError(f"seismic_patch must be 3-D, got {seismic.shape}")
    if not np.isfinite(well_log).all() or not np.isfinite(seismic).all():
        raise ValueError("sample contains NaN/Inf")


def class_support(samples: Sequence[Mapping[str, Any]]) -> np.ndarray:
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    validate_label_ids(labels)
    return np.bincount(labels, minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]


def _samples_by_id(samples: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {sample_id(sample): sample for sample in samples}
    if len(result) != len(samples):
        raise ValueError("stable lithofacies sample IDs are not unique")
    return result


def build_lithofacies_split_manifest(samples: Sequence[Mapping[str, Any]]) -> SplitManifest:
    """Freeze F-5 and honestly downgrade requested five folds to LOGO-4."""
    if not samples:
        raise ValueError("cannot split zero samples")
    for sample in samples:
        validate_p4_sample(sample)
    ids = [sample_id(sample) for sample in samples]
    groups = [_sample_family(sample) for sample in samples]
    actual = set(groups)
    expected = set(DEVELOPMENT_FAMILIES) | {TEST_FAMILY}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(f"unapproved mother families in P4 data: {unexpected}")
    missing_development = sorted(set(DEVELOPMENT_FAMILIES) - actual)
    if missing_development:
        raise ValueError(f"P4 LOGO-4 requires all development families: {missing_development}")
    manifest = build_group_folds(
        ids,
        groups,
        group_key="mother_family",
        test_groups=(TEST_FAMILY,),
        requested_n_splits=REQUESTED_N_SPLITS,
        seed=2693,
        max_splits_by_support=EFFECTIVE_N_SPLITS,
        support_reason="only four independent non-test mother families have real multimodal samples",
        metadata={
            "track_id": "lithofacies",
            "task_id": "gm09_genetic_facies_9class",
            "class_names": CLASS_NAMES,
            "frozen_test_priority": True,
            "splitter": "leave_one_mother_family_out",
            "forbidden_downgrade": "never split samples from one mother family to manufacture five folds",
        },
    )
    lookup = _samples_by_id(samples)
    folds: list[Fold] = []
    for fold in manifest.folds:
        train_samples = [lookup[sid] for sid in fold.train_sample_ids]
        validation_samples = [lookup[sid] for sid in fold.validation_sample_ids]
        train_counts = class_support(train_samples)
        validation_counts = class_support(validation_samples)
        folds.append(
            replace(
                fold,
                support={
                    "class_names": CLASS_NAMES,
                    "train_class_support": train_counts.tolist(),
                    "validation_class_support": validation_counts.tolist(),
                    "train_missing_class_ids": np.flatnonzero(train_counts == 0).tolist(),
                    "validation_missing_class_ids": np.flatnonzero(validation_counts == 0).tolist(),
                    "fixed_output_class_count": len(CLASS_NAMES),
                    "coverage_policy": (
                        "keep nine logits; zero loss weight for a fold-train-unseen class; "
                        "report fixed-schema and observed-support metrics"
                    ),
                },
            )
        )
    supported = replace(manifest, folds=tuple(folds))
    if supported.effective_n_splits != EFFECTIVE_N_SPLITS:
        raise RuntimeError("lithofacies P4 split must be the honest four-family LOGO downgrade")
    if any(len(fold.validation_groups) != 1 for fold in supported.folds):
        raise RuntimeError("each LOGO fold must hold out exactly one mother family")
    validate_manifest(supported)
    return supported


@dataclass(frozen=True)
class FoldPreprocessor:
    """Statistics fitted from one fold's training mother families only."""

    log_stats: tuple[Mapping[str, Any] | None, ...]
    seismic_stats: Mapping[str, Any]
    class_support: tuple[int, ...]
    class_weights: tuple[float, ...]
    fit_sample_ids: tuple[str, ...]
    fit_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoldPreprocessor":
        values = dict(payload)
        values["log_stats"] = tuple(values["log_stats"])
        values["class_support"] = tuple(int(value) for value in values["class_support"])
        values["class_weights"] = tuple(float(value) for value in values["class_weights"])
        values["fit_sample_ids"] = tuple(values["fit_sample_ids"])
        values["fit_families"] = tuple(values["fit_families"])
        return cls(**values)


def _stored_stats(sample: Mapping[str, Any]) -> tuple[list[NormStats], NormStats]:
    payload = sample.get("meta", {}).get("normalization_stats")
    if not isinstance(payload, Mapping):
        raise ValueError("sample lacks reversible stored normalization_stats")
    logs = payload.get("logs", {})
    missing = [channel for channel in LOG_CHANNELS if channel not in logs]
    if missing:
        raise ValueError(f"stored log normalization is missing channels: {missing}")
    return [NormStats.from_dict(dict(logs[channel])) for channel in LOG_CHANNELS], NormStats.from_dict(
        dict(payload["seismic"])
    )


def _recover_physical(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert the stored normalization before any fold-local statistic is fit."""
    validate_p4_sample(sample)
    stored_logs, stored_seismic = _stored_stats(sample)
    well_log = np.asarray(sample["well_log_seq"], dtype=np.float64)
    channel_count = len(LOG_CHANNELS)
    values = well_log[:channel_count]
    masks = well_log[channel_count:] > 0.5
    physical = np.zeros_like(values, dtype=np.float64)
    for index, stats in enumerate(stored_logs):
        if masks[index].any():
            physical[index, masks[index]] = denormalize(values[index, masks[index]], stats)
    seismic = denormalize(np.asarray(sample["seismic_patch"], dtype=np.float64), stored_seismic)
    return physical, masks, seismic


def _tempered_class_weights(counts: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    supported = counts > 0
    frequency = counts[supported] / counts[supported].sum()
    weights[supported] = 1.0 / np.sqrt(frequency)
    weights[supported] /= weights[supported].mean()
    return weights


def fit_fold_preprocessor(samples: Sequence[Mapping[str, Any]]) -> FoldPreprocessor:
    """Fit reversible feature statistics and class weights on fold-train only."""
    if not samples:
        raise ValueError("fold-train samples must not be empty")
    physical = [_recover_physical(sample) for sample in samples]
    stats: list[Mapping[str, Any] | None] = []
    for channel_index in range(len(LOG_CHANNELS)):
        observed = [
            values[channel_index, masks[channel_index]]
            for values, masks, _ in physical
            if masks[channel_index].any()
        ]
        joined = np.concatenate(observed) if observed else np.asarray([], dtype=np.float64)
        stats.append(fit_zscore(joined).to_dict() if joined.size >= 2 else None)
    seismic_values = np.concatenate([seismic.reshape(-1) for _, _, seismic in physical])
    seismic_stats = fit_zscore(seismic_values).to_dict()
    counts = class_support(samples)
    weights = _tempered_class_weights(counts)
    return FoldPreprocessor(
        log_stats=tuple(stats),
        seismic_stats=seismic_stats,
        class_support=tuple(int(value) for value in counts),
        class_weights=tuple(float(value) for value in weights),
        fit_sample_ids=tuple(sorted(sample_id(sample) for sample in samples)),
        fit_families=tuple(sorted({_sample_family(sample) for sample in samples})),
    )


def apply_fold_preprocessor(
    samples: Sequence[Mapping[str, Any]], preprocessor: FoldPreprocessor
) -> list[dict[str, Any]]:
    """Apply immutable fold-train statistics to train or validation samples."""
    transformed: list[dict[str, Any]] = []
    seismic_stats = NormStats.from_dict(dict(preprocessor.seismic_stats))
    for source in samples:
        physical, masks, seismic = _recover_physical(source)
        normalized = np.zeros_like(physical, dtype=np.float32)
        effective_masks = masks.copy()
        unseen_channels: list[str] = []
        for index, stats_payload in enumerate(preprocessor.log_stats):
            if stats_payload is None:
                effective_masks[index] = False
                unseen_channels.append(LOG_CHANNELS[index])
                continue
            stats = NormStats.from_dict(dict(stats_payload))
            normalized[index, effective_masks[index]] = normalize(
                physical[index, effective_masks[index]], stats
            ).astype(np.float32)
        result = copy.deepcopy(dict(source))
        result["well_log_seq"] = np.concatenate(
            (normalized, effective_masks.astype(np.float32)), axis=0
        ).astype(np.float32)
        result["seismic_patch"] = normalize(seismic, seismic_stats).astype(np.float32)
        result.setdefault("meta", {})["p4_fold_preprocessing"] = {
            "fit_scope": "fold_train_mother_families_only",
            "fit_families": preprocessor.fit_families,
            "unseen_log_channels_masked": unseen_channels,
        }
        transformed.append(result)
    return transformed


def samples_to_model_batch(samples: Sequence[Mapping[str, Any]], *, device: str = "cpu") -> ModelBatch:
    """Strictly adapt stored samples to the shared ModelBatch envelope."""
    if not samples:
        raise ValueError("cannot adapt an empty batch")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in torch-free portable environments
        raise RuntimeError("PyTorch is required for the lithofacies model adapter") from exc
    for sample in samples:
        validate_p4_sample(sample)
    well_log = torch.as_tensor(
        np.stack([np.asarray(sample["well_log_seq"], dtype=np.float32) for sample in samples]),
        device=device,
    )
    seismic = torch.as_tensor(
        np.stack([np.asarray(sample["seismic_patch"], dtype=np.float32) for sample in samples]),
        device=device,
    )
    labels = torch.as_tensor([int(sample["label"]) for sample in samples], dtype=torch.long, device=device)
    channel_count = len(LOG_CHANNELS)
    centers = [sample.get("position", {}).get("center_md_m") for sample in samples]
    return ModelBatch(
        inputs={"well_log_seq": well_log, "seismic_patch": seismic},
        targets={TARGET_NAME: labels},
        input_masks={"well_log_observed": well_log[:, channel_count:, :] > 0.5},
        target_masks={TARGET_NAME: torch.ones_like(labels, dtype=torch.bool)},
        sample_ids=[sample_id(sample) for sample in samples],
        groups={"mother_family": [_sample_family(sample) for sample in samples]},
        coordinates={
            "center_md_m": np.asarray(
                [np.nan if center is None else float(center) for center in centers], dtype=np.float64
            ),
            "twt_ms": np.asarray(
                [float(sample.get("position", {}).get("time_ms", np.nan)) for sample in samples],
                dtype=np.float64,
            ),
        },
        metadata={"class_names": CLASS_NAMES, "pipeline_version": PIPELINE_VERSION},
    )


def model_output_from_logits(logits: Any) -> ModelOutput:
    """Keep training output as raw logits; probability conversion is separate."""
    shape = tuple(int(value) for value in logits.shape)
    if len(shape) != 2 or shape[1] != len(CLASS_NAMES):
        raise ValueError(f"lithofacies logits must have shape [B,9], got {shape}")
    try:
        finite = bool(logits.isfinite().all())
    except AttributeError:
        finite = bool(np.isfinite(np.asarray(logits)).all())
    if not finite:
        raise ValueError("lithofacies logits contain NaN/Inf")
    return ModelOutput(raw={TARGET_NAME: logits})


def softmax_probabilities(output: ModelOutput) -> Any:
    """Convert archived/inference logits to probabilities, never during CE training."""
    logits = output.raw[TARGET_NAME]
    try:
        import torch

        if isinstance(logits, torch.Tensor):
            probabilities = torch.softmax(logits, dim=1)
            return ModelOutput(raw=output.raw, transformed={TARGET_NAME: probabilities}, aux=output.aux)
    except ImportError:
        pass
    array = np.asarray(logits, dtype=np.float64)
    shifted = array - array.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    return ModelOutput(raw=output.raw, transformed={TARGET_NAME: probabilities}, aux=output.aux)


def cross_entropy_loss(output: ModelOutput, batch: ModelBatch, class_weights: Any | None = None) -> Any:
    """Consume logits directly with an explicit fold-train-only weight vector."""
    try:
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for cross-entropy training") from exc
    logits = output.raw[TARGET_NAME]
    labels = batch.targets[TARGET_NAME] if batch.targets is not None else None
    if labels is None:
        raise ValueError("cross-entropy requires targets")
    return functional.cross_entropy(logits, labels, weight=class_weights, reduction="sum")


def reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    if n_bins < 2:
        raise ValueError("n_bins must be >=2")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    bins: list[dict[str, Any]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (confidence <= upper if index == n_bins - 1 else confidence < upper)
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else 0.0
        mean_confidence = float(confidence[selected].mean()) if count else 0.0
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {"n_bins": n_bins, "expected_calibration_error": float(ece), "bins": bins}


def classification_metrics_from_logits(labels: Sequence[int], logits: np.ndarray) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    validate_label_ids(labels_array)
    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.shape != (labels_array.size, len(CLASS_NAMES)) or not np.isfinite(logits_array).all():
        raise ValueError("logits must be finite [N,9] and match labels")
    probability_output = softmax_probabilities(model_output_from_logits(logits_array))
    probabilities = np.asarray(probability_output.transformed[TARGET_NAME], dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels_array, predictions), 1)
    metrics = classification_metrics_from_confusion(confusion)
    selected = np.clip(probabilities[np.arange(labels_array.size), labels_array], 1e-12, 1.0)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels_array]
    row_totals = confusion.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals > 0,
    )
    calibration = reliability_bins(labels_array, probabilities)
    metrics.update(
        {
            "fixed_schema_macro_f1": metrics["macro_f1"],
            "negative_log_likelihood": float(-np.log(selected).mean()),
            "multiclass_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
            "expected_calibration_error": calibration["expected_calibration_error"],
            "calibration": calibration,
            "confusion_matrix_row_normalized": row_normalized.tolist(),
        }
    )
    return metrics


def fit_temperature(labels: Sequence[int], logits: np.ndarray) -> dict[str, Any]:
    """Fit one scalar on pooled OOF logits using a deterministic bounded grid."""
    labels_array = np.asarray(labels, dtype=np.int64)
    logits_array = np.asarray(logits, dtype=np.float64)
    candidates = np.geomspace(0.25, 4.0, 121)
    losses = []
    for temperature in candidates:
        metrics = classification_metrics_from_logits(labels_array, logits_array / temperature)
        losses.append(metrics["negative_log_likelihood"])
    best_index = int(np.argmin(losses))
    return {
        "method": "temperature_scaling_grid",
        "fit_scope": "pooled_development_oof_only",
        "temperature": float(candidates[best_index]),
        "negative_log_likelihood": float(losses[best_index]),
        "candidate_count": int(candidates.size),
        "test_labels_used": False,
    }


def prediction_records(
    samples: Sequence[Mapping[str, Any]], logits: np.ndarray, *, temperature: float = 1.0
) -> list[dict[str, Any]]:
    logits_array = np.asarray(logits, dtype=np.float64)
    probabilities = np.asarray(
        softmax_probabilities(model_output_from_logits(logits_array / temperature)).transformed[TARGET_NAME]
    )
    predictions = probabilities.argmax(axis=1)
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        truth = int(sample["label"])
        position = sample.get("position", {})
        records.append(
            {
                "sample_id": sample_id(sample),
                "well_id": position.get("well_name"),
                "family_id": _sample_family(sample),
                "center_md_m": position.get("center_md_m"),
                "twt_ms": position.get("time_ms"),
                "true_class_id": truth,
                "true_class_name": CLASS_NAMES[truth],
                "predicted_class_id": int(predictions[index]),
                "predicted_class_name": CLASS_NAMES[int(predictions[index])],
                "confidence": float(probabilities[index, predictions[index]]),
                "error": bool(predictions[index] != truth),
                "logits": logits_array[index].tolist(),
                "probabilities": probabilities[index].tolist(),
            }
        )
    return records


def finite_center_md(records: Sequence[Mapping[str, Any]]) -> bool:
    return bool(records) and all(
        record.get("center_md_m") is not None and math.isfinite(float(record["center_md_m"]))
        for record in records
    )
