#!/usr/bin/env python3
"""Fault P5.1 R0 lane gates and bounded R1 protocol-mechanism audit.

The runner accepts only hash-locked development inputs.  It deliberately
contains no holdout input surface.  R1-A/B demonstrate why complement-as-
negative labels and random splits are scientifically invalid; R1-C applies the
formal sparse-label contract and stops before model construction when audited
negative support is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import segyio
from sklearn.metrics import average_precision_score


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (PROJECT_ROOT, TRACK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from _code.ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.preprocess import fit_zscore, normalize  # noqa: E402
from p4_contract import build_fault_masks, fault_task_spec  # noqa: E402
from p4_split import SpatialSample, build_buffered_spatial_cv  # noqa: E402


ROOT_SEED = 2693
MODEL_ID = "fault_local_logistic"
BUFFER_INLINES = 8
REQUESTED_FOLDS = 5
PATCH_SHAPE = (17, 33)
DEVELOPMENT_INLINE_COUNT = 48
PARAMETER_UPDATES = 4
DEFAULT_LOCK = TRACK_DIR / "p5_r01_development_lock.json"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_r01_protocol"
FORMAL_CV = TRACK_DIR / "_outputs" / "p4_preflight" / "buffered_cv_plan.json"
STAGE3_DATA = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_data_manifest.json"
LANES = ("synthetic_only", "masked_weak_label", "formal_audited")


class FaultP5R01Error(RuntimeError):
    """Input evidence or a scientific stop condition is invalid."""


@dataclass(frozen=True)
class DevelopmentSamples:
    amplitudes: np.ndarray
    sparse_positive: np.ndarray
    sample_ids: tuple[str, ...]
    inlines: np.ndarray
    kinds: tuple[str, ...]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.amplitudes.ndim != 4 or self.amplitudes.shape[1] != 1:
            raise ValueError("amplitudes must have [N,1,H,W] shape")
        if self.sparse_positive.shape != self.amplitudes.shape[:1] + self.amplitudes.shape[2:]:
            raise ValueError("sparse_positive must match [N,H,W]")
        if len(self.sample_ids) != len(self.amplitudes) or len(self.inlines) != len(self.amplitudes):
            raise ValueError("sample metadata length mismatch")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("development sample IDs must be unique")
        if not np.isfinite(self.amplitudes).all() or not np.isin(self.sparse_positive, (0, 1)).all():
            raise ValueError("development arrays must be finite and labels binary")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FaultP5R01Error(f"expected JSON object for {path.name}")
    return payload


def _firewall() -> dict[str, Any]:
    return {
        "runner_accepts_test_inputs": False,
        "frozen_test_accessed": False,
        "test_labels_accessed": False,
        "test_predictions_accessed": False,
        "test_metrics_accessed": False,
        "historical_holdout_arrays_accessed": False,
        "selection_or_ranking_performed": False,
    }


def _verify_root_seed(root_seed: int) -> None:
    if root_seed != ROOT_SEED:
        raise ValueError(f"root seed is frozen at {ROOT_SEED}")


def verify_development_lock(
    lock_path: Path,
    *,
    seismic_amplitude: Path,
    fault_points: Path,
    seismic_index: Path,
) -> dict[str, Any]:
    """Hash-lock the three explicit development inputs without accepting a test role."""

    lock = _read_json(lock_path)
    if lock.get("protocol") != "fault-p5-r01-development-lock-v1" or lock.get("role") != "development_only":
        raise FaultP5R01Error("development lock protocol/role is invalid")
    expected = lock.get("input_sha256")
    if not isinstance(expected, dict) or set(expected) != {
        "seismic_amplitude", "fault_points", "seismic_index"
    }:
        raise FaultP5R01Error("development lock must contain exactly three source hashes")
    paths = {
        "seismic_amplitude": seismic_amplitude,
        "fault_points": fault_points,
        "seismic_index": seismic_index,
    }
    observed: dict[str, str] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FaultP5R01Error(f"explicit development source is missing: {role}")
        observed[role] = hash_file(path)
        if observed[role] != expected[role]:
            raise FaultP5R01Error(f"development source hash mismatch: {role}")
    inline_range = lock.get("development_inline_range")
    if (
        not isinstance(inline_range, list)
        or len(inline_range) != 2
        or not all(isinstance(value, int) for value in inline_range)
        or inline_range[0] > inline_range[1]
    ):
        raise FaultP5R01Error("development inline range is invalid")
    return {
        "lock_sha256": hash_file(lock_path),
        "input_sha256": observed,
        "development_inline_range": inline_range,
    }


def _nearest_time_indices(times_ms: np.ndarray, query_ms: np.ndarray) -> np.ndarray:
    samples = np.asarray(times_ms, dtype=np.float64)
    query = np.asarray(query_ms, dtype=np.float64)
    right = np.clip(np.searchsorted(samples, query, side="left"), 1, len(samples) - 1)
    left = right - 1
    return np.where(
        np.abs(query - samples[left]) <= np.abs(samples[right] - query), left, right
    ).astype(np.int32)


def _fault_sticks(vertex_codes: np.ndarray) -> list[np.ndarray]:
    sticks: list[np.ndarray] = []
    current: list[int] = []
    for index, raw_code in enumerate(np.asarray(vertex_codes).tolist()):
        code = int(raw_code)
        if code == 1:
            if current:
                raise FaultP5R01Error("malformed fault-stick sequence")
            current = [index]
        elif code == 2:
            if not current:
                raise FaultP5R01Error("fault-stick middle has no start")
            current.append(index)
        elif code == 3:
            if not current:
                raise FaultP5R01Error("fault-stick end has no start")
            current.append(index)
            sticks.append(np.asarray(current, dtype=np.int32))
            current = []
        else:
            raise FaultP5R01Error(f"unknown fault-stick vertex code: {code}")
    if current:
        raise FaultP5R01Error("unterminated fault-stick sequence")
    return sticks


def _rasterize_fault_voxels(faults: Mapping[str, np.ndarray], index: Mapping[str, np.ndarray]) -> np.ndarray:
    inline = np.asarray(faults["inline"], dtype=np.int32)
    crossline = np.asarray(faults["crossline"], dtype=np.int32)
    time_index = _nearest_time_indices(index["samples_ms"], faults["twt_ms"])
    chunks: list[np.ndarray] = []
    for stick in _fault_sticks(faults["stick_no"]):
        points = np.column_stack((inline[stick], crossline[stick], time_index[stick])).astype(float)
        for start, end in zip(points[:-1], points[1:]):
            steps = int(np.max(np.abs(end - start))) + 1
            chunks.append(np.rint(np.linspace(start, end, steps)).astype(np.int32))
    if not chunks:
        raise FaultP5R01Error("no fault-stick segment could be rasterized")
    return np.unique(np.concatenate(chunks, axis=0), axis=0)


def _group_voxels(voxels: np.ndarray) -> dict[int, np.ndarray]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for inline, crossline, time_index in voxels.tolist():
        grouped.setdefault(int(inline), []).append((int(crossline), int(time_index)))
    return {inline: np.asarray(points, dtype=np.int32) for inline, points in grouped.items()}


def _label_patch(
    grouped: Mapping[int, np.ndarray], inline: int, crossline: int, time_index: int
) -> np.ndarray:
    height, width = PATCH_SHAPE
    half_height, half_width = height // 2, width // 2
    label = np.zeros(PATCH_SHAPE, dtype=np.uint8)
    points = grouped.get(inline)
    if points is None:
        return label
    rows = points[:, 0] - (crossline - half_height)
    columns = points[:, 1] - (time_index - half_width)
    keep = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    label[rows[keep], columns[keep]] = 1
    return label


def _choose_development_centres(
    grouped: Mapping[int, np.ndarray],
    index: Mapping[str, np.ndarray],
    development_range: Sequence[int],
    *,
    root_seed: int,
) -> tuple[list[tuple[int, int, int, str]], dict[str, Any]]:
    """Choose only observed fault-voxel centres on development inlines."""

    height, width = PATCH_SHAPE
    half_height, half_width = height // 2, width // 2
    il_low, il_high = map(int, development_range)
    xl_low = int(index["xl_min"]) + half_height
    xl_high = int(index["xl_max"]) - half_height
    time_low = half_width
    time_high = len(index["samples_ms"]) - half_width - 1
    eligible: list[int] = []
    eligible_points: dict[int, np.ndarray] = {}
    for inline in sorted(grouped):
        if not il_low <= inline <= il_high:
            continue
        points = grouped[inline]
        keep = (
            (points[:, 0] >= xl_low)
            & (points[:, 0] <= xl_high)
            & (points[:, 1] >= time_low)
            & (points[:, 1] <= time_high)
        )
        if np.any(keep):
            eligible.append(inline)
            eligible_points[inline] = points[keep]
    if len(eligible) < DEVELOPMENT_INLINE_COUNT:
        raise FaultP5R01Error("not enough development inlines for the bounded mechanism audit")
    selected_positions = np.linspace(0, len(eligible) - 1, DEVELOPMENT_INLINE_COUNT, dtype=int)
    selected_inlines = [eligible[position] for position in selected_positions]
    centres: list[tuple[int, int, int, str]] = []
    for inline in selected_inlines:
        points = eligible_points[inline]
        positive = points[len(points) // 2]
        centres.append((inline, int(positive[0]), int(positive[1]), "fault_positive"))
    return centres, {
        "eligible_development_inlines": len(eligible),
        "selected_fault_centre_inlines": selected_inlines,
        "selection_seed": root_seed,
        "sampling_role": "observed_fault_voxel_centres_only",
        "random_or_annotation_free_negative_centres_generated": False,
    }


def load_development_samples(
    *,
    seismic_amplitude: Path,
    fault_points: Path,
    seismic_index: Path,
    lock_evidence: Mapping[str, Any],
    root_seed: int,
) -> tuple[DevelopmentSamples, dict[str, Any]]:
    """Read a fixed tiny development-only sample without invoking the legacy builder."""

    _verify_root_seed(root_seed)
    with np.load(fault_points, allow_pickle=False) as archive:
        faults = {key: archive[key] for key in ("inline", "crossline", "twt_ms", "stick_no")}
    with np.load(seismic_index, allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    voxels = _rasterize_fault_voxels(faults, index)
    grouped = _group_voxels(voxels)
    centres, sampling = _choose_development_centres(
        grouped,
        index,
        lock_evidence["development_inline_range"],
        root_seed=root_seed,
    )
    height, width = PATCH_SHAPE
    amplitudes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    inlines: list[int] = []
    kinds: list[str] = []
    il_min, xl_min, n_xl = int(index["il_min"]), int(index["xl_min"]), int(index["n_xl"])
    with segyio.open(str(seismic_amplitude), "r", ignore_geometry=True) as volume:
        for inline, crossline, time_index, kind in centres:
            patch = np.empty(PATCH_SHAPE, dtype=np.float32)
            for row, crossline_at_row in enumerate(range(crossline - height // 2, crossline + height // 2 + 1)):
                trace_index = (inline - il_min) * n_xl + (crossline_at_row - xl_min)
                patch[row] = np.asarray(
                    volume.trace[trace_index][time_index - width // 2 : time_index + width // 2 + 1],
                    dtype=np.float32,
                )
            if not np.isfinite(patch).all() or float(np.std(patch)) <= 1e-8:
                raise FaultP5R01Error("development seismic patch is non-finite or constant")
            label = _label_patch(grouped, inline, crossline, time_index)
            if kind == "fault_positive" and not label.any():
                raise FaultP5R01Error("positive diagnostic patch lost its sparse stick")
            if kind != "fault_positive" and label.any():
                raise FaultP5R01Error("proxy diagnostic patch contains a sparse stick")
            amplitudes.append(patch[None])
            labels.append(label)
            inlines.append(inline)
            kinds.append(kind)
            sample_ids.append(f"dev:il{inline}:xl{crossline}:t{time_index}:{kind}")
    samples = DevelopmentSamples(
        amplitudes=np.stack(amplitudes),
        sparse_positive=np.stack(labels),
        sample_ids=tuple(sample_ids),
        inlines=np.asarray(inlines, dtype=np.int32),
        kinds=tuple(kinds),
        source_hashes=dict(lock_evidence["input_sha256"]),
    )
    return samples, {
        **sampling,
        "sample_count": len(samples.sample_ids),
        "positive_patch_count": sum(kind == "fault_positive" for kind in samples.kinds),
        "proxy_patch_count": sum(kind != "fault_positive" for kind in samples.kinds),
        "rasterized_positive_voxels_in_development": int(
            np.sum(
                (voxels[:, 0] >= lock_evidence["development_inline_range"][0])
                & (voxels[:, 0] <= lock_evidence["development_inline_range"][1])
            )
        ),
    }


def _random_split(samples: DevelopmentSamples, root_seed: int) -> dict[str, np.ndarray]:
    """Deliberately illegal sample-level random split, stratified only for diagnostics."""

    rng = np.random.default_rng(root_seed)
    train: list[int] = []
    validation: list[int] = []
    for kind in sorted(set(samples.kinds)):
        indices = np.asarray([index for index, value in enumerate(samples.kinds) if value == kind])
        rng.shuffle(indices)
        cut = max(1, len(indices) // 4)
        validation.extend(indices[:cut].tolist())
        train.extend(indices[cut:].tolist())
    return {
        "train": np.asarray(sorted(train), dtype=np.int32),
        "validation": np.asarray(sorted(validation), dtype=np.int32),
        "excluded_buffer": np.asarray([], dtype=np.int32),
    }


def _spatial_split(samples: DevelopmentSamples, buffer_inlines: int) -> dict[str, np.ndarray]:
    unique_inlines = sorted(set(map(int, samples.inlines.tolist())))
    validation_count = max(2, len(unique_inlines) // 4)
    validation_inlines = set(unique_inlines[-validation_count:])
    validation_start = min(validation_inlines)
    train_inlines = {value for value in unique_inlines if value < validation_start - buffer_inlines}
    buffer = set(unique_inlines) - validation_inlines - train_inlines
    if not train_inlines or not validation_inlines:
        raise FaultP5R01Error("bounded spatial split has an empty side")
    return {
        "train": np.asarray(
            [index for index, inline in enumerate(samples.inlines) if int(inline) in train_inlines],
            dtype=np.int32,
        ),
        "validation": np.asarray(
            [index for index, inline in enumerate(samples.inlines) if int(inline) in validation_inlines],
            dtype=np.int32,
        ),
        "excluded_buffer": np.asarray(
            [index for index, inline in enumerate(samples.inlines) if int(inline) in buffer],
            dtype=np.int32,
        ),
    }


def _class_weights(labels: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise FaultP5R01Error("diagnostic fold-train lacks binary support")
    positive_weight = labels.size / (2.0 * positives)
    negative_weight = labels.size / (2.0 * negatives)
    weights = np.where(labels == 1, positive_weight, negative_weight).astype(np.float32)
    return weights, {
        "positive_count": positives,
        "negative_count": negatives,
        "positive_weight": float(positive_weight),
        "negative_weight": float(negative_weight),
        "fit_scope": "fold_train_only",
    }


def _f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_bool = np.asarray(truth, dtype=bool)
    prediction_bool = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth_bool & prediction_bool))
    fp = int(np.sum(~truth_bool & prediction_bool))
    fn = int(np.sum(truth_bool & ~prediction_bool))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 1.0


def _fit_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 19)
    scores = [_f1(labels, probabilities >= threshold) for threshold in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def _split_audit(samples: DevelopmentSamples, split: Mapping[str, np.ndarray]) -> dict[str, Any]:
    train = split["train"]
    validation = split["validation"]
    train_ids = {samples.sample_ids[index] for index in train}
    validation_ids = {samples.sample_ids[index] for index in validation}
    train_inlines = set(map(int, samples.inlines[train].tolist()))
    validation_inlines = set(map(int, samples.inlines[validation].tolist()))
    distances = [abs(left - right) for left in train_inlines for right in validation_inlines]
    return {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "excluded_buffer_samples": len(split["excluded_buffer"]),
        "sample_id_overlap_count": len(train_ids & validation_ids),
        "inline_overlap_count": len(train_inlines & validation_inlines),
        "minimum_train_validation_inline_distance": min(distances) if distances else None,
        "train_inline_extent": [min(train_inlines), max(train_inlines)],
        "validation_inline_extent": [min(validation_inlines), max(validation_inlines)],
    }


def run_invalid_diagnostic(
    samples: DevelopmentSamples,
    *,
    protocol_id: str,
    split: Mapping[str, np.ndarray],
    root_seed: int,
) -> dict[str, Any]:
    """Fit the one frozen baseline on explicitly invalid complement labels."""

    _verify_root_seed(root_seed)
    train_indices = split["train"]
    validation_indices = split["validation"]
    train_x = samples.amplitudes[train_indices]
    validation_x = samples.amplitudes[validation_indices]
    train_y = samples.sparse_positive[train_indices]
    validation_y = samples.sparse_positive[validation_indices]
    stats = fit_zscore(train_x)
    train_x_normalized = normalize(train_x, stats).astype(np.float32)
    validation_x_normalized = normalize(validation_x, stats).astype(np.float32)
    train_weights, class_weight_evidence = _class_weights(train_y)
    validation_weights = np.ones_like(validation_y, dtype=np.float32)
    model = discover_model("fault", MODEL_ID).build(fault_task_spec(), seed=root_seed)
    losses: list[float] = []
    for _ in range(PARAMETER_UPDATES):
        losses.append(float(model.train_batch(train_x_normalized, train_y, train_weights)))
    validation_loss = float(
        model.loss_batch(validation_x_normalized, validation_y, validation_weights)
    )
    train_probability = np.asarray(model.predict_batch(train_x_normalized), dtype=float)
    validation_probability = np.asarray(model.predict_batch(validation_x_normalized), dtype=float)
    threshold, train_f1 = _fit_threshold(train_y.ravel(), train_probability.ravel())
    split_evidence = _split_audit(samples, split)
    fit_sample_ids = [samples.sample_ids[index] for index in train_indices]
    split_hash = hash_payload(
        {
            "train_sample_ids": [samples.sample_ids[index] for index in train_indices],
            "validation_sample_ids": [samples.sample_ids[index] for index in validation_indices],
            "excluded_buffer_sample_ids": [
                samples.sample_ids[index] for index in split["excluded_buffer"]
            ],
            "buffer_inlines": BUFFER_INLINES if "spatial" in protocol_id else 0,
        }
    )
    return {
        "protocol": "fault-p5-r1-invalid-diagnostic-v1",
        "experiment_id": protocol_id,
        "status": "completed_diagnostic",
        "scientifically_invalid": True,
        "diagnostic_only": True,
        "ranking_status": "not_rankable",
        "reason_codes": [
            "UNLABELLED_COMPLEMENT_USED_AS_NEGATIVE",
            "PROTOCOL_MECHANISM_DIAGNOSTIC_ONLY",
        ],
        "model_id": MODEL_ID,
        "root_seed": root_seed,
        "fixed_budget": {
            "parameter_updates": PARAMETER_UPDATES,
            "hpo_performed": False,
            "model_selection_performed": False,
        },
        "operations": {
            "model_built": True,
            "training_invoked": True,
            "development_prediction_computed_in_memory": True,
            "prediction_artifact_written": False,
            "checkpoint_written": False,
            "hpo_invoked": False,
        },
        "split": {**split_evidence, "split_hash": split_hash},
        "label_support": {
            "train_positive_voxels": int(train_y.sum()),
            "train_illegal_complement_voxels": int(train_y.size - train_y.sum()),
            "validation_positive_voxels": int(validation_y.sum()),
            "validation_illegal_complement_voxels": int(validation_y.size - validation_y.sum()),
            "verified_negative_voxels": 0,
        },
        "fold_train_fit": {
            "preprocessing": {
                **stats.to_dict(),
                "fit_scope": "fold_train_only",
                "fit_sample_ids_sha256": hash_payload(fit_sample_ids),
            },
            "class_weights": class_weight_evidence,
            "target_transform": {"name": "identity", "fit_scope": "none"},
            "threshold": {
                "value": threshold,
                "fit_scope": "fold_train_only",
                "train_diagnostic_f1": train_f1,
            },
            "calibration": {"performed": False, "fit_scope": "not_invoked"},
        },
        "diagnostic_metrics": {
            "role": "invalid_protocol_mechanism_only",
            "train_loss_last": losses[-1],
            "validation_loss": validation_loss,
            "validation_average_precision": float(
                average_precision_score(validation_y.ravel(), validation_probability.ravel())
            ),
            "validation_f1_at_train_threshold": _f1(
                validation_y.ravel(), validation_probability.ravel() >= threshold
            ),
        },
        "test_firewall": _firewall(),
    }


def formal_mask_gate(samples: DevelopmentSamples, *, root_seed: int) -> dict[str, Any]:
    """Apply the legal mask and CV contract; stop before building a model if unsupported."""

    _verify_root_seed(root_seed)
    verified_negative = np.zeros_like(samples.sparse_positive, dtype=bool)
    masks = build_fault_masks(
        samples.sparse_positive,
        samples.kinds,
        verified_negative_mask=verified_negative,
    )
    spatial_samples = [
        SpatialSample(
            sample_id=sample_id,
            inline=int(inline),
            positive_count=int(masks.positive[index].sum()),
            verified_negative_count=int(masks.verified_negative[index].sum()),
            proxy_count=int(masks.proxy[index].sum()),
        )
        for index, (sample_id, inline) in enumerate(zip(samples.sample_ids, samples.inlines))
    ]
    plan = build_buffered_spatial_cv(
        spatial_samples,
        requested_n_splits=REQUESTED_FOLDS,
        buffer_inlines=BUFFER_INLINES,
    )
    allowed = plan.effective_n_splits >= 2 and int(masks.verified_negative.sum()) > 0
    if allowed:
        raise FaultP5R01Error("unexpected formal support requires a separately approved execution")
    return {
        "protocol": "fault-p5-r1-formal-gate-v1",
        "experiment_id": "C_legal_mask_buffered_spatial",
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason_codes": [
            "AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING",
            "FEWER_THAN_TWO_LEGAL_BUFFERED_FOLDS",
            "STOPPED_BEFORE_MODEL_BUILD",
        ],
        "mask_counts": {
            "positive": int(masks.positive.sum()),
            "verified_negative": int(masks.verified_negative.sum()),
            "valid_label": int(masks.valid_label.sum()),
            "unknown": int(masks.unknown.sum()),
            "proxy": int(masks.proxy.sum()),
        },
        "mask_invariants": {
            "stick_is_positive": True,
            "unlabelled_is_unknown": True,
            "unknown_valid_label_is_false": True,
            "proxy_is_separate_from_valid_label": True,
        },
        "split": {
            "requested_folds": REQUESTED_FOLDS,
            "effective_folds": plan.effective_n_splits,
            "buffer_inlines": plan.buffer_inlines,
            "status": plan.status,
            "reason": plan.downgrade_reason,
            "split_hash": plan.stable_hash(),
        },
        "operations": {
            "model_built": False,
            "training_invoked": False,
            "prediction_generated": False,
            "metric_computed": False,
        },
        "test_firewall": _firewall(),
    }


def _synthetic_gate(synthetic_manifest: Path | None, registry: Path | None) -> dict[str, Any]:
    reasons: list[str] = []
    manifest: dict[str, Any] = {}
    registry_text = ""
    if synthetic_manifest is None or not synthetic_manifest.is_file():
        reasons.append("REGISTERED_DENSE_SYNTHETIC_DATASET_MISSING")
    else:
        manifest = _read_json(synthetic_manifest)
    if registry is None or not registry.is_file():
        reasons.append("DATA_REGISTRY_UNAVAILABLE")
    else:
        registry_text = registry.read_text(encoding="utf-8")
    dataset_id = str(manifest.get("dataset_id", ""))
    prohibited_fixture = any(
        token in dataset_id.lower() or token in str(manifest.get("source_role", "")).lower()
        for token in ("fixture", "contract", "synthetic_verified_batch")
    )
    train_volumes = tuple(map(str, manifest.get("train_volume_ids", ())))
    validation_volumes = tuple(map(str, manifest.get("validation_volume_ids", ())))
    requirements = {
        "registered": bool(dataset_id and dataset_id in registry_text),
        "dense_ground_truth": manifest.get("dense_ground_truth") is True,
        "independent_generated_volume_split": bool(
            train_volumes
            and validation_volumes
            and set(train_volumes).isdisjoint(validation_volumes)
        ),
        "contract_fixture_prohibited": not prohibited_fixture,
        "source_hash_present": isinstance(manifest.get("source_sha256"), str)
        and len(manifest.get("source_sha256", "")) == 64,
        "label_hash_present": isinstance(manifest.get("label_sha256"), str)
        and len(manifest.get("label_sha256", "")) == 64,
    }
    if manifest and not all(requirements.values()):
        reasons.append("SYNTHETIC_DATA_CONTRACT_INCOMPLETE")
    ready = bool(manifest) and not reasons and all(requirements.values())
    return {
        "lane_id": "synthetic_only",
        "data_ready": ready,
        "train_allowed": ready,
        "rank_allowed": ready,
        "ranking_scope": "within_synthetic_only_lane" if ready else None,
        "reason_codes": reasons or ["SYNTHETIC_LANE_CONTRACT_SATISFIED"],
        "requirements": requirements,
        "hashes": {
            "source": hash_payload({"manifest": manifest.get("source_sha256"), "registry": hash_payload(registry_text)}),
            "label": hash_payload({"label": manifest.get("label_sha256"), "dense": manifest.get("dense_ground_truth")}),
            "split": hash_payload({"train": train_volumes, "validation": validation_volumes}),
            "config": hash_payload({"lane": "synthetic_only", "seed": ROOT_SEED}),
        },
        "seed": ROOT_SEED,
        "test_firewall": _firewall(),
    }


def build_r0_gates(
    *,
    lock_evidence: Mapping[str, Any],
    synthetic_manifest: Path | None = None,
    data_registry: Path | None = None,
) -> list[dict[str, Any]]:
    synthetic = _synthetic_gate(synthetic_manifest, data_registry)
    source_hashes = dict(lock_evidence["input_sha256"])
    weak_semantics = {
        "stick": "positive",
        "unlabelled": "unknown",
        "valid_label_mask_for_unlabelled": False,
        "proxy_separate": True,
    }
    weak = {
        "lane_id": "masked_weak_label",
        "data_ready": True,
        "train_allowed": False,
        "rank_allowed": False,
        "ranking_scope": None,
        "reason_codes": [
            "MASKED_WEAK_OBJECTIVE_NOT_CONFIGURED",
            "AUDITED_EVALUATION_NEGATIVES_MISSING",
        ],
        "requirements": weak_semantics,
        "hashes": {
            "source": hash_payload(source_hashes),
            "label": hash_payload({"fault_points": source_hashes["fault_points"], **weak_semantics}),
            "split": hash_payload({"development_inline_range": lock_evidence["development_inline_range"], "buffer": BUFFER_INLINES}),
            "config": hash_payload({"lane": "masked_weak_label", "seed": ROOT_SEED, "objective": None}),
        },
        "seed": ROOT_SEED,
        "test_firewall": _firewall(),
    }
    cv = _read_json(FORMAL_CV)
    stage3 = _read_json(STAGE3_DATA)
    verified_negative = int(stage3["coverage"]["negative_provenance"]["verified_negative_labels"])
    effective_folds = int(cv.get("effective_n_splits", 0))
    formal_ready = verified_negative > 0 and effective_folds >= 2
    formal_reasons: list[str] = []
    if verified_negative == 0:
        formal_reasons.append("AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING")
    if effective_folds < 2:
        formal_reasons.append("FEWER_THAN_TWO_LEGAL_BUFFERED_FOLDS")
    formal = {
        "lane_id": "formal_audited",
        "data_ready": formal_ready,
        "train_allowed": formal_ready,
        "rank_allowed": formal_ready,
        "ranking_scope": "within_formal_audited_lane" if formal_ready else None,
        "reason_codes": formal_reasons or ["FORMAL_AUDITED_CONTRACT_SATISFIED"],
        "requirements": {
            "coverage_audited_verified_negative_count": verified_negative,
            "minimum_legal_buffered_folds": 2,
            "effective_legal_buffered_folds": effective_folds,
            "each_fold_train_and_validation_has_both_classes": formal_ready,
        },
        "hashes": {
            "source": hash_payload(source_hashes),
            "label": hash_file(STAGE3_DATA),
            "split": hash_file(FORMAL_CV),
            "config": hash_payload({"lane": "formal_audited", "seed": ROOT_SEED, "buffer": BUFFER_INLINES, "requested_folds": REQUESTED_FOLDS}),
        },
        "seed": ROOT_SEED,
        "test_firewall": _firewall(),
    }
    gates = [synthetic, weak, formal]
    if tuple(gate["lane_id"] for gate in gates) != LANES:
        raise AssertionError("R0 lane roster changed")
    for gate in gates:
        if set(gate["hashes"]) != {"source", "label", "split", "config"}:
            raise AssertionError("R0 gate hash contract changed")
    return gates


def run_r01(
    output_dir: Path,
    *,
    seismic_amplitude: Path,
    fault_points: Path,
    seismic_index: Path,
    development_lock: Path,
    synthetic_manifest: Path | None = None,
    data_registry: Path | None = None,
    root_seed: int = ROOT_SEED,
) -> dict[str, Any]:
    """Execute R0 and the bounded R1 diagnostic without any holdout surface."""

    _verify_root_seed(root_seed)
    lock = verify_development_lock(
        development_lock,
        seismic_amplitude=seismic_amplitude,
        fault_points=fault_points,
        seismic_index=seismic_index,
    )
    gates = build_r0_gates(
        lock_evidence=lock,
        synthetic_manifest=synthetic_manifest,
        data_registry=data_registry,
    )
    samples, sampling = load_development_samples(
        seismic_amplitude=seismic_amplitude,
        fault_points=fault_points,
        seismic_index=seismic_index,
        lock_evidence=lock,
        root_seed=root_seed,
    )
    illegal_label_hash = hashlib.sha256(
        np.ascontiguousarray(samples.sparse_positive).tobytes()
        + json.dumps(samples.sample_ids, separators=(",", ":")).encode("utf-8")
        + b"unlabelled_complement_as_zero"
    ).hexdigest()
    random_result = run_invalid_diagnostic(
        samples,
        protocol_id="A_illegal_complement_random_split",
        split=_random_split(samples, root_seed),
        root_seed=root_seed,
    )
    spatial_result = run_invalid_diagnostic(
        samples,
        protocol_id="B_illegal_complement_buffered_spatial_split",
        split=_spatial_split(samples, BUFFER_INLINES),
        root_seed=root_seed,
    )
    diagnostic_config_hash = hash_payload(
        {
            "root_seed": root_seed,
            "model_id": MODEL_ID,
            "patch_shape": PATCH_SHAPE,
            "development_inline_count": DEVELOPMENT_INLINE_COUNT,
            "parameter_updates": PARAMETER_UPDATES,
            "buffer_inlines": BUFFER_INLINES,
            "hpo": False,
        }
    )
    for result in (random_result, spatial_result):
        result["hashes"] = {
            "source": hash_payload(lock["input_sha256"]),
            "label": illegal_label_hash,
            "split": result["split"]["split_hash"],
            "config": diagnostic_config_hash,
        }
    formal_result = formal_mask_gate(samples, root_seed=root_seed)
    formal_result["hashes"] = {
        "source": hash_payload(lock["input_sha256"]),
        "label": hash_payload(
            {
                "sparse_positive_sha256": hashlib.sha256(
                    np.ascontiguousarray(samples.sparse_positive).tobytes()
                ).hexdigest(),
                "verified_negative_count": 0,
                "unlabelled_semantics": "unknown_valid_label_false",
                "proxy_separate": True,
            }
        ),
        "split": formal_result["split"]["split_hash"],
        "config": hash_payload(
            {
                "lane": "formal_audited",
                "root_seed": root_seed,
                "requested_folds": REQUESTED_FOLDS,
                "buffer_inlines": BUFFER_INLINES,
            }
        ),
    }
    metric_names = ("validation_average_precision", "validation_f1_at_train_threshold")
    diagnostic_delta = {
        name: float(
            random_result["diagnostic_metrics"][name]
            - spatial_result["diagnostic_metrics"][name]
        )
        for name in metric_names
    }
    r0_payload = {
        "protocol": "fault-p5-r0-lane-gates-v1",
        "track_id": "fault",
        "lanes": gates,
        "lane_isolation": {
            "cross_lane_ranking_forbidden": True,
            "labels_and_negatives_never_promoted_between_lanes": True,
        },
    }
    r1_payload = {
        "protocol": "fault-p5-r1-mechanism-v1",
        "track_id": "fault",
        "root_seed": root_seed,
        "model_id": MODEL_ID,
        "input_lock": lock,
        "hashes": {
            "source": hash_payload(lock["input_sha256"]),
            "label": illegal_label_hash,
            "split": hash_payload(
                {
                    "A": random_result["split"]["split_hash"],
                    "B": spatial_result["split"]["split_hash"],
                    "C": formal_result["split"]["split_hash"],
                }
            ),
            "config": diagnostic_config_hash,
        },
        "sampling": sampling,
        "experiments": [random_result, spatial_result, formal_result],
        "diagnostic_delta_random_minus_spatial": {
            "role": "scientifically_invalid_protocol_mechanism_only",
            "values": diagnostic_delta,
            "formal_performance_claim": False,
            "winner_selected": False,
        },
        "ranking_status": "not_rankable",
        "future_boundary": "ten-or-more-model fair comparison is deferred to R2",
        "test_firewall": _firewall(),
    }
    summary = {
        "protocol": "fault-p5-r01-summary-v1",
        "track_id": "fault",
        "status": "completed_with_formal_lane_blocked",
        "ranking_status": "not_rankable",
        "root_seed": root_seed,
        "r0": {
            gate["lane_id"]: {
                key: gate[key]
                for key in ("data_ready", "train_allowed", "rank_allowed", "reason_codes")
            }
            for gate in gates
        },
        "r1": {
            "A": random_result["status"],
            "B": spatial_result["status"],
            "C": formal_result["status"],
            "official_metric_or_winner_produced": False,
        },
        "test_firewall": _firewall(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "r0_lane_gates.json", r0_payload)
    atomic_write_json(output_dir / "r1_mechanism.json", r1_payload)
    atomic_write_json(output_dir / "summary.json", summary)
    readme = """# Fault P5.1 R0/R1 portable evidence\n\nThis directory contains lane gates and a bounded development-only mechanism audit.\nA/B intentionally use an invalid complement-as-negative label only to diagnose split leakage;\nthey are not rankable. C uses the legal mask and buffered split contract and stops before\nmodel construction because audited negatives and two legal folds are absent. No holdout,\nprediction archive, checkpoint, winner, HPO result, or official metric is present.\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    artifact_names = ("r0_lane_gates.json", "r1_mechanism.json", "summary.json", "README.md")
    manifest = {
        "protocol": "fault-p5-r01-artifact-manifest-v1",
        "artifacts": {
            name: {
                "sha256": hash_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in artifact_names
        },
        "input_hashes": lock["input_sha256"],
        "source_code_hashes": {
            "fault_p5_r01.py": hash_file(Path(__file__).resolve()),
            "p4_contract.py": hash_file(TRACK_DIR / "p4_contract.py"),
            "p4_split.py": hash_file(TRACK_DIR / "p4_split.py"),
            "p5_r01_development_lock.json": hash_file(development_lock),
        },
        "config_hash": hash_payload(
            {
                "root_seed": root_seed,
                "model_id": MODEL_ID,
                "patch_shape": PATCH_SHAPE,
                "development_inline_count": DEVELOPMENT_INLINE_COUNT,
                "updates": PARAMETER_UPDATES,
                "buffer_inlines": BUFFER_INLINES,
            }
        ),
        "test_firewall": _firewall(),
    }
    atomic_write_json(output_dir / "artifact_manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-segy", type=Path, required=True)
    parser.add_argument("--fault-points", type=Path, required=True)
    parser.add_argument("--seismic-index", type=Path, required=True)
    parser.add_argument("--development-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--synthetic-manifest", type=Path)
    parser.add_argument("--data-registry", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_r01(
        args.output_dir,
        seismic_amplitude=args.development_segy,
        fault_points=args.fault_points,
        seismic_index=args.seismic_index,
        development_lock=args.development_lock,
        synthetic_manifest=args.synthetic_manifest,
        data_registry=args.data_registry,
        root_seed=args.root_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
