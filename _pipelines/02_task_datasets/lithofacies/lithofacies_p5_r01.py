#!/usr/bin/env python3
"""Development-only P5.1 R0/R1 protocol checks for GM09 lithofacies.

``prepare`` is the only HDF5-facing command.  It opens exactly ``train.h5``,
validates the four development mother families, reverses the historical stored
normalization, and writes an ignored NPZ envelope plus portable R0 evidence.
``run`` consumes only that envelope and executes the fixed-SGD R1 mechanism
comparison.  Neither command has a holdout path, loader, metric, or inference
entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    TEST_FAMILY,
    _recover_physical,
    lithofacies_task_spec,
    sample_id,
    validate_p4_sample,
)
from pipeline_contract import (  # noqa: E402
    LOG_CHANNELS,
    TARGET_CURVE_TYPE,
    TARGET_SOURCE,
    classification_metrics_from_confusion,
)


ROOT_SEED = 2693
NUM_CLASSES = len(CLASS_NAMES)
CONTEXT_LENGTH = 33
R0_SCHEMA = "lithofacies-p5.1-r0-v1"
BATCH_SCHEMA = "lithofacies-p5.1-development-envelope-v1"
R1_SCHEMA = "lithofacies-p5.1-r1-v1"
EXPECTED_CONDITIONS = 8
EXPECTED_CELLS = EXPECTED_CONDITIONS * EFFECTIVE_N_SPLITS
CLASSIFIER_CONFIG: dict[str, Any] = {
    "estimator": "sklearn.linear_model.SGDClassifier",
    "loss": "log_loss",
    "penalty": "l2",
    "alpha": 1e-4,
    "max_iter": 64,
    "tol": None,
    "shuffle": True,
    "random_state": ROOT_SEED,
    "early_stopping": False,
    "average": False,
    "inference": "stable_softmax_over_decision_function",
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _track_owned(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != TRACK_DIR.resolve() and TRACK_DIR.resolve() not in resolved.parents:
        raise ValueError(f"R0/R1 artifacts must stay below {TRACK_DIR}")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = _track_owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path = _track_owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def lane_matrix(*, finite_center_md_count: int) -> list[dict[str, Any]]:
    """Return orthogonal modality/task lanes without manufacturing S samples."""
    rows: list[dict[str, Any]] = []
    for modality in ("W", "M"):
        for task_lane in ("P", "S"):
            if task_lane == "P":
                rows.append(
                    {
                        "modality_lane": modality,
                        "task_lane": task_lane,
                        "lane_id": f"{modality}-{task_lane}",
                        "status": "available",
                        "rank_policy": "separate_lane_only",
                    }
                )
            else:
                rows.append(
                    {
                        "modality_lane": modality,
                        "task_lane": task_lane,
                        "lane_id": f"{modality}-{task_lane}",
                        "status": "not_rankable",
                        "reason": (
                            "development archive has no finite center_md_m; TWT, row order, "
                            "interval midpoint, and repeated center labels are forbidden substitutes"
                        ),
                        "finite_center_md_count": int(finite_center_md_count),
                    }
                )
    return rows


def _read_development_hdf5(dataset_root: Path) -> tuple[list[dict[str, Any]], Path]:
    """Read the explicit development archive; no alternative split is accepted."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific gate
        raise RuntimeError("R0 prepare requires an interpreter with h5py") from exc
    path = dataset_root.resolve() / "train.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            sample = {
                "seismic_patch": group["seismic_patch"][()],
                "well_log_seq": group["well_log_seq"][()],
                "label": group["label"][()],
                "position": json.loads(group.attrs["position"]),
                "meta": json.loads(group.attrs["meta"]),
            }
            validate_p4_sample(sample)
            samples.append(sample)
    if not samples:
        raise ValueError("development train archive is empty")
    families = {str(sample["meta"]["family_id"]) for sample in samples}
    if families != set(DEVELOPMENT_FAMILIES) or TEST_FAMILY in families:
        raise RuntimeError(
            f"development firewall requires exactly {list(DEVELOPMENT_FAMILIES)}, got {sorted(families)}"
        )
    partitions = {str(sample["meta"].get("partition")) for sample in samples}
    if not partitions.issubset({"train", "guard"}):
        raise RuntimeError(f"development archive contains forbidden partitions: {sorted(partitions)}")
    for sample in samples:
        trace = sample["meta"].get("label_trace", {})
        if trace.get("source") != TARGET_SOURCE or trace.get("curve_type") != TARGET_CURVE_TYPE:
            raise RuntimeError("development label provenance changed from GM09/GENETIC FACIES")
    return samples, path


def _physical_arrays(samples: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    physical = [_recover_physical(sample) for sample in samples]
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    families = np.asarray([str(sample["meta"]["family_id"]) for sample in samples], dtype=np.str_)
    wells = np.asarray([str(sample["position"]["well_name"]) for sample in samples], dtype=np.str_)
    identifiers = np.asarray([sample_id(sample) for sample in samples], dtype=np.str_)
    if len(set(identifiers.tolist())) != len(samples):
        raise RuntimeError("development sample IDs are not unique")
    interval_keys = np.asarray(
        [
            "|".join(
                (
                    str(sample["position"]["well_name"]),
                    str(sample["meta"]["label_trace"]["member"]),
                    str(sample["meta"]["label_trace"]["excel_row"]),
                )
            )
            for sample in samples
        ],
        dtype=np.str_,
    )
    centers = np.asarray(
        [
            np.nan
            if sample["position"].get("center_md_m") is None
            else float(sample["position"]["center_md_m"])
            for sample in samples
        ],
        dtype=np.float64,
    )
    times = np.asarray(
        [float(sample["position"].get("time_ms", np.nan)) for sample in samples], dtype=np.float64
    )
    return {
        "physical_logs": np.stack([values for values, _, _ in physical]).astype(np.float32),
        "log_masks": np.stack([masks for _, masks, _ in physical]).astype(np.uint8),
        "physical_seismic": np.stack([seismic for _, _, seismic in physical]).astype(np.float32),
        "labels": labels,
        "families": families,
        "wells": wells,
        "sample_ids": identifiers,
        "interval_keys": interval_keys,
        "center_md_m": centers,
        "time_ms": times,
    }


def build_r0_contract(
    arrays: Mapping[str, np.ndarray], *, development_sha256: str
) -> dict[str, Any]:
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    families = np.asarray(arrays["families"]).astype(str)
    finite_md = int(np.isfinite(np.asarray(arrays["center_md_m"], dtype=np.float64)).sum())
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)[:NUM_CLASSES]
    if set(np.unique(labels).tolist()) != set(range(NUM_CLASSES)):
        raise RuntimeError("development data no longer contains the fixed nine-class schema")
    task_spec = lithofacies_task_spec()
    if tuple(task_spec.primary_metrics) != ("fixed_schema_macro_f1",):
        raise RuntimeError("TaskSpec primary metric drifted from fixed_schema_macro_f1")
    contract: dict[str, Any] = {
        "schema_version": R0_SCHEMA,
        "status": "PASS_WITH_STRUCTURED_S_STOP",
        "task_id": "gm09_genetic_facies_9class",
        "root_seed": ROOT_SEED,
        "label_contract": {
            "source": TARGET_SOURCE,
            "curve_type": TARGET_CURVE_TYPE,
            "field": "Litho Class",
            "class_names": list(CLASS_NAMES),
            "class_count": NUM_CLASSES,
            "excluded": ["LITH", "UNKNOWN", "UNDEFINED", "out-of-interval"],
            "target_transform": "identity_class_id",
        },
        "metric_contract": {
            "primary_metric": "fixed_schema_macro_f1",
            "primary_metric_count": 1,
            "supported_class_macro_f1_role": "diagnostic_only",
            "worst_family_metric": "worst_family_fixed_schema_macro_f1",
            "zero_support_policy": "retain all nine classes as finite zeros",
        },
        "lane_contract": {
            "axes": {"modality": ["W", "M"], "task": ["P", "S"]},
            "lanes": lane_matrix(finite_center_md_count=finite_md),
            "cross_lane_ranking": False,
        },
        "split_contract": {
            "splitter": "leave_one_mother_family_out",
            "requested_n_splits": 5,
            "effective_n_splits": EFFECTIVE_N_SPLITS,
            "development_families": list(DEVELOPMENT_FAMILIES),
            "group_key": "mother_family",
            "split_before": ["normalization", "missing_processing", "class_weights", "window_selection"],
        },
        "preprocessing_contract": {
            "fit_scope": "fold_train_only",
            "log_channels": list(LOG_CHANNELS),
            "missing_contract": "13 normalized values plus 13 binary observed masks",
            "class_weighting": "sqrt_inverse_frequency_fold_train_only",
            "threshold": "none_closed_set_multiclass",
            "calibration": "none_in_R0_R1",
        },
        "sealed_holdout": {
            "identity": TEST_FAMILY,
            "identity_only": True,
            "prior_test_consumed": True,
            "fresh_blind": False,
            "physical_test_accessed": False,
            "known_holdout_artifacts_read": False,
        },
        "completion_contract": {
            "future_minimum_completion_rate": 0.80,
            "full_fold_seed_axes_required": True,
            "seventy_five_percent_rankable": False,
            "r1_formal_ranking": False,
        },
        "development_evidence": {
            "data_basename": "train.h5",
            "data_sha256": development_sha256,
            "sample_count": int(labels.size),
            "family_counts": dict(sorted(Counter(families.tolist()).items())),
            "class_support": class_counts.tolist(),
            "well_log_shape": list(np.asarray(arrays["physical_logs"]).shape[1:]),
            "well_log_with_mask_shape": [2 * len(LOG_CHANNELS), CONTEXT_LENGTH],
            "seismic_shape": list(np.asarray(arrays["physical_seismic"]).shape[1:]),
            "finite_center_md_count": finite_md,
        },
    }
    contract["contract_hash"] = _stable_hash(contract)
    return contract


def prepare(dataset_root: Path, batch_file: Path, output_dir: Path) -> dict[str, Any]:
    batch_file = _track_owned(batch_file)
    output_dir = _track_owned(output_dir)
    samples, input_path = _read_development_hdf5(dataset_root)
    arrays = _physical_arrays(samples)
    input_hash = _hash_file(input_path)
    contract = build_r0_contract(arrays, development_sha256=input_hash)
    manifest = {
        "schema_version": BATCH_SCHEMA,
        "task_id": contract["task_id"],
        "root_seed": ROOT_SEED,
        "class_names": list(CLASS_NAMES),
        "development_data": {"basename": input_path.name, "sha256": input_hash},
        "r0_contract_hash": contract["contract_hash"],
        "development_families": list(DEVELOPMENT_FAMILIES),
        "sample_count": int(len(samples)),
        "physical_test_accessed": False,
        "known_holdout_artifacts_read": False,
    }
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(batch_file, manifest=np.asarray(_canonical_json(manifest)), **arrays)
    _atomic_json(output_dir / "r0_contract.json", contract)
    return {
        "status": contract["status"],
        "r0_contract": "r0_contract.json",
        "r0_contract_hash": contract["contract_hash"],
        "batch_basename": batch_file.name,
        "batch_sha256": _hash_file(batch_file),
        "development_data_sha256": input_hash,
        "sample_count": len(samples),
        "physical_test_accessed": False,
    }


def load_batch(batch_file: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(batch_file, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
    required = {
        "physical_logs", "log_masks", "physical_seismic", "labels", "families", "wells",
        "sample_ids", "interval_keys", "center_md_m", "time_ms",
    }
    if manifest.get("schema_version") != BATCH_SCHEMA or required - set(arrays):
        raise ValueError("unknown or incomplete R0 development envelope")
    if tuple(manifest.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("R0 envelope changed the fixed GM09 schema")
    if manifest.get("physical_test_accessed") is not False:
        raise RuntimeError("R0 envelope violates the sealed-holdout firewall")
    families = set(np.asarray(arrays["families"]).astype(str).tolist())
    if families != set(DEVELOPMENT_FAMILIES) or TEST_FAMILY in families:
        raise RuntimeError("R0 envelope contains a non-development mother family")
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError("R0 envelope labels violate the fixed nine-class schema")
    logs = np.asarray(arrays["physical_logs"])
    masks = np.asarray(arrays["log_masks"])
    seismic = np.asarray(arrays["physical_seismic"])
    if logs.shape != (len(labels), len(LOG_CHANNELS), CONTEXT_LENGTH):
        raise ValueError(f"unexpected physical log shape: {logs.shape}")
    if masks.shape != logs.shape or not np.isin(masks, (0, 1)).all():
        raise ValueError("log missing masks are not binary/aligned")
    if seismic.shape != (len(labels), 3, 3, CONTEXT_LENGTH):
        raise ValueError(f"unexpected seismic shape: {seismic.shape}")
    if not np.isfinite(logs).all() or not np.isfinite(seismic).all():
        raise ValueError("R0 envelope contains NaN/Inf features")
    return arrays, manifest


def random_kfold4(sample_count: int) -> list[dict[str, Any]]:
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=EFFECTIVE_N_SPLITS, shuffle=True, random_state=ROOT_SEED)
    return [
        {"fold_id": fold_id, "train_indices": train, "validation_indices": validation}
        for fold_id, (train, validation) in enumerate(splitter.split(np.arange(sample_count)))
    ]


def logo4(families: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(families).astype(str)
    if set(values.tolist()) != set(DEVELOPMENT_FAMILIES):
        raise ValueError("LOGO4 input does not contain the exact development family roster")
    folds = []
    for fold_id, held_out in enumerate(DEVELOPMENT_FAMILIES):
        validation = np.flatnonzero(values == held_out)
        train = np.flatnonzero(values != held_out)
        if not train.size or not validation.size:
            raise ValueError(f"LOGO4 family {held_out} is empty")
        folds.append(
            {
                "fold_id": fold_id,
                "train_indices": train,
                "validation_indices": validation,
                "validation_family": held_out,
            }
        )
    return folds


def _fit_fold_preprocessing(
    arrays: Mapping[str, np.ndarray], train_indices: np.ndarray, validation_indices: np.ndarray,
    *, modality: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    logs = np.asarray(arrays["physical_logs"], dtype=np.float64)
    masks = np.asarray(arrays["log_masks"], dtype=bool)
    seismic = np.asarray(arrays["physical_seismic"], dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    normalized_logs = {
        "train": np.zeros_like(logs[train_indices], dtype=np.float32),
        "validation": np.zeros_like(logs[validation_indices], dtype=np.float32),
    }
    effective_masks = {
        "train": masks[train_indices].copy(),
        "validation": masks[validation_indices].copy(),
    }
    log_stats: list[dict[str, Any] | None] = []
    unseen_channels: list[str] = []
    for channel_index, channel in enumerate(LOG_CHANNELS):
        observed = logs[train_indices, channel_index][masks[train_indices, channel_index]]
        if observed.size < 2:
            log_stats.append(None)
            unseen_channels.append(channel)
            effective_masks["train"][:, channel_index] = False
            effective_masks["validation"][:, channel_index] = False
            continue
        mean = float(observed.mean())
        std = float(observed.std() + 1e-8)
        log_stats.append({"mean": mean, "std": std})
        for split, indices in (("train", train_indices), ("validation", validation_indices)):
            selected = effective_masks[split][:, channel_index]
            values = logs[indices, channel_index]
            normalized_logs[split][:, channel_index][selected] = (
                (values[selected] - mean) / std
            ).astype(np.float32)
    well = {
        split: np.concatenate((normalized_logs[split], effective_masks[split].astype(np.float32)), axis=1)
        for split in ("train", "validation")
    }
    seismic_stats: dict[str, float] | None = None
    seismic_values: dict[str, np.ndarray] = {}
    if modality == "M":
        mean = float(seismic[train_indices].mean())
        std = float(seismic[train_indices].std() + 1e-8)
        seismic_stats = {"mean": mean, "std": std}
        seismic_values = {
            "train": ((seismic[train_indices] - mean) / std).astype(np.float32),
            "validation": ((seismic[validation_indices] - mean) / std).astype(np.float32),
        }
    counts = np.bincount(labels[train_indices], minlength=NUM_CLASSES)[:NUM_CLASSES]
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    supported = counts > 0
    frequency = counts[supported] / counts[supported].sum()
    weights[supported] = 1.0 / np.sqrt(frequency)
    weights[supported] /= weights[supported].mean()
    sample_weights = weights[labels[train_indices]]
    fit_ids = sorted(np.asarray(arrays["sample_ids"])[train_indices].astype(str).tolist())
    fit_families = sorted(np.unique(np.asarray(arrays["families"])[train_indices].astype(str)).tolist())
    evidence = {
        "fit_scope": "fold_train_only",
        "fit_sample_ids_hash": hashlib.sha256("\n".join(fit_ids).encode("utf-8")).hexdigest(),
        "fit_families": fit_families,
        "train_class_support": counts.tolist(),
        "class_weights": weights.tolist(),
        "log_stats": log_stats,
        "seismic_stats": seismic_stats,
        "unseen_log_channels_masked": unseen_channels,
        "target_transform": "identity_class_id",
        "threshold": None,
        "calibration": None,
    }
    if modality == "W":
        seismic_values = {"train": np.empty((len(train_indices), 0)), "validation": np.empty((len(validation_indices), 0))}
    return well["train"], well["validation"], sample_weights, {
        **evidence,
        "seismic_train": seismic_values["train"],
        "seismic_validation": seismic_values["validation"],
    }


def _features(well: np.ndarray, seismic: np.ndarray, *, modality: str, representation: str) -> np.ndarray:
    if representation == "center":
        center = well.shape[-1] // 2
        pieces = [well[:, :, center]]
        if modality == "M":
            pieces.append(seismic[:, :, :, center].reshape(len(well), -1))
    elif representation == "window":
        pieces = [well.reshape(len(well), -1)]
        if modality == "M":
            pieces.append(seismic.reshape(len(well), -1))
    else:
        raise ValueError(f"unknown representation: {representation}")
    result = np.concatenate(pieces, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("R1 fold features contain NaN/Inf")
    return result


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
    result = classification_metrics_from_confusion(confusion)
    return {
        "accuracy": result["accuracy"],
        "fixed_schema_macro_f1": result["macro_f1"],
        "supported_class_macro_f1": result["supported_class_macro_f1"],
        "supported_class_metric_role": "diagnostic_only",
        "per_class": result["per_class"],
        "confusion_matrix": result["confusion_matrix"],
        "evaluated_samples": result["evaluated_samples"],
    }


def _shifted_window_pairs(
    arrays: Mapping[str, np.ndarray], train_indices: np.ndarray, validation_indices: np.ndarray,
) -> int:
    """Count cross-fold pairs sharing an exact shifted log window within one source interval."""
    logs = np.round(np.asarray(arrays["physical_logs"], dtype=np.float64), decimals=6)
    masks = np.asarray(arrays["log_masks"], dtype=bool)
    wells = np.asarray(arrays["wells"]).astype(str)
    intervals = np.asarray(arrays["interval_keys"]).astype(str)
    train_groups: dict[tuple[str, str], list[int]] = {}
    validation_groups: dict[tuple[str, str], list[int]] = {}
    for index in train_indices:
        train_groups.setdefault((wells[index], intervals[index]), []).append(int(index))
    for index in validation_indices:
        validation_groups.setdefault((wells[index], intervals[index]), []).append(int(index))
    pairs = 0
    for key in sorted(set(train_groups) & set(validation_groups)):
        for left in train_groups[key]:
            for right in validation_groups[key]:
                matched = False
                for shift in range(1, min(17, CONTEXT_LENGTH - 16)):
                    comparisons = (
                        (slice(shift, None), slice(None, -shift)),
                        (slice(None, -shift), slice(shift, None)),
                    )
                    for left_slice, right_slice in comparisons:
                        left_mask = masks[left, :, left_slice]
                        right_mask = masks[right, :, right_slice]
                        if not np.array_equal(left_mask, right_mask) or int(left_mask.sum()) < 8:
                            continue
                        if np.allclose(
                            logs[left, :, left_slice][left_mask],
                            logs[right, :, right_slice][right_mask],
                            rtol=0.0,
                            atol=1e-6,
                        ):
                            matched = True
                            break
                    if matched:
                        break
                pairs += int(matched)
    return pairs


def leakage_diagnostics(
    arrays: Mapping[str, np.ndarray], train_indices: np.ndarray, validation_indices: np.ndarray,
) -> dict[str, Any]:
    def overlap(name: str) -> list[str]:
        values = np.asarray(arrays[name]).astype(str)
        return sorted(set(values[train_indices].tolist()) & set(values[validation_indices].tolist()))

    family_overlap = overlap("families")
    well_overlap = overlap("wells")
    interval_overlap = overlap("interval_keys")
    return {
        "family_overlap": family_overlap,
        "family_overlap_count": len(family_overlap),
        "well_overlap": well_overlap,
        "well_overlap_count": len(well_overlap),
        "source_interval_overlap_count": len(interval_overlap),
        "exact_shifted_window_pair_count": _shifted_window_pairs(
            arrays, train_indices, validation_indices
        ),
    }


def _fold_hash(
    sample_ids: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray,
) -> str:
    payload = {
        "train_sample_ids": sorted(sample_ids[train_indices].astype(str).tolist()),
        "validation_sample_ids": sorted(sample_ids[validation_indices].astype(str).tolist()),
    }
    return _stable_hash(payload)


def _run_cell(
    arrays: Mapping[str, np.ndarray], *, modality: str, representation: str,
    split_name: str, split_role: str, fold: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from sklearn.linear_model import SGDClassifier

    train_indices = np.asarray(fold["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(fold["validation_indices"], dtype=np.int64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    families = np.asarray(arrays["families"]).astype(str)
    leakage = leakage_diagnostics(arrays, train_indices, validation_indices)
    if split_role == "legal_grouped" and (
        leakage["family_overlap_count"] or leakage["well_overlap_count"]
        or leakage["source_interval_overlap_count"] or leakage["exact_shifted_window_pair_count"]
    ):
        raise RuntimeError(f"LOGO4 leakage detected: {leakage}")
    train_well, validation_well, sample_weights, preprocessing = _fit_fold_preprocessing(
        arrays, train_indices, validation_indices, modality=modality
    )
    validation_family_set = sorted(set(families[validation_indices].tolist()))
    if set(preprocessing["fit_families"]) & set(validation_family_set) and split_role == "legal_grouped":
        raise RuntimeError("LOGO4 validation family entered fold-train preprocessing")
    seismic_train = np.asarray(preprocessing.pop("seismic_train"))
    seismic_validation = np.asarray(preprocessing.pop("seismic_validation"))
    x_train = _features(
        train_well, seismic_train, modality=modality, representation=representation
    )
    x_validation = _features(
        validation_well, seismic_validation, modality=modality, representation=representation
    )
    classifier_kwargs = {
        key: value
        for key, value in CLASSIFIER_CONFIG.items()
        if key not in {"estimator", "inference"}
    }
    classifier = SGDClassifier(**classifier_kwargs)
    classifier.fit(x_train, labels[train_indices], sample_weight=sample_weights)
    scores = np.asarray(classifier.decision_function(x_validation), dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(classifier.classes_):
        raise RuntimeError(f"unexpected multiclass decision shape: {scores.shape}")
    scores -= scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(scores, -745.0, 0.0))
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    fixed_probabilities = np.zeros((len(validation_indices), NUM_CLASSES), dtype=np.float64)
    fixed_probabilities[:, np.asarray(classifier.classes_, dtype=np.int64)] = probabilities
    if not np.isfinite(fixed_probabilities).all() or not np.allclose(
        fixed_probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise RuntimeError("SGDClassifier emitted invalid fixed-schema probabilities")
    predictions = fixed_probabilities.argmax(axis=1).astype(np.int64)
    metrics = _metrics(labels[validation_indices], predictions)
    result = {
        "schema_version": R1_SCHEMA,
        "status": "PASS",
        "task_id": "gm09_genetic_facies_9class",
        "task_lane": "P",
        "modality_lane": modality,
        "representation": representation,
        "split_name": split_name,
        "split_role": split_role,
        "rank_eligible": False,
        "rank_status": "not_rankable" if split_role == "diagnostic_only" else "not_ranked_R1_mechanism_only",
        "fold_id": int(fold["fold_id"]),
        "root_seed": ROOT_SEED,
        "classifier_config_hash": _stable_hash(CLASSIFIER_CONFIG),
        "split_hash": _fold_hash(np.asarray(arrays["sample_ids"]), train_indices, validation_indices),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_families": sorted(set(families[train_indices].tolist())),
        "validation_families": validation_family_set,
        "preprocessing": preprocessing,
        "leakage": leakage,
        "metrics": metrics,
        "physical_test_accessed": False,
        "formal_model_ranking": False,
        "hpo": False,
    }
    return result, validation_indices, predictions


def _condition_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for representation, random_name in (
        ("center", "random_depth_kfold4"),
        ("window", "random_window_kfold4"),
    ):
        for split_name, role in ((random_name, "diagnostic_only"), ("logo4", "legal_grouped")):
            for modality in ("W", "M"):
                specs.append(
                    {
                        "representation": representation,
                        "split_name": split_name,
                        "split_role": role,
                        "modality": modality,
                    }
                )
    return specs


def run_r1(batch_file: Path, output_dir: Path) -> tuple[dict[str, Any], int]:
    batch_file = batch_file.resolve()
    output_dir = _track_owned(output_dir)
    arrays, batch_manifest = load_batch(batch_file)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    families = np.asarray(arrays["families"]).astype(str)
    sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
    random_folds = random_kfold4(len(labels))
    grouped_folds = logo4(families)
    results: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for spec in _condition_specs():
        folds = random_folds if spec["split_role"] == "diagnostic_only" else grouped_folds
        oof_predictions = np.full(len(labels), -1, dtype=np.int64)
        coverage = np.zeros(len(labels), dtype=np.int64)
        cell_results: list[dict[str, Any]] = []
        for fold in folds:
            try:
                result, validation_indices, predictions = _run_cell(arrays, fold=fold, **spec)
                oof_predictions[validation_indices] = predictions
                coverage[validation_indices] += 1
            except Exception as exc:  # fail-closed evidence; never substitute another split/seed
                result = {
                    "schema_version": R1_SCHEMA,
                    "status": "FAIL",
                    "task_lane": "P",
                    "modality_lane": spec["modality"],
                    "representation": spec["representation"],
                    "split_name": spec["split_name"],
                    "split_role": spec["split_role"],
                    "rank_eligible": False,
                    "rank_status": "not_rankable",
                    "fold_id": int(fold["fold_id"]),
                    "root_seed": ROOT_SEED,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-10:],
                    "physical_test_accessed": False,
                    "formal_model_ranking": False,
                    "hpo": False,
                }
            results.append(result)
            cell_results.append(result)
        complete = all(result["status"] == "PASS" for result in cell_results)
        exact_cover = bool(np.all(coverage == 1)) if complete else False
        condition_id = "-".join(
            (spec["modality"], "P", spec["representation"], spec["split_name"])
        )
        if complete and exact_cover:
            aggregate_metrics = _metrics(labels, oof_predictions)
            per_family = {}
            for family in DEVELOPMENT_FAMILIES:
                selected = families == family
                per_family[family] = _metrics(labels[selected], oof_predictions[selected])
            worst_family = min(
                per_family, key=lambda family: per_family[family]["fixed_schema_macro_f1"]
            )
            leakage_totals = {
                key: int(sum(result["leakage"][key] for result in cell_results))
                for key in (
                    "family_overlap_count", "well_overlap_count",
                    "source_interval_overlap_count", "exact_shifted_window_pair_count",
                )
            }
            condition = {
                "condition_id": condition_id,
                "status": "PASS",
                "task_lane": "P",
                "modality_lane": spec["modality"],
                "representation": spec["representation"],
                "split_name": spec["split_name"],
                "split_role": spec["split_role"],
                "rank_eligible": False,
                "rank_status": (
                    "not_rankable" if spec["split_role"] == "diagnostic_only"
                    else "not_ranked_R1_mechanism_only"
                ),
                "oof_coverage": {
                    "expected": len(labels),
                    "covered_once": int((coverage == 1).sum()),
                    "missing": int((coverage == 0).sum()),
                    "duplicates": int((coverage > 1).sum()),
                },
                "metrics": aggregate_metrics,
                "per_family_fixed_schema_macro_f1": {
                    family: metrics["fixed_schema_macro_f1"]
                    for family, metrics in per_family.items()
                },
                "worst_family": worst_family,
                "worst_family_fixed_schema_macro_f1": per_family[worst_family][
                    "fixed_schema_macro_f1"
                ],
                "leakage_totals": leakage_totals,
            }
        else:
            condition = {
                "condition_id": condition_id,
                "status": "not_rankable",
                "rank_eligible": False,
                "reason": "one or more preregistered folds failed or OOF coverage is incomplete",
                "completed_folds": sum(result["status"] == "PASS" for result in cell_results),
                "expected_folds": EFFECTIVE_N_SPLITS,
            }
        conditions.append(condition)
    if len(results) != EXPECTED_CELLS or len(conditions) != EXPECTED_CONDITIONS:
        raise RuntimeError("R1 preregistered condition/cell roster changed")
    completed_cells = sum(result["status"] == "PASS" for result in results)
    condition_lookup = {condition["condition_id"]: condition for condition in conditions}
    paired_deltas = []
    for representation, random_name in (
        ("center", "random_depth_kfold4"),
        ("window", "random_window_kfold4"),
    ):
        for modality in ("W", "M"):
            random_condition = condition_lookup[f"{modality}-P-{representation}-{random_name}"]
            logo_condition = condition_lookup[f"{modality}-P-{representation}-logo4"]
            if random_condition["status"] == logo_condition["status"] == "PASS":
                paired_deltas.append(
                    {
                        "modality_lane": modality,
                        "representation": representation,
                        "random_split": random_name,
                        "legal_split": "logo4",
                        "random_minus_logo_fixed_schema_macro_f1": (
                            random_condition["metrics"]["fixed_schema_macro_f1"]
                            - logo_condition["metrics"]["fixed_schema_macro_f1"]
                        ),
                        "interpretation": "protocol_mechanism_diagnostic_only_not_a_model_rank",
                    }
                )
    summary: dict[str, Any] = {
        "schema_version": R1_SCHEMA,
        "status": "PASS" if completed_cells == EXPECTED_CELLS else "not_rankable",
        "task_id": "gm09_genetic_facies_9class",
        "root_seed": ROOT_SEED,
        "classifier": CLASSIFIER_CONFIG,
        "classifier_config_hash": _stable_hash(CLASSIFIER_CONFIG),
        "primary_metric": "fixed_schema_macro_f1",
        "supported_class_metric_role": "diagnostic_only",
        "formal_model_ranking": False,
        "hpo": False,
        "expected_conditions": EXPECTED_CONDITIONS,
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": completed_cells,
        "completion_rate": completed_cells / EXPECTED_CELLS,
        "conditions": conditions,
        "leakage_matrix": [
            {
                "condition_id": condition["condition_id"],
                "modality_lane": condition.get("modality_lane"),
                "representation": condition.get("representation"),
                "split_name": condition.get("split_name"),
                "split_role": condition.get("split_role"),
                **condition.get("leakage_totals", {}),
            }
            for condition in conditions
        ],
        "paired_protocol_deltas": paired_deltas,
        "s_lane": {
            "status": "not_rankable",
            "finite_center_md_count": int(np.isfinite(arrays["center_md_m"]).sum()),
            "reason": "no real finite center_md_m; sequence fabrication is forbidden",
        },
        "sealed_holdout": {
            "identity": TEST_FAMILY,
            "fresh_blind": False,
            "physical_test_accessed": False,
            "known_holdout_artifacts_read": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "r1_results.jsonl"
    summary_path = output_dir / "r1_summary.json"
    _atomic_jsonl(results_path, results)
    _atomic_json(summary_path, summary)
    r0_path = output_dir / "r0_contract.json"
    if not r0_path.is_file():
        raise FileNotFoundError("R1 output directory lacks the prepared r0_contract.json")
    artifact_manifest = {
        "schema_version": "lithofacies-p5.1-r01-artifacts-v1",
        "status": summary["status"],
        "development_data": dict(batch_manifest["development_data"]),
        "development_batch": {"basename": batch_file.name, "sha256": _hash_file(batch_file)},
        "config_hash": summary["classifier_config_hash"],
        "r0_contract": {"path": "r0_contract.json", "sha256": _hash_file(r0_path)},
        "r1_results": {"path": "r1_results.jsonl", "sha256": _hash_file(results_path)},
        "r1_summary": {"path": "r1_summary.json", "sha256": _hash_file(summary_path)},
        "split_hashes": sorted(
            {result["split_hash"] for result in results if result["status"] == "PASS"}
        ),
        "physical_test_accessed": False,
        "known_holdout_artifacts_read": False,
        "absolute_paths_serialized": False,
        "checkpoints_written": False,
    }
    _atomic_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return summary, int(summary["status"] != "PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="validate and envelope development train data")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--batch-file", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = subparsers.add_parser("run", help="run the fixed-SGD R1 protocol comparison")
    run_parser.add_argument("--batch-file", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(args.dataset_root, args.batch_file, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    summary, exit_code = run_r1(args.batch_file, args.output_dir)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "completed_cells": summary["completed_cells"],
                "expected_cells": summary["expected_cells"],
                "paired_protocol_deltas": summary["paired_protocol_deltas"],
                "s_lane": summary["s_lane"],
                "physical_test_accessed": summary["sealed_holdout"]["physical_test_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
