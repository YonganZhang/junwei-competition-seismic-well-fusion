#!/usr/bin/env python3
"""P5.1 R0 contract freeze and R1 split-mechanism diagnostic.

The runner accepts only the explicitly hash-locked development ``train.h5``
and ``guard.npz`` assets.  It has no frozen/known-holdout argument, loader, or
metric surface.  R1 intentionally compares an invalid random depth-point split
with the legal mother-well-family LOGO split using one fixed ridge-SGD model;
the comparison is protocol evidence, never a model leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import reservoir_p5_stage2 as stage2  # noqa: E402
from _code.ml_framework.contracts import TaskSpec  # noqa: E402
from _models.property.reservoir_ridge import build_model as build_ridge  # noqa: E402
from p5_contract import (  # noqa: E402
    FORBIDDEN_INPUTS,
    INPUT_WHITELIST,
    TARGETS,
    build_task_spec,
)


ROOT_SEED = 2693
FROZEN_TEST_FAMILY = "15/9-F-15"
DEVELOPMENT_FAMILIES = ("15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12")
LOGO_VALIDATION_ORDER = ("15/9-F-12", "15/9-19", "15/9-F-11", "15/9-F-1")
EXPECTED_SOURCE_HASHES = {
    "development_train_h5": "b1962a89b049dd2c23ff2fbf857b5daf69de8b40c2f1f5166205f9bc3df70ab2",
    "development_guard_npz": "67b3866d1975f0ddb32b7016d3e7ba6fa595a5cbbd03cb6b0231a74851ca77fe",
}
TARGET_CONFIG = {
    "PHIF": {
        "stored_index": 0,
        "unit": "fraction",
        "model_domain": "identity",
        "physical_transform": "identity",
        "label_field": "PHIF",
    },
    "KLOGH": {
        "stored_index": 1,
        "unit": "mD",
        "model_domain": "log1p(KLOGH_mD)",
        "physical_transform": "expm1",
        "label_field": "KLOGH",
    },
    "SW": {
        "stored_index": 2,
        "unit": "fraction",
        "model_domain": "identity",
        "physical_transform": "identity",
        "label_field": "SW",
    },
}
R1_MODEL_CONFIG = {
    "model_id": "reservoir_ridge",
    "n_features": 153,
    "learning_rate": 0.002,
    "l2_strength": 0.001,
    "update_steps": 64,
    "update_unit": "full_fold_train_batch",
    "hpo": False,
}
LANES = {
    "scratch_flat_fusion": {
        "lane_id": "tabular_cpu",
        "training_paradigm": "scratch",
        "scientific_modality": "real_ST0202_seismic_patch_plus_raw_well_log_sequence",
        "interface_view": "153D_flat_fusion",
        "rank_scope": "scratch_flat_fusion_only",
    },
    "tabiclv2_tabular_pretrained": {
        "lane_id": "tabular_pretrained_cpu",
        "training_paradigm": "pretrained",
        "scientific_modality": "real_ST0202_seismic_patch_plus_raw_well_log_sequence",
        "interface_view": "153D_flat_fusion",
        "rank_scope": "pretrained_only",
    },
    "monai_seismic_3d_gpu": {
        "lane_id": "seismic_3d_gpu",
        "training_paradigm": "scratch",
        "scientific_modality": "real_ST0202_seismic_patch",
        "interface_view": "3D_tensor",
        "rank_scope": "seismic_3d_only",
    },
}
TABICL_EXPECTED = {
    "revision": "46b91961db4f8873dd049ec09990698a435e1e29",
    "license": "BSD-3-Clause",
    "checkpoint_name": "tabicl-regressor-v2-20260212.ckpt",
    "repository": "jingang/TabICL",
    "repository_snapshot": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
    "size_bytes": 114324594,
    "sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
}
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p5_r01"
SOURCE_LOCK_PATH = PROJECT_ROOT / "_models" / "property" / "source_lock.json"


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _source_lock() -> dict[str, Any]:
    return json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))


def _validated_tabicl_entry() -> dict[str, Any]:
    entry = dict(_source_lock()["models"]["tabiclv2_regressor"])
    weights = dict(entry["weights"])
    checks = {
        "revision": entry.get("revision"),
        "license": entry.get("license"),
        "checkpoint_name": weights.get("checkpoint_name"),
        "repository": weights.get("repository"),
        "repository_snapshot": weights.get("repository_snapshot"),
        "size_bytes": weights.get("size_bytes"),
        "sha256": weights.get("sha256"),
    }
    if checks != TABICL_EXPECTED:
        raise ValueError(f"TabICLv2 official source identity changed: {checks}")
    required_policy = {
        "license_status": "approved",
        "official_release_available": True,
        "private": False,
        "gated": False,
        "missing_local_status": "artifact_unavailable",
        "integrity_failure_status": "integrity_mismatch",
        "auto_download": False,
    }
    if any(weights.get(key) != value for key, value in required_policy.items()):
        raise ValueError("TabICLv2 checkpoint availability/license policy changed")
    return entry


def build_r0_contract() -> dict[str, Any]:
    task_spec = build_task_spec()
    if task_spec.targets != TARGETS or tuple(task_spec.target_masks) != TARGETS:
        raise ValueError("property targets/masks changed")
    if tuple(task_spec.input_whitelist) != INPUT_WHITELIST:
        raise ValueError("property input whitelist changed")
    if tuple(task_spec.forbidden_inputs) != FORBIDDEN_INPUTS:
        raise ValueError("property forbidden-input contract changed")
    if tuple(task_spec.metadata["development_groups"]) != DEVELOPMENT_FAMILIES:
        raise ValueError("development mother-well families changed")
    if task_spec.metadata["frozen_test_family"] != FROZEN_TEST_FAMILY:
        raise ValueError("frozen-test family changed")
    tabicl = _validated_tabicl_entry()
    return {
        "schema_version": 1,
        "phase": "P5.1_R0",
        "track_id": "property",
        "status": "contract_frozen",
        "zero_training": True,
        "root_seed": ROOT_SEED,
        "label_version": task_spec.label_version,
        "targets": {
            target: {
                **TARGET_CONFIG[target],
                "mask": task_spec.target_masks[target],
                "target_transform": task_spec.target_transform[target],
                "inverse_transform": task_spec.inverse_transform[target],
                "independent_from_other_target_masks": True,
            }
            for target in TARGETS
        },
        "source_joint_complete_case_filter": True,
        "source_filter_note": (
            "the existing real-data builder retained rows where PHIF, KLOGH and SW were all finite; "
            "R1 still trains and reports each target independently and does not claim recovered extra rows"
        ),
        "input_whitelist": list(INPUT_WHITELIST),
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
        "development": {
            "group_key": "mother_well_family",
            "families": list(DEVELOPMENT_FAMILIES),
            "logo_validation_order": list(LOGO_VALIDATION_ORDER),
        },
        "frozen_test": {
            "family": FROZEN_TEST_FAMILY,
            "access": False,
            "loader_implemented": False,
            "metrics_allowed": False,
            "fresh_blind_claim_allowed": False,
        },
        "lanes": LANES,
        "tabiclv2": {
            "official_code_available": True,
            "official_checkpoint_available": True,
            "local_checkpoint_provisioned": False,
            "local_status": tabicl["weights"]["missing_local_status"],
            "source_lock_entry": tabicl,
        },
        "source_lock_sha256": _hash_file(SOURCE_LOCK_PATH),
        "r1_model_config": R1_MODEL_CONFIG,
        "r1_purpose": "protocol_mechanism_diagnostic_only",
        "r1_final_model_ranking": False,
    }


def _validate_source(path: Path, role: str) -> dict[str, Any]:
    path = Path(path)
    expected_name = "train.h5" if role == "development_train_h5" else "guard.npz"
    if path.name != expected_name or not path.is_file():
        raise FileNotFoundError(f"{role} must be an explicit existing {expected_name}")
    actual_hash = _hash_file(path)
    if actual_hash != EXPECTED_SOURCE_HASHES[role]:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected={EXPECTED_SOURCE_HASHES[role]}, actual={actual_hash}"
        )
    return {
        "role": role,
        "logical_name": expected_name,
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size,
        "path_persisted": False,
    }


def _validate_development_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("development rows are empty")
    families = {str(row["family_id"]) for row in rows}
    if FROZEN_TEST_FAMILY in families:
        raise RuntimeError("frozen-test family reached the R0/R1 development reader")
    if families != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError(
            f"development family identity changed: expected={DEVELOPMENT_FAMILIES}, actual={sorted(families)}"
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("development sample IDs are not unique")
    for row in rows:
        if np.asarray(row["seismic"]).shape != (3, 3, 9):
            raise ValueError("development seismic patch shape changed")
        if np.asarray(row["logs"]).shape != (9, 8):
            raise ValueError("development well-log sequence shape changed")
        if np.asarray(row["label"]).reshape(-1).size < len(TARGETS):
            raise ValueError("development label cannot provide PHIF/KLOGH/SW")


def _make_target_task_spec(target: str) -> TaskSpec:
    config = TARGET_CONFIG[target]
    return TaskSpec(
        track_id="property",
        task_id=f"reservoir_property_p51_r1_{target.lower()}",
        task_type="regression",
        input_modalities=("tabular_flat_view",),
        targets=(target,),
        units={target: config["unit"]},
        label_version=f"property-{target.lower()}-cpi-v1",
        target_masks={target: f"isfinite({target})"},
        group_keys=("mother_well_family",),
        target_transform={target: config["model_domain"]},
        inverse_transform={target: config["physical_transform"]},
        train_loss={target: "MSE_in_model_domain"},
        inference_transform={target: "preserve_unbounded_raw_prediction"},
        threshold_policy={},
        calibration_policy={target: "none"},
        primary_metrics=("physical_unclipped_RMSE",),
        secondary_metrics=("physical_MAE", "physical_R2", "physical_Pearson"),
        guardrail_metrics=("finite_prediction", "raw_physical_boundary_violation_rate"),
        metric_directions={
            "physical_unclipped_RMSE": "minimize",
            "physical_MAE": "minimize",
            "physical_R2": "maximize",
            "physical_Pearson": "maximize",
            "finite_prediction": "maximize",
            "raw_physical_boundary_violation_rate": "minimize",
        },
        hpo={"enabled": False, "test_access": "forbidden"},
        visualizer_id="reservoir_property_r1_protocol_diagnostic",
        required_figures=("r1_random_vs_logo_oof_diagnostic",),
        input_whitelist=INPUT_WHITELIST,
        forbidden_inputs=FORBIDDEN_INPUTS,
        metadata={
            "root_seed": ROOT_SEED,
            "source_joint_complete_case_filter": True,
            "protocol_diagnostic_only": True,
        },
    )


def _random_kfold(sample_count: int) -> list[dict[str, Any]]:
    if sample_count < 4:
        raise ValueError("RandomKFold4 requires at least four samples")
    permutation = np.random.default_rng(ROOT_SEED).permutation(sample_count)
    folds: list[dict[str, Any]] = []
    all_indices = np.arange(sample_count, dtype=int)
    for fold_id, validation in enumerate(np.array_split(permutation, 4)):
        validation = np.sort(validation.astype(int))
        train = np.setdiff1d(all_indices, validation, assume_unique=True)
        folds.append({"fold_id": fold_id, "train": train, "validation": validation})
    return folds


def _logo4(families: np.ndarray) -> list[dict[str, Any]]:
    family_values = set(families.astype(str).tolist())
    if family_values != set(DEVELOPMENT_FAMILIES):
        raise ValueError("LOGO4 requires all four frozen development families")
    folds: list[dict[str, Any]] = []
    for fold_id, validation_family in enumerate(LOGO_VALIDATION_ORDER):
        validation = np.flatnonzero(families == validation_family)
        train = np.flatnonzero(families != validation_family)
        if not len(train) or not len(validation):
            raise RuntimeError(f"LOGO4 fold {fold_id} is empty")
        folds.append({"fold_id": fold_id, "train": train, "validation": validation})
    return folds


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("regression metrics require nonempty paired arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise FloatingPointError("regression metrics received non-finite values")
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = None if total <= 0.0 else float(1.0 - np.sum(residual**2) / total)
    if actual.size < 2 or float(np.std(actual)) == 0.0 or float(np.std(predicted)) == 0.0:
        pearson = None
        pearson_reason = "undefined_for_fewer_than_two_points_or_constant_vector"
    else:
        pearson = float(np.corrcoef(actual, predicted)[0, 1])
        pearson_reason = None
    values = [mae, rmse] + ([] if r2 is None else [r2]) + ([] if pearson is None else [pearson])
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("regression metrics produced non-finite values")
    return {
        "count": int(actual.size),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "R2_reason": None if r2 is not None else "undefined_for_constant_truth",
        "Pearson": pearson,
        "Pearson_reason": pearson_reason,
    }


def _physical_values(target: str, model_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model_values = np.asarray(model_values, dtype=np.float64)
    if target == "KLOGH":
        with np.errstate(over="raise", invalid="raise"):
            raw = np.expm1(model_values)
        bounded = np.maximum(raw, 0.0)
    elif target in {"PHIF", "SW"}:
        raw = model_values.copy()
        bounded = np.clip(raw, 0.0, 1.0)
    else:
        raise KeyError(target)
    if not np.isfinite(raw).all() or not np.isfinite(bounded).all():
        raise FloatingPointError(f"{target} physical transform produced non-finite values")
    return raw, bounded


def _boundary_diagnostic(target: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if target in {"PHIF", "SW"}:
        invalid = (values < 0.0) | (values > 1.0)
        rule = "prediction<0_or_prediction>1"
    else:
        invalid = values < 0.0
        rule = "prediction_mD<0"
    return {
        "rule": rule,
        "invalid_count": int(invalid.sum()),
        "invalid_fraction": float(invalid.mean()),
        "raw_min": float(values.min()),
        "raw_max": float(values.max()),
    }


def _metrics_by_target(
    target: str,
    truth_model: np.ndarray,
    prediction_model: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    truth_raw, _ = _physical_values(target, truth_model)
    prediction_raw, prediction_bounded = _physical_values(target, prediction_model)
    per_family = {
        family: _regression_metrics(
            truth_raw[families == family], prediction_raw[families == family]
        )
        for family in sorted(set(families.tolist()))
    }
    worst = max(per_family, key=lambda family: (per_family[family]["RMSE"], family))
    payload = {
        "physical_unit": TARGET_CONFIG[target]["unit"],
        "physical_unclipped": _regression_metrics(truth_raw, prediction_raw),
        "physical_bounded_diagnostic": _regression_metrics(truth_raw, prediction_bounded),
        "model_domain": _regression_metrics(truth_model, prediction_model),
        "per_mother_family_physical_unclipped": per_family,
        "worst_mother_family": {"family_id": worst, **per_family[worst]},
        "raw_prediction_boundary": _boundary_diagnostic(target, prediction_raw),
    }
    if target == "KLOGH":
        payload["log1p_diagnostic"] = payload["model_domain"]
    return payload


def _stats_payload(stats: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {key: np.asarray(value).tolist() for key, value in stats.items()}


def _run_fold(
    *,
    target: str,
    seismic: np.ndarray,
    logs: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    sample_ids: Sequence[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    fold_id: int,
    protocol: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_groups = sorted(set(families[train_indices].tolist()))
    validation_groups = sorted(set(families[validation_indices].tolist()))
    overlap = sorted(set(train_groups) & set(validation_groups))
    if protocol == "mother_family_logo4" and overlap:
        raise RuntimeError(f"LOGO4 fold {fold_id} has mother-family overlap")
    if protocol == "random_depth_kfold4" and not overlap:
        raise RuntimeError(f"random diagnostic fold {fold_id} unexpectedly has no family overlap")

    stats = stage2._fit_stats(seismic[train_indices], logs[train_indices])
    _, _, features = stage2._transform(seismic, logs, stats)
    task_spec = _make_target_task_spec(target)
    model = build_ridge(
        task_spec,
        n_features=R1_MODEL_CONFIG["n_features"],
        learning_rate=R1_MODEL_CONFIG["learning_rate"],
        l2_strength=R1_MODEL_CONFIG["l2_strength"],
        seed=ROOT_SEED,
    )
    train_x = features[train_indices]
    train_y = labels[train_indices, None]
    first_loss = None
    final_loss = None
    started = time.perf_counter()
    for _ in range(R1_MODEL_CONFIG["update_steps"]):
        final_loss = float(model.train_batch((train_x, train_y)))
        if first_loss is None:
            first_loss = final_loss
    prediction = model.predict_array(features[validation_indices])[:, 0]
    wall_seconds = time.perf_counter() - started
    if not np.isfinite(prediction).all() or not math.isfinite(float(final_loss)):
        raise FloatingPointError("R1 fold produced a non-finite prediction or loss")
    stats_serialized = _stats_payload(stats)
    evidence = {
        "target": target,
        "protocol": protocol,
        "fold_id": fold_id,
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "mother_family_overlap": overlap,
        "train_sample_ids_sha256": _hash_payload(sorted(sample_ids[index] for index in train_indices)),
        "validation_sample_ids_sha256": _hash_payload(
            sorted(sample_ids[index] for index in validation_indices)
        ),
        "preprocessing": {
            "fit": "fold_train_only",
            "fit_sample_ids_sha256": _hash_payload(
                sorted(sample_ids[index] for index in train_indices)
            ),
            "fit_validation_overlap": False,
            "stats_sha256": _hash_payload(stats_serialized),
            "target_statistics_fitted": False,
            "target_transform": TARGET_CONFIG[target]["model_domain"],
            "target_transform_fitted": False,
            "class_weights": "not_applicable_regression",
            "threshold": "none",
            "calibration": "none",
        },
        "model": {
            **R1_MODEL_CONFIG,
            "seed": ROOT_SEED,
            "first_train_loss": first_loss,
            "final_train_loss": final_loss,
        },
        "wall_seconds": wall_seconds,
        "test_firewall": {
            "test_access": False,
            "frozen_test_family_seen": False,
            "test_metrics": False,
        },
    }
    return prediction, evidence


def _oof_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "sample_id",
        "family_id",
        "well_id",
        "depth_m",
        "fold_id",
        "truth_model_domain",
        "prediction_model_domain",
        "truth_physical",
        "prediction_physical_unclipped",
        "prediction_physical_bounded",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _protocol_manifest(
    folds: Sequence[Mapping[str, Any]], families: np.ndarray, sample_ids: Sequence[str]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for fold in folds:
        train = np.asarray(fold["train"], dtype=int)
        validation = np.asarray(fold["validation"], dtype=int)
        train_groups = sorted(set(families[train].tolist()))
        validation_groups = sorted(set(families[validation].tolist()))
        values.append(
            {
                "fold_id": int(fold["fold_id"]),
                "train_count": int(len(train)),
                "validation_count": int(len(validation)),
                "train_groups": train_groups,
                "validation_groups": validation_groups,
                "mother_family_overlap": sorted(set(train_groups) & set(validation_groups)),
                "train_sample_ids_sha256": _hash_payload(
                    sorted(sample_ids[index] for index in train)
                ),
                "validation_sample_ids_sha256": _hash_payload(
                    sorted(sample_ids[index] for index in validation)
                ),
            }
        )
    return values


def _run_target(
    target: str,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    result_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    target_index = TARGET_CONFIG[target]["stored_index"]
    eligible = [row for row in rows if np.isfinite(np.asarray(row["label"])[target_index])]
    support = {
        family: sum(str(row["family_id"]) == family for row in eligible)
        for family in DEVELOPMENT_FAMILIES
    }
    if any(count == 0 for count in support.values()) or len(eligible) < 4:
        return (
            {
                "target": target,
                "status": "blocked",
                "reason": "insufficient_label_support_for_four_development_mother_families",
                "support_by_family": support,
                "final_model_ranking": False,
            },
            {
                "target": target,
                "status": "blocked",
                "support_by_family": support,
            },
            [],
        )
    sample_ids = [str(row["sample_id"]) for row in eligible]
    families = np.asarray([str(row["family_id"]) for row in eligible])
    wells = [str(row["well_id"]) for row in eligible]
    depths = np.asarray([float(row["depth_m"]) for row in eligible], dtype=np.float64)
    seismic = np.stack([np.asarray(row["seismic"], dtype=np.float64) for row in eligible])
    logs = np.stack([np.asarray(row["logs"], dtype=np.float64) for row in eligible])
    labels = np.asarray(
        [float(np.asarray(row["label"], dtype=np.float64)[target_index]) for row in eligible]
    )
    if target == "KLOGH" and np.any(labels < 0.0):
        raise ValueError("development KLOGH log1p labels must be nonnegative")

    protocols = {
        "random_depth_kfold4": _random_kfold(len(eligible)),
        "mother_family_logo4": _logo4(families),
    }
    protocol_summaries: dict[str, Any] = {}
    protocol_manifests: dict[str, Any] = {}
    written: list[Path] = []
    for protocol, folds in protocols.items():
        oof = np.full(len(eligible), np.nan, dtype=np.float64)
        assignments = np.zeros(len(eligible), dtype=np.int64)
        fold_ids = np.full(len(eligible), -1, dtype=np.int64)
        fold_evidence: list[dict[str, Any]] = []
        for fold in folds:
            validation = np.asarray(fold["validation"], dtype=int)
            prediction, evidence = _run_fold(
                target=target,
                seismic=seismic,
                logs=logs,
                labels=labels,
                families=families,
                sample_ids=sample_ids,
                train_indices=np.asarray(fold["train"], dtype=int),
                validation_indices=validation,
                fold_id=int(fold["fold_id"]),
                protocol=protocol,
            )
            oof[validation] = prediction
            assignments[validation] += 1
            fold_ids[validation] = int(fold["fold_id"])
            truth_raw, _ = _physical_values(target, labels[validation])
            predicted_raw, _ = _physical_values(target, prediction)
            evidence["validation_metrics_physical_unclipped"] = _regression_metrics(
                truth_raw, predicted_raw
            )
            fold_evidence.append(evidence)
            result_rows.append(evidence)
        if not np.array_equal(assignments, np.ones(len(eligible), dtype=np.int64)):
            raise RuntimeError(f"{target}/{protocol} did not produce exactly one OOF prediction per row")
        if not np.isfinite(oof).all() or np.any(fold_ids < 0):
            raise RuntimeError(f"{target}/{protocol} OOF is incomplete")
        metrics = _metrics_by_target(target, labels, oof, families)
        truth_raw, _ = _physical_values(target, labels)
        prediction_raw, prediction_bounded = _physical_values(target, oof)
        csv_rows = [
            {
                "sample_id": sample_ids[index],
                "family_id": families[index],
                "well_id": wells[index],
                "depth_m": f"{depths[index]:.8g}",
                "fold_id": int(fold_ids[index]),
                "truth_model_domain": f"{labels[index]:.12g}",
                "prediction_model_domain": f"{oof[index]:.12g}",
                "truth_physical": f"{truth_raw[index]:.12g}",
                "prediction_physical_unclipped": f"{prediction_raw[index]:.12g}",
                "prediction_physical_bounded": f"{prediction_bounded[index]:.12g}",
            }
            for index in range(len(eligible))
        ]
        oof_path = output_dir / target.lower() / f"{protocol}_oof.csv"
        _atomic_text(oof_path, _oof_csv(csv_rows))
        written.append(oof_path)
        protocol_summaries[protocol] = {
            "status": "protocol_diagnostic_complete",
            "split_legality": "invalid_diagnostic" if protocol.startswith("random") else "legal",
            "eligible_count": len(eligible),
            "support_by_family": support,
            "metrics": metrics,
            "folds": fold_evidence,
            "oof_artifact": oof_path.relative_to(output_dir).as_posix(),
            "oof_sha256": _hash_file(oof_path),
            "test_access": False,
        }
        protocol_manifests[protocol] = {
            "kind": "RandomKFold4_shuffle_seed2693"
            if protocol.startswith("random")
            else "mother_well_family_LOGO4",
            "legal_for_model_assessment": not protocol.startswith("random"),
            "folds": _protocol_manifest(folds, families, sample_ids),
        }
    random_rmse = protocol_summaries["random_depth_kfold4"]["metrics"][
        "physical_unclipped"
    ]["RMSE"]
    logo_rmse = protocol_summaries["mother_family_logo4"]["metrics"][
        "physical_unclipped"
    ]["RMSE"]
    summary = {
        "target": target,
        "status": "protocol_diagnostic_complete",
        "unit": TARGET_CONFIG[target]["unit"],
        "model_domain": TARGET_CONFIG[target]["model_domain"],
        "independent_target_mask": True,
        "eligible_count": len(eligible),
        "support_by_family": support,
        "sample_ids_sha256": _hash_payload(sorted(sample_ids)),
        "protocols": protocol_summaries,
        "diagnostic_gap_logo_minus_random_physical_RMSE": float(logo_rmse - random_rmse),
        "interpretation": (
            "random depth-point performance is an intentionally leakage-prone diagnostic; "
            "only LOGO reflects legal development generalization"
        ),
        "final_model_ranking": False,
        "test_access": False,
    }
    manifest = {
        "target": target,
        "status": "frozen",
        "group_key": "mother_well_family",
        "eligible_count": len(eligible),
        "sample_ids_sha256": _hash_payload(sorted(sample_ids)),
        "support_by_family": support,
        "protocols": protocol_manifests,
    }
    return summary, manifest, written


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def run_r01(train_h5: Path, guard_npz: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    r0 = build_r0_contract()
    r0_path = output_dir / "r0_contract.json"
    _atomic_json(r0_path, r0)
    source_assets = [
        _validate_source(Path(train_h5), "development_train_h5"),
        _validate_source(Path(guard_npz), "development_guard_npz"),
    ]
    rows = stage2._read_records(
        Path(train_h5), Path(guard_npz), "reservoir_property_p51_r1"
    )
    _validate_development_rows(rows)

    result_rows: list[dict[str, Any]] = []
    target_summaries: dict[str, Any] = {}
    target_manifests: dict[str, Any] = {}
    artifact_paths: list[Path] = [r0_path]
    for target in TARGETS:
        summary, manifest, written = _run_target(target, rows, output_dir, result_rows)
        target_summaries[target] = summary
        target_manifests[target] = manifest
        artifact_paths.extend(written)

    split_manifest = {
        "schema_version": 1,
        "phase": "P5.1_R1",
        "track_id": "property",
        "root_seed": ROOT_SEED,
        "selection_registered_before_modeling": True,
        "source_assets": source_assets,
        "source_paths_persisted": False,
        "development_families": list(DEVELOPMENT_FAMILIES),
        "frozen_test_family": FROZEN_TEST_FAMILY,
        "source_joint_complete_case_filter": True,
        "targets": target_manifests,
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family_seen": False,
        },
    }
    split_manifest["split_manifest_sha256"] = _hash_payload(split_manifest)
    split_path = output_dir / "r1_split_manifest.json"
    _atomic_json(split_path, split_manifest)
    artifact_paths.append(split_path)

    results_path = output_dir / "r1_results.jsonl"
    _write_jsonl(results_path, result_rows)
    artifact_paths.append(results_path)
    complete = all(
        summary["status"] == "protocol_diagnostic_complete"
        for summary in target_summaries.values()
    )
    summary = {
        "schema_version": 1,
        "phase": "P5.1_R0_R1",
        "track_id": "property",
        "r0_status": r0["status"],
        "r1_status": "protocol_diagnostic_complete" if complete else "blocked",
        "root_seed": ROOT_SEED,
        "protocol_diagnostic_only": True,
        "final_model_ranking": False,
        "hpo": False,
        "model_config": R1_MODEL_CONFIG,
        "targets": target_summaries,
        "split_manifest_sha256": _hash_file(split_path),
        "results_sha256": _hash_file(results_path),
        "run_wall_seconds": time.perf_counter() - started,
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family_seen": False,
            "fresh_blind_claim": False,
        },
    }
    summary_path = output_dir / "r1_summary.json"
    _atomic_json(summary_path, summary)
    artifact_paths.append(summary_path)

    run_manifest = {
        "schema_version": 1,
        "phase": "P5.1_R0_R1",
        "track_id": "property",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "command_template": (
            "python3 _pipelines/02_task_datasets/reservoir/reservoir_p5_r01.py run "
            "--train-h5 <development_train.h5> --guard-npz <development_guard.npz>"
        ),
        "source_assets": source_assets,
        "config_sha256": _hash_payload(R1_MODEL_CONFIG),
        "source_lock_sha256": _hash_file(SOURCE_LOCK_PATH),
        "split_manifest_sha256": _hash_file(split_path),
        "source_paths_persisted": False,
        "test_access": False,
    }
    run_path = output_dir / "run_manifest.json"
    _atomic_json(run_path, run_manifest)
    artifact_paths.append(run_path)

    artifact_manifest = {
        "schema_version": 1,
        "track_id": "property",
        "phase": "P5.1_R0_R1",
        "artifacts": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _hash_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(set(artifact_paths))
        ],
        "absolute_paths_persisted": False,
        "large_runtime_artifacts_persisted": False,
        "test_access": False,
    }
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    _atomic_json(artifact_manifest_path, artifact_manifest)
    return {
        **summary,
        "artifact_manifest_sha256": _hash_file(artifact_manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="freeze and print the zero-training R0 contract")
    audit.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run = subparsers.add_parser("run", help="run bounded development-only R1 diagnostics")
    run.add_argument("--train-h5", type=Path, required=True)
    run.add_argument("--guard-npz", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        contract = build_r0_contract()
        path = args.output_dir / "r0_contract.json"
        _atomic_json(path, contract)
        print(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    summary = run_r01(args.train_h5, args.guard_npz, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["r1_status"] == "protocol_diagnostic_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
