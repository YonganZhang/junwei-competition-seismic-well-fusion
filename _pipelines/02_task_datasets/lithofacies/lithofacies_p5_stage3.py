#!/usr/bin/env python3
"""P5 Stage-3 multiseed LOGO4 confirmation for the GM09 P lane.

The runner is development-only.  It projects the frozen P4 mother-family
contract onto the existing development archive, reuses the Stage-2 model and
budget implementation unchanged, and records all 3 models x 4 folds x 3
repeat seeds.  No command accepts a frozen-test loader, path, label, or metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.lithofacies.p5_adapter_common import OptionalDependencyUnavailable  # noqa: E402
from lithofacies_p5_stage2 import (  # noqa: E402
    ESTIMATOR_WALL_LIMIT_SECONDS,
    NEURAL_PARAMETER_UPDATE_LIMIT,
    NEURAL_WALL_LIMIT_SECONDS,
    P_BATCH_SIZE,
    P_CONTEXT_LENGTH,
    P_TRAIN_SAMPLE_LIMIT,
    P_VALIDATION_SAMPLE_LIMIT,
    TINY_GATE_UPDATES,
    _atomic_write_json,
    _atomic_write_text,
    _estimator_pilot,
    _metric_payload,
    _portable_environment,
    _stable_hash,
    _stage2_model_config,
    _torch_pilot,
    _track_owned,
    _verify_external_gpu_lock,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    TEST_FAMILY,
    apply_fold_preprocessor,
    class_support,
    classification_metrics_from_logits,
    fit_fold_preprocessor,
    sample_id,
)
from p5_stage1 import (  # noqa: E402
    _balanced_take,
    _p_arrays,
    _read_development_hdf5,
    _sha256,
    build_development_logo4,
    load_source_lock,
)


ROOT_SEED = 2693
TASK_ID = "gm09_genetic_facies_9class"
LANE = "P"
TOP3 = (
    "xgboost_multisoftprob_window",
    "catboost_multiclass_window",
    "inceptiontime_window",
)
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
FOLD_IDS = tuple(range(EFFECTIVE_N_SPLITS))
EXPECTED_CELL_COUNT = len(TOP3) * len(FOLD_IDS) * len(REPEAT_SEEDS)
MIN_LEGAL_COMPLETION = 0.80
BOOTSTRAP_SAMPLES = 10_000

BATCH_SCHEMA = "lithofacies-p5-stage3-batch-v1"
CELL_SCHEMA = "lithofacies-p5-stage3-cell-v1"
PARTIAL_SCHEMA = "lithofacies-p5-stage3-partial-v1"
SUMMARY_SCHEMA = "lithofacies-p5-stage3-summary-v1"
LEADERBOARD_SCHEMA = "lithofacies-p5-stage3-gm09-p-leaderboard-v1"
OOF_SCHEMA = "lithofacies-p5-stage3-oof-manifest-v1"
VISUALIZATION_SCHEMA = "lithofacies-p5-stage3-visualization-manifest-v1"

CANONICAL_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_stage3"
RESULTS_FILENAME = "p5_stage3_results.jsonl"
SUMMARY_FILENAME = "p5_stage3_summary.json"
LEADERBOARD_FILENAME = "p5_stage3_gm09_p_leaderboard.json"
OOF_MANIFEST_FILENAME = "p5_stage3_oof_manifest.json"
VISUALIZATION_MANIFEST_FILENAME = "p5_stage3_visualization_manifest.json"
P4_DATA_MANIFEST = TRACK_DIR / "_outputs" / "split_manifest.json"
STAGE2_SUMMARY = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_summary.json"
STAGE2_LEADERBOARD = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_p_leaderboard.json"


def _reason(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value % (2**31 - 1) or ROOT_SEED


def component_seeds(model_id: str, fold_id: int, repeat_id: int) -> dict[str, int]:
    model_seed = REPEAT_SEEDS[repeat_id]
    return {
        "model": model_seed,
        "loader": _stable_seed(model_seed, model_id, fold_id, "loader"),
        "sampler": _stable_seed(model_seed, model_id, fold_id, "sampler"),
        "diagnostic": _stable_seed(model_seed, model_id, fold_id, "diagnostic"),
    }


def stage2_budget_contract() -> dict[str, Any]:
    """Return the immutable Stage-2 budget reused by all Stage-3 cells."""
    payload = {
        "well_shape": [26, P_CONTEXT_LENGTH],
        "seismic_shape": [3, 3, P_CONTEXT_LENGTH],
        "fold_train_sample_limit": P_TRAIN_SAMPLE_LIMIT,
        "fold_validation_sample_limit": P_VALIDATION_SAMPLE_LIMIT,
        "batch_size": P_BATCH_SIZE,
        "neural_parameter_updates": NEURAL_PARAMETER_UPDATE_LIMIT,
        "tiny_gate_updates_included": TINY_GATE_UPDATES,
        "neural_wall_limit_seconds": NEURAL_WALL_LIMIT_SECONDS,
        "estimator_wall_limit_seconds": ESTIMATOR_WALL_LIMIT_SECONDS,
        "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
        "xgboost": {"rounds": 40, "max_depth": 2},
        "catboost": {"iterations": 40, "depth": 3},
        "inceptiontime": {"nf": 8, "kernel_size": 31},
        "pretrained_weights": False,
        "hpo": False,
    }
    return {**payload, "budget_hash": _stable_hash(payload)}


def locked_loss_contract(model_id: str) -> str:
    values = {
        "xgboost_multisoftprob_window": "sqrt_inverse_frequency_weighted_multi_softprob",
        "catboost_multiclass_window": "sqrt_inverse_frequency_weighted_multiclass",
        "inceptiontime_window": "cross_entropy_sqrt_inverse_frequency",
    }
    return values[model_id]


def _load_frozen_sources() -> dict[str, Any]:
    source_lock = load_source_lock()
    by_id = {model["model_id"]: model for model in source_lock["models"]}
    if any(model_id not in by_id for model_id in TOP3):
        raise RuntimeError("Stage-3 top-3 is absent from the source lock")
    if any(by_id[model_id]["leaderboard_lane"] != LANE for model_id in TOP3):
        raise RuntimeError("Stage-3 top-3 contains a cross-lane candidate")
    stage2_summary = json.loads(STAGE2_SUMMARY.read_text(encoding="utf-8"))
    stage2_board = json.loads(STAGE2_LEADERBOARD.read_text(encoding="utf-8"))
    observed_top3 = tuple(entry["model_id"] for entry in stage2_board["entries"][:3])
    if observed_top3 != TOP3 or stage2_board.get("primary_metric") != "fixed_schema_macro_f1":
        raise RuntimeError("Stage-2 accepted top-3 or fixed-nine primary metric changed")
    if stage2_summary.get("frozen_test_accessed") is not False:
        raise RuntimeError("Stage-2 source evidence violates the test firewall")
    return {
        "source_lock": source_lock,
        "by_id": by_id,
        "source_lock_sha256": _sha256(TRACK_DIR / "p5_source_lock.json"),
        "stage2_summary_sha256": _sha256(STAGE2_SUMMARY),
        "stage2_leaderboard_sha256": _sha256(STAGE2_LEADERBOARD),
        "stage2_fold0_split_hash": stage2_summary["split_hash"],
    }


def _validate_p4_data_manifest() -> dict[str, Any]:
    """Read only the split/data contract from the tracked P4 data manifest."""
    payload = json.loads(P4_DATA_MANIFEST.read_text(encoding="utf-8"))
    split = payload.get("split_contract", {})
    assignments = split.get("frozen_family_partitions", {})
    if assignments.get(TEST_FAMILY) != "test":
        raise RuntimeError("P4 data manifest no longer freezes the F-5 family")
    usable = split.get("usable_families", {})
    development = tuple(usable.get("train", ())) + tuple(usable.get("guard", ()))
    if set(development) != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError("P4 data manifest development families changed")
    if tuple(usable.get("test", ())) != (TEST_FAMILY,):
        raise RuntimeError("P4 data manifest test-family contract changed")
    return {
        "path": str(P4_DATA_MANIFEST.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(P4_DATA_MANIFEST),
        "split_unit": split.get("unit"),
        "assignment_before": split.get("assignment_before"),
        "development_families": list(DEVELOPMENT_FAMILIES),
        "frozen_test_family": TEST_FAMILY,
        "test_details_consumed": False,
    }


def _sample_metadata(samples: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    well = np.asarray([str(sample.get("position", {}).get("well_name", "")) for sample in samples])
    family = np.asarray([str(sample.get("meta", {}).get("family_id", "")) for sample in samples])
    twt = np.asarray(
        [float(sample.get("position", {}).get("time_ms", np.nan)) for sample in samples],
        dtype=np.float64,
    )
    md = np.asarray(
        [
            np.nan
            if sample.get("position", {}).get("center_md_m") is None
            else float(sample["position"]["center_md_m"])
            for sample in samples
        ],
        dtype=np.float64,
    )
    logs = np.stack([np.asarray(sample["well_log_seq"], dtype=np.float32) for sample in samples])
    missing_fraction = 1.0 - logs[:, 13:, :].mean(axis=(1, 2))
    return {
        "well_id": well,
        "family_id": family,
        "twt_ms": twt,
        "center_md_m": md,
        "well_log_missing_fraction": missing_fraction.astype(np.float32),
        "seismic_available": np.ones(len(samples), dtype=np.uint8),
    }


def prepare_stage3_batch(dataset_root: Path, batch_file: Path) -> dict[str, Any]:
    """Prepare all four P4 development folds while opening only the train archive."""
    batch_file = _track_owned(batch_file)
    frozen = _load_frozen_sources()
    p4_data = _validate_p4_data_manifest()
    samples, hdf5_path = _read_development_hdf5(dataset_root)
    folds = build_development_logo4(samples)
    if len(folds) != len(FOLD_IDS):
        raise RuntimeError("P4 development projection is not LOGO4")

    arrays: dict[str, Any] = {}
    fold_manifests: list[dict[str, Any]] = []
    all_validation_ids: list[str] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        if fold_id not in FOLD_IDS:
            raise RuntimeError("unexpected P4 fold id")
        train_raw = list(fold["train"])
        validation_raw = list(fold["validation"])
        preprocessor = fit_fold_preprocessor(train_raw)
        train_all = apply_fold_preprocessor(train_raw, preprocessor)
        validation_all = apply_fold_preprocessor(validation_raw, preprocessor)
        train = _balanced_take(train_all, P_TRAIN_SAMPLE_LIMIT)
        validation = _balanced_take(validation_all, P_VALIDATION_SAMPLE_LIMIT)
        train_values = _p_arrays(train)
        validation_values = _p_arrays(validation)
        prefix = f"f{fold_id}"
        arrays.update(
            {
                f"{prefix}_train_well": train_values[0],
                f"{prefix}_train_seismic": train_values[1],
                f"{prefix}_train_labels": train_values[2],
                f"{prefix}_train_ids": train_values[3],
                f"{prefix}_validation_well": validation_values[0],
                f"{prefix}_validation_seismic": validation_values[1],
                f"{prefix}_validation_labels": validation_values[2],
                f"{prefix}_validation_ids": validation_values[3],
                f"{prefix}_class_counts": np.asarray(preprocessor.class_support, dtype=np.int64),
                f"{prefix}_class_weights": np.asarray(preprocessor.class_weights, dtype=np.float32),
                f"{prefix}_preprocessor": np.asarray(
                    json.dumps(preprocessor.to_dict(), ensure_ascii=False, sort_keys=True)
                ),
            }
        )
        metadata = _sample_metadata(validation)
        for key, values in metadata.items():
            arrays[f"{prefix}_validation_{key}"] = values

        full_train_ids = [sample_id(sample) for sample in train_raw]
        full_validation_ids = [sample_id(sample) for sample in validation_raw]
        selected_train_ids = [str(value) for value in train_values[3].tolist()]
        selected_validation_ids = [str(value) for value in validation_values[3].tolist()]
        if set(selected_validation_ids) != set(full_validation_ids):
            raise RuntimeError("validation cap silently dropped P4 fold samples")
        compatibility_payload = {
            "fold_id": fold_id,
            "train_groups": fold["train_groups"],
            "validation_groups": fold["validation_groups"],
            "train_sample_ids": selected_train_ids,
            "validation_sample_ids": selected_validation_ids,
        }
        compatibility_hash = _stable_hash(compatibility_payload)
        if fold_id == 0 and compatibility_hash != frozen["stage2_fold0_split_hash"]:
            raise RuntimeError("Stage-3 fold 0 no longer matches the accepted Stage-2 split")
        partition_payload = {
            "fold_id": fold_id,
            "train_groups": fold["train_groups"],
            "validation_groups": fold["validation_groups"],
            "train_sample_ids": sorted(full_train_ids),
            "validation_sample_ids": sorted(full_validation_ids),
        }
        preprocessor_payload = preprocessor.to_dict()
        fold_manifests.append(
            {
                "fold_id": fold_id,
                "train_groups": list(fold["train_groups"]),
                "validation_groups": list(fold["validation_groups"]),
                "full_train_samples": len(train_raw),
                "selected_train_samples": len(train),
                "full_validation_samples": len(validation_raw),
                "selected_validation_samples": len(validation),
                "train_class_support": class_support(train_raw).tolist(),
                "validation_class_support": class_support(validation_raw).tolist(),
                "partition_hash": _stable_hash(partition_payload),
                "stage2_compatibility_hash": compatibility_hash,
                "training_selection_hash": _stable_hash(selected_train_ids),
                "preprocessor_hash": _stable_hash(preprocessor_payload),
                "preprocessor_fit_scope": "fold_train_mother_families_only",
                "preprocessor_fit_families": list(preprocessor.fit_families),
                "class_weight_fit_scope": "fold_train_mother_families_only",
                "target_transform": "identity_class_id",
                "calibration": "not_applied_locked_stage2_configuration",
            }
        )
        all_validation_ids.extend(full_validation_ids)

    development_ids = sorted(sample_id(sample) for sample in samples)
    if sorted(all_validation_ids) != development_ids or len(set(all_validation_ids)) != len(samples):
        raise RuntimeError("LOGO4 validation folds do not form a one-time development cover")
    split_payload = {
        "splitter": "P4_leave_one_mother_family_out",
        "requested_n_splits": 5,
        "effective_n_splits": EFFECTIVE_N_SPLITS,
        "development_families": list(DEVELOPMENT_FAMILIES),
        "folds": [
            {
                "fold_id": fold["fold_id"],
                "train_groups": fold["train_groups"],
                "validation_groups": fold["validation_groups"],
                "partition_hash": fold["partition_hash"],
            }
            for fold in fold_manifests
        ],
    }
    manifest = {
        "schema_version": BATCH_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "class_names": CLASS_NAMES,
        "class_count": len(CLASS_NAMES),
        "top3": TOP3,
        "repeat_seeds": REPEAT_SEEDS,
        "expected_cells": EXPECTED_CELL_COUNT,
        "budget": stage2_budget_contract(),
        "p4_data_manifest": p4_data,
        "split_contract": split_payload,
        "split_hash": _stable_hash(split_payload),
        "folds": fold_manifests,
        "development_sample_count": len(samples),
        "development_sample_ids_hash": _stable_hash(development_ids),
        "loaded_files": [hdf5_path.name],
        "development_hdf5_sha256": _sha256(hdf5_path),
        "source_lock_sha256": frozen["source_lock_sha256"],
        "stage2_summary_sha256": frozen["stage2_summary_sha256"],
        "stage2_leaderboard_sha256": frozen["stage2_leaderboard_sha256"],
        "frozen_test_family": TEST_FAMILY,
        "frozen_test_accessed": False,
        "test_metrics_used": False,
    }
    arrays["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(batch_file, **arrays)
    return {
        "schema_version": BATCH_SCHEMA,
        "batch_file": str(batch_file.relative_to(PROJECT_ROOT)),
        "batch_sha256": _sha256(batch_file),
        "split_hash": manifest["split_hash"],
        "development_samples": len(samples),
        "folds": fold_manifests,
        "loaded_files": manifest["loaded_files"],
        "frozen_test_accessed": False,
    }


def load_stage3_batch(batch_file: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(batch_file, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
        manifest = json.loads(str(archive["manifest"].item()))
    if manifest.get("schema_version") != BATCH_SCHEMA:
        raise ValueError("unknown Stage-3 batch schema")
    if tuple(manifest.get("class_names", ())) != CLASS_NAMES or int(manifest.get("class_count", 0)) != 9:
        raise ValueError("Stage-3 batch changed the fixed GM09 nine-class schema")
    if tuple(manifest.get("top3", ())) != TOP3 or tuple(manifest.get("repeat_seeds", ())) != REPEAT_SEEDS:
        raise ValueError("Stage-3 batch changed the frozen roster or repeat seeds")
    if manifest.get("frozen_test_accessed") is not False or manifest.get("test_metrics_used") is not False:
        raise RuntimeError("Stage-3 batch violates the test firewall")
    if manifest.get("budget") != stage2_budget_contract():
        raise ValueError("Stage-3 batch changed the Stage-2 budget")
    validation_cover: list[str] = []
    for fold_id in FOLD_IDS:
        fold = manifest["folds"][fold_id]
        expected_validation = DEVELOPMENT_FAMILIES[fold_id]
        if fold["fold_id"] != fold_id or fold["validation_groups"] != [expected_validation]:
            raise ValueError("Stage-3 batch is not the P4 LOGO4 projection")
        if TEST_FAMILY in set(fold["train_groups"]) | set(fold["validation_groups"]):
            raise RuntimeError("frozen family entered a Stage-3 fold")
        prefix = f"f{fold_id}"
        train_well = arrays[f"{prefix}_train_well"]
        validation_well = arrays[f"{prefix}_validation_well"]
        train_seismic = arrays[f"{prefix}_train_seismic"]
        validation_seismic = arrays[f"{prefix}_validation_seismic"]
        if tuple(train_well.shape[1:]) != (26, P_CONTEXT_LENGTH) or tuple(validation_well.shape[1:]) != (26, P_CONTEXT_LENGTH):
            raise ValueError("Stage-3 well-log context changed")
        if tuple(train_seismic.shape[1:]) != (3, 3, P_CONTEXT_LENGTH) or tuple(validation_seismic.shape[1:]) != (3, 3, P_CONTEXT_LENGTH):
            raise ValueError("Stage-3 seismic context changed")
        if len(train_well) > P_TRAIN_SAMPLE_LIMIT or len(validation_well) > P_VALIDATION_SAMPLE_LIMIT:
            raise ValueError("Stage-3 sample budget exceeded")
        for values in (train_well[:, 13:, :], validation_well[:, 13:, :]):
            if not np.isin(values, (0.0, 1.0)).all():
                raise ValueError("Stage-3 missing-mask rows are not binary")
        train_ids = set(str(value) for value in arrays[f"{prefix}_train_ids"].tolist())
        validation_ids = set(str(value) for value in arrays[f"{prefix}_validation_ids"].tolist())
        if train_ids & validation_ids:
            raise RuntimeError("Stage-3 fold has train/validation sample overlap")
        validation_cover.extend(validation_ids)
    if len(validation_cover) != manifest["development_sample_count"] or len(set(validation_cover)) != len(validation_cover):
        raise RuntimeError("Stage-3 validation folds do not cover development exactly once")
    return arrays, manifest


def _fold_arrays(arrays: Mapping[str, np.ndarray], fold_id: int) -> dict[str, np.ndarray]:
    prefix = f"f{fold_id}"
    return {
        "p_train_well": arrays[f"{prefix}_train_well"],
        "p_train_seismic": arrays[f"{prefix}_train_seismic"],
        "p_train_labels": arrays[f"{prefix}_train_labels"],
        "p_train_ids": arrays[f"{prefix}_train_ids"],
        "p_validation_well": arrays[f"{prefix}_validation_well"],
        "p_validation_seismic": arrays[f"{prefix}_validation_seismic"],
        "p_validation_labels": arrays[f"{prefix}_validation_labels"],
        "p_validation_ids": arrays[f"{prefix}_validation_ids"],
        "class_counts": arrays[f"{prefix}_class_counts"],
        "class_weights": arrays[f"{prefix}_class_weights"],
    }


def _prediction_path(output: Path, model_id: str, fold_id: int, repeat_id: int) -> Path:
    return output.parent / "predictions" / f"{model_id}__fold{fold_id}__repeat{repeat_id}.npz"


def _portable_traceback(lines: Sequence[str]) -> list[str]:
    """Remove host/worktree roots while preserving actionable failure frames."""
    portable: list[str] = []
    roots = (str(PROJECT_ROOT), str(TRACK_DIR))
    for line in lines:
        value = str(line)
        for root in roots:
            value = value.replace(root + "/", "")
            value = value.replace(root, ".")
        portable.append(value)
    return portable


def _archive_prediction(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    fold_id: int,
    labels: np.ndarray,
    logits: np.ndarray,
) -> dict[str, Any]:
    path = _track_owned(path)
    prefix = f"f{fold_id}_validation"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=arrays[f"{prefix}_ids"],
        labels=np.asarray(labels, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
        family=arrays[f"{prefix}_family_id"],
        well=arrays[f"{prefix}_well_id"],
        center_md_m=arrays[f"{prefix}_center_md_m"],
        twt_ms=arrays[f"{prefix}_twt_ms"],
        well_log_missing_fraction=arrays[f"{prefix}_well_log_missing_fraction"],
        seismic_available=arrays[f"{prefix}_seismic_available"],
    )
    return {
        "path": str(path.relative_to(CANONICAL_OUTPUT_DIR)),
        "sha256": _sha256(path),
        "rows": int(len(labels)),
        "logit_shape": list(logits.shape),
        "retained_in_commit": False,
    }


def _input_budget(fold_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_layout": "center_window_to_Bx9",
        "sample_selection": "same_fold_balanced_take_up_to_stage2_limit",
        "fold_train_sample_limit": P_TRAIN_SAMPLE_LIMIT,
        "fold_validation_sample_limit": P_VALIDATION_SAMPLE_LIMIT,
        "fold_train_samples_used": int(fold_manifest["selected_train_samples"]),
        "fold_validation_samples_used": int(fold_manifest["selected_validation_samples"]),
        "well_value_channels": 13,
        "well_missing_mask_channels": 13,
        "seismic_spatial_shape": [3, 3],
        "context_positions": P_CONTEXT_LENGTH,
        "pretrained_weights": False,
        "budget_hash": stage2_budget_contract()["budget_hash"],
    }


def cell_key(result: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(result["task_id"]),
        str(result["lane"]),
        str(result["model_id"]),
        int(result["fold_id"]),
        int(result["repeat_id"]),
    )


def expected_cell_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (TASK_ID, LANE, model_id, fold_id, repeat_id)
        for model_id in TOP3
        for fold_id in FOLD_IDS
        for repeat_id in range(len(REPEAT_SEEDS))
    )


def validate_cell_result(result: Mapping[str, Any]) -> None:
    required = {
        "task_id", "lane", "model_id", "fold_id", "repeat_id", "seed", "status",
        "reason", "split_hash", "fold_partition_hash", "input_budget", "wall_seconds",
        "wall_limit_seconds", "validation_metrics", "frozen_test_accessed", "loss_contract",
        "test_metrics_used", "rank_eligible", "parameter_updates",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"Stage-3 cell lacks required fields: {missing}")
    key = cell_key(result)
    if key not in set(expected_cell_keys()):
        raise ValueError("Stage-3 cell is outside the frozen 36-cell roster")
    repeat_id = int(result["repeat_id"])
    if int(result["seed"]) != REPEAT_SEEDS[repeat_id]:
        raise ValueError("Stage-3 cell changed the frozen repeat seed")
    if result["lane"] != LANE:
        raise ValueError("cross-lane Stage-3 cell is forbidden")
    fold_id = int(result["fold_id"])
    expected_validation = DEVELOPMENT_FAMILIES[fold_id]
    expected_train = [
        family for family in DEVELOPMENT_FAMILIES if family != expected_validation
    ]
    if list(result.get("validation_groups", ())) != [expected_validation]:
        raise ValueError("Stage-3 cell changed the frozen P4 validation family")
    if list(result.get("train_groups", ())) != expected_train:
        raise ValueError("Stage-3 cell changed the frozen P4 fold-train families")
    if TEST_FAMILY in set(result.get("train_groups", ())) | set(result.get("validation_groups", ())):
        raise RuntimeError("frozen F-5 family entered a Stage-3 cell")
    for name in ("split_hash", "fold_partition_hash"):
        value = str(result[name])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Stage-3 {name} is not a stable SHA-256 digest")
    if result.get("hpo") is not False:
        raise ValueError("Stage-3 HPO is forbidden")
    if result["loss_contract"] != locked_loss_contract(str(result["model_id"])):
        raise ValueError("Stage-3 cell changed the Stage-2 loss")
    if result["status"] not in {"PASS", "SKIP", "FAIL", "TIMEOUT"}:
        raise ValueError("invalid Stage-3 status")
    if result["frozen_test_accessed"] is not False or result["test_metrics_used"] is not False:
        raise RuntimeError("Stage-3 cell violates the frozen-test firewall")
    budget = result["input_budget"]
    if budget.get("budget_hash") != stage2_budget_contract()["budget_hash"]:
        raise ValueError("Stage-3 cell changed the Stage-2 budget")
    if int(budget["context_positions"]) != P_CONTEXT_LENGTH:
        raise ValueError("Stage-3 cell changed the context budget")
    if int(budget["fold_train_samples_used"]) > P_TRAIN_SAMPLE_LIMIT or int(budget["fold_validation_samples_used"]) > P_VALIDATION_SAMPLE_LIMIT:
        raise ValueError("Stage-3 cell exceeded the sample budget")
    if int(result["parameter_updates"]) > NEURAL_PARAMETER_UPDATE_LIMIT:
        raise ValueError("Stage-3 cell exceeded the update budget")
    preprocessing = result.get("preprocessing", {})
    if preprocessing.get("fit_scope") != "fold_train_mother_families_only":
        raise ValueError("Stage-3 preprocessing was not fit on fold-train only")
    if preprocessing.get("class_weight_fit_scope") != "fold_train_mother_families_only":
        raise ValueError("Stage-3 class weights were not fit on fold-train only")
    if set(preprocessing.get("fit_families", ())) != set(result.get("train_groups", ())):
        raise ValueError("Stage-3 preprocessing fit families disagree with fold-train")
    if set(preprocessing.get("fit_families", ())) & set(result.get("validation_groups", ())):
        raise ValueError("Stage-3 preprocessing crossed into fold-validation")
    if preprocessing.get("target_transform") != "identity_class_id":
        raise ValueError("Stage-3 changed the fixed target transform")
    if preprocessing.get("calibration") != "not_applied_locked_stage2_configuration":
        raise ValueError("Stage-3 calibration changed the Stage-2 configuration")
    if result["model_id"] == "inceptiontime_window" and result["status"] == "PASS":
        if int(result["parameter_updates"]) != NEURAL_PARAMETER_UPDATE_LIMIT:
            raise ValueError("InceptionTime did not reuse the Stage-2 update count")
    if result["model_id"] != "inceptiontime_window" and int(result["parameter_updates"]) != 0:
        raise ValueError("estimator cell reports neural parameter updates")
    if result["status"] == "PASS":
        config = result.get("model_config", {})
        if int(config.get("num_classes", 0)) != 9:
            raise ValueError("Stage-3 cell changed the nine-class output")
        if tuple(config.get("well_log_shape", ())) != (26, P_CONTEXT_LENGTH):
            raise ValueError("Stage-3 cell changed the well-log shape")
        if tuple(config.get("seismic_shape", ())) != (3, 3, P_CONTEXT_LENGTH):
            raise ValueError("Stage-3 cell changed the seismic shape")
        if result["model_id"] == "xgboost_multisoftprob_window":
            actual = (int(config.get("rounds", 0)), int(config.get("max_depth", 0)), int(config.get("seed", -1)))
            if actual != (40, 2, int(result["seed"])):
                raise ValueError("XGBoost Stage-2 configuration changed")
        elif result["model_id"] == "catboost_multiclass_window":
            actual = (int(config.get("iterations", 0)), int(config.get("depth", 0)), int(config.get("seed", -1)))
            if actual != (40, 3, int(result["seed"])):
                raise ValueError("CatBoost Stage-2 configuration changed")
        elif result["model_id"] == "inceptiontime_window":
            if (int(config.get("nf", 0)), int(config.get("kernel_size", 0))) != (8, 31):
                raise ValueError("InceptionTime Stage-2 configuration changed")
            if result.get("optimizer") != "AdamW(lr=0.001,weight_decay=0.0001)":
                raise ValueError("InceptionTime Stage-2 optimizer changed")
            if int(result.get("tiny_gate", {}).get("updates", -1)) != TINY_GATE_UPDATES:
                raise ValueError("InceptionTime Stage-2 tiny gate changed")
    if result["rank_eligible"]:
        if result["status"] != "PASS" or result["validation_metrics"] is None:
            raise ValueError("rank-eligible Stage-3 cell lacks a legal metric")
        metrics = result["validation_metrics"]
        for name in ("fixed_schema_macro_f1", "supported_class_macro_f1"):
            if not math.isfinite(float(metrics[name])):
                raise ValueError(f"Stage-3 metric {name} is not finite")
        if float(result["wall_seconds"]) > float(result["wall_limit_seconds"]):
            raise ValueError("over-budget Stage-3 cell cannot rank")
    elif result["status"] == "PASS":
        raise ValueError("successful Stage-3 cell must be rank eligible")
    if result["status"] != "PASS" and result["reason"] is None:
        raise ValueError("non-passing Stage-3 cell lacks a structured reason")


def run_cells(
    batch_file: Path,
    output: Path,
    model_ids: Sequence[str],
    fold_ids: Sequence[int],
    repeat_ids: Sequence[int],
    *,
    device: str,
) -> tuple[dict[str, Any], int]:
    output = _track_owned(output)
    _verify_external_gpu_lock(device)
    arrays, manifest = load_stage3_batch(batch_file)
    frozen = _load_frozen_sources()
    invalid_models = sorted(set(model_ids) - set(TOP3))
    if invalid_models:
        raise ValueError(f"models are outside the frozen Stage-3 top-3: {invalid_models}")
    if any(fold_id not in FOLD_IDS for fold_id in fold_ids):
        raise ValueError("Stage-3 fold selection is outside LOGO4")
    if any(repeat_id not in range(len(REPEAT_SEEDS)) for repeat_id in repeat_ids):
        raise ValueError("Stage-3 repeat selection is outside the frozen three seeds")

    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        lock = frozen["by_id"][model_id]
        for fold_id in fold_ids:
            fold_manifest = manifest["folds"][fold_id]
            fold_values = _fold_arrays(arrays, fold_id)
            for repeat_id in repeat_ids:
                seeds = component_seeds(model_id, fold_id, repeat_id)
                started = time.monotonic()
                rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                backend_hint: str | None = None
                base: dict[str, Any] = {
                    "schema_version": CELL_SCHEMA,
                    "track_id": "lithofacies",
                    "task_id": TASK_ID,
                    "lane": LANE,
                    "model_id": model_id,
                    "fold_id": fold_id,
                    "repeat_id": repeat_id,
                    "seed": seeds["model"],
                    "component_seeds": seeds,
                    "status": None,
                    "reason": None,
                    "source_revision": lock["revision"],
                    "source_lock_sha256": frozen["source_lock_sha256"],
                    "split_hash": manifest["split_hash"],
                    "fold_partition_hash": fold_manifest["partition_hash"],
                    "train_groups": fold_manifest["train_groups"],
                    "validation_groups": fold_manifest["validation_groups"],
                    "input_budget": _input_budget(fold_manifest),
                    "loss_contract": locked_loss_contract(model_id),
                    "hpo": False,
                    "preprocessing": {
                        "fit_scope": fold_manifest["preprocessor_fit_scope"],
                        "fit_families": fold_manifest["preprocessor_fit_families"],
                        "preprocessor_hash": fold_manifest["preprocessor_hash"],
                        "class_weight_fit_scope": fold_manifest["class_weight_fit_scope"],
                        "target_transform": fold_manifest["target_transform"],
                        "calibration": fold_manifest["calibration"],
                    },
                    "frozen_test_accessed": False,
                    "test_metrics_used": False,
                    "rank_eligible": False,
                    "parameter_updates": 0,
                    "technical_retry": None,
                }
                try:
                    discovered = discover_model("lithofacies", model_id)
                    if discovered.capabilities.get("leaderboard_lane") != LANE:
                        raise RuntimeError("adapter lane disagrees with the frozen P lane")
                    backend = str(discovered.capabilities.get("backend"))
                    backend_hint = backend
                    suffix = ".pt" if backend == "torch" else ".pkl"
                    checkpoint = (
                        output.parent
                        / "checkpoints"
                        / f"{model_id}__fold{fold_id}__repeat{repeat_id}{suffix}"
                    )
                    if backend == "torch":
                        evidence = _torch_pilot(
                            discovered,
                            lock,
                            fold_values,
                            device_name=device,
                            checkpoint_path=checkpoint,
                            seeds=seeds,
                        )
                        wall_limit = NEURAL_WALL_LIMIT_SECONDS
                    elif backend == "estimator":
                        if device != "cpu":
                            raise ValueError("estimator Stage-3 cells must run on CPU")
                        evidence = _estimator_pilot(
                            discovered,
                            lock,
                            fold_values,
                            checkpoint_path=checkpoint,
                            seeds=seeds,
                        )
                        wall_limit = ESTIMATOR_WALL_LIMIT_SECONDS
                    else:
                        raise ValueError(f"unknown adapter backend {backend!r}")
                    validation_labels = np.asarray(evidence.pop("validation_labels"), dtype=np.int64)
                    validation_logits = np.asarray(evidence.pop("validation_logits"), dtype=np.float32)
                    metrics = _metric_payload(
                        validation_labels,
                        validation_logits,
                        fold_manifest["validation_groups"][0],
                    )
                    prediction = _archive_prediction(
                        _prediction_path(output, model_id, fold_id, repeat_id),
                        arrays,
                        fold_id,
                        validation_labels,
                        validation_logits,
                    )
                    wall_seconds = time.monotonic() - started
                    if wall_seconds > wall_limit:
                        status = "TIMEOUT"
                        reason = _reason(
                            "wall_budget_exceeded",
                            f"cell exceeded its {wall_limit:g}s frozen Stage-2 wall budget",
                            wall_limit_seconds=wall_limit,
                        )
                        rank_eligible = False
                    else:
                        status = "PASS"
                        reason = None
                        rank_eligible = True
                except OptionalDependencyUnavailable as exc:
                    evidence = {}
                    prediction = None
                    metrics = None
                    wall_limit = ESTIMATOR_WALL_LIMIT_SECONDS if backend_hint == "estimator" else NEURAL_WALL_LIMIT_SECONDS
                    status = "SKIP"
                    reason = _reason("missing_optional_dependency", str(exc), dependency=exc.dependency)
                    rank_eligible = False
                    wall_seconds = time.monotonic() - started
                except Exception as exc:  # structured fail-loud evidence, no metric fabrication
                    evidence = {}
                    prediction = None
                    metrics = None
                    wall_limit = ESTIMATOR_WALL_LIMIT_SECONDS if backend_hint == "estimator" else NEURAL_WALL_LIMIT_SECONDS
                    status = "FAIL"
                    reason = _reason(
                        "stage3_cell_failure",
                        str(exc),
                        exception=type(exc).__name__,
                        traceback=_portable_traceback(traceback.format_exc().splitlines()[-12:]),
                    )
                    rank_eligible = False
                    wall_seconds = time.monotonic() - started
                rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                result = {
                    **base,
                    "status": status,
                    "reason": reason,
                    "rank_eligible": rank_eligible,
                    "wall_seconds": wall_seconds,
                    "wall_limit_seconds": wall_limit,
                    "peak_resources": {
                        "process_peak_rss_kib": int(max(rss_before, rss_after)),
                        "peak_vram_bytes": int(evidence.get("peak_vram_bytes", 0)),
                    },
                    "environment": _portable_environment(device),
                    "gpu_lock": {
                        "required": device.startswith("cuda"),
                        "mechanism": "external_flock" if device.startswith("cuda") else "not_applicable",
                        "verified": bool(device.startswith("cuda")),
                        "device": device,
                        "wait_excluded_from_cell_wall": True,
                    },
                    "validation_metrics": metrics,
                    "oof_prediction": prediction,
                    **evidence,
                }
                result.pop("peak_vram_bytes", None)
                validate_cell_result(result)
                results.append(result)

    partial = {
        "schema_version": PARTIAL_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "source_lock_sha256": frozen["source_lock_sha256"],
        "batch_sha256": _sha256(batch_file),
        "split_hash": manifest["split_hash"],
        "budget_hash": stage2_budget_contract()["budget_hash"],
        "models": results,
        "frozen_test_accessed": False,
        "test_metrics_used": False,
    }
    _atomic_write_json(output, partial)
    exit_code = int(any(result["status"] in {"FAIL", "TIMEOUT"} for result in results))
    return partial, exit_code


def validate_result_collection(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for source in results:
        validate_cell_result(source)
        key = cell_key(source)
        if key in by_key:
            raise ValueError(f"duplicate Stage-3 cell: {key}")
        by_key[key] = dict(source)
    expected = expected_cell_keys()
    missing = [key for key in expected if key not in by_key]
    unexpected = sorted(set(by_key) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Stage-3 36-cell roster mismatch: missing={missing}, unexpected={unexpected}")
    ordered = [by_key[key] for key in expected]
    if len({result["split_hash"] for result in ordered}) != 1:
        raise ValueError("Stage-3 result collection mixes split hashes")
    for fold_id in FOLD_IDS:
        hashes = {
            result["fold_partition_hash"]
            for result in ordered
            if result["fold_id"] == fold_id
        }
        if len(hashes) != 1:
            raise ValueError(f"Stage-3 fold {fold_id} mixes partition hashes")
    return ordered


def _bootstrap_ci(values: Sequence[float], model_id: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return (math.nan, math.nan)
    rng = np.random.default_rng(_stable_seed(ROOT_SEED, "stage3-bootstrap", model_id))
    samples = rng.choice(array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def build_leaderboard(results: Sequence[Mapping[str, Any]], split_hash: str) -> dict[str, Any]:
    expected = len(expected_cell_keys())
    legal = [result for result in results if result["status"] == "PASS" and result["rank_eligible"]]
    overall_completion = len(legal) / expected
    entries = []
    for model_id in TOP3:
        cells = [result for result in results if result["model_id"] == model_id]
        valid = [result for result in cells if result["status"] == "PASS" and result["rank_eligible"]]
        completion = len(valid) / (len(FOLD_IDS) * len(REPEAT_SEEDS))
        fold_means = {
            str(fold_id): float(
                np.mean(
                    [
                        result["validation_metrics"]["fixed_schema_macro_f1"]
                        for result in valid
                        if result["fold_id"] == fold_id
                    ]
                )
            )
            for fold_id in FOLD_IDS
            if any(result["fold_id"] == fold_id for result in valid)
        }
        seed_means = {
            str(REPEAT_SEEDS[repeat_id]): float(
                np.mean(
                    [
                        result["validation_metrics"]["fixed_schema_macro_f1"]
                        for result in valid
                        if result["repeat_id"] == repeat_id
                    ]
                )
            )
            for repeat_id in range(len(REPEAT_SEEDS))
            if any(result["repeat_id"] == repeat_id for result in valid)
        }
        model_rank_eligible = (
            completion >= MIN_LEGAL_COMPLETION
            and len(fold_means) == len(FOLD_IDS)
            and len(seed_means) == len(REPEAT_SEEDS)
        )
        fixed_values = [float(result["validation_metrics"]["fixed_schema_macro_f1"]) for result in valid]
        supported_values = [float(result["validation_metrics"]["supported_class_macro_f1"]) for result in valid]
        ci = _bootstrap_ci(fixed_values, model_id) if fixed_values else (math.nan, math.nan)
        entries.append(
            {
                "rank": None,
                "model_id": model_id,
                "status": "eligible" if model_rank_eligible else "not_rankable",
                "legal_cells": len(valid),
                "expected_cells": len(FOLD_IDS) * len(REPEAT_SEEDS),
                "completion_rate": completion,
                "fixed_schema_macro_f1_mean": float(np.mean(fixed_values)) if fixed_values else None,
                "fixed_schema_macro_f1_ci95": list(ci) if fixed_values else None,
                "worst_fold_fixed_schema_macro_f1": min(fold_means.values()) if len(fold_means) == len(FOLD_IDS) else None,
                "seed_mean_std": float(np.std(list(seed_means.values()), ddof=1)) if len(seed_means) > 1 else None,
                "supported_class_macro_f1_mean_diagnostic": float(np.mean(supported_values)) if supported_values else None,
                "fold_means": fold_means,
                "seed_means": seed_means,
                "wall_seconds_mean": float(np.mean([result["wall_seconds"] for result in valid])) if valid else None,
                "wall_seconds_max": float(np.max([result["wall_seconds"] for result in valid])) if valid else None,
                "peak_vram_bytes_max": int(max((result["peak_resources"]["peak_vram_bytes"] for result in valid), default=0)),
            }
        )
    ranked = [entry for entry in entries if entry["status"] == "eligible"]
    ranked.sort(
        key=lambda entry: (
            -float(entry["fixed_schema_macro_f1_mean"]),
            -float(entry["worst_fold_fixed_schema_macro_f1"]),
            float(entry["seed_mean_std"]),
            float(entry["wall_seconds_mean"]),
            str(entry["model_id"]),
        )
    )
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
    entries = ranked + [entry for entry in entries if entry["status"] != "eligible"]
    status = "ranked" if overall_completion >= MIN_LEGAL_COMPLETION and len(ranked) >= 2 else "not_rankable"
    return {
        "schema_version": LEADERBOARD_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": status,
        "reason": None if status == "ranked" else "legal completion below 80% or fewer than two eligible candidates",
        "expected_cells": expected,
        "legal_cells": len(legal),
        "completion_rate": overall_completion,
        "minimum_completion_rate": MIN_LEGAL_COMPLETION,
        "primary_metric": "fixed_schema_macro_f1_mean",
        "supported_class_metric_role": "diagnostic_only",
        "bootstrap": {
            "method": "nonparametric_cell_bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "confidence": 0.95,
            "seed_derivation": "sha256(root_seed,stage3-bootstrap,model_id)",
        },
        "tie_breakers": [
            "worst_fold_fixed_schema_macro_f1_desc",
            "seed_mean_std_asc",
            "wall_seconds_mean_asc",
            "model_id_asc",
        ],
        "split_hash": split_hash,
        "entries": entries,
        "development_only": True,
        "frozen_test_accessed": False,
    }


def _load_prediction(output_dir: Path, entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = output_dir / entry["path"]
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"OOF prediction archive missing or hash-mismatched: {entry['path']}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_oof_manifest(
    results: Sequence[Mapping[str, Any]], output_dir: Path, development_sample_count: int
) -> dict[str, Any]:
    entries = []
    for result in results:
        prediction = result.get("oof_prediction")
        if result["status"] != "PASS" or prediction is None:
            continue
        arrays = _load_prediction(output_dir, prediction)
        if len(arrays["labels"]) != int(prediction["rows"]):
            raise RuntimeError("OOF prediction row count changed")
        entries.append(
            {
                "model_id": result["model_id"],
                "fold_id": result["fold_id"],
                "repeat_id": result["repeat_id"],
                "seed": result["seed"],
                **prediction,
            }
        )
    coverage = []
    for model_id in TOP3:
        for repeat_id, seed in enumerate(REPEAT_SEEDS):
            selected = [
                entry for entry in entries
                if entry["model_id"] == model_id and entry["repeat_id"] == repeat_id
            ]
            sample_ids: list[str] = []
            for entry in selected:
                arrays = _load_prediction(output_dir, entry)
                sample_ids.extend(str(value) for value in arrays["sample_ids"].tolist())
            coverage.append(
                {
                    "model_id": model_id,
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "folds": sorted(entry["fold_id"] for entry in selected),
                    "prediction_rows": len(sample_ids),
                    "unique_sample_ids": len(set(sample_ids)),
                    "expected_development_samples": development_sample_count,
                    "complete_oof_cover": (
                        len(selected) == len(FOLD_IDS)
                        and len(sample_ids) == development_sample_count
                        and len(set(sample_ids)) == development_sample_count
                    ),
                }
            )
    return {
        "schema_version": OOF_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "entries": entries,
        "coverage": coverage,
        "full_predictions_committed": False,
        "prediction_storage": "track-private ignored runtime directory",
        "development_only": True,
        "frozen_test_accessed": False,
    }


def _save_figure(figure: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return {
        "path": str(path.relative_to(CANONICAL_OUTPUT_DIR)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def render_visualizations(
    results: Sequence[Mapping[str, Any]],
    leaderboard: Mapping[str, Any],
    oof_manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Render track figures only from archived development OOF predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    eligible = [entry for entry in leaderboard["entries"] if entry["rank"] is not None]
    if not eligible:
        raise RuntimeError("cannot visualize Stage-3 without an eligible development model")
    selected_model = eligible[0]["model_id"]
    prediction_entries = [
        entry for entry in oof_manifest["entries"] if entry["model_id"] == selected_model
    ]
    archived = [_load_prediction(output_dir, entry) for entry in prediction_entries]
    labels = np.concatenate([item["labels"] for item in archived]).astype(np.int64)
    logits = np.concatenate([item["logits"] for item in archived]).astype(np.float64)
    md = np.concatenate([item["center_md_m"] for item in archived]).astype(np.float64)
    missing = np.concatenate([item["well_log_missing_fraction"] for item in archived]).astype(np.float64)
    seismic_available = np.concatenate([item["seismic_available"] for item in archived]).astype(bool)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    predictions = probability.argmax(axis=1)
    metrics = classification_metrics_from_logits(labels, logits)
    entries: list[dict[str, Any]] = []

    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, matrix, title, fmt in (
        (axes[0], confusion, "Fixed-nine confusion (counts)", "d"),
        (axes[1], normalized, "Fixed-nine confusion (row-normalized)", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(9), range(9))
        axis.set_yticks(range(9), range(9))
        for row in range(9):
            for column in range(9):
                axis.text(column, row, format(matrix[row, column], fmt), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046)
    figure = _save_figure(fig, figures_dir / "fixed9_confusion.png")
    plt.close(fig)
    entries.append({"figure_id": "fixed9_confusion", "status": "PASS", **figure})

    per_class = metrics["per_class"]
    x = np.arange(9)
    width = 0.25
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(x - width, [row["precision"] for row in per_class], width, label="precision")
    axis.bar(x, [row["recall"] for row in per_class], width, label="recall")
    axis.bar(x + width, [row["f1"] for row in per_class], width, label="F1")
    axis.set_xticks(x, [f"{index}\n{CLASS_NAMES[index]}" for index in range(9)], rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Fixed-nine per-class precision / recall / F1")
    axis.legend()
    figure = _save_figure(fig, figures_dir / "fixed9_per_class_pr_f1.png")
    plt.close(fig)
    entries.append({"figure_id": "fixed9_per_class_pr_f1", "status": "PASS", **figure})

    calibration = metrics["calibration"]["bins"]
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    populated = [item for item in calibration if item["count"]]
    axis.plot(
        [item["mean_confidence"] for item in populated],
        [item["accuracy"] for item in populated],
        marker="o",
        label=f"raw softmax (ECE={metrics['expected_calibration_error']:.3f})",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title("Development OOF reliability (no calibration refit)")
    axis.legend()
    figure = _save_figure(fig, figures_dir / "calibration_reliability.png")
    plt.close(fig)
    entries.append({"figure_id": "calibration_reliability", "status": "PASS", **figure})

    fig, axes = plt.subplots(1, len(TOP3), figsize=(15, 4.8), sharey=True)
    for axis, model_id in zip(axes, TOP3):
        matrix = np.full((len(FOLD_IDS), len(REPEAT_SEEDS)), np.nan)
        for result in results:
            if result["model_id"] == model_id and result["status"] == "PASS":
                matrix[result["fold_id"], result["repeat_id"]] = result["validation_metrics"]["fixed_schema_macro_f1"]
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=max(0.3, float(np.nanmax(matrix))))
        axis.set_title(model_id)
        axis.set_xlabel("Repeat seed")
        axis.set_xticks(range(3), [str(seed) for seed in REPEAT_SEEDS], rotation=30, ha="right", fontsize=7)
        axis.set_yticks(range(4), [f"fold {fold_id}" for fold_id in FOLD_IDS])
        for row in range(4):
            for column in range(3):
                if math.isfinite(float(matrix[row, column])):
                    axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046)
    axes[0].set_ylabel("P4 LOGO fold")
    fig.suptitle("Fixed-nine Macro-F1 by fold and repeat seed")
    figure = _save_figure(fig, figures_dir / "fold_seed_matrix.png")
    plt.close(fig)
    entries.append({"figure_id": "fold_seed_matrix", "status": "PASS", **figure})

    bins = np.asarray([0.0, 0.25, 0.5, 0.75, 1.000001])
    bucket = np.clip(np.digitize(missing, bins, right=False) - 1, 0, len(bins) - 2)
    bucket_count = np.asarray([(bucket == index).sum() for index in range(len(bins) - 1)])
    bucket_accuracy = np.asarray(
        [
            float((predictions[bucket == index] == labels[bucket == index]).mean())
            if bucket_count[index]
            else 0.0
            for index in range(len(bins) - 1)
        ]
    )
    fig, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(range(4), bucket_accuracy, color="tab:blue")
    axis.set_xticks(range(4), ["0-.25", ".25-.50", ".50-.75", ".75-1.0"])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Fraction of missing well-log positions")
    axis.set_ylabel("OOF accuracy")
    axis.set_title(f"Missing-modality diagnostic; seismic available={seismic_available.mean():.1%}")
    for bar, count in zip(bars, bucket_count):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={count}", ha="center", va="bottom")
    figure = _save_figure(fig, figures_dir / "missing_modality_diagnostic.png")
    plt.close(fig)
    entries.append({"figure_id": "missing_modality_diagnostic", "status": "PASS", **figure})

    if np.isfinite(md).all() and len(md):
        order = np.argsort(md)
        fig, axes = plt.subplots(1, 3, figsize=(10, 9), sharey=True)
        axes[0].step(labels[order], md[order], where="mid")
        axes[0].set_title("Ground truth")
        axes[1].step(predictions[order], md[order], where="mid")
        axes[1].set_title("Prediction")
        axes[2].plot(probability.max(axis=1)[order], md[order])
        axes[2].set_title("Confidence")
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Measured depth (m)")
        figure = _save_figure(fig, figures_dir / "continuous_depth_facies_track.png")
        plt.close(fig)
        entries.append({"figure_id": "continuous_depth_facies_track", "status": "PASS", **figure})
    else:
        fig, axis = plt.subplots(figsize=(10, 3.5))
        axis.axis("off")
        axis.text(
            0.5,
            0.55,
            "Continuous measured-depth facies track: NOT FEASIBLE",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
        )
        axis.text(
            0.5,
            0.30,
            "The development archive stores no finite center_md_m.\nInterval midpoints and row order are not valid substitutes.",
            ha="center",
            va="center",
            fontsize=11,
        )
        figure = _save_figure(fig, figures_dir / "continuous_depth_track_not_feasible.png")
        plt.close(fig)
        entries.append(
            {
                "figure_id": "continuous_depth_facies_track",
                "status": "not_feasible",
                "reason": "development OOF archives have no finite center_md_m; midpoint fabrication forbidden",
                "finite_md_rows": int(np.isfinite(md).sum()),
                **figure,
            }
        )

    return {
        "schema_version": VISUALIZATION_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "selected_model": selected_model,
        "selection_source": LEADERBOARD_FILENAME,
        "input_oof_manifest": OOF_MANIFEST_FILENAME,
        "input_prediction_entries": len(prediction_entries),
        "calibration_fit": "not_applied_locked_stage2_configuration",
        "figures": entries,
        "renderer": "lithofacies_p5_stage3.py render",
        "rebuild_command": (
            "python3 _pipelines/02_task_datasets/lithofacies/lithofacies_p5_stage3.py "
            "render --results _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3/p5_stage3_results.jsonl "
            "--leaderboard _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3/p5_stage3_gm09_p_leaderboard.json "
            "--oof-manifest _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3/p5_stage3_oof_manifest.json "
            "--output-dir _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3"
        ),
        "development_only": True,
        "frozen_test_accessed": False,
    }


def finalize_results(
    inputs: Sequence[Path], batch_file: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir = _track_owned(output_dir)
    _, batch_manifest = load_stage3_batch(batch_file)
    partials = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not partials:
        raise ValueError("Stage-3 finalize requires partial results")
    for partial in partials:
        if partial.get("frozen_test_accessed") is not False or partial.get("test_metrics_used") is not False:
            raise RuntimeError("Stage-3 partial violates the test firewall")
        if partial.get("split_hash") != batch_manifest["split_hash"]:
            raise ValueError("Stage-3 partial split hash changed")
        if partial.get("budget_hash") != stage2_budget_contract()["budget_hash"]:
            raise ValueError("Stage-3 partial budget hash changed")
    collected = [result for partial in partials for result in partial.get("models", [])]
    for result in collected:
        reason = result.get("reason")
        if isinstance(reason, dict) and isinstance(reason.get("traceback"), list):
            reason["traceback"] = _portable_traceback(reason["traceback"])
    results = validate_result_collection(collected)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    _atomic_write_text(
        results_path,
        "".join(
            json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for result in results
        ),
    )
    leaderboard = build_leaderboard(results, batch_manifest["split_hash"])
    leaderboard_path = output_dir / LEADERBOARD_FILENAME
    _atomic_write_json(leaderboard_path, leaderboard)
    oof_manifest = build_oof_manifest(
        results, output_dir, int(batch_manifest["development_sample_count"])
    )
    oof_path = output_dir / OOF_MANIFEST_FILENAME
    _atomic_write_json(oof_path, oof_manifest)
    visualization = render_visualizations(results, leaderboard, oof_manifest, output_dir)
    visualization_path = output_dir / VISUALIZATION_MANIFEST_FILENAME
    _atomic_write_json(visualization_path, visualization)

    counts = {
        status.lower(): sum(result["status"] == status for result in results)
        for status in ("PASS", "SKIP", "FAIL", "TIMEOUT")
    }
    legal = counts["pass"]
    completion = legal / EXPECTED_CELL_COUNT
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "class_names": CLASS_NAMES,
        "class_count": 9,
        "top3": TOP3,
        "repeat_seeds": REPEAT_SEEDS,
        "fold_ids": FOLD_IDS,
        "expected_cells": EXPECTED_CELL_COUNT,
        "recorded_cells": len(results),
        "legal_cells": legal,
        "completion_rate": completion,
        "minimum_completion_rate": MIN_LEGAL_COMPLETION,
        "status": "ranked" if completion >= MIN_LEGAL_COMPLETION else "not_rankable",
        "status_counts": counts,
        "passed_cells": counts["pass"],
        "skipped_cells": counts["skip"],
        "failed_cells": counts["fail"],
        "timeout_cells": counts["timeout"],
        "budget": stage2_budget_contract(),
        "source_lock_sha256": batch_manifest["source_lock_sha256"],
        "stage2_summary_sha256": batch_manifest["stage2_summary_sha256"],
        "stage2_leaderboard_sha256": batch_manifest["stage2_leaderboard_sha256"],
        "batch_sha256": _sha256(batch_file),
        "split_hash": batch_manifest["split_hash"],
        "results_sha256": _sha256(results_path),
        "leaderboard_sha256": _sha256(leaderboard_path),
        "oof_manifest_sha256": _sha256(oof_path),
        "visualization_manifest_sha256": _sha256(visualization_path),
        "leaderboard": {
            "path": LEADERBOARD_FILENAME,
            "status": leaderboard["status"],
            "primary_metric": leaderboard["primary_metric"],
            "winner": next((entry["model_id"] for entry in leaderboard["entries"] if entry["rank"] == 1), None),
        },
        "oof": {
            "path": OOF_MANIFEST_FILENAME,
            "complete_covers": sum(entry["complete_oof_cover"] for entry in oof_manifest["coverage"]),
            "expected_covers": len(TOP3) * len(REPEAT_SEEDS),
            "full_predictions_committed": False,
        },
        "visualization": {
            "path": VISUALIZATION_MANIFEST_FILENAME,
            "figure_count": len(visualization["figures"]),
            "depth_track_status": next(
                entry["status"] for entry in visualization["figures"]
                if entry["figure_id"] == "continuous_depth_facies_track"
            ),
        },
        "development_only": True,
        "frozen_test_accessed": False,
        "test_metrics_used": False,
    }
    _atomic_write_json(output_dir / SUMMARY_FILENAME, summary)
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return TOP3
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("--models must not be empty")
    return values


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("selection must not be empty")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-batch", help="prepare P4 LOGO4 development folds")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--batch-file", type=Path, required=True)
    run = subparsers.add_parser("run", help="run frozen Stage-3 cells")
    run.add_argument("--batch-file", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models", type=_parse_models, default=TOP3)
    run.add_argument("--folds", type=_parse_ints, default=FOLD_IDS)
    run.add_argument("--repeats", type=_parse_ints, default=tuple(range(len(REPEAT_SEEDS))))
    run.add_argument("--device", default="cpu")
    finalize = subparsers.add_parser("finalize", help="create canonical Stage-3 artifacts")
    finalize.add_argument("--inputs", nargs="+", type=Path, required=True)
    finalize.add_argument("--batch-file", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    render = subparsers.add_parser("render", help="rebuild figures from archived OOF predictions")
    render.add_argument("--results", type=Path, required=True)
    render.add_argument("--leaderboard", type=Path, required=True)
    render.add_argument("--oof-manifest", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare-batch":
        payload = prepare_stage3_batch(args.dataset_root, args.batch_file)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        payload, exit_code = run_cells(
            args.batch_file,
            args.output,
            args.models,
            args.folds,
            args.repeats,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "recorded_cells": len(payload["models"]),
                    "status_counts": {
                        status: sum(result["status"] == status for result in payload["models"])
                        for status in ("PASS", "SKIP", "FAIL", "TIMEOUT")
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return exit_code
    if args.command == "render":
        results = read_jsonl(args.results)
        leaderboard = json.loads(args.leaderboard.read_text(encoding="utf-8"))
        oof_manifest = json.loads(args.oof_manifest.read_text(encoding="utf-8"))
        payload = render_visualizations(results, leaderboard, oof_manifest, _track_owned(args.output_dir))
        _atomic_write_json(_track_owned(args.output_dir) / VISUALIZATION_MANIFEST_FILENAME, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    summary = finalize_results(args.inputs, args.batch_file, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["failed_cells"] + summary["timeout_cells"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
