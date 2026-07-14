#!/usr/bin/env python3
"""Known-holdout confirmation for the frozen Stage-3 GM09 P winner.

This track-private runner deliberately does not reuse or reset the P4 Torch
lifecycle.  Its four commands separate HDF5 access from the XGBoost runtime:

1. ``prepare-development`` opens only development ``train.h5``;
2. ``refit`` trains the frozen winner without any holdout input;
3. ``prepare-holdout`` durably consumes the historically seen holdout before
   opening it and writes an ignored inference envelope;
4. ``confirm`` performs inference and archives portable evidence.

Every transition is fail-closed and single-use.  F-5 is explicitly a known,
reusable holdout and must never be described as a fresh blind test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from lithofacies_p5_stage2 import (  # noqa: E402
    ESTIMATOR_WALL_LIMIT_SECONDS,
    P_CONTEXT_LENGTH,
    P_TRAIN_SAMPLE_LIMIT,
    P_VALIDATION_SAMPLE_LIMIT,
    _atomic_write_json,
    _portable_environment,
    _seed_everything,
    _stable_hash,
    _track_owned,
)
from lithofacies_p5_stage3 import (  # noqa: E402
    LEADERBOARD_FILENAME as STAGE3_LEADERBOARD_FILENAME,
    RESULTS_FILENAME as STAGE3_RESULTS_FILENAME,
    SUMMARY_FILENAME as STAGE3_SUMMARY_FILENAME,
    locked_loss_contract,
    stage2_budget_contract,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    FoldPreprocessor,
    TEST_FAMILY,
    apply_fold_preprocessor,
    class_support,
    classification_metrics_from_logits,
    fit_fold_preprocessor,
    lithofacies_task_spec,
    sample_id,
    validate_p4_sample,
)
from p5_stage1 import (  # noqa: E402
    _balanced_take,
    _p_arrays,
    _read_development_hdf5,
    _sha256,
    load_source_lock,
)


ROOT_SEED = 2693
TASK_ID = "gm09_genetic_facies_9class"
LANE = "P"
WINNER_MODEL_ID = "xgboost_multisoftprob_window"
EVIDENCE_CLASS = "previously_seen_reusable_holdout"
EXPECTED_DEVELOPMENT_SAMPLES = 447
EXPECTED_HOLDOUT_SAMPLES = 120
EXPECTED_HOLDOUT_SUPPORT = (2, 31, 2, 17, 3, 24, 41, 0, 0)

STAGE3_DIR = TRACK_DIR / "_outputs" / "p5_stage3"
STAGE3_SUMMARY = STAGE3_DIR / STAGE3_SUMMARY_FILENAME
STAGE3_LEADERBOARD = STAGE3_DIR / STAGE3_LEADERBOARD_FILENAME
STAGE3_RESULTS = STAGE3_DIR / STAGE3_RESULTS_FILENAME
P4_DATA_MANIFEST = TRACK_DIR / "_outputs" / "split_manifest.json"
HISTORICAL_METRICS = TRACK_DIR / "_outputs" / "multimodal_mlp" / "metrics.json"
HISTORICAL_RUN_MANIFEST = TRACK_DIR / "_outputs" / "multimodal_mlp" / "run_manifest.json"
CANONICAL_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_stage4_confirmation"

STATE_CONFIG_FROZEN = "CONFIG_FROZEN"
STATE_REFIT_COMPLETE = "REFIT_COMPLETE"
STATE_HOLDOUT_CONSUMED = "KNOWN_HOLDOUT_CONSUMED"
STATE_CONFIRMATION_COMPLETE = "CONFIRMATION_COMPLETE"

CONFIG_SCHEMA = "lithofacies-p5-stage4-frozen-config-v1"
LIFECYCLE_SCHEMA = "lithofacies-p5-stage4-lifecycle-v1"
DEVELOPMENT_BATCH_SCHEMA = "lithofacies-p5-stage4-development-batch-v1"
HOLDOUT_BATCH_SCHEMA = "lithofacies-p5-stage4-known-holdout-batch-v1"
METRICS_SCHEMA = "lithofacies-p5-stage4-known-holdout-metrics-v1"
PREDICTIONS_SCHEMA = "lithofacies-p5-stage4-known-holdout-predictions-v1"
SUMMARY_SCHEMA = "lithofacies-p5-stage4-summary-v1"
ARTIFACT_SCHEMA = "lithofacies-p5-stage4-artifact-manifest-v1"
VISUALIZATION_SCHEMA = "lithofacies-p5-stage4-visualization-manifest-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _relative(path: Path, root: Path = CANONICAL_OUTPUT_DIR) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _stable_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES) or not np.isfinite(values).all():
        raise ValueError("Stage-4 logits must be finite [N,9]")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Stage-4 probabilities are not finite simplex rows")
    return probabilities


def validate_stage3_winner_payloads(
    summary: Mapping[str, Any],
    leaderboard: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed if the accepted Stage-3 winner or its budget changed."""
    if summary.get("schema_version") != "lithofacies-p5-stage3-summary-v1":
        raise ValueError("unexpected Stage-3 summary schema")
    if summary.get("task_id") != TASK_ID or summary.get("lane") != LANE:
        raise ValueError("Stage-3 task/lane changed")
    if summary.get("status") != "ranked" or summary.get("leaderboard", {}).get("winner") != WINNER_MODEL_ID:
        raise ValueError("Stage-3 winner is not the accepted XGBoost P candidate")
    if summary.get("frozen_test_accessed") is not False or summary.get("test_metrics_used") is not False:
        raise RuntimeError("Stage-3 selection used holdout evidence")
    budget = stage2_budget_contract()
    if summary.get("budget") != budget or budget.get("hpo") is not False:
        raise ValueError("Stage-3 frozen budget changed")
    if leaderboard.get("schema_version") != "lithofacies-p5-stage3-gm09-p-leaderboard-v1":
        raise ValueError("unexpected Stage-3 leaderboard schema")
    if leaderboard.get("primary_metric") != "fixed_schema_macro_f1_mean":
        raise ValueError("Stage-3 primary metric changed")
    if leaderboard.get("supported_class_metric_role") != "diagnostic_only":
        raise ValueError("supported-class metric entered Stage-3 ranking")
    if leaderboard.get("frozen_test_accessed") is not False:
        raise RuntimeError("Stage-3 leaderboard accessed the holdout")
    ranked = [entry for entry in leaderboard.get("entries", ()) if entry.get("rank") == 1]
    if len(ranked) != 1 or ranked[0].get("model_id") != WINNER_MODEL_ID:
        raise ValueError("Stage-3 rank-one entry changed")
    winner_entry = ranked[0]
    if winner_entry.get("status") != "eligible" or winner_entry.get("legal_cells") != 12:
        raise ValueError("Stage-3 winner lacks all 12 legal cells")
    winner_results = [result for result in results if result.get("model_id") == WINNER_MODEL_ID]
    if len(winner_results) != 12:
        raise ValueError("Stage-3 results do not contain 12 XGBoost cells")
    expected_seeds = {1867973658, 2137841944, 3902865753}
    if {int(result["seed"]) for result in winner_results} != expected_seeds:
        raise ValueError("Stage-3 XGBoost repeat seeds changed")
    for result in winner_results:
        if result.get("status") != "PASS" or result.get("rank_eligible") is not True:
            raise ValueError("Stage-3 XGBoost cell is not legally complete")
        if result.get("frozen_test_accessed") is not False or result.get("test_metrics_used") is not False:
            raise RuntimeError("Stage-3 XGBoost cell used holdout evidence")
        if result.get("loss_contract") != locked_loss_contract(WINNER_MODEL_ID):
            raise ValueError("Stage-3 XGBoost loss contract changed")
        config = result.get("model_config", {})
        if int(config.get("rounds", 0)) != 40 or int(config.get("max_depth", 0)) != 2:
            raise ValueError("Stage-3 XGBoost configuration changed")
        if tuple(config.get("well_log_shape", ())) != (26, P_CONTEXT_LENGTH):
            raise ValueError("Stage-3 well-log shape changed")
        if tuple(config.get("seismic_shape", ())) != (3, 3, P_CONTEXT_LENGTH):
            raise ValueError("Stage-3 seismic shape changed")
        if result.get("input_budget", {}).get("budget_hash") != budget["budget_hash"]:
            raise ValueError("Stage-3 cell budget hash changed")
    return {
        "winner_model_id": WINNER_MODEL_ID,
        "winner_rank": 1,
        "fixed_schema_macro_f1_mean": float(winner_entry["fixed_schema_macro_f1_mean"]),
        "fixed_schema_macro_f1_ci95": winner_entry["fixed_schema_macro_f1_ci95"],
        "worst_fold_fixed_schema_macro_f1": float(
            winner_entry["worst_fold_fixed_schema_macro_f1"]
        ),
        "legal_cells": 12,
        "primary_metric": "fixed_schema_macro_f1_mean",
        "supported_class_metric_role": "diagnostic_only",
        "stage3_split_hash": summary["split_hash"],
        "stage3_budget_hash": budget["budget_hash"],
    }


def verify_stage3_winner() -> dict[str, Any]:
    summary = _read_json(STAGE3_SUMMARY)
    leaderboard = _read_json(STAGE3_LEADERBOARD)
    results = _read_jsonl(STAGE3_RESULTS)
    if _sha256(STAGE3_LEADERBOARD) != summary.get("leaderboard_sha256"):
        raise RuntimeError("Stage-3 leaderboard hash mismatch")
    if _sha256(STAGE3_RESULTS) != summary.get("results_sha256"):
        raise RuntimeError("Stage-3 results hash mismatch")
    evidence = validate_stage3_winner_payloads(summary, leaderboard, results)
    return {
        **evidence,
        "summary_path": str(STAGE3_SUMMARY.relative_to(PROJECT_ROOT)),
        "summary_sha256": _sha256(STAGE3_SUMMARY),
        "leaderboard_path": str(STAGE3_LEADERBOARD.relative_to(PROJECT_ROOT)),
        "leaderboard_sha256": _sha256(STAGE3_LEADERBOARD),
        "results_path": str(STAGE3_RESULTS.relative_to(PROJECT_ROOT)),
        "results_sha256": _sha256(STAGE3_RESULTS),
    }


def verify_data_contract_manifest() -> dict[str, Any]:
    payload = _read_json(P4_DATA_MANIFEST)
    split = payload.get("split_contract", {})
    assignments = split.get("frozen_family_partitions", {})
    usable = split.get("usable_families", {})
    development = tuple(usable.get("train", ())) + tuple(usable.get("guard", ()))
    if set(development) != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError("tracked data manifest changed the four development families")
    if tuple(usable.get("test", ())) != (TEST_FAMILY,) or assignments.get(TEST_FAMILY) != "test":
        raise RuntimeError("tracked data manifest changed the F-5 holdout identity")
    if int(payload.get("sample_counts", {}).get("train", 0)) + int(
        payload.get("sample_counts", {}).get("guard", 0)
    ) != EXPECTED_DEVELOPMENT_SAMPLES:
        raise RuntimeError("tracked development sample count changed")
    if int(payload.get("sample_counts", {}).get("test", 0)) != EXPECTED_HOLDOUT_SAMPLES:
        raise RuntimeError("tracked holdout sample count changed")
    test_counts = payload.get("class_counts", {}).get("test", {})
    support = tuple(int(test_counts.get(class_name, 0)) for class_name in CLASS_NAMES)
    if support != EXPECTED_HOLDOUT_SUPPORT:
        raise RuntimeError("tracked F-5 class support changed")
    return {
        "path": str(P4_DATA_MANIFEST.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(P4_DATA_MANIFEST),
        "split_unit": split.get("unit"),
        "assignment_before": split.get("assignment_before"),
        "development_families": list(DEVELOPMENT_FAMILIES),
        "holdout_family": TEST_FAMILY,
        "development_samples": EXPECTED_DEVELOPMENT_SAMPLES,
        "holdout_samples": EXPECTED_HOLDOUT_SAMPLES,
        "holdout_class_support": list(EXPECTED_HOLDOUT_SUPPORT),
    }


def verify_prior_consumption() -> dict[str, Any]:
    metrics = _read_json(HISTORICAL_METRICS)
    run = _read_json(HISTORICAL_RUN_MANIFEST)
    historical = metrics.get("per_well_test_metrics", {}).get(TEST_FAMILY)
    if not isinstance(historical, Mapping) or int(historical.get("evaluated_samples", 0)) != 120:
        raise RuntimeError("historical F-5 metric evidence is missing")
    if run.get("test_loaded_after_best_checkpoint_selection") is not True:
        raise RuntimeError("historical test-consumption manifest changed")
    return {
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "historical_metrics_path": str(HISTORICAL_METRICS.relative_to(PROJECT_ROOT)),
        "historical_metrics_sha256": _sha256(HISTORICAL_METRICS),
        "historical_run_manifest_path": str(HISTORICAL_RUN_MANIFEST.relative_to(PROJECT_ROOT)),
        "historical_run_manifest_sha256": _sha256(HISTORICAL_RUN_MANIFEST),
        "historical_f5_accuracy": float(historical["accuracy"]),
        "historical_f5_fixed_schema_macro_f1": float(historical["macro_f1"]),
    }


def _frozen_config(stage3: Mapping[str, Any], data: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    source_lock = load_source_lock()
    candidates = [
        model for model in source_lock["models"] if model.get("model_id") == WINNER_MODEL_ID
    ]
    if len(candidates) != 1 or candidates[0].get("leaderboard_lane") != LANE:
        raise RuntimeError("source lock no longer contains the P-lane XGBoost winner")
    source = candidates[0]
    budget = stage2_budget_contract()
    payload = {
        "schema_version": CONFIG_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "model_id": WINNER_MODEL_ID,
        "seed": ROOT_SEED,
        "class_names": list(CLASS_NAMES),
        "class_count": len(CLASS_NAMES),
        "model_config": {
            "num_classes": 9,
            "well_log_shape": [26, P_CONTEXT_LENGTH],
            "seismic_shape": [3, 3, P_CONTEXT_LENGTH],
            "rounds": int(budget["xgboost"]["rounds"]),
            "max_depth": int(budget["xgboost"]["max_depth"]),
            "seed": ROOT_SEED,
        },
        "loss_contract": locked_loss_contract(WINNER_MODEL_ID),
        "budget": budget,
        "refit_policy": {
            "preprocessor_fit_scope": "all_four_development_mother_families_only",
            "class_weight_fit_scope": "all_four_development_mother_families_only",
            "preprocessor_fit_samples": EXPECTED_DEVELOPMENT_SAMPLES,
            "training_sample_selection": "stage2_balanced_take_after_preprocessing",
            "training_sample_limit": P_TRAIN_SAMPLE_LIMIT,
            "holdout_sample_limit": P_VALIDATION_SAMPLE_LIMIT,
            "calibration": "none_locked_stage3_raw_softmax_diagnostics_only",
            "target_transform": "identity_class_id",
            "hpo": False,
        },
        "stage3_winner_evidence": dict(stage3),
        "data_contract": dict(data),
        "prior_holdout_evidence": dict(prior),
        "source_lock": {
            "path": str((TRACK_DIR / "p5_source_lock.json").relative_to(PROJECT_ROOT)),
            "sha256": _sha256(TRACK_DIR / "p5_source_lock.json"),
            "revision": source["revision"],
            "license": source["license"],
            "pretrained_weights_used": False,
        },
        "p4_state_reset": False,
    }
    return {**payload, "config_hash": _hash_payload(payload)}


def _lifecycle_path(output_dir: Path) -> Path:
    return output_dir / "lifecycle.json"


def _load_lifecycle(output_dir: Path) -> dict[str, Any]:
    payload = _read_json(_lifecycle_path(output_dir))
    if payload.get("schema_version") != LIFECYCLE_SCHEMA:
        raise ValueError("unknown Stage-4 lifecycle schema")
    return payload


def require_state(output_dir: Path, expected: str) -> dict[str, Any]:
    lifecycle = _load_lifecycle(output_dir)
    if lifecycle.get("state") != expected:
        raise RuntimeError(
            f"Stage-4 requires {expected}; current state is {lifecycle.get('state')}"
        )
    return lifecycle


def _write_lifecycle(output_dir: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(_lifecycle_path(output_dir), payload)


def _validate_development(samples: Sequence[Mapping[str, Any]]) -> None:
    if len(samples) != EXPECTED_DEVELOPMENT_SAMPLES:
        raise ValueError("development archive no longer contains 447 samples")
    families = {str(sample.get("meta", {}).get("family_id")) for sample in samples}
    if families != set(DEVELOPMENT_FAMILIES) or TEST_FAMILY in families:
        raise RuntimeError("development archive violates LOGO4/F-5 isolation")
    for sample in samples:
        validate_p4_sample(sample)


def _read_known_holdout(dataset_root: Path) -> tuple[list[dict[str, Any]], Path]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("prepare-holdout requires an approved environment with h5py") from exc
    path = dataset_root.resolve() / "test.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            samples.append(
                {
                    "seismic_patch": group["seismic_patch"][()],
                    "well_log_seq": group["well_log_seq"][()],
                    "label": group["label"][()],
                    "position": json.loads(group.attrs["position"]),
                    "meta": json.loads(group.attrs["meta"]),
                }
            )
    if len(samples) != EXPECTED_HOLDOUT_SAMPLES:
        raise ValueError("known holdout no longer contains 120 samples")
    families = {str(sample.get("meta", {}).get("family_id")) for sample in samples}
    if families != {TEST_FAMILY}:
        raise RuntimeError("known holdout contains a family other than F-5")
    for sample in samples:
        validate_p4_sample(sample)
        if sample.get("meta", {}).get("partition") != "test":
            raise RuntimeError("known holdout sample is not marked test")
    if tuple(int(value) for value in class_support(samples)) != EXPECTED_HOLDOUT_SUPPORT:
        raise RuntimeError("known holdout label support changed")
    return samples, path


def _validate_frozen_config(output_dir: Path) -> dict[str, Any]:
    frozen = _read_json(output_dir / "frozen_config.json")
    config_hash = frozen.get("config_hash")
    without_hash = dict(frozen)
    without_hash.pop("config_hash", None)
    if config_hash != _hash_payload(without_hash):
        raise RuntimeError("Stage-4 frozen config hash mismatch")
    if frozen.get("model_id") != WINNER_MODEL_ID or frozen.get("seed") != ROOT_SEED:
        raise RuntimeError("Stage-4 frozen winner/seed changed")
    if frozen.get("budget") != stage2_budget_contract():
        raise RuntimeError("Stage-4 frozen Stage-2 budget changed")
    return frozen


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_npz(path: Path, schema: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
        manifest = json.loads(str(archive["manifest"].item()))
    if manifest.get("schema_version") != schema:
        raise ValueError("unknown Stage-4 runtime envelope schema")
    if manifest.get("fresh_blind") is not False or manifest.get("prior_test_consumed") is not True:
        raise RuntimeError("Stage-4 runtime envelope misstates holdout independence")
    return arrays, manifest


def prepare_development(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = _track_owned(output_dir)
    lifecycle_path = _lifecycle_path(output_dir)
    if lifecycle_path.exists():
        raise RuntimeError("Stage-4 lifecycle already exists; never reset or overwrite it")
    started = time.monotonic()
    stage3 = verify_stage3_winner()
    data = verify_data_contract_manifest()
    prior = verify_prior_consumption()
    frozen = _frozen_config(stage3, data, prior)
    development_raw, train_path = _read_development_hdf5(dataset_root)
    _validate_development(development_raw)
    preprocessor = fit_fold_preprocessor(development_raw)
    if set(preprocessor.fit_families) != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError("Stage-4 preprocessor did not fit all development families")
    development_all = apply_fold_preprocessor(development_raw, preprocessor)
    selected = _balanced_take(development_all, P_TRAIN_SAMPLE_LIMIT)
    selected_families = sorted(
        {str(sample.get("meta", {}).get("family_id")) for sample in selected}
    )
    if set(selected_families) != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError("Stage-2 training cap dropped an entire development family")
    train_well, train_seismic, train_labels, train_ids = _p_arrays(selected)
    if tuple(train_well.shape[1:]) != (26, P_CONTEXT_LENGTH):
        raise ValueError("Stage-4 development well-log shape changed")
    if tuple(train_seismic.shape[1:]) != (3, 3, P_CONTEXT_LENGTH):
        raise ValueError("Stage-4 development seismic shape changed")
    preprocessor_payload = preprocessor.to_dict()
    preprocessor_hash = _hash_payload(preprocessor_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "frozen_config.json", frozen)
    _atomic_write_json(output_dir / "preprocessor.json", preprocessor_payload)
    batch_path = output_dir / "runtime" / "development_refit.npz"
    batch_manifest = {
        "schema_version": DEVELOPMENT_BATCH_SCHEMA,
        "config_hash": frozen["config_hash"],
        "preprocessor_hash": preprocessor_hash,
        "development_families": list(DEVELOPMENT_FAMILIES),
        "full_development_samples": len(development_raw),
        "selected_training_samples": len(selected),
        "selected_training_families": selected_families,
        "full_class_support": class_support(development_raw).tolist(),
        "selected_class_support": class_support(selected).tolist(),
        "budget_hash": frozen["budget"]["budget_hash"],
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "holdout_accessed": False,
    }
    _save_npz(
        batch_path,
        {
            "manifest": np.asarray(json.dumps(batch_manifest, ensure_ascii=False, sort_keys=True)),
            "train_well": train_well,
            "train_seismic": train_seismic,
            "train_labels": train_labels,
            "train_ids": train_ids,
            "class_counts": np.asarray(preprocessor.class_support, dtype=np.int64),
            "class_weights": np.asarray(preprocessor.class_weights, dtype=np.float32),
        },
    )
    elapsed = time.monotonic() - started
    development_evidence = {
        **batch_manifest,
        "train_hdf5_basename": train_path.name,
        "train_hdf5_sha256": _sha256(train_path),
        "development_batch_path": _relative(batch_path),
        "development_batch_sha256": _sha256(batch_path),
        "preprocessor_path": "preprocessor.json",
        "preprocessor_sha256": _sha256(output_dir / "preprocessor.json"),
        "preprocessor_fit_sample_ids_hash": _stable_hash(
            sorted(preprocessor.fit_sample_ids)
        ),
        "selection_ids_hash": _stable_hash([str(value) for value in train_ids.tolist()]),
        "elapsed_seconds": elapsed,
    }
    _atomic_write_json(output_dir / "development_evidence.json", development_evidence)
    lifecycle = {
        "schema_version": LIFECYCLE_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "state": STATE_CONFIG_FROZEN,
        "config_hash": frozen["config_hash"],
        "development_batch_sha256": _sha256(batch_path),
        "preprocessor_hash": preprocessor_hash,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "p4_state_reset": False,
        "known_holdout_consumed_at": None,
        "transitions": [
            {
                "state": STATE_CONFIG_FROZEN,
                "evidence": {
                    "config_hash": frozen["config_hash"],
                    "development_batch_sha256": _sha256(batch_path),
                    "holdout_accessed": False,
                },
            }
        ],
    }
    _write_lifecycle(output_dir, lifecycle)
    return {
        "status": STATE_CONFIG_FROZEN,
        "config_hash": frozen["config_hash"],
        "development_samples": len(development_raw),
        "selected_training_samples": len(selected),
        "holdout_accessed": False,
        "elapsed_seconds": elapsed,
    }


def refit_winner(output_dir: Path) -> dict[str, Any]:
    output_dir = _track_owned(output_dir)
    lifecycle = require_state(output_dir, STATE_CONFIG_FROZEN)
    frozen = _validate_frozen_config(output_dir)
    batch_path = output_dir / "runtime" / "development_refit.npz"
    if _sha256(batch_path) != lifecycle.get("development_batch_sha256"):
        raise RuntimeError("development batch hash mismatch")
    arrays, manifest = _load_npz(batch_path, DEVELOPMENT_BATCH_SCHEMA)
    if manifest.get("holdout_accessed") is not False:
        raise RuntimeError("development refit envelope accessed the holdout")
    if manifest.get("config_hash") != frozen["config_hash"]:
        raise RuntimeError("development refit envelope config mismatch")
    if set(manifest.get("development_families", ())) != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError("development refit envelope changed mother families")
    train_well = np.asarray(arrays["train_well"], dtype=np.float32)
    train_seismic = np.asarray(arrays["train_seismic"], dtype=np.float32)
    train_labels = np.asarray(arrays["train_labels"], dtype=np.int64)
    if len(train_labels) != P_TRAIN_SAMPLE_LIMIT:
        raise RuntimeError("Stage-4 refit did not reuse the 320-sample budget")
    discovered = discover_model("lithofacies", WINNER_MODEL_ID)
    capabilities = discovered.capabilities
    if capabilities.get("leaderboard_lane") != LANE or capabilities.get("backend") != "estimator":
        raise RuntimeError("frozen winner is no longer the P-lane estimator adapter")
    started = time.monotonic()
    _seed_everything(ROOT_SEED)
    model = discovered.build(lithofacies_task_spec(), **frozen["model_config"])
    fit_loss = float(
        model.fit_stage1(
            train_well,
            train_seismic,
            train_labels,
            class_counts=arrays["class_counts"],
        )
    )
    if not math.isfinite(fit_loss):
        raise RuntimeError("Stage-4 XGBoost refit loss is not finite")
    reference_logits = np.asarray(
        model.predict_logits(train_well, train_seismic), dtype=np.float32
    )
    _stable_probabilities(reference_logits)
    checkpoint_path = output_dir / "runtime" / "refit_checkpoint.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": "lithofacies-p5-stage4-refit-checkpoint-v1",
                "model_id": WINNER_MODEL_ID,
                "config_hash": frozen["config_hash"],
                "model_config": frozen["model_config"],
                "budget_hash": frozen["budget"]["budget_hash"],
                "preprocessor_hash": manifest["preprocessor_hash"],
                "model": model,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with checkpoint_path.open("rb") as handle:
        reloaded = pickle.load(handle)
    roundtrip_logits = np.asarray(
        reloaded["model"].predict_logits(train_well, train_seismic), dtype=np.float32
    )
    roundtrip_max_abs = float(np.max(np.abs(reference_logits - roundtrip_logits)))
    if roundtrip_max_abs != 0.0:
        raise RuntimeError("Stage-4 refit checkpoint changed XGBoost logits")
    elapsed = time.monotonic() - started
    if elapsed > ESTIMATOR_WALL_LIMIT_SECONDS:
        raise RuntimeError("Stage-4 refit exceeded the frozen estimator wall budget")
    checkpoint_hash = _sha256(checkpoint_path)
    evidence = {
        "schema_version": "lithofacies-p5-stage4-refit-evidence-v1",
        "status": STATE_REFIT_COMPLETE,
        "model_id": WINNER_MODEL_ID,
        "seed": ROOT_SEED,
        "config_hash": frozen["config_hash"],
        "budget_hash": frozen["budget"]["budget_hash"],
        "loss_contract": frozen["loss_contract"],
        "development_families": list(DEVELOPMENT_FAMILIES),
        "preprocessor_fit_scope": "all_four_development_mother_families_only",
        "class_weight_fit_scope": "all_four_development_mother_families_only",
        "preprocessor_fit_samples": manifest["full_development_samples"],
        "selected_training_samples": len(train_labels),
        "full_class_support": manifest["full_class_support"],
        "selected_class_support": manifest["selected_class_support"],
        "fit_loss": fit_loss,
        "estimator_fit_calls": 1,
        "boost_rounds": frozen["model_config"]["rounds"],
        "checkpoint": {
            "path": _relative(checkpoint_path),
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
            "roundtrip": "PASS",
            "roundtrip_max_abs": roundtrip_max_abs,
            "retained_in_commit": False,
        },
        "environment": _portable_environment("cpu"),
        "holdout_accessed": False,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "wall_limit_seconds": ESTIMATOR_WALL_LIMIT_SECONDS,
        "elapsed_seconds": elapsed,
    }
    _atomic_write_json(output_dir / "refit_evidence.json", evidence)
    lifecycle["state"] = STATE_REFIT_COMPLETE
    lifecycle["checkpoint_sha256"] = checkpoint_hash
    lifecycle["transitions"].append(
        {
            "state": STATE_REFIT_COMPLETE,
            "evidence": {
                "config_hash": frozen["config_hash"],
                "checkpoint_sha256": checkpoint_hash,
                "holdout_accessed": False,
            },
        }
    )
    _write_lifecycle(output_dir, lifecycle)
    return {
        "status": STATE_REFIT_COMPLETE,
        "fit_loss": fit_loss,
        "checkpoint_sha256": checkpoint_hash,
        "holdout_accessed": False,
        "elapsed_seconds": elapsed,
    }


def prepare_known_holdout(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = _track_owned(output_dir)
    lifecycle = require_state(output_dir, STATE_REFIT_COMPLETE)
    frozen = _validate_frozen_config(output_dir)
    refit = _read_json(output_dir / "refit_evidence.json")
    checkpoint_path = output_dir / refit["checkpoint"]["path"]
    if _sha256(checkpoint_path) != lifecycle.get("checkpoint_sha256"):
        raise RuntimeError("refit checkpoint hash mismatch before holdout consumption")
    if refit.get("holdout_accessed") is not False:
        raise RuntimeError("refit evidence reports premature holdout access")
    consumed_at = datetime.now(timezone.utc).isoformat()
    lifecycle["state"] = STATE_HOLDOUT_CONSUMED
    lifecycle["known_holdout_consumed_at"] = consumed_at
    lifecycle["transitions"].append(
        {
            "state": STATE_HOLDOUT_CONSUMED,
            "evidence": {
                "config_hash": frozen["config_hash"],
                "checkpoint_sha256": lifecycle["checkpoint_sha256"],
                "holdout_family": TEST_FAMILY,
                "prior_test_consumed": True,
                "fresh_blind": False,
                "evidence_class": EVIDENCE_CLASS,
                "consumed_before_hdf5_open": True,
                "consumed_at": consumed_at,
            },
        }
    )
    # Durable single-use transition must precede the only Stage-4 test HDF5 open.
    _write_lifecycle(output_dir, lifecycle)
    started = time.monotonic()
    holdout_raw, test_path = _read_known_holdout(dataset_root)
    preprocessor = FoldPreprocessor.from_dict(
        _read_json(output_dir / "preprocessor.json")
    )
    holdout = apply_fold_preprocessor(holdout_raw, preprocessor)
    well, seismic, labels, ids = _p_arrays(holdout)
    if len(labels) > P_VALIDATION_SAMPLE_LIMIT:
        raise RuntimeError("known holdout exceeds the frozen Stage-2 validation budget")
    if tuple(well.shape[1:]) != (26, P_CONTEXT_LENGTH) or tuple(seismic.shape[1:]) != (
        3,
        3,
        P_CONTEXT_LENGTH,
    ):
        raise ValueError("known holdout input shape changed")
    well_ids = np.asarray(
        [str(sample.get("position", {}).get("well_name", "")) for sample in holdout_raw]
    )
    families = np.asarray(
        [str(sample.get("meta", {}).get("family_id", "")) for sample in holdout_raw]
    )
    center_md = np.asarray(
        [
            np.nan
            if sample.get("position", {}).get("center_md_m") is None
            else float(sample["position"]["center_md_m"])
            for sample in holdout_raw
        ],
        dtype=np.float64,
    )
    twt = np.asarray(
        [float(sample.get("position", {}).get("time_ms", np.nan)) for sample in holdout_raw],
        dtype=np.float64,
    )
    missing = 1.0 - well[:, 13:, :].mean(axis=(1, 2))
    batch_path = output_dir / "runtime" / "known_holdout.npz"
    batch_manifest = {
        "schema_version": HOLDOUT_BATCH_SCHEMA,
        "config_hash": frozen["config_hash"],
        "checkpoint_sha256": lifecycle["checkpoint_sha256"],
        "preprocessor_sha256": _sha256(output_dir / "preprocessor.json"),
        "holdout_family": TEST_FAMILY,
        "holdout_samples": len(labels),
        "class_support": class_support(holdout_raw).tolist(),
        "known_holdout_consumed_at": consumed_at,
        "consumed_before_hdf5_open": True,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "calibration_fit": "none_test_labels_never_fit",
    }
    _save_npz(
        batch_path,
        {
            "manifest": np.asarray(json.dumps(batch_manifest, ensure_ascii=False, sort_keys=True)),
            "well": well,
            "seismic": seismic,
            "labels": labels,
            "sample_ids": ids,
            "well_ids": well_ids,
            "families": families,
            "center_md_m": center_md,
            "twt_ms": twt,
            "well_log_missing_fraction": missing.astype(np.float32),
            "seismic_available": np.ones(len(labels), dtype=np.uint8),
        },
    )
    elapsed = time.monotonic() - started
    evidence = {
        **batch_manifest,
        "test_hdf5_basename": test_path.name,
        "test_hdf5_sha256": _sha256(test_path),
        "holdout_batch_path": _relative(batch_path),
        "holdout_batch_sha256": _sha256(batch_path),
        "finite_center_md_rows": int(np.isfinite(center_md).sum()),
        "elapsed_seconds": elapsed,
    }
    _atomic_write_json(output_dir / "holdout_data_evidence.json", evidence)
    lifecycle["holdout_batch_sha256"] = _sha256(batch_path)
    lifecycle["transitions"][-1]["evidence"]["holdout_batch_sha256"] = _sha256(
        batch_path
    )
    lifecycle["transitions"][-1]["evidence"]["holdout_samples"] = len(labels)
    _write_lifecycle(output_dir, lifecycle)
    return {
        "status": STATE_HOLDOUT_CONSUMED,
        "holdout_family": TEST_FAMILY,
        "holdout_samples": len(labels),
        "class_support": class_support(holdout_raw).tolist(),
        "prior_test_consumed": True,
        "fresh_blind": False,
        "elapsed_seconds": elapsed,
    }


def _save_figure(figure: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def render_confirmation_figures(
    output_dir: Path,
    metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    entries: list[dict[str, Any]] = []
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for axis, matrix, title, format_string in (
        (axes[0], confusion, "Known F-5 confusion (counts)", "d"),
        (axes[1], normalized, "Known F-5 confusion (row-normalized)", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(9))
        axis.set_yticks(range(9))
        axis.set_title(title)
        for row in range(9):
            for column in range(9):
                axis.text(column, row, format(matrix[row, column], format_string), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046)
    figure = _save_figure(fig, figures_dir / "fixed9_confusion.png")
    plt.close(fig)
    entries.append({"figure_id": "fixed9_confusion", "status": "PASS", **figure})

    per_class = metrics["per_class"]
    x = np.arange(9)
    width = 0.24
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(x - width, [row["precision"] for row in per_class], width, label="precision")
    axis.bar(x, [row["recall"] for row in per_class], width, label="recall")
    axis.bar(x + width, [row["f1"] for row in per_class], width, label="F1")
    axis.set_xticks(x, [f"{index}\n{CLASS_NAMES[index]}" for index in range(9)], rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    support_axis = axis.twinx()
    support_axis.plot(x, [row["support"] for row in per_class], "ko--", label="support")
    support_axis.set_ylabel("Support")
    axis.set_title("Known F-5 fixed-nine precision / recall / F1 / support")
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = support_axis.get_legend_handles_labels()
    axis.legend(handles + extra_handles, labels + extra_labels, loc="upper left")
    figure = _save_figure(fig, figures_dir / "fixed9_per_class_pr_f1_support.png")
    plt.close(fig)
    entries.append({"figure_id": "fixed9_per_class_pr_f1_support", "status": "PASS", **figure})

    bins = [row for row in metrics["calibration"]["bins"] if row["count"]]
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    axis.plot(
        [row["mean_confidence"] for row in bins],
        [row["accuracy"] for row in bins],
        marker="o",
        label=f"raw softmax (ECE={metrics['expected_calibration_error']:.3f})",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title("Known F-5 reliability; no test calibration fit")
    axis.legend()
    figure = _save_figure(fig, figures_dir / "calibration_reliability.png")
    plt.close(fig)
    entries.append({"figure_id": "calibration_reliability", "status": "PASS", **figure})

    finite_md = [
        record for record in records
        if record.get("center_md_m") is not None and math.isfinite(float(record["center_md_m"]))
    ]
    if len(finite_md) == len(records) and records:
        ordered = sorted(records, key=lambda record: float(record["center_md_m"]))
        md = [float(record["center_md_m"]) for record in ordered]
        truth = [int(record["true_class_id"]) for record in ordered]
        prediction = [int(record["predicted_class_id"]) for record in ordered]
        confidence = [float(record["confidence"]) for record in ordered]
        error = [int(record["error"]) for record in ordered]
        fig, axes = plt.subplots(1, 4, figsize=(12, 9), sharey=True)
        axes[0].step(truth, md, where="mid")
        axes[0].set_title("Ground truth")
        axes[1].step(prediction, md, where="mid")
        axes[1].set_title("Prediction")
        axes[2].plot(confidence, md)
        axes[2].set_title("Confidence")
        axes[3].step(error, md, where="mid")
        axes[3].set_title("Error")
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Measured depth (m)")
        figure = _save_figure(fig, figures_dir / "continuous_depth_facies_track.png")
        plt.close(fig)
        entries.append({"figure_id": "continuous_depth_facies_track", "status": "PASS", **figure})
    else:
        fig, axis = plt.subplots(figsize=(10, 3.5))
        axis.axis("off")
        axis.text(0.5, 0.58, "Continuous measured-depth track: NOT FEASIBLE", ha="center", va="center", fontsize=16, weight="bold")
        axis.text(0.5, 0.30, "F-5 archive stores no finite center_md_m.\nInterval midpoints and row order are not substitutes.", ha="center", va="center", fontsize=11)
        figure = _save_figure(fig, figures_dir / "continuous_depth_track_not_feasible.png")
        plt.close(fig)
        entries.append(
            {
                "figure_id": "continuous_depth_facies_track",
                "status": "not_feasible",
                "reason": "known F-5 archive has no finite center_md_m; midpoint fabrication forbidden",
                "finite_md_rows": len(finite_md),
                **figure,
            }
        )
    return {
        "schema_version": VISUALIZATION_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "figures": entries,
    }


def _prediction_records(
    arrays: Mapping[str, np.ndarray], logits: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    predictions = probabilities.argmax(axis=1)
    records: list[dict[str, Any]] = []
    for index in range(len(predictions)):
        truth = int(arrays["labels"][index])
        predicted = int(predictions[index])
        md = float(arrays["center_md_m"][index])
        records.append(
            {
                "sample_id": str(arrays["sample_ids"][index]),
                "well_id": str(arrays["well_ids"][index]),
                "family_id": str(arrays["families"][index]),
                "center_md_m": md if math.isfinite(md) else None,
                "twt_ms": float(arrays["twt_ms"][index]),
                "true_class_id": truth,
                "true_class_name": CLASS_NAMES[truth],
                "predicted_class_id": predicted,
                "predicted_class_name": CLASS_NAMES[predicted],
                "confidence": float(probabilities[index, predicted]),
                "error": predicted != truth,
                "well_log_missing_fraction": float(
                    arrays["well_log_missing_fraction"][index]
                ),
                "seismic_available": bool(arrays["seismic_available"][index]),
                "logits": logits[index].astype(float).tolist(),
                "probabilities": probabilities[index].astype(float).tolist(),
            }
        )
    return records


def _artifact_entry(output_dir: Path, path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path, output_dir),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def confirm_known_holdout(output_dir: Path) -> dict[str, Any]:
    output_dir = _track_owned(output_dir)
    lifecycle = require_state(output_dir, STATE_HOLDOUT_CONSUMED)
    frozen = _validate_frozen_config(output_dir)
    holdout_path = output_dir / "runtime" / "known_holdout.npz"
    if _sha256(holdout_path) != lifecycle.get("holdout_batch_sha256"):
        raise RuntimeError("known-holdout envelope hash mismatch")
    arrays, manifest = _load_npz(holdout_path, HOLDOUT_BATCH_SCHEMA)
    if manifest.get("holdout_family") != TEST_FAMILY:
        raise RuntimeError("known-holdout envelope changed the F-5 identity")
    if tuple(int(value) for value in manifest.get("class_support", ())) != EXPECTED_HOLDOUT_SUPPORT:
        raise RuntimeError("known-holdout envelope changed label support")
    checkpoint_path = output_dir / "runtime" / "refit_checkpoint.pkl"
    if _sha256(checkpoint_path) != lifecycle.get("checkpoint_sha256"):
        raise RuntimeError("refit checkpoint hash mismatch at confirmation")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if checkpoint.get("config_hash") != frozen["config_hash"]:
        raise RuntimeError("refit checkpoint no longer matches frozen config")
    started = time.monotonic()
    logits = np.asarray(
        checkpoint["model"].predict_logits(arrays["well"], arrays["seismic"]),
        dtype=np.float32,
    )
    probabilities = _stable_probabilities(logits)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    metrics = classification_metrics_from_logits(labels.tolist(), logits)
    metrics["fixed_schema_macro_f1"] = float(metrics["macro_f1"])
    metrics.update(
        {
            "schema_version": METRICS_SCHEMA,
            "track_id": "lithofacies",
            "task_id": TASK_ID,
            "lane": LANE,
            "model_id": WINNER_MODEL_ID,
            "seed": ROOT_SEED,
            "holdout_family": TEST_FAMILY,
            "primary_metric": "fixed_schema_macro_f1",
            "supported_class_metric_role": "diagnostic_only",
            "calibration_role": "raw_softmax_diagnostic_only",
            "calibration_fit": "none_test_labels_never_fit",
            "prior_test_consumed": True,
            "fresh_blind": False,
            "evidence_class": EVIDENCE_CLASS,
            "config_hash": frozen["config_hash"],
            "checkpoint_sha256": lifecycle["checkpoint_sha256"],
        }
    )
    if int(metrics["evaluated_samples"]) != EXPECTED_HOLDOUT_SAMPLES:
        raise RuntimeError("Stage-4 did not evaluate all 120 F-5 samples")
    if tuple(row["support"] for row in metrics["per_class"]) != EXPECTED_HOLDOUT_SUPPORT:
        raise RuntimeError("Stage-4 metric support changed")
    for key in (
        "accuracy",
        "fixed_schema_macro_f1",
        "supported_class_macro_f1",
        "negative_log_likelihood",
        "multiclass_brier",
        "expected_calibration_error",
    ):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"Stage-4 metric {key} is not finite")
    records = _prediction_records(arrays, logits, probabilities)
    predictions_payload = {
        "schema_version": PREDICTIONS_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "model_id": WINNER_MODEL_ID,
        "seed": ROOT_SEED,
        "holdout_family": TEST_FAMILY,
        "config_hash": frozen["config_hash"],
        "checkpoint_sha256": lifecycle["checkpoint_sha256"],
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "calibration_fit": "none_test_labels_never_fit",
        "records": records,
    }
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"
    _atomic_write_json(predictions_path, predictions_payload)
    _atomic_write_json(metrics_path, metrics)
    visualization = render_confirmation_figures(output_dir, metrics, records)
    visualization_path = output_dir / "visualization_manifest.json"
    _atomic_write_json(visualization_path, visualization)
    elapsed = time.monotonic() - started
    if elapsed > ESTIMATOR_WALL_LIMIT_SECONDS:
        raise RuntimeError("Stage-4 confirmation exceeded the estimator wall budget")
    lifecycle["state"] = STATE_CONFIRMATION_COMPLETE
    lifecycle["transitions"].append(
        {
            "state": STATE_CONFIRMATION_COMPLETE,
            "evidence": {
                "predictions_sha256": _sha256(predictions_path),
                "metrics_sha256": _sha256(metrics_path),
                "visualization_manifest_sha256": _sha256(visualization_path),
                "evaluated_samples": EXPECTED_HOLDOUT_SAMPLES,
                "fixed_schema_macro_f1": metrics["fixed_schema_macro_f1"],
            },
        }
    )
    _write_lifecycle(output_dir, lifecycle)
    artifacts: list[dict[str, Any]] = []
    for filename, role in (
        ("frozen_config.json", "configuration"),
        ("preprocessor.json", "preprocessing"),
        ("development_evidence.json", "development_data"),
        ("refit_evidence.json", "refit"),
        ("holdout_data_evidence.json", "known_holdout_data"),
        ("predictions.json", "prediction"),
        ("metrics.json", "metric"),
        ("visualization_manifest.json", "visualization_manifest"),
        ("lifecycle.json", "lifecycle"),
    ):
        artifacts.append(_artifact_entry(output_dir, output_dir / filename, role))
    for figure in visualization["figures"]:
        artifacts.append(
            _artifact_entry(output_dir, output_dir / figure["path"], "visualization")
        )
    artifact_manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "runtime_checkpoint_committed": False,
        "runtime_batches_committed": False,
        "artifacts": artifacts,
    }
    artifact_path = output_dir / "artifact_manifest.json"
    _atomic_write_json(artifact_path, artifact_manifest)
    depth = next(
        figure for figure in visualization["figures"]
        if figure["figure_id"] == "continuous_depth_facies_track"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": STATE_CONFIRMATION_COMPLETE,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": LANE,
        "model_id": WINNER_MODEL_ID,
        "seed": ROOT_SEED,
        "config_hash": frozen["config_hash"],
        "stage3_winner": frozen["stage3_winner_evidence"],
        "holdout_family": TEST_FAMILY,
        "holdout_samples": EXPECTED_HOLDOUT_SAMPLES,
        "class_support": list(EXPECTED_HOLDOUT_SUPPORT),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "fixed_schema_macro_f1": metrics["fixed_schema_macro_f1"],
            "supported_class_macro_f1_diagnostic": metrics["supported_class_macro_f1"],
            "balanced_accuracy_supported_classes": metrics["balanced_accuracy"],
            "negative_log_likelihood": metrics["negative_log_likelihood"],
            "multiclass_brier": metrics["multiclass_brier"],
            "expected_calibration_error": metrics["expected_calibration_error"],
        },
        "primary_metric": "fixed_schema_macro_f1",
        "supported_class_metric_role": "diagnostic_only",
        "calibration_fit": "none_test_labels_never_fit",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "evidence_class": EVIDENCE_CLASS,
        "known_holdout_consumed_at": lifecycle["known_holdout_consumed_at"],
        "p4_state_reset": False,
        "depth_track_status": depth["status"],
        "predictions_sha256": _sha256(predictions_path),
        "metrics_sha256": _sha256(metrics_path),
        "visualization_manifest_sha256": _sha256(visualization_path),
        "artifact_manifest_sha256": _sha256(artifact_path),
        "checkpoint_sha256": lifecycle["checkpoint_sha256"],
        "runtime_checkpoint_committed": False,
        "runtime_batches_committed": False,
        "environment": _portable_environment("cpu"),
        "elapsed_seconds": elapsed,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-development", help="freeze winner and prepare development-only refit envelope"
    )
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    refit = subparsers.add_parser("refit", help="refit frozen XGBoost on development only")
    refit.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    holdout = subparsers.add_parser(
        "prepare-holdout", help="durably consume then read historically seen F-5"
    )
    holdout.add_argument("--dataset-root", type=Path, required=True)
    holdout.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    confirm = subparsers.add_parser("confirm", help="evaluate and archive known-holdout evidence")
    confirm.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare-development":
            result = prepare_development(args.dataset_root, args.output_dir)
        elif args.command == "refit":
            result = refit_winner(args.output_dir)
        elif args.command == "prepare-holdout":
            result = prepare_known_holdout(args.dataset_root, args.output_dir)
        else:
            result = confirm_known_holdout(args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
