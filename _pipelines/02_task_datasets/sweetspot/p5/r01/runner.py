"""Bounded P5.1 R0/R1 runner for seven independent sweetspot targets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split

from _models.sweetspot.logistic_classifier import build_model as build_logistic
from _models.sweetspot.robust_linear import build_model as build_robust_linear
from .contracts import CONTRACT_ORDER, HERE, ROOT_SEED, canonical_sha256, load_contracts, sha256_file, task_spec
from .data import R01Dataset, build_development_datasets


DEFAULT_OUTPUT_DIR = HERE / "_outputs"
OUTPUT_FILES = (
    "r0_contract_registry.json", "r0_data_audit.json", "r1_results.jsonl",
    "r1_summary.json", "artifact_manifest.json",
)


def _json_bytes(payload: Any, *, indent: int | None = 2) -> bytes:
    return (json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False,
    ) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return {"mae": None, "rmse": None, "spearman": None}
    residual = prediction[valid] - truth[valid]
    rho = spearmanr(truth[valid], prediction[valid]).statistic if valid.sum() > 1 else np.nan
    return {
        "mae": _finite(np.mean(np.abs(residual))),
        "rmse": _finite(np.sqrt(np.mean(np.square(residual)))),
        "spearman": _finite(rho),
    }


def binary_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(probability)
    truth = truth[valid]
    probability = probability[valid]
    if not len(truth):
        return {"average_precision": None, "brier": None, "f1_at_0.5": None}
    ap = average_precision_score(truth, probability) if len(np.unique(truth)) == 2 else np.nan
    return {
        "average_precision": _finite(ap),
        "brier": _finite(brier_score_loss(truth, probability)),
        "f1_at_0.5": _finite(f1_score(truth, probability >= 0.5, zero_division=0)),
    }


def _target_forward(contract: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    if contract["label"].get("model_transform") == "log1p":
        if np.any(values < 0):
            raise ValueError(f"{contract['target_id']}: log1p target contains negative values")
        return np.log1p(values)
    return np.asarray(values, dtype=float)


def _target_inverse(contract: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    if contract["label"].get("model_transform") == "log1p":
        return np.maximum(0.0, np.expm1(values))
    return np.asarray(values, dtype=float)


def _preprocessing_evidence(model: Any, train_indices: np.ndarray, dataset: R01Dataset) -> dict[str, Any]:
    steps = model.pipeline.named_steps
    imputer = steps["imputer"]
    scaler = steps["scaler"]
    statistics = {
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=float).round(12).tolist(),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).round(12).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).round(12).tolist(),
    }
    estimator = steps.get("regressor", steps.get("classifier"))
    iterations = getattr(estimator, "n_iter_", None)
    iteration_count = int(np.max(np.asarray(iterations))) if iterations is not None else None
    max_iter = getattr(estimator, "max_iter", None)
    return {
        "fit_scope": "fold_train_only",
        "fit_sample_count": int(len(train_indices)),
        "fit_sample_ids_sha256": canonical_sha256([dataset.sample_ids[index] for index in train_indices]),
        "statistics_sha256": canonical_sha256(statistics),
        "class_weight_fit_scope": "fold_train_only" if dataset.task_type == "binary" else "not_applicable",
        "target_transform_fit_scope": "fixed_formula_no_fit",
        "threshold_fit_scope": "fixed_0.5_no_fit" if dataset.task_type == "binary" else "not_applicable",
        "calibration": "none",
        "optimizer_iterations": iteration_count,
        "optimizer_max_iter": max_iter,
        "optimizer_reached_budget": bool(
            iteration_count is not None and max_iter is not None and iteration_count >= max_iter
        ),
    }


def _fit_predict(
    contract: Mapping[str, Any], dataset: R01Dataset, train_indices: np.ndarray, validation_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = task_spec(contract)
    target_name = dataset.target_name
    train_targets = _target_forward(contract, dataset.targets[train_indices])
    if dataset.task_type == "binary":
        model = build_logistic(spec, c=1.0, class_weight="balanced", max_iter=500)
    else:
        model = build_robust_linear(spec, estimator="huber", alpha=1e-4, epsilon=1.35)
    model.fit(
        dataset.features[train_indices],
        {target_name: train_targets},
        {target_name: np.ones(len(train_indices), dtype=bool)},
    )
    output = model.predict(dataset.features[validation_indices])
    if dataset.task_type == "binary":
        prediction = np.asarray(output.transformed[target_name], dtype=float)
    else:
        prediction = _target_inverse(contract, np.asarray(output.raw[target_name], dtype=float))
    return prediction, _preprocessing_evidence(model, train_indices, dataset)


def legal_group_folds(dataset: R01Dataset) -> list[dict[str, Any]]:
    groups = np.asarray(dataset.groups, dtype=object)
    folds: list[dict[str, Any]] = []
    for fold_index, validation_group in enumerate(dataset.development_groups):
        validation = np.flatnonzero(groups == validation_group)
        train = np.flatnonzero(groups != validation_group)
        if not len(validation):
            folds.append({
                "fold_index": fold_index, "validation_group": validation_group,
                "status": "blocked", "reason": "development_group_has_no_eligible_samples",
            })
            continue
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("legal group fold has train/validation group overlap")
        folds.append({
            "fold_index": fold_index, "validation_group": validation_group,
            "status": "ready", "train_indices": train, "validation_indices": validation,
            "train_groups": sorted(set(groups[train].tolist())),
            "validation_groups": [validation_group],
        })
    return folds


def _evaluate_legal(contract: Mapping[str, Any], dataset: R01Dataset) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    completed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for fold in legal_group_folds(dataset):
        if fold["status"] != "ready":
            blocked.append(fold)
            continue
        train = fold.pop("train_indices")
        validation = fold.pop("validation_indices")
        try:
            prediction, preprocessing = _fit_predict(contract, dataset, train, validation)
        except (ValueError, RuntimeError) as exc:
            blocked.append({**fold, "status": "blocked", "reason": type(exc).__name__, "detail": str(exc)})
            continue
        truth = dataset.targets[validation]
        metrics = binary_metrics(truth, prediction) if dataset.task_type == "binary" else regression_metrics(truth, prediction)
        completed.append({
            **fold, "status": "completed", "train_count": len(train), "validation_count": len(validation),
            "metrics": metrics, "preprocessing": preprocessing,
            "horizon_purge_days": 30 if dataset.target_id in {"T3", "T4"} else None,
            "horizon_purge_application": (
                "validation well is wholly held out; this is stricter than excluding only origins within 30 days"
                if dataset.target_id in {"T3", "T4"} else None
            ),
            "validation_cutoff_min": min(dataset.cutoffs[index] for index in validation) if dataset.target_id in {"T3", "T4"} else None,
            "validation_cutoff_max": max(dataset.cutoffs[index] for index in validation) if dataset.target_id in {"T3", "T4"} else None,
        })
        predictions.append(prediction)
        truths.append(truth)
    if not completed:
        return {
            "status": "blocked", "selection_status": "not_rankable", "reason": "no_legal_fold_completed",
            "completed_folds": [], "blocked_folds": blocked, "metrics": None,
        }
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    metrics = binary_metrics(truth, prediction) if dataset.task_type == "binary" else regression_metrics(truth, prediction)
    return {
        "status": "completed" if not blocked else "partial",
        "selection_status": "not_rankable_protocol_mechanism_only",
        "split": (
            "held_well_logo_with_rolling_origins_and_30d_horizon_purge"
            if dataset.target_id in {"T3", "T4"} else "development_mother_well_logo"
        ),
        "completed_fold_count": len(completed), "expected_fold_count": len(dataset.development_groups),
        "scored_sample_count": len(truth), "metrics": metrics,
        "completed_folds": completed, "blocked_folds": blocked,
    }


def _evaluate_random(contract: Mapping[str, Any], dataset: R01Dataset) -> dict[str, Any]:
    indices = np.arange(len(dataset.sample_ids))
    stratify = dataset.targets.astype(int) if dataset.task_type == "binary" else None
    try:
        train, validation = train_test_split(
            indices, test_size=0.2, random_state=ROOT_SEED, shuffle=True, stratify=stratify,
        )
        prediction, preprocessing = _fit_predict(contract, dataset, train, validation)
    except (ValueError, RuntimeError) as exc:
        return {
            "status": "blocked", "selection_status": "invalid_for_selection",
            "rankability": "not_rankable", "reason": type(exc).__name__, "detail": str(exc),
        }
    truth = dataset.targets[validation]
    metrics = binary_metrics(truth, prediction) if dataset.task_type == "binary" else regression_metrics(truth, prediction)
    overlap = sorted(set(np.asarray(dataset.groups, dtype=object)[train]) & set(np.asarray(dataset.groups, dtype=object)[validation]))
    return {
        "status": "diagnostic_only", "selection_status": "invalid_for_selection",
        "rankability": "not_rankable", "split": "random_sample_80_20",
        "train_count": len(train), "validation_count": len(validation),
        "group_overlap": overlap, "metrics": metrics, "preprocessing": preprocessing,
    }


def _leakage_gap(contract: Mapping[str, Any], legal: Mapping[str, Any], random_lane: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(legal.get("metrics"), Mapping) or not isinstance(random_lane.get("metrics"), Mapping):
        return None
    gap: dict[str, Any] = {}
    for metric, direction in contract["metrics"]["directions"].items():
        legal_value = legal["metrics"].get(metric)
        random_value = random_lane["metrics"].get(metric)
        if legal_value is None or random_value is None:
            gap[metric] = None
        elif direction == "minimize":
            gap[metric] = float(legal_value - random_value)
        else:
            gap[metric] = float(random_value - legal_value)
    return {
        "definition": "positive means random-sample diagnostic appears more optimistic than legal group validation",
        "values": gap,
    }


def _blocked_result(contract: Mapping[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "target_id": contract["target_id"], "task_id": contract["task_id"],
        "head_name": contract["head_name"], "status": status,
        "rankability": "not_rankable", "reason": reason,
        "sample_count": 0, "groups": [],
        "coverage": {
            "sample_count": 0, "spatial_scale": contract["support"]["spatial_scale"],
            "time_coverage": contract["support"]["time_scale"],
        },
        "legal_lane": None, "random_leakage_diagnostic": None, "leakage_gap": None,
        "test_firewall": {
            "physical_test_h5_accessed": False, "known_holdout_accessed": False,
            "frozen_test_metrics_accessed": False, "fresh_blind_claimed": False,
        },
    }


def run_r01(source_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    random.seed(ROOT_SEED)
    np.random.seed(ROOT_SEED)
    contracts = load_contracts()
    contract_registry = {
        "schema_version": "sweetspot-p5.1-r0-contract-registry/v1", "root_seed": ROOT_SEED,
        "target_order": list(CONTRACT_ORDER),
        "targets": {target_id: {
            "contract_path": contract["contract_path"], "contract_sha256": contract["contract_sha256"],
            "task_id": contract["task_id"], "head_name": contract["head_name"],
            "status": contract["status"], "semantic_name": contract["semantic_name"],
            "truth_class": contract["truth_class"], "field_truth": contract["field_truth"],
            "units": contract["units"], "label": contract["label"],
            "support": contract["support"], "censoring": contract["censoring"],
            "split": contract["split"],
            "warnings": contract["warnings"],
        } for target_id, contract in contracts.items()},
        "independent_heads": True, "aggregate_sweetspot_score": None,
        "test_access": "forbidden",
    }
    datasets, data_audit = build_development_datasets(source_root)
    results: list[dict[str, Any]] = []
    for target_id in CONTRACT_ORDER:
        contract = contracts[target_id]
        if target_id == "T5":
            results.append(_blocked_result(contract, "not_feasible", "only simulation proxy; no approved field truth"))
            continue
        dataset = datasets.get(target_id)
        if dataset is None:
            results.append(_blocked_result(contract, "blocked", "development-only reconstruction unavailable"))
            continue
        legal = _evaluate_legal(contract, dataset)
        random_lane = _evaluate_random(contract, dataset)
        status = "r1_completed" if legal["status"] == "completed" else "blocked_or_partial"
        result = {
            "target_id": target_id, "task_id": contract["task_id"], "head_name": contract["head_name"],
            "status": status, "rankability": "not_rankable", "r1_scope": "protocol_mechanism_only",
            "model": contract["r1_lane"]["model"], "model_config": {
                "regression": {"estimator": "huber", "alpha": 0.0001, "epsilon": 1.35},
                "binary": {"C": 1.0, "class_weight": "balanced", "threshold": 0.5},
                "hpo": False,
            }[dataset.task_type],
            "sample_count": len(dataset.sample_ids), "groups": list(dataset.development_groups),
            "sample_sha256": dataset.sample_sha256, "feature_count": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names), "coverage": dataset.coverage,
            "provenance": dataset.provenance, "legal_lane": legal,
            "random_leakage_diagnostic": random_lane,
            "leakage_gap": _leakage_gap(contract, legal, random_lane),
            "proxy_warnings": contract["warnings"],
            "additional_lanes": (
                {"formal_failure_survival": "unapproved_blocked"} if target_id == "T4" else
                {"historical_multimodal_lane": "blocked_no_portable_development_seismic_features"}
                if target_id in {"T6", "T7"} else {}
            ),
            "test_firewall": {
                "physical_test_h5_accessed": False, "known_holdout_accessed": False,
                "frozen_test_metrics_accessed": False, "historical_test_metrics_used": False,
                "fresh_blind_claimed": False,
            },
        }
        results.append(result)
    boards = {
        result["target_id"]: {
            "head_name": result["head_name"], "status": "not_rankable_protocol_mechanism_only",
            "entries": [] if result["legal_lane"] is None else [{
                "model": result.get("model"), "metrics": result["legal_lane"].get("metrics"),
                "eligible_for_selection": False,
            }],
        }
        for result in results
    }
    summary = {
        "schema_version": "sweetspot-p5.1-r1-summary/v1", "root_seed": ROOT_SEED,
        "purpose": "R1 semantic and leakage mechanism validation; not final model ranking",
        "target_order": list(CONTRACT_ORDER), "target_boards": boards,
        "aggregate_sweetspot_board": None, "aggregate_sweetspot_score": None,
        "ten_model_fair_comparison": "deferred_to_R2",
        "target_status": {result["target_id"]: result["status"] for result in results},
        "test_firewall": {
            "physical_test_h5_accessed": False, "known_holdout_accessed": False,
            "frozen_test_metrics_accessed": False, "historical_test_metrics_used": False,
            "fresh_blind_claimed": False,
        },
    }
    return contract_registry, data_audit, results, summary


def write_outputs(output_dir: Path, payloads: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]], *, overwrite: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite R0/R1 artifacts: {[path.name for path in existing]}")
    contract_registry, data_audit, results, summary = payloads
    serialized = {
        "r0_contract_registry.json": _json_bytes(contract_registry),
        "r0_data_audit.json": _json_bytes(data_audit),
        "r1_results.jsonl": b"".join(_json_bytes(row, indent=None) for row in results),
    }
    results_sha256 = canonical_sha256(results)
    summary = {**summary, "results_canonical_sha256": results_sha256}
    serialized["r1_summary.json"] = _json_bytes(summary)
    for name, content in serialized.items():
        _atomic_write(output_dir / name, content)
    manifest = {
        "schema_version": "sweetspot-p5.1-r01-artifact-manifest/v1",
        "files": {name: {"sha256": sha256_file(output_dir / name), "size_bytes": (output_dir / name).stat().st_size} for name in sorted(serialized)},
        "input_hashes": {
            "contracts_sha256": canonical_sha256({target: item["contract_sha256"] for target, item in contract_registry["targets"].items()}),
            "source_manifest_sha256": data_audit["source_manifest_sha256"],
            "sample_sha256": data_audit["dataset_sample_sha256"],
            "split_manifest_sha256": data_audit["dataset_split_sha256"],
            "config_sha256": canonical_sha256({"root_seed": ROOT_SEED, "model": "fixed robust linear/logistic", "hpo": False}),
        },
        "test_firewall": summary["test_firewall"],
        "portable_paths": True,
    }
    _atomic_write(output_dir / "artifact_manifest.json", _json_bytes(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None, help="root containing approved raw Volve archives")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace only this R0/R1 portable output set")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    payloads = run_r01(args.source_root)
    manifest = write_outputs(args.output_dir, payloads, overwrite=args.overwrite)
    summary = payloads[3]
    print(json.dumps({
        "status": "completed", "elapsed_seconds": round(time.monotonic() - started, 3),
        "target_status": summary["target_status"], "artifact_count": len(manifest["files"]) + 1,
        "test_accessed": False, "final_ranking_published": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
