"""Leakage-safe Chronos-2 evaluation for sweetspot T3.

The run compares the archived P5 XGBoost leader with:

* B1: causal 30-day history mean;
* F0: source-locked Chronos-2 with past-only production covariates;
* F1: a one-parameter convex blend of F0 and B1, with the weight selected
  from fold-train labels only.

No known-holdout or frozen-test API exists in this module.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error

from _models.sweetspot.p7_chronos2 import (
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    PREDICTION_LENGTH,
    forecast_oil,
    forecast_water_risk_scores,
    load_pipeline,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "t3_chronos2_cv"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT
SOURCE_LOCK_PATH = HERE / "source_lock.v1.json"
P5_LEADERBOARD_DIR = (
    PROJECT_ROOT
    / "_pipelines"
    / "02_task_datasets"
    / "sweetspot"
    / "p5"
    / "_outputs"
    / "stage3_cv"
    / "leaderboards"
)
T3_FOLDS = (0, 1, 2, 3)
T4_FOLDS = (0, 1, 2)
WEIGHT_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 101))
ROOT_SEED = 2693

labels_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_labels"
)
data_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_data"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid prediction vector")
    rho = spearmanr(actual, prediction).statistic
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "spearman": float(rho),
    }


def history_mean(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (7, PREDICTION_LENGTH):
        raise ValueError(f"unexpected T3 sequence shape: {values.shape}")
    prediction = np.nanmean(values[:, 0, :], axis=1)
    if not np.isfinite(prediction).all():
        raise ValueError("history mean received an all-missing oil history")
    return prediction


def choose_convex_weight(
    foundation_prediction: np.ndarray,
    history_prediction: np.ndarray,
    train_target: np.ndarray,
    *,
    grid: Sequence[float] = WEIGHT_GRID,
) -> tuple[float, float]:
    """Select one blend weight using fold-train labels only."""
    foundation = np.asarray(foundation_prediction, dtype=np.float64).reshape(-1)
    history = np.asarray(history_prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if not (foundation.shape == history.shape == target.shape):
        raise ValueError("train-only blend arrays must have identical shapes")
    candidates = np.asarray(tuple(grid), dtype=np.float64)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("blend grid is empty")
    if np.any((candidates < 0.0) | (candidates > 1.0)):
        raise ValueError("blend weights must stay in [0, 1]")
    errors = np.asarray(
        [
            mean_absolute_error(
                target,
                weight * foundation + (1.0 - weight) * history,
            )
            for weight in candidates
        ],
        dtype=np.float64,
    )
    selected = int(np.argmin(errors))
    return float(candidates[selected]), float(errors[selected])


def _archived_baseline(target_id: str) -> dict[str, Any]:
    leaderboard_path = P5_LEADERBOARD_DIR / f"{target_id}.json"
    payload = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    eligible = [
        row
        for row in payload["entries"]
        if row.get("eligible_for_ranking") and row.get("rank") == 1
    ]
    if len(eligible) != 1:
        raise ValueError(f"archived {target_id} leaderboard has no unique rank-1 baseline")
    row = eligible[0]
    return {
        "model_id": row["model_id"],
        "primary_metric": payload["primary_metric"],
        "primary_mean": float(row["primary_mean"]),
        "fold_means": {key: float(value) for key, value in row["fold_means"].items()},
        "leaderboard_path": str(leaderboard_path.relative_to(PROJECT_ROOT)),
        "leaderboard_sha256": _sha256_file(leaderboard_path),
        "known_holdout_used_for_selection": False,
    }


def _validate_source_lock() -> dict[str, Any]:
    lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"source lock mismatch for {key}")
    return lock


def _resolve_snapshot(*, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
            allow_patterns=("config.json", "model.safetensors"),
        )
    ).resolve()


def _validate_snapshot(snapshot: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    config = snapshot / "config.json"
    weights = snapshot / "model.safetensors"
    observed = {
        "config_sha256": _sha256_file(config),
        "weights_sha256": _sha256_file(weights),
        "weights_size_bytes": int(weights.stat().st_size),
    }
    for key, value in observed.items():
        if lock.get(key) != value:
            raise ValueError(f"source-locked model artifact mismatch for {key}")
    return observed


@contextmanager
def gpu_lock(path: Path | None, *, device: str) -> Iterator[dict[str, Any]]:
    if device != "cuda":
        yield {"mechanism": None, "path": None, "device": "cpu"}
        return
    if path is None:
        raise ValueError("device=cuda requires --gpu-lock-path")
    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield {"mechanism": "flock", "path": lock_path.name, "device": "cuda:0"}
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _aggregate(method_id: str, folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("mae", "rmse", "spearman")
    return {
        "method_id": method_id,
        "fold_count": len(folds),
        "folds": list(folds),
        "macro_fold_mean": {
            metric: float(np.mean([row["metrics"][metric] for row in folds]))
            for metric in metrics
        },
    }


def run(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
    gpu_lock_path: Path | None = None,
    local_files_only: bool = False,
    batch_size: int = 196,
) -> dict[str, Any]:
    start = time.perf_counter()
    source_lock = _validate_source_lock()
    snapshot = _resolve_snapshot(local_files_only=local_files_only)
    artifact_observation = _validate_snapshot(snapshot, source_lock)
    audit = labels_module.validate_label_mapping()
    split = audit.split_manifest("T3")
    observed_folds = tuple(int(row["fold_id"]) for row in split["folds"])
    if observed_folds != T3_FOLDS:
        raise ValueError(f"T3 frozen fold contract changed: {observed_folds}")
    t4_split = audit.split_manifest("T4")
    observed_t4_folds = tuple(int(row["fold_id"]) for row in t4_split["folds"])
    if observed_t4_folds != T4_FOLDS:
        raise ValueError(f"T4 frozen fold contract changed: {observed_t4_folds}")

    with gpu_lock(gpu_lock_path, device=device) as lock_evidence:
        pipeline = load_pipeline(snapshot, device=device)
        fold_rows: dict[str, list[dict[str, Any]]] = {
            "B1_history_mean": [],
            "F0_chronos2_past_covariates": [],
            "F1_chronos2_train_blend": [],
        }
        weights: list[dict[str, Any]] = []
        for fold_id in T3_FOLDS:
            data = data_module.load_development_pilot_data(
                audit,
                "T3",
                source_root=Path(source_root),
                fold_id=fold_id,
            )
            train_daily, train_foundation = forecast_oil(
                pipeline,
                data.train_sequence,
                mode="past_covariates",
                batch_size=batch_size,
            )
            validation_daily, validation_foundation = forecast_oil(
                pipeline,
                data.validation_sequence,
                mode="past_covariates",
                batch_size=batch_size,
            )
            train_history = history_mean(data.train_sequence)
            validation_history = history_mean(data.validation_sequence)
            weight, train_mae = choose_convex_weight(
                train_foundation,
                train_history,
                data.train_target,
            )
            validation_blend = (
                weight * validation_foundation
                + (1.0 - weight) * validation_history
            )
            common = {
                "fold_id": int(fold_id),
                "train_samples": len(data.train_sample_ids),
                "validation_samples": len(data.validation_sample_ids),
                "validation_groups": sorted(set(data.validation_groups)),
                "train_sample_ids_sha256": _canonical_sha256(list(data.train_sample_ids)),
                "validation_sample_ids_sha256": _canonical_sha256(list(data.validation_sample_ids)),
                "input_budget_sha256": data.input_budget_sha256,
                "split_sha256": data.split_sha256,
            }
            fold_rows["B1_history_mean"].append(
                {**common, "metrics": _metrics(data.validation_target, validation_history)}
            )
            fold_rows["F0_chronos2_past_covariates"].append(
                {
                    **common,
                    "metrics": _metrics(data.validation_target, validation_foundation),
                    "daily_forecast_sha256": hashlib.sha256(
                        validation_daily.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )
            fold_rows["F1_chronos2_train_blend"].append(
                {
                    **common,
                    "metrics": _metrics(data.validation_target, validation_blend),
                    "prediction_sha256": hashlib.sha256(
                        validation_blend.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )
            weights.append(
                {
                    "fold_id": int(fold_id),
                    "chronos_weight": weight,
                    "history_weight": 1.0 - weight,
                    "train_selection_metric": "mae",
                    "train_selection_mae": train_mae,
                    "grid_step": 0.01,
                    "validation_labels_used_for_weight_selection": False,
                }
            )
            del train_daily
        t4_rows: list[dict[str, Any]] = []
        for fold_id in T4_FOLDS:
            data = data_module.load_development_pilot_data(
                audit,
                "T4",
                source_root=Path(source_root),
                fold_id=fold_id,
            )
            train_scores, quantiles = forecast_water_risk_scores(
                pipeline,
                data.train_sequence,
                batch_size=min(batch_size, 128),
            )
            validation_scores, validation_quantiles = forecast_water_risk_scores(
                pipeline,
                data.validation_sequence,
                batch_size=min(batch_size, 128),
            )
            if not np.array_equal(quantiles, validation_quantiles):
                raise ValueError("T4 Chronos quantile grid changed within a fold")
            train_ap = np.asarray(
                [
                    average_precision_score(data.train_target, train_scores[:, index])
                    for index in range(train_scores.shape[1])
                ],
                dtype=np.float64,
            )
            quantile_index = int(np.argmax(train_ap))
            validation_score = validation_scores[:, quantile_index]
            t4_rows.append(
                {
                    "fold_id": int(fold_id),
                    "train_samples": len(data.train_sample_ids),
                    "validation_samples": len(data.validation_sample_ids),
                    "validation_groups": sorted(set(data.validation_groups)),
                    "selected_quantile": float(quantiles[quantile_index]),
                    "train_selection_average_precision": float(train_ap[quantile_index]),
                    "validation_average_precision": float(
                        average_precision_score(data.validation_target, validation_score)
                    ),
                    "validation_labels_used_for_quantile_selection": False,
                    "input_budget_sha256": data.input_budget_sha256,
                    "split_sha256": data.split_sha256,
                    "validation_score_sha256": hashlib.sha256(
                        validation_score.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )

    methods = {
        method_id: _aggregate(method_id, rows)
        for method_id, rows in fold_rows.items()
    }
    archived = _archived_baseline("T3")
    t4_archived = _archived_baseline("T4")
    selected = methods["F1_chronos2_train_blend"]["macro_fold_mean"]["mae"]
    naive = methods["B1_history_mean"]["macro_fold_mean"]["mae"]
    baseline = archived["primary_mean"]
    improvement = 100.0 * (baseline - selected) / baseline
    naive_improvement = 100.0 * (naive - selected) / naive
    promote = selected < baseline and selected < naive
    t4_macro_ap = float(
        np.mean([row["validation_average_precision"] for row in t4_rows])
    )
    t4_promote = t4_macro_ap > float(t4_archived["primary_mean"])
    summary = {
        "schema_version": "sweetspot-p7-chronos2-t3-cv/v1",
        "target": {
            "target_id": "T3",
            "lane": "productivity",
            "definition": "mean BORE_OIL_VOL over the next 30 calendar days",
            "history_days": 30,
            "forecast_days": 30,
            "primary_metric": "mae",
            "direction": "minimize",
        },
        "foundation": {
            "family": "Gaia time-series foundation lane",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "real_pretrained_weights_loaded": True,
            "input_mode": "oil target plus six past-only production covariates",
            "future_covariates_used": False,
            "cross_learning": False,
            "point_forecast": "mean of non-negative daily median forecasts",
            "artifact": artifact_observation,
            "source_lock_path": str(SOURCE_LOCK_PATH.relative_to(PROJECT_ROOT)),
            "source_lock_sha256": _sha256_file(SOURCE_LOCK_PATH),
        },
        "evaluation": {
            "folds": list(T3_FOLDS),
            "t4_folds": list(T4_FOLDS),
            "split_policy": "existing P4 held-well rolling-origin development folds",
            "train_sample_limit": data_module.TRAIN_SAMPLE_LIMIT,
            "validation_sample_limit": data_module.VALIDATION_SAMPLE_LIMIT,
            "known_holdout_accessed": False,
            "frozen_test_accessed": False,
            "historical_holdout_metrics_used_for_selection": False,
            "validation_labels_used_for_blend_selection": False,
            "selection_scope": "fold-train labels choose one convex blend weight; fold-validation reports metrics",
        },
        "methods": methods,
        "blend_weights": weights,
        "archived_p5_baseline": archived,
        "t4_experiment": {
            "target_id": "T4",
            "method_id": "F0_chronos2_future_water_risk",
            "history_days": 7,
            "forecast_days": 30,
            "train_selection": "choose forecast quantile by fold-train average precision",
            "risk_score": "largest non-negative seven-day mean in the 30-day water forecast",
            "folds": t4_rows,
            "macro_fold_average_precision": t4_macro_ap,
            "archived_p5_baseline": t4_archived,
            "promotion_status": "PROMOTE" if t4_promote else "REJECT",
            "reason": (
                "foundation risk score beats archived CatBoost"
                if t4_promote
                else "seven-day context foundation score does not beat archived CatBoost"
            ),
            "known_holdout_accessed": False,
            "frozen_test_accessed": False,
        },
        "decision": {
            "selected_method": "F1_chronos2_train_blend",
            "selected_macro_fold_mae": selected,
            "archived_xgboost_macro_fold_mae": baseline,
            "causal_history_mean_macro_fold_mae": naive,
            "mae_reduction_vs_archived_xgboost_percent": improvement,
            "mae_reduction_vs_history_mean_percent": naive_improvement,
            "promotion_status": "PROMOTE" if promote else "REJECT",
            "promotion_rule": "selected MAE must beat both archived XGBoost and causal history-mean control",
            "t4_status": "PROMOTE" if t4_promote else "REJECTED_NO_GAIN",
            "t4_chronos_macro_fold_average_precision": t4_macro_ap,
            "t4_catboost_macro_fold_average_precision": float(t4_archived["primary_mean"]),
            "non_temporal_tracks_status": "FOUNDATION_NOT_APPLICABLE",
        },
        "resource": {
            "wall_seconds": float(time.perf_counter() - start),
            "device": device,
            "gpu_lock": lock_evidence,
            "batch_size": int(batch_size),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "chronos_forecasting": importlib.metadata.version("chronos-forecasting"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runtime_boundary": {
            "external_api_calls": False,
            "foundation_download_allowed": not local_files_only,
            "raw_predictions_persisted": False,
            "checkpoint_written": False,
            "root_seed": ROOT_SEED,
        },
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=196)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(
        source_root=args.source_root,
        output_dir=args.output_dir,
        device=args.device,
        gpu_lock_path=args.gpu_lock_path,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
