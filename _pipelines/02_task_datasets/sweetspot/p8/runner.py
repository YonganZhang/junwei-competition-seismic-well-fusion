"""Exact-calendar Chronos-2 development diagnostic.

This run intentionally does not reuse the archived P5 XGBoost score because
the daily-grid sample universe differs from the old observation-index sample
universe.  It compares Chronos and a train-only blend against the causal
history mean, archives no row-level prediction, and cannot promote the model.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from _models.sweetspot.p7_chronos2 import (
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    forecast_oil_calendar,
    load_pipeline,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "t3_chronos2_calendar_cv"
SOURCE_LOCK_PATH = HERE / "source_lock.v1.json"
FOLDS = (0, 1, 2, 3)
WEIGHT_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 101))

labels_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_labels"
)
calendar_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p8.calendar_data"
)
p7_runner = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p7.runner"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid calendar prediction vector")
    rho = float(spearmanr(actual, prediction).statistic)
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "spearman": rho,
    }


def _history_mean(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (7, 30):
        raise ValueError(f"unexpected calendar sequence shape: {values.shape}")
    prediction = np.nanmean(values[:, 0, :], axis=1)
    if not np.isfinite(prediction).all():
        raise ValueError("calendar history mean received an all-missing oil history")
    return prediction


def _calendar_features(
    sequences: np.ndarray,
    *,
    train_medians: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (7, 30):
        raise ValueError(f"unexpected calendar sequence shape: {values.shape}")
    flat = values.reshape(len(values), -1)
    missing = ~np.isfinite(flat)
    if train_medians is None:
        medians = np.nanmedian(flat, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    else:
        medians = np.asarray(train_medians, dtype=np.float64)
        if medians.shape != (flat.shape[1],) or not np.isfinite(medians).all():
            raise ValueError("calendar tree medians must be finite and train-fitted")
    filled = np.where(missing, medians[None, :], flat)
    observed = (~missing).sum(axis=1, keepdims=True).astype(np.float64)
    features = np.concatenate((filled, missing.astype(np.float64), observed), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("calendar tree features are non-finite")
    return features, medians


def _fit_calendar_tree(
    train_sequence: np.ndarray,
    train_target: np.ndarray,
    validation_sequence: np.ndarray,
    *,
    seed: int,
    shuffle_target: bool = False,
) -> np.ndarray:
    train_x, medians = _calendar_features(train_sequence)
    validation_x, _ = _calendar_features(
        validation_sequence, train_medians=medians
    )
    target = np.asarray(train_target, dtype=np.float64).copy()
    if shuffle_target:
        target = np.random.default_rng(seed).permutation(target)
    estimator = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=seed,
    )
    estimator.fit(train_x, target)
    prediction = np.asarray(estimator.predict(validation_x), dtype=np.float64)
    if prediction.shape != (len(validation_x),) or not np.isfinite(prediction).all():
        raise ValueError("calendar tree produced invalid predictions")
    return prediction


def _shuffle_history_order(
    sequences: np.ndarray,
    *,
    fold_id: int,
) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float32).copy()
    rng = np.random.default_rng(2693 + int(fold_id))
    for sample in range(len(values)):
        order = rng.permutation(values.shape[-1])
        values[sample] = values[sample][:, order]
    return values


def _choose_weight(
    foundation: np.ndarray,
    history: np.ndarray,
    target: np.ndarray,
    grid: Sequence[float] = WEIGHT_GRID,
) -> tuple[float, float]:
    return p7_runner.choose_convex_weight(
        foundation, history, target, grid=grid
    )


def _validate_source_lock() -> Mapping[str, Any]:
    payload = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "source_revision": "75a2dfe228e39d55283f477c01399b9e53d89661",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"calendar Chronos source lock mismatch for {key}")
    return payload


def _resolve_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            allow_patterns=("config.json", "model.safetensors"),
        )
    ).resolve()


@contextmanager
def gpu_lock(path: Path | None, *, device: str) -> Iterator[Mapping[str, Any]]:
    if not device.startswith("cuda"):
        yield {"mechanism": None, "device": device}
        return
    if path is None:
        raise ValueError("CUDA calendar evaluation requires --gpu-lock-path")
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield {"mechanism": "flock", "device": device, "lock_name": resolved.name}
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        metric: float(np.mean([row["metrics"][metric] for row in rows]))
        for metric in ("mae", "rmse", "spearman")
    }


def run(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
    gpu_lock_path: Path | None = None,
    batch_size: int = 196,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_lock = _validate_source_lock()
    snapshot = _resolve_snapshot()
    artifact = p7_runner._validate_snapshot(snapshot, source_lock)
    audit = labels_module.validate_label_mapping()
    rows: dict[str, list[dict[str, Any]]] = {
        "B1_calendar_history_mean": [],
        "B2_calendar_extra_trees": [],
        "C0_calendar_extra_trees_target_shuffle": [],
        "F0_chronos2_calendar": [],
        "C1_chronos2_history_order_shuffle": [],
        "F1_chronos2_train_blend_calendar": [],
    }
    weights: list[dict[str, Any]] = []
    with gpu_lock(gpu_lock_path, device=device) as lock_evidence:
        pipeline = load_pipeline(snapshot, device=device)
        for fold_id in FOLDS:
            data = calendar_module.load_calendar_pilot_data(
                audit,
                source_root=Path(source_root),
                fold_id=fold_id,
            )
            train_daily, train_foundation = forecast_oil_calendar(
                pipeline,
                data.train_sequence,
                data.train_timestamps,
                data.train_sample_ids,
                batch_size=batch_size,
            )
            validation_daily, validation_foundation = forecast_oil_calendar(
                pipeline,
                data.validation_sequence,
                data.validation_timestamps,
                data.validation_sample_ids,
                batch_size=batch_size,
            )
            _, validation_foundation_shuffled = forecast_oil_calendar(
                pipeline,
                _shuffle_history_order(
                    data.validation_sequence, fold_id=fold_id
                ),
                data.validation_timestamps,
                data.validation_sample_ids,
                batch_size=batch_size,
            )
            train_history = _history_mean(data.train_sequence)
            validation_history = _history_mean(data.validation_sequence)
            validation_tree = _fit_calendar_tree(
                data.train_sequence,
                data.train_target,
                data.validation_sequence,
                seed=2693 + fold_id,
            )
            validation_tree_control = _fit_calendar_tree(
                data.train_sequence,
                data.train_target,
                data.validation_sequence,
                seed=2693 + fold_id,
                shuffle_target=True,
            )
            weight, train_selection_mae = _choose_weight(
                train_foundation, train_history, data.train_target
            )
            validation_blend = (
                weight * validation_foundation
                + (1.0 - weight) * validation_history
            )
            common = {
                "fold_id": fold_id,
                "train_samples": len(data.train_sample_ids),
                "validation_samples": len(data.validation_sample_ids),
                "train_groups": sorted(set(data.train_groups)),
                "validation_groups": sorted(set(data.validation_groups)),
                "train_sample_ids_sha256": _canonical_sha256(
                    list(data.train_sample_ids)
                ),
                "validation_sample_ids_sha256": _canonical_sha256(
                    list(data.validation_sample_ids)
                ),
                "split_sha256": data.split_sha256,
                "input_budget_sha256": data.input_budget_sha256,
            }
            rows["B1_calendar_history_mean"].append(
                {**common, "metrics": _metrics(data.validation_target, validation_history)}
            )
            rows["B2_calendar_extra_trees"].append(
                {**common, "metrics": _metrics(data.validation_target, validation_tree)}
            )
            rows["C0_calendar_extra_trees_target_shuffle"].append(
                {
                    **common,
                    "metrics": _metrics(
                        data.validation_target, validation_tree_control
                    ),
                    "control": "train_targets_permuted_with_fixed_seed",
                }
            )
            rows["F0_chronos2_calendar"].append(
                {
                    **common,
                    "metrics": _metrics(data.validation_target, validation_foundation),
                    "daily_forecast_sha256": hashlib.sha256(
                        validation_daily.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )
            rows["C1_chronos2_history_order_shuffle"].append(
                {
                    **common,
                    "metrics": _metrics(
                        data.validation_target, validation_foundation_shuffled
                    ),
                    "control": (
                        "validation_history_days_permuted_per_sample_while_"
                        "timestamps_and_values_are_preserved"
                    ),
                }
            )
            rows["F1_chronos2_train_blend_calendar"].append(
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
                    "fold_id": fold_id,
                    "chronos_weight": weight,
                    "history_weight": 1.0 - weight,
                    "train_selection_mae": train_selection_mae,
                    "validation_labels_used_for_weight_selection": False,
                }
            )
            del train_daily
    methods = {
        method: {
            "fold_count": len(fold_rows),
            "folds": fold_rows,
            "macro_fold_mean": _aggregate(fold_rows),
        }
        for method, fold_rows in rows.items()
    }
    foundation_mae = methods["F0_chronos2_calendar"]["macro_fold_mean"]["mae"]
    history_mae = methods["B1_calendar_history_mean"]["macro_fold_mean"]["mae"]
    tree_mae = methods["B2_calendar_extra_trees"]["macro_fold_mean"]["mae"]
    tree_control_mae = methods[
        "C0_calendar_extra_trees_target_shuffle"
    ]["macro_fold_mean"]["mae"]
    foundation_control_mae = methods[
        "C1_chronos2_history_order_shuffle"
    ]["macro_fold_mean"]["mae"]
    effect_supported = bool(
        foundation_mae < history_mae
        and foundation_mae < tree_mae
        and foundation_mae < tree_control_mae
        and foundation_mae < foundation_control_mae
    )
    summary = {
        "schema_version": "sweetspot-p8-chronos-calendar/v1",
        "foundation": {
            "model_id": MODEL_ID,
            "weights_revision": MODEL_REVISION,
            "source_revision": source_lock["source_revision"],
            "license": MODEL_LICENSE,
            "artifact": artifact,
            "real_pretrained_weights_loaded": True,
            "source_lock_path": str(SOURCE_LOCK_PATH.relative_to(PROJECT_ROOT)),
            "source_lock_sha256": _sha256_file(SOURCE_LOCK_PATH),
        },
        "calendar_contract": {
            "frequency": "D",
            "history_days": 30,
            "forecast_days": 30,
            "missing_dates_reindexed_not_zero_filled": True,
            "legacy_p7_observation_index_evidence_promoted": False,
        },
        "evaluation": {
            "folds": list(FOLDS),
            "known_holdout_accessed": False,
            "frozen_test_accessed": False,
            "validation_labels_used_for_blend_selection": False,
        },
        "methods": methods,
        "train_only_blend_weights": weights,
        "decision": {
            "state": (
                "EFFECT_SUPPORTED_NOT_PROMOTED"
                if effect_supported
                else "CONNECTED_NO_PROMOTION"
            ),
            "default_enabled": False,
            "effect_supported": effect_supported,
            "reason": (
                "Chronos beats the causal mean, same-grid ExtraTrees, target-shuffle, "
                "and history-order-shuffle controls on the locked development folds; "
                "default promotion still requires a same-architecture random-init control."
                if effect_supported
                else "The routed foundation model did not clear every registered control."
            ),
        },
        "runtime": {
            "device": device,
            "gpu_lock": dict(lock_evidence),
            "duration_seconds": time.perf_counter() - started,
            "raw_predictions_persisted": False,
            "checkpoint_written": False,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=196)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run(**vars(args))
    print(json.dumps(summary["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
