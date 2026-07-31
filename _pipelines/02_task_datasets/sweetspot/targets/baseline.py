"""Train targets 1--4 with development-only HPO and one frozen-test read.

The runner deliberately keeps the four cases independent.  Every estimator is
loaded from the canonical ``_models/sweetspot`` registry, every preprocessing
pipeline is fitted inside a development fold, and the frozen test rows are
released exactly once after configuration freeze and development refit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, r2_score

from _code.ml_framework.artifacts import ArtifactManifest, atomic_write_json, hash_file, hash_payload
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint
from _code.ml_framework.hpo import HPOPlan, rank_trials, run_fixed_trials
from _code.ml_framework.lifecycle import ExperimentLifecycle, ExperimentState
from _code.ml_framework.model_discovery import discover_model
from _code.ml_framework.run_layout import create_run_layout
from _code.ml_framework.seeding import DEFAULT_ROOT_SEED, derive_seed, seed_everything

from . import visualize


TARGET_IDS = (
    "reservoir_quality",
    "hydrocarbon_pay",
    "productivity",
    "water_breakthrough",
)
BASE = "_pipelines.02_task_datasets.sweetspot.targets"
CASE_NAME = "baseline_v1"


@dataclass(frozen=True)
class TargetRuntime:
    target_id: str
    target_key: str
    model_id: str
    problem: str
    metric_name: str
    metric_direction: str
    transformed_target: bool


RUNTIMES = {
    "reservoir_quality": TargetRuntime(
        "reservoir_quality", "target", "robust_linear", "regression", "mae", "minimize", True,
    ),
    "hydrocarbon_pay": TargetRuntime(
        "hydrocarbon_pay", "target", "logistic_classifier", "binary", "average_precision", "maximize", False,
    ),
    "productivity": TargetRuntime(
        "productivity", "target_future_30d_mean_oil_sm3_day", "robust_linear",
        "regression", "mae", "minimize", True,
    ),
    "water_breakthrough": TargetRuntime(
        "water_breakthrough", "event_within_30d", "logistic_classifier",
        "binary", "average_precision", "maximize", False,
    ),
}


class FrozenTestGate:
    """Release predeclared test row indices once and only once."""

    def __init__(self, sample_ids: Sequence[str], frozen_ids: Sequence[str]) -> None:
        index = {sample_id: row for row, sample_id in enumerate(sample_ids)}
        missing = sorted(set(frozen_ids) - set(index))
        if missing:
            raise ValueError(f"frozen test IDs are absent from dataset: {missing[:3]}")
        self._indices = np.asarray([index[sample_id] for sample_id in frozen_ids], dtype=int)
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def consume(self) -> np.ndarray:
        if self._consumed:
            raise RuntimeError("frozen test rows have already been consumed")
        self._consumed = True
        return self._indices.copy()


def _module(target_id: str) -> Any:
    if target_id not in TARGET_IDS:
        raise ValueError(f"target_id must be one of {TARGET_IDS}")
    return importlib.import_module(f"{BASE}.{target_id}.contract")


def _target_values(runtime: TargetRuntime, dataset: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(dataset[runtime.target_key], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{runtime.target_id}: target contains non-finite values")
    if runtime.problem == "binary" and set(np.unique(values)) != {0.0, 1.0}:
        raise ValueError(f"{runtime.target_id}: binary target must contain both 0 and 1")
    if runtime.transformed_target and np.any(values < 0):
        raise ValueError(f"{runtime.target_id}: log1p target contains negative values")
    return values


def _fit_values(runtime: TargetRuntime, values: np.ndarray) -> np.ndarray:
    return np.log1p(values) if runtime.transformed_target else values


def _physical_values(runtime: TargetRuntime, values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(values)) if runtime.transformed_target else values


def _candidate_configs(runtime: TargetRuntime) -> list[dict[str, Any]]:
    if runtime.problem == "binary":
        return [
            {"c": c, "class_weight": class_weight, "max_iter": 500}
            for class_weight in ("balanced", None)
            for c in (0.01, 0.1, 1.0, 10.0)
        ]
    return [
        {"estimator": "ridge", "alpha": 1e-6},
        {"estimator": "ridge", "alpha": 1e-4},
        {"estimator": "ridge", "alpha": 1e-2},
        {"estimator": "ridge", "alpha": 1.0},
        {"estimator": "ridge", "alpha": 10.0},
        {"estimator": "huber", "alpha": 1e-6, "epsilon": 1.20},
        {"estimator": "huber", "alpha": 1e-4, "epsilon": 1.35},
        {"estimator": "huber", "alpha": 1e-2, "epsilon": 1.60},
    ]


def _pilot_configs(runtime: TargetRuntime) -> list[dict[str, Any]]:
    """Return the preregistered 20-trial pilot grid for simple models."""
    if runtime.problem == "binary":
        return [
            {"c": float(c), "class_weight": class_weight, "max_iter": 750}
            for class_weight in ("balanced", None)
            for c in np.logspace(-3.0, 2.0, 10)
        ]
    ridge = [
        {"estimator": "ridge", "alpha": float(alpha)}
        for alpha in np.logspace(-5.5, 1.5, 10)
    ]
    huber = [
        {"estimator": "huber", "alpha": alpha, "epsilon": epsilon}
        for alpha in (1e-5, 1e-3)
        for epsilon in (1.10, 1.25, 1.40, 1.60, 1.90)
    ]
    return ridge + huber


def _stable_subsample(indices: np.ndarray, sample_ids: Sequence[str], seed: int, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    ranked = sorted(
        indices.tolist(),
        key=lambda row: hashlib.sha256(f"{seed}\0{sample_ids[row]}".encode("utf-8")).hexdigest(),
    )
    return np.asarray(ranked[:limit], dtype=int)


def _indices(sample_ids: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    by_id = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise ValueError(f"split contains unknown sample IDs: {missing[:3]}")
    return np.asarray([by_id[sample_id] for sample_id in selected], dtype=int)


def _build_and_fit(spec: Any, runtime: TargetRuntime, config: Mapping[str, Any], x: np.ndarray, y: np.ndarray) -> Any:
    model = discover_model("sweetspot", runtime.model_id).build(spec, **dict(config))
    target = spec.targets[0]
    model.fit(x, {target: y}, {target: np.ones(len(y), dtype=bool)})
    return model


def _predict(model: Any, spec: Any, runtime: TargetRuntime, x: np.ndarray) -> np.ndarray:
    output = model.predict(x)
    target = spec.targets[0]
    if runtime.problem == "binary":
        if output.transformed is None:
            raise RuntimeError("binary model did not return probabilities")
        prediction = np.asarray(output.transformed[target], dtype=float)
    else:
        prediction = _physical_values(runtime, np.asarray(output.raw[target], dtype=float))
    if prediction.shape != (len(x),) or not np.all(np.isfinite(prediction)):
        raise ValueError("model prediction is non-finite or has the wrong shape")
    return prediction


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residual = predicted - observed
    correlation = spearmanr(observed, predicted).statistic
    constant = bool(np.allclose(observed, observed[0]))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": None if constant else float(r2_score(observed, predicted)),
        "r2_reason": "constant observed target" if constant else None,
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "sample_count": int(len(observed)),
    }


def _best_threshold(observed: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 17)
    scored = [(float(f1_score(observed, probability >= value)), float(value)) for value in candidates]
    return max(scored, key=lambda item: (item[0], -item[1]))[1]


def _binary_metrics(observed: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "average_precision": float(average_precision_score(observed, probability)),
        "brier": float(brier_score_loss(observed, probability)),
        "f1": float(f1_score(observed, probability >= threshold)),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(observed)),
        "sample_count": int(len(observed)),
    }


def _prediction_frame(
    target_id: str,
    dataset: Mapping[str, Any],
    indices: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "sample_id": np.asarray(dataset["sample_ids"], dtype=object)[indices],
        "well": np.asarray(dataset.get("wellbores", dataset["groups"]), dtype=object)[indices],
        "observed": observed,
    })
    if target_id in {"reservoir_quality", "hydrocarbon_pay"}:
        frame["depth_m"] = np.asarray(dataset["depth_m"], dtype=float)[indices]
    else:
        frame["cutoff_date"] = np.asarray(dataset["cutoff_dates"], dtype=object)[indices]
    frame["probability" if target_id in {"hydrocarbon_pay", "water_breakthrough"} else "prediction"] = prediction
    return frame


def _extra_metrics(target_id: str, frame: pd.DataFrame, metrics: dict[str, Any], threshold: float) -> None:
    if target_id == "hydrocarbon_pay":
        errors = []
        for _, part in frame.groupby("well"):
            depths = np.sort(part["depth_m"].to_numpy(float))
            spacing = float(np.median(np.diff(depths))) if len(depths) > 1 else 0.0
            observed = spacing * float(part["observed"].sum())
            predicted = spacing * float((part["probability"] >= threshold).sum())
            errors.append(abs(predicted - observed))
        metrics["net_thickness_mae_m"] = float(np.mean(errors))
    if target_id == "productivity":
        count = max(1, int(math.ceil(len(frame) * 0.10)))
        truth = set(frame.nlargest(count, "observed")["sample_id"])
        predicted = set(frame.nlargest(count, "prediction")["sample_id"])
        metrics["topk_hit"] = float(len(truth & predicted) / count)


def _trainer_state() -> dict[str, Any]:
    return {
        "next_epoch": 1,
        "global_step": 1,
        "best_epoch": 0,
        "best_val_loss": 0.0,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "history": [{"epoch": 0, "kind": "closed_form_or_converged_sklearn_fit"}],
    }


def _save_model_checkpoint(
    path: Path,
    model: Any,
    *,
    config_hash: str,
    split_hash: str,
    seed_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    return save_checkpoint(
        path,
        epoch=0,
        model_state={"pipeline": model.pipeline, "fitted": True},
        optimizer_state={"name": "sklearn_internal", "state": "converged"},
        scheduler_state={"name": "none", "last_epoch": 0},
        scaler_state={"enabled": False},
        config_hash=config_hash,
        split_hash=split_hash,
        trainer_state=_trainer_state(),
        seed_report=seed_report,
        environment={"python": platform.python_version(), "numpy": np.__version__},
        extra={"config": dict(config)},
        include_torch_rng=False,
    )


def _restore_checkpoint(path: Path, spec: Any, runtime: TargetRuntime) -> tuple[Any, Mapping[str, Any]]:
    payload = load_checkpoint(path)
    config = payload["extra"]["config"]
    model = discover_model("sweetspot", runtime.model_id).build(spec, **config["model_config"])
    model.pipeline = payload["model_state"]["pipeline"]
    model._fitted = bool(payload["model_state"]["fitted"])
    return model, payload


def _write_manifest(run_root: Path, run_id: str) -> None:
    manifest = ArtifactManifest(run_id=run_id, root=run_root)
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        role = "checkpoint" if path.suffix == ".pkl" else "figure" if path.suffix == ".png" else "evidence"
        manifest.register(path.relative_to(run_root).as_posix(), role=role)
    manifest.write()
    manifest.verify()


def run_target(
    target_id: str,
    output_dir: Path,
    *,
    source_root: Path | None = None,
    root_seed: int = DEFAULT_ROOT_SEED,
    sanity_train_limit: int = 4000,
    run_pilot: bool = True,
) -> dict[str, Any]:
    """Execute one real-data target.  ``output_dir`` must be new or empty."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    runtime = RUNTIMES[target_id]
    contract = _module(target_id)
    spec = contract.task_spec()
    dataset, split, data_evidence = contract.build_dataset_and_manifest(source_root)
    sample_ids = list(dataset["sample_ids"])
    x = np.asarray(dataset["features"], dtype=float)
    physical_y = _target_values(runtime, dataset)
    fit_y = _fit_values(runtime, physical_y)
    if x.ndim != 2 or x.shape[0] != len(sample_ids):
        raise ValueError("features must be a sample-aligned matrix")
    run_root = create_run_layout(output_dir)
    split_hash = split.stable_hash()
    seed_report = seed_everything(root_seed, include_torch=False).to_dict()
    lifecycle = ExperimentLifecycle(f"sweetspot.{target_id}.{CASE_NAME}")
    atomic_write_json(run_root / "task_spec.json", spec.to_dict())
    atomic_write_json(run_root / "split_manifest.json", split.to_dict())
    atomic_write_json(run_root / "data_evidence.json", data_evidence)
    lifecycle.advance(ExperimentState.SPLIT_LOCKED, {"split_hash": split_hash})

    smoke_fold = split.folds[0]
    smoke_rows = _stable_subsample(
        _indices(sample_ids, smoke_fold.train_sample_ids), sample_ids,
        derive_seed(root_seed, "model", target_id, "smoke"), min(256, sanity_train_limit),
    )
    smoke_model = _build_and_fit(spec, runtime, _candidate_configs(runtime)[0], x[smoke_rows], fit_y[smoke_rows])
    smoke_prediction = _predict(smoke_model, spec, runtime, x[smoke_rows[: min(32, len(smoke_rows))]])
    lifecycle.advance(ExperimentState.SMOKE_PASSED, {
        "sample_count": int(len(smoke_rows)), "finite_predictions": bool(np.all(np.isfinite(smoke_prediction))),
    })

    plan = HPOPlan(direction=runtime.metric_direction, sampler="fixed_grid_dependency_light")
    atomic_write_json(run_root / "hpo" / "plan.json", {
        **asdict(plan),
        "executed_stage": "sanity_8_then_pilot_20" if run_pilot else "sanity_8_fixed_candidates",
        "pilot_status": "scheduled" if run_pilot else "skipped_by_explicit_runner_option",
        "test_access": "forbidden",
        "train_limit_per_fold_for_regression_sanity": sanity_train_limit,
    })

    def objective(config: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
        fold_scores: list[float] = []
        for fold in split.folds:
            train_rows = _indices(sample_ids, fold.train_sample_ids)
            if runtime.problem == "regression":
                train_rows = _stable_subsample(train_rows, sample_ids, seed + fold.fold_id, sanity_train_limit)
            validation_rows = _indices(sample_ids, fold.validation_sample_ids)
            model = _build_and_fit(spec, runtime, config, x[train_rows], fit_y[train_rows])
            prediction = _predict(model, spec, runtime, x[validation_rows])
            if runtime.problem == "binary":
                score = average_precision_score(physical_y[validation_rows], prediction)
            else:
                score = np.mean(np.abs(physical_y[validation_rows] - prediction))
            fold_scores.append(float(score))
        return {"fold_scores": fold_scores, "guardrails": {"test_rows_accessed": 0.0}}

    trials = run_fixed_trials(
        _candidate_configs(runtime), objective, root_seed=root_seed,
        output_dir=run_root / "hpo", metric_direction=runtime.metric_direction,
    )
    pilot_trials = []
    if run_pilot:
        pilot_trials = run_fixed_trials(
            _pilot_configs(runtime), objective, root_seed=derive_seed(root_seed, "hpo_sampler", "pilot"),
            output_dir=run_root / "hpo" / "pilot", metric_direction=runtime.metric_direction,
        )
    all_trials = [*trials, *pilot_trials]
    selected_trial = rank_trials(all_trials, direction=runtime.metric_direction)[0]
    selected_config = dict(selected_trial.params)
    atomic_write_json(run_root / "hpo" / "selection.json", {
        "trial_id": selected_trial.trial_id,
        "trial_stage": "pilot" if selected_trial in pilot_trials else "sanity",
        "metric": runtime.metric_name,
        "direction": runtime.metric_direction,
        "fold_mean": selected_trial.mean,
        "fold_scores": selected_trial.fold_scores,
        "model_config": selected_config,
        "selection_scope": "development folds only",
    })

    oof_rows: list[int] = []
    oof_predictions: list[float] = []
    fold_records: list[dict[str, Any]] = []
    fold_config = {
        "task_spec": spec.to_dict(), "model_id": runtime.model_id,
        "model_config": selected_config, "root_seed": root_seed,
    }
    fold_config_hash = hash_payload(fold_config)
    for fold in split.folds:
        train_rows = _indices(sample_ids, fold.train_sample_ids)
        validation_rows = _indices(sample_ids, fold.validation_sample_ids)
        model = _build_and_fit(spec, runtime, selected_config, x[train_rows], fit_y[train_rows])
        prediction = _predict(model, spec, runtime, x[validation_rows])
        fold_dir = run_root / "folds" / f"fold_{fold.fold_id}"
        checkpoint = _save_model_checkpoint(
            fold_dir / "checkpoint_best.pkl", model, config_hash=fold_config_hash,
            split_hash=split_hash, seed_report=seed_report, config=fold_config,
        )
        fold_records.append({
            "fold_id": fold.fold_id,
            "train_groups": list(fold.train_groups),
            "validation_groups": list(fold.validation_groups),
            "train_samples": int(len(train_rows)),
            "validation_samples": int(len(validation_rows)),
            "checkpoint_sha256": hash_file(checkpoint),
        })
        oof_rows.extend(validation_rows.tolist())
        oof_predictions.extend(prediction.tolist())
    if sorted(oof_rows) != sorted(_indices(sample_ids, split.development_sample_ids).tolist()):
        raise RuntimeError("OOF rows do not cover development exactly once")
    order = np.argsort(np.asarray(oof_rows))
    oof_rows_array = np.asarray(oof_rows, dtype=int)[order]
    oof_prediction_array = np.asarray(oof_predictions, dtype=float)[order]
    oof_observed = physical_y[oof_rows_array]
    threshold = _best_threshold(oof_observed, oof_prediction_array) if runtime.problem == "binary" else 0.5
    oof_metrics = (
        _binary_metrics(oof_observed, oof_prediction_array, threshold)
        if runtime.problem == "binary" else _regression_metrics(oof_observed, oof_prediction_array)
    )
    oof_frame = _prediction_frame(target_id, dataset, oof_rows_array, oof_observed, oof_prediction_array)
    _extra_metrics(target_id, oof_frame, oof_metrics, threshold)
    oof_frame.to_csv(run_root / "oof" / "predictions.csv", index=False)
    atomic_write_json(run_root / "oof" / "metrics.json", oof_metrics)
    atomic_write_json(run_root / "oof" / "folds.json", fold_records)
    lifecycle.advance(ExperimentState.CV_COMPLETE, {
        "oof_metrics_hash": hash_payload(oof_metrics), "sample_count": int(len(oof_frame)),
    })

    frozen_config = {
        "task_spec": spec.to_dict(),
        "model_id": runtime.model_id,
        "model_config": selected_config,
        "root_seed": root_seed,
        "threshold": float(threshold),
        "target_transform": "log1p" if runtime.transformed_target else "identity",
        "selection_scope": "development_oof_only",
    }
    config_hash = hash_payload(frozen_config)
    atomic_write_json(run_root / "config.json", frozen_config)
    lifecycle.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": config_hash})

    development_rows = _indices(sample_ids, split.development_sample_ids)
    refit_model = _build_and_fit(spec, runtime, selected_config, x[development_rows], fit_y[development_rows])
    refit_checkpoint = _save_model_checkpoint(
        run_root / "refit" / "checkpoint_best.pkl", refit_model,
        config_hash=config_hash, split_hash=split_hash, seed_report=seed_report, config=frozen_config,
    )
    checkpoint_hash = hash_file(refit_checkpoint)
    lifecycle.advance(ExperimentState.REFIT_COMPLETE, {"checkpoint_hash": checkpoint_hash})

    gate = FrozenTestGate(sample_ids, split.test_sample_ids)
    lifecycle.consume_test(config_hash=config_hash, checkpoint_hash=checkpoint_hash, split_hash=split_hash)
    test_rows = gate.consume()
    restored_model, restored = _restore_checkpoint(refit_checkpoint, spec, runtime)
    if restored["config_hash"] != config_hash or restored["split_hash"] != split_hash:
        raise RuntimeError("restored checkpoint does not match frozen run hashes")
    test_prediction = _predict(restored_model, spec, runtime, x[test_rows])
    test_observed = physical_y[test_rows]
    test_metrics = (
        _binary_metrics(test_observed, test_prediction, threshold)
        if runtime.problem == "binary" else _regression_metrics(test_observed, test_prediction)
    )
    test_frame = _prediction_frame(target_id, dataset, test_rows, test_observed, test_prediction)
    _extra_metrics(target_id, test_frame, test_metrics, threshold)
    test_frame.to_csv(run_root / "frozen_test" / "predictions.csv", index=False)
    atomic_write_json(run_root / "frozen_test" / "metrics.json", test_metrics)
    visualize.render(
        target_id, run_root / "frozen_test" / "predictions.csv",
        run_root / "visualizations", frozen_threshold=threshold,
    )
    lifecycle.advance(ExperimentState.VERIFIED, {
        "metrics_hash": hash_payload(test_metrics),
        "prediction_hash": hash_file(run_root / "frozen_test" / "predictions.csv"),
        "test_rows_accessed": int(len(test_rows)),
    })
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())
    status = {
        "target_number": TARGET_IDS.index(target_id) + 1,
        "target_id": target_id,
        "task_id": spec.task_id,
        "label_version": spec.label_version,
        "status": contract.STATUS,
        "model_id": runtime.model_id,
        "model_config": selected_config,
        "hpo_sanity_trials": len(trials),
        "hpo_pilot_trials": len(pilot_trials),
        "hpo_pilot_status": "complete" if run_pilot else "skipped_by_explicit_runner_option",
        "requested_folds": split.requested_n_splits,
        "effective_folds": split.effective_n_splits,
        "downgrade_reason": split.downgrade_reason,
        "development_samples": int(len(development_rows)),
        "test_samples": int(len(test_rows)),
        "test_consumed_once": gate.consumed,
        "oof_metrics": oof_metrics,
        "frozen_test_metrics": test_metrics,
        "proxy_warning": spec.metadata.get("proxy_warning") or spec.metadata.get("label_semantics"),
    }
    atomic_write_json(run_root / "status.json", status)
    _write_manifest(run_root, lifecycle.experiment_id)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=(*TARGET_IDS, "all"), default="all")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--sanity-train-limit", type=int, default=4000)
    parser.add_argument("--skip-pilot", action="store_true", help="run only the eight-trial sanity stage")
    args = parser.parse_args()
    selected = TARGET_IDS if args.target == "all" else (args.target,)
    results = {}
    for target_id in selected:
        output = (
            args.output_root / target_id / CASE_NAME
            if args.output_root is not None
            else Path(__file__).resolve().parent / target_id / "_outputs" / CASE_NAME
        )
        results[target_id] = run_target(
            target_id, output, source_root=args.source_root,
            root_seed=args.seed, sanity_train_limit=args.sanity_train_limit,
            run_pilot=not args.skip_pilot,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
