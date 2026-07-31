#!/usr/bin/env python3
"""P5 Stage-4 confirmation on the historically seen reconstruction holdouts.

This is deliberately not a blind-test runner.  It freezes the two Stage-3
development winners, refits the same PyKrige adapter within its frozen input
and wall-time budgets, and reports a labelled known-holdout confirmation.
It never writes into a P4 run root or changes the P4 lifecycle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping, Sequence

_AUX_SITE_PACKAGES = os.environ.get("VOLVE_P5_AUX_SITE_PACKAGES")
if _AUX_SITE_PACKAGES:
    sys.path.append(_AUX_SITE_PACKAGES)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from ml_framework.contracts import ModelBatch  # noqa: E402
from ml_framework.model_discovery import discover_model  # noqa: E402
from ml_framework.seeding import seed_everything  # noqa: E402

import p4_reconstruction as p4  # noqa: E402
import p5_stage1 as stage1  # noqa: E402
import reconstruction_p5_stage2 as stage2  # noqa: E402


SCHEMA_VERSION = "p5-stage4-reconstruction-confirmation-v1"
SUMMARY_SCHEMA_VERSION = "p5-stage4-reconstruction-summary-v1"
EVIDENCE_CLASS = "previously_seen_reusable_holdout"
ROOT_SEED = 2693
WINNER = "pykrige_ok3d"
MODES = ("strict", "conditional")
POINT_TRAIN_VOXELS = stage2.POINT_TRAIN_VOXELS
PREDICTION_CHUNK_SIZE = stage2.VALIDATION_VOXELS
DEFAULT_OUTPUT_DIR = HERE / "p5_stage4_confirmation"


def _portable_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"canonical Stage-4 artifact is outside the project: {path}") from exc


def _atomic_npz(path: Path, **arrays: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage3_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (HERE / "p5_stage3_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def frozen_stage3_winner(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    leaderboard_path = HERE / f"p5_stage3_leaderboard_{mode}.json"
    leaderboard = _json(leaderboard_path)
    summary = _json(HERE / "p5_stage3_summary.json")
    if leaderboard.get("lane") != mode or leaderboard.get("task_id") != p4.protocol(mode).task_id:
        raise RuntimeError("Stage-3 leaderboard lane/task identity mismatch")
    if leaderboard.get("rankable") is not True or leaderboard.get("development_only") is not True:
        raise RuntimeError("Stage-3 leaderboard is not rankable development-only evidence")
    if leaderboard.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-3 leaderboard reports frozen-test access")
    winner = min(leaderboard["entries"], key=lambda item: int(item["rank"]))
    if winner.get("model_id") != WINNER or int(winner.get("rank", -1)) != 1:
        raise RuntimeError(f"{mode} frozen Stage-3 winner is not {WINNER}")
    split_hash = str(leaderboard["split_hash"])
    if summary["split_hashes"].get(mode) != split_hash:
        raise RuntimeError("Stage-3 summary and leaderboard split hashes differ")
    rows = [
        row
        for row in _stage3_rows()
        if row["lane"] == mode and row["model_id"] == WINNER and row["status"] == "passed"
    ]
    if len(rows) != int(winner["passed_cells"]):
        raise RuntimeError("Stage-3 winner cell count differs from results evidence")
    budgets = {json.dumps(row["budget"], sort_keys=True) for row in rows}
    if len(budgets) != 1:
        raise RuntimeError("Stage-3 winner cells do not share one resource budget")
    budget = json.loads(next(iter(budgets)))
    expected_budget = stage2.budget_for(discover_model("reconstruction", WINNER).capabilities)
    if budget != expected_budget or budget != {
        "model_class": "traditional_cpu",
        "max_updates": 1,
        "max_wall_seconds": 300,
    }:
        raise RuntimeError("Stage-3 PyKrige budget is not the frozen 1-fit/300-second budget")
    templates: set[str] = set()
    for row in rows:
        config = dict(row["model_config"])
        config.pop("seed", None)
        templates.add(json.dumps(config, sort_keys=True))
    if len(templates) != 1:
        raise RuntimeError("Stage-3 winner cells have non-seed configuration drift")
    return {
        "leaderboard": _portable_project_path(leaderboard_path),
        "leaderboard_sha256": hash_file(leaderboard_path),
        "split_hash": split_hash,
        "winner_entry": winner,
        "passed_result_hashes": [row["result_hash"] for row in rows],
        "config_without_seed": json.loads(next(iter(templates))),
        "budget": budget,
    }


def verify_prior_exposure(mode: str, data_dir: Path) -> dict[str, Any]:
    """Prove that this physical holdout was scored before P4 Stage-4."""
    active = p4.protocol(mode)
    result_path = HERE / f"results_{mode}.json"
    run_manifest_path = HERE / f"run_manifest_{mode}.json"
    result = _json(result_path)
    run_manifest = _json(run_manifest_path)
    if result.get("evaluation_mode") != mode or run_manifest.get("evaluation_mode") != mode:
        raise RuntimeError("legacy exposure evidence belongs to a different mode")
    if result.get("test", {}).get("patch_i_blocks") != list(active.test_i_blocks):
        raise RuntimeError("legacy exposure used different physical test I-blocks")
    model_metrics = result.get("models", {}).get(result.get("primary_baseline"), {})
    if not all(math.isfinite(float(model_metrics[name])) for name in ("rmse", "mae")):
        raise RuntimeError("legacy exposure does not contain finite holdout metrics")
    actual_hashes: dict[str, str] = {}
    for container in ("train.h5", "test.h5"):
        path = data_dir / container
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hashes[container] = hash_file(path)
        expected = run_manifest["input_data_sha256"][f"_data/processed/reconstruction/{container}"]
        if actual_hashes[container] != expected:
            raise RuntimeError(f"{container} differs from the historically scored dataset")
    # P4 run roots are isolated under this directory by the track contract.
    # Do not mistake Stage-4's own frozen_config.json for a formal P4 state.
    p4_runs = HERE / "_outputs" / "runs"
    formal_p4_state = (
        list(p4_runs.rglob("lifecycle.json")) + list(p4_runs.rglob("frozen_config.json"))
        if p4_runs.is_dir()
        else []
    )
    return {
        "prior_test_consumed": True,
        "evidence_class": EVIDENCE_CLASS,
        "fresh_blind": False,
        "legacy_same_holdout_metrics_present": True,
        "formal_p4_test_consumed_artifact_present": bool(formal_p4_state),
        "legacy_result": _portable_project_path(result_path),
        "legacy_result_sha256": hash_file(result_path),
        "legacy_run_manifest": _portable_project_path(run_manifest_path),
        "legacy_run_manifest_sha256": hash_file(run_manifest_path),
        "test_i_blocks": list(active.test_i_blocks),
        "legacy_primary_baseline": result["primary_baseline"],
        "legacy_primary_metrics": {
            "rmse": float(model_metrics["rmse"]),
            "mae": float(model_metrics["mae"]),
        },
        "canonical_input_sha256": actual_hashes,
    }


def _point_batch(mode: str, features: np.ndarray, target: np.ndarray) -> ModelBatch:
    spec = p4.task_spec(mode)
    target_name = spec.targets[0]
    count = int(target.size)
    return ModelBatch(
        inputs={"features": np.asarray(features, dtype=np.float64)},
        targets={target_name: np.asarray(target, dtype=np.float64)},
        input_masks={"features": np.ones((count,), dtype=bool)},
        target_masks={target_name: np.ones((count,), dtype=bool)},
        sample_ids=[f"stage4_{mode}_development_refit"],
        groups={"evaluation_mode": [mode]},
        coordinates={"xyz": np.asarray(features[:, -3:], dtype=np.float64)},
        metadata={
            "evaluation_mode": mode,
            "scope": "all legal development preprocessing; frozen 512-point model-fit cap",
            "prior_test_consumed": True,
            "fresh_blind": False,
        },
    )


def _test_features(
    mode: str,
    development_records: Sequence[p4.PatchRecord],
    test_records: Sequence[p4.PatchRecord],
    preprocess_report: Mapping[str, Any],
) -> tuple[p4.FlatCells, np.ndarray, dict[str, Any]]:
    active = p4.protocol(mode)
    development_cells = p4.flatten_records(development_records)
    test_cells = p4.flatten_records(test_records)
    development_constraints = p4.constraints_from_records(development_records)
    test_constraints = p4.constraints_from_records(test_records)
    if mode == "strict" and test_constraints.shape[0] != 0:
        raise RuntimeError("strict known holdout contains a forbidden test-region constraint")
    allowed = (
        np.concatenate([development_constraints, test_constraints])
        if mode == "conditional" and test_constraints.shape[0]
        else development_constraints
    )
    fallback = float(np.mean(development_cells.target))
    raw, feature_names = p4._raw_features(mode, test_cells, allowed, fallback=fallback)  # noqa: SLF001
    p4.assert_feature_contract(mode, feature_names)
    features = p4._normalization_from_report(raw, preprocess_report)  # noqa: SLF001
    audit = {
        "evaluation_mode": mode,
        "development_constraints": int(development_constraints.shape[0]),
        "development_constraints_used": (
            int(development_constraints.shape[0]) if active.idw_feature_name else 0
        ),
        "test_constraints_present": int(test_constraints.shape[0]),
        "test_constraints_used": int(test_constraints.shape[0]) if mode == "conditional" else 0,
        "guard_constraints_used": 0,
        "strict_test_target_or_future_feature_used": False,
        "strict_reference_derived_well_value_used": False,
        "exact_constraint_cells_excluded_from_metrics": int(test_cells.observed_mask.sum()),
        "conditional_reconstruction_not_strict_holdout": mode == "conditional",
    }
    return test_cells, features, audit


def _predict_chunks(model: Any, features: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(features), PREDICTION_CHUNK_SIZE):
        values = np.asarray(
            model.predict_array(features[start : start + PREDICTION_CHUNK_SIZE]),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise FloatingPointError("Stage-4 PyKrige prediction contains non-finite values")
        chunks.append(values)
    prediction = np.concatenate(chunks)
    if prediction.shape != (len(features),):
        raise RuntimeError("Stage-4 chunked prediction changed output shape")
    return prediction


def _dense(values: np.ndarray, indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    volume = np.full(shape, np.nan, dtype=np.float64)
    volume[tuple(indices.T)] = values
    return volume


def _best_slice(volume: np.ndarray, axis: int) -> np.ndarray:
    reduced = tuple(index for index in range(3) if index != axis)
    support = np.sum(np.isfinite(volume), axis=reduced)
    return np.take(volume, int(np.argmax(support)), axis=axis)


def _radial_log_spectrum(volume: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(volume)
    if not np.any(finite):
        raise ValueError("spectrum diagnostic has no finite voxel")
    filled = np.where(finite, volume, float(np.mean(volume[finite])))
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(filled))))
    grids = np.meshgrid(
        *[np.fft.fftshift(np.fft.fftfreq(size)) for size in filled.shape], indexing="ij"
    )
    radius = np.sqrt(sum(grid**2 for grid in grids))
    edges = np.linspace(0.0, float(radius.max()) + 1e-12, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.asarray(
        [
            float(np.mean(magnitude[(radius >= left) & (radius < right)]))
            for left, right in zip(edges[:-1], edges[1:])
        ]
    )
    return centers, profile


def render_figure(
    path: Path,
    *,
    mode: str,
    truth_values: np.ndarray,
    prediction_values: np.ndarray,
    indices_kji: np.ndarray,
    volume_shape_kji: tuple[int, int, int],
    metrics: Mapping[str, Any],
    constraint_audit: Mapping[str, Any],
) -> Path:
    truth = _dense(truth_values, indices_kji, volume_shape_kji)
    prediction = _dense(prediction_values, indices_kji, volume_shape_kji)
    residual = prediction - truth
    fig, axes = plt.subplots(4, 3, figsize=(17, 21), constrained_layout=True)
    names = ("K/time-depth", "J/crossline", "I/inline")
    property_min = float(min(np.min(truth_values), np.min(prediction_values)))
    property_max = float(max(np.max(truth_values), np.max(prediction_values)))
    residual_limit = max(float(np.max(np.abs(prediction_values - truth_values))), 1e-8)
    for row, (axis, name) in enumerate(zip(range(3), names)):
        for column, (volume, title) in enumerate(
            ((truth, "reference"), (prediction, "reconstruction"), (residual, "residual"))
        ):
            kwargs: dict[str, Any] = {"origin": "lower", "aspect": "auto"}
            if column == 2:
                kwargs.update(cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit)
            else:
                kwargs.update(cmap="viridis", vmin=property_min, vmax=property_max)
            image = axes[row, column].imshow(_best_slice(volume, axis), **kwargs)
            axes[row, column].set_title(f"{name}: {title} porosity")
            fig.colorbar(image, ax=axes[row, column], shrink=0.75)

    target_sorted = np.sort(truth_values)
    prediction_sorted = np.sort(prediction_values)
    probability = np.arange(1, len(target_sorted) + 1, dtype=np.float64) / len(target_sorted)
    axes[3, 0].plot(target_sorted, probability, label="reference")
    axes[3, 0].plot(prediction_sorted, probability, label="prediction")
    axes[3, 0].set(
        title="Known-holdout porosity ECDF (diagnostic only)",
        xlabel="porosity",
        ylabel="empirical cumulative probability",
    )
    axes[3, 0].legend()

    frequency, truth_spectrum = _radial_log_spectrum(truth)
    _, prediction_spectrum = _radial_log_spectrum(prediction)
    axes[3, 1].plot(frequency, truth_spectrum, label="reference")
    axes[3, 1].plot(frequency, prediction_spectrum, label="prediction")
    axes[3, 1].set(
        title="Full known-holdout 3-D radial log-spectrum diagnostic",
        xlabel="spatial frequency",
        ylabel="mean log(1+|FFT|)",
    )
    axes[3, 1].legend()

    names_map = p4.metric_names(mode)
    axes[3, 2].axis("off")
    caveat = (
        f"CONDITIONAL: {constraint_audit['test_constraints_used']} test-region constraints used; "
        "not strict holdout."
        if mode == "conditional"
        else "STRICT spatial block: no guard/test-region porosity constraint used."
    )
    axes[3, 2].text(
        0.0,
        1.0,
        "\n".join(
            [
                "KNOWN HOLDOUT CONFIRMATION",
                "prior_test_consumed=true",
                "fresh_blind=false",
                f"RMSE={float(metrics[names_map['rmse']]):.6f}",
                f"MAE={float(metrics[names_map['mae']]):.6f}",
                f"spectral log-RMSE={float(metrics[names_map['spectral_log_rmse']]):.6f}",
                caveat,
                "ECDF is diagnostic only; no CDF score is used for selection.",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.suptitle(f"Volve Stage-4 reconstruction — {mode} — {WINNER}", fontsize=16)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(temporary, format="png", dpi=150)
    plt.close(fig)
    os.replace(temporary, path)
    return path


def _write_state(mode_dir: Path, mode: str, state: str, **evidence: Any) -> Path:
    return atomic_write_json(
        mode_dir / "confirmation_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "task_id": p4.protocol(mode).task_id,
            "mode": mode,
            "state": state,
            "prior_test_consumed": True,
            "evidence_class": EVIDENCE_CLASS,
            "fresh_blind": False,
            "evidence": evidence,
        },
    )


def run_mode(mode: str, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if Path(sys.executable).parent.parent.name != "torch-common":
        raise RuntimeError("Stage-4 must use the frozen torch-common primary environment")
    if os.environ.get("VOLVE_P5_AUX_DEPENDENCY_GROUP") != "tabular-cpu":
        raise RuntimeError("Stage-4 requires the already-provisioned tabular-cpu auxiliary group")
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    winner = frozen_stage3_winner(mode)
    exposure = verify_prior_exposure(mode, data_dir)
    catalog = p4.scan_patch_catalog(data_dir)
    split_manifest = p4.build_spatial_manifest(mode, catalog)
    if split_manifest.stable_hash() != winner["split_hash"]:
        raise RuntimeError("Stage-4 physical split differs from the frozen Stage-3 split hash")
    active = p4.protocol(mode)
    development_records = p4.load_patch_records(active.development_i_blocks, data_dir)
    prepared = p4.prepare_full_development(mode, development_records)
    selected = stage2._sample_indices(prepared.train_target.size, POINT_TRAIN_VOXELS)  # noqa: SLF001
    train_batch = _point_batch(
        mode, prepared.train_features[selected], prepared.train_target[selected]
    )
    config = stage1.model_config(
        WINNER, p4.task_spec(mode), train_batch, device="cpu", seed=ROOT_SEED
    )
    config_without_seed = dict(config)
    config_without_seed.pop("seed", None)
    if config_without_seed != winner["config_without_seed"]:
        raise RuntimeError("Stage-4 config differs from the frozen Stage-3 winner config")
    if int(config["n_training_samples"]) != POINT_TRAIN_VOXELS:
        raise RuntimeError("Stage-4 model-fit sample count differs from the frozen cap")
    source_lock = stage1.load_source_lock()
    frozen_payload = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "task_id": active.task_id,
        "mode": mode,
        "model_id": WINNER,
        "root_seed": ROOT_SEED,
        "model_config": config,
        "budget": winner["budget"],
        "split_hash": winner["split_hash"],
        "selection_source": "P5 Stage-3 buffered-development OOF leaderboard only",
        "source": stage2._portable_source(WINNER, source_lock),  # noqa: SLF001
        "stage3_leaderboard": winner["leaderboard"],
        "stage3_leaderboard_sha256": winner["leaderboard_sha256"],
        "stage3_passed_result_hashes": winner["passed_result_hashes"],
        "prior_exposure": exposure,
        "prior_test_consumed": True,
        "evidence_class": EVIDENCE_CLASS,
        "fresh_blind": False,
        "cdf_diagnostic_only": True,
    }
    frozen_payload["config_hash"] = hash_payload(frozen_payload)
    frozen_path = atomic_write_json(mode_dir / "frozen_config.json", frozen_payload)
    _write_state(
        mode_dir,
        mode,
        "CONFIG_FROZEN",
        config_hash=frozen_payload["config_hash"],
        split_hash=winner["split_hash"],
    )

    discovered = discover_model("reconstruction", WINNER)
    budget = winner["budget"]
    started = time.monotonic()
    holdout_access_started = False
    try:
        seed_report = seed_everything(ROOT_SEED, strict=True, include_torch=False).to_dict()
        with stage2._wall_timeout(int(budget["max_wall_seconds"])):  # noqa: SLF001
            model = discovered.build(p4.task_spec(mode), **config)
            train_step = dict(model.train_batch(train_batch))
            if not math.isfinite(float(train_step["loss"])):
                raise FloatingPointError("Stage-4 refit loss is non-finite")
            checkpoint_path = mode_dir / "refit_checkpoint.npz"
            model.save_checkpoint(checkpoint_path)
            restored = discovered.build(p4.task_spec(mode), **config)
            restored.load_checkpoint(checkpoint_path)
            probe = prepared.train_features[selected[: min(64, len(selected))]]
            roundtrip_error = float(
                np.max(np.abs(model.predict_array(probe) - restored.predict_array(probe)))
            )
            if roundtrip_error > 1e-8:
                raise AssertionError("Stage-4 checkpoint round-trip exceeded tolerance")
            refit_payload = {
                "schema_version": SCHEMA_VERSION,
                "mode": mode,
                "model_id": WINNER,
                "status": "passed",
                "updates": int(model.update_count),
                "train_loss": float(train_step["loss"]),
                "development_patch_count": len(development_records),
                "development_active_cells": int(prepared.train_target.size),
                "preprocess_fit_scope": "all legal development active cells",
                "preprocess_report": prepared.preprocess_report,
                "model_fit_selection": "deterministic linspace over all legal development active cells",
                "model_fit_cells": int(len(selected)),
                "model_fit_index_sha256": hash_payload(selected.tolist()),
                "constraint_audit": prepared.constraint_audit,
                "checkpoint": "refit_checkpoint.npz",
                "checkpoint_sha256": hash_file(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_roundtrip_max_abs_error": roundtrip_error,
                "seed_report": seed_report,
            }
            refit_path = atomic_write_json(mode_dir / "refit.json", refit_payload)
            _write_state(
                mode_dir,
                mode,
                "REFIT_COMPLETE",
                config_hash=frozen_payload["config_hash"],
                checkpoint_sha256=refit_payload["checkpoint_sha256"],
                split_hash=winner["split_hash"],
            )

            # This is intentionally a reusable, already-seen holdout.  Record
            # access before reading its arrays so a crash cannot be mislabelled.
            _write_state(
                mode_dir,
                mode,
                "KNOWN_HOLDOUT_ACCESS_STARTED",
                prior_test_consumed=True,
                historical_result_sha256=exposure["legacy_result_sha256"],
                split_hash=winner["split_hash"],
            )
            holdout_access_started = True
            test_records = p4.load_patch_records(active.test_i_blocks, data_dir)
            test_cells, test_features, constraint_audit = _test_features(
                mode, development_records, test_records, prepared.preprocess_report
            )
            metric_mask = ~test_cells.observed_mask
            if not np.any(metric_mask):
                raise RuntimeError("known holdout has no legal metric cells")
            metric_features = test_features[metric_mask]
            prediction = _predict_chunks(restored, metric_features)
            truth = np.asarray(test_cells.target[metric_mask], dtype=np.float64)
            indices = np.asarray(test_cells.indices_kji[metric_mask], dtype=np.int64)
            metrics = p4.regression_metrics(
                mode,
                truth,
                prediction,
                indices_kji=indices,
                volume_shape_kji=test_cells.volume_shape_kji,
                train_range=(
                    float(np.min(prepared.train_target)),
                    float(np.max(prepared.train_target)),
                ),
            )
            predictions_path = _atomic_npz(
                mode_dir / "predictions.npz",
                mode=np.asarray(mode),
                task_id=np.asarray(active.task_id),
                evidence_class=np.asarray(EVIDENCE_CLASS),
                prior_test_consumed=np.asarray(True),
                fresh_blind=np.asarray(False),
                sample_ids=np.asarray(test_cells.sample_ids[metric_mask]),
                indices_kji=indices,
                volume_shape_kji=np.asarray(test_cells.volume_shape_kji, dtype=np.int64),
                truth=truth,
                prediction=prediction,
                residual=prediction - truth,
            )
            metric_payload = {
                "schema_version": SCHEMA_VERSION,
                "track_id": "reconstruction",
                "task_id": active.task_id,
                "mode": mode,
                "model_id": WINNER,
                "status": "passed",
                "prior_test_consumed": True,
                "evidence_class": EVIDENCE_CLASS,
                "fresh_blind": False,
                "evaluation_scope": "historically seen mode-specific spatial holdout",
                "metrics": metrics,
                "test_patch_count": len(test_records),
                "test_active_cells": int(test_cells.target.size),
                "metric_voxels": int(np.sum(metric_mask)),
                "excluded_exact_constraint_cells": int(np.sum(test_cells.observed_mask)),
                "constraint_audit": constraint_audit,
                "cdf": {
                    "status": "diagnostic_only",
                    "definition": "unweighted empirical CDF over unique legal holdout metric voxels",
                    "used_for_selection": False,
                    "numeric_score_reported": False,
                },
                "spectral_metric": {
                    "name": p4.metric_names(mode)["spectral_log_rmse"],
                    "definition": "3-D mean-filled log1p(abs(rfftn)) voxel-spectrum RMSE",
                },
            }
            metrics_path = atomic_write_json(mode_dir / "metrics.json", metric_payload)
            figure_path = render_figure(
                mode_dir / "confirmation.png",
                mode=mode,
                truth_values=truth,
                prediction_values=prediction,
                indices_kji=indices,
                volume_shape_kji=test_cells.volume_shape_kji,
                metrics=metrics,
                constraint_audit=constraint_audit,
            )
        wall_seconds = time.monotonic() - started
        stage2.validate_budget(budget, int(model.update_count), wall_seconds)
        refit_payload["model_wall_seconds_including_known_holdout_inference"] = wall_seconds
        atomic_write_json(refit_path, refit_payload)
        files = [
            frozen_path,
            refit_path,
            checkpoint_path,
            predictions_path,
            metrics_path,
            figure_path,
        ]
        artifact_manifest = {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "task_id": active.task_id,
            "mode": mode,
            "prior_test_consumed": True,
            "evidence_class": EVIDENCE_CLASS,
            "fresh_blind": False,
            "artifacts": {
                path.name: {"sha256": hash_file(path), "bytes": path.stat().st_size}
                for path in files
            },
        }
        manifest_path = atomic_write_json(mode_dir / "manifest.json", artifact_manifest)
        status = {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "task_id": active.task_id,
            "mode": mode,
            "model_id": WINNER,
            "status": "passed",
            "reason": None,
            "prior_test_consumed": True,
            "evidence_class": EVIDENCE_CLASS,
            "fresh_blind": False,
            "split_hash": winner["split_hash"],
            "config_hash": frozen_payload["config_hash"],
            "manifest": "manifest.json",
            "manifest_sha256": hash_file(manifest_path),
            "wall_seconds": wall_seconds,
            "metrics": metrics,
            "counts": {
                "development_patches": len(development_records),
                "development_active_cells": int(prepared.train_target.size),
                "model_fit_cells": int(len(selected)),
                "test_patches": len(test_records),
                "test_active_cells": int(test_cells.target.size),
                "metric_voxels": int(np.sum(metric_mask)),
                "excluded_exact_constraint_cells": int(np.sum(test_cells.observed_mask)),
            },
            "resources": {
                "device": "cpu",
                "updates": int(model.update_count),
                "max_wall_seconds": int(budget["max_wall_seconds"]),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "prediction_chunk_size": PREDICTION_CHUNK_SIZE,
            },
            "environment": {
                "primary_environment": "torch-common",
                "aux_dependency_group": "tabular-cpu",
                "python": platform.python_version(),
                "python_executable": Path(sys.executable).name,
                "downloads_performed_bytes": 0,
            },
        }
        status["result_hash"] = hash_payload(status)
        atomic_write_json(mode_dir / "status.json", status)
        _write_state(
            mode_dir,
            mode,
            "CONFIRMATION_COMPLETE",
            result_hash=status["result_hash"],
            manifest_sha256=status["manifest_sha256"],
            prior_test_consumed=True,
            fresh_blind=False,
        )
        return status
    except stage2.PilotTimeout as exc:
        wall_seconds = time.monotonic() - started
        status = {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "task_id": active.task_id,
            "mode": mode,
            "model_id": WINNER,
            "status": "timeout",
            "reason": {"code": "budget_timeout", "message": str(exc)},
            "prior_test_consumed": True,
            "evidence_class": EVIDENCE_CLASS,
            "fresh_blind": False,
            "holdout_access_started": holdout_access_started,
            "split_hash": winner["split_hash"],
            "config_hash": frozen_payload["config_hash"],
            "wall_seconds": wall_seconds,
            "metrics": None,
        }
        status["result_hash"] = hash_payload(status)
        atomic_write_json(mode_dir / "status.json", status)
        _write_state(
            mode_dir,
            mode,
            "TIMEOUT",
            reason=status["reason"],
            holdout_access_started=holdout_access_started,
        )
        return status


def validate_mode_output(mode: str, output_dir: Path) -> dict[str, Any]:
    mode_dir = output_dir / mode
    status = _json(mode_dir / "status.json")
    if status.get("schema_version") != SCHEMA_VERSION or status.get("mode") != mode:
        raise ValueError("Stage-4 status schema/mode mismatch")
    if status.get("model_id") != WINNER or status.get("task_id") != p4.protocol(mode).task_id:
        raise ValueError("Stage-4 status model/task mismatch")
    if status.get("prior_test_consumed") is not True:
        raise ValueError("Stage-4 status hides prior holdout consumption")
    if status.get("evidence_class") != EVIDENCE_CLASS or status.get("fresh_blind") is not False:
        raise ValueError("Stage-4 evidence is mislabelled as fresh blind")
    expected_hash = hash_payload({key: value for key, value in status.items() if key != "result_hash"})
    if status.get("result_hash") != expected_hash:
        raise ValueError("Stage-4 status result hash mismatch")
    winner = frozen_stage3_winner(mode)
    if status.get("split_hash") != winner["split_hash"]:
        raise ValueError("Stage-4 status split differs from frozen Stage-3")
    if status.get("status") != "passed":
        if not (status.get("reason") or {}).get("code") or status.get("metrics") is not None:
            raise ValueError("Stage-4 non-passed result lacks a structured reason")
        return status
    manifest_path = mode_dir / status["manifest"]
    if hash_file(manifest_path) != status["manifest_sha256"]:
        raise ValueError("Stage-4 manifest hash mismatch")
    manifest = _json(manifest_path)
    if manifest.get("prior_test_consumed") is not True or manifest.get("fresh_blind") is not False:
        raise ValueError("Stage-4 manifest evidence label mismatch")
    for relative, evidence in manifest["artifacts"].items():
        path = mode_dir / relative
        if not path.is_file() or hash_file(path) != evidence["sha256"]:
            raise ValueError(f"Stage-4 artifact hash mismatch: {relative}")
        if path.stat().st_size != int(evidence["bytes"]):
            raise ValueError(f"Stage-4 artifact size mismatch: {relative}")
    metrics_payload = _json(mode_dir / "metrics.json")
    if metrics_payload.get("cdf", {}).get("status") != "diagnostic_only":
        raise ValueError("Stage-4 CDF is not explicitly diagnostic-only")
    if metrics_payload.get("constraint_audit", {}).get(
        "conditional_reconstruction_not_strict_holdout"
    ) is not (mode == "conditional"):
        raise ValueError("Stage-4 conditional caveat is inconsistent")
    names = p4.metric_names(mode)
    for name in ("rmse", "mae", "spectral_log_rmse"):
        if not math.isfinite(float(status["metrics"][names[name]])):
            raise ValueError(f"Stage-4 metric {name} is non-finite")
    with np.load(mode_dir / "predictions.npz", allow_pickle=False) as archive:
        if bool(archive["fresh_blind"]) or not bool(archive["prior_test_consumed"]):
            raise ValueError("Stage-4 prediction archive evidence label mismatch")
        truth = np.asarray(archive["truth"], dtype=np.float64)
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
        residual = np.asarray(archive["residual"], dtype=np.float64)
        if truth.shape != prediction.shape or not np.allclose(prediction - truth, residual):
            raise ValueError("Stage-4 compact prediction archive is inconsistent")
        if truth.size != int(status["counts"]["metric_voxels"]):
            raise ValueError("Stage-4 prediction count differs from status")
    return status


def collate(output_dir: Path) -> dict[str, Any]:
    statuses = {mode: validate_mode_output(mode, output_dir) for mode in MODES}
    if any(status["status"] != "passed" for status in statuses.values()):
        raise RuntimeError("Stage-4 confirmation cannot collate non-passed mode outputs")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "winner": {mode: WINNER for mode in MODES},
        "prior_test_consumed": True,
        "evidence_class": EVIDENCE_CLASS,
        "fresh_blind": False,
        "claim": "known-holdout confirmation only; not an independent blind test",
        "modes_are_independent": True,
        "conditional_reconstruction_not_strict_holdout": True,
        "results": {
            mode: {
                "status": status["status"],
                "result_hash": status["result_hash"],
                "split_hash": status["split_hash"],
                "config_hash": status["config_hash"],
                "metrics": status["metrics"],
                "counts": status["counts"],
                "wall_seconds": status["wall_seconds"],
                "manifest": f"{mode}/manifest.json",
                "manifest_sha256": status["manifest_sha256"],
                "predictions": f"{mode}/predictions.npz",
                "predictions_sha256": hash_file(output_dir / mode / "predictions.npz"),
                "figure": f"{mode}/confirmation.png",
                "figure_sha256": hash_file(output_dir / mode / "confirmation.png"),
            }
            for mode, status in statuses.items()
        },
    }
    summary["summary_hash"] = hash_payload(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-mode", help="run one labelled known-holdout confirmation")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    aggregate = subparsers.add_parser("collate", help="validate both mode outputs and summarize")
    aggregate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-mode":
        print(json.dumps(run_mode(args.mode, args.data_dir, args.output_dir), indent=2))
    elif args.command == "collate":
        print(json.dumps(collate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
