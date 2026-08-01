#!/usr/bin/env python3
"""P18 CIG-Bench RGT external-drift kriging diagnostic.

The pretrained RGT network reads only the assembled development seismic
amplitude.  Its target-free relative-geological-time volume is supplied as a
single specified external drift to PyKrige UniversalKriging3D.  Each fold keeps
the committed P5 conditional split, the exact 512 training labels, the 2,048
validation rows, linear variogram, and four lag bins used by the strong
OrdinaryKriging3D baseline.  This module exposes no holdout path argument.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import p11_residual_fusion as base  # noqa: E402
import p14_geophysical_fm as p14  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
from _models.reconstruction import cigbench_rgt  # noqa: E402


SCHEMA_VERSION = "reconstruction-p18-cigbench-ked/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p18_cigbench_ked"
CANONICAL_INFER_SHAPE = (400, 512, 512)
RESOURCE_ALIGNED_INFER_SHAPE = (128, 256, 256)
VARIOGRAM_MODEL = "linear"
NLAGS = 4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2693
OUTCOME_TOLERANCE = 1e-12


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    if target.shape != prediction.shape or not np.all(np.isfinite(error)):
        raise ValueError("metric inputs must be aligned and finite")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rows": int(len(target)),
    }


def _outcome(delta: float) -> str:
    if delta < -OUTCOME_TOLERANCE:
        return "win"
    if delta > OUTCOME_TOLERANCE:
        return "loss"
    return "tie"


def sample_rgt(volume: np.ndarray, indices_kji: np.ndarray) -> np.ndarray:
    """Select scalar external-drift values at exact global KJI cells."""

    volume = np.asarray(volume, dtype=np.float64)
    indices = np.asarray(indices_kji, dtype=np.int64)
    if volume.ndim != 3 or indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("RGT volume and KJI indices have invalid shapes")
    if np.any(indices < 0) or np.any(indices >= np.asarray(volume.shape)):
        raise IndexError("KJI index lies outside the RGT volume")
    result = volume[indices[:, 0], indices[:, 1], indices[:, 2]]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("sampled RGT drift is non-finite")
    return result


def _as_plain(values: Any) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        if np.any(np.ma.getmaskarray(values)):
            raise FloatingPointError("PyKrige returned masked predictions")
        values = values.data
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("PyKrige returned non-finite predictions")
    return result


def fit_predict_ok3d(
    train_xyz: np.ndarray,
    train_target: np.ndarray,
    validation_xyz: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recompute the unchanged P5 OrdinaryKriging3D control."""

    from pykrige.ok3d import OrdinaryKriging3D

    train_xyz = np.asarray(train_xyz, dtype=np.float64)
    validation_xyz = np.asarray(validation_xyz, dtype=np.float64)
    model = OrdinaryKriging3D(
        train_xyz[:, 0],
        train_xyz[:, 1],
        train_xyz[:, 2],
        np.asarray(train_target, dtype=np.float64),
        variogram_model=VARIOGRAM_MODEL,
        nlags=NLAGS,
        verbose=False,
        enable_plotting=False,
    )
    prediction, variance = model.execute(
        "points",
        validation_xyz[:, 0],
        validation_xyz[:, 1],
        validation_xyz[:, 2],
        backend="vectorized",
    )
    prediction = _as_plain(prediction)
    variance = _as_plain(variance)
    return prediction, {
        "variogram_model": VARIOGRAM_MODEL,
        "nlags": NLAGS,
        "variogram_parameters": np.asarray(
            model.variogram_model_parameters, dtype=np.float64
        ).tolist(),
        "variance_min": float(np.min(variance)),
        "variance_max": float(np.max(variance)),
    }


def fit_predict_ked(
    train_xyz: np.ndarray,
    train_target: np.ndarray,
    validation_xyz: np.ndarray,
    train_rgt: np.ndarray,
    validation_rgt: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Kriging with one specified external RGT drift and unchanged variogram."""

    from pykrige.uk3d import UniversalKriging3D

    train_xyz = np.asarray(train_xyz, dtype=np.float64)
    validation_xyz = np.asarray(validation_xyz, dtype=np.float64)
    train_rgt = np.asarray(train_rgt, dtype=np.float64)
    validation_rgt = np.asarray(validation_rgt, dtype=np.float64)
    if np.std(train_rgt) < 1e-8:
        raise RuntimeError("outer-train RGT drift is effectively constant")
    model = UniversalKriging3D(
        train_xyz[:, 0],
        train_xyz[:, 1],
        train_xyz[:, 2],
        np.asarray(train_target, dtype=np.float64),
        variogram_model=VARIOGRAM_MODEL,
        nlags=NLAGS,
        drift_terms=["specified"],
        specified_drift=[train_rgt],
        verbose=False,
        enable_plotting=False,
    )
    prediction, variance = model.execute(
        "points",
        validation_xyz[:, 0],
        validation_xyz[:, 1],
        validation_xyz[:, 2],
        backend="vectorized",
        specified_drift_arrays=[validation_rgt],
    )
    prediction = _as_plain(prediction)
    variance = _as_plain(variance)
    return prediction, {
        "kriging": "UniversalKriging3D with specified external drift",
        "drift_terms": ["specified:RGT"],
        "variogram_model": VARIOGRAM_MODEL,
        "nlags": NLAGS,
        "variogram_parameters": np.asarray(
            model.variogram_model_parameters, dtype=np.float64
        ).tolist(),
        "train_rgt_min": float(np.min(train_rgt)),
        "train_rgt_max": float(np.max(train_rgt)),
        "train_rgt_std": float(np.std(train_rgt)),
        "variance_min": float(np.min(variance)),
        "variance_max": float(np.max(variance)),
    }


def _kriging_coordinates(
    stage3_root: Path,
    fold: p17.FoldSamples,
    oof: base.OOFDevelopment,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Read the standardized XYZ columns that P5 passed to PyKrige exactly."""

    root = (
        Path(stage3_root).expanduser().resolve()
        / "cache"
        / base.EXPECTED_LANE
        / f"fold_{fold.fold_id:02d}"
    )
    train_path = root / "point_train.npz"
    validation_path = root / "point_validation.npz"
    base.ensure_no_holdout_paths([train_path, validation_path])
    with np.load(train_path, allow_pickle=False) as payload:
        train_values = np.asarray(payload["input_values"], dtype=np.float64)
        train_target = np.asarray(payload["target_values"], dtype=np.float64)
    with np.load(validation_path, allow_pickle=False) as payload:
        validation_values = np.asarray(payload["input_values"], dtype=np.float64)
        validation_target = np.asarray(payload["target_values"], dtype=np.float64)
    mask = oof.fold_ids == fold.fold_id
    np.testing.assert_allclose(train_target, fold.train_target, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        validation_target, oof.target[mask], rtol=0.0, atol=0.0
    )
    if train_values.shape != (512, 7) or validation_values.shape != (2048, 7):
        raise RuntimeError("P5 point-cache shape drift")
    return train_values[:, -3:], validation_values[:, -3:], {
        "point_train_npz_sha256": base._sha256(train_path),  # noqa: SLF001
        "point_validation_npz_sha256": base._sha256(  # noqa: SLF001
            validation_path
        ),
    }


def _whole_fold_bootstrap(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    """Resample five whole spatial folds, never individual voxels."""

    unique = np.asarray(sorted(np.unique(fold_ids).tolist()), dtype=np.int64)
    if unique.tolist() != list(base.FOLD_IDS):
        raise ValueError("whole-fold bootstrap requires the five locked folds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    masks = {int(fold): fold_ids == fold for fold in unique}
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draw_target = np.concatenate([target[masks[int(fold)]] for fold in sampled])
        draw_baseline = np.concatenate(
            [baseline[masks[int(fold)]] for fold in sampled]
        )
        draw_candidate = np.concatenate(
            [candidate[masks[int(fold)]] for fold in sampled]
        )
        deltas[draw] = (
            _metrics(draw_target, draw_candidate)["rmse"]
            - _metrics(draw_target, draw_baseline)["rmse"]
        )
    return {
        "unit": "whole locked spatial fold",
        "independent_units": 5,
        "voxel_independent_resampling": False,
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "rmse_delta_ked_minus_pykrige": {
            "estimate": float(
                _metrics(target, candidate)["rmse"]
                - _metrics(target, baseline)["rmse"]
            ),
            "ci95": [
                float(np.quantile(deltas, 0.025)),
                float(np.quantile(deltas, 0.975)),
            ],
            "probability_below_zero": float(np.mean(deltas < 0.0)),
        },
        "caveat": (
            "Only five coarse spatial units exist; this descriptive interval "
            "is not a large-sample significance claim."
        ),
    }


def _rgt_quality(rgt: np.ndarray, seismic: np.ndarray) -> dict[str, Any]:
    rgt = np.asarray(rgt, dtype=np.float64)
    seismic = np.asarray(seismic, dtype=np.float64)
    depth = np.broadcast_to(
        np.linspace(0.0, 1.0, rgt.shape[0])[:, None, None], rgt.shape
    )
    vertical = np.diff(rgt, axis=0)
    return {
        "shape_kji": list(rgt.shape),
        "finite": bool(np.all(np.isfinite(rgt))),
        "minimum": float(np.min(rgt)),
        "maximum": float(np.max(rgt)),
        "mean": float(np.mean(rgt)),
        "standard_deviation": float(np.std(rgt)),
        "positive_vertical_difference_fraction": float(np.mean(vertical > 0.0)),
        "negative_vertical_difference_fraction": float(np.mean(vertical < 0.0)),
        "pearson_with_depth_index": float(np.corrcoef(rgt.ravel(), depth.ravel())[0, 1]),
        "pearson_with_seismic_amplitude": float(
            np.corrcoef(rgt.ravel(), seismic.ravel())[0, 1]
        ),
        "rgt_sha256": _array_sha256(np.asarray(rgt, dtype=np.float32)),
    }


def _run_rgt(
    *,
    seismic: np.ndarray,
    weight_path: Path,
    weight_audit: Mapping[str, Any],
    device: str,
    infer_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    started = time.time()
    if device.startswith("cuda"):
        device_index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats()
    predictor = cigbench_rgt.build_predictor(
        restore_path=weight_path,
        device=device,
        infer_shape=infer_shape,
        expected_weight_sha256=str(weight_audit["sha256"]),
        use_autocast=True,
    )
    # Channel 0 is the true seismic amplitude. Channels 1/2 are engineered
    # local-RMS and vertical-gradient attributes, not extra surveys.
    rgt, seismic_used = predictor.predict(
        np.asarray(seismic[0], dtype=np.float32),
        clp_s=2.0,
        horizon_rgt=None,
        horizon_mask=None,
        resize_back=True,
        normalize_output=True,
    )
    peak = (
        int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda")
        else 0
    )
    del predictor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    rgt = np.asarray(rgt, dtype=np.float32)
    seismic_used = np.asarray(seismic_used, dtype=np.float32)
    if rgt.shape != seismic.shape[1:] or not np.all(np.isfinite(rgt)):
        raise RuntimeError("CIG-Bench RGT output shape/finite check failed")
    return rgt, seismic_used, {
        "infer_shape": [int(value) for value in infer_shape],
        "upstream_recommended_shape": list(CANONICAL_INFER_SHAPE),
        "matches_upstream_recommended_shape": (
            tuple(int(value) for value in infer_shape) == CANONICAL_INFER_SHAPE
        ),
        "input_channel": "seismic_amplitude (seismic_patch[0])",
        "clp_s": 2.0,
        "horizon_constraints": False,
        "resize_back": True,
        "normalize_output": True,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": peak,
    }


def probe_canonical(
    *,
    data_dir: Path,
    weight_cache: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    """Attempt the exact upstream grid and persist an honest feasibility result."""

    started = time.time()
    inputs = base.resolve_dev_inputs(data_dir)
    base.ensure_no_holdout_paths([weight_cache, output_path])
    weight_path, weight_audit = cigbench_rgt.resolve_weight(cache_dir=weight_cache)
    seismic, _, volume_audit = p14.assemble_seismic_volume(inputs.train_h5)
    try:
        _run_rgt(
            seismic=seismic,
            weight_path=weight_path,
            weight_audit=weight_audit,
            device=device,
            infer_shape=CANONICAL_INFER_SHAPE,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = "CUDA_OUT_OF_MEMORY"
        error_type = type(exc).__name__
        error_message = str(exc)
    else:
        status = "PASSED"
        error_type = None
        error_message = None
    record = {
        "schema_version": "reconstruction-p18-cigbench-canonical-probe/v1",
        "status": status,
        "infer_shape": list(CANONICAL_INFER_SHAPE),
        "input_volume_shape": volume_audit["assembled_volume_shape_kji"],
        "device": device,
        "weight_sha256": weight_audit["sha256"],
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": time.time() - started,
        "development_only": True,
        "frozen_holdout_opened": False,
    }
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _evaluate(
    *,
    stage3_root: Path,
    oof: base.OOFDevelopment,
    folds: Sequence[p17.FoldSamples],
    rgt: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ked_prediction = np.empty_like(oof.target, dtype=np.float64)
    recomputed = np.empty_like(oof.target, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        mask = oof.fold_ids == fold.fold_id
        train_xyz, validation_xyz, source_hashes = _kriging_coordinates(
            stage3_root, fold, oof
        )
        train_rgt = sample_rgt(rgt, fold.train_indices_kji)
        validation_rgt = sample_rgt(rgt, fold.validation_indices_kji)
        baseline_started = time.time()
        baseline_fold, baseline_model = fit_predict_ok3d(
            train_xyz, fold.train_target, validation_xyz
        )
        baseline_seconds = time.time() - baseline_started
        ked_started = time.time()
        candidate_fold, ked_model = fit_predict_ked(
            train_xyz,
            fold.train_target,
            validation_xyz,
            train_rgt,
            validation_rgt,
        )
        ked_seconds = time.time() - ked_started
        recomputed[mask] = baseline_fold
        ked_prediction[mask] = candidate_fold
        committed_metrics = _metrics(oof.target[mask], oof.baseline[mask])
        recomputed_metrics = _metrics(oof.target[mask], baseline_fold)
        candidate_metrics = _metrics(oof.target[mask], candidate_fold)
        delta = candidate_metrics["rmse"] - committed_metrics["rmse"]
        fold_rows.append(
            {
                "fold_id": int(fold.fold_id),
                "independent_spatial_unit": True,
                "train_labels": int(len(fold.train_target)),
                "validation_rows": int(np.sum(mask)),
                "pykrige_committed": committed_metrics,
                "pykrige_recomputed": recomputed_metrics,
                "ked": candidate_metrics,
                "rmse_delta_ked_minus_pykrige": delta,
                "outcome": _outcome(delta),
                "baseline_runtime_seconds": baseline_seconds,
                "ked_runtime_seconds": ked_seconds,
                "baseline_model": baseline_model,
                "ked_model": ked_model,
                "source_hashes": source_hashes,
            }
        )
    baseline_metrics = _metrics(oof.target, oof.baseline)
    recomputed_metrics = _metrics(oof.target, recomputed)
    ked_metrics = _metrics(oof.target, ked_prediction)
    max_abs = float(np.max(np.abs(recomputed - oof.baseline)))
    rmse_delta = ked_metrics["rmse"] - baseline_metrics["rmse"]
    counts = {
        label: sum(row["outcome"] == label for row in fold_rows)
        for label in ("win", "loss", "tie")
    }
    bootstrap = _whole_fold_bootstrap(
        target=oof.target,
        baseline=oof.baseline,
        candidate=ked_prediction,
        fold_ids=oof.fold_ids,
    )
    experiment = {
        "primary_comparison": "CIG-Bench RGT KED vs committed PyKrige OOF",
        "baseline": {"pykrige_ok3d_repeat_0": baseline_metrics},
        "baseline_recomputation": {
            "metrics": recomputed_metrics,
            "max_abs_prediction_difference_vs_committed_oof": max_abs,
            "rmse_difference_vs_committed_oof": (
                recomputed_metrics["rmse"] - baseline_metrics["rmse"]
            ),
            "same_model_and_protocol": True,
            "within_numerical_tolerance_1e_9": max_abs <= 1e-9,
        },
        "ked_metrics": ked_metrics,
        "rmse_delta_ked_minus_pykrige": rmse_delta,
        "relative_rmse_change": rmse_delta / baseline_metrics["rmse"],
        "per_fold": fold_rows,
        "independent_fold_outcome_counts": counts,
        "independent_fold_win_rate": counts["win"] / len(base.FOLD_IDS),
        "whole_fold_bootstrap": bootstrap,
        "decision": {
            "state": (
                "POSITIVE_DEVELOPMENT_SIGNAL"
                if rmse_delta < 0.0 and counts["win"] >= 4
                else "NO_ROBUST_DEVELOPMENT_GAIN"
            ),
            "default_enabled": False,
            "promotion_rule": "at least 1% pooled RMSE gain and wins in >=4/5 folds",
            "promotion_rule_passed": bool(
                -rmse_delta / baseline_metrics["rmse"] >= 0.01
                and counts["win"] >= 4
            ),
            "frozen_holdout_remains_sealed": True,
        },
    }
    payload = {
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "target": np.asarray(oof.target, dtype=np.float64),
        "pykrige_committed_prediction": np.asarray(oof.baseline, dtype=np.float64),
        "pykrige_recomputed_prediction": recomputed,
        "ked_prediction": ked_prediction,
        "rgt_at_validation": sample_rgt(rgt, oof.indices_kji),
        "pykrige_error": np.asarray(oof.baseline - oof.target, dtype=np.float64),
        "ked_error": np.asarray(ked_prediction - oof.target, dtype=np.float64),
    }
    return experiment, payload


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    experiment = result["experiment"]
    baseline = experiment["baseline"]["pykrige_ok3d_repeat_0"]["rmse"]
    ked = experiment["ked_metrics"]["rmse"]
    delta = experiment["rmse_delta_ked_minus_pykrige"]
    counts = experiment["independent_fold_outcome_counts"]
    ci = experiment["whole_fold_bootstrap"]["rmse_delta_ked_minus_pykrige"][
        "ci95"
    ]
    inference = result["rgt_inference"]
    lines = [
        "# P18 CIG-Bench RGT external-drift kriging",
        "",
        "## Result",
        "",
        f"- Committed PyKrige development OOF RMSE: `{baseline:.12f}`.",
        f"- CIG-Bench RGT KED development OOF RMSE: `{ked:.12f}`.",
        f"- KED minus PyKrige RMSE: `{delta:+.12f}` "
        f"(relative `{experiment['relative_rmse_change']:+.6%}`).",
        f"- Five genuinely independent spatial-fold outcomes: "
        f"`{counts['win']} win / {counts['loss']} loss / {counts['tie']} tie`; "
        "there are no random-seed pseudo-repeats.",
        f"- Whole-spatial-fold bootstrap 95% CI for KED minus PyKrige RMSE: "
        f"`[{ci[0]:+.12f}, {ci[1]:+.12f}]`.",
        f"- Decision: `{experiment['decision']['state']}`; the candidate remains "
        "disabled and the frozen holdout remains sealed.",
        "",
        "## RGT construction and geological rationale",
        "",
        "- The 20 non-overlapping development patches are reassembled into the "
        "real contiguous `63 x 100 x 72` K/J/I volume. Only "
        "`seismic_patch[0]` is sent to RGT-Est because it is seismic amplitude; "
        "channels 1/2 are already engineered local-RMS and vertical-gradient "
        "attributes, not independent seismic acquisitions.",
        "- RGT is aligned as `(T,H,W)=(K,J,I)`: K is the vertical/depositional "
        "sampling axis and J/I are the two lateral trace axes. The returned RGT "
        "is resized back to the exact development grid before KJI lookup.",
        f"- Upstream recommends `400 x 512 x 512`; this run used "
        f"`{inference['infer_shape']}`. Canonical-shape status is "
        f"`{inference['matches_upstream_recommended_shape']}`.",
        f"- The persisted canonical feasibility probe status is "
        f"`{inference['canonical_feasibility_probe']['status']}`. A fallback "
        "shape is allowed only after the same weight and real development "
        "volume produce a CUDA out-of-memory result; it is fixed from resource "
        "limits, never chosen by target RMSE.",
        "- UniversalKriging3D receives one specified RGT drift. Coordinates, "
        "512 labels/fold, linear variogram, `nlags=4`, and 2,048 validation "
        "rows/fold are unchanged from the OrdinaryKriging3D control.",
        "- The drift is fixed before reading any fold target and is never fitted "
        "to porosity. It expresses a mature KED prior: cells at similar relative "
        "geological time may share a trend even when Euclidean depth differs.",
        "",
        "## Control and provenance checks",
        "",
        f"- Drift-disabled OK3D recomputation max absolute difference from the "
        f"committed OOF archive: "
        f"`{experiment['baseline_recomputation']['max_abs_prediction_difference_vs_committed_oof']:.3e}`.",
        "- Installed `cig-bench=="
        f"{result['source']['cig_bench']['version']}`; "
        f"wheel metadata and official upstream declare MIT. Weight SHA-256: "
        f"`{result['source']['weight']['sha256']}`.",
        "- The installed wheel metadata homepage is a placeholder; provenance is "
        "therefore locked to the official `douyimin/CIG-bench` URL, imported "
        "source-file hashes, ModelScope model ID, and downloaded weight hash.",
        "",
        "## Fold breakdown",
        "",
        "| fold | PyKrige RMSE | RGT-KED RMSE | delta | outcome |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in experiment["per_fold"]:
        lines.append(
            f"| {row['fold_id']} | {row['pykrige_committed']['rmse']:.9f} | "
            f"{row['ked']['rmse']:.9f} | "
            f"{row['rmse_delta_ked_minus_pykrige']:+.9f} | "
            f"{row['outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "- These are development-only results from five spatial units. The "
            "20,000 bootstrap draws resample folds, not voxels, and do not create "
            "20,000 independent observations.",
            "- RGT quality has no ground-truth horizon labels in this dataset; "
            "the volume is a pretrained target-free structural prior, not a "
            "validated geological-age label.",
            "- PropertyPredictor was not run: its documented interface is "
            "conditional on a sparse property/well volume, so treating it as a "
            "standalone zero-shot porosity predictor would not be a fair or "
            "documented comparison. KED was the preregistered priority.",
            "- No claim is made that any metric change is caused by pretrained "
            "knowledge without a same-architecture random-init RGT ablation.",
            "- Only `train.h5`, audited P5 development caches, and development "
            "OOF artifacts were opened. No frozen evaluation surface was read.",
            "",
        ]
    )
    (output_dir / "evidence.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    weight_cache: Path,
    device: str = "cuda:0",
    infer_shape: Sequence[int] = CANONICAL_INFER_SHAPE,
    canonical_probe: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = base.resolve_dev_inputs(data_dir)
    base.ensure_no_holdout_paths([stage3_root, output_dir, weight_cache])
    package_audit = cigbench_rgt.verify_package()
    weight_path, weight_audit = cigbench_rgt.resolve_weight(cache_dir=weight_cache)
    requested_shape = tuple(int(value) for value in infer_shape)
    if requested_shape != CANONICAL_INFER_SHAPE:
        if canonical_probe is None:
            raise ValueError(
                "noncanonical RGT inference requires a persisted canonical probe"
            )
        canonical_probe = Path(canonical_probe).expanduser().resolve()
        base.ensure_no_holdout_paths([canonical_probe])
        canonical_probe_record = json.loads(
            canonical_probe.read_text(encoding="utf-8")
        )
        if (
            canonical_probe_record.get("status") != "CUDA_OUT_OF_MEMORY"
            or canonical_probe_record.get("weight_sha256")
            != weight_audit["sha256"]
        ):
            raise RuntimeError("canonical probe does not justify fallback inference")
    else:
        canonical_probe_record = {
            "status": "NOT_REQUIRED_CANONICAL_RUN",
            "infer_shape": list(CANONICAL_INFER_SHAPE),
        }
    oof = base.load_oof_development(stage3_root)
    folds, sample_audit = p17.load_fold_samples(
        stage3_root=stage3_root,
        train_h5=inputs.train_h5,
        oof=oof,
    )
    seismic, active, volume_audit = p14.assemble_seismic_volume(inputs.train_h5)
    rgt, seismic_used, rgt_audit = _run_rgt(
        seismic=seismic,
        weight_path=weight_path,
        weight_audit=weight_audit,
        device=device,
        infer_shape=infer_shape,
    )
    rgt_path = output_dir / "rgt_volume.npz"
    np.savez_compressed(
        rgt_path,
        rgt=np.asarray(rgt, dtype=np.float32),
        seismic_amplitude_used=np.asarray(seismic_used, dtype=np.float32),
        active_mask=np.asarray(active, dtype=np.uint8),
    )
    experiment, payload = _evaluate(
        stage3_root=stage3_root,
        oof=oof,
        folds=folds,
        rgt=rgt,
    )
    prediction_path = output_dir / "predictions.npz"
    np.savez_compressed(prediction_path, **payload)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED",
        "development_only": True,
        "source": {
            "cig_bench": package_audit,
            "weight": weight_audit,
            "pykrige": {
                "distribution": "pykrige",
                "version": importlib.metadata.version("pykrige"),
                "license": "BSD-3-Clause",
                "upstream_url": "https://github.com/GeoStat-Framework/PyKrige",
            },
        },
        "data": {
            "train_h5_sha256": base._sha256(inputs.train_h5),  # noqa: SLF001
            "volume_audit": volume_audit,
            "sample_audit": sample_audit,
            "source_oof_rows": int(len(oof.target)),
            "source_oof_records": list(oof.source_records),
        },
        "rgt_inference": {
            **rgt_audit,
            "canonical_feasibility_probe": canonical_probe_record,
            "quality": _rgt_quality(rgt, seismic[0]),
            "artifact": {
                "path": str(rgt_path.relative_to(PROJECT_ROOT)),
                "sha256": base._sha256(rgt_path),  # noqa: SLF001
                "bytes": rgt_path.stat().st_size,
            },
        },
        "experiment": experiment,
        "prediction_artifact": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "sha256": base._sha256(prediction_path),  # noqa: SLF001
            "bytes": prediction_path.stat().st_size,
            "rows": int(len(oof.target)),
        },
        "property_predictor": {
            "executed": False,
            "reason": (
                "documented predictor requires a sparse property/well volume; "
                "a standalone zero-shot comparison would be out of interface"
            ),
        },
        "holdout_firewall": {
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": [
                "seismic_patch[0:3]",
                "seismic_patch[3:6]",
                "seismic_patch[8]",
            ],
            "label_dataset_read_by_rgt": False,
            "test_path_argument_exists": False,
            "frozen_holdout_opened": False,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "device": device,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_evidence(output_dir, result)
    return result


def verify_evidence(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected P18 CIG-Bench KED schema")
    if summary["holdout_firewall"]["frozen_holdout_opened"]:
        raise RuntimeError("frozen holdout firewall failed")
    if summary["experiment"]["decision"]["default_enabled"]:
        raise RuntimeError("development-only KED candidate must remain disabled")
    prediction_record = summary["prediction_artifact"]
    prediction_path = PROJECT_ROOT / prediction_record["path"]
    if base._sha256(prediction_path) != prediction_record["sha256"]:  # noqa: SLF001
        raise RuntimeError("prediction artifact hash mismatch")
    rgt_record = summary["rgt_inference"]["artifact"]
    rgt_path = PROJECT_ROOT / rgt_record["path"]
    if base._sha256(rgt_path) != rgt_record["sha256"]:  # noqa: SLF001
        raise RuntimeError("RGT artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.float64)
        baseline = np.asarray(
            payload["pykrige_committed_prediction"], dtype=np.float64
        )
        recomputed = np.asarray(
            payload["pykrige_recomputed_prediction"], dtype=np.float64
        )
        candidate = np.asarray(payload["ked_prediction"], dtype=np.float64)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
        indices = np.asarray(payload["indices_kji"], dtype=np.int64)
    reported = summary["experiment"]
    baseline_metrics = _metrics(target, baseline)
    recomputed_metrics = _metrics(target, recomputed)
    candidate_metrics = _metrics(target, candidate)
    np.testing.assert_allclose(
        [baseline_metrics["rmse"], candidate_metrics["rmse"]],
        [
            reported["baseline"]["pykrige_ok3d_repeat_0"]["rmse"],
            reported["ked_metrics"]["rmse"],
        ],
        rtol=0.0,
        atol=1e-15,
    )
    fold_checks = []
    for row in reported["per_fold"]:
        mask = fold_ids == row["fold_id"]
        if int(np.sum(mask)) != 2048:
            raise RuntimeError("persisted fold row count changed")
        candidate_fold = _metrics(target[mask], candidate[mask])
        np.testing.assert_allclose(
            candidate_fold["rmse"], row["ked"]["rmse"], rtol=0.0, atol=1e-15
        )
        fold_checks.append(
            {
                "fold_id": row["fold_id"],
                "rows": int(np.sum(mask)),
                "outcome": row["outcome"],
            }
        )
    verification = {
        "schema_version": "reconstruction-p18-cigbench-ked-verification/v1",
        "status": "PASSED",
        "summary_sha256": base._sha256(summary_path),  # noqa: SLF001
        "prediction_sha256": prediction_record["sha256"],
        "rgt_sha256": rgt_record["sha256"],
        "rows": int(len(target)),
        "indices_kji_sha256": _array_sha256(indices),
        "baseline_metrics_recomputed": baseline_metrics,
        "baseline_control_metrics_recomputed": recomputed_metrics,
        "ked_metrics_recomputed": candidate_metrics,
        "fold_checks": fold_checks,
        "firewall_checks": {
            "independent_spatial_folds": len(np.unique(fold_ids)),
            "train_labels_per_fold": summary["data"]["sample_audit"][
                "train_labels_per_fold"
            ],
            "frozen_holdout_opened": False,
            "default_enabled": False,
        },
    }
    verification_path = output_dir / "verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_names = (
        "canonical_probe.json",
        "evidence.md",
        "predictions.npz",
        "rgt_volume.npz",
        "summary.json",
        "verification.json",
    )
    manifest = {
        name: base._sha256(output_dir / name)  # noqa: SLF001
        for name in manifest_names
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verification


def _shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != 3 or any(item <= 0 or item % 16 for item in parts):
        raise argparse.ArgumentTypeError(
            "infer shape must be three comma-separated positive multiples of 16"
        )
    return parts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-dev")
    check.add_argument("--data-dir", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument("--weight-cache", type=Path, required=True)
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument(
        "--infer-shape", type=_shape, default=CANONICAL_INFER_SHAPE
    )
    execute.add_argument("--canonical-probe", type=Path)
    probe = subparsers.add_parser("probe-canonical")
    probe.add_argument("--data-dir", type=Path, required=True)
    probe.add_argument("--weight-cache", type=Path, required=True)
    probe.add_argument("--output-path", type=Path, required=True)
    probe.add_argument("--device", default="cuda:0")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check-dev":
        inputs = base.resolve_dev_inputs(args.data_dir)
        print(json.dumps({"accepted": str(inputs.train_h5), "holdout_opened": False}))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_evidence(args.output_dir), sort_keys=True))
        return 0
    if args.command == "probe-canonical":
        values = vars(args)
        values.pop("command")
        print(json.dumps(probe_canonical(**values), sort_keys=True))
        return 0
    values = vars(args)
    values.pop("command")
    result = run(**values)
    print(json.dumps(result["experiment"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
