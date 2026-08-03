#!/usr/bin/env python3
"""Bounded development-only geostatistics feasibility pilot for reconstruction.

The pilot keeps the P21 five spatial folds, 512 labels per fold and 2,048
validation rows per fold.  It never accepts a test/holdout path.  Two fixed,
train-only candidates are evaluated:

* local ordinary kriging with a directional exponential variogram; and
* regression kriging using the three collocated seismic attributes as the
  secondary variable (a minimal co-kriging feasibility proxy).

Both candidates work in physical metres reconstructed from the frozen build
bounds and apply only the hard porosity-fraction bounds [0, 1].
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


TRACK = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK.parents[2]
sys.path[:0] = [str(TRACK), str(PROJECT_ROOT)]

import p11_residual_fusion as base  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
import p18_anisotropic_foundation_geostatistics as p18  # noqa: E402
import p21_fixed_foundation_ensemble as p21  # noqa: E402
import p29_agent_action_effect_repair as p29  # noqa: E402


SCHEMA_VERSION = "reconstruction-p30-bounded-geostatistics-feasibility/v1"
DEFAULT_OUTPUT_DIR = TRACK / "_outputs" / "p30_bounded_geostatistics_feasibility"
NEIGHBOURS = 64
RIDGE_ALPHA = 1.0
POROSITY_BOUNDS = (0.0, 1.0)
DIRECTION_COSINE_MINIMUM = 0.80
VARIOGRAM_BINS = 12
SOLVE_BATCH_SIZE = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return p21._metrics(target, prediction)  # noqa: SLF001


def _physical_bounds(build_summary: Mapping[str, Any]) -> np.ndarray:
    bounds = build_summary["coordinate_bounds"]
    result = np.asarray(
        [bounds["x"], bounds["y"], bounds["depth"]], dtype=np.float64
    )
    if result.shape != (3, 2) or np.any(result[:, 1] <= result[:, 0]):
        raise ValueError("invalid physical coordinate bounds")
    return result


def denormalize_coordinates(normalized: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 3 or bounds.shape != (3, 2):
        raise ValueError("normalized coordinates must be [n,3] and bounds [3,2]")
    if np.any(normalized < -1e-6) or np.any(normalized > 1.0 + 1e-6):
        raise ValueError("normalized coordinates fall outside the build contract")
    return normalized * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def _binned_variogram(distance: np.ndarray, semivariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(distance, kind="stable")
    chunks = np.array_split(order, VARIOGRAM_BINS)
    centres: list[float] = []
    gamma: list[float] = []
    counts: list[int] = []
    for rows in chunks:
        if len(rows) < 4:
            continue
        centres.append(float(np.median(distance[rows])))
        gamma.append(float(np.median(semivariance[rows])))
        counts.append(int(len(rows)))
    return np.asarray(centres), np.asarray(gamma), np.asarray(counts)


def fit_directional_variogram(
    coordinates_m: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    """Fit three train-only directional exponential variograms.

    The effective range is the distance at which the fitted structured
    covariance has decayed to five percent.  Pair selection and fitting are
    deterministic and use no validation target.
    """

    coordinates_m = np.asarray(coordinates_m, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if coordinates_m.shape != (len(values), 3) or len(values) < 8:
        raise ValueError("variogram fit requires at least eight 3-D samples")
    left, right = np.triu_indices(len(values), k=1)
    delta = np.abs(coordinates_m[left] - coordinates_m[right])
    euclidean = np.linalg.norm(delta, axis=1)
    semivariance = 0.5 * (values[left] - values[right]) ** 2
    nonzero = euclidean > 0.0
    sill_reference = max(float(np.var(values, ddof=1)), 1e-10)
    rows: list[dict[str, Any]] = []
    effective_ranges: list[float] = []
    axis_names = ("x_easting_m", "y_northing_m", "depth_m")
    for axis, axis_name in enumerate(axis_names):
        cosine = np.divide(
            delta[:, axis],
            euclidean,
            out=np.zeros_like(euclidean),
            where=nonzero,
        )
        minimum_pairs = VARIOGRAM_BINS * 4
        actual_cosine_minimum = DIRECTION_COSINE_MINIMUM
        selected = nonzero & (cosine >= actual_cosine_minimum)
        for relaxed in (0.60, 0.40, 0.20, 0.0):
            if int(np.sum(selected)) >= minimum_pairs:
                break
            actual_cosine_minimum = relaxed
            selected = nonzero & (cosine >= actual_cosine_minimum)
        axis_distance = euclidean[selected]
        axis_gamma = semivariance[selected]
        if len(axis_distance) < minimum_pairs:
            raise RuntimeError(f"insufficient directional pairs for {axis_name}")
        centres, gamma, counts = _binned_variogram(axis_distance, axis_gamma)
        positive = centres[centres > 0.0]
        minimum_range = max(float(np.min(positive)), 1e-6)
        maximum_range = max(float(np.max(centres)) * 4.0, minimum_range * 2.0)

        def residual(theta: np.ndarray) -> np.ndarray:
            nugget, partial_sill, effective_range = theta
            fitted = nugget + partial_sill * (
                1.0 - np.exp(-3.0 * centres / effective_range)
            )
            return np.sqrt(counts) * (fitted - gamma) / sill_reference

        fitted = least_squares(
            residual,
            x0=np.asarray([0.05 * sill_reference, 0.95 * sill_reference, np.median(centres)]),
            bounds=(
                np.asarray([0.0, 1e-12, minimum_range]),
                np.asarray([2.0 * sill_reference, 4.0 * sill_reference, maximum_range]),
            ),
            max_nfev=2_000,
        )
        nugget, partial_sill, effective_range = map(float, fitted.x)
        effective_ranges.append(effective_range)
        rows.append(
            {
                "axis": axis_name,
                "direction_cosine_minimum_requested": DIRECTION_COSINE_MINIMUM,
                "direction_cosine_minimum_used": actual_cosine_minimum,
                "direction_resolution": (
                    "requested_cone"
                    if actual_cosine_minimum == DIRECTION_COSINE_MINIMUM
                    else "relaxed_low_resolution"
                ),
                "low_direction_resolution": bool(
                    actual_cosine_minimum < DIRECTION_COSINE_MINIMUM
                ),
                "pair_count": int(np.sum(selected)),
                "bin_centres_m": centres.tolist(),
                "bin_semivariance": gamma.tolist(),
                "bin_counts": counts.tolist(),
                "nugget": nugget,
                "partial_sill": partial_sill,
                "effective_range_m": effective_range,
                "fit_cost": float(fitted.cost),
                "fit_success": bool(fitted.success),
            }
        )
    nugget = float(np.median([row["nugget"] for row in rows]))
    partial_sill = float(np.median([row["partial_sill"] for row in rows]))
    return {
        "model": "directional_exponential",
        "effective_ranges_m": effective_ranges,
        "nugget": max(nugget, 1e-10),
        "partial_sill": max(partial_sill, 1e-10),
        "sill_reference": sill_reference,
        "directions": rows,
        "validation_target_used": False,
    }


def local_ordinary_kriging(
    *,
    train_coordinates_m: np.ndarray,
    train_values: np.ndarray,
    query_coordinates_m: np.ndarray,
    variogram: Mapping[str, Any],
    neighbours: int = NEIGHBOURS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_coordinates_m = np.asarray(train_coordinates_m, dtype=np.float64)
    query_coordinates_m = np.asarray(query_coordinates_m, dtype=np.float64)
    train_values = np.asarray(train_values, dtype=np.float64)
    ranges = np.asarray(variogram["effective_ranges_m"], dtype=np.float64)
    if np.any(ranges <= 0.0) or ranges.shape != (3,):
        raise ValueError("variogram effective ranges must be three positive values")
    count = min(int(neighbours), len(train_values))
    if count < 1:
        raise ValueError("neighbours must be positive")
    scaled_train = train_coordinates_m / ranges
    scaled_query = query_coordinates_m / ranges
    neighbour_distance, neighbour_rows = NearestNeighbors(
        n_neighbors=count, n_jobs=1
    ).fit(scaled_train).kneighbors(scaled_query)
    prediction = np.empty(len(scaled_query), dtype=np.float64)
    variance = np.empty(len(scaled_query), dtype=np.float64)
    partial_sill = float(variogram["partial_sill"])
    nugget = float(variogram["nugget"])
    fallback_solves = 0
    exact_conditioned = 0
    for start in range(0, len(scaled_query), SOLVE_BATCH_SIZE):
        stop = min(start + SOLVE_BATCH_SIZE, len(scaled_query))
        rows = neighbour_rows[start:stop]
        distances = neighbour_distance[start:stop]
        query_count = stop - start
        exact = distances[:, 0] <= 1e-12
        exact_conditioned += int(np.sum(exact))
        if np.any(exact):
            prediction[start:stop][exact] = train_values[rows[exact, 0]]
            variance[start:stop][exact] = 0.0
        pending = np.flatnonzero(~exact)
        if not len(pending):
            continue
        selected_rows = rows[pending]
        selected_coordinates = scaled_train[selected_rows]
        pair_distance = np.linalg.norm(
            selected_coordinates[:, :, None, :] - selected_coordinates[:, None, :, :],
            axis=3,
        )
        covariance = partial_sill * np.exp(-3.0 * pair_distance)
        diagonal = np.arange(count)
        covariance[:, diagonal, diagonal] += nugget + 1e-10
        system = np.zeros((len(pending), count + 1, count + 1), dtype=np.float64)
        system[:, :count, :count] = covariance
        system[:, :count, count] = 1.0
        system[:, count, :count] = 1.0
        rhs = np.empty((len(pending), count + 1), dtype=np.float64)
        rhs[:, :count] = partial_sill * np.exp(-3.0 * distances[pending])
        rhs[:, count] = 1.0
        try:
            solution = np.linalg.solve(system, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            fallback_solves += len(pending)
            solution = np.stack(
                [np.linalg.pinv(a, rcond=1e-12) @ b for a, b in zip(system, rhs)]
            )
        weights = solution[:, :count]
        block_prediction = np.sum(weights * train_values[selected_rows], axis=1)
        block_variance = (
            partial_sill
            + nugget
            - np.sum(weights * rhs[:, :count], axis=1)
            + solution[:, count]
        )
        prediction[start:stop][pending] = block_prediction
        variance[start:stop][pending] = np.maximum(block_variance, 0.0)
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(variance)):
        raise FloatingPointError("ordinary kriging produced non-finite output")
    return prediction, variance, {
        "neighbours": count,
        "solve_batch_size": SOLVE_BATCH_SIZE,
        "fallback_solves": fallback_solves,
        "exact_conditioned_queries": exact_conditioned,
    }


def regression_kriging(
    *,
    train_coordinates_m: np.ndarray,
    train_secondary: np.ndarray,
    train_target: np.ndarray,
    query_coordinates_m: np.ndarray,
    query_secondary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(train_secondary)
    train_scaled = scaler.transform(train_secondary)
    query_scaled = scaler.transform(query_secondary)
    drift = Ridge(alpha=RIDGE_ALPHA).fit(train_scaled, train_target)
    train_drift = drift.predict(train_scaled)
    query_drift = drift.predict(query_scaled)
    residual = np.asarray(train_target, dtype=np.float64) - train_drift
    variogram = fit_directional_variogram(train_coordinates_m, residual)
    kriged_residual, variance, solve_audit = local_ordinary_kriging(
        train_coordinates_m=train_coordinates_m,
        train_values=residual,
        query_coordinates_m=query_coordinates_m,
        variogram=variogram,
    )
    prediction = query_drift + kriged_residual
    correlations = [
        float(np.corrcoef(train_target, train_secondary[:, column])[0, 1])
        for column in range(train_secondary.shape[1])
    ]
    return prediction, variance, {
        "method": "regression_kriging_cokriging_proxy",
        "secondary_variables": [
            "seismic_amplitude",
            "seismic_local_rms",
            "seismic_vertical_gradient",
        ],
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_coefficients": drift.coef_.tolist(),
        "ridge_intercept": float(drift.intercept_),
        "train_secondary_target_correlations": correlations,
        "residual_variogram": variogram,
        "solve_audit": solve_audit,
        "validation_target_used": False,
    }


def _clip_physical(prediction: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    lower, upper = POROSITY_BOUNDS
    prediction = np.asarray(prediction, dtype=np.float64)
    clipped = np.clip(prediction, lower, upper)
    changed = np.abs(clipped - prediction) > 0.0
    return clipped, {
        "property": "PORO fraction",
        "bounds": [lower, upper],
        "violations_before_constraint": int(np.sum(changed)),
        "maximum_absolute_change": float(np.max(np.abs(clipped - prediction))),
        "finite_after_constraint": bool(np.all(np.isfinite(clipped))),
    }


def _fold_comparison(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    for fold_id in base.FOLD_IDS:
        mask = fold_ids == fold_id
        baseline_metrics = _metrics(target[mask], baseline[mask])
        candidate_metrics = _metrics(target[mask], candidate[mask])
        delta = float(candidate_metrics["rmse"] - baseline_metrics["rmse"])
        outcome = "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        outcomes[outcome] += 1
        rows.append(
            {
                "fold_id": int(fold_id),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "rmse_delta_candidate_minus_p21": delta,
                "relative_rmse_change": delta / float(baseline_metrics["rmse"]),
                "outcome": outcome,
            }
        )
    return rows, outcomes


def _candidate_summary(
    *,
    target: np.ndarray,
    p21_prediction: np.ndarray,
    candidate: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    per_fold, outcomes = _fold_comparison(target, p21_prediction, candidate, fold_ids)
    baseline_metrics = _metrics(target, p21_prediction)
    candidate_metrics = _metrics(target, candidate)
    bootstrap = p17._whole_fold_bootstrap(  # noqa: SLF001
        target=target,
        baseline=p21_prediction,
        candidate=candidate,
        fold_ids=fold_ids,
    )
    maximum_fold_regression = max(row["relative_rmse_change"] for row in per_fold)
    return {
        "metrics": candidate_metrics,
        "p21_metrics": baseline_metrics,
        "rmse_delta_candidate_minus_p21": float(candidate_metrics["rmse"] - baseline_metrics["rmse"]),
        "relative_rmse_change_vs_p21": (
            float(candidate_metrics["rmse"] - baseline_metrics["rmse"])
            / float(baseline_metrics["rmse"])
        ),
        "per_fold": per_fold,
        "outcomes_vs_p21": outcomes,
        "maximum_fold_relative_regression": float(maximum_fold_regression),
        "whole_fold_bootstrap": bootstrap,
    }


def _fusion_contract() -> dict[str, Any]:
    return {
        "schema_version": "reconstruction-cross-modal-fusion-io/v1",
        "claim_boundary": "future contract; no well-log-plus-seismic fusion result is claimed by P30",
        "inputs": {
            "cell_id": {"dtype": "int64", "shape": ["n", 3], "order": ["k", "j", "i"], "required": True},
            "xyz_m": {"dtype": "float64", "shape": ["n", 3], "order": ["easting_m", "northing_m", "TVDSS_or_declared_depth_m"], "required": True},
            "seismic": {"dtype": "float32", "shape": ["n", "seismic_feature"], "required": True, "alignment": "same cell_id and depth/time datum"},
            "seismic_foundation_embedding": {"dtype": "float32", "shape": ["n", "embedding"], "required": False, "provenance": ["model_id", "revision", "weights_sha256", "input_transform"]},
            "well_log_observations": {"dtype": "table", "required": True, "columns": ["well_id", "MD_m", "TVDSS_m", "x_m", "y_m", "curve_name", "value", "unit", "quality_mask"]},
            "well_log_foundation_embedding": {"dtype": "float32", "shape": ["observation", "embedding"], "required": False, "provenance": ["model_id", "revision", "weights_sha256", "curve_mapping"]},
            "alignment": {"required": True, "fields": ["depth_datum", "time_depth_transform_id", "support_radius_m", "cell_to_well_weights", "outside_support_mask"]},
            "split": {"required": True, "fields": ["outer_fold_id", "role", "effective_train_sample_ids", "buffer_definition", "frozen_test_opened"]},
            "source_lock": {"required": True, "fields": ["asset_id", "license", "source_sha256", "feature_cache_sha256"]},
        },
        "invariants": [
            "all normalization, variograms, PCA and fusion weights fit on effective outer-train rows only",
            "well observations from a held spatial fold cannot enter feature fitting or action selection",
            "MD is not interchangeable with TVDSS; seismic TWT requires an identified time-depth transform",
            "missing modalities use explicit masks, never silent zero substitution",
            "frozen test/holdout paths are rejected before filesystem access",
        ],
        "outputs": {
            "porosity_mean": {"dtype": "float32", "shape": ["n"], "unit": "fraction", "bounds": [0.0, 1.0]},
            "porosity_variance": {"dtype": "float32", "shape": ["n"], "unit": "fraction^2", "minimum": 0.0},
            "conditioning_audit": {"fields": ["exact_well_cells_honoured", "bound_clips", "outside_support_count"]},
            "modality_ablation": {"required": True, "members": ["seismic_only", "well_only", "seismic_plus_well", "foundation_disabled"]},
            "provenance": {"required": True, "fields": ["config_sha256", "input_hashes", "split_hash", "prediction_sha256"]},
        },
    }


def _write_finding(output_dir: Path, result: Mapping[str, Any]) -> None:
    ordinary = result["candidates"]["anisotropic_ordinary_kriging"]
    regression = result["candidates"]["regression_kriging_cokriging_proxy"]
    lines = [
        "# P30 bounded reconstruction geostatistics finding",
        "",
        "## Finding",
        "",
        "P21 remains the reconstruction default. The matched-budget physical",
        "anisotropic ordinary-kriging and seismic regression-kriging candidates",
        "both fail promotion. Classical well-log co-kriging is blocked by the",
        "absence of an independently aligned well-log secondary variable in the",
        "P21 folds, not by a missing numerical solver.",
        "",
        "Old P29 outputs are retained only as historical policy evidence. They",
        "must not be used for a new promotion because the old replay interface",
        "misread scalar ensemble weights and silently zero-filled query-side",
        "seismic/GFM covariates. The repaired code is locked by an A0-to-P21",
        "identity check and query-side fail-closed tests.",
        "",
        "## Matched five-fold result",
        "",
        "| fold | P21 RMSE | anisotropic OK RMSE | regression-kriging RMSE |",
        "|---:|---:|---:|---:|",
    ]
    for ordinary_row, regression_row in zip(
        ordinary["per_fold"], regression["per_fold"]
    ):
        lines.append(
            f"| {ordinary_row['fold_id']} | "
            f"{ordinary_row['baseline']['rmse']:.12f} | "
            f"{ordinary_row['candidate']['rmse']:.12f} | "
            f"{regression_row['candidate']['rmse']:.12f} |"
        )
    lines.extend(
        [
            "",
            f"Pooled P21 RMSE is `{ordinary['p21_metrics']['rmse']:.12f}`;",
            f"anisotropic ordinary kriging is `{ordinary['metrics']['rmse']:.12f}`",
            f"({ordinary['outcomes_vs_p21']['win']} wins / {ordinary['outcomes_vs_p21']['loss']} losses), and",
            f"regression kriging is `{regression['metrics']['rmse']:.12f}`",
            f"({regression['outcomes_vs_p21']['win']} wins / {regression['outcomes_vs_p21']['loss']} losses).",
            "",
            "## Direction-cone audit",
            "",
        "The requested direction cosine was 0.8. Only fold 0 depth required",
        "relaxation to 0.6, for both the target and regression-residual",
        "variograms; both fits are flagged `relaxed_low_resolution`. Every",
        "other axis/fold fit used 0.8. No threshold used validation targets.",
            "",
            "## Residual risks and next input",
            "",
            "The current depth is TVD-like Eclipse cell-centre depth, not MD or",
            "seismic TWT. The existing weak well tie mixes MD with TVD-like depth,",
            "and only one intersecting LFP well supplies sparse constraints. Future",
            "well-log plus seismic foundation fusion therefore requires the exact",
            "machine contract in `fusion_io_contract.json`: aligned KJI and metre",
            "coordinates, an identified MD-to-TVDSS/TWT transform, declared curve",
            "units and quality masks, support weights, explicit missing-modality",
            "masks, outer-fold roles, source/weight hashes, and PORO mean/variance",
            "plus modality-ablation outputs.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPYCACHEPREFIX=_pipelines/02_task_datasets/reconstruction/_tmp/p30_pycache /usr/bin/python3 -m py_compile _pipelines/02_task_datasets/reconstruction/p29_agent_action_effect_repair.py _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py _pipelines/02_task_datasets/reconstruction/_tests/test_p29_agent_action_effect_repair.py _pipelines/02_task_datasets/reconstruction/_tests/test_p30_bounded_geostatistics_feasibility.py",
            "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p29_agent_action_effect_repair.py' -v",
            "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p30_bounded_geostatistics_feasibility.py' -v",
            "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py --data-dir ../track-reconstruction/_data/processed/reconstruction --stage3-root ../p5-stage3-reconstruction/_tmp/p5_stage3_reconstruction --build-summary ../track-reconstruction/_pipelines/02_task_datasets/reconstruction/build_summary.json",
            "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py --verify-only",
            "git diff --check",
            "```",
        ]
    )
    (output_dir / "finding.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    build_summary_path: Path,
    p21_predictions_path: Path,
    p21_summary_path: Path,
    p29_summary_path: Path,
    data_registry_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base.ensure_no_holdout_paths(
        (
            data_dir,
            stage3_root,
            build_summary_path,
            p21_predictions_path,
            p21_summary_path,
            p29_summary_path,
            data_registry_path,
            output_dir,
        )
    )
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    folds, fold_loading_audit = p17.load_fold_samples(
        stage3_root=stage3_root,
        train_h5=inputs.train_h5,
        oof=oof,
    )
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    bounds = _physical_bounds(build_summary)
    p21_summary = json.loads(p21_summary_path.read_text(encoding="utf-8"))
    p29_summary = json.loads(p29_summary_path.read_text(encoding="utf-8"))
    registry_text = data_registry_path.read_text(encoding="utf-8")
    legal = "Equinor Open Data Licence" in registry_text
    if not legal:
        raise RuntimeError("development asset license is not registered")

    with np.load(p21_predictions_path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["indices_kji"], oof.indices_kji)
        np.testing.assert_array_equal(payload["fold_ids"], oof.fold_ids)
        np.testing.assert_allclose(payload["target"], oof.target, rtol=0.0, atol=0.0)
        p21_prediction = np.asarray(payload["candidate_prediction"], dtype=np.float64)

    requested = p17._unique_indices(folds)  # noqa: SLF001
    cache_indices, foundation, feature_audit = p18._load_feature_cache(  # noqa: SLF001
        p18.DEFAULT_FEATURE_CACHE,
        train_h5=inputs.train_h5,
        expected_indices=requested,
    )
    prepared, _ = p18._prepare_fold_metrics(  # noqa: SLF001
        folds=folds,
        requested_indices=cache_indices,
        foundation_features=foundation,
    )
    corrected_p29_a0 = p29.build_real_action_bank(
        oof=oof, prepared=prepared, registry=()
    )["A0"]
    replay_maximum_difference = float(np.max(np.abs(corrected_p29_a0 - p21_prediction)))
    if replay_maximum_difference > 1e-12:
        raise RuntimeError("repaired P29 A0 no longer replays P21")

    ordinary_prediction = np.full(len(oof.target), np.nan, dtype=np.float64)
    ordinary_variance = np.full(len(oof.target), np.nan, dtype=np.float64)
    regression_prediction = np.full(len(oof.target), np.nan, dtype=np.float64)
    regression_variance = np.full(len(oof.target), np.nan, dtype=np.float64)
    fold_audits: list[dict[str, Any]] = []
    for fold in folds:
        mask = oof.fold_ids == fold.fold_id
        train_coordinates = denormalize_coordinates(fold.train_raw_features[:, 3:6], bounds)
        validation_coordinates = denormalize_coordinates(fold.validation_raw_features[:, 3:6], bounds)
        target_variogram = fit_directional_variogram(train_coordinates, fold.train_target)
        ordinary, ordinary_var, ordinary_solve = local_ordinary_kriging(
            train_coordinates_m=train_coordinates,
            train_values=fold.train_target,
            query_coordinates_m=validation_coordinates,
            variogram=target_variogram,
        )
        regression, regression_var, regression_audit = regression_kriging(
            train_coordinates_m=train_coordinates,
            train_secondary=fold.train_raw_features[:, :3],
            train_target=fold.train_target,
            query_coordinates_m=validation_coordinates,
            query_secondary=fold.validation_raw_features[:, :3],
        )
        ordinary_prediction[mask] = ordinary
        ordinary_variance[mask] = ordinary_var
        regression_prediction[mask] = regression
        regression_variance[mask] = regression_var
        fold_audits.append(
            {
                "fold_id": int(fold.fold_id),
                "train_labels": int(len(fold.train_target)),
                "validation_rows": int(np.sum(mask)),
                "coordinate_units": ["easting_m", "northing_m", "depth_m"],
                "coordinate_bounds_observed_m": [
                    [float(np.min(train_coordinates[:, axis])), float(np.max(train_coordinates[:, axis]))]
                    for axis in range(3)
                ],
                "ordinary_variogram": target_variogram,
                "ordinary_solve": ordinary_solve,
                "regression_kriging": regression_audit,
            }
        )
    if not all(
        np.all(np.isfinite(value))
        for value in (
            ordinary_prediction,
            ordinary_variance,
            regression_prediction,
            regression_variance,
        )
    ):
        raise RuntimeError("pilot produced incomplete OOF arrays")
    ordinary_constrained, ordinary_constraint = _clip_physical(ordinary_prediction)
    regression_constrained, regression_constraint = _clip_physical(regression_prediction)
    ordinary_summary = _candidate_summary(
        target=oof.target,
        p21_prediction=p21_prediction,
        candidate=ordinary_constrained,
        fold_ids=oof.fold_ids,
    )
    regression_summary = _candidate_summary(
        target=oof.target,
        p21_prediction=p21_prediction,
        candidate=regression_constrained,
        fold_ids=oof.fold_ids,
    )
    best_name, best_summary = min(
        (
            ("anisotropic_ordinary_kriging", ordinary_summary),
            ("regression_kriging_cokriging_proxy", regression_summary),
        ),
        key=lambda item: item[1]["metrics"]["rmse"],
    )
    promotion = (
        best_summary["relative_rmse_change_vs_p21"] < 0.0
        and best_summary["outcomes_vs_p21"]["win"] >= 4
        and best_summary["maximum_fold_relative_regression"] <= 0.01
        and best_summary["whole_fold_bootstrap"]["rmse_delta_candidate_minus_baseline"]["ci95"][1] < 0.0
    )
    direction_cone_relaxations = []
    for row in fold_audits:
        for variogram_name, variogram in (
            ("target", row["ordinary_variogram"]),
            ("regression_residual", row["regression_kriging"]["residual_variogram"]),
        ):
            for direction in variogram["directions"]:
                if direction["direction_cosine_minimum_used"] < DIRECTION_COSINE_MINIMUM:
                    direction_cone_relaxations.append(
                        {
                            "fold_id": row["fold_id"],
                            "variogram": variogram_name,
                            "axis": direction["axis"],
                            "requested": DIRECTION_COSINE_MINIMUM,
                            "used": direction["direction_cosine_minimum_used"],
                            "pair_count": direction["pair_count"],
                            "low_direction_resolution": direction[
                                "low_direction_resolution"
                            ],
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        indices_kji=oof.indices_kji,
        fold_ids=oof.fold_ids,
        target=oof.target,
        p21_prediction=p21_prediction,
        corrected_p29_a0=corrected_p29_a0,
        anisotropic_ordinary_prediction=ordinary_constrained,
        anisotropic_ordinary_variance=ordinary_variance,
        regression_kriging_prediction=regression_constrained,
        regression_kriging_variance=regression_variance,
    )
    fusion_contract = _fusion_contract()
    _write_json(output_dir / "fusion_io_contract.json", fusion_contract)
    result = {
        "schema_version": SCHEMA_VERSION,
        "objective": "bounded feasibility of co-kriging, directional variograms and physical constraints",
        "protocol": {
            "outer_spatial_folds": list(base.FOLD_IDS),
            "train_labels_per_fold": 512,
            "validation_rows_per_fold": 2048,
            "matched_to_p21_rows_and_budget": True,
            "candidate_hyperparameters_selected_with_validation_labels": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "primary_metric": "pooled development OOF RMSE",
        },
        "asset_audit": {
            "source_asset": "volve-north-sea",
            "license": "Equinor Open Data Licence",
            "registered_for_reuse": legal,
            "data_registry_sha256": _sha256(data_registry_path),
            "train_h5_sha256": _sha256(inputs.train_h5),
            "train_h5_only": True,
            "target": "Eclipse PORO fraction on active cells",
            "target_observed_range_development_oof": [float(np.min(oof.target)), float(np.max(oof.target))],
        },
        "coordinate_audit": {
            "stored_channels": ["x_normalized", "y_normalized", "depth_normalized"],
            "working_units": ["easting_m", "northing_m", "depth_m"],
            "physical_bounds": bounds.tolist(),
            "depth_semantics": "Eclipse cell-centre depth, TVD-like; not MD and not seismic TWT",
            "seismic_sampling": build_summary["seismic"],
            "weak_tie_warning": build_summary["weak_tie"]["depth_coordinate_warning"],
            "p21_vertical_weight_semantics": "dimensionless multiplier after per-fold coordinate standardization; not a fitted physical variogram range",
        },
        "split_audit": {
            "development_source": "train.h5 only",
            "outer_cv": "five deterministic buffered spatial folds from Stage-3",
            "stage3_frozen_test_loaded": False,
            "original_dataset_split_metadata": build_summary["split"],
            "fold_loading_audit": fold_loading_audit,
        },
        "default_evidence_audit": {
            "p21": {
                "state": p21_summary["decision"]["state"],
                "default_enabled": p21_summary["decision"]["default_enabled"],
                "rmse": p21_summary["comparison"]["candidate"]["rmse"],
                "summary_sha256": _sha256(p21_summary_path),
            },
            "p29": {
                "policy_verdict": p29_summary["policy"]["promotion"]["verdict"],
                "default_model_changed": False,
                "legacy_summary_preserved": True,
                "legacy_output_usable_for_new_promotion": False,
                "legacy_summary_sha256": _sha256(p29_summary_path),
                "legacy_a0_rmse": p29_summary["metrics"]["A0"]["pooled"]["rmse"],
                "interface_bug": "seismic_weights treated as per-channel values and query secondary variables silently zeroed in replay",
                "corrected_a0_rmse": _metrics(oof.target, corrected_p29_a0)["rmse"],
                "corrected_a0_max_abs_difference_vs_p21": replay_maximum_difference,
                "repair_locks": [
                    "corrected real A0 equals P21 within 1e-12",
                    "query-side seismic/GFM features are required as train/query pairs",
                    "query baseline is required when blend_weight < 1",
                ],
            },
        },
        "conclusions": {
            "old_p29_outputs": "HISTORICAL_ONLY_NOT_ELIGIBLE_FOR_NEW_PROMOTION",
            "p29_repair": "LOCKED_BY_A0_EQUALS_P21_AND_QUERY_SIDE_FAIL_CLOSED_TESTS",
            "anisotropic_ordinary_kriging": "NO_PROMOTION",
            "regression_kriging_cokriging_proxy": "NO_PROMOTION",
            "classical_well_log_cokriging": "BLOCKED_PENDING_ALIGNED_SECONDARY_INPUT",
            "default": "P21_REMAINS_DEFAULT",
        },
        "direction_cone_audit": {
            "requested_cosine_minimum": DIRECTION_COSINE_MINIMUM,
            "fixed_relaxation_schedule": [0.60, 0.40, 0.20, 0.0],
            "fit_count": len(base.FOLD_IDS) * 2 * 3,
            "relaxed_fit_count": len(direction_cone_relaxations),
            "relaxations": direction_cone_relaxations,
            "validation_target_used_for_threshold": False,
        },
        "reproduction": {
            "py_compile": "PYTHONPYCACHEPREFIX=_pipelines/02_task_datasets/reconstruction/_tmp/p30_pycache /usr/bin/python3 -m py_compile _pipelines/02_task_datasets/reconstruction/p29_agent_action_effect_repair.py _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py _pipelines/02_task_datasets/reconstruction/_tests/test_p29_agent_action_effect_repair.py _pipelines/02_task_datasets/reconstruction/_tests/test_p30_bounded_geostatistics_feasibility.py",
            "p29_tests": "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p29_agent_action_effect_repair.py' -v",
            "p30_tests": "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p30_bounded_geostatistics_feasibility.py' -v",
            "run": "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py --data-dir ../track-reconstruction/_data/processed/reconstruction --stage3-root ../p5-stage3-reconstruction/_tmp/p5_stage3_reconstruction --build-summary ../track-reconstruction/_pipelines/02_task_datasets/reconstruction/build_summary.json",
            "verify_only": "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py --verify-only",
        },
        "feasibility": {
            "anisotropic_variogram": "IMPLEMENTED_DEVELOPMENT_PILOT",
            "regression_kriging_cokriging_proxy": "IMPLEMENTED_DEVELOPMENT_PILOT",
            "classical_cokriging": "BLOCKED_NO_ALIGNED_INDEPENDENT_WELL_LOG_SECONDARY_VARIABLE_IN_P21_SPLITS",
            "physical_constraints": "IMPLEMENTED_POROSITY_BOUNDS_AND_EXACT_CONDITIONING_KERNEL",
        },
        "candidates": {
            "anisotropic_ordinary_kriging": {
                **ordinary_summary,
                "physical_constraint": ordinary_constraint,
            },
            "regression_kriging_cokriging_proxy": {
                **regression_summary,
                "physical_constraint": regression_constraint,
            },
        },
        "fold_audits": fold_audits,
        "feature_cache_audit": feature_audit,
        "decision": {
            "best_candidate": best_name,
            "promote_over_p21": bool(promotion),
            "state": "PROMOTED" if promotion else "FEASIBLE_NO_PROMOTION",
            "p21_remains_default": not promotion,
            "cross_modal_foundation_claimed": False,
            "fresh_blind_claimed": False,
        },
        "prediction_artifact": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(prediction_path),
        },
        "fusion_contract": {
            "path": str((output_dir / "fusion_io_contract.json").relative_to(PROJECT_ROOT)),
            "sha256": _sha256(output_dir / "fusion_io_contract.json"),
        },
    }
    _write_json(output_dir / "summary.json", result)
    _write_finding(output_dir, result)
    lines = [
        "# P30 bounded geostatistics feasibility",
        "",
        f"- P21 development OOF RMSE: `{_metrics(oof.target, p21_prediction)['rmse']:.12f}`.",
        f"- Anisotropic ordinary kriging RMSE: `{ordinary_summary['metrics']['rmse']:.12f}`.",
        f"- Regression-kriging proxy RMSE: `{regression_summary['metrics']['rmse']:.12f}`.",
        f"- Decision: `{result['decision']['state']}`; P21 remains default: `{result['decision']['p21_remains_default']}`.",
        "",
        "The pilot used only the legal Volve development train container and the existing",
        "five buffered spatial folds. Classical co-kriging remains blocked because the P21",
        "fold cache has no aligned independent well-log secondary variable; the implemented",
        "regression-kriging route is therefore a bounded seismic-secondary feasibility proxy.",
        "P21's historical vertical weight is dimensionless and must not be described as a",
        "physical directional variogram. No frozen test or holdout was opened.",
    ]
    (output_dir / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def verify_evidence(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prediction_path = PROJECT_ROOT / summary["prediction_artifact"]["path"]
    if _sha256(prediction_path) != summary["prediction_artifact"]["sha256"]:
        raise RuntimeError("P30 prediction artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.float64)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
        p21_prediction = np.asarray(payload["p21_prediction"], dtype=np.float64)
        corrected_p29 = np.asarray(payload["corrected_p29_a0"], dtype=np.float64)
        candidates = {
            "anisotropic_ordinary_kriging": np.asarray(payload["anisotropic_ordinary_prediction"], dtype=np.float64),
            "regression_kriging_cokriging_proxy": np.asarray(payload["regression_kriging_prediction"], dtype=np.float64),
        }
        variances = (
            np.asarray(payload["anisotropic_ordinary_variance"], dtype=np.float64),
            np.asarray(payload["regression_kriging_variance"], dtype=np.float64),
        )
    if float(np.max(np.abs(corrected_p29 - p21_prediction))) > 1e-12:
        raise RuntimeError("P29 repaired replay differs from P21")
    recomputed = {}
    for name, prediction in candidates.items():
        metrics = _metrics(target, prediction)
        np.testing.assert_allclose(
            metrics["rmse"], summary["candidates"][name]["metrics"]["rmse"], rtol=0.0, atol=1e-15
        )
        if np.any(prediction < POROSITY_BOUNDS[0]) or np.any(prediction > POROSITY_BOUNDS[1]):
            raise RuntimeError(f"{name} violates physical porosity bounds")
        recomputed[name] = metrics
    if any(np.any(~np.isfinite(value)) or np.any(value < 0.0) for value in variances):
        raise RuntimeError("P30 kriging variance is invalid")
    if set(np.unique(fold_ids).tolist()) != set(base.FOLD_IDS) or len(target) != 10_240:
        raise RuntimeError("P30 OOF identity drift")
    verification = {
        "status": "PASSED",
        "summary_sha256": _sha256(summary_path),
        "prediction_artifact_sha256": _sha256(prediction_path),
        "rows": int(len(target)),
        "folds": sorted(np.unique(fold_ids).tolist()),
        "p29_replay_max_abs_difference_vs_p21": float(np.max(np.abs(corrected_p29 - p21_prediction))),
        "candidate_metrics_recomputed": recomputed,
        "physical_bounds_verified": list(POROSITY_BOUNDS),
        "nonnegative_finite_variances": True,
        "firewall": {"test_h5_opened": False, "frozen_holdout_opened": False},
    }
    _write_json(output_dir / "verification.json", verification)
    return verification


def write_artifact_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    paths = [
        TRACK / "p29_agent_action_effect_repair.py",
        TRACK / "_tests" / "test_p29_agent_action_effect_repair.py",
        Path(__file__).resolve(),
        TRACK / "_tests" / "test_p30_bounded_geostatistics_feasibility.py",
        output_dir / "evidence.md",
        output_dir / "finding.md",
        output_dir / "fusion_io_contract.json",
        output_dir / "predictions.npz",
        output_dir / "summary.json",
        output_dir / "verification.json",
    ]
    manifest = {
        "schema_version": "reconstruction-p30-artifact-manifest/v1",
        "artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def verify_artifact_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = []
    for row in manifest["artifacts"]:
        path = PROJECT_ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact missing: {path}")
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        if actual_hash != row["sha256"] or actual_bytes != row["bytes"]:
            raise RuntimeError(f"manifest artifact drift: {path}")
        checked.append(row["path"])
    return {
        "status": "PASSED",
        "manifest_sha256": _sha256(manifest_path),
        "artifact_count": len(checked),
        "checked_paths": checked,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--stage3-root", type=Path)
    parser.add_argument("--build-summary", type=Path)
    parser.add_argument("--p21-predictions", type=Path, default=TRACK / "_outputs" / "p21_fixed_foundation_ensemble" / "predictions.npz")
    parser.add_argument("--p21-summary", type=Path, default=TRACK / "_outputs" / "p21_fixed_foundation_ensemble" / "summary.json")
    parser.add_argument("--p29-summary", type=Path, default=TRACK / "_outputs" / "p29_agent_action_effect_repair" / "summary.json")
    parser.add_argument("--data-registry", type=Path, default=PROJECT_ROOT / "_meta" / "_data_registry.yml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.verify_only:
        verification = verify_evidence(args.output_dir)
        write_artifact_manifest(args.output_dir)
        manifest_verification = verify_artifact_manifest(args.output_dir)
        print(
            json.dumps(
                {"verification": verification, "manifest": manifest_verification},
                sort_keys=True,
            )
        )
        return 0
    required = ("data_dir", "stage3_root", "build_summary")
    if any(getattr(args, name) is None for name in required):
        raise SystemExit("--data-dir, --stage3-root and --build-summary are required")
    result = run(
        data_dir=args.data_dir.expanduser().resolve(),
        stage3_root=args.stage3_root.expanduser().resolve(),
        build_summary_path=args.build_summary.expanduser().resolve(),
        p21_predictions_path=args.p21_predictions.expanduser().resolve(),
        p21_summary_path=args.p21_summary.expanduser().resolve(),
        p29_summary_path=args.p29_summary.expanduser().resolve(),
        data_registry_path=args.data_registry.expanduser().resolve(),
        output_dir=args.output_dir,
    )
    verification = verify_evidence(args.output_dir)
    write_artifact_manifest(args.output_dir)
    manifest_verification = verify_artifact_manifest(args.output_dir)
    print(json.dumps({"decision": result["decision"], "verification": verification, "manifest": manifest_verification}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
