#!/usr/bin/env python3
"""P17 foundation-informed nonstationary geostatistics for reconstruction.

The frozen seismic foundation model is not asked to predict porosity.  Its
trace representations instead deform the neighbourhood metric used by a
local kernel interpolator.  Each outer fold uses exactly the committed 512
``point_train`` labels and the unchanged 2,048-row PyKrige validation archive.
All PCA/scaling operations are fitted on that fold's 512 training rows only.

This phase is a development optimisation, not a foundation-model ablation.
Matched random-init and causal contribution attribution are deliberately
deferred; only the genuine pretrained route is executed here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import p11_residual_fusion as base  # noqa: E402
import p14_geophysical_fm as p14  # noqa: E402


SCHEMA_VERSION = "reconstruction-p17-foundation-geostatistics/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p17_foundation_geostatistics"
DEFAULT_FEATURE_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p17_foundation_geostatistics"
    / "gfm_point_features.npz"
)
MATCH_TOLERANCE = 1e-8
PCA_COMPONENTS = 16
METRIC_WEIGHT_PAIRS = (
    (0.05, 0.00),
    (0.05, 0.10),
    (0.05, 0.20),
    (0.10, 0.10),
    (0.10, 0.20),
    (0.15, 0.00),
    (0.15, 0.10),
    (0.15, 0.20),
    (0.15, 0.40),
    (0.25, 0.10),
    (0.25, 0.20),
    (0.35, 0.20),
    (0.50, 0.20),
)
NEIGHBOUR_COUNTS = (32, 64, 128)
BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
DISTANCE_POWER = 2.0
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2693


@dataclass(frozen=True)
class FoldSamples:
    fold_id: int
    train_target: np.ndarray
    train_raw_features: np.ndarray
    validation_raw_features: np.ndarray
    train_indices_kji: np.ndarray
    validation_indices_kji: np.ndarray
    source_hashes: Mapping[str, str]


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _raw_from_preprocessed(
    values: np.ndarray,
    stats: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Invert the six non-constraint z-scored input columns exactly."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 7:
        raise ValueError("point cache must contain seven input columns")
    if len(stats) != 7:
        raise ValueError("preprocess statistics must cover seven columns")
    raw = np.column_stack(
        [
            matrix[:, column] * float(stats[column]["std"])
            + float(stats[column]["mean"])
            for column in range(1, 7)
        ]
    )
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError("inverted point features are non-finite")
    return raw


def _coordinate_index(train_h5: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read target-free physical coordinates for every active development cell."""

    import h5py

    coordinates: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    patch_keys: list[str] = []
    with h5py.File(train_h5, "r") as handle:
        for key in sorted(handle):
            group = handle[key]
            metadata = json.loads(group.attrs["meta"])
            start = np.asarray(metadata["patch_start_kji"], dtype=np.int64)
            active = np.asarray(group["seismic_patch"][8], dtype=np.float32) > 0.5
            local = np.argwhere(active)
            xyz = np.asarray(
                group["seismic_patch"][3:6], dtype=np.float64
            )[:, active].T
            if xyz.shape != (len(local), 3):
                raise RuntimeError("coordinate channel shape mismatch")
            coordinates.append(xyz)
            indices.append(local + start)
            patch_keys.append(key)
    all_coordinates = np.concatenate(coordinates)
    all_indices = np.concatenate(indices)
    if len(np.unique(all_indices, axis=0)) != len(all_indices):
        raise RuntimeError("active coordinate index contains duplicate KJI cells")
    return all_coordinates, all_indices, {
        "active_coordinate_rows": int(len(all_indices)),
        "patch_count": len(patch_keys),
        "patch_keys_sha256": hashlib.sha256(
            "\n".join(patch_keys).encode("utf-8")
        ).hexdigest(),
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": ["seismic_patch[3:6]", "seismic_patch[8]"],
        "label_dataset_read": False,
    }


def _match_coordinates(
    tree: cKDTree,
    all_indices: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, float]:
    distance, row = tree.query(np.asarray(xyz, dtype=np.float64), k=1)
    maximum = float(np.max(distance))
    if maximum > MATCH_TOLERANCE:
        raise RuntimeError(f"coordinate-to-KJI mismatch: max distance {maximum}")
    matched = np.asarray(all_indices[row], dtype=np.int64)
    if len(np.unique(matched, axis=0)) != len(matched):
        raise RuntimeError("point sample coordinates do not map one-to-one")
    return matched, maximum


def load_fold_samples(
    *,
    stage3_root: Path,
    train_h5: Path,
    oof: base.OOFDevelopment,
) -> tuple[tuple[FoldSamples, ...], dict[str, Any]]:
    """Load the legal 512-point train budget and recover exact KJI cells."""

    stage3_root = Path(stage3_root).expanduser().resolve()
    manifest_path = stage3_root / "cache" / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("frozen_test_i_blocks_loaded"):
        raise RuntimeError("Stage-3 cache reports frozen-test access")
    coordinates, all_indices, coordinate_audit = _coordinate_index(train_h5)
    tree = cKDTree(coordinates)
    folds: list[FoldSamples] = []
    maximum_match_distance = 0.0
    for fold_id in base.FOLD_IDS:
        root = stage3_root / "cache" / base.EXPECTED_LANE / f"fold_{fold_id:02d}"
        stats_path = root / "preprocess_stats.json"
        train_path = root / "point_train.npz"
        validation_path = root / "point_validation.npz"
        train_json_path = root / "point_train.json"
        validation_json_path = root / "point_validation.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        train_json = json.loads(train_json_path.read_text(encoding="utf-8"))
        validation_json = json.loads(
            validation_json_path.read_text(encoding="utf-8")
        )
        if train_json["npz_sha256"] != base._sha256(train_path):  # noqa: SLF001
            raise RuntimeError(f"fold {fold_id} point_train hash mismatch")
        if validation_json["npz_sha256"] != base._sha256(validation_path):  # noqa: SLF001
            raise RuntimeError(f"fold {fold_id} point_validation hash mismatch")
        with np.load(train_path, allow_pickle=False) as payload:
            train_values = np.asarray(payload["input_values"], dtype=np.float64)
            train_target = np.asarray(payload["target_values"], dtype=np.float64)
        with np.load(validation_path, allow_pickle=False) as payload:
            validation_values = np.asarray(
                payload["input_values"], dtype=np.float64
            )
            validation_target = np.asarray(
                payload["target_values"], dtype=np.float64
            )
        if train_values.shape != (512, 7) or train_target.shape != (512,):
            raise RuntimeError(f"fold {fold_id} changed the 512-point train budget")
        validation_mask = oof.fold_ids == fold_id
        if validation_values.shape != (int(validation_mask.sum()), 7):
            raise RuntimeError(f"fold {fold_id} validation row count mismatch")
        np.testing.assert_allclose(
            validation_target,
            oof.target[validation_mask],
            rtol=0.0,
            atol=0.0,
        )
        train_raw = _raw_from_preprocessed(train_values, stats["stats"])
        validation_raw = _raw_from_preprocessed(
            validation_values, stats["stats"]
        )
        train_indices, train_distance = _match_coordinates(
            tree, all_indices, train_raw[:, 3:6]
        )
        validation_indices, validation_distance = _match_coordinates(
            tree, all_indices, validation_raw[:, 3:6]
        )
        np.testing.assert_array_equal(
            validation_indices,
            oof.indices_kji[validation_mask],
        )
        maximum_match_distance = max(
            maximum_match_distance,
            train_distance,
            validation_distance,
        )
        folds.append(
            FoldSamples(
                fold_id=fold_id,
                train_target=train_target,
                train_raw_features=train_raw,
                validation_raw_features=validation_raw,
                train_indices_kji=train_indices,
                validation_indices_kji=validation_indices,
                source_hashes={
                    "point_train_npz": train_json["npz_sha256"],
                    "point_validation_npz": validation_json["npz_sha256"],
                    "preprocess_stats": base._sha256(stats_path),  # noqa: SLF001
                },
            )
        )
    return tuple(folds), {
        **coordinate_audit,
        "coordinate_match_tolerance": MATCH_TOLERANCE,
        "maximum_coordinate_match_distance": maximum_match_distance,
        "train_labels_per_fold": 512,
        "validation_rows_per_fold": int(np.sum(oof.fold_ids == base.FOLD_IDS[0])),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_indices_sha256": _array_sha256(fold.train_indices_kji),
                "validation_indices_sha256": _array_sha256(
                    fold.validation_indices_kji
                ),
                "source_hashes": dict(fold.source_hashes),
            }
            for fold in folds
        ],
    }


def _unique_indices(folds: Sequence[FoldSamples]) -> np.ndarray:
    rows = np.concatenate(
        [
            np.concatenate(
                [fold.train_indices_kji, fold.validation_indices_kji]
            )
            for fold in folds
        ]
    )
    return np.unique(rows, axis=0)


def _feature_rows(
    requested: np.ndarray,
    features: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    lookup = {tuple(row): pos for pos, row in enumerate(requested.tolist())}
    try:
        positions = np.asarray(
            [lookup[tuple(row)] for row in indices.tolist()], dtype=np.int64
        )
    except KeyError as exc:
        raise RuntimeError("requested KJI cell is absent from GFM cache") from exc
    selected = features[:, positions].transpose(1, 0, 2).reshape(
        len(positions), -1
    )
    if not np.all(np.isfinite(selected)):
        raise FloatingPointError("selected GFM features are non-finite")
    return np.asarray(selected, dtype=np.float64)


def _kernel_prediction(
    *,
    train_target: np.ndarray,
    train_raw: np.ndarray,
    validation_raw: np.ndarray,
    train_foundation: np.ndarray,
    validation_foundation: np.ndarray,
    neighbours: int,
    foundation_weight: float,
    seismic_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Foundation-deformed local kernel interpolation with train-only PCA."""

    scaler = StandardScaler().fit(train_foundation)
    train_scaled = scaler.transform(train_foundation)
    validation_scaled = scaler.transform(validation_foundation)
    components = min(PCA_COMPONENTS, len(train_target) - 1, train_scaled.shape[1])
    pca = PCA(n_components=components, random_state=BOOTSTRAP_SEED).fit(
        train_scaled
    )
    train_latent = pca.transform(train_scaled)
    validation_latent = pca.transform(validation_scaled)
    latent_scale = np.std(train_latent, axis=0)
    latent_scale[latent_scale < 1e-8] = 1.0
    train_latent /= latent_scale
    validation_latent /= latent_scale

    coordinate_scale = np.std(train_raw[:, 3:6], axis=0)
    coordinate_scale[coordinate_scale < 1e-8] = 1.0
    seismic_scale = np.std(train_raw[:, 0:3], axis=0)
    seismic_scale[seismic_scale < 1e-8] = 1.0
    train_metric = np.column_stack(
        [
            train_raw[:, 3:6] / coordinate_scale,
            seismic_weight * train_raw[:, 0:3] / seismic_scale,
            foundation_weight * train_latent,
        ]
    )
    validation_metric = np.column_stack(
        [
            validation_raw[:, 3:6] / coordinate_scale,
            seismic_weight * validation_raw[:, 0:3] / seismic_scale,
            foundation_weight * validation_latent,
        ]
    )
    model = NearestNeighbors(n_neighbors=int(neighbours), n_jobs=-1).fit(
        train_metric
    )
    distance, neighbour_rows = model.kneighbors(validation_metric)
    weights = 1.0 / np.maximum(distance, 1e-8) ** DISTANCE_POWER
    prediction = np.sum(
        weights * train_target[neighbour_rows], axis=1
    ) / np.sum(weights, axis=1)
    return prediction, {
        "pca_components": int(components),
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "foundation_weight": foundation_weight,
        "seismic_weight": seismic_weight,
        "distance_power": DISTANCE_POWER,
        "neighbours": int(neighbours),
        "fit_rows": int(len(train_target)),
        "all_transforms_fit_on_outer_train_only": True,
        "target_used_for_metric_fit": False,
    }


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rows": int(len(error)),
    }


def _whole_fold_bootstrap(
    *, target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, fold_ids: np.ndarray
) -> dict[str, Any]:
    unique = np.asarray(base.FOLD_IDS, dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([np.flatnonzero(fold_ids == fold) for fold in sampled])
        base_rmse = np.sqrt(np.mean((baseline[rows] - target[rows]) ** 2))
        candidate_rmse = np.sqrt(np.mean((candidate[rows] - target[rows]) ** 2))
        deltas[draw] = candidate_rmse - base_rmse
    return {
        "unit": "whole spatial fold",
        "independent_units": len(unique),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "rmse_delta_candidate_minus_baseline": {
            "mean": float(np.mean(deltas)),
            "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
            "probability_candidate_better": float(np.mean(deltas < 0.0)),
        },
    }


def evaluate(
    *,
    oof: base.OOFDevelopment,
    folds: Sequence[FoldSamples],
    requested_indices: np.ndarray,
    foundation_features: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidates: dict[str, np.ndarray] = {}
    fit_audits: dict[str, list[dict[str, Any]]] = {}
    for foundation_weight, seismic_weight in METRIC_WEIGHT_PAIRS:
        for neighbours in NEIGHBOUR_COUNTS:
            kernel = np.full(len(oof.target), np.nan, dtype=np.float64)
            fold_audits: list[dict[str, Any]] = []
            for fold in folds:
                validation = oof.fold_ids == fold.fold_id
                train_foundation = _feature_rows(
                    requested_indices,
                    foundation_features,
                    fold.train_indices_kji,
                )
                validation_foundation = _feature_rows(
                    requested_indices,
                    foundation_features,
                    fold.validation_indices_kji,
                )
                fold_prediction, audit = _kernel_prediction(
                    train_target=fold.train_target,
                    train_raw=fold.train_raw_features,
                    validation_raw=fold.validation_raw_features,
                    train_foundation=train_foundation,
                    validation_foundation=validation_foundation,
                    neighbours=neighbours,
                    foundation_weight=foundation_weight,
                    seismic_weight=seismic_weight,
                )
                kernel[validation] = fold_prediction
                fold_audits.append({"fold_id": fold.fold_id, **audit})
            if not np.all(np.isfinite(kernel)):
                raise RuntimeError("kernel OOF prediction is incomplete")
            for blend in BLEND_WEIGHTS:
                name = (
                    f"gfm_metric_f{foundation_weight:.2f}_s{seismic_weight:.2f}"
                    f"_k{neighbours}_blend_{blend:.2f}"
                )
                candidates[name] = (
                    (1.0 - blend) * oof.baseline + blend * kernel
                )
                fit_audits[name] = [
                    {**row, "pykrige_blend_weight": 1.0 - blend}
                    for row in fold_audits
                ]
    baseline_metrics = _metrics(oof.target, oof.baseline)
    candidate_metrics = {
        name: _metrics(oof.target, prediction)
        for name, prediction in candidates.items()
    }
    best_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["rmse"])
    best_prediction = candidates[best_name]
    per_fold: list[dict[str, Any]] = []
    outcome_counts = {"win": 0, "loss": 0, "tie": 0}
    for fold_id in base.FOLD_IDS:
        mask = oof.fold_ids == fold_id
        baseline_fold = _metrics(oof.target[mask], oof.baseline[mask])
        candidate_fold = _metrics(oof.target[mask], best_prediction[mask])
        delta = candidate_fold["rmse"] - baseline_fold["rmse"]
        outcome = "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        outcome_counts[outcome] += 1
        per_fold.append(
            {
                "fold_id": fold_id,
                "baseline": baseline_fold,
                "candidate": candidate_fold,
                "rmse_delta_candidate_minus_baseline": delta,
                "outcome": outcome,
            }
        )
    best_metrics = candidate_metrics[best_name]
    delta = best_metrics["rmse"] - baseline_metrics["rmse"]
    bootstrap = _whole_fold_bootstrap(
        target=oof.target,
        baseline=oof.baseline,
        candidate=best_prediction,
        fold_ids=oof.fold_ids,
    )
    state = (
        "DEVELOPMENT_SIGNAL"
        if delta < 0.0
        else "VERIFIED_NO_PROMOTION"
    )
    experiment = {
        "primary_metric": "pooled development OOF RMSE (lower is better)",
        "baseline": {"pykrige_ok3d_repeat_0": baseline_metrics},
        "candidate_grid": {
            name: {
                **candidate_metrics[name],
                "rmse_delta_vs_pykrige": candidate_metrics[name]["rmse"]
                - baseline_metrics["rmse"],
            }
            for name in sorted(candidate_metrics)
        },
        "search_space": {
            "metric_weight_pairs": [list(pair) for pair in METRIC_WEIGHT_PAIRS],
            "neighbour_counts": list(NEIGHBOUR_COUNTS),
            "blend_weights": list(BLEND_WEIGHTS),
            "candidate_count": len(candidates),
            "no_foundation_control_included": False,
            "ablation_deferred": True,
        },
        "selected_candidate": best_name,
        "selected_metrics": best_metrics,
        "rmse_delta_vs_pykrige": delta,
        "relative_rmse_change_vs_pykrige": delta / baseline_metrics["rmse"],
        "per_fold": per_fold,
        "independent_fold_outcome_counts": outcome_counts,
        "whole_fold_bootstrap": bootstrap,
        "decision": {
            "state": state,
            "default_enabled": False,
            "reason": (
                "development-only signal; frozen holdout remains sealed and "
                "foundation attribution is deferred until later ablation"
                if delta < 0.0
                else "no pooled development improvement over PyKrige"
            ),
            "pretrained_foundation_used": True,
            "pretrained_contribution_claimed": False,
            "ablation_deferred": True,
        },
        "fit_audits": fit_audits[best_name],
    }
    payload = {
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "target": np.asarray(oof.target, dtype=np.float64),
        "baseline_prediction": np.asarray(oof.baseline, dtype=np.float64),
        "candidate_prediction": np.asarray(best_prediction, dtype=np.float64),
        "baseline_error": np.asarray(oof.baseline - oof.target, dtype=np.float64),
        "candidate_error": np.asarray(best_prediction - oof.target, dtype=np.float64),
    }
    return experiment, payload


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    experiment = result["experiment"]
    base_rmse = experiment["baseline"]["pykrige_ok3d_repeat_0"]["rmse"]
    candidate_rmse = experiment["selected_metrics"]["rmse"]
    counts = experiment["independent_fold_outcome_counts"]
    ci = experiment["whole_fold_bootstrap"][
        "rmse_delta_candidate_minus_baseline"
    ]["ci95"]
    lines = [
        "# P17 foundation-informed nonstationary geostatistics",
        "",
        "## Result",
        "",
        f"- PyKrige pooled development OOF RMSE: `{base_rmse:.12f}`.",
        f"- Selected pretrained-GFM kernel RMSE: `{candidate_rmse:.12f}`.",
        f"- Selected candidate: `{experiment['selected_candidate']}`.",
        f"- Delta (candidate - PyKrige): `{candidate_rmse - base_rmse:+.12f}`.",
        f"- Independent spatial-fold outcomes: {counts['win']} wins / "
        f"{counts['loss']} losses / {counts['tie']} ties.",
        f"- Whole-fold bootstrap delta CI95: `[{ci[0]:+.12f}, {ci[1]:+.12f}]`.",
        f"- Decision: `{experiment['decision']['state']}`; default remains disabled.",
        "",
        "## Method",
        "",
        "The genuine frozen ThinkOnward GFM encodes input seismic traces. "
        "Within every outer spatial fold, its representations are scaled and "
        "reduced by PCA using only the 512 legal training points.  Physical "
        "coordinates, local seismic attributes and the reduced GFM coordinates "
        "jointly define a nonstationary neighbourhood metric.  Inverse-distance "
        "kernel interpolation then estimates porosity and is conservatively "
        "blended with the unchanged PyKrige prediction.",
        "",
        "## Boundary",
        "",
        "- Exactly 512 training labels and 2,048 validation rows are used per fold.",
        "- No frozen test or `test.h5` path exists in the CLI.",
        "- Encoder inputs are target-free seismic and coordinates only.",
        "- This phase does not run a matched random-init/no-foundation ablation; "
        "causal attribution to pretraining is therefore not claimed.",
        "- Any positive result is development evidence, not a final test claim.",
        "",
    ]
    (output_dir / "evidence.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    source_root: Path,
    snapshot_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    feature_cache: Path = DEFAULT_FEATURE_CACHE,
    device: str = "cuda:0",
    batch_size: int = 24,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(HERE)
    except ValueError as exc:
        raise ValueError("P17 output must remain inside reconstruction") from exc
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(Path(stage3_root))
    folds, sample_audit = load_fold_samples(
        stage3_root=Path(stage3_root), train_h5=inputs.train_h5, oof=oof
    )
    requested_indices = _unique_indices(folds)
    proxy = SimpleNamespace(indices_kji=requested_indices)
    features, feature_audit = p14.get_projected_features(
        weight_mode="pretrained",
        inputs=inputs,
        oof=proxy,
        source_root=source_root,
        snapshot_path=snapshot_path,
        feature_cache=feature_cache,
        device=device,
        batch_size=batch_size,
    )
    experiment, prediction_payload = evaluate(
        oof=oof,
        folds=folds,
        requested_indices=requested_indices,
        foundation_features=features,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction_errors.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "implementation": {
            "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "script_sha256": base._sha256(Path(__file__)),  # noqa: SLF001
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "model": {
            "id": "thinkonward/geophysical-foundation-model",
            "snapshot_revision": p14.gfm.SNAPSHOT_REVISION,
            "encoder_weight_mode": "pretrained",
            "frozen": True,
            "role": "nonstationary neighbourhood metric; not direct porosity prediction",
        },
        "source_oof": {
            "rows": int(len(oof.target)),
            "records": list(oof.source_records),
            "indices_kji_sha256": _array_sha256(oof.indices_kji),
        },
        "sample_audit": sample_audit,
        "foundation_feature_audit": feature_audit,
        "experiment": experiment,
        "prediction_error_artifact": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "sha256": base._sha256(prediction_path),  # noqa: SLF001
            "rows": int(len(oof.target)),
        },
        "holdout_firewall": {
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": [
                "seismic_patch[0:3]",
                "seismic_patch[3:6]",
                "seismic_patch[8]",
            ],
            "label_dataset_read_by_encoder": False,
            "test_path_argument_exists": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": device,
            "feature_cache_reused": feature_audit["cache_reused"],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_evidence(output_dir, result)
    return result


def verify_evidence(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Independently recompute persisted OOF metrics and artifact hashes."""

    output_dir = Path(output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unexpected P17 evidence schema")
    record = summary["prediction_error_artifact"]
    prediction_path = PROJECT_ROOT / record["path"]
    if base._sha256(prediction_path) != record["sha256"]:  # noqa: SLF001
        raise RuntimeError("prediction artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.float64)
        baseline = np.asarray(payload["baseline_prediction"], dtype=np.float64)
        candidate = np.asarray(payload["candidate_prediction"], dtype=np.float64)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
        indices_kji = np.asarray(payload["indices_kji"], dtype=np.int64)
    if len(target) != record["rows"] or target.shape != baseline.shape:
        raise RuntimeError("persisted OOF row count mismatch")
    if candidate.shape != target.shape or fold_ids.shape != target.shape:
        raise RuntimeError("persisted prediction shape mismatch")
    baseline_metrics = _metrics(target, baseline)
    candidate_metrics = _metrics(target, candidate)
    reported = summary["experiment"]
    np.testing.assert_allclose(
        baseline_metrics["rmse"],
        reported["baseline"]["pykrige_ok3d_repeat_0"]["rmse"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        candidate_metrics["rmse"],
        reported["selected_metrics"]["rmse"],
        rtol=0.0,
        atol=1e-15,
    )
    fold_checks = []
    for row in reported["per_fold"]:
        mask = fold_ids == row["fold_id"]
        if int(mask.sum()) != 2048:
            raise RuntimeError("unexpected validation rows in persisted fold")
        base_fold = _metrics(target[mask], baseline[mask])
        candidate_fold = _metrics(target[mask], candidate[mask])
        np.testing.assert_allclose(
            [base_fold["rmse"], candidate_fold["rmse"]],
            [row["baseline"]["rmse"], row["candidate"]["rmse"]],
            rtol=0.0,
            atol=1e-15,
        )
        fold_checks.append(
            {
                "fold_id": row["fold_id"],
                "rows": int(mask.sum()),
                "baseline_rmse": base_fold["rmse"],
                "candidate_rmse": candidate_fold["rmse"],
            }
        )
    if summary["sample_audit"]["train_labels_per_fold"] != 512:
        raise RuntimeError("training-label budget changed")
    if summary["holdout_firewall"]["frozen_holdout_opened"]:
        raise RuntimeError("frozen holdout firewall failed")
    if reported["decision"]["default_enabled"]:
        raise RuntimeError("development-only candidate must remain disabled")
    verification = {
        "schema_version": "reconstruction-p17-independent-verification/v1",
        "status": "PASSED",
        "summary_sha256": base._sha256(summary_path),  # noqa: SLF001
        "prediction_artifact_sha256": record["sha256"],
        "rows": int(len(target)),
        "indices_kji_sha256": _array_sha256(indices_kji),
        "baseline_metrics_recomputed": baseline_metrics,
        "candidate_metrics_recomputed": candidate_metrics,
        "fold_checks": fold_checks,
        "firewall_checks": {
            "train_labels_per_fold": 512,
            "validation_rows_per_fold": 2048,
            "test_h5_opened": False,
            "default_enabled": False,
        },
    }
    verification_path = output_dir / "verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-dev")
    check.add_argument("--data-dir", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--snapshot-path", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--batch-size", type=int, default=24)
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
        verification = verify_evidence(args.output_dir)
        print(json.dumps(verification, sort_keys=True))
        return 0
    values = vars(args)
    values.pop("command")
    result = run(**values)
    experiment = result["experiment"]
    print(
        json.dumps(
            {
                "baseline_rmse": experiment["baseline"][
                    "pykrige_ok3d_repeat_0"
                ]["rmse"],
                "candidate_rmse": experiment["selected_metrics"]["rmse"],
                "selected_candidate": experiment["selected_candidate"],
                "state": experiment["decision"]["state"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
