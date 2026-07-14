#!/usr/bin/env python3
"""P5.1 R0/R1 development-only conditional reconstruction audit.

R0 freezes provenance, a buffered development pseudo-test split, geometry-only
pseudo-well positions, feature/config/mask hashes and a narrow HDF5 access
audit.  R1 trains one fixed linear model once and evaluates B0/B1/shuffled on
the same development pseudo-test cells.  This module deliberately has no
known/frozen-holdout or physical ``test.h5`` surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from ml_framework.model_discovery import discover_model  # noqa: E402
from ml_framework.preprocess import (  # noqa: E402
    denoise_identity,
    denormalize,
    fit_zscore,
    normalize,
)

import p4_reconstruction as p4  # noqa: E402


SCHEMA_VERSION = "p5.1-r01-reconstruction-v1"
ROOT_SEED = 2693
MODE = "conditional"
DEVELOPMENT_I_BLOCKS = (0, 1, 2, 3)
REQUESTED_FOLDS = 5
PSEUDO_TEST_FOLD_ID = 2
BUFFER_BLOCKS = 1
TRAIN_PSEUDO_WELLS = 64
PSEUDO_TEST_WELLS = 32
MIN_BAND_SUPPORT = 32
FEATURE_NAMES = (
    "conditional_idw_porosity",
    "seismic_amplitude",
    "seismic_local_rms",
    "seismic_vertical_gradient",
    "x_normalized",
    "y_normalized",
    "depth_normalized",
)
MODEL_CONFIG: dict[str, Any] = {
    "model_id": "reconstruction_linear_sgd",
    "learning_rate": 0.01,
    "ridge_alpha": 0.1,
    "updates": 100,
    "loss": "mse_valid_label_mean",
    "target_transform": "identity",
    "output_transform": "identity",
    "hpo": False,
}
DISTANCE_QUANTILES = (0.25, 0.50, 0.75)
ANISOTROPY = np.asarray([1.0, 1.0, 3.0], dtype=np.float64)


@dataclass(frozen=True)
class RecordGeometry:
    source_key: str
    sample_id: str
    i_block: int
    j_block: int
    k_block: int
    patch_start_kji: tuple[int, int, int]
    patch_shape_kji: tuple[int, int, int]
    active_flat_indices: np.ndarray
    cell_start: int
    cell_stop: int


@dataclass(frozen=True)
class GeometryBundle:
    records: tuple[RecordGeometry, ...]
    indices_kji: np.ndarray
    seismic: np.ndarray
    coordinates: np.ndarray
    original_observed_mask: np.ndarray
    cell_i_blocks: np.ndarray
    cell_k_blocks: np.ndarray
    volume_shape_kji: tuple[int, int, int]
    access_audit: Mapping[str, Any]


@dataclass(frozen=True)
class R0Prepared:
    geometry: GeometryBundle
    split_manifest: Mapping[str, Any]
    train_mask: np.ndarray
    pseudo_test_mask: np.ndarray
    train_pseudo_indices: np.ndarray
    pseudo_test_indices: np.ndarray
    common_metric_mask: np.ndarray
    pseudo_test_distances: np.ndarray
    distance_edges: tuple[float, ...]
    manifest: Mapping[str, Any]


def _relative(path: Path) -> str:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return resolved.relative_to(root).as_posix()
    return path.name


def _hash_arrays(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _train_path(data_dir: Path | None) -> Path:
    root = (
        Path(data_dir)
        if data_dir is not None
        else PROJECT_ROOT / "_data" / "processed" / "reconstruction"
    )
    path = root / "train.h5"
    if not path.is_file():
        raise FileNotFoundError(f"development physical train.h5 is missing: {path}")
    if path.name != "train.h5":
        raise RuntimeError("R0/R1 accepts only the physical train.h5 development container")
    return path


def _sample_id(meta: Mapping[str, Any]) -> str:
    k_block, j_block, i_block = (int(value) for value in meta["patch_index_kji"])
    return f"k{k_block:02d}_j{j_block:02d}_i{i_block:02d}"


def load_development_geometry(data_dir: Path | None = None) -> GeometryBundle:
    """Read only train.h5 geometry/input slices; never read PORO or well_log_seq."""
    path = _train_path(data_dir)
    records: list[RecordGeometry] = []
    indices: list[np.ndarray] = []
    seismic: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    i_blocks: list[np.ndarray] = []
    k_blocks: list[np.ndarray] = []
    max_shape = np.zeros(3, dtype=np.int64)
    cursor = 0
    with h5py.File(path, "r") as handle:
        for source_key in sorted(handle):
            group = handle[source_key]
            meta = json.loads(group.attrs["meta"])
            k_block, j_block, i_block = (
                int(value) for value in meta["patch_index_kji"]
            )
            if i_block not in DEVELOPMENT_I_BLOCKS:
                raise RuntimeError(
                    "physical train.h5 contains a non-development I-block; firewall refuses it"
                )
            start = tuple(int(value) for value in meta["patch_start_kji"])
            shape = tuple(int(value) for value in meta["patch_shape_kji"])
            patch = group["seismic_patch"]
            if patch.shape != (9, *shape):
                raise ValueError(f"invalid seismic patch shape for {source_key}: {patch.shape}")
            # Channel 6 contains reference-derived sparse PORO and is deliberately
            # not read.  Channel 7 is only the local provenance mask.
            signal_and_coordinates = np.asarray(patch[0:6], dtype=np.float32)
            masks = np.asarray(patch[7:9], dtype=np.float32)
            active_flat = np.flatnonzero(masks[1].reshape(-1) > 0.5)
            if not active_flat.size:
                continue
            local = np.indices(shape, dtype=np.int64).reshape(3, -1).T[active_flat]
            global_kji = local + np.asarray(start, dtype=np.int64)
            count = int(active_flat.size)
            indices.append(global_kji)
            seismic.append(
                denoise_identity(signal_and_coordinates[0:3].reshape(3, -1).T[active_flat])
            )
            coordinates.append(
                signal_and_coordinates[3:6].reshape(3, -1).T[active_flat]
            )
            observed.append(masks[0].reshape(-1)[active_flat] > 0.5)
            i_blocks.append(np.full(count, i_block, dtype=np.int16))
            k_blocks.append(np.full(count, k_block, dtype=np.int16))
            records.append(
                RecordGeometry(
                    source_key=source_key,
                    sample_id=_sample_id(meta),
                    i_block=i_block,
                    j_block=j_block,
                    k_block=k_block,
                    patch_start_kji=start,
                    patch_shape_kji=shape,
                    active_flat_indices=active_flat.astype(np.int64),
                    cell_start=cursor,
                    cell_stop=cursor + count,
                )
            )
            cursor += count
            max_shape = np.maximum(max_shape, np.asarray(start) + np.asarray(shape))
    if not records:
        raise ValueError("physical train.h5 contains no conditional development cells")
    bundle = GeometryBundle(
        records=tuple(records),
        indices_kji=np.concatenate(indices).astype(np.int64),
        seismic=np.concatenate(seismic).astype(np.float64),
        coordinates=np.concatenate(coordinates).astype(np.float64),
        original_observed_mask=np.concatenate(observed).astype(bool),
        cell_i_blocks=np.concatenate(i_blocks),
        cell_k_blocks=np.concatenate(k_blocks),
        volume_shape_kji=tuple(int(value) for value in max_shape),
        access_audit={
            "physical_containers_opened": ["train.h5"],
            "physical_test_h5_opened": False,
            "datasets_read": [
                "group.attrs.meta",
                "seismic_patch[0:6]",
                "seismic_patch[7:9]",
            ],
            "reference_sparse_poro_channel_6_read": False,
            "well_log_seq_read": False,
            "known_or_frozen_metrics_read": False,
            "known_or_frozen_predictions_read": False,
        },
    )
    if set(np.unique(bundle.cell_i_blocks).tolist()) != set(DEVELOPMENT_I_BLOCKS):
        raise ValueError("conditional development I-block coverage differs from I0-3")
    return bundle


def load_development_target(
    geometry: GeometryBundle, data_dir: Path | None = None
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Read only development PORO after geometry-only pseudo-well selection."""
    path = _train_path(data_dir)
    values: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        for record in geometry.records:
            target = np.asarray(handle[record.source_key]["label"][()], dtype=np.float32)
            if target.shape != record.patch_shape_kji:
                raise ValueError(f"invalid label shape for {record.source_key}")
            values.append(target.reshape(-1)[record.active_flat_indices])
    merged = np.concatenate(values).astype(np.float64)
    if merged.shape != (geometry.indices_kji.shape[0],) or not np.all(np.isfinite(merged)):
        raise ValueError("development target is non-finite or misaligned")
    return merged, {
        "physical_containers_opened": ["train.h5"],
        "datasets_read": ["label"],
        "physical_test_h5_opened": False,
        "well_log_seq_read": False,
        "selection_frozen_before_target_read": True,
        "development_label_hash": _hash_arrays(target=merged),
    }


def _contiguous_buckets(groups: Sequence[int], n_splits: int) -> list[tuple[int, ...]]:
    quotient, remainder = divmod(len(groups), n_splits)
    buckets: list[tuple[int, ...]] = []
    start = 0
    for index in range(n_splits):
        size = quotient + (1 if index < remainder else 0)
        buckets.append(tuple(int(value) for value in groups[start : start + size]))
        start += size
    return buckets


def build_development_split(geometry: GeometryBundle) -> Mapping[str, Any]:
    """Mirror the frozen P4 conditional buffered K-fold rule without test access."""
    groups = sorted(int(value) for value in np.unique(geometry.cell_k_blocks))
    effective = min(REQUESTED_FOLDS, len(groups))
    if effective < 2:
        raise ValueError("fewer than two development K groups")
    buckets = _contiguous_buckets(groups, effective)
    validation = buckets[PSEUDO_TEST_FOLD_ID]
    purged = tuple(
        value
        for value in groups
        if value not in validation
        and min(abs(value - held) for held in validation) <= BUFFER_BLOCKS
    )
    train = tuple(value for value in groups if value not in validation and value not in purged)
    if not train or not validation:
        raise ValueError("buffered pseudo-test fold has empty train or validation support")
    record = {
        "contract": "p4_reconstruction.build_spatial_manifest conditional development fold",
        "requested_n_splits": REQUESTED_FOLDS,
        "effective_n_splits": effective,
        "fold_id": PSEUDO_TEST_FOLD_ID,
        "axis": "k_block",
        "buffer_blocks": BUFFER_BLOCKS,
        "development_i_blocks": list(DEVELOPMENT_I_BLOCKS),
        "effective_train_k_blocks": list(train),
        "purged_k_blocks": list(purged),
        "pseudo_test_k_blocks": list(validation),
        "effective_train_sample_ids": sorted(
            item.sample_id for item in geometry.records if item.k_block in train
        ),
        "purged_sample_ids": sorted(
            item.sample_id for item in geometry.records if item.k_block in purged
        ),
        "pseudo_test_sample_ids": sorted(
            item.sample_id for item in geometry.records if item.k_block in validation
        ),
        "known_or_frozen_test_sample_ids_read": False,
    }
    return {**record, "split_hash": hash_payload(record)}


def select_spatial_points(
    coordinates: np.ndarray, indices_kji: np.ndarray, count: int
) -> np.ndarray:
    """Geometry-only deterministic farthest-point sampling; target is not an input."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    indices = np.asarray(indices_kji, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or indices.shape != xyz.shape:
        raise ValueError("spatial point selection expects aligned [N,3] geometry")
    if count < 2 or xyz.shape[0] < count:
        raise ValueError("insufficient geometry support for requested pseudo-wells")
    order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
    scaled = xyz[order] * ANISOTROPY
    centre = scaled.mean(axis=0)
    first = int(np.argmin(np.sum((scaled - centre) ** 2, axis=1)))
    chosen = [first]
    minimum = np.sum((scaled - scaled[first]) ** 2, axis=1)
    minimum[first] = -1.0
    for _ in range(1, count):
        next_index = int(np.argmax(minimum))
        if minimum[next_index] < 0:
            raise ValueError("pseudo-well farthest-point selection exhausted")
        chosen.append(next_index)
        distance = np.sum((scaled - scaled[next_index]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[np.asarray(chosen, dtype=np.int64)] = -1.0
    return order[np.asarray(chosen, dtype=np.int64)]


def _nearest_distance(query: np.ndarray, wells: np.ndarray) -> np.ndarray:
    if wells.shape[0] < 2:
        raise ValueError("distance bands require at least two pseudo-wells")
    distance, _ = cKDTree(wells * ANISOTROPY).query(query * ANISOTROPY, k=1)
    return np.asarray(distance, dtype=np.float64)


def _metric_mask_and_bands(
    geometry: GeometryBundle,
    pseudo_test_mask: np.ndarray,
    pseudo_test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    validation_global = np.flatnonzero(pseudo_test_mask)
    selected_global = validation_global[pseudo_test_indices]
    common = pseudo_test_mask.copy()
    common[selected_global] = False
    validation_coordinates = geometry.coordinates[pseudo_test_mask]
    well_coordinates = validation_coordinates[pseudo_test_indices]
    distance = _nearest_distance(validation_coordinates, well_coordinates)
    metric_distance = distance[~np.isin(validation_global, selected_global)]
    quantiles = tuple(float(value) for value in np.quantile(metric_distance, DISTANCE_QUANTILES))
    if not all(math.isfinite(value) for value in quantiles):
        raise ValueError("non-finite distance-band boundary")
    if any(right <= left for left, right in zip(quantiles, quantiles[1:])):
        raise ValueError("distance-band boundaries are degenerate")
    return common, distance, (0.0, *quantiles, math.inf)


def _source_record(relative_path: str) -> Mapping[str, Any]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative_path, "sha256": hash_file(path), "bytes": path.stat().st_size}


def _selected_kji(
    geometry: GeometryBundle, mask: np.ndarray, local_indices: np.ndarray
) -> np.ndarray:
    return geometry.indices_kji[mask][local_indices].astype(np.int64)


def prepare_r0(data_dir: Path | None = None) -> R0Prepared:
    geometry = load_development_geometry(data_dir)
    split = build_development_split(geometry)
    train_mask = np.isin(
        geometry.cell_k_blocks, np.asarray(split["effective_train_k_blocks"])
    )
    pseudo_test_mask = np.isin(
        geometry.cell_k_blocks, np.asarray(split["pseudo_test_k_blocks"])
    )
    train_selection = select_spatial_points(
        geometry.coordinates[train_mask], geometry.indices_kji[train_mask], TRAIN_PSEUDO_WELLS
    )
    test_selection = select_spatial_points(
        geometry.coordinates[pseudo_test_mask],
        geometry.indices_kji[pseudo_test_mask],
        PSEUDO_TEST_WELLS,
    )
    common_mask, distance, distance_edges = _metric_mask_and_bands(
        geometry, pseudo_test_mask, test_selection
    )
    train_kji = _selected_kji(geometry, train_mask, train_selection)
    test_kji = _selected_kji(geometry, pseudo_test_mask, test_selection)
    build_summary = json.loads((HERE / "build_summary.json").read_text(encoding="utf-8"))
    model_inspection = json.loads(
        (HERE / "model_inspection.json").read_text(encoding="utf-8")
    )
    source_records = [
        _source_record("_pipelines/02_task_datasets/reconstruction/build_dataset.py"),
        _source_record("_pipelines/02_task_datasets/reconstruction/build_summary.json"),
        _source_record("_pipelines/02_task_datasets/reconstruction/model_inspection.json"),
        _source_record("_pipelines/02_task_datasets/reconstruction/p4_reconstruction.py"),
        _source_record("_models/reconstruction/reconstruction_linear_sgd.py"),
    ]
    config_payload = {
        **MODEL_CONFIG,
        "root_seed": ROOT_SEED,
        "feature_names": list(FEATURE_NAMES),
        "train_pseudo_wells": TRAIN_PSEUDO_WELLS,
        "pseudo_test_wells": PSEUDO_TEST_WELLS,
        "pseudo_well_rule": "geometry_only_anisotropic_farthest_point_sampling",
        "shuffle_rule": "seeded_nonzero_cyclic_value_rotation_at_fixed_locations",
        "distance_band_rule": "pseudo-test non-condition-cell distance quantiles 0/25/50/75/100%",
    }
    sample_payload = [
        {
            "sample_id": item.sample_id,
            "i_block": item.i_block,
            "j_block": item.j_block,
            "k_block": item.k_block,
            "patch_start_kji": list(item.patch_start_kji),
            "patch_shape_kji": list(item.patch_shape_kji),
        }
        for item in geometry.records
    ]
    mask_indices = geometry.indices_kji[common_mask]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R0_zero_training_contract",
        "track_id": "reconstruction",
        "mode": MODE,
        "root_seed": ROOT_SEED,
        "evidence_scope": "development_protocol_mechanism_only",
        "fresh_blind": False,
        "field_generalization": False,
        "formal_lane_status": "blocked",
        "known_holdout_status": "not_read_not_consumed",
        "sources": source_records,
        "label": {
            "name": "Volve final Eclipse PORO on active cells",
            "units": "fraction",
            "source": "Eclipse VOLVE_2016.INIT PORO mapped through VOLVE_2016.GRID",
            "rms_role": "independent value-multiset cross-check only",
            "independent_measured_phie": False,
            "label_version": p4.protocol(MODE).label_version,
        },
        "rms_cross_check": {
            "role": "value-multiset cross-check only; no RMS spatial mapping is claimed",
            "member": model_inspection["rms_reference"]["member"],
            "format": model_inspection["rms_reference"]["format"],
            "exact_nonzero_porosity_multiset_match": model_inspection["rms_reference"][
                "porosity_multiset_exactly_matches_eclipse_nonzero_phif_nw"
            ],
            "spatial_mapping": model_inspection["rms_reference"]["spatial_mapping"],
        },
        "coordinates": {
            "channels": ["x_normalized", "y_normalized", "depth_normalized"],
            "source": "Eclipse active-cell centres",
            "normalization": "fixed full-grid minmax geometry transform from build_dataset",
            "bounds": build_summary["coordinate_bounds"],
            "fold_train_statistical_preprocessing_required": True,
        },
        "seismic": {
            **build_summary["seismic"],
            "source": "ST0202 post-stack time SEG-Y sampled at Eclipse cells",
            "fixed_project_weak_well_tie_retained": True,
        },
        "weak_tie": {
            **build_summary["weak_tie"],
            "disclosure": (
                "B0 retains the fixed project-level weak well tie/time sampling and is not a "
                "claim of zero well-derived information."
            ),
        },
        "existing_sparse_poro": {
            **build_summary["sparse_wells"],
            "value_source": "Eclipse reference PORO sampled at mapped cells",
            "synthetic_reference_revealed": True,
            "independent_measured_phie": False,
            "used_by_r1": False,
        },
        "conditions": {
            "B0": {
                "formal_name": "no_pseudo_test_PORO_condition",
                "pseudo_test_poro_supplied": False,
                "fixed_weak_tie_seismic_sampling_retained": True,
            },
            "B1": {
                "formal_name": "correct_synthetic_reference_revealed_pseudo_wells",
                "pseudo_test_poro_supplied": True,
            },
            "shuffled": {
                "formal_name": "seed2693_shuffled_pseudo_well_values_fixed_locations",
                "pseudo_test_poro_supplied": True,
            },
        },
        "split": split,
        "pseudo_well_selection": {
            "selection_frozen_before_target_values_read": True,
            "selection_inputs": ["normalized_xyz", "global_kji", "split_membership"],
            "target_is_not_function_argument": True,
            "rule": "geometry_only_anisotropic_farthest_point_sampling",
            "anisotropy_xyz": ANISOTROPY.tolist(),
            "train_count": int(train_kji.shape[0]),
            "pseudo_test_count": int(test_kji.shape[0]),
            "train_indices_kji": train_kji.tolist(),
            "pseudo_test_indices_kji": test_kji.tolist(),
            "train_selection_hash": _hash_arrays(indices_kji=train_kji),
            "pseudo_test_selection_hash": _hash_arrays(indices_kji=test_kji),
        },
        "distance_bands": {
            "space": "normalized_xyz_with_depth_anisotropy_3",
            "boundary_rule": "quantiles computed from geometry before target values are read",
            "edges": [None if math.isinf(value) else value for value in distance_edges],
            "minimum_support": MIN_BAND_SUPPORT,
        },
        "access_audit": dict(geometry.access_audit),
        "test_firewall": {
            "physical_test_h5_opened": False,
            "known_or_frozen_arrays_read": False,
            "known_or_frozen_metrics_read": False,
            "known_or_frozen_predictions_read": False,
            "global_well_log_seq_read": False,
            "only_physical_train_h5_development_channels": True,
        },
        "hashes": {
            "sample_hash": hash_payload(sample_payload),
            "block_split_hash": split["split_hash"],
            "label_source_hash": hash_payload(
                {
                    "name": "Volve final Eclipse PORO on active cells",
                    "label_version": p4.protocol(MODE).label_version,
                    "build_dataset_sha256": source_records[0]["sha256"],
                    "build_summary_sha256": source_records[1]["sha256"],
                    "model_inspection_sha256": source_records[2]["sha256"],
                }
            ),
            "sparse_condition_source_hash": hash_payload(
                {
                    "value_source": "Eclipse reference PORO sampled at mapped cells",
                    "synthetic_reference_revealed": True,
                    "n_observation_rows": build_summary["sparse_wells"]["n_observation_rows"],
                    "mapping": build_summary["sparse_wells"]["mapping"],
                }
            ),
            "geometry_input_hash": _hash_arrays(
                indices_kji=geometry.indices_kji,
                seismic=geometry.seismic,
                coordinates=geometry.coordinates,
                original_observed_mask=geometry.original_observed_mask,
            ),
            "feature_hash": hash_payload(list(FEATURE_NAMES)),
            "config_hash": hash_payload(config_payload),
            "common_metric_mask_hash": _hash_arrays(indices_kji=mask_indices),
        },
        "config": config_payload,
        "support": {
            "development_cells": int(geometry.indices_kji.shape[0]),
            "effective_train_cells": int(train_mask.sum()),
            "pseudo_test_cells": int(pseudo_test_mask.sum()),
            "common_metric_cells": int(common_mask.sum()),
            "exact_pseudo_test_cells_excluded_from_all_conditions": int(test_kji.shape[0]),
        },
    }
    manifest["r0_manifest_hash"] = hash_payload(manifest)
    return R0Prepared(
        geometry=geometry,
        split_manifest=split,
        train_mask=train_mask,
        pseudo_test_mask=pseudo_test_mask,
        train_pseudo_indices=train_selection,
        pseudo_test_indices=test_selection,
        common_metric_mask=common_mask,
        pseudo_test_distances=distance,
        distance_edges=distance_edges,
        manifest=manifest,
    )


def _constraints(
    geometry: GeometryBundle,
    target: np.ndarray,
    mask: np.ndarray,
    local_indices: np.ndarray,
    values: np.ndarray | None = None,
) -> np.ndarray:
    coordinates = geometry.coordinates[mask][local_indices]
    selected_values = target[mask][local_indices] if values is None else np.asarray(values)
    return np.column_stack([coordinates, selected_values]).astype(np.float64)


def _idw(query: np.ndarray, constraints: np.ndarray) -> np.ndarray:
    if constraints.shape[0] < 2:
        raise ValueError("IDW requires at least two pre-registered pseudo-wells")
    distance, indices = cKDTree(constraints[:, :3] * ANISOTROPY).query(
        query * ANISOTROPY, k=min(8, constraints.shape[0])
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    exact = distance < 1e-12
    weights = 1.0 / np.maximum(distance, 1e-6) ** 2
    prediction = np.sum(weights * constraints[:, 3][indices], axis=1) / np.sum(weights, axis=1)
    exact_rows = np.flatnonzero(exact.any(axis=1))
    if exact_rows.size:
        first = np.argmax(exact[exact_rows], axis=1)
        prediction[exact_rows] = constraints[:, 3][indices[exact_rows, first]]
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("pseudo-well IDW feature is non-finite")
    return prediction


def _raw_features(
    geometry: GeometryBundle, mask: np.ndarray, constraints: np.ndarray
) -> np.ndarray:
    values = np.column_stack(
        [
            _idw(geometry.coordinates[mask], constraints),
            geometry.seismic[mask],
            geometry.coordinates[mask],
        ]
    ).astype(np.float64)
    if values.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("invalid R1 feature matrix")
    return values


def _fit_preprocess(train_raw: np.ndarray) -> tuple[list[Any], np.ndarray, float]:
    stats = [fit_zscore(train_raw[:, column]) for column in range(train_raw.shape[1])]
    transformed = np.column_stack(
        [normalize(train_raw[:, column], stats[column]) for column in range(train_raw.shape[1])]
    )
    restored = np.column_stack(
        [denormalize(transformed[:, column], stats[column]) for column in range(train_raw.shape[1])]
    )
    error = float(np.max(np.abs(restored - train_raw)))
    if error > 1e-10:
        raise ValueError(f"fold-train preprocessing round-trip failed: {error}")
    return stats, transformed.astype(np.float64), error


def _apply_preprocess(raw: np.ndarray, stats: Sequence[Any]) -> np.ndarray:
    return np.column_stack(
        [normalize(raw[:, column], stats[column]) for column in range(raw.shape[1])]
    ).astype(np.float64)


def _model_checkpoint_payload(model: Any) -> Mapping[str, Any]:
    return {
        "model_id": MODEL_CONFIG["model_id"],
        "weights": np.asarray(model.weights, dtype=np.float64).tolist(),
        "bias": float(model.bias),
        "update_count": int(model.update_count),
        "config": MODEL_CONFIG,
        "root_seed": ROOT_SEED,
    }


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Mapping[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("metrics require aligned non-empty arrays")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("metric inputs are non-finite")
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else None,
        "voxel_count": int(target.size),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
    }
    for name in ("rmse", "mae", "bias", "target_mean", "target_std", "prediction_mean", "prediction_std"):
        if not math.isfinite(float(metrics[name])):
            raise FloatingPointError(f"non-finite metric {name}")
    if metrics["r2"] is not None and not math.isfinite(float(metrics["r2"])):
        raise FloatingPointError("non-finite r2")
    return metrics


def _distance_band_metrics(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    distances: np.ndarray,
    edges: Sequence[float],
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (distances > lower) & (distances <= upper)
        if index == 0:
            selected = (distances >= lower) & (distances <= upper)
        support = int(selected.sum())
        if support < MIN_BAND_SUPPORT:
            raise ValueError(
                f"distance band {index} support {support} < required {MIN_BAND_SUPPORT}"
            )
        condition_metrics = {
            name: _regression_metrics(target[selected], prediction[selected])
            for name, prediction in predictions.items()
        }
        records.append(
            {
                "band_id": index,
                "lower_exclusive": lower if index else None,
                "upper_inclusive": None if math.isinf(upper) else upper,
                "voxel_count": support,
                "conditions": condition_metrics,
                "delta_rmse": {
                    "B1_minus_B0": float(
                        condition_metrics["B1"]["rmse"] - condition_metrics["B0"]["rmse"]
                    ),
                    "shuffled_minus_B0": float(
                        condition_metrics["shuffled"]["rmse"]
                        - condition_metrics["B0"]["rmse"]
                    ),
                },
            }
        )
    return records


def _blocked_result(
    prepared: R0Prepared,
    target_audit: Mapping[str, Any],
    code: str,
    message: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1_development_protocol_mechanism",
        "status": "blocked",
        "rankable": False,
        "reason": {"code": code, "message": message},
        "r0_manifest_hash": prepared.manifest["r0_manifest_hash"],
        "target_access_audit": dict(target_audit),
        "development_protocol_mechanism_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "formal_lane_status": "blocked",
        "test_firewall": prepared.manifest["test_firewall"],
    }


def run_r1(prepared: R0Prepared, data_dir: Path | None = None) -> Mapping[str, Any]:
    target, target_audit = load_development_target(prepared.geometry, data_dir)
    geometry = prepared.geometry
    train_values = target[prepared.train_mask][prepared.train_pseudo_indices]
    test_values = target[prepared.pseudo_test_mask][prepared.pseudo_test_indices]
    if train_values.size < 2 or test_values.size < 2:
        return _blocked_result(
            prepared, target_audit, "insufficient_pseudo_wells", "fewer than two conditions"
        )
    if np.unique(np.round(train_values, 10)).size < 2 or np.unique(np.round(test_values, 10)).size < 2:
        return _blocked_result(
            prepared,
            target_audit,
            "insufficient_distinct_pseudo_well_values",
            "pseudo-well values contain fewer than two distinct values",
        )

    train_constraints = _constraints(
        geometry, target, prepared.train_mask, prepared.train_pseudo_indices
    )
    correct_test_constraints = _constraints(
        geometry, target, prepared.pseudo_test_mask, prepared.pseudo_test_indices
    )
    shift = 1 + ROOT_SEED % (test_values.size - 1)
    shuffled_values = np.roll(test_values, shift)
    if np.array_equal(shuffled_values, test_values):
        return _blocked_result(
            prepared,
            target_audit,
            "shuffled_well_degenerate",
            "pre-registered seed2693 cyclic value permutation is identical",
        )
    shuffled_test_constraints = _constraints(
        geometry,
        target,
        prepared.pseudo_test_mask,
        prepared.pseudo_test_indices,
        shuffled_values,
    )

    train_raw = _raw_features(geometry, prepared.train_mask, train_constraints)
    stats, train_features, roundtrip_error = _fit_preprocess(train_raw)
    task_spec = p4.task_spec(MODE)
    discovered = discover_model("reconstruction", str(MODEL_CONFIG["model_id"]))
    model = discovered.build(
        task_spec,
        n_features=len(FEATURE_NAMES),
        n_training_samples=int(prepared.train_mask.sum()),
        learning_rate=float(MODEL_CONFIG["learning_rate"]),
        ridge_alpha=float(MODEL_CONFIG["ridge_alpha"]),
    )
    train_target = target[prepared.train_mask]
    losses: list[float] = []
    for _ in range(int(MODEL_CONFIG["updates"])):
        losses.append(float(model.train_batch((train_features, train_target))))
    if not losses or not np.all(np.isfinite(losses)):
        raise FloatingPointError("non-finite fixed-budget training history")
    checkpoint_payload = _model_checkpoint_payload(model)
    checkpoint_hash = hash_payload(checkpoint_payload)

    condition_constraints = {
        "B0": train_constraints,
        "B1": np.concatenate([train_constraints, correct_test_constraints]),
        "shuffled": np.concatenate([train_constraints, shuffled_test_constraints]),
    }
    predictions_all: dict[str, np.ndarray] = {}
    feature_hashes: dict[str, str] = {}
    for name, constraints in condition_constraints.items():
        raw = _raw_features(geometry, prepared.pseudo_test_mask, constraints)
        feature_hashes[name] = _hash_arrays(features=raw)
        features = _apply_preprocess(raw, stats)
        predictions_all[name] = np.asarray(model.predict_array(features), dtype=np.float64)
    if hash_payload(_model_checkpoint_payload(model)) != checkpoint_hash:
        raise RuntimeError("model state changed during condition inference")

    validation_global = np.flatnonzero(prepared.pseudo_test_mask)
    common_local = prepared.common_metric_mask[validation_global]
    metric_target = target[prepared.common_metric_mask]
    metric_predictions = {
        name: values[common_local] for name, values in predictions_all.items()
    }
    metrics = {
        name: _regression_metrics(metric_target, values)
        for name, values in metric_predictions.items()
    }
    metric_distances = prepared.pseudo_test_distances[common_local]
    band_metrics = _distance_band_metrics(
        metric_target, metric_predictions, metric_distances, prepared.distance_edges
    )
    difference = metric_predictions["B1"] - metric_predictions["B0"]
    sensitivity = {
        "conditional_feature_weight": float(np.asarray(model.weights)[0]),
        "B0_B1_changed_voxels": int(np.count_nonzero(np.abs(difference) > 1e-12)),
        "B0_B1_max_abs_prediction_difference": float(np.max(np.abs(difference))),
        "B0_B1_rms_prediction_difference": float(np.sqrt(np.mean(difference**2))),
        "feature_hash_B0_differs_from_B1": feature_hashes["B0"] != feature_hashes["B1"],
    }
    condition_aware = bool(
        abs(sensitivity["conditional_feature_weight"]) > 1e-12
        and sensitivity["B0_B1_changed_voxels"] > 0
        and sensitivity["B0_B1_max_abs_prediction_difference"] > 1e-12
        and sensitivity["feature_hash_B0_differs_from_B1"]
    )
    status = "passed" if condition_aware else "condition_unaware"
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1_development_protocol_mechanism",
        "status": status,
        "rankable": False,
        "root_seed": ROOT_SEED,
        "model_id": MODEL_CONFIG["model_id"],
        "model_config": MODEL_CONFIG,
        "same_fixed_model_checkpoint_all_conditions": True,
        "checkpoint": {
            "storage": "in_memory_compact_state_only_no_checkpoint_file",
            "sha256": checkpoint_hash,
            "update_count": int(model.update_count),
        },
        "training": {
            "fit_scope": "effective_fold_train_only",
            "effective_train_cells": int(prepared.train_mask.sum()),
            "pseudo_well_count": int(train_constraints.shape[0]),
            "updates": len(losses),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_minimum": min(losses),
            "preprocessing": "ml_framework.preprocess.fit_zscore fold-train only",
            "preprocess_stats": [item.to_dict() for item in stats],
            "preprocess_roundtrip_max_abs_error": roundtrip_error,
            "target_transform": "identity",
            "hpo_performed": False,
        },
        "conditions": {
            "B0": {
                "formal_name": "no_pseudo_test_PORO_condition",
                "pseudo_test_constraints": 0,
                "fixed_fold_train_constraints": int(train_constraints.shape[0]),
                "fixed_weak_tie_seismic_sampling_retained": True,
            },
            "B1": {
                "formal_name": "correct_synthetic_reference_revealed_pseudo_wells",
                "pseudo_test_constraints": int(correct_test_constraints.shape[0]),
                "values_hash": _hash_arrays(values=test_values),
            },
            "shuffled": {
                "formal_name": "seed2693_shuffled_pseudo_well_values_fixed_locations",
                "pseudo_test_constraints": int(shuffled_test_constraints.shape[0]),
                "shift": int(shift),
                "values_hash": _hash_arrays(values=shuffled_values),
                "non_identity": True,
            },
        },
        "common_metric_mask": {
            "sha256": prepared.manifest["hashes"]["common_metric_mask_hash"],
            "voxel_count": int(prepared.common_metric_mask.sum()),
            "exact_pseudo_test_cells_excluded_from_all_conditions": int(
                prepared.pseudo_test_indices.size
            ),
        },
        "metrics": metrics,
        "delta_rmse": {
            "B1_minus_B0": float(metrics["B1"]["rmse"] - metrics["B0"]["rmse"]),
            "B1_minus_shuffled": float(
                metrics["B1"]["rmse"] - metrics["shuffled"]["rmse"]
            ),
            "shuffled_minus_B0": float(
                metrics["shuffled"]["rmse"] - metrics["B0"]["rmse"]
            ),
            "B0_minus_B1_improvement": float(
                metrics["B0"]["rmse"] - metrics["B1"]["rmse"]
            ),
        },
        "well_information_gain_gate": (
            "B1 RMSE must be lower than both B0 and the fixed-location shuffled-value control"
        ),
        "well_information_gain_supported": bool(
            metrics["B1"]["rmse"] < metrics["B0"]["rmse"]
            and metrics["B1"]["rmse"] < metrics["shuffled"]["rmse"]
        ),
        "condition_sensitivity": sensitivity,
        "distance_bands": band_metrics,
        "prediction_hashes": {
            name: _hash_arrays(prediction=values)
            for name, values in metric_predictions.items()
        },
        "feature_hashes": feature_hashes,
        "r0_manifest_hash": prepared.manifest["r0_manifest_hash"],
        "split_hash": prepared.split_manifest["split_hash"],
        "config_hash": prepared.manifest["hashes"]["config_hash"],
        "target_access_audit": dict(target_audit),
        "test_firewall": prepared.manifest["test_firewall"],
        "development_protocol_mechanism_only": True,
        "synthetic_reference_revealed": True,
        "independent_measured_phie": False,
        "fresh_blind": False,
        "field_generalization": False,
        "formal_lane_status": "blocked",
        "ten_model_ranking_performed": False,
        "interpretation": (
            "This R1 result tests development-only conditional-channel mechanism. It is not "
            "a model ranking, field-generalization result, or fresh-blind test."
        ),
    }
    return result


def _report_text(r0: Mapping[str, Any], r1: Mapping[str, Any]) -> str:
    lines = [
        "# Reconstruction P5.1 R0/R1 development-only evidence",
        "",
        "## Scope",
        "",
        "This is `development_protocol_mechanism_only`. It is not a model ranking, "
        "fresh-blind test, known-holdout confirmation, or field-generalization claim.",
        "",
        "B0 is formally `no_pseudo_test_PORO_condition`. It retains the fixed project-level "
        "weak MD→TWT well tie used for seismic sampling, so it is not a claim of zero "
        "well-derived information.",
        "",
        "Pseudo-well PORO values are deterministic `synthetic/reference-revealed` Eclipse "
        "target samples, not independently measured PHIE.",
        "",
        "## R0",
        "",
        f"- Split hash: `{r0['hashes']['block_split_hash']}`",
        f"- Sample hash: `{r0['hashes']['sample_hash']}`",
        f"- Feature hash: `{r0['hashes']['feature_hash']}`",
        f"- Config hash: `{r0['hashes']['config_hash']}`",
        f"- Common metric-mask hash: `{r0['hashes']['common_metric_mask_hash']}`",
        "- Physical `test.h5`, global `well_log_seq`, known metrics and known predictions read: no.",
        "",
        "## R1",
        "",
        f"- Status: `{r1['status']}`; formal rankable: `false`.",
    ]
    if r1.get("metrics"):
        for name in ("B0", "B1", "shuffled"):
            metric = r1["metrics"][name]
            lines.append(
                f"- {name}: RMSE={metric['rmse']:.10f}, MAE={metric['mae']:.10f}, "
                f"bias={metric['bias']:.10f}, R²={metric['r2']:.10f}."
            )
        lines.extend(
            [
                f"- ΔRMSE(B1-B0): {r1['delta_rmse']['B1_minus_B0']:.10f}.",
                f"- ΔRMSE(B1-shuffled): {r1['delta_rmse']['B1_minus_shuffled']:.10f}.",
                f"- Well-information gain supported on this development block: "
                f"`{str(r1['well_information_gain_supported']).lower()}`.",
                f"- Shared in-memory checkpoint hash: `{r1['checkpoint']['sha256']}`.",
            ]
        )
        if any(r1["metrics"][name]["r2"] is not None and r1["metrics"][name]["r2"] <= 0 for name in ("B0", "B1", "shuffled")):
            lines.append(
                "- At least one R² is non-positive; this cannot support a spatial-generalization-success claim."
            )
    else:
        lines.append(f"- Block reason: `{r1['reason']['code']}` — {r1['reason']['message']}.")
    lines.extend(
        [
            "",
            "Formal/fresh-blind/field-generalization lanes remain blocked. The historical "
            "holdout was not opened, consumed, scored, or relabelled by R0/R1.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    prepared: R0Prepared,
    r1: Mapping[str, Any] | None,
    output_dir: Path,
) -> Mapping[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    r0_path = atomic_write_json(output_dir / "r0_manifest.json", prepared.manifest)
    roles = [(r0_path.name, "R0 provenance/split/firewall contract")]
    if r1 is not None:
        r1_path = atomic_write_json(output_dir / "r1_results.json", r1)
        report_path = output_dir / "P5_R01_REPORT.md"
        report_path.write_text(_report_text(prepared.manifest, r1), encoding="utf-8")
        roles.extend(
            [
                (r1_path.name, "R1 development-only mechanism metrics"),
                (report_path.name, "human-readable scientific boundary report"),
            ]
        )
    manifest = ArtifactManifest(run_id="reconstruction-p5.1-r01", root=output_dir)
    for relative_path, role in roles:
        manifest.register(relative_path, role=role)
    manifest_path = manifest.write("artifact_manifest.json")
    manifest.verify()
    return {
        "output_dir": _relative(output_dir),
        "artifact_manifest": _relative(manifest_path),
        "artifacts": manifest.to_dict()["artifacts"],
    }


def write_data_gate_blocked(output_dir: Path) -> Mapping[str, Any]:
    """Persist portable fail-closed evidence when development HDF5 is absent."""
    reason = {
        "code": "development_train_h5_not_provisioned",
        "message": (
            "The read-only physical train.h5 development asset is not provisioned in this "
            "worktree. R0 sample/block/mask hashes and R1 metrics cannot be computed without "
            "building data or reusing an unaudited cache, both forbidden by this goal."
        ),
    }
    source_records = [
        _source_record("_pipelines/02_task_datasets/reconstruction/build_dataset.py"),
        _source_record("_pipelines/02_task_datasets/reconstruction/build_summary.json"),
        _source_record("_pipelines/02_task_datasets/reconstruction/model_inspection.json"),
        _source_record("_pipelines/02_task_datasets/reconstruction/p4_reconstruction.py"),
        _source_record("_models/reconstruction/reconstruction_linear_sgd.py"),
    ]
    build_summary = json.loads((HERE / "build_summary.json").read_text(encoding="utf-8"))
    model_inspection = json.loads(
        (HERE / "model_inspection.json").read_text(encoding="utf-8")
    )
    r0 = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R0_zero_training_contract",
        "status": "blocked",
        "rankable": False,
        "reason": reason,
        "required_data_role": "_data/processed/reconstruction/train.h5",
        "sources": source_records,
        "label": {
            "name": "Volve final Eclipse PORO on active cells",
            "source": "Eclipse reference target",
            "units": "fraction",
            "synthetic_reference_revealed_condition_values": True,
            "independent_measured_phie": False,
        },
        "rms_cross_check": {
            "role": "value-multiset cross-check only; no RMS spatial mapping is claimed",
            "member": model_inspection["rms_reference"]["member"],
            "format": model_inspection["rms_reference"]["format"],
            "exact_nonzero_porosity_multiset_match": model_inspection["rms_reference"][
                "porosity_multiset_exactly_matches_eclipse_nonzero_phif_nw"
            ],
            "spatial_mapping": model_inspection["rms_reference"]["spatial_mapping"],
        },
        "coordinates": {
            "source": "Eclipse active-cell centres",
            "channels": ["x_normalized", "y_normalized", "depth_normalized"],
            "bounds": build_summary["coordinate_bounds"],
            "normalization": "fixed full-grid minmax geometry transform",
            "fold_train_statistical_preprocessing_required": True,
        },
        "seismic": {
            "source": "ST0202 post-stack time SEG-Y sampled with fixed project weak tie",
            "fixed_project_weak_well_tie_retained": True,
            "attribute_definitions": build_summary["seismic"]["attribute_definitions"],
        },
        "weak_tie": {
            "method": build_summary["weak_tie"]["method"],
            "depth_coordinate_warning": build_summary["weak_tie"]["depth_coordinate_warning"],
            "B0_is_not_zero_well_derived_information": True,
        },
        "existing_sparse_poro": {
            "n_observation_rows": build_summary["sparse_wells"]["n_observation_rows"],
            "n_wells_with_constraints": build_summary["sparse_wells"]["n_wells_with_constraints"],
            "value_source": "Eclipse reference PORO sampled at mapped cells",
            "used_by_r1": False,
        },
        "conditions": {
            "B0": {
                "formal_name": "no_pseudo_test_PORO_condition",
                "fixed_weak_tie_seismic_sampling_retained": True,
            },
            "B1": "correct synthetic/reference-revealed development pseudo-well PORO",
            "shuffled": "same locations with seed2693 non-identity value permutation",
        },
        "planned_split": {
            "development_i_blocks": list(DEVELOPMENT_I_BLOCKS),
            "requested_folds": REQUESTED_FOLDS,
            "pseudo_test_fold_id": PSEUDO_TEST_FOLD_ID,
            "buffer_blocks": BUFFER_BLOCKS,
            "status": "not_instantiated_without_development_geometry",
        },
        "feature_contract": {
            "names": list(FEATURE_NAMES),
            "denoise": "ml_framework.preprocess.denoise_identity",
            "statistical_preprocessing": "ml_framework.preprocess.fit_zscore fold-train only",
            "target_transform": "identity",
        },
        "hashes": {
            "feature_hash": hash_payload(list(FEATURE_NAMES)),
            "config_hash": hash_payload(
                {**MODEL_CONFIG, "root_seed": ROOT_SEED, "feature_names": list(FEATURE_NAMES)}
            ),
            "sample_hash": None,
            "block_split_hash": None,
            "common_metric_mask_hash": None,
        },
        "test_firewall": {
            "physical_test_h5_opened": False,
            "known_or_frozen_arrays_read": False,
            "known_or_frozen_metrics_read": False,
            "known_or_frozen_predictions_read": False,
            "global_well_log_seq_read": False,
            "historical_cache_used": False,
        },
        "development_protocol_mechanism_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "formal_lane_status": "blocked",
    }
    r0["r0_manifest_hash"] = hash_payload(r0)
    r1 = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1_development_protocol_mechanism",
        "status": "blocked",
        "rankable": False,
        "reason": reason,
        "metrics": None,
        "r0_manifest_hash": r0["r0_manifest_hash"],
        "development_protocol_mechanism_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "formal_lane_status": "blocked",
        "ten_model_ranking_performed": False,
        "test_firewall": r0["test_firewall"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "r0_manifest.json", r0)
    atomic_write_json(output_dir / "r1_results.json", r1)
    report = "\n".join(
        [
            "# Reconstruction P5.1 R0/R1 development-only evidence",
            "",
            "## Status",
            "",
            "R0 and R1 are `blocked/not_rankable`: the physical development `train.h5` is "
            "not provisioned in this worktree. No build, cache reuse, training, prediction or "
            "metric backfill was performed.",
            "",
            "The source-level audit still confirms that pseudo-well PORO would be "
            "`synthetic/reference-revealed` Eclipse target values, not independently measured "
            "PHIE. B0 would retain the fixed project weak MD→TWT tie and therefore would not "
            "mean zero well-derived information.",
            "",
            "Physical `test.h5`, historical predictions/metrics and global `well_log_seq` were "
            "not opened. Formal, fresh-blind and field-generalization lanes remain blocked.",
            "",
        ]
    )
    (output_dir / "P5_R01_REPORT.md").write_text(report, encoding="utf-8")
    artifact_manifest = ArtifactManifest(run_id="reconstruction-p5.1-r01-blocked", root=output_dir)
    for name, role in (
        ("r0_manifest.json", "blocked R0 source/data-gate audit"),
        ("r1_results.json", "blocked R1 data-gate result"),
        ("P5_R01_REPORT.md", "human-readable blocked evidence"),
    ):
        artifact_manifest.register(name, role=role)
    manifest_path = artifact_manifest.write("artifact_manifest.json")
    artifact_manifest.verify()
    return {
        "output_dir": _relative(output_dir),
        "artifact_manifest": _relative(manifest_path),
        "artifacts": artifact_manifest.to_dict()["artifacts"],
        "r0_status": "blocked",
        "r1_status": "blocked",
        "reason": reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("audit", "write the zero-training R0 development contract"),
        ("run", "run R0 then one fixed-model development-only R1"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--data-dir", type=Path)
        child.add_argument("--output-dir", type=Path, default=HERE / "p5_r01_evidence")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        prepared = prepare_r0(args.data_dir)
    except FileNotFoundError:
        blocked = write_data_gate_blocked(args.output_dir)
        print(
            json.dumps(
                {
                    "command": args.command,
                    "fresh_blind": False,
                    "physical_test_h5_opened": False,
                    **blocked,
                },
                indent=2,
            )
        )
        raise SystemExit(2)
    result = None if args.command == "audit" else run_r1(prepared, args.data_dir)
    outputs = write_outputs(prepared, result, args.output_dir)
    print(
        json.dumps(
            {
                "command": args.command,
                "r0_status": "passed",
                "r1_status": None if result is None else result["status"],
                "fresh_blind": False,
                "physical_test_h5_opened": False,
                **outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
