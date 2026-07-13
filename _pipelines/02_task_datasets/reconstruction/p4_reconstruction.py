#!/usr/bin/env python3
"""P4 contract-first orchestration for Volve 3-D porosity reconstruction.

The historical ``baseline.py`` remains the canonical Ridge+IDW evidence.  This
module adds the frozen-test/development-CV experiment plumbing required by P4
without changing those metrics.  Conditional and strict reconstruction are
different tasks and never share result directories or metric names.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import ArtifactManifest, atomic_write_json, hash_file, hash_payload  # noqa: E402
from ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from ml_framework.contracts import TaskSpec  # noqa: E402
from ml_framework.cv import run_development_cv  # noqa: E402
from ml_framework.hpo import HPOPlan, run_fixed_trials  # noqa: E402
from ml_framework.lifecycle import ExperimentLifecycle, ExperimentState  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from ml_framework.preprocess import NormStats, denoise_identity, denormalize, fit_zscore, normalize  # noqa: E402
from ml_framework.run_layout import create_run_layout  # noqa: E402
from ml_framework.seeding import DEFAULT_ROOT_SEED, SeedTree, seed_everything  # noqa: E402
from ml_framework.splits import Fold, SplitManifest, validate_manifest  # noqa: E402
from ml_framework.trainer import StepResult, TrainerConfig, TrainerState, train_with_validation  # noqa: E402


MODES = ("conditional", "strict")
MODEL_NAMES = ("ridge_linear", "reconstruction_linear_sgd", "reconstruction_tiny_mlp")
SEISMIC_FEATURES = (
    "seismic_amplitude",
    "seismic_local_rms",
    "seismic_vertical_gradient",
)
COORDINATE_FEATURES = ("x_normalized", "y_normalized", "depth_normalized")


@dataclass(frozen=True)
class Protocol:
    mode: str
    task_id: str
    target_name: str
    label_version: str
    development_i_blocks: tuple[int, ...]
    guard_i_blocks: tuple[int, ...]
    test_i_blocks: tuple[int, ...]
    idw_feature_name: str | None
    metric_prefix: str
    conditional_test_constraints: bool


PROTOCOLS = {
    "conditional": Protocol(
        mode="conditional",
        task_id="volve_porosity_conditional_reconstruction",
        target_name="conditional_eclipse_porosity",
        label_version="volve-eclipse-poro-conditional-v1",
        development_i_blocks=(0, 1, 2, 3),
        guard_i_blocks=(),
        test_i_blocks=(4, 5),
        idw_feature_name="conditional_idw_porosity",
        metric_prefix="conditional",
        conditional_test_constraints=True,
    ),
    "strict": Protocol(
        mode="strict",
        task_id="volve_porosity_strict_spatial_reconstruction",
        target_name="strict_eclipse_porosity",
        label_version="volve-eclipse-poro-strict-v1",
        development_i_blocks=(4, 5),
        guard_i_blocks=(3,),
        test_i_blocks=(0, 1, 2),
        idw_feature_name=None,
        metric_prefix="strict",
        conditional_test_constraints=False,
    ),
}


@dataclass(frozen=True)
class PatchLocation:
    sample_id: str
    source_split: str
    source_key: str
    k_block: int
    j_block: int
    i_block: int
    patch_start_kji: tuple[int, int, int]
    patch_shape_kji: tuple[int, int, int]


@dataclass
class PatchRecord:
    location: PatchLocation
    seismic_patch: np.ndarray
    label: np.ndarray
    well_log_seq: np.ndarray

    @property
    def sample_id(self) -> str:
        return self.location.sample_id

    @property
    def i_block(self) -> int:
        return self.location.i_block

    @property
    def j_block(self) -> int:
        return self.location.j_block

    @property
    def k_block(self) -> int:
        return self.location.k_block


@dataclass
class FlatCells:
    sample_ids: np.ndarray
    indices_kji: np.ndarray
    seismic: np.ndarray
    coordinates: np.ndarray
    observed_mask: np.ndarray
    target: np.ndarray
    volume_shape_kji: tuple[int, int, int]


@dataclass
class PreparedFold:
    feature_names: tuple[str, ...]
    train_features: np.ndarray
    train_target: np.ndarray
    validation_features: np.ndarray
    validation_target: np.ndarray
    validation_metric_mask: np.ndarray
    validation_cells: FlatCells
    preprocess_report: dict[str, Any]
    constraint_audit: dict[str, Any]


def protocol(mode: str) -> Protocol:
    if mode not in PROTOCOLS:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return PROTOCOLS[mode]


def metric_names(mode: str) -> dict[str, str]:
    prefix = protocol(mode).metric_prefix
    return {
        "rmse": f"{prefix}_rmse",
        "mae": f"{prefix}_mae",
        "bias": f"{prefix}_bias",
        "r2": f"{prefix}_r2",
        "pearson_r": f"{prefix}_pearson_r",
        "spectral_log_rmse": f"{prefix}_spectral_log_rmse",
        "out_of_range_rate": f"{prefix}_out_of_range_rate",
    }


def hpo_plan() -> HPOPlan:
    """Return the frozen optional HPO plan; no Optuna import is required."""
    return HPOPlan(
        sanity_trials=8,
        pilot_trials=20,
        top_configs=3,
        confirm_seeds=3,
        sampler="random_then_tpe",
        pruner="nop",
        direction="minimize",
    )


def fixed_baseline_configs() -> list[dict[str, Any]]:
    """Dependency-light fixed configurations; these are not an HPO result."""
    return [
        {"model": "ridge_linear", "learning_rate": 0.01, "ridge_alpha": 10.0},
        {"model": "reconstruction_linear_sgd", "learning_rate": 0.01, "ridge_alpha": 0.1},
        {"model": "reconstruction_tiny_mlp", "learning_rate": 0.005, "ridge_alpha": 0.1},
    ]


def task_spec(mode: str) -> TaskSpec:
    """Build one mode-specific TaskSpec; no mutable TaskSpec is shared."""
    active = protocol(mode)
    metrics = metric_names(mode)
    input_whitelist = (
        *((active.idw_feature_name,) if active.idw_feature_name is not None else ()),
        *SEISMIC_FEATURES,
        *COORDINATE_FEATURES,
    )
    forbidden = (
        active.target_name,
        "conditional_eclipse_porosity",
        "strict_eclipse_porosity",
        "eclipse_reference_porosity",
        "rms_reference_porosity",
        "future_porosity",
        "derived_reference_feature",
        "test_region_well_porosity",
    )
    return TaskSpec(
        track_id="reconstruction",
        task_id=active.task_id,
        task_type="reconstruction",
        input_modalities=(
            ("post_stack_seismic", "coordinates", "sparse_well_constraints")
            if mode == "conditional"
            else ("post_stack_seismic", "coordinates")
        ),
        targets=(active.target_name,),
        units={active.target_name: "fraction"},
        label_version=active.label_version,
        target_masks={active.target_name: "eclipse_active_and_not_exact_well_cell"},
        group_keys=("i_block", "k_block"),
        target_transform={active.target_name: "identity"},
        inverse_transform={active.target_name: "identity"},
        train_loss={active.target_name: {"name": "mse", "reduction": "valid_label_mean"}},
        inference_transform={active.target_name: "identity"},
        threshold_policy={},
        calibration_policy={},
        primary_metrics=(metrics["rmse"],),
        metric_directions={
            metrics["rmse"]: "minimize",
            metrics["mae"]: "minimize",
            metrics["bias"]: "minimize",
            metrics["r2"]: "maximize",
            metrics["pearson_r"]: "maximize",
            metrics["spectral_log_rmse"]: "minimize",
            metrics["out_of_range_rate"]: "minimize",
        },
        secondary_metrics=(
            metrics["mae"],
            metrics["bias"],
            metrics["r2"],
            metrics["pearson_r"],
            metrics["spectral_log_rmse"],
        ),
        guardrail_metrics=(metrics["out_of_range_rate"],),
        spatial_buffer={"fold_axis": "k_block", "buffer_blocks": 1, "test_axis": "i_block"},
        hpo={**asdict(hpo_plan()), "primary_metric": metrics["rmse"]},
        visualizer_id=f"reconstruction_{mode}_volume_diagnostics",
        required_figures=(
            f"{mode}_inline_truth_prediction_residual",
            f"{mode}_crossline_truth_prediction_residual",
            f"{mode}_time_depth_truth_prediction_residual",
            f"{mode}_error_distribution",
            f"{mode}_attribute_distribution",
            f"{mode}_spectral_diagnostic",
        ),
        input_whitelist=input_whitelist,
        forbidden_inputs=forbidden,
        metadata={
            "evaluation_mode": mode,
            "development_i_blocks": list(active.development_i_blocks),
            "guard_i_blocks": list(active.guard_i_blocks),
            "frozen_test_i_blocks": list(active.test_i_blocks),
            "conditional_reconstruction_not_strict_holdout": mode == "conditional",
            "strict_test_constraint_policy": (
                "not_applicable_conditional_mode"
                if mode == "conditional"
                else (
                    "all sparse porosity values are excluded from strict model inputs because the current "
                    "values were sampled from Eclipse reference target cells; test/guard/future truth is also forbidden"
                )
            ),
            "output_parameterization": "identity",
            "test_history": "legacy spatial block has already been reported; not a new blind field test",
        },
    )


def assert_feature_contract(mode: str, feature_names: Sequence[str]) -> None:
    spec = task_spec(mode)
    names = set(feature_names)
    unknown = sorted(names - set(spec.input_whitelist))
    leaked = sorted(names & set(spec.forbidden_inputs))
    if leaked:
        raise ValueError(f"{mode} model features contain forbidden truth/future fields: {leaked}")
    if unknown:
        raise ValueError(f"{mode} model features are outside TaskSpec input_whitelist: {unknown}")


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    if data_dir is not None:
        return data_dir.resolve()
    override = os.environ.get("RECONSTRUCTION_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "_data" / "processed" / "reconstruction"


def _location(source_split: str, source_key: str, meta: Mapping[str, Any]) -> PatchLocation:
    k_block, j_block, i_block = (int(value) for value in meta["patch_index_kji"])
    start = tuple(int(value) for value in meta["patch_start_kji"])
    shape = tuple(int(value) for value in meta["patch_shape_kji"])
    return PatchLocation(
        sample_id=f"k{k_block:02d}_j{j_block:02d}_i{i_block:02d}",
        source_split=source_split,
        source_key=source_key,
        k_block=k_block,
        j_block=j_block,
        i_block=i_block,
        patch_start_kji=start,
        patch_shape_kji=shape,
    )


def scan_patch_catalog(data_dir: Path | None = None) -> list[PatchLocation]:
    """Read split metadata only; this does not access labels or model inputs."""
    root = resolve_data_dir(data_dir)
    missing = [root / f"{split}.h5" for split in ("train", "test") if not (root / f"{split}.h5").is_file()]
    if missing:
        raise FileNotFoundError("missing unified reconstruction dataset: " + ", ".join(str(path) for path in missing))
    locations: list[PatchLocation] = []
    for source_split in ("train", "test"):
        with h5py.File(root / f"{source_split}.h5", "r") as handle:
            for source_key in sorted(handle):
                meta = json.loads(handle[source_key].attrs["meta"])
                locations.append(_location(source_split, source_key, meta))
    sample_ids = [item.sample_id for item in locations]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("patch catalog sample IDs are not globally unique")
    return locations


def load_patch_records(
    i_blocks: Iterable[int],
    data_dir: Path | None = None,
) -> list[PatchRecord]:
    """Load only requested I-block arrays from the unified HDF5 schema."""
    allowed = set(int(value) for value in i_blocks)
    if not allowed:
        raise ValueError("at least one I-block is required")
    root = resolve_data_dir(data_dir)
    records: list[PatchRecord] = []
    for source_split in ("train", "test"):
        path = root / f"{source_split}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            for source_key in sorted(handle):
                group = handle[source_key]
                meta = json.loads(group.attrs["meta"])
                location = _location(source_split, source_key, meta)
                if location.i_block not in allowed:
                    continue
                patch = np.asarray(group["seismic_patch"][()], dtype=np.float32)
                label = np.asarray(group["label"][()], dtype=np.float32)
                wells = np.asarray(group["well_log_seq"][()], dtype=np.float32)
                if patch.shape[0] != 9 or label.shape != patch.shape[1:]:
                    raise ValueError(f"invalid unified sample shapes for {location.sample_id}")
                records.append(PatchRecord(location, patch, label, wells))
    if not records:
        raise ValueError(f"no reconstruction patches found for I-blocks {sorted(allowed)}")
    return records


def _contiguous_buckets(groups: Sequence[int], n_splits: int) -> list[tuple[int, ...]]:
    q, remainder = divmod(len(groups), n_splits)
    buckets: list[tuple[int, ...]] = []
    start = 0
    for index in range(n_splits):
        size = q + (1 if index < remainder else 0)
        buckets.append(tuple(groups[start : start + size]))
        start += size
    return buckets


def build_spatial_manifest(
    mode: str,
    catalog: Sequence[PatchLocation],
    *,
    requested_n_splits: int = 5,
    buffer_blocks: int = 1,
) -> SplitManifest:
    """Freeze contiguous I-block test first, then buffered K-block CV."""
    active = protocol(mode)
    if requested_n_splits < 2:
        raise ValueError("requested_n_splits must be >=2")
    if buffer_blocks < 0:
        raise ValueError("buffer_blocks must be >=0")
    test = [item for item in catalog if item.i_block in active.test_i_blocks]
    development = [item for item in catalog if item.i_block in active.development_i_blocks]
    guard = [item for item in catalog if item.i_block in active.guard_i_blocks]
    if not test or not development:
        raise ValueError(f"{mode} catalog lacks frozen test or development patches")
    test_blocks = sorted({item.i_block for item in test})
    if test_blocks != list(range(test_blocks[0], test_blocks[-1] + 1)):
        raise ValueError("frozen test I-blocks must form one continuous interval")
    k_groups = sorted({item.k_block for item in development})
    effective = min(requested_n_splits, len(k_groups))
    downgrade: list[str] = []
    if effective < requested_n_splits:
        downgrade.append(f"only {len(k_groups)} independent development K-blocks are available")

    while effective >= 2:
        buckets = _contiguous_buckets(k_groups, effective)
        viable = True
        for validation_k in buckets:
            purged_k = {
                candidate
                for candidate in k_groups
                if candidate not in validation_k
                and min(abs(candidate - held) for held in validation_k) <= buffer_blocks
            }
            fit_k = set(k_groups) - set(validation_k) - purged_k
            if not fit_k:
                viable = False
                break
        if viable:
            break
        effective -= 1
    if effective < 2:
        raise ValueError("spatial buffer leaves fewer than two viable development folds")
    if effective < min(requested_n_splits, len(k_groups)):
        downgrade.append(f"K-block buffer={buffer_blocks} limits viable folds to {effective}")

    dev_ids = {item.sample_id for item in development}
    folds: list[Fold] = []
    for fold_id, validation_k in enumerate(_contiguous_buckets(k_groups, effective)):
        validation_k_set = set(validation_k)
        validation_ids = tuple(item.sample_id for item in development if item.k_block in validation_k_set)
        candidate_train_ids = tuple(item.sample_id for item in development if item.k_block not in validation_k_set)
        purged_k = {
            candidate
            for candidate in k_groups
            if candidate not in validation_k_set
            and min(abs(candidate - held) for held in validation_k_set) <= buffer_blocks
        }
        purged_ids = tuple(item.sample_id for item in development if item.k_block in purged_k)
        effective_ids = tuple(item for item in candidate_train_ids if item not in set(purged_ids))
        if not effective_ids:
            raise ValueError(f"fold {fold_id} has no train patches after spatial purge")
        folds.append(
            Fold(
                fold_id=fold_id,
                train_groups=tuple(f"dev_k{value:02d}" for value in k_groups if value not in validation_k_set),
                validation_groups=tuple(f"dev_k{value:02d}" for value in validation_k),
                train_sample_ids=candidate_train_ids,
                validation_sample_ids=validation_ids,
                purge={
                    "axis": "k_block",
                    "buffer_blocks": buffer_blocks,
                    "purged_k_blocks": sorted(purged_k),
                    "purged_train_sample_ids": list(purged_ids),
                    "effective_train_sample_ids": list(effective_ids),
                },
                support={
                    "candidate_train_patches": len(candidate_train_ids),
                    "purged_train_patches": len(purged_ids),
                    "effective_train_patches": len(effective_ids),
                    "validation_patches": len(validation_ids),
                },
            )
        )
    manifest = SplitManifest(
        manifest_version="p4-reconstruction-v1",
        group_key="continuous_i_test_then_buffered_k_cv",
        requested_n_splits=requested_n_splits,
        effective_n_splits=effective,
        downgrade_reason="; ".join(downgrade) or None,
        test_groups=tuple(f"test_i{value:02d}" for value in test_blocks),
        test_sample_ids=tuple(item.sample_id for item in test),
        development_groups=tuple(f"dev_k{value:02d}" for value in k_groups),
        development_sample_ids=tuple(item.sample_id for item in development),
        folds=tuple(folds),
        metadata={
            "evaluation_mode": mode,
            "freeze_order": "continuous test I-blocks frozen before development CV",
            "development_i_blocks": list(active.development_i_blocks),
            "guard_i_blocks": list(active.guard_i_blocks),
            "guard_sample_ids": [item.sample_id for item in guard],
            "test_i_blocks": test_blocks,
            "test_i_blocks_contiguous": True,
            "fold_axis": "k_block",
            "buffer_blocks": buffer_blocks,
            "purge_encoding_note": (
                "shared SplitManifest requires candidate train+validation coverage; effective fit IDs "
                "are the strict subset in fold.purge.effective_train_sample_ids"
            ),
            "oof_unit": "patch",
            "development_sample_count": len(dev_ids),
        },
    )
    validate_manifest(manifest)
    validate_buffered_manifest(manifest, catalog)
    return manifest


def validate_buffered_manifest(manifest: SplitManifest, catalog: Sequence[PatchLocation]) -> None:
    by_id = {item.sample_id: item for item in catalog}
    seen: list[str] = []
    for fold in manifest.folds:
        validation_ids = set(fold.validation_sample_ids)
        effective_ids = set(fold.purge["effective_train_sample_ids"])
        purged_ids = set(fold.purge["purged_train_sample_ids"])
        if effective_ids & validation_ids or purged_ids & validation_ids:
            raise ValueError(f"fold {fold.fold_id} buffer/train overlaps validation")
        if effective_ids | purged_ids != set(fold.train_sample_ids):
            raise ValueError(f"fold {fold.fold_id} effective+purged IDs do not equal candidate train IDs")
        validation_k = {by_id[item].k_block for item in validation_ids}
        buffer_blocks = int(fold.purge["buffer_blocks"])
        for sample_id in effective_ids:
            if min(abs(by_id[sample_id].k_block - value) for value in validation_k) <= buffer_blocks:
                raise ValueError(f"fold {fold.fold_id} effective train sample violates K buffer")
        seen.extend(fold.validation_sample_ids)
    if sorted(seen) != sorted(manifest.development_sample_ids):
        raise ValueError("buffered CV OOF patch coverage is not exactly once")


def _select_records(records: Sequence[PatchRecord], sample_ids: Iterable[str]) -> list[PatchRecord]:
    wanted = set(sample_ids)
    selected = [item for item in records if item.sample_id in wanted]
    if {item.sample_id for item in selected} != wanted:
        missing = sorted(wanted - {item.sample_id for item in selected})
        raise ValueError(f"record selection is missing sample IDs: {missing}")
    return selected


def flatten_records(records: Sequence[PatchRecord]) -> FlatCells:
    if not records:
        raise ValueError("cannot flatten zero patches")
    sample_ids: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    seismic: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    target: list[np.ndarray] = []
    max_shape = np.zeros(3, dtype=np.int64)
    for record in records:
        patch = record.seismic_patch
        active = patch[8].ravel() > 0.5
        start = np.asarray(record.location.patch_start_kji, dtype=np.int64)
        patch_shape = np.asarray(record.location.patch_shape_kji, dtype=np.int64)
        max_shape = np.maximum(max_shape, start + patch_shape)
        # Empty active-cell patches remain split/OOF samples, but naturally
        # contribute zero voxels to fitting and metrics.
        if not np.any(active):
            continue
        local_kji = np.indices(record.label.shape).reshape(3, -1).T[active]
        global_kji = local_kji + start
        sample_ids.append(np.full(int(active.sum()), record.sample_id, dtype=f"U{max(1, len(record.sample_id))}"))
        indices.append(global_kji)
        seismic.append(denoise_identity(patch[0:3].reshape(3, -1).T[active]).astype(np.float64))
        coordinates.append(patch[3:6].reshape(3, -1).T[active].astype(np.float64))
        observed.append((patch[7].ravel() > 0.5)[active])
        target.append(record.label.ravel()[active].astype(np.float64))
    if not target:
        raise ValueError("selected patches contain zero active Eclipse cells")
    return FlatCells(
        sample_ids=np.concatenate(sample_ids),
        indices_kji=np.concatenate(indices),
        seismic=np.concatenate(seismic),
        coordinates=np.concatenate(coordinates),
        observed_mask=np.concatenate(observed),
        target=np.concatenate(target),
        volume_shape_kji=tuple(int(value) for value in max_shape),
    )


def constraints_from_records(records: Sequence[PatchRecord]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record in records:
        patch = record.seismic_patch
        active = patch[8].ravel() > 0.5
        observed = (patch[7].ravel() > 0.5) & (patch[8].ravel() > 0.5)
        if np.any(observed):
            xyz = patch[3:6].reshape(3, -1).T[observed]
            porosity = record.label.ravel()[observed, None]
            rows.append(np.concatenate([xyz, porosity], axis=1).astype(np.float64))
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    values = np.concatenate(rows)
    unique: dict[tuple[float, float, float], float] = {}
    for row in values:
        key = tuple(float(value) for value in row[:3])
        if key in unique and not math.isclose(unique[key], float(row[3]), abs_tol=1e-7):
            raise ValueError("duplicate well coordinate has inconsistent porosity")
        unique[key] = float(row[3])
    return np.asarray([(*key, value) for key, value in unique.items()], dtype=np.float64)


def idw_predict(coordinates: np.ndarray, constraints: np.ndarray, *, fallback: float | None = None) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    if constraints.shape[0] == 0:
        if fallback is None or not math.isfinite(fallback):
            raise ValueError("IDW has zero constraints and no finite fold-train fallback")
        return np.full(coordinates.shape[0], fallback, dtype=np.float64)
    anisotropy = np.asarray([1.0, 1.0, 3.0], dtype=np.float64)
    tree = cKDTree(constraints[:, :3] * anisotropy)
    distance, indices = tree.query(coordinates * anisotropy, k=min(8, constraints.shape[0]))
    if distance.ndim == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    exact = distance < 1e-10
    weights = 1.0 / np.maximum(distance, 1e-6) ** 2
    prediction = np.sum(weights * constraints[:, 3][indices], axis=1) / np.sum(weights, axis=1)
    exact_rows = np.flatnonzero(exact.any(axis=1))
    if exact_rows.size:
        first = np.argmax(exact[exact_rows], axis=1)
        prediction[exact_rows] = constraints[:, 3][indices[exact_rows, first]]
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("IDW produced non-finite values")
    return prediction


def _raw_features(
    mode: str,
    cells: FlatCells,
    constraints: np.ndarray,
    *,
    fallback: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    active = protocol(mode)
    names = (
        *((active.idw_feature_name,) if active.idw_feature_name is not None else ()),
        *SEISMIC_FEATURES,
        *COORDINATE_FEATURES,
    )
    assert_feature_contract(mode, names)
    if active.idw_feature_name is None:
        values = np.column_stack([cells.seismic, cells.coordinates]).astype(np.float64)
    else:
        idw = idw_predict(cells.coordinates, constraints, fallback=fallback)
        values = np.column_stack([idw, cells.seismic, cells.coordinates]).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("raw reconstruction features are non-finite")
    return values, names


def prepare_fold(
    mode: str,
    fold: Fold,
    development_records: Sequence[PatchRecord],
) -> PreparedFold:
    """Fit every preprocessing statistic and well filter on fold-train only."""
    effective_ids = tuple(fold.purge["effective_train_sample_ids"])
    train_records = _select_records(development_records, effective_ids)
    validation_records = _select_records(development_records, fold.validation_sample_ids)
    train_cells = flatten_records(train_records)
    validation_cells = flatten_records(validation_records)
    constraints = constraints_from_records(train_records)
    fallback = float(np.mean(train_cells.target))
    train_raw, names = _raw_features(mode, train_cells, constraints, fallback=fallback)
    validation_raw, validation_names = _raw_features(mode, validation_cells, constraints, fallback=fallback)
    if names != validation_names:
        raise RuntimeError("train/validation feature names differ")
    stats = [fit_zscore(train_raw[:, column]) for column in range(train_raw.shape[1])]
    train_features = np.column_stack(
        [normalize(train_raw[:, column], stats[column]) for column in range(train_raw.shape[1])]
    )
    validation_features = np.column_stack(
        [normalize(validation_raw[:, column], stats[column]) for column in range(validation_raw.shape[1])]
    )
    restored = np.column_stack(
        [denormalize(train_features[:, column], stats[column]) for column in range(train_features.shape[1])]
    )
    roundtrip_error = float(np.max(np.abs(restored - train_raw)))
    if roundtrip_error > 1e-10:
        raise ValueError(f"fold preprocessing round-trip failed: {roundtrip_error}")
    metric_mask = ~validation_cells.observed_mask
    if not np.any(metric_mask):
        raise ValueError(f"fold {fold.fold_id} has no non-constraint validation cells")
    return PreparedFold(
        feature_names=names,
        train_features=np.asarray(train_features, dtype=np.float64),
        train_target=train_cells.target,
        validation_features=np.asarray(validation_features, dtype=np.float64),
        validation_target=validation_cells.target,
        validation_metric_mask=metric_mask,
        validation_cells=validation_cells,
        preprocess_report={
            "fit_scope": "fold.purge.effective_train_sample_ids only",
            "effective_train_sample_ids": list(effective_ids),
            "validation_sample_ids": list(fold.validation_sample_ids),
            "purged_train_sample_ids": list(fold.purge["purged_train_sample_ids"]),
            "feature_names": list(names),
            "stats": [item.to_dict() for item in stats],
            "roundtrip_max_abs_error": roundtrip_error,
            "target_transform": "identity",
            "denoise": "ml_framework.preprocess.denoise_identity",
        },
        constraint_audit={
            "fit_constraint_count": int(constraints.shape[0]),
            "constraints_supplied_to_model": (
                int(constraints.shape[0]) if protocol(mode).idw_feature_name is not None else 0
            ),
            "validation_constraints_used": 0,
            "purged_constraints_used": 0,
            "zero_constraint_fallback": (
                "fold_train_target_mean"
                if constraints.shape[0] == 0 and protocol(mode).idw_feature_name is not None
                else None
            ),
            "fallback_value": (
                fallback
                if constraints.shape[0] == 0 and protocol(mode).idw_feature_name is not None
                else None
            ),
            "strict_reference_derived_well_values_excluded": mode == "strict",
        },
    )


def _model_kwargs(
    model_name: str,
    n_features: int,
    n_training_samples: int,
    learning_rate: float,
    ridge_alpha: float,
    model_seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n_features": n_features,
        "n_training_samples": n_training_samples,
        "learning_rate": learning_rate,
        "ridge_alpha": ridge_alpha,
    }
    if model_name == "reconstruction_tiny_mlp":
        kwargs["seed"] = model_seed
    return kwargs


def _model_state(model: object) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, value in vars(model).items():
        if isinstance(value, np.ndarray):
            state[name] = value.copy()
        elif isinstance(value, (str, int, float, bool, type(None))):
            state[name] = value
    if not state:
        raise ValueError("model exposes no serializable state")
    return state


def _restore_model_state(model: object, state: Mapping[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, value.copy() if isinstance(value, np.ndarray) else value)


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "implementation": platform.python_implementation(),
    }


def _training_config_payload(
    *,
    mode: str,
    model_name: str,
    model_kwargs: Mapping[str, Any],
    epochs: int,
) -> dict[str, Any]:
    return {
        "task_spec": task_spec(mode).to_dict(),
        "model": model_name,
        "model_kwargs": dict(model_kwargs),
        "epochs": epochs,
        "loss": "mse_valid_label_mean",
        "output_transform": "identity",
    }


def train_model(
    *,
    mode: str,
    model_name: str,
    prepared: PreparedFold,
    output_dir: Path,
    split_hash: str,
    epochs: int,
    learning_rate: float,
    ridge_alpha: float,
    root_seed: int,
) -> tuple[object, TrainerState, Path]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    seed_report = seed_everything(root_seed, include_torch=False).to_dict()
    model_seed = SeedTree(root_seed).seed("model", mode, model_name)
    kwargs = _model_kwargs(
        model_name,
        prepared.train_features.shape[1],
        prepared.train_target.size,
        learning_rate,
        ridge_alpha,
        model_seed,
    )
    model = discover_model("reconstruction", model_name).build(task_spec(mode), **kwargs)
    config_payload = _training_config_payload(
        mode=mode,
        model_name=model_name,
        model_kwargs=kwargs,
        epochs=epochs,
    )
    config_hash = hash_payload(config_payload)
    output_dir.mkdir(parents=True, exist_ok=True)

    def train_step(batch: tuple[np.ndarray, np.ndarray]) -> StepResult:
        loss = float(model.train_batch(batch))
        return StepResult(loss_sum=loss * batch[1].size, valid_count=int(batch[1].size))

    def validation_step(batch: tuple[np.ndarray, np.ndarray]) -> StepResult:
        loss = float(model.validation_loss(batch))
        return StepResult(loss_sum=loss * batch[1].size, valid_count=int(batch[1].size))

    def checkpoint_writer(state: TrainerState, path: Path) -> None:
        save_checkpoint(
            path,
            epoch=max(0, state.next_epoch - 1),
            model_state=_model_state(model),
            optimizer_state={
                "name": "numpy_gradient_descent",
                "learning_rate": learning_rate,
                "update_count": int(getattr(model, "update_count", state.global_step)),
            },
            scheduler_state={"name": "constant", "last_epoch": state.next_epoch},
            scaler_state={"enabled": False, "kind": "numpy_float64"},
            config_hash=config_hash,
            split_hash=split_hash,
            trainer_state=state.to_dict(),
            seed_report=seed_report,
            environment=_environment(),
            extra={"mode": mode, "model": model_name, "config": config_payload},
            include_torch_rng=False,
        )

    state = train_with_validation(
        train_step=train_step,
        validation_step=validation_step,
        train_batches_fn=lambda: [(prepared.train_features, prepared.train_target)],
        validation_batches_fn=lambda: [(prepared.validation_features, prepared.validation_target)],
        config=TrainerConfig(max_epochs=epochs, min_epochs=epochs, patience=None),
        output_dir=output_dir,
        checkpoint_writer=checkpoint_writer,
    )
    best_path = output_dir / "checkpoint_best.pkl"
    best = load_checkpoint(best_path)
    _restore_model_state(model, best["model_state"])
    atomic_write_json(output_dir / "run_config.json", config_payload)
    return model, state, best_path


def _dense_volume(values: np.ndarray, indices_kji: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    dense = np.full(shape, np.nan, dtype=np.float64)
    dense[tuple(indices_kji.T)] = values
    return dense


def spectral_log_rmse(
    target: np.ndarray,
    prediction: np.ndarray,
    indices_kji: np.ndarray,
    shape: tuple[int, int, int],
) -> float:
    truth = _dense_volume(target, indices_kji, shape)
    pred = _dense_volume(prediction, indices_kji, shape)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not np.any(valid):
        raise ValueError("spectral metric has no valid voxels")
    truth_fill = np.where(valid, truth, float(np.mean(truth[valid])))
    pred_fill = np.where(valid, pred, float(np.mean(pred[valid])))
    truth_spectrum = np.log1p(np.abs(np.fft.rfftn(truth_fill)))
    pred_spectrum = np.log1p(np.abs(np.fft.rfftn(pred_fill)))
    return float(np.sqrt(np.mean((truth_spectrum - pred_spectrum) ** 2)))


def regression_metrics(
    mode: str,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    indices_kji: np.ndarray,
    volume_shape_kji: tuple[int, int, int],
    train_range: tuple[float, float],
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("metric target/prediction must be matching non-empty vectors")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metric inputs must be finite")
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    pearson_defined = bool(np.std(target) > 0 and np.std(prediction) > 0)
    raw = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else None,
        "pearson_r": float(np.corrcoef(target, prediction)[0, 1]) if pearson_defined else None,
        "spectral_log_rmse": spectral_log_rmse(
            target, prediction, indices_kji, volume_shape_kji
        ),
        "out_of_range_rate": float(
            np.mean((prediction < train_range[0]) | (prediction > train_range[1]))
        ),
        "voxel_count": int(target.size),
        "pearson_r_defined": pearson_defined,
    }
    for name in ("rmse", "mae", "bias", "spectral_log_rmse", "out_of_range_rate"):
        if not math.isfinite(raw[name]):
            raise FloatingPointError(f"non-finite metric {name}")
    for name in ("r2", "pearson_r"):
        if raw[name] is not None and not math.isfinite(raw[name]):
            raise FloatingPointError(f"non-finite metric {name}")
    names = metric_names(mode)
    return {
        "evaluation_mode": mode,
        "task_id": protocol(mode).task_id,
        **{names.get(key, key): value for key, value in raw.items()},
    }


def save_prediction_archive(
    path: Path,
    *,
    mode: str,
    cells: FlatCells,
    prediction: np.ndarray,
    amplitude: np.ndarray | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != cells.target.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction archive contains invalid prediction")
    values = cells.seismic[:, 0] if amplitude is None else np.asarray(amplitude, dtype=np.float64)
    np.savez_compressed(
        path,
        mode=np.asarray(mode),
        task_id=np.asarray(protocol(mode).task_id),
        sample_ids=cells.sample_ids,
        indices_kji=cells.indices_kji,
        volume_shape_kji=np.asarray(cells.volume_shape_kji, dtype=np.int64),
        truth=cells.target,
        prediction=prediction,
        residual=prediction - cells.target,
        amplitude=values,
    )
    return path


def _lifecycle_from_dict(payload: Mapping[str, Any]) -> ExperimentLifecycle:
    return ExperimentLifecycle(
        experiment_id=str(payload["experiment_id"]),
        state=ExperimentState(payload["state"]),
        evidence={key: dict(value) for key, value in payload.get("evidence", {}).items()},
        test_consumed_at=payload.get("test_consumed_at"),
    )


def _write_lifecycle(run_root: Path, life: ExperimentLifecycle) -> Path:
    return atomic_write_json(run_root / "lifecycle.json", life.to_dict())


def _read_lifecycle(run_root: Path) -> ExperimentLifecycle:
    path = run_root / "lifecycle.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _lifecycle_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _assert_run_mode(run_root: Path, mode: str) -> None:
    spec_path = run_root / "task_spec.json"
    if not spec_path.is_file():
        return
    archived = json.loads(spec_path.read_text(encoding="utf-8"))
    expected = protocol(mode)
    if archived.get("task_id") != expected.task_id or archived.get("metadata", {}).get("evaluation_mode") != mode:
        raise RuntimeError("run root belongs to a different reconstruction mode/task")


def initialize_run(
    *,
    mode: str,
    run_root: Path,
    catalog: Sequence[PatchLocation],
    root_seed: int = DEFAULT_ROOT_SEED,
    requested_n_splits: int = 5,
    buffer_blocks: int = 1,
) -> tuple[SplitManifest, ExperimentLifecycle]:
    _assert_run_mode(run_root, mode)
    create_run_layout(run_root)
    spec = task_spec(mode)
    manifest = build_spatial_manifest(
        mode,
        catalog,
        requested_n_splits=requested_n_splits,
        buffer_blocks=buffer_blocks,
    )
    atomic_write_json(run_root / "task_spec.json", spec.to_dict())
    atomic_write_json(run_root / "split_manifest.json", manifest.to_dict())
    seed_report = seed_everything(root_seed, include_torch=False).to_dict()
    atomic_write_json(run_root / "seed_report.json", seed_report)
    atomic_write_json(run_root / "environment.json", _environment())
    atomic_write_json(run_root / "hpo" / "plan.json", asdict(hpo_plan()))
    atomic_write_json(
        run_root / "hpo" / "fixed_baselines.json",
        {"metric_direction": "minimize", "configs": fixed_baseline_configs(), "executed": False},
    )
    life = ExperimentLifecycle(f"{mode}:{run_root.name}")
    life.advance(ExperimentState.SPLIT_LOCKED, {"split_hash": manifest.stable_hash()})
    _write_lifecycle(run_root, life)
    return manifest, life


def refresh_artifact_manifest(run_root: Path) -> Path:
    manifest = ArtifactManifest(run_root.name, run_root)
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(run_root).as_posix()
        role = relative.split("/", 1)[0] if "/" in relative else path.stem
        manifest.register(relative, role=role)
    return manifest.write()


def run_development_baseline_cv(
    *,
    mode: str,
    model_name: str,
    run_root: Path,
    catalog: Sequence[PatchLocation],
    development_records: Sequence[PatchRecord],
    epochs: int = 20,
    learning_rate: float = 0.01,
    ridge_alpha: float = 10.0,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> dict[str, Any]:
    manifest_path = run_root / "split_manifest.json"
    _assert_run_mode(run_root, mode)
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        folds = tuple(Fold(**item) for item in payload.pop("folds"))
        manifest = SplitManifest(folds=folds, **payload)
        life = _read_lifecycle(run_root)
    else:
        manifest, life = initialize_run(
            mode=mode,
            run_root=run_root,
            catalog=catalog,
            root_seed=root_seed,
        )
    if life.state == ExperimentState.SPLIT_LOCKED:
        life.advance(ExperimentState.SMOKE_PASSED, {"unit_contract_tiny_smoke": "passed"})
        _write_lifecycle(run_root, life)
    if life.state != ExperimentState.SMOKE_PASSED:
        raise RuntimeError(f"CV requires SMOKE_PASSED, got {life.state.value}")
    oof: list[tuple[FlatCells, np.ndarray, np.ndarray]] = []

    def fold_runner(fold: Fold) -> Mapping[str, Any]:
        prepared = prepare_fold(mode, fold, development_records)
        fold_dir = run_root / "folds" / f"fold_{fold.fold_id}"
        atomic_write_json(fold_dir / "preprocess_stats.json", prepared.preprocess_report)
        atomic_write_json(fold_dir / "constraint_audit.json", prepared.constraint_audit)
        model, state, checkpoint = train_model(
            mode=mode,
            model_name=model_name,
            prepared=prepared,
            output_dir=fold_dir / "training",
            split_hash=manifest.stable_hash(),
            epochs=epochs,
            learning_rate=learning_rate,
            ridge_alpha=ridge_alpha,
            root_seed=SeedTree(root_seed).seed("cv", fold.fold_id),
        )
        prediction = np.asarray(model.predict_array(prepared.validation_features), dtype=np.float64)
        mask = prepared.validation_metric_mask
        metric_cells = FlatCells(
            sample_ids=prepared.validation_cells.sample_ids[mask],
            indices_kji=prepared.validation_cells.indices_kji[mask],
            seismic=prepared.validation_cells.seismic[mask],
            coordinates=prepared.validation_cells.coordinates[mask],
            observed_mask=prepared.validation_cells.observed_mask[mask],
            target=prepared.validation_cells.target[mask],
            volume_shape_kji=prepared.validation_cells.volume_shape_kji,
        )
        metrics = regression_metrics(
            mode,
            metric_cells.target,
            prediction[mask],
            indices_kji=metric_cells.indices_kji,
            volume_shape_kji=metric_cells.volume_shape_kji,
            train_range=(float(prepared.train_target.min()), float(prepared.train_target.max())),
        )
        save_prediction_archive(
            fold_dir / "predictions.npz",
            mode=mode,
            cells=metric_cells,
            prediction=prediction[mask],
        )
        oof.append((metric_cells, prediction[mask], prepared.train_target))
        return {
            "validation_sample_ids": fold.validation_sample_ids,
            "metrics": metrics,
            "valid_label_count": int(mask.sum()),
            "best_epoch": state.best_epoch,
            "checkpoint": checkpoint.relative_to(run_root).as_posix(),
            "constraint_audit": prepared.constraint_audit,
        }

    primary = task_spec(mode).primary_metrics[0]
    summary = run_development_cv(
        manifest,
        fold_runner,
        output_dir=run_root,
        primary_metric=primary,
        metric_direction="minimize",
    )
    all_cells = FlatCells(
        sample_ids=np.concatenate([item[0].sample_ids for item in oof]),
        indices_kji=np.concatenate([item[0].indices_kji for item in oof]),
        seismic=np.concatenate([item[0].seismic for item in oof]),
        coordinates=np.concatenate([item[0].coordinates for item in oof]),
        observed_mask=np.concatenate([item[0].observed_mask for item in oof]),
        target=np.concatenate([item[0].target for item in oof]),
        volume_shape_kji=tuple(
            max(item[0].volume_shape_kji[axis] for item in oof) for axis in range(3)
        ),
    )
    all_prediction = np.concatenate([item[1] for item in oof])
    train_min = min(float(item[2].min()) for item in oof)
    train_max = max(float(item[2].max()) for item in oof)
    pooled = regression_metrics(
        mode,
        all_cells.target,
        all_prediction,
        indices_kji=all_cells.indices_kji,
        volume_shape_kji=all_cells.volume_shape_kji,
        train_range=(train_min, train_max),
    )
    save_prediction_archive(run_root / "oof" / "predictions.npz", mode=mode, cells=all_cells, prediction=all_prediction)
    atomic_write_json(run_root / "oof" / "metrics.json", pooled)
    summary["pooled_oof_metrics"] = pooled
    atomic_write_json(run_root / "oof" / "summary.json", summary)
    life.advance(ExperimentState.CV_COMPLETE, {"oof_hash": hash_file(run_root / "oof" / "predictions.npz")})
    _write_lifecycle(run_root, life)
    refresh_artifact_manifest(run_root)
    return summary


def run_fixed_baseline_plan(
    *,
    output_dir: Path,
    objective: Any,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> list[Any]:
    """Run fixed configurations only; callers own development-only objective data."""
    return run_fixed_trials(
        fixed_baseline_configs(),
        objective,
        root_seed=root_seed,
        output_dir=output_dir,
        metric_direction="minimize",
    )


def prepare_full_development(
    mode: str,
    development_records: Sequence[PatchRecord],
) -> PreparedFold:
    """Fit refit preprocessing on all development cells after config freeze."""
    cells = flatten_records(development_records)
    constraints = constraints_from_records(development_records)
    fallback = float(np.mean(cells.target))
    raw, names = _raw_features(mode, cells, constraints, fallback=fallback)
    stats = [fit_zscore(raw[:, column]) for column in range(raw.shape[1])]
    features = np.column_stack(
        [normalize(raw[:, column], stats[column]) for column in range(raw.shape[1])]
    ).astype(np.float64)
    restored = np.column_stack(
        [denormalize(features[:, column], stats[column]) for column in range(features.shape[1])]
    )
    roundtrip_error = float(np.max(np.abs(restored - raw)))
    if roundtrip_error > 1e-10:
        raise ValueError(f"refit preprocessing round-trip failed: {roundtrip_error}")
    metric_mask = ~cells.observed_mask
    return PreparedFold(
        feature_names=names,
        train_features=features,
        train_target=cells.target,
        # Refit epoch count is frozen from CV.  The development batch is reused
        # only for loss monitoring; it does not select a new epoch or config.
        validation_features=features,
        validation_target=cells.target,
        validation_metric_mask=metric_mask,
        validation_cells=cells,
        preprocess_report={
            "fit_scope": "all development patches after CONFIG_FROZEN",
            "effective_train_sample_ids": sorted({str(value) for value in cells.sample_ids}),
            "validation_sample_ids": [],
            "purged_train_sample_ids": [],
            "feature_names": list(names),
            "stats": [item.to_dict() for item in stats],
            "roundtrip_max_abs_error": roundtrip_error,
            "target_transform": "identity",
            "denoise": "ml_framework.preprocess.denoise_identity",
            "refit_loss_monitor_note": "development loss only; epoch count was frozen before refit",
        },
        constraint_audit={
            "fit_constraint_count": int(constraints.shape[0]),
            "constraints_supplied_to_model": (
                int(constraints.shape[0]) if protocol(mode).idw_feature_name is not None else 0
            ),
            "test_constraints_used": 0,
            "guard_constraints_used": 0,
            "zero_constraint_fallback": (
                "development_target_mean"
                if constraints.shape[0] == 0 and protocol(mode).idw_feature_name is not None
                else None
            ),
            "strict_reference_derived_well_values_excluded": mode == "strict",
        },
    )


def freeze_and_refit(
    *,
    mode: str,
    model_name: str,
    run_root: Path,
    development_records: Sequence[PatchRecord],
    epochs: int,
    learning_rate: float,
    ridge_alpha: float,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> dict[str, Any]:
    """Freeze a development-selected config and refit on all development data."""
    _assert_run_mode(run_root, mode)
    life = _read_lifecycle(run_root)
    if life.state != ExperimentState.CV_COMPLETE:
        raise RuntimeError(f"refit requires CV_COMPLETE, got {life.state.value}")
    manifest_payload = json.loads((run_root / "split_manifest.json").read_text(encoding="utf-8"))
    folds = tuple(Fold(**item) for item in manifest_payload.pop("folds"))
    manifest = SplitManifest(folds=folds, **manifest_payload)
    prepared = prepare_full_development(mode, development_records)
    model_seed = SeedTree(root_seed).seed("model", mode, model_name)
    kwargs = _model_kwargs(
        model_name,
        prepared.train_features.shape[1],
        prepared.train_target.size,
        learning_rate,
        ridge_alpha,
        model_seed,
    )
    frozen_config = _training_config_payload(
        mode=mode,
        model_name=model_name,
        model_kwargs=kwargs,
        epochs=epochs,
    )
    frozen_config["selection_source"] = "development OOF only"
    frozen_config["epoch_rule"] = "explicit frozen refit epoch count"
    # train_model hashes only executable training fields; keep the frozen file
    # byte-for-byte aligned with that payload for lifecycle verification.
    executable_config = _training_config_payload(
        mode=mode,
        model_name=model_name,
        model_kwargs=kwargs,
        epochs=epochs,
    )
    config_hash = hash_payload(executable_config)
    atomic_write_json(run_root / "frozen_config.json", {**frozen_config, "config_hash": config_hash})
    life.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": config_hash})
    _write_lifecycle(run_root, life)
    atomic_write_json(run_root / "refit" / "preprocess_stats.json", prepared.preprocess_report)
    atomic_write_json(run_root / "refit" / "constraint_audit.json", prepared.constraint_audit)
    _, state, _ = train_model(
        mode=mode,
        model_name=model_name,
        prepared=prepared,
        output_dir=run_root / "refit" / "training",
        split_hash=manifest.stable_hash(),
        epochs=epochs,
        learning_rate=learning_rate,
        ridge_alpha=ridge_alpha,
        root_seed=root_seed,
    )
    checkpoint = run_root / "refit" / "training" / "checkpoint_last.pkl"
    loaded = load_checkpoint(checkpoint)
    if loaded["config_hash"] != config_hash:
        raise RuntimeError("refit checkpoint config hash differs from frozen config")
    checkpoint_hash = hash_file(checkpoint)
    life.advance(
        ExperimentState.REFIT_COMPLETE,
        {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint": checkpoint.relative_to(run_root).as_posix(),
            "epochs": state.next_epoch,
        },
    )
    _write_lifecycle(run_root, life)
    refresh_artifact_manifest(run_root)
    return {
        "mode": mode,
        "model": model_name,
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "checkpoint": checkpoint.relative_to(run_root).as_posix(),
        "epochs": state.next_epoch,
    }


def _normalization_from_report(raw: np.ndarray, report: Mapping[str, Any]) -> np.ndarray:
    stats = [NormStats.from_dict(dict(item)) for item in report["stats"]]
    if raw.shape[1] != len(stats):
        raise ValueError("refit preprocessing feature count does not match test features")
    return np.column_stack(
        [normalize(raw[:, column], stats[column]) for column in range(raw.shape[1])]
    ).astype(np.float64)


def run_frozen_test_once(
    *,
    mode: str,
    run_root: Path,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Consume and evaluate frozen test exactly once, in that order."""
    _assert_run_mode(run_root, mode)
    active = protocol(mode)
    life = _read_lifecycle(run_root)
    if life.state != ExperimentState.REFIT_COMPLETE:
        raise RuntimeError(f"test requires REFIT_COMPLETE, got {life.state.value}")
    frozen = json.loads((run_root / "frozen_config.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((run_root / "split_manifest.json").read_text(encoding="utf-8"))
    folds = tuple(Fold(**item) for item in manifest_payload.pop("folds"))
    manifest = SplitManifest(folds=folds, **manifest_payload)
    checkpoint_relative = life.evidence[ExperimentState.REFIT_COMPLETE.value]["checkpoint"]
    checkpoint = run_root / checkpoint_relative
    checkpoint_hash = hash_file(checkpoint)
    config_hash = str(frozen["config_hash"])
    split_hash = manifest.stable_hash()
    # The consumed state is durably written before any test block HDF5 array is
    # loaded.  A crash after this line still burns the campaign as required.
    consume_frozen_test_once(
        run_root=run_root,
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        split_hash=split_hash,
    )

    development_records = load_patch_records(active.development_i_blocks, data_dir)
    test_records = load_patch_records(active.test_i_blocks, data_dir)
    test_cells = flatten_records(test_records)
    development_constraints = constraints_from_records(development_records)
    test_constraints = constraints_from_records(test_records)
    if mode == "strict" and test_constraints.shape[0] != 0:
        raise RuntimeError("strict test contains well constraints; target/input firewall refuses evaluation")
    allowed_constraints = (
        np.concatenate([development_constraints, test_constraints])
        if mode == "conditional" and test_constraints.shape[0]
        else development_constraints
    )
    development_cells = flatten_records(development_records)
    fallback = float(np.mean(development_cells.target))
    raw, feature_names = _raw_features(mode, test_cells, allowed_constraints, fallback=fallback)
    assert_feature_contract(mode, feature_names)
    preprocess = json.loads((run_root / "refit" / "preprocess_stats.json").read_text(encoding="utf-8"))
    features = _normalization_from_report(raw, preprocess)
    checkpoint_payload = load_checkpoint(checkpoint)
    config = checkpoint_payload["extra"]["config"]
    archived_task_spec = TaskSpec.from_dict(config["task_spec"])
    model = discover_model("reconstruction", config["model"]).build(
        archived_task_spec, **config["model_kwargs"]
    )
    _restore_model_state(model, checkpoint_payload["model_state"])
    prediction = np.asarray(model.predict_array(features), dtype=np.float64)
    metric_mask = ~test_cells.observed_mask
    metric_cells = FlatCells(
        sample_ids=test_cells.sample_ids[metric_mask],
        indices_kji=test_cells.indices_kji[metric_mask],
        seismic=test_cells.seismic[metric_mask],
        coordinates=test_cells.coordinates[metric_mask],
        observed_mask=test_cells.observed_mask[metric_mask],
        target=test_cells.target[metric_mask],
        volume_shape_kji=test_cells.volume_shape_kji,
    )
    metrics = regression_metrics(
        mode,
        metric_cells.target,
        prediction[metric_mask],
        indices_kji=metric_cells.indices_kji,
        volume_shape_kji=metric_cells.volume_shape_kji,
        train_range=(float(development_cells.target.min()), float(development_cells.target.max())),
    )
    constraint_audit = {
        "evaluation_mode": mode,
        "development_constraints": int(development_constraints.shape[0]),
        "development_constraints_used": (
            int(development_constraints.shape[0]) if active.idw_feature_name is not None else 0
        ),
        "test_constraints_present": int(test_constraints.shape[0]),
        "test_constraints_used": int(test_constraints.shape[0]) if mode == "conditional" else 0,
        "guard_constraints_used": 0,
        "strict_test_target_or_future_feature_used": False,
        "strict_reference_derived_well_value_used": False,
        "exact_constraint_cells_excluded_from_metrics": int(test_cells.observed_mask.sum()),
    }
    save_prediction_archive(
        run_root / "frozen_test" / "predictions.npz",
        mode=mode,
        cells=metric_cells,
        prediction=prediction[metric_mask],
    )
    atomic_write_json(run_root / "frozen_test" / "metrics.json", metrics)
    atomic_write_json(run_root / "frozen_test" / "constraint_audit.json", constraint_audit)
    refresh_artifact_manifest(run_root)
    return {"metrics": metrics, "constraint_audit": constraint_audit}


def consume_frozen_test_once(
    *,
    run_root: Path,
    config_hash: str,
    checkpoint_hash: str,
    split_hash: str,
) -> ExperimentLifecycle:
    """Persist TEST_CONSUMED before any caller is allowed to read test labels."""
    life = _read_lifecycle(run_root)
    life.consume_test(
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        split_hash=split_hash,
    )
    _write_lifecycle(run_root, life)
    return life


def synthetic_catalog_and_records() -> tuple[list[PatchLocation], list[PatchRecord]]:
    """Small deterministic six-I/seven-K volume for unit/tiny smoke only."""
    shape = (2, 3, 4)
    locations: list[PatchLocation] = []
    payloads: list[tuple[PatchLocation, np.ndarray, np.ndarray]] = []
    constraint_rows: list[list[float]] = []
    for i_block in range(6):
        for k_block in range(7):
            location = PatchLocation(
                sample_id=f"k{k_block:02d}_j00_i{i_block:02d}",
                source_split="test" if i_block >= 4 else "train",
                source_key=f"synthetic_{k_block}_{i_block}",
                k_block=k_block,
                j_block=0,
                i_block=i_block,
                patch_start_kji=(k_block * shape[0], 0, i_block * shape[2]),
                patch_shape_kji=shape,
            )
            grid = np.indices(shape)
            global_k = grid[0] + k_block * shape[0]
            global_j = grid[1]
            global_i = grid[2] + i_block * shape[2]
            x = global_i / float(6 * shape[2] - 1)
            y = global_j / float(shape[1] - 1)
            z = global_k / float(7 * shape[0] - 1)
            amplitude = np.sin(global_i / 3.0) + np.cos(global_k / 2.0)
            target = 0.08 + 0.05 * x + 0.02 * y + 0.03 * z + 0.005 * amplitude
            patch = np.zeros((9, *shape), dtype=np.float32)
            patch[0] = amplitude
            patch[1] = np.sqrt(amplitude**2 + 0.01)
            patch[2] = np.gradient(amplitude, axis=0)
            patch[3], patch[4], patch[5] = x, y, z
            patch[8] = 1.0
            has_constraint = (i_block == 3 and k_block == 6) or (i_block == 4)
            if has_constraint:
                local = (0, 1, 1)
                patch[(7, *local)] = 1.0
                patch[(6, *local)] = target[local]
                constraint_rows.append(
                    [float(x[local]), float(y[local]), float(z[local]), float(target[local])]
                )
            locations.append(location)
            payloads.append((location, patch, target.astype(np.float32)))
    global_wells = np.asarray(constraint_rows, dtype=np.float32)
    records = [PatchRecord(location, patch, target, global_wells.copy()) for location, patch, target in payloads]
    return locations, records


def feasibility_report(catalog: Sequence[PatchLocation], records: Sequence[PatchRecord] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "feasible": [
            "conditional and strict mode-specific TaskSpec",
            "continuous spatial-block test manifests",
            "buffered development OOF CV",
            "fixed Ridge/linear-SGD/tiny-MLP baselines",
        ],
        "not_feasible": [
            {
                "claim": "new blind cross-field generalization",
                "reason": "only one Volve reference volume is available and the historical spatial test was already reported",
            },
            {
                "claim": "cross-well generalization",
                "reason": "only one physical well intersects active final-grid cells",
            },
            {
                "claim": "fully constrained five-fold conditional IDW CV",
                "reason": (
                    "conditional development contains one sparse constraint; a validation/buffer "
                    "fold can have zero fold-train constraints"
                ),
                "mitigation": (
                    "use the audited fold-train target-mean IDW fallback without importing "
                    "frozen-test constraints"
                ),
            },
            {
                "claim": "strict P4 IDW from independent measured well porosity",
                "reason": (
                    "available sparse porosity values were sampled from Eclipse reference target "
                    "cells, so strict P4 excludes them as derived truth"
                ),
                "needed": "independently measured and spatially registered porosity well logs",
            },
        ],
        "warnings": [],
    }
    if records is not None:
        conditional_dev = [item for item in records if item.i_block in protocol("conditional").development_i_blocks]
        count = int(constraints_from_records(conditional_dev).shape[0])
        report["conditional_development_constraint_count"] = count
        if count <= 1:
            report["warnings"].append(
                "conditional development has one sparse constraint; folds that purge it use the explicit fold-train-mean IDW fallback"
            )
    report["catalog_sample_count"] = len(catalog)
    return report


def real_data_smoke(data_dir: Path | None = None) -> dict[str, Any]:
    """Read-only real-data contract smoke; no model training or test prediction."""
    root = resolve_data_dir(data_dir)
    catalog = scan_patch_catalog(root)
    summaries: dict[str, Any] = {}
    for mode in MODES:
        active = protocol(mode)
        manifest = build_spatial_manifest(mode, catalog)
        development = load_patch_records(active.development_i_blocks, root)
        first = prepare_fold(mode, manifest.folds[0], development)
        summaries[mode] = {
            "task_id": active.task_id,
            "requested_n_splits": manifest.requested_n_splits,
            "effective_n_splits": manifest.effective_n_splits,
            "development_patches": len(manifest.development_sample_ids),
            "test_patches": len(manifest.test_sample_ids),
            "first_fold_train_cells": int(first.train_target.size),
            "first_fold_validation_cells": int(first.validation_target.size),
            "first_fold_constraints": first.constraint_audit["fit_constraint_count"],
            "first_fold_constraints_supplied_to_model": first.constraint_audit[
                "constraints_supplied_to_model"
            ],
            "finite_features": bool(
                np.all(np.isfinite(first.train_features))
                and np.all(np.isfinite(first.validation_features))
            ),
        }
    return {"data_dir": str(root), "catalog_patches": len(catalog), "modes": summaries}


def tiny_smoke(output_dir: Path, mode: str = "strict", model_name: str = "ridge_linear") -> dict[str, Any]:
    catalog, records = synthetic_catalog_and_records()
    active = protocol(mode)
    manifest = build_spatial_manifest(mode, catalog)
    development = [item for item in records if item.i_block in active.development_i_blocks]
    prepared = prepare_fold(mode, manifest.folds[0], development)
    model, state, checkpoint = train_model(
        mode=mode,
        model_name=model_name,
        prepared=prepared,
        output_dir=output_dir,
        split_hash=manifest.stable_hash(),
        epochs=3,
        learning_rate=0.01,
        ridge_alpha=0.1,
        root_seed=DEFAULT_ROOT_SEED,
    )
    prediction = model.predict_array(prepared.validation_features)
    return {
        "mode": mode,
        "model": model_name,
        "epochs": state.next_epoch,
        "best_epoch": state.best_epoch,
        "checkpoint": str(checkpoint),
        "finite_prediction": bool(np.all(np.isfinite(prediction))),
        "prediction_shape": list(prediction.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    specs = subparsers.add_parser("task-specs", help="print both independent TaskSpecs")
    specs.add_argument("--mode", choices=(*MODES, "both"), default="both")

    plan = subparsers.add_parser("plan", help="print fixed baselines and minimize HPO plan")
    plan.add_argument("--mode", choices=MODES, default="strict")

    real = subparsers.add_parser("real-smoke", help="read-only real HDF5 split/CV smoke")
    real.add_argument("--data-dir", type=Path)

    tiny = subparsers.add_parser("tiny-smoke", help="three-epoch synthetic development smoke")
    tiny.add_argument("--mode", choices=MODES, default="strict")
    tiny.add_argument("--model", choices=MODEL_NAMES, default="ridge_linear")
    tiny.add_argument("--output-dir", type=Path, required=True)

    prepare = subparsers.add_parser("prepare", help="freeze TaskSpec/test/CV manifest without training")
    prepare.add_argument("--mode", choices=MODES, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path)
    prepare.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)

    cv_command = subparsers.add_parser("cv", help="run development-only buffered CV")
    cv_command.add_argument("--mode", choices=MODES, required=True)
    cv_command.add_argument("--run-root", type=Path, required=True)
    cv_command.add_argument("--data-dir", type=Path)
    cv_command.add_argument("--model", choices=MODEL_NAMES, default="ridge_linear")
    cv_command.add_argument("--epochs", type=int, default=20)
    cv_command.add_argument("--learning-rate", type=float, default=0.01)
    cv_command.add_argument("--ridge-alpha", type=float, default=10.0)
    cv_command.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)

    refit = subparsers.add_parser("refit", help="freeze config and refit all development data")
    refit.add_argument("--mode", choices=MODES, required=True)
    refit.add_argument("--run-root", type=Path, required=True)
    refit.add_argument("--data-dir", type=Path)
    refit.add_argument("--model", choices=MODEL_NAMES, default="ridge_linear")
    refit.add_argument("--epochs", type=int, required=True)
    refit.add_argument("--learning-rate", type=float, default=0.01)
    refit.add_argument("--ridge-alpha", type=float, default=10.0)
    refit.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)

    test = subparsers.add_parser("test", help="single-use frozen-test inference")
    test.add_argument("--mode", choices=MODES, required=True)
    test.add_argument("--run-root", type=Path, required=True)
    test.add_argument("--data-dir", type=Path)

    args = parser.parse_args()
    if args.command == "task-specs":
        modes = MODES if args.mode == "both" else (args.mode,)
        print(json.dumps({mode: task_spec(mode).to_dict() for mode in modes}, indent=2))
    elif args.command == "plan":
        print(
            json.dumps(
                {
                    "task_id": protocol(args.mode).task_id,
                    "primary_metric": task_spec(args.mode).primary_metrics[0],
                    "hpo": asdict(hpo_plan()),
                    "fixed_baselines": fixed_baseline_configs(),
                    "long_hpo_executed": False,
                },
                indent=2,
            )
        )
    elif args.command == "real-smoke":
        print(json.dumps(real_data_smoke(args.data_dir), indent=2))
    elif args.command == "tiny-smoke":
        print(json.dumps(tiny_smoke(args.output_dir, args.mode, args.model), indent=2))
    elif args.command == "prepare":
        catalog = scan_patch_catalog(args.data_dir)
        manifest, life = initialize_run(
            mode=args.mode,
            run_root=args.run_root,
            catalog=catalog,
            root_seed=args.root_seed,
        )
        active = protocol(args.mode)
        records = load_patch_records(active.development_i_blocks, args.data_dir)
        report = feasibility_report(catalog, records)
        atomic_write_json(args.run_root / "not_feasible.json", report)
        refresh_artifact_manifest(args.run_root)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "state": life.state.value,
                    "split_hash": manifest.stable_hash(),
                    "effective_n_splits": manifest.effective_n_splits,
                    "not_feasible": report["not_feasible"],
                },
                indent=2,
            )
        )
    elif args.command == "cv":
        active = protocol(args.mode)
        catalog = scan_patch_catalog(args.data_dir)
        development = load_patch_records(active.development_i_blocks, args.data_dir)
        summary = run_development_baseline_cv(
            mode=args.mode,
            model_name=args.model,
            run_root=args.run_root,
            catalog=catalog,
            development_records=development,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            ridge_alpha=args.ridge_alpha,
            root_seed=args.root_seed,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "refit":
        active = protocol(args.mode)
        development = load_patch_records(active.development_i_blocks, args.data_dir)
        print(
            json.dumps(
                freeze_and_refit(
                    mode=args.mode,
                    model_name=args.model,
                    run_root=args.run_root,
                    development_records=development,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    ridge_alpha=args.ridge_alpha,
                    root_seed=args.root_seed,
                ),
                indent=2,
            )
        )
    elif args.command == "test":
        print(
            json.dumps(
                run_frozen_test_once(mode=args.mode, run_root=args.run_root, data_dir=args.data_dir),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
