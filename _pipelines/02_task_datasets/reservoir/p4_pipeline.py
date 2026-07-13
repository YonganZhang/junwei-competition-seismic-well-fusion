"""Leakage-safe P4 runner shared by sweet-spot targets 6 and 7.

The runner consumes the real reservoir HDF5/guard artifacts produced by this
track, but it owns independent target manifests, fold-only preprocessing, OOF
predictions, a frozen configuration, one development refit and one frozen-test
consumption per label version.  The shared P4 framework remains read-only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
from ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from ml_framework.contracts import TaskSpec  # noqa: E402
from ml_framework.cv import run_development_cv  # noqa: E402
from ml_framework.lifecycle import ExperimentLifecycle, ExperimentState  # noqa: E402
from ml_framework.model_discovery import discover_model  # noqa: E402
from ml_framework.preprocess import denoise_identity  # noqa: E402
from ml_framework.run_layout import create_run_layout  # noqa: E402
from ml_framework.seeding import SeedTree, derive_seed, seed_everything  # noqa: E402
from ml_framework.splits import Fold, SplitManifest, build_group_folds  # noqa: E402
from ml_framework.trainer import StepResult, TrainerConfig, train_with_validation  # noqa: E402


DEFAULT_TEST_FAMILY = "15/9-F-15"
INPUT_CHANNELS = ("GR", "RT", "NPHI", "RHOB")
FORBIDDEN_INPUTS = (
    "PHIF",
    "PHIE",
    "LFP_PHIE",
    "KLOGH",
    "KLOGH_NEW",
    "KLOGV",
    "SW",
    "BVW",
    "SWIRR",
    "VSH",
    "LFP_VSH",
)


@dataclass(frozen=True)
class TargetDefinition:
    """Storage and physical semantics for one independent property task."""

    name: str
    stored_index: int
    stored_domain: str
    physical_unit: str
    physical_bounds: tuple[float | None, float | None]

    def to_physical(self, values: np.ndarray, *, prediction: bool) -> np.ndarray:
        domain = np.asarray(values, dtype=np.float64)
        if self.stored_domain == "identity":
            physical = domain.copy()
        elif self.stored_domain == "log1p":
            # The physical lower bound is enforced only for inference.  Truth
            # values are checked rather than silently changed.
            if prediction:
                domain = np.maximum(domain, 0.0)
            physical = np.expm1(domain)
        else:
            raise ValueError(f"unsupported stored domain: {self.stored_domain}")
        if prediction:
            lower, upper = self.physical_bounds
            if lower is not None:
                physical = np.maximum(physical, lower)
            if upper is not None:
                physical = np.minimum(physical, upper)
        return physical

    def from_physical(self, values: np.ndarray) -> np.ndarray:
        physical = np.asarray(values, dtype=np.float64)
        if self.stored_domain == "identity":
            return physical.copy()
        if self.stored_domain == "log1p":
            if np.any(physical < 0):
                raise ValueError(f"{self.name} contains negative physical values")
            return np.log1p(physical)
        raise ValueError(f"unsupported stored domain: {self.stored_domain}")


POROSITY_PHIF = TargetDefinition("PHIF", 0, "identity", "fraction", (0.0, 1.0))
PERMEABILITY_KLOGH = TargetDefinition("KLOGH", 1, "log1p", "mD", (0.0, None))


@dataclass(frozen=True)
class IndexedSample:
    sample_id: str
    family_id: str
    well_id: str
    depth_m: float
    position: Mapping[str, Any]
    source_kind: str
    source_path: Path
    source_key: str | int
    label_valid: bool


@dataclass(frozen=True)
class LoadedSample:
    index: IndexedSample
    seismic_patch: np.ndarray
    well_log_seq: np.ndarray
    target_domain: float


@dataclass(frozen=True)
class FeatureStats:
    seismic_mean: np.ndarray
    seismic_std: np.ndarray
    log_mean: np.ndarray
    log_std: np.ndarray
    target_mean: float
    target_std: float
    fit_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "per-feature zscore with explicit observed log masks",
            "seismic_mean": self.seismic_mean.tolist(),
            "seismic_std": self.seismic_std.tolist(),
            "log_mean": self.log_mean.tolist(),
            "log_std": self.log_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_scope": "current fold train well families only",
            "input_channels": list(INPUT_CHANNELS),
            "denoise": "denoise_identity",
        }


def _stable_sample_id(
    task_id: str,
    *,
    family_id: str,
    well_id: str,
    depth_m: float,
    source_kind: str,
    source_key: str | int,
) -> str:
    raw = "|".join(
        [task_id, family_id, well_id, f"{depth_m:.6f}", source_kind, str(source_key)]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{task_id}-{digest}"


def _metadata(meta_text: str, position_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = json.loads(meta_text)
    position = json.loads(position_text)
    for key in ("well_id", "family_id", "depth_m"):
        if key not in meta:
            raise ValueError(f"real reservoir sample metadata missing {key!r}")
    return meta, position


def scan_real_sources(
    *,
    processed_dir: Path,
    guard_path: Path,
    task_id: str,
    definition: TargetDefinition,
) -> list[IndexedSample]:
    """Read identifiers and label masks only; feature/label values stay closed."""
    processed_dir = Path(processed_dir)
    guard_path = Path(guard_path)
    required = [processed_dir / "train.h5", processed_dir / "test.h5", guard_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing real reservoir data: " + ", ".join(str(p) for p in missing))

    records: list[IndexedSample] = []
    for split in ("train", "test"):
        path = processed_dir / f"{split}.h5"
        with h5py.File(path, "r") as handle:
            for key in sorted(handle.keys()):
                group = handle[key]
                label = np.asarray(group["label"][()], dtype=np.float64).reshape(-1)
                if label.size <= definition.stored_index:
                    raise ValueError(f"{path}:{key} label shape cannot provide {definition.name}")
                meta, position = _metadata(group.attrs["meta"], group.attrs["position"])
                records.append(
                    IndexedSample(
                        sample_id=_stable_sample_id(
                            task_id,
                            family_id=meta["family_id"],
                            well_id=meta["well_id"],
                            depth_m=float(meta["depth_m"]),
                            source_kind="hdf5",
                            source_key=key,
                        ),
                        family_id=meta["family_id"],
                        well_id=meta["well_id"],
                        depth_m=float(meta["depth_m"]),
                        position=position,
                        source_kind="hdf5",
                        source_path=path,
                        source_key=key,
                        label_valid=bool(np.isfinite(label[definition.stored_index])),
                    )
                )

    with np.load(guard_path, allow_pickle=False) as archive:
        required_keys = {"seismic_patch", "well_log_seq", "label", "position_json", "meta_json"}
        if not required_keys <= set(archive.files):
            raise ValueError(f"guard archive missing keys: {sorted(required_keys - set(archive.files))}")
        labels = np.asarray(archive["label"], dtype=np.float64)
        for index in range(len(labels)):
            meta, position = _metadata(str(archive["meta_json"][index]), str(archive["position_json"][index]))
            records.append(
                IndexedSample(
                    sample_id=_stable_sample_id(
                        task_id,
                        family_id=meta["family_id"],
                        well_id=meta["well_id"],
                        depth_m=float(meta["depth_m"]),
                        source_kind="guard_npz",
                        source_key=index,
                    ),
                    family_id=meta["family_id"],
                    well_id=meta["well_id"],
                    depth_m=float(meta["depth_m"]),
                    position=position,
                    source_kind="guard_npz",
                    source_path=guard_path,
                    source_key=index,
                    label_valid=bool(np.isfinite(labels[index, definition.stored_index])),
                )
            )
    if len({record.sample_id for record in records}) != len(records):
        raise RuntimeError("stable sample IDs are not globally unique")
    return records


def label_availability(records: Sequence[IndexedSample]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for family in sorted({record.family_id for record in records}):
        subset = [record for record in records if record.family_id == family]
        valid = [record for record in subset if record.label_valid]
        families[family] = {
            "samples_scanned": len(subset),
            "valid_labels": len(valid),
            "wells_with_valid_labels": sorted({record.well_id for record in valid}),
        }
    return {
        "selection_inputs": "label availability only; no features, model, HPO or metrics",
        "families": families,
        "total_samples_scanned": len(records),
        "total_valid_labels": sum(record.label_valid for record in records),
    }


def select_frozen_test_family(
    records: Sequence[IndexedSample], preferred: str = DEFAULT_TEST_FAMILY
) -> tuple[str, dict[str, Any]]:
    """Freeze test using only preregistered label support."""
    valid_counts = {
        family: sum(record.label_valid for record in records if record.family_id == family)
        for family in sorted({record.family_id for record in records})
    }
    valid_families = [family for family, count in valid_counts.items() if count > 0]
    if preferred in valid_families and len([f for f in valid_families if f != preferred]) >= 2:
        return preferred, {
            "policy": "prefer F-15 when it has labels and leaves at least two development families",
            "selected_family": preferred,
            "fallback_used": False,
            "valid_label_counts": valid_counts,
        }
    candidates = [family for family in valid_families if len([f for f in valid_families if f != family]) >= 2]
    if not candidates:
        raise RuntimeError(
            "label availability cannot support one frozen test plus at least two development families"
        )
    selected = sorted(candidates, key=lambda family: (-valid_counts[family], family))[0]
    return selected, {
        "policy": "fallback: largest preregistered valid-label count, then lexical family ID",
        "selected_family": selected,
        "fallback_used": True,
        "fallback_reason": f"preferred family {preferred!r} lacked sufficient registered support",
        "valid_label_counts": valid_counts,
    }


def build_independent_manifest(
    records: Sequence[IndexedSample],
    *,
    test_family: str,
    target_name: str,
    label_version: str,
    seed: int,
    selection_record: Mapping[str, Any],
) -> SplitManifest:
    valid = [record for record in records if record.label_valid]
    return build_group_folds(
        [record.sample_id for record in valid],
        [record.family_id for record in valid],
        group_key="mother_well_family",
        test_groups=[test_family],
        requested_n_splits=5,
        seed=derive_seed(seed, "split", target_name, label_version),
        metadata={
            "target": target_name,
            "label_version": label_version,
            "selection_registered_before_modeling": True,
            "test_selection": dict(selection_record),
            "interpolation_windowing_augmentation_fit_after_split": True,
            "development_cv": "LOGO equivalent; requested5, effective=min(5, development families)",
        },
    )


def load_samples(records: Sequence[IndexedSample], definition: TargetDefinition) -> list[LoadedSample]:
    """Load numerical arrays for an already-authorized set of sample IDs."""
    by_id: dict[str, LoadedSample] = {}
    hdf_paths = sorted({record.source_path for record in records if record.source_kind == "hdf5"})
    for path in hdf_paths:
        selected = [record for record in records if record.source_path == path]
        with h5py.File(path, "r") as handle:
            for record in selected:
                group = handle[str(record.source_key)]
                label = np.asarray(group["label"][()], dtype=np.float64).reshape(-1)
                by_id[record.sample_id] = LoadedSample(
                    record,
                    np.asarray(group["seismic_patch"][()], dtype=np.float64),
                    np.asarray(group["well_log_seq"][()], dtype=np.float64),
                    float(label[definition.stored_index]),
                )
    npz_paths = sorted({record.source_path for record in records if record.source_kind == "guard_npz"})
    for path in npz_paths:
        selected = [record for record in records if record.source_path == path]
        with np.load(path, allow_pickle=False) as archive:
            for record in selected:
                index = int(record.source_key)
                by_id[record.sample_id] = LoadedSample(
                    record,
                    np.asarray(archive["seismic_patch"][index], dtype=np.float64),
                    np.asarray(archive["well_log_seq"][index], dtype=np.float64),
                    float(archive["label"][index, definition.stored_index]),
                )
    missing = [record.sample_id for record in records if record.sample_id not in by_id]
    if missing:
        raise RuntimeError(f"failed to load {len(missing)} authorized samples")
    loaded = [by_id[record.sample_id] for record in records]
    for sample in loaded:
        if sample.seismic_patch.shape != (3, 3, 9) or sample.well_log_seq.shape != (9, 8):
            raise ValueError("unexpected real reservoir input shape")
        if not np.isfinite(sample.seismic_patch).all() or not np.isfinite(sample.well_log_seq).all():
            raise ValueError("real input contains non-finite values")
        if not math.isfinite(sample.target_domain):
            raise ValueError("authorized target is non-finite")
    return loaded


def fit_fold_statistics(samples: Sequence[LoadedSample]) -> FeatureStats:
    if not samples:
        raise ValueError("cannot fit preprocessing from zero fold-train samples")
    seismic = denoise_identity(np.stack([sample.seismic_patch for sample in samples])).reshape(len(samples), -1)
    logs = np.stack([sample.well_log_seq for sample in samples])
    values = denoise_identity(logs[:, :, :4]).reshape(len(samples), -1)
    masks = (logs[:, :, 4:8] > 0.5).reshape(len(samples), -1)
    seismic_mean = seismic.mean(axis=0)
    seismic_std = seismic.std(axis=0) + 1e-8
    log_mean = np.zeros(values.shape[1], dtype=np.float64)
    log_std = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        observed = values[masks[:, column], column]
        if observed.size:
            log_mean[column] = observed.mean()
            log_std[column] = observed.std() + 1e-8
    targets = np.asarray([sample.target_domain for sample in samples], dtype=np.float64)
    stats = FeatureStats(
        seismic_mean=seismic_mean,
        seismic_std=seismic_std,
        log_mean=log_mean,
        log_std=log_std,
        target_mean=float(targets.mean()),
        target_std=float(targets.std() + 1e-8),
        fit_sample_ids=tuple(sample.index.sample_id for sample in samples),
    )
    if not all(
        np.isfinite(array).all()
        for array in (stats.seismic_mean, stats.seismic_std, stats.log_mean, stats.log_std)
    ):
        raise RuntimeError("fold-train preprocessing produced non-finite statistics")
    return stats


def transform_inputs(samples: Sequence[LoadedSample], stats: FeatureStats) -> np.ndarray:
    seismic = denoise_identity(np.stack([sample.seismic_patch for sample in samples])).reshape(len(samples), -1)
    logs = np.stack([sample.well_log_seq for sample in samples])
    values = denoise_identity(logs[:, :, :4]).reshape(len(samples), -1)
    masks = (logs[:, :, 4:8] > 0.5).astype(np.float64).reshape(len(samples), -1)
    seismic_norm = (seismic - stats.seismic_mean) / stats.seismic_std
    log_norm = ((values - stats.log_mean) / stats.log_std) * masks
    features = np.concatenate([seismic_norm, log_norm, masks], axis=1)
    if features.shape[1] != 153 or not np.isfinite(features).all():
        raise RuntimeError(f"invalid multimodal feature matrix: {features.shape}")
    return features


def transform_target(samples: Sequence[LoadedSample], stats: FeatureStats) -> np.ndarray:
    values = np.asarray([sample.target_domain for sample in samples], dtype=np.float64)
    return ((values - stats.target_mean) / stats.target_std).reshape(-1, 1)


def inverse_normalized_target(values: np.ndarray, stats: FeatureStats) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1) * stats.target_std + stats.target_mean


class EpochBatches:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> None:
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("batch source must be nonempty and aligned")
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __call__(self) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        return [
            (self.x[index], self.y[index])
            for start in range(0, len(indices), self.batch_size)
            for index in [indices[start : start + self.batch_size]]
        ]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("metric vectors must be nonempty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metric vectors must be finite")
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator <= 0.0:
        r2, r2_reason = None, "undefined because ground truth is constant"
    else:
        r2 = float(1.0 - np.sum(residual ** 2) / denominator)
        r2_reason = None
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "R2_reason": r2_reason}


def evaluate_predictions(
    actual_domain: np.ndarray,
    predicted_domain: np.ndarray,
    definition: TargetDefinition,
    *,
    interval_low_domain: np.ndarray | None = None,
    interval_high_domain: np.ndarray | None = None,
) -> dict[str, Any]:
    actual_physical = definition.to_physical(actual_domain, prediction=False)
    predicted_physical = definition.to_physical(predicted_domain, prediction=True)
    result: dict[str, Any] = {
        "target": definition.name,
        "physical_unit": definition.physical_unit,
        "physical": regression_metrics(actual_physical, predicted_physical),
    }
    if definition.stored_domain == "log1p":
        result["log1p_domain"] = regression_metrics(actual_domain, np.maximum(predicted_domain, 0.0))
        result["transform"] = "train in log1p(KLOGH); physical KLOGH=expm1(clamp(log_prediction,0))"
    if interval_low_domain is not None and interval_high_domain is not None:
        low = definition.to_physical(interval_low_domain, prediction=True)
        high = definition.to_physical(interval_high_domain, prediction=True)
        result["uncertainty"] = {
            "method": "symmetric 90% OOF absolute-residual interval in model domain",
            "empirical_test_coverage": float(np.mean((actual_physical >= low) & (actual_physical <= high))),
            "mean_interval_width": float(np.mean(high - low)),
            "unit": definition.physical_unit,
        }
    return result


def _model_state(model: Any) -> dict[str, Any]:
    if hasattr(model, "weights") and hasattr(model, "bias"):
        return {
            "kind": "linear",
            "weights": np.asarray(model.weights).copy(),
            "bias": np.asarray(model.bias).copy(),
            "update_count": int(model.update_count),
        }
    if hasattr(model, "params"):
        return {
            "kind": "tiny_mlp",
            "params": {key: np.asarray(value).copy() for key, value in model.params.items()},
            "mom1": {key: np.asarray(value).copy() for key, value in model.mom1.items()},
            "mom2": {key: np.asarray(value).copy() for key, value in model.mom2.items()},
            "update_count": int(model.update_count),
        }
    raise TypeError(f"unsupported reservoir model state: {type(model).__name__}")


def _restore_model_state(model: Any, state: Mapping[str, Any]) -> None:
    if state["kind"] == "linear":
        model.weights = np.asarray(state["weights"], dtype=np.float64).copy()
        model.bias = np.asarray(state["bias"], dtype=np.float64).copy()
        model.update_count = int(state["update_count"])
        return
    if state["kind"] == "tiny_mlp":
        for key in model.params:
            model.params[key] = np.asarray(state["params"][key], dtype=np.float64).copy()
            model.mom1[key] = np.asarray(state["mom1"][key], dtype=np.float64).copy()
            model.mom2[key] = np.asarray(state["mom2"][key], dtype=np.float64).copy()
        model.update_count = int(state["update_count"])
        return
    raise ValueError(f"unsupported checkpoint model kind: {state['kind']!r}")


def _model_config(
    *, model_name: str, n_features: int, learning_rate: float, l2_strength: float, seed: int
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "n_features": n_features,
        "n_outputs": 1,
        "hidden_dim": 24,
        "learning_rate": learning_rate,
        "l2_strength": l2_strength,
        "weight_decay": l2_strength,
        "seed": seed,
    }


def _build_model(config: Mapping[str, Any], task_spec: TaskSpec) -> Any:
    kwargs = dict(config)
    model_name = str(kwargs.pop("model_name"))
    n_outputs = int(kwargs.pop("n_outputs"))
    if n_outputs != len(task_spec.targets):
        raise ValueError("model config output count does not match TaskSpec targets")
    if model_name == "tiny_mlp":
        kwargs.pop("l2_strength", None)
    else:
        kwargs.pop("weight_decay", None)
    return discover_model("property", model_name).build(task_spec, **kwargs)


def _seed_checkpoint_report(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    report = seed_everything(seed, strict=True, include_torch=False)
    return (
        {"root_seed": report.root_seed, "seed_tree": report.seed_tree},
        {**report.environment, "python": platform.python_version(), "runner": "reservoir-p4-numpy"},
    )


def _checkpoint_writer(
    *,
    model: Any,
    stats: FeatureStats,
    model_config: Mapping[str, Any],
    config_hash: str,
    split_hash: str,
    seed: int,
    target_name: str,
):
    seed_report, environment = _seed_checkpoint_report(seed)

    def writer(state: Any, path: Path) -> None:
        save_checkpoint(
            path,
            epoch=max(0, state.next_epoch - 1),
            model_state=_model_state(model),
            optimizer_state={
                "type": "model-owned SGD/Adam",
                "learning_rate": model_config["learning_rate"],
                "l2_strength": model_config["l2_strength"],
            },
            scheduler_state=None,
            scaler_state=stats.to_dict(),
            config_hash=config_hash,
            split_hash=split_hash,
            trainer_state=state.to_dict(),
            seed_report=seed_report,
            environment=environment,
            extra={"target": target_name, "model_name": model_config["model_name"]},
            include_torch_rng=False,
        )

    return writer


def _select_loaded(samples: Sequence[LoadedSample], ids: Iterable[str]) -> list[LoadedSample]:
    wanted = set(ids)
    selected = [sample for sample in samples if sample.index.sample_id in wanted]
    if {sample.index.sample_id for sample in selected} != wanted:
        raise RuntimeError("loaded development samples do not match locked fold IDs")
    return selected


def tiny_development_check(
    samples: Sequence[LoadedSample], *, task_spec: TaskSpec, model_name: str, seed: int
) -> dict[str, Any]:
    families = sorted({sample.index.family_id for sample in samples})
    if len(families) < 2:
        raise RuntimeError("tiny check requires two development families")
    chosen: list[LoadedSample] = []
    for family in families[:2]:
        chosen.extend([sample for sample in samples if sample.index.family_id == family][:8])
    stats = fit_fold_statistics(chosen)
    x = transform_inputs(chosen, stats)
    y = transform_target(chosen, stats)
    config = _model_config(
        model_name=model_name,
        n_features=x.shape[1],
        learning_rate=0.01,
        l2_strength=1e-3,
        seed=derive_seed(seed, "model", "tiny"),
    )
    model = _build_model(config, task_spec)
    initial = model.validation_loss((x, y))
    for _ in range(12):
        model.train_batch((x, y))
    final = model.validation_loss((x, y))
    prediction = model.predict_array(x)
    if prediction.shape != (len(chosen), 1) or not np.isfinite(prediction).all() or final >= initial:
        raise RuntimeError(f"tiny check failed: initial={initial}, final={final}")
    return {
        "status": "passed",
        "model": model_name,
        "samples": len(chosen),
        "families": families[:2],
        "initial_loss": float(initial),
        "final_loss": float(final),
    }


def _train_fold(
    fold: Fold,
    *,
    development: Sequence[LoadedSample],
    task_spec: TaskSpec,
    definition: TargetDefinition,
    run_root: Path,
    split_hash: str,
    base_config: Mapping[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    train_samples = _select_loaded(development, fold.train_sample_ids)
    validation_samples = _select_loaded(development, fold.validation_sample_ids)
    if {sample.index.family_id for sample in train_samples} & {
        sample.index.family_id for sample in validation_samples
    }:
        raise RuntimeError("fold train/validation mother-well families overlap")
    stats = fit_fold_statistics(train_samples)
    if set(stats.fit_sample_ids) != set(fold.train_sample_ids):
        raise RuntimeError("preprocessing was not fitted on exactly the locked fold train IDs")
    x_train = transform_inputs(train_samples, stats)
    y_train = transform_target(train_samples, stats)
    x_val = transform_inputs(validation_samples, stats)
    y_val = transform_target(validation_samples, stats)
    fold_seed = derive_seed(seed, "model", definition.name, fold.fold_id)
    model_config = {**base_config, "n_features": x_train.shape[1], "seed": fold_seed}
    model = _build_model(model_config, task_spec)
    config_hash = hash_payload({"fold": fold.fold_id, **model_config, "epochs": epochs, "batch_size": batch_size})
    fold_dir = run_root / "folds" / f"fold_{fold.fold_id}"
    state = train_with_validation(
        train_step=lambda batch: StepResult(
            loss_sum=model.train_batch(batch) * len(batch[0]), valid_count=len(batch[0])
        ),
        validation_step=lambda batch: StepResult(
            loss_sum=model.validation_loss(batch) * len(batch[0]), valid_count=len(batch[0])
        ),
        train_batches_fn=EpochBatches(
            x_train, y_train, batch_size, True, derive_seed(seed, "loader", definition.name, fold.fold_id)
        ),
        validation_batches_fn=EpochBatches(x_val, y_val, batch_size, False, fold_seed),
        config=TrainerConfig(max_epochs=epochs, min_epochs=min(10, epochs), patience=max(5, epochs // 6)),
        output_dir=fold_dir,
        checkpoint_writer=_checkpoint_writer(
            model=model,
            stats=stats,
            model_config=model_config,
            config_hash=config_hash,
            split_hash=split_hash,
            seed=fold_seed,
            target_name=definition.name,
        ),
    )
    best = load_checkpoint(fold_dir / "checkpoint_best.pkl")
    _restore_model_state(model, best["model_state"])
    prediction_domain = inverse_normalized_target(model.predict_array(x_val), stats)
    actual_domain = np.asarray([sample.target_domain for sample in validation_samples])
    metrics = evaluate_predictions(actual_domain, prediction_domain, definition)
    atomic_write_json(fold_dir / "normalization.json", stats.to_dict())
    _write_predictions(
        fold_dir / "predictions.csv",
        validation_samples,
        actual_domain,
        prediction_domain,
        definition,
    )
    return {
        "validation_sample_ids": [sample.index.sample_id for sample in validation_samples],
        "valid_label_count": len(validation_samples),
        "metrics": {"physical_MAE": metrics["physical"]["MAE"]},
        "all_metrics": metrics,
        "best_epoch": state.best_epoch,
        "best_validation_loss": state.best_val_loss,
        "normalization_fit_sample_ids_hash": hash_payload(sorted(stats.fit_sample_ids)),
        "prediction_domain": prediction_domain,
        "actual_domain": actual_domain,
    }


def _write_predictions(
    path: Path,
    samples: Sequence[LoadedSample],
    actual_domain: np.ndarray,
    predicted_domain: np.ndarray,
    definition: TargetDefinition,
    *,
    interval_low_domain: np.ndarray | None = None,
    interval_high_domain: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_physical = definition.to_physical(actual_domain, prediction=False)
    predicted_physical = definition.to_physical(predicted_domain, prediction=True)
    low = None if interval_low_domain is None else definition.to_physical(interval_low_domain, prediction=True)
    high = None if interval_high_domain is None else definition.to_physical(interval_high_domain, prediction=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "well_id",
                "mother_well_family",
                "depth_m",
                "actual_model_domain",
                "predicted_model_domain",
                f"actual_{definition.physical_unit}",
                f"predicted_{definition.physical_unit}",
                f"interval_low_{definition.physical_unit}",
                f"interval_high_{definition.physical_unit}",
            ]
        )
        for index, sample in enumerate(samples):
            writer.writerow(
                [
                    sample.index.sample_id,
                    sample.index.well_id,
                    sample.index.family_id,
                    sample.index.depth_m,
                    actual_domain[index],
                    predicted_domain[index],
                    actual_physical[index],
                    predicted_physical[index],
                    "" if low is None else low[index],
                    "" if high is None else high[index],
                ]
            )


def _refit_development(
    *,
    samples: Sequence[LoadedSample],
    task_spec: TaskSpec,
    definition: TargetDefinition,
    run_root: Path,
    split_hash: str,
    frozen_config: Mapping[str, Any],
    config_hash: str,
    seed: int,
) -> tuple[Any, FeatureStats, Path, dict[str, Any]]:
    stats = fit_fold_statistics(samples)
    x = transform_inputs(samples, stats)
    y = transform_target(samples, stats)
    model_config = {
        **dict(frozen_config["model"]),
        "n_features": x.shape[1],
        "seed": derive_seed(seed, "model", definition.name, "refit"),
    }
    model = _build_model(model_config, task_spec)
    batch_size = int(frozen_config["batch_size"])
    epochs = int(frozen_config["refit_epochs"])
    batches = EpochBatches(
        x, y, batch_size, True, derive_seed(seed, "loader", definition.name, "refit")
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(epochs):
        weighted_loss = 0.0
        count = 0
        for batch in batches():
            loss = float(model.train_batch(batch))
            weighted_loss += loss * len(batch[0])
            count += len(batch[0])
            global_step += 1
        history.append({"epoch": epoch, "refit_train_loss": weighted_loss / count, "valid_count": count})
    from ml_framework.trainer import TrainerState

    state = TrainerState(
        next_epoch=epochs,
        global_step=global_step,
        best_epoch=epochs - 1,
        best_val_loss=float(frozen_config["development_primary_score"]),
        epochs_without_improvement=0,
        stopped_early=False,
        history=history,
    )
    refit_dir = run_root / "refit"
    checkpoint_path = refit_dir / "checkpoint_best.pkl"
    writer = _checkpoint_writer(
        model=model,
        stats=stats,
        model_config=model_config,
        config_hash=config_hash,
        split_hash=split_hash,
        seed=int(model_config["seed"]),
        target_name=definition.name,
    )
    writer(state, checkpoint_path)
    atomic_write_json(refit_dir / "normalization.json", stats.to_dict())
    atomic_write_json(
        refit_dir / "history.json",
        {
            "history": history,
            "epoch_budget_source": "median best epoch across development LOGO folds",
            "validation_during_refit": False,
            "all_development_samples_used": True,
        },
    )
    restored = load_checkpoint(checkpoint_path)
    _restore_model_state(model, restored["model_state"])
    return model, stats, checkpoint_path, state.to_dict()


def _plot_prediction_truth(
    actual: np.ndarray, predicted: np.ndarray, definition: TargetDefinition, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.scatter(actual, predicted, s=12, alpha=0.6)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set_xlabel(f"Truth ({definition.physical_unit})")
    axis.set_ylabel(f"Prediction ({definition.physical_unit})")
    axis.set_title(f"{definition.name}: frozen-test prediction vs truth")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_residuals(
    actual: np.ndarray, predicted: np.ndarray, depths: np.ndarray, definition: TargetDefinition, path: Path
) -> None:
    residual = predicted - actual
    fig, axis = plt.subplots(figsize=(6.0, 4.8))
    points = axis.scatter(actual, residual, c=depths, cmap="viridis", s=13, alpha=0.7)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel(f"Truth ({definition.physical_unit})")
    axis.set_ylabel(f"Prediction - truth ({definition.physical_unit})")
    axis.set_title(f"{definition.name}: residual diagnostic")
    fig.colorbar(points, ax=axis, label="Measured depth (m)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_depth_curve(
    samples: Sequence[LoadedSample],
    actual: np.ndarray,
    predicted: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    definition: TargetDefinition,
    path: Path,
) -> None:
    depths = np.asarray([sample.index.depth_m for sample in samples])
    order = np.argsort(depths)
    fig, axis = plt.subplots(figsize=(6.2, 7.2))
    axis.scatter(actual[order], depths[order], label="truth", s=11, alpha=0.75)
    axis.scatter(predicted[order], depths[order], label="prediction", s=11, alpha=0.65)
    # Do not connect horizontal-well crossings; intervals are shown as points.
    axis.hlines(depths[order], low[order], high[order], color="tab:orange", alpha=0.14, linewidth=0.8)
    axis.invert_yaxis()
    axis.set_xlabel(f"{definition.name} ({definition.physical_unit})")
    axis.set_ylabel("Measured depth (m)")
    axis.set_title(f"{definition.name}: frozen-test depth track with 90% OOF interval")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_distribution_uncertainty(
    actual: np.ndarray,
    predicted: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    definition: TargetDefinition,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].hist(actual, bins=25, alpha=0.55, label="truth")
    axes[0].hist(predicted, bins=25, alpha=0.55, label="prediction")
    axes[0].set_xlabel(f"{definition.name} ({definition.physical_unit})")
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Frozen-test distribution")
    axes[0].legend()
    widths = high - low
    covered = (actual >= low) & (actual <= high)
    axes[1].hist(widths, bins=25, alpha=0.75, color="tab:orange")
    axes[1].set_xlabel(f"90% interval width ({definition.physical_unit})")
    axes[1].set_ylabel("Samples")
    axes[1].set_title(f"OOF residual interval; coverage={covered.mean():.3f}")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_visualizations(
    *,
    samples: Sequence[LoadedSample],
    actual_domain: np.ndarray,
    predicted_domain: np.ndarray,
    low_domain: np.ndarray,
    high_domain: np.ndarray,
    definition: TargetDefinition,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = definition.to_physical(actual_domain, prediction=False)
    predicted = definition.to_physical(predicted_domain, prediction=True)
    low = definition.to_physical(low_domain, prediction=True)
    high = definition.to_physical(high_domain, prediction=True)
    depths = np.asarray([sample.index.depth_m for sample in samples])
    paths = {
        "prediction_vs_truth.png": lambda path: _plot_prediction_truth(actual, predicted, definition, path),
        "residuals.png": lambda path: _plot_residuals(actual, predicted, depths, definition, path),
        "depth_curve.png": lambda path: _plot_depth_curve(
            samples, actual, predicted, low, high, definition, path
        ),
        "distribution_uncertainty.png": lambda path: _plot_distribution_uncertainty(
            actual, predicted, low, high, definition, path
        ),
    }
    for name, plotter in paths.items():
        plotter(output_dir / name)
    return list(paths)


def run_p4_target(
    *,
    task_spec: TaskSpec,
    definition: TargetDefinition,
    processed_dir: Path,
    guard_path: Path,
    output_dir: Path,
    mode: str = "baseline",
    model_name: str = "reservoir_ridge",
    epochs: int = 48,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    l2_strength: float = 1e-3,
    seed: int = 2693,
) -> dict[str, Any]:
    if mode not in {"audit", "smoke", "baseline"}:
        raise ValueError("mode must be audit, smoke or baseline")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or l2_strength < 0:
        raise ValueError("training parameters are outside their valid range")
    if tuple(task_spec.targets) != (definition.name,):
        raise ValueError("TaskSpec and target definition disagree")
    if set(task_spec.input_whitelist) & set(task_spec.forbidden_inputs):
        raise ValueError("TaskSpec contains an input leakage field")

    output_dir = Path(output_dir)
    run_root = create_run_layout(output_dir)
    atomic_write_json(run_root / "task_spec.json", task_spec.to_dict())
    indexed = scan_real_sources(
        processed_dir=processed_dir,
        guard_path=guard_path,
        task_id=task_spec.task_id,
        definition=definition,
    )
    availability = label_availability(indexed)
    test_family, selection = select_frozen_test_family(indexed)
    availability.update(
        {
            "target": definition.name,
            "label_version": task_spec.label_version,
            "mask": task_spec.target_masks[definition.name],
            "test_selection": selection,
            "source_files": {
                "train_hdf5_sha256": hash_file(Path(processed_dir) / "train.h5"),
                "test_hdf5_sha256": hash_file(Path(processed_dir) / "test.h5"),
                "guard_npz_sha256": hash_file(Path(guard_path)),
            },
            "source_paths_persisted": False,
        }
    )
    atomic_write_json(run_root / "label_availability.json", availability)
    manifest = build_independent_manifest(
        indexed,
        test_family=test_family,
        target_name=definition.name,
        label_version=task_spec.label_version,
        seed=seed,
        selection_record=selection,
    )
    split_hash = manifest.stable_hash()
    atomic_write_json(run_root / "split_manifest.json", manifest.to_dict())
    atomic_write_json(
        run_root / "hpo" / "plan.json",
        {
            "status": "not_run",
            "reason": "P4 delivery runs the preregistered simple baseline; no long HPO was authorized",
            "objective": "development OOF physical-unit MAE",
            "direction": "minimize",
            "test_access": "forbidden",
            "future_search_space": task_spec.hpo,
        },
    )
    lifecycle = ExperimentLifecycle(f"{task_spec.task_id}-{task_spec.label_version}")
    lifecycle.advance(
        ExperimentState.SPLIT_LOCKED,
        {"split_hash": split_hash, "test_family": test_family, "selection_basis": "label_availability"},
    )
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())
    if mode == "audit":
        status = {
            "status": "audit_complete",
            "target": definition.name,
            "split_hash": split_hash,
            "test_family": test_family,
            "requested_folds": manifest.requested_n_splits,
            "effective_folds": manifest.effective_n_splits,
        }
        atomic_write_json(run_root / "status.json", status)
        return status

    valid_by_id = {record.sample_id: record for record in indexed if record.label_valid}
    development_records = [valid_by_id[sid] for sid in manifest.development_sample_ids]
    # Numeric frozen-test arrays are intentionally not loaded here.
    development = load_samples(development_records, definition)
    tiny = tiny_development_check(
        development,
        task_spec=task_spec,
        model_name=model_name,
        seed=seed,
    )
    lifecycle.advance(ExperimentState.SMOKE_PASSED, tiny)
    atomic_write_json(run_root / "tiny_smoke.json", tiny)
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())
    if mode == "smoke":
        status = {
            "status": "smoke_passed",
            "target": definition.name,
            "test_consumed": False,
            "effective_folds": manifest.effective_n_splits,
            "tiny": tiny,
        }
        atomic_write_json(run_root / "status.json", status)
        return status

    base_config = _model_config(
        model_name=model_name,
        n_features=153,
        learning_rate=learning_rate,
        l2_strength=l2_strength,
        seed=seed,
    )
    fold_payloads: dict[int, dict[str, Any]] = {}

    def fold_runner(fold: Fold) -> Mapping[str, Any]:
        payload = _train_fold(
            fold,
            development=development,
            task_spec=task_spec,
            definition=definition,
            run_root=run_root,
            split_hash=split_hash,
            base_config=base_config,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        fold_payloads[fold.fold_id] = payload
        # Large vectors are persisted in the fold CSV, not duplicated in JSON.
        return {key: value for key, value in payload.items() if key not in {"prediction_domain", "actual_domain"}}

    cv_summary = run_development_cv(
        manifest,
        fold_runner,
        output_dir=run_root,
        primary_metric="physical_MAE",
        metric_direction="minimize",
    )
    ordered_predictions: dict[str, tuple[float, float]] = {}
    for fold in manifest.folds:
        payload = fold_payloads[fold.fold_id]
        for sid, actual, predicted in zip(
            payload["validation_sample_ids"], payload["actual_domain"], payload["prediction_domain"]
        ):
            ordered_predictions[str(sid)] = (float(actual), float(predicted))
    if set(ordered_predictions) != set(manifest.development_sample_ids):
        raise RuntimeError("OOF predictions do not cover development IDs exactly once")
    oof_actual = np.asarray([ordered_predictions[sid][0] for sid in manifest.development_sample_ids])
    oof_prediction = np.asarray([ordered_predictions[sid][1] for sid in manifest.development_sample_ids])
    oof_metrics = evaluate_predictions(oof_actual, oof_prediction, definition)
    uncertainty_quantile = float(np.quantile(np.abs(oof_prediction - oof_actual), 0.9))
    atomic_write_json(
        run_root / "oof" / "metrics.json",
        {**oof_metrics, "absolute_residual_q90_model_domain": uncertainty_quantile},
    )
    _write_predictions(
        run_root / "oof" / "predictions.csv",
        _select_loaded(development, manifest.development_sample_ids),
        oof_actual,
        oof_prediction,
        definition,
    )
    lifecycle.advance(
        ExperimentState.CV_COMPLETE,
        {
            "oof_predictions_hash": hash_file(run_root / "oof" / "predictions.csv"),
            "primary_metric": "physical_MAE",
            "primary_score": oof_metrics["physical"]["MAE"],
            "effective_folds": manifest.effective_n_splits,
        },
    )
    best_epochs = [int(payload["best_epoch"]) + 1 for payload in fold_payloads.values()]
    refit_epochs = max(1, int(round(float(np.median(best_epochs)))))
    frozen_config = {
        "model": base_config,
        "batch_size": batch_size,
        "requested_cv_epochs": epochs,
        "refit_epochs": refit_epochs,
        "refit_epoch_budget_source": "median fold best epoch",
        "development_primary_score": float(oof_metrics["physical"]["MAE"]),
        "primary_metric": "physical_MAE",
        "metric_direction": "minimize",
        "uncertainty_absolute_residual_q90_model_domain": uncertainty_quantile,
        "hpo_performed": False,
        "root_seed": seed,
    }
    config_hash = hash_payload(frozen_config)
    atomic_write_json(run_root / "config.json", frozen_config)
    lifecycle.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": config_hash})
    model, refit_stats, checkpoint_path, _refit_state = _refit_development(
        samples=development,
        task_spec=task_spec,
        definition=definition,
        run_root=run_root,
        split_hash=split_hash,
        frozen_config=frozen_config,
        config_hash=config_hash,
        seed=seed,
    )
    checkpoint_hash = hash_file(checkpoint_path)
    lifecycle.advance(
        ExperimentState.REFIT_COMPLETE,
        {
            "checkpoint_hash": checkpoint_hash,
            "development_samples": len(development),
            "test_loaded": False,
        },
    )
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())

    # The single-use firewall is advanced before numerical test arrays are opened.
    lifecycle.consume_test(
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        split_hash=split_hash,
    )
    test_records = [valid_by_id[sid] for sid in manifest.test_sample_ids]
    test_samples = load_samples(test_records, definition)
    x_test = transform_inputs(test_samples, refit_stats)
    prediction_domain = inverse_normalized_target(model.predict_array(x_test), refit_stats)
    actual_domain = np.asarray([sample.target_domain for sample in test_samples])
    low_domain = prediction_domain - uncertainty_quantile
    high_domain = prediction_domain + uncertainty_quantile
    metrics = evaluate_predictions(
        actual_domain,
        prediction_domain,
        definition,
        interval_low_domain=low_domain,
        interval_high_domain=high_domain,
    )
    atomic_write_json(run_root / "frozen_test" / "metrics.json", metrics)
    _write_predictions(
        run_root / "frozen_test" / "predictions.csv",
        test_samples,
        actual_domain,
        prediction_domain,
        definition,
        interval_low_domain=low_domain,
        interval_high_domain=high_domain,
    )
    figures = _write_visualizations(
        samples=test_samples,
        actual_domain=actual_domain,
        predicted_domain=prediction_domain,
        low_domain=low_domain,
        high_domain=high_domain,
        definition=definition,
        output_dir=run_root / "visualizations",
    )
    lifecycle.advance(
        ExperimentState.VERIFIED,
        {
            "metrics_hash": hash_file(run_root / "frozen_test" / "metrics.json"),
            "prediction_hash": hash_file(run_root / "frozen_test" / "predictions.csv"),
            "required_figures": figures,
        },
    )
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())
    status = {
        "status": "complete",
        "target": definition.name,
        "label_version": task_spec.label_version,
        "model": model_name,
        "test_family": test_family,
        "test_consumed_once": True,
        "test_samples": len(test_samples),
        "development_samples": len(development),
        "requested_folds": manifest.requested_n_splits,
        "effective_folds": manifest.effective_n_splits,
        "downgrade_reason": manifest.downgrade_reason,
        "physical_metrics": metrics["physical"],
        "log1p_diagnostics": metrics.get("log1p_domain"),
        "unresolved": [
            "only one frozen-test mother-well family is available",
            "weak MD-to-TWT tie and existing Hugin-filtered samples are inherited from the reservoir build",
            "uncertainty is an OOF residual interval, not a calibrated probabilistic model",
        ],
    }
    atomic_write_json(run_root / "status.json", status)
    artifact_manifest = ArtifactManifest(f"{task_spec.task_id}-{task_spec.label_version}", run_root)
    canonical = {
        "task_spec.json": "task_spec",
        "label_availability.json": "label_availability",
        "split_manifest.json": "split_manifest",
        "config.json": "frozen_config",
        "lifecycle.json": "test_firewall",
        "tiny_smoke.json": "tiny_smoke",
        "hpo/plan.json": "hpo_plan",
        "oof/summary.json": "cv_summary",
        "oof/metrics.json": "oof_metrics",
        "oof/predictions.csv": "oof_predictions",
        "refit/checkpoint_best.pkl": "checkpoint",
        "refit/normalization.json": "train_only_preprocessing",
        "refit/history.json": "refit_history",
        "frozen_test/metrics.json": "frozen_test_metrics",
        "frozen_test/predictions.csv": "frozen_test_predictions",
        "status.json": "status",
    }
    for figure in figures:
        canonical[f"visualizations/{figure}"] = "visualization"
    for relative, role in canonical.items():
        artifact_manifest.register(relative, role=role)
    artifact_manifest.write()
    return status


def _canonical_archive_well(member: str) -> tuple[str, str] | None:
    parts = member.split("/")
    if len(parts) < 4:
        return None
    raw = parts[2].replace("_", "/", 1)
    match = re.match(r"^(15/9-19|15/9-F-\d+)", raw)
    return (raw, match.group(1)) if match else None


def audit_phie_label_version(well_log_zip: Path, output_dir: Path, task_spec: TaskSpec) -> dict[str, Any]:
    """Audit exact PHIE separately; LFP_PHIE is evidence only, never an alias."""
    well_log_zip = Path(well_log_zip)
    if not well_log_zip.is_file():
        raise FileNotFoundError(well_log_zip)
    exact_hits: list[dict[str, Any]] = []
    lfp_hits: list[dict[str, Any]] = []
    scanned = 0
    with zipfile.ZipFile(well_log_zip) as archive:
        members = [
            member
            for member in archive.namelist()
            if member.lower().endswith(".las")
            and ("/05.PETROPHYSICAL INTERPRETATION/" in member or "/06.LFP/" in member)
        ]
        for member in members:
            scanned += 1
            header = archive.read(member).decode("latin-1", errors="replace").split("~A", 1)[0]
            well = _canonical_archive_well(member)
            if well is None:
                continue
            row = {"member": member, "well_id": well[0], "family_id": well[1]}
            if re.search(r"(?mi)^\s*PHIE\s*\.", header):
                exact_hits.append(row)
            if re.search(r"(?mi)^\s*LFP_PHIE\s*\.", header):
                lfp_hits.append(row)
    exact_families = sorted({row["family_id"] for row in exact_hits})
    lfp_families = sorted({row["family_id"] for row in lfp_hits})
    feasible = len(exact_families) >= 3 and DEFAULT_TEST_FAMILY in exact_families
    evidence = {
        "label_version": task_spec.label_version,
        "target": "PHIE",
        "archive_sha256": hash_file(well_log_zip),
        "source_path_persisted": False,
        "las_files_scanned": scanned,
        "exact_PHIE": {"hits": exact_hits, "families": exact_families},
        "LFP_PHIE_not_an_alias_or_target": {"hits": lfp_hits, "families": lfp_families},
        "feasibility_gate": {
            "requires": "exact PHIE in frozen test plus at least two development mother-well families",
            "feasible": feasible,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "task_spec.json", task_spec.to_dict())
    atomic_write_json(output_dir / "label_availability.json", evidence)
    status = {
        "status": "feasible_not_run" if feasible else "not_feasible",
        "target": "PHIE",
        "label_version": task_spec.label_version,
        "reason": (
            "exact PHIE met the preregistered family gate; run remains separate from PHIF"
            if feasible
            else "exact PHIE has insufficient mother-well-family support; LFP_PHIE is not substituted"
        ),
        "exact_phie_families": exact_families,
        "lfp_phie_families_excluded": lfp_families,
        "mixed_with_PHIF": False,
    }
    atomic_write_json(output_dir / "status.json", status)
    return status
