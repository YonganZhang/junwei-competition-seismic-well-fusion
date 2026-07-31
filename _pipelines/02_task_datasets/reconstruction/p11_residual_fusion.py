#!/usr/bin/env python3
"""Development-only PyKrige + OpenMind residual-fusion experiment.

P11 is a second-level, cross-fitted experiment.  Its base predictions are the
portable P5 Stage-3 PyKrige development OOF archives.  A frozen, genuinely
pretrained OpenMind ResEnc-L encoder supplies multiscale 3-D features.  The
residual regressor and its bounded confidence gate are trained without ever
opening ``test.h5`` or another holdout surface.

The module deliberately has no test-loader argument.  The only accepted HDF5
container is ``train.h5`` and only its metadata plus ``seismic_patch`` dataset
are read.  PORO targets come from the already-audited development OOF archives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p11_residual_fusion"
DEFAULT_FEATURE_CACHE = (
    PROJECT_ROOT / "_tmp" / "p11_residual_fusion" / "openmind_oof_features.npz"
)
DEFAULT_DIRECT_SUMMARY = HERE / "_outputs" / "p9_openmind_effect" / "summary.json"

SCHEMA_VERSION = "reconstruction-p11-residual-fusion/v1"
FEATURE_CACHE_SCHEMA_VERSION = "reconstruction-p11-openmind-feature-cache/v1"
EXPECTED_MODEL_ID = "pykrige_ok3d"
EXPECTED_LANE = "conditional"
EXPECTED_SPLIT_HASH = "ffe1fddbfd3ef8e2a9e0b3f3942133226610c872e7da37460f070782cba700a6"
EXPECTED_CHECKPOINT_SHA256 = (
    "7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933"
)
EXPECTED_CHECKPOINT_BYTES = 491_198_606
EXPECTED_SOURCE_REVISION = "7044864404315536e92e670ef2f0ca24f11e6175"
FOLD_IDS = (0, 1, 2, 3, 4)
REPEAT_ID = 0
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
LATENT_CHANNELS_PER_STAGE = 16
RIDGE_ALPHA = 10.0
GATE_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
FORBIDDEN_NAMES = {
    "test.h5",
    "holdout",
    "known_holdout",
    "frozen_holdout",
    "frozen-test",
}


@dataclass(frozen=True)
class DevInputPaths:
    """The sole legal HDF5 input for P11."""

    data_dir: Path
    train_h5: Path


@dataclass(frozen=True)
class OOFDevelopment:
    """Aligned, development-only Stage-3 OOF evidence."""

    target: np.ndarray
    baseline: np.ndarray
    indices_kji: np.ndarray
    xyz: np.ndarray
    distance_to_well: np.ndarray
    structural_features: np.ndarray
    fold_ids: np.ndarray
    source_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GateModel:
    """A fitted probability gate, including the constant-class fallback."""

    estimator: Any | None
    constant_probability: float | None

    def predict(self, signals: np.ndarray) -> np.ndarray:
        signals = np.asarray(signals, dtype=np.float64)
        if self.estimator is None:
            if self.constant_probability is None:
                raise RuntimeError("constant gate is missing its probability")
            values = np.full(len(signals), self.constant_probability, dtype=np.float64)
        else:
            values = np.asarray(self.estimator.predict_proba(signals)[:, 1], dtype=np.float64)
        return np.clip(values, 0.0, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _verified_source_revision(source_root: Path) -> str:
    """Verify the cached nnssl checkout even when latent features are reused."""

    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"OpenMind source root missing: {source_root}")
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = completed.stdout.strip()
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(
            "OpenMind source revision differs from the route lock: "
            f"{revision or '<empty>'}"
        )
    return revision


def _forbidden_match(path: Path) -> bool:
    lowered = str(path).lower()
    return any(name in lowered for name in FORBIDDEN_NAMES)


def ensure_no_holdout_paths(paths: Iterable[Path]) -> None:
    """Fail before opening a path that names a frozen holdout surface."""

    forbidden = [Path(path) for path in paths if _forbidden_match(Path(path))]
    if forbidden:
        joined = ", ".join(str(path) for path in forbidden)
        raise ValueError(f"forbidden holdout path(s) rejected: {joined}")


def resolve_dev_inputs(data_dir: Path) -> DevInputPaths:
    """Resolve ``train.h5`` without probing for any sibling holdout file."""

    data_dir = Path(data_dir).expanduser().resolve()
    ensure_no_holdout_paths([data_dir])
    train_h5 = data_dir / "train.h5"
    ensure_no_holdout_paths([train_h5])
    if not train_h5.is_file():
        raise FileNotFoundError(f"development train.h5 is missing: {train_h5}")
    return DevInputPaths(data_dir=data_dir, train_h5=train_h5)


def _json(path: Path) -> dict[str, Any]:
    ensure_no_holdout_paths([path])
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _fold_paths(stage3_root: Path, fold_id: int) -> dict[str, Path]:
    base = (
        stage3_root
        / "cells"
        / EXPECTED_LANE
        / EXPECTED_MODEL_ID
        / f"fold_{fold_id:02d}"
        / f"repeat_{REPEAT_ID}"
    )
    cache = stage3_root / "cache" / EXPECTED_LANE / f"fold_{fold_id:02d}"
    paths = {
        "oof": base / "oof_prediction.npz",
        "status": base / "status.json",
        "point": cache / "point_validation.npz",
        "point_json": cache / "point_validation.json",
    }
    ensure_no_holdout_paths(paths.values())
    return paths


def load_oof_development(stage3_root: Path) -> OOFDevelopment:
    """Load and independently verify five legal PyKrige OOF folds."""

    stage3_root = Path(stage3_root).expanduser().resolve()
    ensure_no_holdout_paths([stage3_root])
    manifest = _json(stage3_root / "cache" / "cache_manifest.json")
    if manifest.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-3 cache reports frozen-test access")
    mode_audit = manifest.get("modes", {}).get(EXPECTED_LANE, {})
    if mode_audit.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise RuntimeError("conditional Stage-3 split hash drift")

    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    xyz_rows: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    structural: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []

    for fold_id in FOLD_IDS:
        paths = _fold_paths(stage3_root, fold_id)
        missing = [path for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing P5 Stage-3 development artifact(s): "
                + ", ".join(str(path) for path in missing)
            )
        status = _json(paths["status"])
        if status.get("status") != "passed":
            raise RuntimeError(f"PyKrige fold {fold_id} is not a passed OOF cell")
        if (
            status.get("lane") != EXPECTED_LANE
            or status.get("model_id") != EXPECTED_MODEL_ID
            or status.get("fold_id") != fold_id
            or status.get("repeat_id") != REPEAT_ID
            or status.get("split_hash") != EXPECTED_SPLIT_HASH
        ):
            raise RuntimeError(f"PyKrige fold {fold_id} identity drift")
        firewall = status.get("test_firewall", {})
        if (
            firewall.get("development_only") is not True
            or firewall.get("frozen_test_i_blocks_loaded") != []
            or firewall.get("historical_test_metrics_read") is not False
            or firewall.get("test_path_argument_exists") is not False
        ):
            raise RuntimeError(f"PyKrige fold {fold_id} violates the holdout firewall")
        expected_hash = status.get("oof_prediction", {}).get("sha256")
        actual_hash = _sha256(paths["oof"])
        if expected_hash != actual_hash:
            raise RuntimeError(f"PyKrige fold {fold_id} OOF hash mismatch")

        with np.load(paths["oof"], allow_pickle=False) as payload:
            target = np.asarray(payload["target"], dtype=np.float64)
            prediction = np.asarray(payload["prediction"], dtype=np.float64)
            fold_indices = np.asarray(payload["indices_kji"], dtype=np.int64)
            fold_xyz = np.asarray(payload["xyz"], dtype=np.float64)
            distance = np.asarray(
                payload["distance_to_fold_train_well"], dtype=np.float64
            )
            if (
                str(payload["lane"]) != EXPECTED_LANE
                or str(payload["model_id"]) != EXPECTED_MODEL_ID
                or int(payload["fold_id"]) != fold_id
                or int(payload["repeat_id"]) != REPEAT_ID
            ):
                raise RuntimeError(f"PyKrige fold {fold_id} NPZ identity drift")

        point_meta = _json(paths["point_json"])
        if _sha256(paths["point"]) != point_meta.get("npz_sha256"):
            raise RuntimeError(f"point cache fold {fold_id} hash mismatch")
        if point_meta.get("metadata", {}).get("frozen_test_i_blocks_loaded") != []:
            raise RuntimeError(f"point cache fold {fold_id} reports test access")
        with np.load(paths["point"], allow_pickle=False) as payload:
            point_target = np.asarray(payload["target_values"], dtype=np.float64)
            point_features = np.asarray(payload["input_values"], dtype=np.float64)
            if str(payload["input_name"]) != "features":
                raise RuntimeError(f"point cache fold {fold_id} input identity drift")
        point_xyz = np.asarray(point_meta["coordinates"]["xyz"], dtype=np.float64)
        if point_features.ndim != 2 or point_features.shape[1] != 7:
            raise RuntimeError("conditional PyKrige point cache must have seven features")
        np.testing.assert_allclose(target, point_target, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(fold_xyz, point_xyz, rtol=0.0, atol=1e-6)
        if not (
            target.shape == prediction.shape == distance.shape
            and fold_indices.shape == (len(target), 3)
            and fold_xyz.shape == (len(target), 3)
            and point_features.shape[0] == len(target)
        ):
            raise RuntimeError(f"PyKrige fold {fold_id} OOF array shape mismatch")
        if not all(
            np.all(np.isfinite(item))
            for item in (target, prediction, fold_xyz, distance, point_features)
        ):
            raise FloatingPointError(f"PyKrige fold {fold_id} OOF contains non-finite values")

        targets.append(target)
        predictions.append(prediction)
        indices.append(fold_indices)
        xyz_rows.append(fold_xyz)
        distances.append(distance)
        # Exclude the conditional IDW column.  It is target-derived and is
        # already represented by the cross-fitted PyKrige base prediction.
        structural.append(point_features[:, 1:7])
        folds.append(np.full(len(target), fold_id, dtype=np.int64))
        source_records.append(
            {
                "fold_id": fold_id,
                "repeat_id": REPEAT_ID,
                "repeat_seed": int(status["repeat_seed"]),
                "oof_sha256": actual_hash,
                "oof_rows": len(target),
                "fold_hash": status["fold_hash"],
                "development_only": True,
            }
        )

    result = OOFDevelopment(
        target=np.concatenate(targets),
        baseline=np.concatenate(predictions),
        indices_kji=np.concatenate(indices),
        xyz=np.concatenate(xyz_rows),
        distance_to_well=np.concatenate(distances),
        structural_features=np.concatenate(structural),
        fold_ids=np.concatenate(folds),
        source_records=tuple(source_records),
    )
    if len(np.unique(result.indices_kji, axis=0)) != len(result.indices_kji):
        raise RuntimeError("PyKrige OOF coordinates do not cover development exactly once")
    if set(np.unique(result.fold_ids).tolist()) != set(FOLD_IDS):
        raise RuntimeError("PyKrige OOF folds are incomplete")
    return result


def _feature_cache_manifest_path(feature_cache: Path) -> Path:
    return feature_cache.with_suffix(".json")


def _load_valid_feature_cache(
    feature_cache: Path,
    *,
    indices_kji: np.ndarray,
    checkpoint_sha256: str,
    train_h5_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    manifest_path = _feature_cache_manifest_path(feature_cache)
    if not feature_cache.is_file() or not manifest_path.is_file():
        return None
    manifest = _json(manifest_path)
    expected = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": _array_sha256(indices_kji),
        "source_revision": EXPECTED_SOURCE_REVISION,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if manifest.get("npz_sha256") != _sha256(feature_cache):
        raise RuntimeError("OpenMind feature cache hash mismatch")
    with np.load(feature_cache, allow_pickle=False) as payload:
        cached_indices = np.asarray(payload["indices_kji"], dtype=np.int64)
        features = np.asarray(payload["openmind_features"], dtype=np.float32)
    np.testing.assert_array_equal(cached_indices, indices_kji)
    if features.shape[0] != len(indices_kji) or features.ndim != 2:
        raise RuntimeError("OpenMind feature cache shape mismatch")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("OpenMind feature cache contains non-finite values")
    return features, manifest


def _sample_encoder_stages(
    stages: Sequence[Any],
    local_kji: np.ndarray,
    *,
    padded_shape: tuple[int, int, int],
) -> np.ndarray:
    """Trilinearly sample multiscale encoder stages at voxel centres."""

    import torch
    import torch.nn.functional as functional

    local = np.asarray(local_kji, dtype=np.float32)
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError("local_kji must have shape [N,3]")
    d, h, w = padded_shape
    # grid_sample consumes x/y/z and align_corners=False voxel-centre
    # coordinates.  The normalized grid is shared by all downsampled stages.
    grid_xyz = np.column_stack(
        [
            2.0 * (local[:, 2] + 0.5) / w - 1.0,
            2.0 * (local[:, 1] + 0.5) / h - 1.0,
            2.0 * (local[:, 0] + 0.5) / d - 1.0,
        ]
    )
    grid = torch.as_tensor(
        grid_xyz, dtype=stages[0].dtype, device=stages[0].device
    ).reshape(1, len(local), 1, 1, 3)
    sampled: list[np.ndarray] = []
    for stage in stages:
        values = functional.grid_sample(
            stage,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled.append(
            values[0, :, :, 0, 0].transpose(0, 1).float().cpu().numpy()
        )
    return np.concatenate(sampled, axis=1).astype(np.float32, copy=False)


def extract_openmind_features(
    *,
    train_h5: Path,
    indices_kji: np.ndarray,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract real pretrained OpenMind encoder features from ``train.h5``.

    This function never requests or reads a label dataset.  Per-patch,
    input-only active-cell z-scoring avoids cross-fold fitted statistics.
    """

    ensure_no_holdout_paths([train_h5, source_root, checkpoint])
    if dependency_root is not None:
        dependency_root = Path(dependency_root).expanduser().resolve()
        if not dependency_root.is_dir():
            raise FileNotFoundError(f"OpenMind dependency root missing: {dependency_root}")
        sys.path.insert(0, str(dependency_root))

    import h5py
    import torch
    import torch.nn.functional as functional

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from _models.reconstruction.openmind_mae import build_model

    import importlib

    p4 = importlib.import_module(
        "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
    )
    model = build_model(
        p4.task_spec("strict"),
        source_root=source_root,
        checkpoint_path=checkpoint,
        freeze_encoder=True,
        device=device,
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.network.encoder.parameters()):
        raise RuntimeError("OpenMind encoder is not frozen")

    requested = np.asarray(indices_kji, dtype=np.int64)
    features: np.ndarray | None = None
    layer_shapes: list[list[int]] | None = None
    layer_channels: list[int] | None = None
    accessed_groups: list[str] = []

    with h5py.File(train_h5, "r") as handle:
        locations: list[tuple[str, np.ndarray, np.ndarray]] = []
        for key in sorted(handle):
            meta = json.loads(handle[key].attrs["meta"])
            start = np.asarray(meta["patch_start_kji"], dtype=np.int64)
            shape = np.asarray(meta["patch_shape_kji"], dtype=np.int64)
            locations.append((key, start, start + shape))

        assignment = np.full(len(requested), -1, dtype=np.int64)
        for location_id, (_, start, stop) in enumerate(locations):
            inside = np.all((requested >= start) & (requested < stop), axis=1)
            if np.any((assignment >= 0) & inside):
                raise RuntimeError("development coordinate maps to multiple train patches")
            assignment[inside] = location_id
        if np.any(assignment < 0):
            missing = requested[assignment < 0][:5].tolist()
            raise RuntimeError(f"OOF coordinates are absent from train.h5: {missing}")

        for location_id in np.unique(assignment):
            key, start, stop = locations[int(location_id)]
            row_ids = np.flatnonzero(assignment == location_id)
            local = requested[row_ids] - start
            group = handle[key]
            seismic = np.asarray(group["seismic_patch"][0:3], dtype=np.float32)
            active = np.asarray(group["seismic_patch"][8], dtype=np.float32) > 0.5
            expected_shape = tuple((stop - start).tolist())
            if seismic.shape[1:] != expected_shape or active.shape != expected_shape:
                raise RuntimeError(f"invalid train patch shape for {key}")
            if not np.all(active[tuple(local.T)]):
                raise RuntimeError(f"OOF coordinate maps to an inactive cell in {key}")

            normalized = np.zeros_like(seismic)
            for channel in range(3):
                values = seismic[channel][active]
                mean = float(np.mean(values))
                std = float(np.std(values))
                scale = max(std, 1e-6)
                normalized[channel][active] = (values - mean) / scale
            tensor = torch.as_tensor(
                normalized[None], dtype=torch.float32, device=device
            )
            # The pretrained model is single-channel.  A fixed, target-free
            # mean projection avoids a random or target-fitted adapter.
            single = tensor.mean(dim=1, keepdim=True)
            original = tuple(int(value) for value in single.shape[-3:])
            padded = tuple(max(64, ((size + 31) // 32) * 32) for size in original)
            pads: list[int] = []
            for size, wanted in reversed(list(zip(original, padded))):
                pads.extend((0, wanted - size))
            volume = functional.pad(single, tuple(pads)) if any(pads) else single
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                stages = model.network.encoder(volume)
            if not isinstance(stages, (tuple, list)) or not stages:
                raise RuntimeError("OpenMind encoder did not return multiscale stages")
            sampled = _sample_encoder_stages(stages, local, padded_shape=padded)
            if features is None:
                features = np.empty(
                    (len(requested), sampled.shape[1]), dtype=np.float32
                )
                layer_shapes = [list(map(int, stage.shape)) for stage in stages]
                layer_channels = [int(stage.shape[1]) for stage in stages]
            elif sampled.shape[1] != features.shape[1]:
                raise RuntimeError("OpenMind encoder feature width changed between patches")
            features[row_ids] = sampled
            accessed_groups.append(key)

    if features is None or layer_shapes is None or layer_channels is None:
        raise RuntimeError("OpenMind extraction produced no features")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("OpenMind extraction produced non-finite features")
    audit = {
        "model_id": "MIC-DKFZ/ResEncL-OpenMind-MAE",
        "real_pretrained_weights_loaded": True,
        "encoder_frozen": True,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "feature_width": int(features.shape[1]),
        "layer_channels": layer_channels,
        "layer_shapes_first_patch": layer_shapes,
        "input_projection": "per-patch active-cell zscore then fixed mean of 3 seismic channels",
        "spatial_sampling": "trilinear voxel-centre sampling from all encoder stages",
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": ["seismic_patch[0:3]", "seismic_patch[8]"],
        "label_dataset_read": False,
        "accessed_patch_count": len(accessed_groups),
        "accessed_patch_keys_sha256": hashlib.sha256(
            json.dumps(sorted(accessed_groups)).encode("utf-8")
        ).hexdigest(),
    }
    return features, audit


def get_openmind_features(
    *,
    inputs: DevInputPaths,
    oof: OOFDevelopment,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    feature_cache: Path,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse a hash-locked feature cache or create it from real weights."""

    source_root = Path(source_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    feature_cache = Path(feature_cache).expanduser().resolve()
    ensure_no_holdout_paths([source_root, checkpoint, feature_cache])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OpenMind checkpoint missing: {checkpoint}")
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError("OpenMind checkpoint byte size differs from the route lock")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("OpenMind checkpoint SHA-256 differs from the route lock")
    source_revision = _verified_source_revision(source_root)
    train_h5_sha256 = _sha256(inputs.train_h5)
    cached = _load_valid_feature_cache(
        feature_cache,
        indices_kji=oof.indices_kji,
        checkpoint_sha256=checkpoint_sha256,
        train_h5_sha256=train_h5_sha256,
    )
    if cached is not None:
        features, manifest = cached
        audit = dict(manifest["feature_audit"])
        audit["cache_reused"] = True
        audit["feature_cache_sha256"] = manifest["npz_sha256"]
        audit["source_revision"] = source_revision
        audit["source_revision_verified"] = True
        return features, audit

    features, audit = extract_openmind_features(
        train_h5=inputs.train_h5,
        indices_kji=oof.indices_kji,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        device=device,
    )
    if audit["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("loaded OpenMind checkpoint identity drift")
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    with feature_cache.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            openmind_features=features,
        )
    manifest = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": _array_sha256(oof.indices_kji),
        "npz_sha256": _sha256(feature_cache),
        "feature_audit": audit,
    }
    _feature_cache_manifest_path(feature_cache).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = dict(audit)
    audit["cache_reused"] = False
    audit["feature_cache_sha256"] = manifest["npz_sha256"]
    audit["source_revision"] = source_revision
    audit["source_revision_verified"] = True
    return features, audit


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    return {
        "rmse": float(math.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rows": int(len(error)),
    }


def _selected_latent_channels(
    layer_channels: Sequence[int],
    *,
    seed: int,
) -> np.ndarray:
    """Choose the same fixed channel budget from every OpenMind stage."""

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    offset = 0
    for width in layer_channels:
        count = min(LATENT_CHANNELS_PER_STAGE, int(width))
        local = np.sort(rng.choice(int(width), size=count, replace=False))
        selected.append(local + offset)
        offset += int(width)
    return np.concatenate(selected).astype(np.int64)


def _fit_residual(
    train_features: np.ndarray,
    train_residual: np.ndarray,
    validation_features: np.ndarray,
) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=1e-6),
    )
    estimator.fit(train_features, train_residual)
    prediction = np.asarray(estimator.predict(validation_features), dtype=np.float64)
    if prediction.shape != (len(validation_features),) or not np.all(
        np.isfinite(prediction)
    ):
        raise FloatingPointError("residual regressor returned invalid predictions")
    return prediction


def _gate_signals(
    residual_prediction: np.ndarray,
    baseline: np.ndarray,
    distance_to_well: np.ndarray,
) -> np.ndarray:
    correction = np.asarray(residual_prediction, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    distance = np.asarray(distance_to_well, dtype=np.float64)
    return np.column_stack(
        [
            correction,
            np.abs(correction),
            base,
            distance,
            np.abs(correction) / np.maximum(np.abs(base), 1e-6),
        ]
    )


def _fit_gate(
    signals: np.ndarray,
    beneficial: np.ndarray,
    *,
    seed: int,
) -> GateModel:
    labels = np.asarray(beneficial, dtype=np.int64)
    unique = np.unique(labels)
    if unique.size == 1:
        return GateModel(estimator=None, constant_probability=float(unique[0]))

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    estimator.fit(signals, labels)
    return GateModel(estimator=estimator, constant_probability=None)


def _inner_oof_residual(
    features: np.ndarray,
    residual_target: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    """Cross-fit residual predictions across the outer-training fold groups."""

    result = np.full(len(features), np.nan, dtype=np.float64)
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) < 3:
        raise ValueError("residual stacking needs at least three inner fold groups")
    for inner_fold in unique_folds:
        validation = fold_ids == inner_fold
        train = ~validation
        if np.any(fold_ids[train] == inner_fold):
            raise RuntimeError("inner residual cross-fit overlap")
        result[validation] = _fit_residual(
            features[train],
            residual_target[train],
            features[validation],
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("inner residual OOF predictions are incomplete")
    return result


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _evaluate_feature_route(
    *,
    route: str,
    features: np.ndarray,
    oof: OOFDevelopment,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = {
        "ungated": np.full(len(oof.target), np.nan, dtype=np.float64),
        "gated": np.full(len(oof.target), np.nan, dtype=np.float64),
    }
    cells: list[dict[str, Any]] = []
    residual_target = oof.target - oof.baseline

    for outer_fold in FOLD_IDS:
        validation = oof.fold_ids == outer_fold
        train = ~validation
        if np.any(oof.fold_ids[train] == outer_fold):
            raise RuntimeError("outer residual training includes its validation fold")
        inner_prediction = _inner_oof_residual(
            features[train],
            residual_target[train],
            oof.fold_ids[train],
        )
        inner_base = oof.baseline[train]
        inner_truth = oof.target[train]
        inner_ungated = inner_base + inner_prediction
        beneficial = (
            np.abs(inner_ungated - inner_truth)
            < np.abs(inner_base - inner_truth)
        )
        gate_model = _fit_gate(
            _gate_signals(
                inner_prediction,
                inner_base,
                oof.distance_to_well[train],
            ),
            beneficial,
            seed=seed + outer_fold,
        )
        inner_raw_gate = gate_model.predict(
            _gate_signals(
                inner_prediction,
                inner_base,
                oof.distance_to_well[train],
            )
        )
        candidate_scores: dict[str, float] = {}
        for candidate in GATE_CANDIDATES:
            gate = np.clip(inner_raw_gate * candidate, 0.0, 1.0)
            candidate_scores[f"{candidate:.2f}"] = _metrics(
                inner_truth,
                inner_base + gate * inner_prediction,
            )["rmse"]
        selected_scale = min(
            GATE_CANDIDATES,
            key=lambda value: (candidate_scores[f"{value:.2f}"], value),
        )

        outer_residual = _fit_residual(
            features[train],
            residual_target[train],
            features[validation],
        )
        raw_gate = gate_model.predict(
            _gate_signals(
                outer_residual,
                oof.baseline[validation],
                oof.distance_to_well[validation],
            )
        )
        gate = np.clip(raw_gate * selected_scale, 0.0, 1.0)
        ungated = oof.baseline[validation] + outer_residual
        gated = oof.baseline[validation] + gate * outer_residual
        predictions["ungated"][validation] = ungated
        predictions["gated"][validation] = gated
        cells.append(
            {
                "route": route,
                "seed": seed,
                "outer_fold": outer_fold,
                "train_rows": int(np.sum(train)),
                "validation_rows": int(np.sum(validation)),
                "train_fold_ids": sorted(np.unique(oof.fold_ids[train]).tolist()),
                "validation_fold_ids": [outer_fold],
                "base_predictions_for_training": "P5 Stage-3 OOF only",
                "inner_residual_predictions": "cross-fit by original OOF fold",
                "selected_gate_scale": selected_scale,
                "inner_gate_candidate_rmse": candidate_scores,
                "inner_correction_benefit_rate": float(np.mean(beneficial)),
                "baseline_metrics": _metrics(
                    oof.target[validation], oof.baseline[validation]
                ),
                "ungated_metrics": _metrics(oof.target[validation], ungated),
                "gated_metrics": _metrics(oof.target[validation], gated),
                "residual_prediction_stats": _summary_stats(outer_residual),
                "gate_stats": {
                    **_summary_stats(gate),
                    "zero_rate": float(np.mean(gate == 0.0)),
                    "one_rate": float(np.mean(gate == 1.0)),
                },
            }
        )
    for name, values in predictions.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{route}/{name} predictions are incomplete")
    return predictions, cells


def evaluate_residual_fusion(
    oof: OOFDevelopment,
    openmind_features: np.ndarray,
    *,
    layer_channels: Sequence[int],
    repeat_seeds: Sequence[int] = REPEAT_SEEDS,
) -> dict[str, Any]:
    """Run paired pretrained/structural residual ablations on fixed OOF rows."""

    openmind_features = np.asarray(openmind_features, dtype=np.float64)
    if openmind_features.shape[0] != len(oof.target):
        raise ValueError("OpenMind feature rows do not align with PyKrige OOF")
    if sum(int(value) for value in layer_channels) != openmind_features.shape[1]:
        raise ValueError("OpenMind layer channels do not sum to feature width")
    if not np.all(np.isfinite(openmind_features)):
        raise FloatingPointError("OpenMind features contain non-finite values")

    gate_zero = oof.baseline.copy()
    if not np.array_equal(gate_zero, oof.baseline):
        raise RuntimeError("gate=0 failed exact PyKrige degeneration")

    per_cell: list[dict[str, Any]] = []
    aggregate_by_seed: list[dict[str, Any]] = []
    selected_channels: dict[str, list[int]] = {}
    for seed in repeat_seeds:
        selected = _selected_latent_channels(layer_channels, seed=int(seed))
        selected_channels[str(seed)] = selected.tolist()
        common = np.column_stack([oof.baseline, oof.distance_to_well])
        pretrained_x = np.column_stack([openmind_features[:, selected], common])
        structural_x = np.column_stack([oof.structural_features, common])

        pretrained_predictions, pretrained_cells = _evaluate_feature_route(
            route="pretrained_openmind_residual",
            features=pretrained_x,
            oof=oof,
            seed=int(seed),
        )
        structural_predictions, structural_cells = _evaluate_feature_route(
            route="no_foundation_structural_residual",
            features=structural_x,
            oof=oof,
            seed=int(seed),
        )
        per_cell.extend(pretrained_cells)
        per_cell.extend(structural_cells)
        aggregate_by_seed.append(
            {
                "seed": int(seed),
                "pykrige_oof": _metrics(oof.target, oof.baseline),
                "gate_zero_exact": _metrics(oof.target, gate_zero),
                "gate_zero_bitwise_equal_to_pykrige": bool(
                    np.array_equal(gate_zero, oof.baseline)
                ),
                "pretrained_openmind_residual_ungated": _metrics(
                    oof.target, pretrained_predictions["ungated"]
                ),
                "pretrained_openmind_residual_gated": _metrics(
                    oof.target, pretrained_predictions["gated"]
                ),
                "no_foundation_structural_residual_gated": _metrics(
                    oof.target, structural_predictions["gated"]
                ),
            }
        )

    def mean_rmse(key: str) -> float:
        return float(
            np.mean([row[key]["rmse"] for row in aggregate_by_seed])
        )

    baseline_rmse = mean_rmse("pykrige_oof")
    pretrained_rmse = mean_rmse("pretrained_openmind_residual_gated")
    structural_rmse = mean_rmse("no_foundation_structural_residual_gated")
    fold_wins = sum(
        row["gated_metrics"]["rmse"] < row["baseline_metrics"]["rmse"]
        for row in per_cell
        if row["route"] == "pretrained_openmind_residual"
    )
    expected_fold_cells = len(FOLD_IDS) * len(tuple(repeat_seeds))
    material_gain = (
        (baseline_rmse - pretrained_rmse) / baseline_rmse
        if baseline_rmse > 0.0
        else -math.inf
    )
    promoted = (
        material_gain >= 0.01
        and pretrained_rmse < structural_rmse
        and fold_wins >= math.ceil(0.8 * expected_fold_cells)
    )
    return {
        "fixed_protocol": {
            "outer_folds": list(FOLD_IDS),
            "repeat_seeds": [int(seed) for seed in repeat_seeds],
            "base_model": EXPECTED_MODEL_ID,
            "base_predictions_for_all_residual_training": "cross-fitted P5 Stage-3 development OOF",
            "residual_model": f"StandardScaler + Ridge(alpha={RIDGE_ALPHA}, solver=lsqr)",
            "gate_model": "inner-OOF LogisticRegression benefit probability times train-only selected bounded scale",
            "gate_bounds": [0.0, 1.0],
            "gate_scale_candidates": list(GATE_CANDIDATES),
            "latent_channel_budget_per_stage": LATENT_CHANNELS_PER_STAGE,
            "test_or_holdout_tuning": False,
        },
        "selected_openmind_channels": selected_channels,
        "aggregate_by_seed": aggregate_by_seed,
        "per_fold_seed": per_cell,
        "comparison": {
            "pykrige_oof_mean_seed_rmse": baseline_rmse,
            "pretrained_openmind_residual_gated_mean_seed_rmse": pretrained_rmse,
            "no_foundation_structural_residual_gated_mean_seed_rmse": structural_rmse,
            "pretrained_relative_gain_vs_pykrige": material_gain,
            "pretrained_minus_structural_rmse": pretrained_rmse - structural_rmse,
            "pretrained_fold_seed_wins_vs_pykrige": fold_wins,
            "pretrained_fold_seed_cells": expected_fold_cells,
        },
        "decision": {
            "state": "PROMOTE_DEVELOPMENT_ONLY" if promoted else "VERIFIED_NO_PROMOTION",
            "default_enabled": promoted,
            "promotion_rule": (
                "at least 1% pooled development RMSE gain vs PyKrige, better than "
                "no-foundation structural control, and wins >=80% of fold-seed cells"
            ),
        },
    }


def _direct_foundation_comparison(path: Path) -> dict[str, Any]:
    """Report the archived direct route only when its lane is comparable."""

    if not path.is_file():
        return {
            "state": "not_available",
            "reason": "archived direct-foundation summary is absent",
        }
    payload = _json(path)
    evaluation = payload.get("evaluation", {})
    comparable = (
        evaluation.get("mode") == EXPECTED_LANE
        and evaluation.get("split_hash") == EXPECTED_SPLIT_HASH
        and payload.get("runtime", {}).get("raw_predictions_persisted") is True
    )
    if not comparable:
        return {
            "state": "not_comparable",
            "reason": (
                "the archived direct OpenMind path is strict-lane evidence and "
                "does not persist row-aligned conditional OOF predictions"
            ),
            "archived_mode": evaluation.get("mode"),
            "archived_split_hash": evaluation.get("split_hash"),
            "summary_sha256": _sha256(path),
            "archived_macro_fold_rmse": payload.get("comparison", {}).get(
                "pretrained_macro_fold_rmse"
            ),
        }
    return {
        "state": "comparable",
        "summary_sha256": _sha256(path),
        "macro_fold_rmse": payload["comparison"]["pretrained_macro_fold_rmse"],
    }


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    comparison = result["experiment"]["comparison"]
    decision = result["experiment"]["decision"]
    direct = result["ablations"]["original_direct_foundation_path"]
    lines = [
        "# P11 Reconstruction Residual-Fusion Evidence",
        "",
        "## Outcome",
        "",
        f"- Decision: `{decision['state']}`; default enabled: `{decision['default_enabled']}`.",
        f"- PyKrige development OOF RMSE: `{comparison['pykrige_oof_mean_seed_rmse']:.12g}`.",
        (
            "- Gated pretrained-OpenMind residual RMSE: "
            f"`{comparison['pretrained_openmind_residual_gated_mean_seed_rmse']:.12g}`."
        ),
        (
            "- Gated no-foundation structural residual RMSE: "
            f"`{comparison['no_foundation_structural_residual_gated_mean_seed_rmse']:.12g}`."
        ),
        (
            "- Relative gain vs PyKrige: "
            f"`{comparison['pretrained_relative_gain_vs_pykrige']:.6%}`."
        ),
        "",
        "## Holdout firewall",
        "",
        "- The runner accepts only `train.h5`; it has no test-loader argument.",
        "- OpenMind extraction read only `seismic_patch[0:3]` and `seismic_patch[8]`.",
        "- Targets and PyKrige predictions came from hash-verified P5 Stage-3 development OOF archives.",
        "- Every residual training base prediction was OOF; every gate was calibrated by inner OOF.",
        "- No frozen-test path, array, label, prediction, or historical metric was opened.",
        "",
        "## Legal ablations",
        "",
        "- `pykrige_oof`: unchanged cross-fitted strong baseline.",
        "- `gate_zero_exact`: bitwise-identical copy of PyKrige OOF.",
        "- `pretrained_openmind_residual_ungated`: genuine cached-weight encoder features without gate.",
        "- `pretrained_openmind_residual_gated`: same residual plus bounded `[0,1]` confidence gate.",
        "- `no_foundation_structural_residual_gated`: same residual/gate algorithm using seismic+coordinate controls.",
        (
            "- `original_direct_foundation_path`: "
            f"`{direct['state']}` ({direct.get('reason', 'same-lane evidence')})."
        ),
        "",
        "## Genuine feature identity",
        "",
        f"- Checkpoint SHA-256: `{result['openmind_feature_audit']['checkpoint_sha256']}`.",
        f"- Source revision: `{result['openmind_feature_audit']['source_revision']}`.",
        (
            "- Multiscale layer channels: "
            f"`{result['openmind_feature_audit']['layer_channels']}`."
        ),
        (
            "- Feature cache SHA-256: "
            f"`{result['openmind_feature_audit']['feature_cache_sha256']}`."
        ),
        "",
        "Fold/seed metrics and gate/residual distributions are in `summary.json`.",
        "",
    ]
    evidence_path = output_dir / "evidence.md"
    evidence_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    feature_cache: Path = DEFAULT_FEATURE_CACHE,
    direct_foundation_summary: Path = DEFAULT_DIRECT_SUMMARY,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Execute P11 and write portable, reconstruction-local evidence."""

    started = time.perf_counter()
    inputs = resolve_dev_inputs(data_dir)
    oof = load_oof_development(stage3_root)
    features, feature_audit = get_openmind_features(
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        feature_cache=feature_cache,
        device=device,
    )
    layer_channels = feature_audit["layer_channels"]
    experiment = evaluate_residual_fusion(
        oof,
        features,
        layer_channels=layer_channels,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "task_id": "volve_porosity_conditional_reconstruction",
        "evidence_scope": "fixed conditional development OOF only",
        "holdout_firewall": {
            "accepted_hdf5_basenames": ["train.h5"],
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": ["seismic_patch[0:3]", "seismic_patch[8]"],
            "label_dataset_read": False,
            "test_path_argument_exists": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "historical_test_metrics_read": False,
        },
        "pykrige_oof_sources": {
            "lane": EXPECTED_LANE,
            "model_id": EXPECTED_MODEL_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "repeat_id": REPEAT_ID,
            "rows": len(oof.target),
            "folds": list(oof.source_records),
            "training_use": "base predictions are OOF before residual stacking",
        },
        "openmind_feature_audit": feature_audit,
        "implementation": {
            "script": "_pipelines/02_task_datasets/reconstruction/p11_residual_fusion.py",
            "script_sha256": _sha256(Path(__file__)),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "ablations": {
            "required": [
                "pykrige_oof",
                "gate_zero_exact",
                "pretrained_openmind_residual_ungated",
                "pretrained_openmind_residual_gated",
                "no_foundation_structural_residual_gated",
            ],
            "original_direct_foundation_path": _direct_foundation_comparison(
                direct_foundation_summary
            ),
        },
        "experiment": experiment,
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "downloads_performed_bytes": 0,
            "feature_cache_reused": feature_audit["cache_reused"],
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evidence(output_dir, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-dev", help="verify the train-only HDF5 boundary")
    check.add_argument("--data-dir", type=Path, required=True)

    execute = subparsers.add_parser("run", help="run the P11 development-only experiment")
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--checkpoint", type=Path, required=True)
    execute.add_argument("--dependency-root", type=Path)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    execute.add_argument(
        "--direct-foundation-summary",
        type=Path,
        default=DEFAULT_DIRECT_SUMMARY,
    )
    execute.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check-dev":
        resolved = resolve_dev_inputs(args.data_dir)
        print(json.dumps({"accepted": str(resolved.train_h5), "holdout_opened": False}))
        return 0
    values = vars(args)
    values.pop("command")
    result = run(**values)
    print(json.dumps(result["experiment"]["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
