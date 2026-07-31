#!/usr/bin/env python3
"""P5 Stage-1 contract smoke runner for the fixed nine-class GM09 task.

The runner has three deliberately separate commands:

``prepare-batch`` reads only the existing development ``train.h5``, constructs
the honest LOGO4 split, fits preprocessing on one fold's training families,
and writes a small ignored NPZ envelope. ``smoke`` consumes that envelope in a
dependency-specific shared environment. ``merge`` combines disjoint smoke
reports. No command accepts or opens a frozen-test loader or path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.contracts import ModelBatch  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    NUM_CLASSES,
    OptionalDependencyUnavailable,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    TARGET_NAME,
    TEST_FAMILY,
    apply_fold_preprocessor,
    class_support,
    fit_fold_preprocessor,
    lithofacies_task_spec,
    sample_id,
    validate_p4_sample,
)


SOURCE_LOCK_PATH = TRACK_DIR / "p5_source_lock.json"
BATCH_SCHEMA = "lithofacies-p5-stage1-batch-v1"
RESULT_SCHEMA = "lithofacies-p5-stage1-result-v1"
FIRST_TEN = (
    "xgboost_multisoftprob_window",
    "catboost_multiclass_window",
    "minirocket_ridge_window",
    "inceptiontime_window",
    "tcn_center_head",
    "balanced_softmax_tcn",
    "moderntcn_window",
    "ms_tcn2_dense",
    "embracenet_missing_modal",
    "multibench_lowrank_tensor_fusion",
)


class LaneUnavailable(RuntimeError):
    """A scientifically required lane input is absent from the real development archive."""

    def __init__(self, lane: str, reason: str) -> None:
        self.lane = lane
        self.reason = reason
        super().__init__(f"{lane}-lane unavailable: {reason}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _track_owned(path: Path) -> Path:
    resolved = path.resolve()
    if TRACK_DIR.resolve() not in resolved.parents:
        raise ValueError(f"P5 runtime artifacts must stay below {TRACK_DIR}")
    return resolved


def load_source_lock() -> dict[str, Any]:
    payload = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    identifiers = tuple(model.get("model_id") for model in models)
    if payload.get("class_count") != NUM_CLASSES or identifiers != FIRST_TEN:
        raise RuntimeError("P5 source lock no longer matches the frozen first-ten GM09 roster")
    for model in models:
        required = {
            "model_id", "leaderboard_lane", "source_url", "revision", "license",
            "dependency_group", "required_imports", "implementation_mode",
            "pretrained_weights", "smoke_config",
        }
        missing = sorted(required - set(model))
        if missing:
            raise RuntimeError(f"source lock {model.get('model_id')} lacks {missing}")
        if model["pretrained_weights"].get("used"):
            raise RuntimeError("Stage-1 source lock forbids pretrained weights")
    return payload


def _read_development_hdf5(dataset_root: Path) -> tuple[list[dict[str, Any]], Path]:
    """Read the development archive only; importing h5py is prepare-only."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("prepare-batch requires an environment with h5py") from exc
    path = dataset_root.resolve() / "train.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            sample = {
                "seismic_patch": group["seismic_patch"][()],
                "well_log_seq": group["well_log_seq"][()],
                "label": group["label"][()],
                "position": json.loads(group.attrs["position"]),
                "meta": json.loads(group.attrs["meta"]),
            }
            validate_p4_sample(sample)
            samples.append(sample)
    if not samples:
        raise ValueError("development train.h5 is empty")
    families = {str(sample["meta"]["family_id"]) for sample in samples}
    expected = set(DEVELOPMENT_FAMILIES)
    if families != expected:
        raise ValueError(
            f"Stage-1 requires exactly four development mother families; got {sorted(families)}"
        )
    if TEST_FAMILY in families:
        raise RuntimeError("F-5 firewall violation in development archive")
    return samples, path


def build_development_logo4(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return four deterministic leave-one-mother-family-out folds."""
    by_family: dict[str, list[Mapping[str, Any]]] = {family: [] for family in DEVELOPMENT_FAMILIES}
    for sample in samples:
        family = str(sample.get("meta", {}).get("family_id", ""))
        if family not in by_family or family == TEST_FAMILY:
            raise ValueError(f"non-development family in LOGO4 input: {family!r}")
        by_family[family].append(sample)
    missing = [family for family, values in by_family.items() if not values]
    if missing:
        raise ValueError(f"LOGO4 lacks development families: {missing}")
    folds: list[dict[str, Any]] = []
    all_ids = {sample_id(sample) for sample in samples}
    for fold_id, validation_family in enumerate(DEVELOPMENT_FAMILIES):
        validation = list(by_family[validation_family])
        train = [
            sample
            for family in DEVELOPMENT_FAMILIES
            if family != validation_family
            for sample in by_family[family]
        ]
        train_ids = {sample_id(sample) for sample in train}
        validation_ids = {sample_id(sample) for sample in validation}
        if train_ids & validation_ids or train_ids | validation_ids != all_ids:
            raise RuntimeError("LOGO4 sample partition is not a disjoint cover")
        folds.append(
            {
                "fold_id": fold_id,
                "train_groups": [family for family in DEVELOPMENT_FAMILIES if family != validation_family],
                "validation_groups": [validation_family],
                "train": train,
                "validation": validation,
                "train_class_support": class_support(train).tolist(),
                "validation_class_support": class_support(validation).tolist(),
            }
        )
    if len(folds) != EFFECTIVE_N_SPLITS:
        raise RuntimeError("GM09 development must use effective LOGO4")
    return folds


def _balanced_take(samples: Sequence[Mapping[str, Any]], maximum: int) -> list[Mapping[str, Any]]:
    buckets: dict[int, list[Mapping[str, Any]]] = {class_id: [] for class_id in range(NUM_CLASSES)}
    for sample in sorted(samples, key=sample_id):
        buckets[int(sample["label"])].append(sample)
    chosen: list[Mapping[str, Any]] = []
    cursor = 0
    while len(chosen) < min(maximum, len(samples)):
        progress = False
        for class_id in range(NUM_CLASSES):
            if cursor < len(buckets[class_id]) and len(chosen) < maximum:
                chosen.append(buckets[class_id][cursor])
                progress = True
        if not progress:
            break
        cursor += 1
    return chosen


def _p_arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([np.asarray(sample["well_log_seq"], dtype=np.float32) for sample in samples]),
        np.stack([np.asarray(sample["seismic_patch"], dtype=np.float32) for sample in samples]),
        np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64),
        np.asarray([sample_id(sample) for sample in samples], dtype=np.str_),
    )


def _sequence_arrays(
    samples: Sequence[Mapping[str, Any]], maximum: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    by_well: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        md = sample.get("position", {}).get("center_md_m")
        if md is None or not math.isfinite(float(md)):
            continue
        well = str(sample.get("position", {}).get("well_name", ""))
        if well:
            by_well.setdefault(well, []).append(sample)
    eligible = {well: values for well, values in by_well.items() if len(values) >= 2}
    if not eligible:
        raise ValueError("S-lane smoke requires at least two real MD-ordered centers in one well")
    well = sorted(eligible, key=lambda key: (-len(eligible[key]), key))[0]
    ordered = sorted(eligible[well], key=lambda sample: (float(sample["position"]["center_md_m"]), sample_id(sample)))
    ordered = ordered[:maximum]
    log_centers = []
    seismic_centers = []
    for sample in ordered:
        logs = np.asarray(sample["well_log_seq"], dtype=np.float32)
        seismic = np.asarray(sample["seismic_patch"], dtype=np.float32)
        log_centers.append(logs[:, logs.shape[-1] // 2])
        seismic_centers.append(seismic[:, :, seismic.shape[-1] // 2])
    well_array = np.stack(log_centers, axis=-1)[None, ...]
    seismic_array = np.stack(seismic_centers, axis=-1)[None, ...]
    labels = np.asarray([[int(sample["label"]) for sample in ordered]], dtype=np.int64)
    sample_ids = np.asarray([sample_id(sample) for sample in ordered], dtype=np.str_)
    md = np.asarray([[float(sample["position"]["center_md_m"]) for sample in ordered]], dtype=np.float64)
    family = str(ordered[0]["meta"]["family_id"])
    if any(str(sample["meta"]["family_id"]) != family for sample in ordered):
        raise RuntimeError("S-lane sequence crossed mother families")
    return well_array, seismic_array, labels, sample_ids, md, well, family


def prepare_batch(
    dataset_root: Path, batch_file: Path, *, max_train: int, max_validation: int,
    sequence_length: int,
) -> dict[str, Any]:
    batch_file = _track_owned(batch_file)
    samples, hdf5_path = _read_development_hdf5(dataset_root)
    folds = build_development_logo4(samples)
    fold = folds[0]
    preprocessor = fit_fold_preprocessor(fold["train"])
    train_all = apply_fold_preprocessor(fold["train"], preprocessor)
    validation_all = apply_fold_preprocessor(fold["validation"], preprocessor)
    train_p = _balanced_take(train_all, max_train)
    validation_p = _balanced_take(validation_all, max_validation)
    p_train = _p_arrays(train_p)
    p_validation = _p_arrays(validation_p)
    try:
        s_train = _sequence_arrays(train_all, sequence_length)
        s_validation = _sequence_arrays(validation_all, sequence_length)
        if s_train[-1] in set(fold["validation_groups"]) or s_validation[-1] in set(fold["train_groups"]):
            raise RuntimeError("S-lane sequence violated the LOGO fold")
        s_lane = {
            "status": "available",
            "train": {"well": s_train[-2], "family": s_train[-1], "positions": int(s_train[2].shape[-1])},
            "validation": {
                "well": s_validation[-2], "family": s_validation[-1],
                "positions": int(s_validation[2].shape[-1]),
            },
        }
        s_reason = None
    except ValueError as exc:
        s_train = None
        s_validation = None
        s_reason = str(exc)
        s_lane = {
            "status": "not_feasible",
            "reason": s_reason,
            "policy": "do not fabricate MD order or repeat center labels across a synthetic sequence",
        }
    manifest = {
        "schema_version": BATCH_SCHEMA,
        "task_id": "gm09_genetic_facies_9class",
        "class_names": CLASS_NAMES,
        "class_count": NUM_CLASSES,
        "requested_n_splits": 5,
        "effective_n_splits": EFFECTIVE_N_SPLITS,
        "splitter": "leave_one_mother_family_out",
        "stage1_fold_id": int(fold["fold_id"]),
        "stage1_train_groups": fold["train_groups"],
        "stage1_validation_groups": fold["validation_groups"],
        "all_folds": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"train", "validation"}
            }
            for candidate in folds
        ],
        "fold_train_class_support": preprocessor.class_support,
        "p_train_samples": len(train_p),
        "p_validation_samples": len(validation_p),
        "s_lane": s_lane,
        "loaded_files": [hdf5_path.name],
        "development_hdf5_sha256": _sha256(hdf5_path),
        "frozen_test_family": TEST_FAMILY,
        "frozen_test_accessed": False,
        "preprocess_fit_scope": "stage1_fold_train_mother_families_only",
        "missing_mask_rows": [13, 26],
        "seismic_shape": [3, 3, 33],
    }
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "manifest": np.asarray(json.dumps(manifest, ensure_ascii=False)),
        "class_counts": np.asarray(preprocessor.class_support, dtype=np.int64),
        "class_weights": np.asarray(preprocessor.class_weights, dtype=np.float32),
        "p_train_well": p_train[0], "p_train_seismic": p_train[1],
        "p_train_labels": p_train[2], "p_train_ids": p_train[3],
        "p_validation_well": p_validation[0], "p_validation_seismic": p_validation[1],
        "p_validation_labels": p_validation[2], "p_validation_ids": p_validation[3],
    }
    if s_train is not None and s_validation is not None:
        arrays.update(
            {
                "s_train_well": s_train[0], "s_train_seismic": s_train[1],
                "s_train_labels": s_train[2], "s_train_ids": s_train[3], "s_train_md": s_train[4],
                "s_validation_well": s_validation[0], "s_validation_seismic": s_validation[1],
                "s_validation_labels": s_validation[2], "s_validation_ids": s_validation[3],
                "s_validation_md": s_validation[4],
            }
        )
    else:
        arrays["s_unavailable_reason"] = np.asarray(s_reason or "real MD order is unavailable")
    np.savez_compressed(batch_file, **arrays)
    return {**manifest, "batch_file": str(batch_file.relative_to(PROJECT_ROOT)), "batch_sha256": _sha256(batch_file)}


def load_batch(batch_file: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(batch_file, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
        manifest = json.loads(str(archive["manifest"].item()))
    if manifest.get("schema_version") != BATCH_SCHEMA:
        raise ValueError("unknown Stage-1 batch schema")
    if tuple(manifest.get("class_names", ())) != CLASS_NAMES or manifest.get("class_count") != NUM_CLASSES:
        raise ValueError("Stage-1 batch changed the fixed GM09 class schema")
    if manifest.get("effective_n_splits") != EFFECTIVE_N_SPLITS:
        raise ValueError("Stage-1 batch is not LOGO4")
    if manifest.get("frozen_test_accessed") is not False:
        raise RuntimeError("Stage-1 batch reports frozen-test access")
    train_groups = set(manifest["stage1_train_groups"])
    validation_groups = set(manifest["stage1_validation_groups"])
    if train_groups & validation_groups or TEST_FAMILY in train_groups | validation_groups:
        raise RuntimeError("Stage-1 batch violates the F-5/mother-family firewall")
    names = ["p_train_well", "p_validation_well"]
    if manifest.get("s_lane", {}).get("status") == "available":
        names.extend(("s_train_well", "s_validation_well"))
    for name in names:
        values = arrays[name]
        if values.shape[1] != 26 or not np.isfinite(values).all():
            raise ValueError(f"{name} does not preserve finite 13-value + 13-mask channels")
        masks = values[:, 13:, :]
        if not np.isin(masks, (0.0, 1.0)).all():
            raise ValueError(f"{name} missing-mask rows are not binary")
    return arrays, manifest


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "xgboost", "catboost", "sktime", "scikit-learn", "tsai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        result.update(
            {
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        result.update({"torch_cuda": None, "cuda_available": False, "cuda_device": None})
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _lane_arrays(arrays: Mapping[str, np.ndarray], lane: str, split: str) -> tuple[np.ndarray, ...]:
    prefix = "p" if lane == "P" else "s"
    if f"{prefix}_{split}_well" not in arrays:
        reason_value = arrays.get("s_unavailable_reason", np.asarray("required real sequence is unavailable"))
        raise LaneUnavailable(lane, str(np.asarray(reason_value).item()))
    return (
        arrays[f"{prefix}_{split}_well"],
        arrays[f"{prefix}_{split}_seismic"],
        arrays[f"{prefix}_{split}_labels"],
    )


def _build_config(lock: Mapping[str, Any], well: np.ndarray, seismic: np.ndarray) -> dict[str, Any]:
    return {
        **dict(lock["smoke_config"]),
        "num_classes": NUM_CLASSES,
        "well_log_shape": tuple(int(value) for value in well.shape[1:]),
        "seismic_shape": tuple(int(value) for value in seismic.shape[1:]),
    }


def _validate_logits(values: np.ndarray, labels: np.ndarray, lane: str) -> None:
    expected = (labels.shape[0], NUM_CLASSES) if lane == "P" else (
        labels.shape[0], NUM_CLASSES, labels.shape[-1]
    )
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"{lane}-lane logits must be finite {expected}, got {values.shape}")


def _softmax_numpy(logits: np.ndarray, lane: str) -> np.ndarray:
    axis = 1
    shifted = logits - logits.max(axis=axis, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=axis, keepdims=True)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=axis), 1.0, atol=1e-5):
        raise ValueError(f"{lane}-lane softmax is invalid")
    return probability


def _torch_smoke(
    discovered: Any, lock: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *,
    device_name: str, checkpoint_path: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    lane = str(lock["leaderboard_lane"])
    train_well, train_seismic, train_labels = _lane_arrays(arrays, lane, "train")
    validation_well, validation_seismic, validation_labels = _lane_arrays(arrays, lane, "validation")
    config = _build_config(lock, train_well, train_seismic)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device_name} but CUDA is unavailable")
    cuda_index: int | None = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        # PyTorch 2.13's memory counter rejects an uninitialized CUDA device.
        torch.cuda.set_device(cuda_index)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    model = discovered.build(lithofacies_task_spec(), **config).to(device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("torch backend did not return torch.nn.Module")
    well_tensor = torch.as_tensor(train_well, dtype=torch.float32, device=device)
    seismic_tensor = torch.as_tensor(train_seismic, dtype=torch.float32, device=device)
    labels_tensor = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    class_counts = torch.as_tensor(arrays["class_counts"], dtype=torch.float32, device=device)
    class_weights = torch.as_tensor(arrays["class_weights"], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(well_tensor, seismic_tensor)
    module_loss = getattr(discovered.module, "stage1_loss", None)
    if module_loss is None:
        loss = functional.cross_entropy(logits, labels_tensor, weight=class_weights)
    else:
        loss = module_loss(
            logits, labels_tensor, class_counts=class_counts, class_weights=class_weights
        )
    if not bool(torch.isfinite(loss)):
        raise ValueError("Stage-1 loss is not finite")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("Stage-1 backward did not produce finite gradients")
    optimizer.step()

    validation_well_tensor = torch.as_tensor(validation_well, dtype=torch.float32, device=device)
    validation_seismic_tensor = torch.as_tensor(validation_seismic, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model(validation_well_tensor, validation_seismic_tensor)
        repeated = model(validation_well_tensor, validation_seismic_tensor)
        changed_seismic = model(validation_well_tensor, validation_seismic_tensor + 0.125)
        changed_masks = validation_well_tensor.clone()
        changed_masks[:, 13:, :] = 1.0 - changed_masks[:, 13:, :]
        changed_mask_prediction = model(changed_masks, validation_seismic_tensor)
    prediction_array = prediction.detach().cpu().numpy()
    _validate_logits(prediction_array, validation_labels, lane)
    _softmax_numpy(prediction_array, lane)
    deterministic_error = float(torch.max(torch.abs(prediction - repeated)).detach().cpu())
    if deterministic_error != 0.0:
        raise RuntimeError(f"evaluation forward is not deterministic: max_abs={deterministic_error}")
    seismic_sensitivity = float(torch.max(torch.abs(prediction - changed_seismic)).detach().cpu())
    mask_sensitivity = float(torch.max(torch.abs(prediction - changed_mask_prediction)).detach().cpu())
    if seismic_sensitivity <= 0.0 or mask_sensitivity <= 0.0:
        raise RuntimeError("adapter did not consume both seismic and well missing-mask inputs")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": RESULT_SCHEMA,
            "model_id": lock["model_id"],
            "source_revision": lock["revision"],
            "class_names": CLASS_NAMES,
            "leaderboard_lane": lane,
            "config": config,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    reloaded_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded = discovered.build(lithofacies_task_spec(), **config).to(device)
    reloaded.load_state_dict(reloaded_payload["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_prediction = reloaded(validation_well_tensor, validation_seismic_tensor)
    roundtrip_error = float(torch.max(torch.abs(prediction - reloaded_prediction)).detach().cpu())
    if roundtrip_error != 0.0:
        raise RuntimeError(f"checkpoint round-trip changed logits: max_abs={roundtrip_error}")
    if device.type == "cuda":
        assert cuda_index is not None
        torch.cuda.synchronize(cuda_index)
        peak_vram = int(torch.cuda.max_memory_allocated(cuda_index))
    else:
        peak_vram = 0
    return {
        "fit": "one_adamw_step",
        "forward": "PASS",
        "loss": float(loss.detach().cpu()),
        "backward": "PASS",
        "checkpoint_roundtrip": "PASS",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_roundtrip_max_abs": roundtrip_error,
        "determinism_max_abs": deterministic_error,
        "seismic_sensitivity_max_abs": seismic_sensitivity,
        "missing_mask_sensitivity_max_abs": mask_sensitivity,
        "validation_output_shape": list(prediction_array.shape),
        "peak_vram_bytes": peak_vram,
        "device": str(device),
    }


def _estimator_smoke(
    discovered: Any, lock: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    lane = str(lock["leaderboard_lane"])
    if lane != "P":
        raise ValueError("current estimator Stage-1 contract supports the P lane only")
    train_well, train_seismic, train_labels = _lane_arrays(arrays, lane, "train")
    validation_well, validation_seismic, validation_labels = _lane_arrays(arrays, lane, "validation")
    config = _build_config(lock, train_well, train_seismic)
    model = discovered.build(lithofacies_task_spec(), **config)
    loss = float(
        model.fit_stage1(
            train_well, train_seismic, train_labels, class_counts=arrays["class_counts"]
        )
    )
    if not math.isfinite(loss):
        raise ValueError("estimator Stage-1 loss is not finite")
    prediction = np.asarray(model.predict_logits(validation_well, validation_seismic))
    _validate_logits(prediction, validation_labels, lane)
    _softmax_numpy(prediction, lane)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": RESULT_SCHEMA,
                "model_id": lock["model_id"],
                "source_revision": lock["revision"],
                "class_names": CLASS_NAMES,
                "leaderboard_lane": lane,
                "config": config,
                "model": model,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with checkpoint_path.open("rb") as handle:
        reloaded = pickle.load(handle)
    reloaded_prediction = np.asarray(
        reloaded["model"].predict_logits(validation_well, validation_seismic)
    )
    roundtrip_error = float(np.max(np.abs(prediction - reloaded_prediction)))
    if roundtrip_error != 0.0:
        raise RuntimeError(f"estimator checkpoint changed logits: max_abs={roundtrip_error}")
    return {
        "fit": "fixed_small_estimator_fit",
        "forward": "PASS",
        "loss": loss,
        "backward": "NOT_APPLICABLE_NON_GRADIENT_ESTIMATOR",
        "checkpoint_roundtrip": "PASS",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_roundtrip_max_abs": roundtrip_error,
        "determinism_max_abs": roundtrip_error,
        "multimodal_feature_count": int(np.prod(train_well.shape[1:]) + np.prod(train_seismic.shape[1:])),
        "missing_mask_rows_preserved": [13, 26],
        "validation_output_shape": list(prediction.shape),
        "peak_vram_bytes": 0,
        "device": "cpu",
    }


def run_smoke(
    batch_file: Path, output: Path, model_ids: Sequence[str], *, device: str,
) -> tuple[dict[str, Any], int]:
    output = _track_owned(output)
    arrays, batch_manifest = load_batch(batch_file)
    source_lock = load_source_lock()
    by_id = {model["model_id"]: model for model in source_lock["models"]}
    invalid = sorted(set(model_ids) - set(by_id))
    if invalid:
        raise ValueError(f"models are outside the frozen first ten: {invalid}")
    results: list[dict[str, Any]] = []
    checkpoint_dir = output.parent / "checkpoints"
    for model_id in model_ids:
        lock = by_id[model_id]
        _seed_everything(int(source_lock["root_seed"]))
        started = time.monotonic()
        before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        base = {
            "model_id": model_id,
            "leaderboard_lane": lock["leaderboard_lane"],
            "dependency_group": lock["dependency_group"],
            "source_url": lock["source_url"],
            "source_revision": lock["revision"],
            "license": lock["license"],
            "pretrained_weights_used": False,
            "formal_metric": False,
            "frozen_test_accessed": False,
        }
        try:
            discovered = discover_model("lithofacies", model_id)
            if discovered.capabilities.get("leaderboard_lane") != lock["leaderboard_lane"]:
                raise RuntimeError("adapter capability lane disagrees with source lock")
            backend = str(discovered.capabilities.get("backend"))
            suffix = ".pt" if backend == "torch" else ".pkl"
            checkpoint = checkpoint_dir / f"{model_id}{suffix}"
            if backend == "torch":
                evidence = _torch_smoke(
                    discovered, lock, arrays, device_name=device, checkpoint_path=checkpoint
                )
            elif backend == "estimator":
                evidence = _estimator_smoke(discovered, lock, arrays, checkpoint_path=checkpoint)
            else:
                raise ValueError(f"unknown adapter backend {backend!r}")
            status = "PASS"
            reason = None
        except OptionalDependencyUnavailable as exc:
            evidence = {}
            status = "SKIP"
            reason = {
                "code": "missing_optional_dependency",
                "dependency": exc.dependency,
                "message": str(exc),
            }
        except LaneUnavailable as exc:
            evidence = {}
            status = "SKIP"
            reason = {
                "code": "lane_not_feasible",
                "lane": exc.lane,
                "message": exc.reason,
            }
        except Exception as exc:  # fail-loud evidence; caller receives non-zero status
            evidence = {}
            status = "FAIL"
            reason = {
                "code": "stage1_contract_failure",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-12:],
            }
        after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results.append(
            {
                **base,
                "status": status,
                "reason": reason,
                "wall_seconds": time.monotonic() - started,
                "process_peak_rss_kib": int(max(before_rss, after_rss)),
                **evidence,
            }
        )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "lithofacies",
        "task_id": "gm09_genetic_facies_9class",
        "class_names": CLASS_NAMES,
        "source_lock_sha256": _sha256(SOURCE_LOCK_PATH),
        "batch_sha256": _sha256(batch_file),
        "batch_manifest": batch_manifest,
        "environment": _environment(),
        "models": results,
        "frozen_test_accessed": False,
    }
    _write_json(output, payload)
    return payload, int(any(result["status"] == "FAIL" for result in results))


def merge_results(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    output = _track_owned(output)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not payloads:
        raise ValueError("merge requires at least one Stage-1 result")
    source_hashes = {payload["source_lock_sha256"] for payload in payloads}
    batch_hashes = {payload["batch_sha256"] for payload in payloads}
    if len(source_hashes) != 1 or len(batch_hashes) != 1:
        raise ValueError("cannot merge results from different source locks or real batches")
    by_id: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if payload.get("frozen_test_accessed") is not False:
            raise RuntimeError("cannot merge a result that accessed frozen test")
        for result in payload["models"]:
            model_id = result["model_id"]
            if model_id in by_id:
                raise ValueError(f"duplicate Stage-1 result for {model_id}")
            by_id[model_id] = result
    missing = [model_id for model_id in FIRST_TEN if model_id not in by_id]
    unexpected = sorted(set(by_id) - set(FIRST_TEN))
    if missing or unexpected:
        raise ValueError(f"merged Stage-1 roster mismatch: missing={missing}, unexpected={unexpected}")
    ordered = [by_id[model_id] for model_id in FIRST_TEN if model_id in by_id]
    counts = {status: sum(result["status"] == status for result in ordered) for status in ("PASS", "SKIP", "FAIL")}
    merged = {
        "schema_version": RESULT_SCHEMA,
        "track_id": "lithofacies",
        "task_id": "gm09_genetic_facies_9class",
        "class_names": CLASS_NAMES,
        "source_lock_sha256": next(iter(source_hashes)),
        "batch_sha256": next(iter(batch_hashes)),
        "candidate_count": len(ordered),
        "status_counts": counts,
        "leaderboards": {
            "P": [result["model_id"] for result in ordered if result["leaderboard_lane"] == "P"],
            "S": [result["model_id"] for result in ordered if result["leaderboard_lane"] == "S"],
        },
        "environments": [payload["environment"] for payload in payloads],
        "models": ordered,
        "frozen_test_accessed": False,
        "formal_ranking": False,
    }
    _write_json(output, merged)
    return merged


def _parse_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return FIRST_TEN
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    if not models:
        raise argparse.ArgumentTypeError("--models must contain at least one model id")
    return models


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-batch", help="read development train.h5 only")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--batch-file", type=Path, required=True)
    prepare.add_argument("--max-train", type=int, default=64)
    prepare.add_argument("--max-validation", type=int, default=16)
    prepare.add_argument("--sequence-length", type=int, default=16)
    smoke = subparsers.add_parser("smoke", help="run fixed Stage-1 models from a prepared batch")
    smoke.add_argument("--batch-file", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--models", type=_parse_models, default=FIRST_TEN)
    smoke.add_argument("--device", default="cpu")
    merge = subparsers.add_parser("merge", help="merge disjoint dependency-group reports")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare-batch":
        if min(args.max_train, args.max_validation, args.sequence_length) < 2:
            raise ValueError("Stage-1 batch limits must all be >=2")
        result = prepare_batch(
            args.dataset_root,
            args.batch_file,
            max_train=args.max_train,
            max_validation=args.max_validation,
            sequence_length=args.sequence_length,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "smoke":
        payload, exit_code = run_smoke(
            args.batch_file, args.output, args.models, device=args.device
        )
        print(json.dumps({"models": payload["models"]}, ensure_ascii=False, indent=2))
        return exit_code
    merged = merge_results(args.inputs, args.output)
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    return int(merged["status_counts"]["FAIL"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
