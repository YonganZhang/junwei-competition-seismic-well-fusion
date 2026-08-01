#!/usr/bin/env python3
"""Fixed-budget P5 Stage-2 development pilot for GM09 lithofacies.

This track-private runner consumes the Stage-1 development-only NPZ envelope.
It has no frozen-test argument or loader.  Nine P-lane candidates are evaluated
on the same first LOGO4 fold and one S-lane candidate is recorded as a
structured skip when real, ordered measured-depth sequences are unavailable.
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

from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    NUM_CLASSES,
    OptionalDependencyUnavailable,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    TEST_FAMILY,
    classification_metrics_from_logits,
    lithofacies_task_spec,
)
from p5_stage1 import (  # noqa: E402
    FIRST_TEN,
    LaneUnavailable,
    _build_config,
    _lane_arrays,
    _sha256,
    _softmax_numpy,
    _validate_logits,
    load_batch,
    load_source_lock,
    prepare_batch,
)


ROOT_SEED = 2693
TASK_ID = "gm09_genetic_facies_9class"
STAGE2_SCHEMA = "lithofacies-p5-stage2-cell-v1"
PARTIAL_SCHEMA = "lithofacies-p5-stage2-partial-v1"
SUMMARY_SCHEMA = "lithofacies-p5-stage2-summary-v1"
LEADERBOARD_SCHEMA = "lithofacies-p5-stage2-p-leaderboard-v1"
FIXED_FOLD_ID = 0
FIXED_VALIDATION_FAMILY = DEVELOPMENT_FAMILIES[FIXED_FOLD_ID]
P_CONTEXT_LENGTH = 33
P_TRAIN_SAMPLE_LIMIT = 320
P_VALIDATION_SAMPLE_LIMIT = 160
P_BATCH_SIZE = 32
NEURAL_PARAMETER_UPDATE_LIMIT = 40
TINY_GATE_UPDATES = 3
NEURAL_WALL_LIMIT_SECONDS = 600.0
ESTIMATOR_WALL_LIMIT_SECONDS = 300.0
GPU_LOCK_PATH = Path(
    os.environ.get(
        "VOLVE_P5_GPU_LOCK",
        str(Path.home() / ".cache" / "volve-p5" / "locks" / "gpu0.lock"),
    )
)
CANONICAL_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_stage2"
RESULTS_FILENAME = "p5_stage2_results.jsonl"
SUMMARY_FILENAME = "p5_stage2_summary.json"
P_LEADERBOARD_FILENAME = "p5_stage2_p_leaderboard.json"


def _stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def derive_cell_seed(model_id: str, component: str) -> int:
    """Derive a stable positive seed without Python's randomized ``hash``."""
    material = f"{ROOT_SEED}|lithofacies|stage2|{model_id}|{component}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value % (2**31 - 1) or ROOT_SEED


def _track_owned(path: Path) -> Path:
    resolved = path.resolve()
    if TRACK_DIR.resolve() not in resolved.parents:
        raise ValueError(f"Stage-2 artifacts must stay below {TRACK_DIR}")
    return resolved


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _package_versions() -> dict[str, str | None]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "torch", "xgboost", "catboost", "sktime", "scikit-learn", "tsai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def _portable_environment(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": _package_versions(),
        "device_type": device.split(":", 1)[0],
    }


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


def _verify_external_gpu_lock(device: str) -> None:
    """Fail closed unless a parent ``flock`` owns the frozen shared GPU lock."""
    if not device.startswith("cuda"):
        return
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - the benchmark host is Linux
        raise RuntimeError("CUDA pilot requires POSIX flock verification") from exc
    GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK_PATH.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    raise RuntimeError(
        "CUDA pilot must be launched under an exclusive flock on the frozen gpu0 lock"
    )


def prepare_stage2_batch(dataset_root: Path, batch_file: Path) -> dict[str, Any]:
    """Prepare one fixed fold from the development archive through Stage-1 code."""
    batch_file = _track_owned(batch_file)
    manifest = prepare_batch(
        dataset_root,
        batch_file,
        max_train=P_TRAIN_SAMPLE_LIMIT,
        max_validation=P_VALIDATION_SAMPLE_LIMIT,
        sequence_length=P_CONTEXT_LENGTH,
    )
    arrays, stored = load_batch(batch_file)
    contract = batch_contract(arrays, stored, batch_file)
    return {
        "schema_version": "lithofacies-p5-stage2-batch-v1",
        "task_id": TASK_ID,
        "batch_file": str(batch_file.relative_to(PROJECT_ROOT)),
        "loaded_files": manifest["loaded_files"],
        **contract,
    }


def batch_contract(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any], batch_file: Path
) -> dict[str, Any]:
    """Validate and hash the immutable fold/input envelope used by every cell."""
    if int(manifest.get("stage1_fold_id", -1)) != FIXED_FOLD_ID:
        raise ValueError("Stage-2 must use the first valid development fold")
    train_groups = tuple(manifest.get("stage1_train_groups", ()))
    validation_groups = tuple(manifest.get("stage1_validation_groups", ()))
    expected_train = tuple(
        family for family in DEVELOPMENT_FAMILIES if family != FIXED_VALIDATION_FAMILY
    )
    if train_groups != expected_train or validation_groups != (FIXED_VALIDATION_FAMILY,):
        raise ValueError("Stage-2 fold no longer matches the frozen LOGO4 family assignment")
    if TEST_FAMILY in set(train_groups) | set(validation_groups):
        raise RuntimeError("frozen family entered the Stage-2 development fold")
    if manifest.get("frozen_test_accessed") is not False:
        raise RuntimeError("Stage-2 input envelope reports frozen-test access")
    train_well, train_seismic, train_labels = _lane_arrays(arrays, "P", "train")
    validation_well, validation_seismic, validation_labels = _lane_arrays(
        arrays, "P", "validation"
    )
    expected_well_shape = (26, P_CONTEXT_LENGTH)
    expected_seismic_shape = (3, 3, P_CONTEXT_LENGTH)
    if tuple(train_well.shape[1:]) != expected_well_shape or tuple(validation_well.shape[1:]) != expected_well_shape:
        raise ValueError("P lane well-log context budget changed")
    if tuple(train_seismic.shape[1:]) != expected_seismic_shape or tuple(validation_seismic.shape[1:]) != expected_seismic_shape:
        raise ValueError("P lane seismic context budget changed")
    if len(train_labels) > P_TRAIN_SAMPLE_LIMIT or len(validation_labels) > P_VALIDATION_SAMPLE_LIMIT:
        raise ValueError("P lane sample budget exceeded")
    split_payload = {
        "fold_id": FIXED_FOLD_ID,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "train_sample_ids": [str(value) for value in arrays["p_train_ids"].tolist()],
        "validation_sample_ids": [str(value) for value in arrays["p_validation_ids"].tolist()],
    }
    if set(split_payload["train_sample_ids"]) & set(split_payload["validation_sample_ids"]):
        raise RuntimeError("Stage-2 train/validation sample IDs overlap")
    return {
        "batch_sha256": _sha256(batch_file),
        "split_hash": _stable_hash(split_payload),
        "fold_id": FIXED_FOLD_ID,
        "train_groups": list(train_groups),
        "validation_groups": list(validation_groups),
        "fold_train_class_support": [int(value) for value in arrays["class_counts"].tolist()],
        "p_train_samples": int(len(train_labels)),
        "p_validation_samples": int(len(validation_labels)),
        "s_lane": dict(manifest.get("s_lane", {})),
        "frozen_test_accessed": False,
    }


def _input_budget(contract: Mapping[str, Any], lane: str) -> dict[str, Any]:
    p_budget = {
        "sample_selection": "same_fixed_fold_all_available_up_to_limit",
        "fold_train_sample_limit": P_TRAIN_SAMPLE_LIMIT,
        "fold_validation_sample_limit": P_VALIDATION_SAMPLE_LIMIT,
        "fold_train_samples_used": int(contract["p_train_samples"]),
        "fold_validation_samples_used": int(contract["p_validation_samples"]),
        "well_value_channels": 13,
        "well_missing_mask_channels": 13,
        "seismic_spatial_shape": [3, 3],
        "context_positions": P_CONTEXT_LENGTH,
        "pretrained_weights": False,
    }
    if lane == "P":
        return {**p_budget, "task_layout": "center_window_to_Bx9"}
    return {
        "task_layout": "real_MD_ordered_sequence_to_Bx9xL",
        "fold_train_sequences_used": 0,
        "fold_validation_sequences_used": 0,
        "well_value_channels": 13,
        "well_missing_mask_channels": 13,
        "seismic_spatial_shape": [3, 3],
        "pretrained_weights": False,
        "availability": contract.get("s_lane", {}),
    }


def _stage2_model_config(
    lock: Mapping[str, Any], well: np.ndarray, seismic: np.ndarray, model_seed: int
) -> dict[str, Any]:
    config = _build_config(lock, well, seismic)
    model_id = str(lock["model_id"])
    if model_id == "xgboost_multisoftprob_window":
        # P5 Stage-2/3 is archived evidence, not the current default-baseline
        # entrypoint. Keep its historical eta explicit after the adapter default
        # moves to the independently validated P17 configuration.
        config.update(
            {
                "rounds": NEURAL_PARAMETER_UPDATE_LIMIT,
                "eta": 0.2,
                "seed": model_seed,
            }
        )
    elif model_id == "catboost_multiclass_window":
        config.update({"iterations": NEURAL_PARAMETER_UPDATE_LIMIT, "seed": model_seed})
    elif model_id == "minirocket_ridge_window":
        config.update({"seed": model_seed})
    return config


def _metric_payload(labels: np.ndarray, logits: np.ndarray, family: str) -> dict[str, Any]:
    metrics = classification_metrics_from_logits(labels.tolist(), logits)
    primary = float(metrics["fixed_schema_macro_f1"])
    supported_diagnostic = float(metrics["supported_class_macro_f1"])
    metrics["worst_family_fixed_schema_macro_f1"] = primary
    metrics["validation_family_metrics"] = {
        family: {
            "sample_count": int(len(labels)),
            "fixed_schema_macro_f1": primary,
            "supported_class_macro_f1": supported_diagnostic,
        }
    }
    for key in (
        "supported_class_macro_f1",
        "fixed_schema_macro_f1",
        "balanced_accuracy",
        "negative_log_likelihood",
        "expected_calibration_error",
        "worst_family_fixed_schema_macro_f1",
    ):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f"validation metric {key} is not finite")
    return metrics


def _torch_loss(
    discovered: Any,
    logits: Any,
    labels: Any,
    class_counts: Any,
    class_weights: Any,
) -> Any:
    import torch.nn.functional as functional

    custom = getattr(discovered.module, "stage1_loss", None)
    if custom is not None:
        return custom(
            logits,
            labels,
            class_counts=class_counts,
            class_weights=class_weights,
        )
    return functional.cross_entropy(logits, labels, weight=class_weights)


def _evaluate_torch(
    model: Any, well: np.ndarray, seismic: np.ndarray, *, device: Any
) -> np.ndarray:
    import torch

    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(well), 64):
            stop = min(start + 64, len(well))
            logits = model(
                torch.as_tensor(well[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(seismic[start:stop], dtype=torch.float32, device=device),
            )
            predictions.append(logits.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _torch_pilot(
    discovered: Any,
    lock: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    device_name: str,
    checkpoint_path: Path,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    import torch

    train_well, train_seismic, train_labels = _lane_arrays(arrays, "P", "train")
    validation_well, validation_seismic, validation_labels = _lane_arrays(
        arrays, "P", "validation"
    )
    config = _stage2_model_config(lock, train_well, train_seismic, int(seeds["model"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device_name} but CUDA is unavailable")
    cuda_index: int | None = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_index)

    _seed_everything(int(seeds["model"]))
    model = discovered.build(lithofacies_task_spec(), **config).to(device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("torch backend did not return torch.nn.Module")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_counts = torch.as_tensor(arrays["class_counts"], dtype=torch.float32, device=device)
    class_weights = torch.as_tensor(arrays["class_weights"], dtype=torch.float32, device=device)
    gate_size = min(P_BATCH_SIZE, len(train_labels))
    gate_well = torch.as_tensor(train_well[:gate_size], dtype=torch.float32, device=device)
    gate_seismic = torch.as_tensor(train_seismic[:gate_size], dtype=torch.float32, device=device)
    gate_labels = torch.as_tensor(train_labels[:gate_size], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        gate_initial = float(
            _torch_loss(
                discovered,
                model(gate_well, gate_seismic),
                gate_labels,
                class_counts,
                class_weights,
            ).detach().cpu()
        )
    gate_losses = []
    for _ in range(TINY_GATE_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        gate_logits = model(gate_well, gate_seismic)
        if tuple(gate_logits.shape) != (gate_size, NUM_CLASSES) or not bool(torch.isfinite(gate_logits).all()):
            raise ValueError("tiny gate logits are not finite [B,9]")
        gate_loss = _torch_loss(
            discovered, gate_logits, gate_labels, class_counts, class_weights
        )
        if not bool(torch.isfinite(gate_loss)):
            raise ValueError("tiny gate loss is not finite")
        gate_loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("tiny gate backward did not produce finite gradients")
        optimizer.step()
        gate_losses.append(float(gate_loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        gate_final = float(
            _torch_loss(
                discovered,
                model(gate_well, gate_seismic),
                gate_labels,
                class_counts,
                class_weights,
            ).detach().cpu()
        )
    if gate_final > gate_initial + 1e-6:
        raise RuntimeError(
            f"tiny-overfit gate did not reduce loss: {gate_initial:.6g}->{gate_final:.6g}"
        )

    sampler = np.random.default_rng(int(seeds["sampler"]))
    remaining_updates = NEURAL_PARAMETER_UPDATE_LIMIT - TINY_GATE_UPDATES
    training_losses: list[float] = []
    order = np.arange(len(train_labels), dtype=np.int64)
    cursor = len(order)
    model.train()
    for _ in range(remaining_updates):
        if cursor + P_BATCH_SIZE > len(order):
            order = sampler.permutation(len(train_labels))
            cursor = 0
        indices = order[cursor : cursor + P_BATCH_SIZE]
        cursor += len(indices)
        optimizer.zero_grad(set_to_none=True)
        well_batch = torch.as_tensor(train_well[indices], dtype=torch.float32, device=device)
        seismic_batch = torch.as_tensor(train_seismic[indices], dtype=torch.float32, device=device)
        labels_batch = torch.as_tensor(train_labels[indices], dtype=torch.long, device=device)
        logits = model(well_batch, seismic_batch)
        loss = _torch_loss(discovered, logits, labels_batch, class_counts, class_weights)
        if not bool(torch.isfinite(loss)):
            raise ValueError("pilot training loss is not finite")
        loss.backward()
        optimizer.step()
        training_losses.append(float(loss.detach().cpu()))

    validation_logits = _evaluate_torch(
        model, validation_well, validation_seismic, device=device
    )
    _validate_logits(validation_logits, validation_labels, "P")
    _softmax_numpy(validation_logits, "P")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": STAGE2_SCHEMA,
            "model_id": lock["model_id"],
            "source_revision": lock["revision"],
            "class_names": CLASS_NAMES,
            "config": config,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "parameter_updates": NEURAL_PARAMETER_UPDATE_LIMIT,
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded = discovered.build(lithofacies_task_spec(), **loaded["config"]).to(device)
    reloaded.load_state_dict(loaded["state_dict"])
    reloaded_logits = _evaluate_torch(
        reloaded, validation_well, validation_seismic, device=device
    )
    roundtrip_error = float(np.max(np.abs(validation_logits - reloaded_logits)))
    if roundtrip_error != 0.0:
        raise RuntimeError(f"Stage-2 checkpoint changed logits: max_abs={roundtrip_error}")
    if device.type == "cuda":
        assert cuda_index is not None
        torch.cuda.synchronize(cuda_index)
        peak_vram = int(torch.cuda.max_memory_allocated(cuda_index))
    else:
        peak_vram = 0
    return {
        "backend": "torch",
        "model_config": config,
        "parameter_updates": NEURAL_PARAMETER_UPDATE_LIMIT,
        "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
        "tiny_gate": {
            "status": "PASS",
            "updates": TINY_GATE_UPDATES,
            "initial_loss": gate_initial,
            "final_loss": gate_final,
            "finite_shape_backward": True,
        },
        "training_loss": {
            "first": training_losses[0],
            "last": training_losses[-1],
            "minimum": min(training_losses),
        },
        "validation_logits": validation_logits,
        "validation_labels": validation_labels,
        "checkpoint": {
            "roundtrip": "PASS",
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "roundtrip_max_abs": roundtrip_error,
            "retained_in_commit": False,
        },
        "peak_vram_bytes": peak_vram,
    }


def _estimator_pilot(
    discovered: Any,
    lock: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    checkpoint_path: Path,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    train_well, train_seismic, train_labels = _lane_arrays(arrays, "P", "train")
    validation_well, validation_seismic, validation_labels = _lane_arrays(
        arrays, "P", "validation"
    )
    config = _stage2_model_config(lock, train_well, train_seismic, int(seeds["model"]))
    _seed_everything(int(seeds["model"]))
    model = discovered.build(lithofacies_task_spec(), **config)
    fit_loss = float(
        model.fit_stage1(
            train_well,
            train_seismic,
            train_labels,
            class_counts=arrays["class_counts"],
        )
    )
    if not math.isfinite(fit_loss):
        raise ValueError("estimator fit loss is not finite")
    validation_logits = np.asarray(
        model.predict_logits(validation_well, validation_seismic), dtype=np.float32
    )
    _validate_logits(validation_logits, validation_labels, "P")
    _softmax_numpy(validation_logits, "P")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": STAGE2_SCHEMA,
                "model_id": lock["model_id"],
                "source_revision": lock["revision"],
                "class_names": CLASS_NAMES,
                "config": config,
                "model": model,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with checkpoint_path.open("rb") as handle:
        reloaded = pickle.load(handle)
    reloaded_logits = np.asarray(
        reloaded["model"].predict_logits(validation_well, validation_seismic),
        dtype=np.float32,
    )
    roundtrip_error = float(np.max(np.abs(validation_logits - reloaded_logits)))
    if roundtrip_error != 0.0:
        raise RuntimeError(f"estimator checkpoint changed logits: max_abs={roundtrip_error}")
    estimator_iterations = int(
        config.get("rounds", config.get("iterations", 1))
    )
    return {
        "backend": "estimator",
        "model_config": config,
        "parameter_updates": 0,
        "estimator_fit_calls": 1,
        "estimator_iterations_or_transforms": estimator_iterations,
        "tiny_gate": {
            "status": "PASS",
            "source": "accepted Stage-1 real-code-path fit plus Stage-2 finite/shape/checkpoint gate",
            "finite_shape_backward": "backward_not_applicable",
        },
        "training_loss": {"fit": fit_loss},
        "validation_logits": validation_logits,
        "validation_labels": validation_labels,
        "checkpoint": {
            "roundtrip": "PASS",
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "roundtrip_max_abs": roundtrip_error,
            "retained_in_commit": False,
        },
        "peak_vram_bytes": 0,
    }


def _reason(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def run_pilot(
    batch_file: Path,
    output: Path,
    model_ids: Sequence[str],
    *,
    device: str,
) -> tuple[dict[str, Any], int]:
    """Run selected preregistered cells; never accepts a dataset or test path."""
    output = _track_owned(output)
    _verify_external_gpu_lock(device)
    arrays, manifest = load_batch(batch_file)
    contract = batch_contract(arrays, manifest, batch_file)
    source_lock = load_source_lock()
    by_id = {model["model_id"]: model for model in source_lock["models"]}
    invalid = sorted(set(model_ids) - set(by_id))
    if invalid:
        raise ValueError(f"models are outside the frozen first ten: {invalid}")
    results: list[dict[str, Any]] = []
    checkpoint_dir = output.parent / "checkpoints"
    for model_id in model_ids:
        lock = by_id[model_id]
        lane = str(lock["leaderboard_lane"])
        backend_hint: str | None = None
        seeds = {
            component: derive_cell_seed(model_id, component)
            for component in ("model", "loader", "sampler", "diagnostic")
        }
        _seed_everything(seeds["model"])
        started = time.monotonic()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        base: dict[str, Any] = {
            "schema_version": STAGE2_SCHEMA,
            "track_id": "lithofacies",
            "task_id": TASK_ID,
            "model_id": model_id,
            "lane": lane,
            "status": None,
            "reason": None,
            "seed": seeds["model"],
            "component_seeds": seeds,
            "source_revision": lock["revision"],
            "source_lock_sha256": _sha256(TRACK_DIR / "p5_source_lock.json"),
            "split_hash": contract["split_hash"],
            "fold_id": FIXED_FOLD_ID,
            "train_groups": contract["train_groups"],
            "validation_groups": contract["validation_groups"],
            "input_budget": _input_budget(contract, lane),
            "frozen_test_accessed": False,
            "test_metrics_used": False,
            "rank_eligible": False,
            "parameter_updates": 0,
        }
        try:
            if lane == "S":
                _lane_arrays(arrays, lane, "train")
                _lane_arrays(arrays, lane, "validation")
                raise RuntimeError("S lane unexpectedly became available without a registered Stage-2 budget")
            discovered = discover_model("lithofacies", model_id)
            if discovered.capabilities.get("leaderboard_lane") != lane:
                raise RuntimeError("adapter lane disagrees with source lock")
            backend = str(discovered.capabilities.get("backend"))
            backend_hint = backend
            suffix = ".pt" if backend == "torch" else ".pkl"
            checkpoint_path = checkpoint_dir / f"{model_id}{suffix}"
            if backend == "torch":
                evidence = _torch_pilot(
                    discovered,
                    lock,
                    arrays,
                    device_name=device,
                    checkpoint_path=checkpoint_path,
                    seeds=seeds,
                )
                wall_limit = NEURAL_WALL_LIMIT_SECONDS
            elif backend == "estimator":
                if device != "cpu":
                    raise ValueError("estimator cells must run on CPU")
                evidence = _estimator_pilot(
                    discovered,
                    lock,
                    arrays,
                    checkpoint_path=checkpoint_path,
                    seeds=seeds,
                )
                wall_limit = ESTIMATOR_WALL_LIMIT_SECONDS
            else:
                raise ValueError(f"unknown Stage-2 backend {backend!r}")
            wall_seconds = time.monotonic() - started
            metrics = _metric_payload(
                evidence.pop("validation_labels"),
                evidence.pop("validation_logits"),
                FIXED_VALIDATION_FAMILY,
            )
            if wall_seconds > wall_limit:
                status = "TIMEOUT"
                reason = _reason(
                    "wall_budget_exceeded",
                    f"cell exceeded its {wall_limit:g}s frozen wall budget",
                    wall_limit_seconds=wall_limit,
                )
                rank_eligible = False
            else:
                status = "PASS"
                reason = None
                rank_eligible = True
        except OptionalDependencyUnavailable as exc:
            evidence = {}
            metrics = None
            wall_limit = (
                ESTIMATOR_WALL_LIMIT_SECONDS
                if backend_hint == "estimator"
                else NEURAL_WALL_LIMIT_SECONDS
            )
            status = "SKIP"
            reason = _reason(
                "missing_optional_dependency", str(exc), dependency=exc.dependency
            )
            rank_eligible = False
            wall_seconds = time.monotonic() - started
        except LaneUnavailable as exc:
            evidence = {}
            metrics = None
            wall_limit = NEURAL_WALL_LIMIT_SECONDS
            status = "SKIP"
            reason = _reason(
                "lane_not_rankable",
                exc.reason,
                lane=exc.lane,
                policy="do not fabricate MD order and do not place S cells on the P leaderboard",
            )
            rank_eligible = False
            wall_seconds = time.monotonic() - started
        except Exception as exc:  # fail-loud, structured evidence for the cell
            evidence = {}
            metrics = None
            wall_limit = (
                ESTIMATOR_WALL_LIMIT_SECONDS
                if backend_hint == "estimator"
                else NEURAL_WALL_LIMIT_SECONDS
            )
            status = "FAIL"
            reason = _reason(
                "stage2_pilot_failure",
                str(exc),
                exception=type(exc).__name__,
                traceback=traceback.format_exc().splitlines()[-12:],
            )
            rank_eligible = False
            wall_seconds = time.monotonic() - started
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {
            **base,
            "status": status,
            "reason": reason,
            "rank_eligible": rank_eligible,
            "wall_seconds": wall_seconds,
            "wall_limit_seconds": wall_limit,
            "peak_resources": {
                "process_peak_rss_kib": int(max(rss_before, rss_after)),
                "peak_vram_bytes": int(evidence.get("peak_vram_bytes", 0)),
            },
            "environment": _portable_environment(device),
            "validation_metrics": metrics,
            **evidence,
        }
        result.pop("peak_vram_bytes", None)
        validate_cell_result(result)
        results.append(result)
    payload = {
        "schema_version": PARTIAL_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "source_lock_sha256": _sha256(TRACK_DIR / "p5_source_lock.json"),
        "batch_sha256": contract["batch_sha256"],
        "split_hash": contract["split_hash"],
        "models": results,
        "frozen_test_accessed": False,
    }
    _atomic_write_json(output, payload)
    exit_code = int(any(result["status"] in {"FAIL", "TIMEOUT"} for result in results))
    return payload, exit_code


def validate_cell_result(result: Mapping[str, Any]) -> None:
    required = {
        "model_id",
        "task_id",
        "lane",
        "status",
        "reason",
        "seed",
        "split_hash",
        "input_budget",
        "wall_seconds",
        "wall_limit_seconds",
        "peak_resources",
        "validation_metrics",
        "frozen_test_accessed",
        "test_metrics_used",
        "rank_eligible",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"Stage-2 cell lacks required fields: {missing}")
    if result["model_id"] not in FIRST_TEN or result["task_id"] != TASK_ID:
        raise ValueError("Stage-2 cell is outside the preregistered lithofacies roster")
    if result["lane"] not in {"P", "S"}:
        raise ValueError("Stage-2 lane must be P or S")
    if result["status"] not in {"PASS", "SKIP", "FAIL", "TIMEOUT"}:
        raise ValueError("Stage-2 status is invalid")
    if result["frozen_test_accessed"] is not False or result["test_metrics_used"] is not False:
        raise RuntimeError("Stage-2 result violates the frozen-test firewall")
    if result["lane"] == "P":
        if int(result["input_budget"]["context_positions"]) != P_CONTEXT_LENGTH:
            raise ValueError("Stage-2 cell changed the P context budget")
        if int(result["input_budget"]["fold_train_samples_used"]) > P_TRAIN_SAMPLE_LIMIT:
            raise ValueError("Stage-2 cell exceeded the train sample budget")
        if int(result["input_budget"]["fold_validation_samples_used"]) > P_VALIDATION_SAMPLE_LIMIT:
            raise ValueError("Stage-2 cell exceeded the validation sample budget")
    if float(result["wall_limit_seconds"]) > (
        ESTIMATOR_WALL_LIMIT_SECONDS
        if result.get("backend") == "estimator"
        else NEURAL_WALL_LIMIT_SECONDS
    ):
        raise ValueError("Stage-2 cell declared an excessive wall budget")
    if int(result.get("parameter_updates", 0)) > NEURAL_PARAMETER_UPDATE_LIMIT:
        raise ValueError("Stage-2 cell exceeded the parameter-update budget")
    if result["rank_eligible"]:
        if result["lane"] != "P" or result["status"] != "PASS":
            raise ValueError("only successful P-lane cells can be rank eligible")
        if result["validation_metrics"] is None:
            raise ValueError("rank-eligible cell lacks validation metrics")
        if float(result["wall_seconds"]) > float(result["wall_limit_seconds"]):
            raise ValueError("over-budget cell cannot be rank eligible")
    if result["lane"] == "S" and result["rank_eligible"]:
        raise ValueError("S-lane cell cannot enter the P leaderboard")


def _leaderboard(results: Sequence[Mapping[str, Any]], split_hash: str) -> dict[str, Any]:
    eligible = [
        result
        for result in results
        if result["lane"] == "P" and result["status"] == "PASS" and result["rank_eligible"]
    ]
    eligible.sort(
        key=lambda result: (
            -float(result["validation_metrics"]["fixed_schema_macro_f1"]),
            -float(result["validation_metrics"]["worst_family_fixed_schema_macro_f1"]),
            float(result["wall_seconds"]),
            str(result["model_id"]),
        )
    )
    entries = []
    for rank, result in enumerate(eligible, start=1):
        metrics = result["validation_metrics"]
        entries.append(
            {
                "rank": rank,
                "model_id": result["model_id"],
                "fixed_schema_macro_f1": metrics["fixed_schema_macro_f1"],
                "worst_family_fixed_schema_macro_f1": metrics[
                    "worst_family_fixed_schema_macro_f1"
                ],
                "supported_class_macro_f1": metrics["supported_class_macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "wall_seconds": result["wall_seconds"],
                "peak_vram_bytes": result["peak_resources"]["peak_vram_bytes"],
                "seed": result["seed"],
            }
        )
    status = "ranked" if len(entries) >= 2 else "not_rankable"
    return {
        "schema_version": LEADERBOARD_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "lane": "P",
        "status": status,
        "reason": None if status == "ranked" else "fewer than two legal P validation results",
        "primary_metric": "fixed_schema_macro_f1",
        "tie_breakers": [
            "worst_family_fixed_schema_macro_f1_desc",
            "wall_seconds_asc",
            "model_id_asc",
        ],
        "split_hash": split_hash,
        "fold_id": FIXED_FOLD_ID,
        "validation_family": FIXED_VALIDATION_FAMILY,
        "development_only": True,
        "formal_test_ranking": False,
        "entries": entries,
        "frozen_test_accessed": False,
    }


def finalize_results(inputs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    """Merge disjoint environment reports into portable canonical artifacts."""
    output_dir = _track_owned(output_dir)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not payloads:
        raise ValueError("finalize requires at least one partial result")
    source_hashes = {payload.get("source_lock_sha256") for payload in payloads}
    batch_hashes = {payload.get("batch_sha256") for payload in payloads}
    split_hashes = {payload.get("split_hash") for payload in payloads}
    if len(source_hashes) != 1 or len(batch_hashes) != 1 or len(split_hashes) != 1:
        raise ValueError("cannot merge partials from different source locks, batches, or splits")
    by_id: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if payload.get("frozen_test_accessed") is not False:
            raise RuntimeError("partial result violates the frozen-test firewall")
        for result in payload.get("models", []):
            validate_cell_result(result)
            model_id = str(result["model_id"])
            if model_id in by_id:
                raise ValueError(f"duplicate Stage-2 result for {model_id}")
            by_id[model_id] = dict(result)
    missing = [model_id for model_id in FIRST_TEN if model_id not in by_id]
    unexpected = sorted(set(by_id) - set(FIRST_TEN))
    if missing or unexpected:
        raise ValueError(f"Stage-2 roster mismatch: missing={missing}, unexpected={unexpected}")
    results = [by_id[model_id] for model_id in FIRST_TEN]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    results_text = "".join(
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for result in results
    )
    _atomic_write_text(results_path, results_text)
    split_hash = next(iter(split_hashes))
    leaderboard = _leaderboard(results, split_hash)
    leaderboard_path = output_dir / P_LEADERBOARD_FILENAME
    _atomic_write_json(leaderboard_path, leaderboard)
    counts = {
        status.lower(): sum(result["status"] == status for result in results)
        for status in ("PASS", "SKIP", "FAIL", "TIMEOUT")
    }
    s_result = next(result for result in results if result["lane"] == "S")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "track_id": "lithofacies",
        "task_id": TASK_ID,
        "class_names": CLASS_NAMES,
        "class_count": NUM_CLASSES,
        "root_seed": ROOT_SEED,
        "expected_cells": len(FIRST_TEN),
        "recorded_cells": len(results),
        "attempted_cells": sum(result["status"] != "SKIP" for result in results),
        "passed_cells": counts["pass"],
        "skipped_cells": counts["skip"],
        "failed_cells": counts["fail"],
        "timeout_cells": counts["timeout"],
        "status_counts": counts,
        "source_lock_sha256": next(iter(source_hashes)),
        "batch_sha256": next(iter(batch_hashes)),
        "split_hash": split_hash,
        "results_sha256": _sha256(results_path),
        "p_leaderboard_sha256": _sha256(leaderboard_path),
        "fixed_fold": {
            "fold_id": FIXED_FOLD_ID,
            "train_groups": results[0]["train_groups"],
            "validation_groups": results[0]["validation_groups"],
        },
        "p_lane": {
            "status": leaderboard["status"],
            "eligible_models": len(leaderboard["entries"]),
            "primary_metric": leaderboard["primary_metric"],
            "leaderboard_file": P_LEADERBOARD_FILENAME,
        },
        "s_lane": {
            "status": "not_rankable",
            "model_id": s_result["model_id"],
            "reason": s_result["reason"],
            "included_in_p_leaderboard": False,
        },
        "artifacts": {
            "results": RESULTS_FILENAME,
            "summary": SUMMARY_FILENAME,
            "p_leaderboard": P_LEADERBOARD_FILENAME,
            "runtime_checkpoints_committed": False,
        },
        "development_only": True,
        "frozen_test_accessed": False,
        "formal_test_ranking": False,
    }
    _atomic_write_json(output_dir / SUMMARY_FILENAME, summary)
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return FIRST_TEN
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("--models must contain at least one model id")
    return models


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-batch", help="prepare fixed development fold only")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--batch-file", type=Path, required=True)
    pilot = subparsers.add_parser("pilot", help="run fixed-budget preregistered cells")
    pilot.add_argument("--batch-file", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--models", type=_parse_models, default=FIRST_TEN)
    pilot.add_argument("--device", default="cpu")
    finalize = subparsers.add_parser("finalize", help="create canonical JSONL/summary/P leaderboard")
    finalize.add_argument("--inputs", type=Path, nargs="+", required=True)
    finalize.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare-batch":
        payload = prepare_stage2_batch(args.dataset_root, args.batch_file)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "pilot":
        payload, exit_code = run_pilot(
            args.batch_file, args.output, args.models, device=args.device
        )
        print(
            json.dumps(
                {
                    "models": [
                        {
                            "model_id": result["model_id"],
                            "lane": result["lane"],
                            "status": result["status"],
                            "reason": result["reason"],
                            "wall_seconds": result["wall_seconds"],
                        }
                        for result in payload["models"]
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return exit_code
    summary = finalize_results(args.inputs, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["status_counts"]["fail"] + summary["status_counts"]["timeout"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
