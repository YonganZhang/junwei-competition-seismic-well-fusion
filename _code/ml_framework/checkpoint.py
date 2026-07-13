"""Framework-neutral, resumable checkpoint envelope including RNG state."""
from __future__ import annotations

import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_VERSION = "p4-v1"
_TRAINER_STATE_FIELDS = {
    "next_epoch",
    "global_step",
    "best_epoch",
    "best_val_loss",
    "epochs_without_improvement",
    "stopped_early",
    "history",
}


def _validate_resume_metadata(
    trainer_state: Mapping[str, Any],
    seed_report: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    missing_trainer = sorted(_TRAINER_STATE_FIELDS - set(trainer_state))
    if missing_trainer:
        raise ValueError(f"trainer_state missing fields: {missing_trainer}")
    if not isinstance(trainer_state.get("history"), list):
        raise ValueError("trainer_state.history must be a list")
    for name in ("next_epoch", "global_step", "best_epoch", "epochs_without_improvement"):
        if not isinstance(trainer_state.get(name), int):
            raise ValueError(f"trainer_state.{name} must be an integer")
    if not isinstance(trainer_state.get("stopped_early"), bool):
        raise ValueError("trainer_state.stopped_early must be boolean")
    if not isinstance(seed_report.get("root_seed"), int) or seed_report["root_seed"] < 0:
        raise ValueError("seed_report.root_seed must be a non-negative integer")
    if not isinstance(seed_report.get("seed_tree"), Mapping) or not seed_report["seed_tree"]:
        raise ValueError("seed_report.seed_tree must be a non-empty mapping")
    if not environment or not all(isinstance(key, str) and key.strip() for key in environment):
        raise ValueError("environment must be a non-empty mapping with named fields")


def capture_rng_state(*, include_torch: bool = True) -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    if include_torch:
        try:
            import torch

            state["torch_cpu"] = torch.get_rng_state()
            state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        except ImportError:
            state["torch_cpu"] = None
            state["torch_cuda"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        import torch

        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model_state: Any,
    optimizer_state: Any,
    scheduler_state: Any,
    scaler_state: Any,
    config_hash: str,
    split_hash: str,
    trainer_state: Mapping[str, Any],
    seed_report: Mapping[str, Any],
    environment: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    include_torch_rng: bool = True,
) -> Path:
    if epoch < 0:
        raise ValueError("epoch must be >=0")
    for name, value in (("config_hash", config_hash), ("split_hash", split_hash)):
        if not value:
            raise ValueError(f"{name} must not be empty")
    _validate_resume_metadata(trainer_state, seed_report, environment)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "rng_state": capture_rng_state(include_torch=include_torch_rng),
        "config_hash": config_hash,
        "split_hash": split_hash,
        "trainer_state": dict(trainer_state),
        "seed_report": dict(seed_report),
        "environment": dict(environment),
        "extra": dict(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    # Checkpoints are trusted local experiment artifacts; never load untrusted pickle files.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported or malformed checkpoint")
    required = {
        "epoch", "model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng_state",
        "config_hash", "split_hash", "trainer_state", "seed_report", "environment", "extra",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing fields: {missing}")
    _validate_resume_metadata(payload["trainer_state"], payload["seed_report"], payload["environment"])
    return payload
