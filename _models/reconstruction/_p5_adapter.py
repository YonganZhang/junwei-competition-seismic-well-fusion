"""Track-local helpers for thin P5 reconstruction model adapters.

This module intentionally has no optional third-party imports.  Model files
remain dynamically discoverable in a clean checkout even when an upstream
package is unavailable; ``build_model`` then raises :class:`AdapterSkip` with
a machine-readable reason instead of importing a similarly named fallback.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec


class AdapterSkip(RuntimeError):
    """Expected Stage-1 skip caused by an explicit dependency/science gate."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


def require_dependency(module_name: str, *, model_id: str, distribution: str) -> Any:
    """Import one locked upstream dependency or produce a structured skip."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise AdapterSkip(
            "dependency_missing",
            f"{model_id} requires import {module_name!r} from {distribution!r}",
            model_id=model_id,
            module=module_name,
            distribution=distribution,
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc


def reconstruction_mode(task_spec: TaskSpec) -> str:
    if task_spec.track_id != "reconstruction" or task_spec.task_type != "reconstruction":
        raise ValueError("P5 reconstruction adapters require a reconstruction TaskSpec")
    mode = str(task_spec.metadata.get("evaluation_mode", ""))
    if mode not in {"strict", "conditional"}:
        raise ValueError("TaskSpec.metadata.evaluation_mode must be strict or conditional")
    if len(task_spec.targets) != 1:
        raise ValueError("P5 reconstruction adapters require exactly one target")
    return mode


def validate_n_features(task_spec: TaskSpec, n_features: int) -> str:
    mode = reconstruction_mode(task_spec)
    expected = len(task_spec.input_whitelist)
    if int(n_features) != expected:
        raise ValueError(
            f"{mode} adapter expected {expected} whitelisted features, got {n_features}"
        )
    if mode == "strict" and any("idw" in name or "well" in name for name in task_spec.input_whitelist):
        raise ValueError("strict TaskSpec must not expose IDW/well-value features")
    return mode


def point_batch_arrays(batch: ModelBatch, task_spec: TaskSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "features" not in batch.inputs:
        raise ValueError("point adapter requires ModelBatch.inputs['features']")
    features = np.asarray(batch.inputs["features"], dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("point features must be finite [N,C]")
    target_name = task_spec.targets[0]
    if batch.targets is None or target_name not in batch.targets:
        raise ValueError("training/evaluation batch is missing the reconstruction target")
    target = np.asarray(batch.targets[target_name], dtype=np.float64).reshape(-1)
    mask = np.asarray(batch.target_masks[target_name], dtype=bool).reshape(-1)
    if target.shape != (features.shape[0],) or mask.shape != target.shape:
        raise ValueError("point target/mask shape does not match features")
    if not np.isfinite(target).all() or not np.any(mask):
        raise ValueError("point target must be finite with at least one valid cell")
    return features, target, mask


def masked_mse(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if target.shape != prediction.shape or mask.shape != target.shape or not np.any(mask):
        raise ValueError("masked MSE inputs must have matching non-empty shapes")
    value = float(np.mean((prediction[mask] - target[mask]) ** 2))
    if not math.isfinite(value):
        raise FloatingPointError("masked MSE is non-finite")
    return value


class FittedPointAdapter:
    """Base class for deterministic CPU interpolation/geostatistical adapters."""

    checkpoint_version = "p5-fitted-point-v1"

    def __init__(self, task_spec: TaskSpec, *, n_features: int, model_id: str) -> None:
        self.task_spec = task_spec
        self.mode = validate_n_features(task_spec, n_features)
        self.n_features = int(n_features)
        self.model_id = model_id
        self.update_count = 0
        self._train_features: np.ndarray | None = None
        self._train_target: np.ndarray | None = None

    def _fit_backend(self, features: np.ndarray, target: np.ndarray) -> None:
        raise NotImplementedError

    def _predict_backend(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def train_batch(self, batch: ModelBatch) -> Mapping[str, Any]:
        features, target, mask = point_batch_arrays(batch, self.task_spec)
        self._train_features = features[mask].copy()
        self._train_target = target[mask].copy()
        self._fit_backend(self._train_features, self._train_target)
        self.update_count += 1
        loss = masked_mse(target, self.predict_array(features), mask)
        return {"loss": loss, "valid_count": int(mask.sum()), "backward": False, "fit": True}

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features or not np.isfinite(values).all():
            raise ValueError(f"expected finite [N,{self.n_features}] features")
        prediction, _ = self._predict_backend(values)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if prediction.shape != (len(values),) or not np.isfinite(prediction).all():
            raise FloatingPointError("point adapter produced invalid prediction")
        return prediction

    def predict(self, batch: ModelBatch) -> ModelOutput:
        features = np.asarray(batch.inputs["features"], dtype=np.float64)
        prediction, uncertainty = self._predict_backend(features)
        target_name = self.task_spec.targets[0]
        raw = {target_name: np.asarray(prediction, dtype=np.float64).reshape(-1)}
        unc = None if uncertainty is None else {
            target_name: np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        }
        return ModelOutput(raw=raw, uncertainty=unc)

    def validation_loss(self, batch: ModelBatch) -> float:
        features, target, mask = point_batch_arrays(batch, self.task_spec)
        return masked_mse(target, self.predict_array(features), mask)

    def save_checkpoint(self, path: Path) -> None:
        if self._train_features is None or self._train_target is None:
            raise RuntimeError("cannot checkpoint an unfitted adapter")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                checkpoint_version=np.asarray(self.checkpoint_version),
                model_id=np.asarray(self.model_id),
                task_id=np.asarray(self.task_spec.task_id),
                mode=np.asarray(self.mode),
                n_features=np.asarray(self.n_features),
                update_count=np.asarray(self.update_count),
                train_features=self._train_features,
                train_target=self._train_target,
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["checkpoint_version"]) != self.checkpoint_version:
                raise ValueError("unsupported fitted-point checkpoint")
            if str(payload["model_id"]) != self.model_id or str(payload["task_id"]) != self.task_spec.task_id:
                raise ValueError("checkpoint model/task mismatch")
            if int(payload["n_features"]) != self.n_features or str(payload["mode"]) != self.mode:
                raise ValueError("checkpoint feature/mode mismatch")
            self._train_features = np.asarray(payload["train_features"], dtype=np.float64)
            self._train_target = np.asarray(payload["train_target"], dtype=np.float64)
            self.update_count = int(payload["update_count"])
        self._fit_backend(self._train_features, self._train_target)
