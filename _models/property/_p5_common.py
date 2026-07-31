"""Shared, dependency-gated building blocks for P5 property adapters.

The adapters intentionally own no data split or preprocessing.  They consume a
validated :class:`ModelBatch`, keep the three target masks independent, and
return model-domain raw outputs plus an explicit physical-domain view.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import pickle
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec


PROPERTY_TARGETS = ("PHIF", "KLOGH", "SW")
SOURCE_LOCK_PATH = Path(__file__).with_name("source_lock.json")


class Stage1GateError(RuntimeError):
    """A machine-readable, expected Stage-1 skip condition."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


def _source_lock() -> dict[str, Any]:
    return json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))


def source_lock_entry(model_id: str) -> dict[str, Any]:
    lock = _source_lock()
    try:
        return dict(lock["models"][model_id])
    except KeyError as exc:
        raise Stage1GateError(
            "source_lock_missing", f"{model_id} is absent from the property source lock", model_id=model_id
        ) from exc


def source_lock_sha256() -> str:
    return hashlib.sha256(SOURCE_LOCK_PATH.read_bytes()).hexdigest()


def require_model_dependencies(model_id: str) -> dict[str, Any]:
    """Import exactly the locked optional dependencies or raise a structured skip."""
    entry = source_lock_entry(model_id)
    imported: dict[str, Any] = {}
    for dependency in entry.get("dependencies", []):
        distribution = str(dependency["distribution"])
        import_name = str(dependency["import_name"])
        expected = dependency.get("version")
        if importlib.util.find_spec(import_name) is None:
            raise Stage1GateError(
                "optional_dependency_missing",
                f"{model_id} requires optional dependency {distribution}",
                model_id=model_id,
                dependency_group=entry["dependency_group"],
                distribution=distribution,
                import_name=import_name,
                expected_version=expected,
            )
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise Stage1GateError(
                "distribution_metadata_missing",
                f"{import_name} imports but distribution metadata for {distribution} is missing",
                model_id=model_id,
                distribution=distribution,
            ) from exc
        if expected is not None and actual != expected:
            raise Stage1GateError(
                "dependency_version_mismatch",
                f"{model_id} requires {distribution}=={expected}, found {actual}",
                model_id=model_id,
                distribution=distribution,
                expected_version=expected,
                actual_version=actual,
            )
        try:
            imported[import_name] = importlib.import_module(import_name)
        except Exception as exc:
            raise Stage1GateError(
                "optional_dependency_import_failed",
                f"failed to import {import_name}: {type(exc).__name__}: {exc}",
                model_id=model_id,
                distribution=distribution,
            ) from exc
    return imported


def require_approved_weight(model_id: str, config: Mapping[str, Any]) -> Path | None:
    """Fail closed for gated/unlocked checkpoints and never trigger a download."""
    weights = source_lock_entry(model_id).get("weights", {"required": False})
    if not weights.get("required", False):
        return None
    if weights.get("license_status") != "approved":
        raise Stage1GateError(
            "weight_license_unconfirmed",
            f"{model_id} checkpoint license is not approved in source_lock.json",
            model_id=model_id,
            checkpoint=weights.get("checkpoint_name"),
            license_status=weights.get("license_status"),
            auto_download=False,
        )
    expected_sha256 = weights.get("sha256")
    if not expected_sha256:
        raise Stage1GateError(
            "weight_sha256_unlocked",
            f"{model_id} checkpoint SHA-256 is not locked",
            model_id=model_id,
            auto_download=False,
        )
    raw_path = config.get("checkpoint_path")
    if not raw_path:
        raise Stage1GateError(
            "weight_checkpoint_missing",
            f"{model_id} requires an explicitly provisioned local checkpoint",
            model_id=model_id,
            checkpoint=weights.get("checkpoint_name"),
            auto_download=False,
        )
    path = Path(raw_path)
    if not path.is_file():
        raise Stage1GateError(
            "weight_checkpoint_missing",
            f"locked checkpoint does not exist: {path}",
            model_id=model_id,
            auto_download=False,
        )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise Stage1GateError(
            "weight_sha256_mismatch",
            f"{model_id} checkpoint SHA-256 does not match source lock",
            model_id=model_id,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
        )
    return path


def validate_property_task_spec(task_spec: TaskSpec) -> None:
    if task_spec.track_id != "property" or task_spec.task_type != "regression":
        raise ValueError("P5 property adapters require a property regression TaskSpec")
    if task_spec.targets != PROPERTY_TARGETS:
        raise ValueError(f"expected targets {PROPERTY_TARGETS}, found {task_spec.targets}")
    if tuple(task_spec.target_masks) != PROPERTY_TARGETS:
        raise ValueError("property target masks must be independently declared in target order")
    expected_transforms = {"PHIF": "identity", "KLOGH": "log1p(KLOGH_mD)", "SW": "identity"}
    if dict(task_spec.target_transform) != expected_transforms:
        raise ValueError("property target transforms do not match the frozen P5 contract")


def _sample_count(batch: ModelBatch) -> int:
    count = len(batch.sample_ids)
    if count <= 0:
        raise ValueError("property batch must be nonempty")
    return count


def feature_matrix(batch: ModelBatch, key: str = "tabular") -> np.ndarray:
    count = _sample_count(batch)
    if key not in batch.inputs:
        raise ValueError(f"property batch is missing input {key!r}")
    values = np.asarray(batch.inputs[key], dtype=np.float64)
    if values.ndim < 2 or values.shape[0] != count or not np.isfinite(values).all():
        raise ValueError(f"input {key!r} must be finite with leading shape [{count}], got {values.shape}")
    return values


def target_arrays(batch: ModelBatch, task_spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    count = _sample_count(batch)
    if batch.targets is None:
        raise ValueError("training/loss requires targets")
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for target in task_spec.targets:
        value = np.asarray(batch.targets[target], dtype=np.float64).reshape(-1)
        mask = np.asarray(batch.target_masks[target], dtype=bool).reshape(-1)
        if value.shape != (count,) or mask.shape != (count,):
            raise ValueError(f"target {target!r} and mask must have shape ({count},)")
        if not np.isfinite(value[mask]).all():
            raise ValueError(f"target {target!r} has non-finite values under its valid mask")
        if not mask.any():
            raise ValueError(f"target {target!r} has no valid labels in this batch")
        values.append(value)
        masks.append(mask)
    return np.column_stack(values), np.column_stack(masks)


def masked_mse(prediction: np.ndarray, batch: ModelBatch, task_spec: TaskSpec) -> dict[str, Any]:
    matrix = np.asarray(prediction, dtype=np.float64)
    if matrix.shape != (len(batch.sample_ids), len(task_spec.targets)) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid prediction matrix {matrix.shape}")
    targets, masks = target_arrays(batch, task_spec)
    per_target = {
        target: float(np.mean((matrix[masks[:, index], index] - targets[masks[:, index], index]) ** 2))
        for index, target in enumerate(task_spec.targets)
    }
    return {
        "loss": float(np.mean(list(per_target.values()))),
        "per_target": per_target,
        "valid_counts": {
            target: int(masks[:, index].sum()) for index, target in enumerate(task_spec.targets)
        },
    }


def property_output(prediction: np.ndarray, task_spec: TaskSpec) -> ModelOutput:
    matrix = np.asarray(prediction, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(task_spec.targets) or not np.isfinite(matrix).all():
        raise ValueError(f"expected finite [N,{len(task_spec.targets)}] prediction, got {matrix.shape}")
    raw = {target: matrix[:, index].copy() for index, target in enumerate(task_spec.targets)}
    transformed = {
        "PHIF": np.clip(raw["PHIF"], 0.0, 1.0),
        "KLOGH": np.expm1(np.maximum(raw["KLOGH"], 0.0)),
        "SW": np.clip(raw["SW"], 0.0, 1.0),
    }
    if not all(np.isfinite(values).all() for values in transformed.values()):
        raise FloatingPointError("physical property transform produced non-finite values")
    return ModelOutput(raw=raw, transformed=transformed, aux={"raw_is_preserved": True})


def seed_torch_runtime(torch: Any, seed: int, *, deterministic: bool = True) -> None:
    """Reset every torch RNG before module/optimizer construction.

    CUDA determinism also needs a cuBLAS workspace policy before the first
    matrix operation.  Stage-1 is a contract smoke, so fail-closed
    deterministic algorithms are preferable to silently accepting replay
    drift.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")


class IndependentTargetEstimatorAdapter:
    """Thin wrapper for libraries exposing a scikit-like estimator API."""

    def __init__(
        self,
        *,
        model_id: str,
        task_spec: TaskSpec,
        estimator_factory: Callable[[str], Any],
        feature_key: str = "tabular",
    ) -> None:
        validate_property_task_spec(task_spec)
        self.model_id = model_id
        self.task_spec = task_spec
        self.estimator_factory = estimator_factory
        self.feature_key = feature_key
        self.estimators: dict[str, Any] = {}

    def fit(self, batch: ModelBatch) -> dict[str, Any]:
        x = feature_matrix(batch, self.feature_key).reshape(len(batch.sample_ids), -1)
        targets, masks = target_arrays(batch, self.task_spec)
        self.estimators = {}
        for index, target in enumerate(self.task_spec.targets):
            estimator = self.estimator_factory(target)
            estimator.fit(x[masks[:, index]], targets[masks[:, index], index])
            self.estimators[target] = estimator
        return masked_mse(self.predict_array(batch), batch, self.task_spec)

    def predict_array(self, batch: ModelBatch) -> np.ndarray:
        if set(self.estimators) != set(self.task_spec.targets):
            raise RuntimeError("estimator adapter must be fitted before prediction")
        x = feature_matrix(batch, self.feature_key).reshape(len(batch.sample_ids), -1)
        prediction = np.column_stack(
            [np.asarray(self.estimators[target].predict(x), dtype=np.float64).reshape(-1)
             for target in self.task_spec.targets]
        )
        if prediction.shape != (len(batch.sample_ids), len(self.task_spec.targets)):
            raise ValueError(f"unexpected estimator prediction shape {prediction.shape}")
        return prediction

    def predict(self, batch: ModelBatch) -> ModelOutput:
        return property_output(self.predict_array(batch), self.task_spec)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "schema_version": 1,
                    "model_id": self.model_id,
                    "targets": self.task_spec.targets,
                    "feature_key": self.feature_key,
                    "estimators": self.estimators,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_checkpoint(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - trusted local Stage-1 artifact
        identity = (payload.get("model_id"), tuple(payload.get("targets", ())), payload.get("feature_key"))
        expected = (self.model_id, self.task_spec.targets, self.feature_key)
        if identity != expected:
            raise ValueError(f"checkpoint identity mismatch: expected={expected}, found={identity}")
        self.estimators = dict(payload["estimators"])


class TorchMultiTargetAdapter:
    """One-step masked training wrapper for small official PyTorch backbones."""

    def __init__(
        self,
        *,
        model_id: str,
        task_spec: TaskSpec,
        torch_module: Any,
        input_builder: Callable[[ModelBatch], tuple[Any, ...]],
        torch: Any,
        learning_rate: float,
        weight_decay: float,
        device: str,
        config: Mapping[str, Any],
    ) -> None:
        validate_property_task_spec(task_spec)
        self.model_id = model_id
        self.task_spec = task_spec
        self.torch = torch
        self.device = torch.device(device)
        self.module = torch_module.to(self.device)
        self.input_builder = input_builder
        self.config = dict(config)
        parameter_groups = (
            self.module.make_parameter_groups()
            if hasattr(self.module, "make_parameter_groups")
            else self.module.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            parameter_groups, lr=float(learning_rate), weight_decay=float(weight_decay)
        )

    def _inputs(self, batch: ModelBatch) -> tuple[Any, ...]:
        result: list[Any] = []
        for value in self.input_builder(batch):
            if value is None:
                result.append(None)
            elif self.torch.is_tensor(value):
                result.append(value.to(self.device))
            else:
                result.append(self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device))
        return tuple(result)

    def _forward(self, batch: ModelBatch) -> Any:
        prediction = self.module(*self._inputs(batch))
        if prediction.ndim not in {2, 3} or prediction.shape[0] != len(batch.sample_ids):
            raise ValueError(f"unexpected torch prediction shape {tuple(prediction.shape)}")
        if prediction.shape[-1] != len(self.task_spec.targets) or not self.torch.isfinite(prediction).all():
            raise FloatingPointError(f"invalid torch prediction shape/values {tuple(prediction.shape)}")
        return prediction

    def _loss(self, prediction: Any, batch: ModelBatch) -> tuple[Any, dict[str, int]]:
        targets_np, masks_np = target_arrays(batch, self.task_spec)
        targets = self.torch.as_tensor(targets_np, dtype=prediction.dtype, device=self.device)
        masks = self.torch.as_tensor(masks_np, dtype=prediction.dtype, device=self.device)
        if prediction.ndim == 3:
            targets = targets[:, None, :]
            masks = masks[:, None, :].expand(-1, prediction.shape[1], -1)
            reduce_dims = (0, 1)
        else:
            reduce_dims = (0,)
        denominator = masks.sum(dim=reduce_dims)
        if bool((denominator <= 0).any()):
            raise ValueError("every property target needs at least one valid label")
        per_target = (((prediction - targets) ** 2) * masks).sum(dim=reduce_dims) / denominator
        return per_target.mean(), {
            target: int(masks_np[:, index].sum()) for index, target in enumerate(self.task_spec.targets)
        }

    def fit(self, batch: ModelBatch) -> dict[str, Any]:
        self.module.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self._forward(batch)
        loss, valid_counts = self._loss(prediction, batch)
        if not self.torch.isfinite(loss):
            raise FloatingPointError("masked torch loss is non-finite")
        loss.backward()
        for parameter in self.module.parameters():
            if parameter.grad is not None and not self.torch.isfinite(parameter.grad).all():
                raise FloatingPointError("torch backward produced non-finite gradients")
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu()), "valid_counts": valid_counts, "backward": True}

    def predict_array(self, batch: ModelBatch) -> np.ndarray:
        self.module.eval()
        with self.torch.no_grad():
            prediction = self._forward(batch)
            if prediction.ndim == 3:
                prediction = prediction.mean(dim=1)
        return prediction.detach().cpu().numpy().astype(np.float64)

    def predict(self, batch: ModelBatch) -> ModelOutput:
        return property_output(self.predict_array(batch), self.task_spec)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "schema_version": 1,
                "model_id": self.model_id,
                "targets": self.task_spec.targets,
                "model_state": self.module.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        payload = self.torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("model_id") != self.model_id or tuple(payload.get("targets", ())) != self.task_spec.targets:
            raise ValueError("torch checkpoint model/target identity mismatch")
        self.module.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
