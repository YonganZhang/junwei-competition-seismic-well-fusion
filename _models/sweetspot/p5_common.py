"""Small adapter helpers shared by the ten P5 sweetspot model modules."""
from __future__ import annotations

import io
import hashlib
import pickle
import resource
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from _code.ml_framework.contracts import ModelOutput, TaskSpec


class AdapterSkip(RuntimeError):
    """Expected Stage-1 skip with a machine-readable reason code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024


def require_single_target(task_spec: TaskSpec, supported: Sequence[str]) -> str:
    if len(task_spec.targets) != 1:
        raise ValueError("P5 sweetspot adapters require exactly one independent target/head")
    if task_spec.task_type not in set(supported):
        raise ValueError(
            f"adapter does not support task_type={task_spec.task_type!r}; supported={tuple(supported)}"
        )
    if not task_spec.metadata.get("single_target_head"):
        raise ValueError("P5 TaskSpec must explicitly declare metadata.single_target_head=true")
    return task_spec.targets[0]


def output_dim(task_spec: TaskSpec) -> int:
    if task_spec.task_type == "multiclass":
        count = int(task_spec.metadata.get("class_count", 0))
        if count < 2:
            raise ValueError("multiclass TaskSpec requires metadata.class_count>=2")
        return count
    return 1


@dataclass
class TabularEstimatorAdapter:
    task_spec: TaskSpec
    estimator: Any
    input_key: str = "tabular"

    def __post_init__(self) -> None:
        self.target = require_single_target(
            self.task_spec, ("binary", "multiclass", "regression", "ranking"),
        )
        self._fitted = False

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Mapping[str, Sequence[float]],
        masks: Mapping[str, Sequence[bool]],
    ) -> "TabularEstimatorAdapter":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets[self.target])
        mask = np.asarray(masks[self.target], dtype=bool)
        if x.ndim != 2 or x.shape[0] != y.shape[0] or mask.shape[0] != y.shape[0]:
            raise ValueError("tabular features, target and mask must be sample-aligned")
        valid = mask & np.isfinite(y)
        if valid.sum() < 2:
            raise ValueError("at least two valid development labels are required")
        if self.task_spec.task_type == "binary" and set(np.unique(y[valid].astype(int))) != {0, 1}:
            raise ValueError("binary development smoke requires both approved classes")
        self.estimator.fit(x[valid], y[valid])
        self._fitted = True
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> ModelOutput:
        if not self._fitted:
            raise RuntimeError("adapter must be fitted before prediction")
        x = np.asarray(features, dtype=np.float64)
        prediction = np.asarray(self.estimator.predict(x))
        if self.task_spec.task_type == "binary":
            probability = np.asarray(self.estimator.predict_proba(x))[:, 1]
            if hasattr(self.estimator, "decision_function"):
                raw = np.asarray(self.estimator.decision_function(x))
            elif hasattr(self.estimator, "predict"):
                try:
                    raw = np.asarray(self.estimator.predict(x, raw_score=True))
                except TypeError:
                    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
                    raw = np.log(clipped / (1.0 - clipped))
            return ModelOutput(raw={self.target: raw}, transformed={self.target: probability})
        if self.task_spec.task_type == "multiclass":
            probability = np.asarray(self.estimator.predict_proba(x))
            return ModelOutput(raw={self.target: probability}, transformed={self.target: probability})
        return ModelOutput(raw={self.target: prediction.reshape(-1)})

    def stage1_smoke(
        self,
        inputs: Mapping[str, Any],
        target: np.ndarray,
        target_mask: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, Any]:
        del seed
        if self.input_key not in inputs:
            raise AdapterSkip("input_modality_missing", f"required input array {self.input_key!r} is absent")
        x = np.asarray(inputs[self.input_key], dtype=np.float64)
        started = time.monotonic()
        self.fit(x, {self.target: target}, {self.target: target_mask})
        probe = x[: min(8, len(x))]
        before = self.predict(probe)
        encoded = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        restored = pickle.loads(encoded)
        after = restored.predict(probe)
        left = np.asarray(before.raw[self.target], dtype=np.float64)
        right = np.asarray(after.raw[self.target], dtype=np.float64)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError("tabular adapter returned non-finite or inconsistent output")
        delta = float(np.max(np.abs(left - right))) if left.size else 0.0
        if delta > 1e-10:
            raise RuntimeError(f"checkpoint round-trip changed predictions by {delta}")
        return {
            "synthetic_batch": "runner-owned",
            "real_development_batch_samples": int(len(x)),
            "raw_output_shape": list(left.shape),
            "finite_output": True,
            "checkpoint_bytes": len(encoded),
            "checkpoint_roundtrip_max_abs_delta": delta,
            "output_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
            "peak_rss_bytes": _rss_bytes(),
            "download_bytes": 0,
            "wall_seconds": time.monotonic() - started,
            "test_accessed": False,
        }


class TorchModuleAdapter:
    """One-target torch head with one-step and in-memory checkpoint smoke."""

    def __init__(
        self,
        task_spec: TaskSpec,
        module_factory: Callable[[], Any],
        *,
        input_key: str,
        device: str,
        forward_fn: Callable[[Any, Mapping[str, Any], Any], Any] | None = None,
    ) -> None:
        self.task_spec = task_spec
        self.target = require_single_target(task_spec, ("binary", "multiclass", "regression"))
        self.module_factory = module_factory
        self.input_key = input_key
        self.device = device
        self.forward_fn = forward_fn
        import torch

        torch.manual_seed(2693)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2693)
        self.module = module_factory().to(device)

    def _forward(self, inputs: Mapping[str, Any], torch: Any) -> Any:
        if self.forward_fn is not None:
            return self.forward_fn(self.module, inputs, torch)
        if self.input_key not in inputs:
            raise AdapterSkip("input_modality_missing", f"required input array {self.input_key!r} is absent")
        value = torch.as_tensor(inputs[self.input_key], dtype=torch.float32, device=self.device)
        return self.module(value)

    def _loss(self, output: Any, target: np.ndarray, mask: np.ndarray, torch: Any) -> Any:
        y = torch.as_tensor(target, device=self.device)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if self.task_spec.task_type == "multiclass":
            if output.ndim != 2:
                raise ValueError("multiclass output must be [batch, class]")
            y = y.long().reshape(-1)
            valid = valid.reshape(-1)
            return torch.nn.functional.cross_entropy(output[valid], y[valid])
        y = y.to(dtype=torch.float32)
        if output.ndim == y.ndim + 1 and output.shape[1] == 1:
            y = y.unsqueeze(1)
            valid = valid.unsqueeze(1)
        if output.shape != y.shape:
            raise ValueError(f"model output {tuple(output.shape)} does not match target {tuple(y.shape)}")
        valid = valid.expand_as(output)
        if not bool(valid.any()):
            raise ValueError("development smoke has zero valid target values")
        if self.task_spec.task_type == "binary":
            return torch.nn.functional.binary_cross_entropy_with_logits(output[valid], y[valid])
        return torch.nn.functional.mse_loss(output[valid], y[valid])

    def stage1_smoke(
        self,
        inputs: Mapping[str, Any],
        target: np.ndarray,
        target_mask: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, Any]:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise AdapterSkip("cuda_unavailable", "CUDA was requested but is unavailable")
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        self.module.train()
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        output = self._forward(inputs, torch)
        loss = self._loss(output, target, target_mask, torch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Stage-1 loss")
        loss.backward()
        optimizer.step()
        self.module.eval()
        with torch.no_grad():
            prediction = self._forward(inputs, torch).detach().cpu()
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("non-finite Stage-1 output")
        buffer = io.BytesIO()
        torch.save(self.module.state_dict(), buffer)
        buffer.seek(0)
        restored = self.module_factory().to(self.device)
        restored.load_state_dict(torch.load(buffer, map_location=self.device, weights_only=True))
        restored.eval()
        original = self.module
        self.module = restored
        try:
            with torch.no_grad():
                replay = self._forward(inputs, torch).detach().cpu()
        finally:
            self.module = original
        delta = float(torch.max(torch.abs(prediction - replay)).item()) if prediction.numel() else 0.0
        if delta > 1e-6:
            raise RuntimeError(f"torch checkpoint round-trip changed predictions by {delta}")
        peak = int(torch.cuda.max_memory_allocated(self.device)) if self.device.startswith("cuda") else 0
        return {
            "real_development_batch_samples": int(prediction.shape[0]),
            "raw_output_shape": list(prediction.shape),
            "finite_output": True,
            "single_step_loss": float(loss.detach().cpu().item()),
            "backward_completed": True,
            "checkpoint_bytes": buffer.getbuffer().nbytes,
            "checkpoint_roundtrip_max_abs_delta": delta,
            "output_sha256": hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
            "device": self.device,
            "peak_vram_bytes": peak,
            "peak_rss_bytes": _rss_bytes(),
            "download_bytes": 0,
            "wall_seconds": time.monotonic() - started,
            "test_accessed": False,
        }


def dependency_skip(package: str, detail: str | None = None) -> AdapterSkip:
    return AdapterSkip(
        "dependency_missing",
        detail or f"optional locked dependency {package!r} is unavailable; no substitute implementation is allowed",
    )
