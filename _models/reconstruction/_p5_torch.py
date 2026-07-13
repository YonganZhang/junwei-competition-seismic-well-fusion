"""Torch execution adapter loaded only after a model's dependency gate passes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec
from _models.reconstruction._p5_adapter import masked_mse, point_batch_arrays, validate_n_features


class TorchRegressionAdapter:
    """One-step regression/checkpoint surface for point or 3-D Torch networks."""

    checkpoint_version = "p5-torch-regression-v1"

    def __init__(
        self,
        torch: Any,
        network: Any,
        task_spec: TaskSpec,
        *,
        model_id: str,
        n_features: int,
        representation: str,
        learning_rate: float,
        weight_decay: float,
        device: str,
        minimum_spatial_size: int = 1,
    ) -> None:
        if representation not in {"point", "volume"}:
            raise ValueError("Torch representation must be point or volume")
        self.torch = torch
        self.task_spec = task_spec
        self.mode = validate_n_features(task_spec, n_features)
        self.model_id = model_id
        self.n_features = int(n_features)
        self.representation = representation
        self.minimum_spatial_size = int(minimum_spatial_size)
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.update_count = 0

    def _volume_arrays(self, batch: ModelBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if "volume" not in batch.inputs:
            raise ValueError("volume adapter requires ModelBatch.inputs['volume']")
        values = np.asarray(batch.inputs["volume"], dtype=np.float32)
        target_name = self.task_spec.targets[0]
        if batch.targets is None or target_name not in batch.targets:
            raise ValueError("volume batch is missing target")
        target = np.asarray(batch.targets[target_name], dtype=np.float32)
        mask = np.asarray(batch.target_masks[target_name], dtype=bool)
        if values.ndim != 5 or values.shape[1] != self.n_features:
            raise ValueError(f"expected volume [B,{self.n_features},D,H,W]")
        if target.shape != (values.shape[0], 1, *values.shape[2:]) or mask.shape != target.shape:
            raise ValueError("volume target/mask shape mismatch")
        if not np.isfinite(values).all() or not np.isfinite(target).all() or not np.any(mask):
            raise ValueError("volume batch must be finite with a non-empty target mask")
        return values, target, mask

    def _pad_volume(self, tensor: Any) -> tuple[Any, tuple[int, int, int]]:
        original = tuple(int(value) for value in tensor.shape[-3:])
        target = []
        for size in original:
            minimum = max(size, self.minimum_spatial_size)
            multiple = self.minimum_spatial_size if self.minimum_spatial_size > 1 else 1
            target.append(((minimum + multiple - 1) // multiple) * multiple)
        pads: list[int] = []
        for size, wanted in reversed(list(zip(original, target))):
            pads.extend([0, wanted - size])
        if any(pads):
            tensor = self.torch.nn.functional.pad(tensor, tuple(pads))
        return tensor, original

    def _forward_tensor(self, features: Any) -> Any:
        if self.representation == "volume":
            padded, original = self._pad_volume(features)
            prediction = self.network(padded)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            prediction = prediction[..., : original[0], : original[1], : original[2]]
        else:
            prediction = self.network(features)
        if prediction.ndim > 1 and prediction.shape[1] == 1 and self.representation == "point":
            prediction = prediction[:, 0]
        return prediction

    def _tensors(self, batch: ModelBatch) -> tuple[Any, Any, Any]:
        if self.representation == "point":
            features, target, mask = point_batch_arrays(batch, self.task_spec)
            features = features.astype(np.float32)
            target = target.astype(np.float32)
        else:
            features, target, mask = self._volume_arrays(batch)
        return (
            self.torch.as_tensor(features, device=self.device),
            self.torch.as_tensor(target, device=self.device),
            self.torch.as_tensor(mask, dtype=self.torch.bool, device=self.device),
        )

    def train_batch(self, batch: ModelBatch) -> Mapping[str, Any]:
        self.network.train()
        features, target, mask = self._tensors(batch)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self._forward_tensor(features)
        if prediction.shape != target.shape:
            raise ValueError(f"Torch prediction shape {prediction.shape} != target {target.shape}")
        loss = ((prediction - target)[mask] ** 2).mean()
        if not bool(self.torch.isfinite(loss)):
            raise FloatingPointError("Torch training loss is non-finite")
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        return {
            "loss": float(loss.detach().cpu()),
            "valid_count": int(mask.sum().detach().cpu()),
            "backward": True,
            "fit": True,
        }

    def predict(self, batch: ModelBatch) -> ModelOutput:
        self.network.eval()
        input_name = "features" if self.representation == "point" else "volume"
        values = np.asarray(batch.inputs[input_name], dtype=np.float32)
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(values, device=self.device)
            prediction = self._forward_tensor(tensor).detach().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Torch adapter produced non-finite prediction")
        return ModelOutput(raw={self.task_spec.targets[0]: prediction})

    def validation_loss(self, batch: ModelBatch) -> float:
        if self.representation == "point":
            _, target, mask = point_batch_arrays(batch, self.task_spec)
        else:
            _, target, mask = self._volume_arrays(batch)
        prediction = np.asarray(self.predict(batch).raw[self.task_spec.targets[0]], dtype=np.float64)
        return masked_mse(np.asarray(target), prediction, np.asarray(mask))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "checkpoint_version": self.checkpoint_version,
                "model_id": self.model_id,
                "task_id": self.task_spec.task_id,
                "mode": self.mode,
                "n_features": self.n_features,
                "representation": self.representation,
                "model_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "torch_cpu_rng_state": self.torch.get_rng_state(),
                "torch_cuda_rng_state": (
                    self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else []
                ),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        payload = self.torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("checkpoint_version") != self.checkpoint_version:
            raise ValueError("unsupported Torch checkpoint")
        expected = (self.model_id, self.task_spec.task_id, self.mode, self.n_features, self.representation)
        actual = (
            payload.get("model_id"), payload.get("task_id"), payload.get("mode"),
            payload.get("n_features"), payload.get("representation"),
        )
        if actual != expected:
            raise ValueError(f"Torch checkpoint contract mismatch: {actual!r} != {expected!r}")
        self.network.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.update_count = int(payload["update_count"])
