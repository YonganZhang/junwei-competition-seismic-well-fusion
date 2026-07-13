"""Thin sparse variational GP adapter backed by GPyTorch."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec
from _models.reconstruction._p5_adapter import (
    masked_mse,
    point_batch_arrays,
    require_dependency,
    validate_n_features,
)


model_id = "gpytorch_svgp"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["coordinates", "seismic", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": True,
        "batch_representation": "point",
        "trainable": True,
        "dependency_group": "geostat-cpu",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    torch = require_dependency("torch", model_id=model_id, distribution="torch")
    gpytorch = require_dependency("gpytorch", model_id=model_id, distribution="gpytorch")
    n_features = int(config["n_features"])
    mode = validate_n_features(task_spec, n_features)
    num_inducing = int(config.get("num_inducing", 16))
    learning_rate = float(config.get("learning_rate", 0.01))
    n_training_samples = int(config.get("n_training_samples", 1))
    device = torch.device(str(config.get("device", "cpu")))
    seed = int(config.get("seed", 2693))
    if num_inducing <= 0 or n_training_samples <= 0:
        raise ValueError("num_inducing and n_training_samples must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inducing = torch.randn(num_inducing, n_features, generator=generator) * 0.05

    class VariationalGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points: Any) -> None:
            distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, distribution, learn_inducing_locations=True
            )
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
            )

        def forward(self, values: Any) -> Any:
            mean = self.mean_module(values)
            covariance = self.covar_module(values)
            return gpytorch.distributions.MultivariateNormal(mean, covariance)

    class SVGPAdapter:
        checkpoint_version = "p5-gpytorch-svgp-v1"

        def __init__(self) -> None:
            self.task_spec = task_spec
            self.model_id = model_id
            self.mode = mode
            self.n_features = n_features
            self.device = device
            self.model = VariationalGP(inducing.clone()).to(device)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.likelihood.parameters()),
                lr=learning_rate,
            )
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood, self.model, num_data=n_training_samples
            )
            self.update_count = 0

        def train_batch(self, batch: ModelBatch) -> Mapping[str, Any]:
            features, target, mask = point_batch_arrays(batch, self.task_spec)
            x = torch.as_tensor(features[mask], dtype=torch.float32, device=device)
            y = torch.as_tensor(target[mask], dtype=torch.float32, device=device)
            self.model.train(); self.likelihood.train(); self.optimizer.zero_grad(set_to_none=True)
            loss = -self.mll(self.model(x), y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("SVGP loss is non-finite")
            loss.backward(); self.optimizer.step(); self.update_count += 1
            return {"loss": float(loss.detach().cpu()), "valid_count": int(mask.sum()), "backward": True, "fit": True}

        def predict(self, batch: ModelBatch) -> ModelOutput:
            features = np.asarray(batch.inputs["features"], dtype=np.float64)
            self.model.eval(); self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = self.likelihood(
                    self.model(torch.as_tensor(features, dtype=torch.float32, device=device))
                )
            mean = posterior.mean.detach().cpu().numpy()
            variance = posterior.variance.detach().cpu().numpy()
            if not np.isfinite(mean).all() or not np.isfinite(variance).all():
                raise FloatingPointError("SVGP posterior is non-finite")
            target_name = self.task_spec.targets[0]
            return ModelOutput(raw={target_name: mean}, uncertainty={target_name: variance})

        def validation_loss(self, batch: ModelBatch) -> float:
            _, target, mask = point_batch_arrays(batch, self.task_spec)
            prediction = np.asarray(self.predict(batch).raw[self.task_spec.targets[0]])
            return masked_mse(target, prediction, mask)

        def save_checkpoint(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "checkpoint_version": self.checkpoint_version,
                "model_id": model_id, "task_id": self.task_spec.task_id, "mode": self.mode,
                "n_features": self.n_features, "model_state": self.model.state_dict(),
                "likelihood_state": self.likelihood.state_dict(),
                "optimizer_state": self.optimizer.state_dict(), "update_count": self.update_count,
                "torch_rng_state": torch.get_rng_state(),
            }, path)

        def load_checkpoint(self, path: Path) -> None:
            payload = torch.load(path, map_location=device, weights_only=False)
            expected = (self.checkpoint_version, model_id, self.task_spec.task_id, self.mode, self.n_features)
            actual = (payload.get("checkpoint_version"), payload.get("model_id"), payload.get("task_id"), payload.get("mode"), payload.get("n_features"))
            if actual != expected:
                raise ValueError("SVGP checkpoint contract mismatch")
            self.model.load_state_dict(payload["model_state"])
            self.likelihood.load_state_dict(payload["likelihood_state"])
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.update_count = int(payload["update_count"])

    return SVGPAdapter()
