"""Deterministic, budget-matched small-head core for the P38 LOGO3 pilot."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


FEATURE_DIM = 32
HIDDEN_DIM = 32
MAX_UPDATES = 120
EVAL_EVERY = 5
LEARNING_RATE = 3e-3
TARGET_BOUNDS = (0.0, 1.0)
PROJECTION_SEED = 2693


def _fixed_projection(width: int, *, stream: int) -> np.ndarray:
    if width <= 0:
        raise ValueError("feature width must be positive")
    if width <= FEATURE_DIM:
        output = np.zeros((width, FEATURE_DIM), dtype=np.float32)
        output[:, :width] = np.eye(width, dtype=np.float32)
        return output
    rng = np.random.default_rng(PROJECTION_SEED + int(stream))
    values = rng.normal(size=(width, FEATURE_DIM))
    q, _ = np.linalg.qr(values)
    return q[:, :FEATURE_DIM].astype(np.float32)


def _balanced_location_scale(
    values: np.ndarray,
    train_indices: np.ndarray,
    parent_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    indices = np.asarray(train_indices, dtype=np.int64)
    parents = np.asarray(parent_index, dtype=np.int64)
    unique = np.unique(parents[indices])
    if len(unique) == 0:
        raise ValueError("empty training parent set")
    parent_means = []
    parent_vars = []
    for parent in unique:
        selected = array[indices[parents[indices] == parent]]
        if len(selected) == 0 or not np.all(np.isfinite(selected)):
            raise FloatingPointError("non-finite or empty parent features")
        parent_means.append(np.mean(selected, axis=0))
        parent_vars.append(np.var(selected, axis=0))
    means = np.stack(parent_means)
    location = np.mean(means, axis=0)
    variance = np.mean(
        np.stack(parent_vars) + (means - location[None, :]) ** 2,
        axis=0,
    )
    scale = np.maximum(np.sqrt(variance), 1e-6)
    return location.astype(np.float32), scale.astype(np.float32)


@dataclass
class ModalityTransform:
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        transformed = ((array - self.mean) / self.scale) @ self.projection
        if not np.all(np.isfinite(transformed)):
            raise FloatingPointError("transformed modality is non-finite")
        return transformed.astype(np.float32)


def fit_modality_transform(
    values: np.ndarray,
    train_indices: np.ndarray,
    parent_index: np.ndarray,
    *,
    stream: int,
) -> ModalityTransform:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("modality features must be finite [N,D]")
    mean, scale = _balanced_location_scale(array, train_indices, parent_index)
    return ModalityTransform(
        mean=mean,
        scale=scale,
        projection=_fixed_projection(array.shape[1], stream=stream),
    )


class BoundedFusionHead(torch.nn.Module):
    """Same-parameter head for every control; modality masks prevent fake input."""

    def __init__(self, *, maximum_gate: float) -> None:
        super().__init__()
        if not (0.0 < maximum_gate <= 1.0):
            raise ValueError("maximum_gate must be in (0,1]")
        self.maximum_gate = float(maximum_gate)
        self.well = torch.nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.seismic = torch.nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.gate = torch.nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.trunk = torch.nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.output = torch.nn.Linear(HIDDEN_DIM, 1)

    def forward(
        self,
        well: torch.Tensor,
        seismic: torch.Tensor,
        well_present: float,
        seismic_present: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_well = torch.nn.functional.gelu(self.well(well)) * float(well_present)
        h_seismic = torch.nn.functional.gelu(self.seismic(seismic)) * float(seismic_present)
        gate = self.maximum_gate * torch.sigmoid(
            self.gate(torch.cat([h_well, h_seismic], dim=-1))
        )
        fused = torch.cat([h_well, gate * h_seismic], dim=-1)
        hidden = torch.nn.functional.gelu(self.trunk(fused))
        return self.output(hidden)[:, 0], gate


def parameter_count() -> int:
    model = BoundedFusionHead(maximum_gate=0.1)
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _seed_all(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _balanced_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parent: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for value in torch.unique(parent):
        selected = parent == value
        losses.append(torch.mean((prediction[selected] - target[selected]) ** 2))
    return torch.stack(losses).mean()


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(error**2)))


@dataclass
class FitBundle:
    model: BoundedFusionHead
    well_transform: ModalityTransform
    seismic_transform: ModalityTransform
    well_present: float
    seismic_present: float
    device: str
    steps: int
    audit: Mapping[str, Any]

    def predict(
        self,
        well_features: np.ndarray,
        seismic_features: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        chosen = np.asarray(indices, dtype=np.int64)
        well = self.well_transform.apply(well_features[chosen])
        seismic = self.seismic_transform.apply(seismic_features[chosen])
        self.model.eval()
        with torch.no_grad():
            prediction, gate = self.model(
                torch.as_tensor(well, dtype=torch.float32, device=self.device),
                torch.as_tensor(seismic, dtype=torch.float32, device=self.device),
                self.well_present,
                self.seismic_present,
            )
        bounded = np.clip(
            prediction.detach().cpu().numpy(), TARGET_BOUNDS[0], TARGET_BOUNDS[1]
        ).astype(np.float32)
        return bounded, gate.detach().cpu().numpy().astype(np.float32)


def fit_head(
    *,
    well_features: np.ndarray,
    seismic_features: np.ndarray,
    target: np.ndarray,
    parent_index: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    well_present: bool,
    seismic_present: bool,
    config: Mapping[str, Any],
    seed: int,
    device: str,
    fixed_steps: int | None = None,
    stream: int = 0,
) -> tuple[FitBundle, np.ndarray | None]:
    _seed_all(seed)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = (
        None if validation_indices is None else np.asarray(validation_indices, dtype=np.int64)
    )
    well_transform = fit_modality_transform(
        well_features, train, parent_index, stream=101 + stream
    )
    seismic_transform = fit_modality_transform(
        seismic_features, train, parent_index, stream=211 + stream
    )
    x_well = well_transform.apply(well_features[train])
    x_seismic = seismic_transform.apply(seismic_features[train])
    y = np.asarray(target[train], dtype=np.float32)
    groups = np.asarray(parent_index[train], dtype=np.int64)
    model = BoundedFusionHead(maximum_gate=float(config["gate_strength"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=float(config["weight_decay"]),
    )
    tw = torch.as_tensor(x_well, dtype=torch.float32, device=device)
    ts = torch.as_tensor(x_seismic, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    tg = torch.as_tensor(groups, dtype=torch.int64, device=device)
    max_updates = int(fixed_steps if fixed_steps is not None else MAX_UPDATES)
    patience_updates = int(config["early_stopping_patience"])
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = math.inf
    best_step = max_updates
    since_best = 0
    first_loss = None
    last_loss = None
    maximum_gradient_norm = 0.0
    for update in range(1, max_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(tw, ts, float(well_present), float(seismic_present))
        loss = _balanced_mse(prediction, ty, tg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("P38 head loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        if validation is not None and (update == 1 or update % EVAL_EVERY == 0):
            provisional = FitBundle(
                model=model,
                well_transform=well_transform,
                seismic_transform=seismic_transform,
                well_present=float(well_present),
                seismic_present=float(seismic_present),
                device=device,
                steps=update,
                audit={},
            )
            val_prediction, _ = provisional.predict(
                well_features, seismic_features, validation
            )
            score = _rmse(target[validation], val_prediction)
            if score < best_rmse - 1e-12:
                best_rmse = score
                best_step = update
                best_state = copy.deepcopy(model.state_dict())
                since_best = 0
            else:
                since_best += EVAL_EVERY
            if since_best >= patience_updates:
                break
    if validation is not None:
        if best_state is None:
            raise RuntimeError("early stopping did not record a finite validation state")
        model.load_state_dict(best_state)
        actual_steps = best_step
    else:
        actual_steps = max_updates
    bundle = FitBundle(
        model=model,
        well_transform=well_transform,
        seismic_transform=seismic_transform,
        well_present=float(well_present),
        seismic_present=float(seismic_present),
        device=device,
        steps=actual_steps,
        audit={
            "train_rows": int(len(train)),
            "train_parent_counts": {
                str(int(parent)): int(np.count_nonzero(groups == parent))
                for parent in np.unique(groups)
            },
            "first_loss": first_loss,
            "last_loss": last_loss,
            "maximum_gradient_norm": maximum_gradient_norm,
            "updates_executed": update,
            "selected_steps": actual_steps,
            "trainable_parameter_count": parameter_count(),
            "frozen_foundation_parameter_count_not_in_head": True,
            "well_source_variance": float(np.var(well_features[train])),
            "seismic_source_variance": float(np.var(seismic_features[train])),
            "well_transformed_variance": float(np.var(x_well)),
            "seismic_transformed_variance": float(np.var(x_seismic)),
        },
    )
    validation_prediction = None
    if validation is not None:
        validation_prediction, _ = bundle.predict(well_features, seismic_features, validation)
    return bundle, validation_prediction


def inner_logo(
    *,
    well_features: np.ndarray,
    seismic_features: np.ndarray,
    target: np.ndarray,
    parent_index: np.ndarray,
    outer_train_parents: Sequence[int],
    well_present: bool,
    seismic_present: bool,
    config: Mapping[str, Any],
    seed: int,
    device: str,
    stream: int,
) -> dict[str, Any]:
    predictions = np.full(len(target), np.nan, dtype=np.float32)
    steps: list[int] = []
    folds: list[dict[str, Any]] = []
    for inner_held in outer_train_parents:
        train = np.flatnonzero(
            np.isin(parent_index, outer_train_parents) & (parent_index != inner_held)
        )
        validation = np.flatnonzero(parent_index == inner_held)
        bundle, predicted = fit_head(
            well_features=well_features,
            seismic_features=seismic_features,
            target=target,
            parent_index=parent_index,
            train_indices=train,
            validation_indices=validation,
            well_present=well_present,
            seismic_present=seismic_present,
            config=config,
            seed=seed + int(inner_held) * 17,
            device=device,
            stream=stream + int(inner_held),
        )
        assert predicted is not None
        predictions[validation] = predicted
        steps.append(bundle.steps)
        folds.append(
            {
                "inner_held_parent_index": int(inner_held),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "rmse": _rmse(target[validation], predicted),
                "selected_steps": bundle.steps,
                "training_audit": dict(bundle.audit),
            }
        )
    selected = np.flatnonzero(np.isin(parent_index, outer_train_parents))
    if not np.all(np.isfinite(predictions[selected])):
        raise RuntimeError("inner LOGO predictions are incomplete")
    per_parent = {
        str(int(parent)): _rmse(
            target[parent_index == parent], predictions[parent_index == parent]
        )
        for parent in outer_train_parents
    }
    return {
        "predictions": predictions,
        "selected_indices": selected,
        "folds": folds,
        "selected_steps": int(np.median(steps)),
        "per_parent_rmse": per_parent,
        "macro_rmse": float(np.mean(list(per_parent.values()))),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("metric arrays must be same-shape and finite")
    error = p - y
    return {
        "rows": int(len(y)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def calibration_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    scale = max(float(sigma), 1e-6)
    error = y - p
    nll = float(np.mean(0.5 * (error / scale) ** 2 + np.log(scale) + 0.5 * np.log(2 * np.pi)))
    z50 = 0.6744897501960817
    z90 = 1.6448536269514722
    return {
        "sigma": scale,
        "gaussian_nll": nll,
        "coverage_50": float(np.mean(np.abs(error) <= z50 * scale)),
        "coverage_90": float(np.mean(np.abs(error) <= z90 * scale)),
        "mean_width_50": float(2.0 * z50 * scale),
        "mean_width_90": float(2.0 * z90 * scale),
    }


def paired_depth_block_bootstrap(
    *,
    target: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    parent_index: np.ndarray,
    md_m: np.ndarray,
    block_m: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    parents = np.unique(parent_index)
    units: dict[int, list[np.ndarray]] = {}
    for parent in parents:
        rows = np.flatnonzero(parent_index == parent)
        block = np.floor((md_m[rows] - np.min(md_m[rows])) / float(block_m)).astype(int)
        units[int(parent)] = [rows[block == value] for value in np.unique(block)]
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        parent_delta = []
        for parent in parents:
            blocks = units[int(parent)]
            sampled = rng.integers(0, len(blocks), size=len(blocks))
            rows = np.concatenate([blocks[index] for index in sampled])
            parent_delta.append(
                _rmse(target[rows], candidate[rows]) - _rmse(target[rows], control[rows])
            )
        deltas[draw] = float(np.mean(parent_delta))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "physical_MD_block_m": float(block_m),
        "blocks_per_parent": {str(key): len(value) for key, value in units.items()},
        "mean_delta": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_candidate_better": float(np.mean(deltas < 0.0)),
    }


def checkpoint_arrays(prefix: str, bundle: FitBundle) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        f"{prefix}__well_mean": bundle.well_transform.mean,
        f"{prefix}__well_scale": bundle.well_transform.scale,
        f"{prefix}__well_projection": bundle.well_transform.projection,
        f"{prefix}__seismic_mean": bundle.seismic_transform.mean,
        f"{prefix}__seismic_scale": bundle.seismic_transform.scale,
        f"{prefix}__seismic_projection": bundle.seismic_transform.projection,
        f"{prefix}__steps": np.asarray(bundle.steps, dtype=np.int32),
        f"{prefix}__well_present": np.asarray(bundle.well_present, dtype=np.float32),
        f"{prefix}__seismic_present": np.asarray(bundle.seismic_present, dtype=np.float32),
    }
    for name, tensor in bundle.model.state_dict().items():
        arrays[f"{prefix}__state__{name}"] = tensor.detach().cpu().numpy()
    return arrays
