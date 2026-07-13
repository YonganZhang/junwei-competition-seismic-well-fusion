"""Explicit multiclass facies losses with a raw-logit model contract."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


LOSS_IDS = (
    "cross_entropy",
    "focal",
    "cross_entropy_plus_dice",
    "cross_entropy_plus_lovasz",
)


@dataclass(frozen=True)
class LossConfig:
    loss_id: str = "cross_entropy"
    focal_gamma: float = 2.0
    compound_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.loss_id not in LOSS_IDS:
            raise ValueError(f"unsupported facies loss {self.loss_id!r}")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be >=0")
        if not 0.0 <= self.compound_weight <= 1.0:
            raise ValueError("compound_weight must be in [0, 1]")


def softmax_probabilities(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Convert archived/inference logits to probabilities outside the model head."""
    if logits.ndim != 4:
        raise ValueError(f"facies logits must be [B,C,H,W], got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("facies logits contain NaN/Inf")
    if not temperature > 0:
        raise ValueError("temperature must be >0")
    probabilities = torch.softmax(logits.float() / float(temperature), dim=1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("softmax probabilities contain NaN/Inf")
    return probabilities


def _weighted_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    log_pt = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    alpha = class_weights[target]
    return (-(1.0 - pt).pow(gamma) * log_pt * alpha).mean()


def _generalized_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = softmax_probabilities(logits)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    class_volume = one_hot.sum(dim=(0, 2, 3))
    present = class_volume > 0
    if not present.any():
        raise ValueError("dice batch contains no valid facies pixels")
    weights = 1.0 / (class_volume[present].square() + epsilon)
    intersection = (probabilities[:, present] * one_hot[:, present]).sum(dim=(0, 2, 3))
    denominator = (probabilities[:, present] + one_hot[:, present]).sum(dim=(0, 2, 3))
    dice = (2.0 * (weights * intersection).sum() + epsilon) / (
        (weights * denominator).sum() + epsilon
    )
    return 1.0 - dice


def _lovasz_gradient(sorted_ground_truth: torch.Tensor) -> torch.Tensor:
    count = sorted_ground_truth.numel()
    total_positive = sorted_ground_truth.sum()
    intersection = total_positive - sorted_ground_truth.float().cumsum(0)
    union = total_positive + (1.0 - sorted_ground_truth.float()).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-12)
    if count > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


def _lovasz_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    probabilities = softmax_probabilities(logits).permute(0, 2, 3, 1).reshape(-1, num_classes)
    flattened_target = target.reshape(-1)
    losses: list[torch.Tensor] = []
    for class_id in range(num_classes):
        foreground = (flattened_target == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_id]).abs()
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_foreground = foreground[permutation]
        losses.append(torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground)))
    if not losses:
        raise ValueError("Lovasz batch contains no configured classes")
    return torch.stack(losses).mean()


class FaciesSegmentationLoss(nn.Module):
    """One explicit loss adapter; the underlying model always returns logits."""

    def __init__(
        self,
        *,
        num_classes: int,
        class_weights: torch.Tensor,
        config: LossConfig,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("facies segmentation requires at least two classes")
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.shape != (num_classes,) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("class_weights must be finite positive [C]")
        self.num_classes = num_classes
        self.config = config
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected logits [B,{self.num_classes},H,W], got {tuple(logits.shape)}"
            )
        if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError(f"target shape {tuple(target.shape)} is not aligned with logits")
        if target.dtype != torch.long:
            raise ValueError("facies targets must use torch.long class IDs")
        if not torch.isfinite(logits).all():
            raise ValueError("loss received NaN/Inf logits")
        if int(target.min()) < 0 or int(target.max()) >= self.num_classes:
            raise ValueError("target IDs violate the configured facies schema")

        ce = F.cross_entropy(logits, target, weight=self.class_weights)
        if self.config.loss_id == "cross_entropy":
            return ce
        if self.config.loss_id == "focal":
            return _weighted_focal_loss(
                logits, target, self.class_weights, self.config.focal_gamma
            )
        if self.config.loss_id == "cross_entropy_plus_dice":
            auxiliary = _generalized_dice_loss(logits, target, self.num_classes)
        elif self.config.loss_id == "cross_entropy_plus_lovasz":
            auxiliary = _lovasz_softmax_loss(logits, target, self.num_classes)
        else:
            raise AssertionError(f"unreachable loss {self.config.loss_id}")
        weight = self.config.compound_weight
        return (1.0 - weight) * ce + weight * auxiliary


def build_loss(
    loss_id: str,
    *,
    num_classes: int,
    class_weights: torch.Tensor,
    focal_gamma: float = 2.0,
    compound_weight: float = 0.5,
) -> FaciesSegmentationLoss:
    return FaciesSegmentationLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        config=LossConfig(
            loss_id=loss_id,
            focal_gamma=focal_gamma,
            compound_weight=compound_weight,
        ),
    )
