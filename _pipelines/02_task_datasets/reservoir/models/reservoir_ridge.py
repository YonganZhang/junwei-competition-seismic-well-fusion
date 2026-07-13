"""L2-regularized multi-output linear SGD baseline for reservoir targets."""
from __future__ import annotations

from ml_framework.model_registry import register_model

from models.reservoir_linear import ReservoirLinearSGD


class ReservoirRidgeSGD(ReservoirLinearSGD):
    """Linear SGD with an L2 penalty on the regression weights."""

    def __init__(
        self,
        *args: object,
        l2_strength: float = 1e-3,
        weight_decay: float | None = None,
        **kwargs: object,
    ) -> None:
        if weight_decay is not None:
            l2_strength = float(weight_decay)
        super().__init__(
            *args,
            l2_strength=l2_strength,
            weight_decay=weight_decay,
            **kwargs,
        )


@register_model("reservoir_ridge")
def build_model(**kwargs) -> ReservoirRidgeSGD:
    return ReservoirRidgeSGD(**kwargs)
