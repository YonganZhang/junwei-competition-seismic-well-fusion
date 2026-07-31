"""Frozen first-ten model-by-target suitability matrix and static gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MODEL_ORDER = (
    "xgboost", "catboost", "lightgbm", "autogluon_limited", "inceptiontime",
    "patchtst", "temporal_fusion_transformer", "seg_spatial_tcn", "graphsage",
    "monai_unet3d",
)
TARGET_ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
_RATINGS = {
    "xgboost": ("A", "A", "B", "B", "C", "A", "A"),
    "catboost": ("A", "A", "B", "B", "C", "A", "A"),
    "lightgbm": ("A", "A", "B", "B", "C", "A", "A"),
    "autogluon_limited": ("A", "A", "A", "A", "C", "A", "A"),
    "inceptiontime": ("A", "A", "C", "C", None, "A", "A"),
    "patchtst": ("B", "B", "A", "A", "C", "B", "B"),
    "temporal_fusion_transformer": (None, None, "A", "A", "C", None, None),
    "seg_spatial_tcn": ("A", "B", None, None, "C", "A", "A"),
    "graphsage": ("A", "B", "B", "B", "A", "B", "B"),
    "monai_unet3d": ("A", "A", "C", "C", "A", "A", "A"),
}


@dataclass(frozen=True)
class MatrixGate:
    model_id: str
    target_id: str
    eligible: bool
    rating: str | None
    smoke_contract: str
    reason_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "target_id": self.target_id,
            "eligible": self.eligible, "rating": self.rating,
            "smoke_contract": self.smoke_contract, "reason_code": self.reason_code,
        }


def matrix_gate(model_id: str, target_id: str) -> MatrixGate:
    if model_id not in MODEL_ORDER:
        raise KeyError(f"unknown P5 sweetspot model {model_id!r}")
    if target_id not in TARGET_ORDER:
        raise KeyError(f"unknown P5 sweetspot target {target_id!r}")
    rating = _RATINGS[model_id][TARGET_ORDER.index(target_id)]
    return MatrixGate(
        model_id=model_id,
        target_id=target_id,
        eligible=rating is not None,
        rating=rating,
        smoke_contract=f"S{target_id[1:]}",
        reason_code=None if rating is not None else "matrix_not_applicable",
    )


def matrix_payload() -> dict[str, Any]:
    return {
        "schema_version": "sweetspot-p5-target-matrix/v1",
        "models": list(MODEL_ORDER),
        "targets": list(TARGET_ORDER),
        "cells": [matrix_gate(model, target).to_dict() for model in MODEL_ORDER for target in TARGET_ORDER],
        "meaning": "A/B/C are suitability ratings after label approval, not evidence of an available label or a score",
    }
