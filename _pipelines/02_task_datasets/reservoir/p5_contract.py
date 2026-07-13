"""Frozen three-target TaskSpec for P5 reservoir-property model adapters."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import TaskSpec


TARGETS = ("PHIF", "KLOGH", "SW")
TARGET_INDICES = {"PHIF": 0, "KLOGH": 1, "SW": 2}
INPUT_WHITELIST = (
    "ST0202_seismic_patch",
    "GR",
    "RT",
    "NPHI",
    "RHOB",
    "GR_observed_mask",
    "RT_observed_mask",
    "NPHI_observed_mask",
    "RHOB_observed_mask",
)
FORBIDDEN_INPUTS = (
    "PHIF",
    "PHIE",
    "LFP_PHIE",
    "KLOGH",
    "KLOGH_NEW",
    "KLOGV",
    "SW",
    "BVW",
    "SWIRR",
    "VSH",
    "LFP_VSH",
)


def build_task_spec() -> TaskSpec:
    return TaskSpec(
        track_id="property",
        task_id="reservoir_property_p5_three_independent_targets",
        task_type="regression",
        input_modalities=("real_ST0202_seismic_patch", "raw_well_log_sequence", "tabular_flat_view"),
        targets=TARGETS,
        units={"PHIF": "fraction", "KLOGH": "mD", "SW": "fraction"},
        label_version="property-phif-klogh-sw-cpi-v1",
        target_masks={
            "PHIF": "isfinite(PHIF)",
            "KLOGH": "isfinite(KLOGH) and KLOGH>=0 before log1p",
            "SW": "isfinite(SW)",
        },
        group_keys=("mother_well_family",),
        target_transform={"PHIF": "identity", "KLOGH": "log1p(KLOGH_mD)", "SW": "identity"},
        inverse_transform={"PHIF": "identity", "KLOGH": "KLOGH_mD=expm1(log1p_output)", "SW": "identity"},
        train_loss={
            target: "independently_masked_mean_squared_error_in_model_domain" for target in TARGETS
        },
        inference_transform={
            "PHIF": "preserve_raw_then_clip_physical_view_to_[0,1]",
            "KLOGH": "preserve_raw_then_expm1(clamp(log1p_output,0))",
            "SW": "preserve_raw_then_clip_physical_view_to_[0,1]",
        },
        threshold_policy={},
        calibration_policy={target: "none_at_stage1" for target in TARGETS},
        primary_metrics=("physical_MAE_macro",),
        secondary_metrics=(
            "PHIF_MAE",
            "PHIF_RMSE",
            "KLOGH_MAE_mD",
            "KLOGH_RMSE_mD",
            "KLOGH_log1p_RMSE",
            "SW_MAE",
            "SW_RMSE",
        ),
        guardrail_metrics=("finite_raw_outputs", "independent_target_valid_counts"),
        metric_directions={
            "physical_MAE_macro": "minimize",
            "PHIF_MAE": "minimize",
            "PHIF_RMSE": "minimize",
            "KLOGH_MAE_mD": "minimize",
            "KLOGH_RMSE_mD": "minimize",
            "KLOGH_log1p_RMSE": "minimize",
            "SW_MAE": "minimize",
            "SW_RMSE": "minimize",
            "finite_raw_outputs": "maximize",
            "independent_target_valid_counts": "maximize",
        },
        hpo={
            "stage1": "disabled",
            "future_objective": "development_LOGO4_OOF_physical_MAE_macro",
            "direction": "minimize",
            "test_access": "forbidden",
        },
        visualizer_id="reservoir_property_three_target",
        required_figures=(
            "per_well_depth_curves.png",
            "prediction_vs_truth.png",
            "residual_spatial_error.png",
            "physical_range_and_uncertainty.png",
        ),
        input_whitelist=INPUT_WHITELIST,
        forbidden_inputs=FORBIDDEN_INPUTS,
        metadata={
            "reservoir_interval": "Hugin Top-Base inherited from the real reservoir build",
            "development_groups": ["15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12"],
            "frozen_test_family": "15/9-F-15",
            "stage1_test_loader": "forbidden_and_not_implemented",
            "targets_are_never_cofiltered": True,
            "KLOGH_NEW_excluded": True,
            "LFP_PHIE_excluded": True,
            "raw_outputs_preserved_before_physical_bounds": True,
        },
    )


def physical_to_model_domain(target: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if target == "KLOGH":
        if np.any(array < 0):
            raise ValueError("KLOGH physical values must be nonnegative")
        return np.log1p(array)
    if target in {"PHIF", "SW"}:
        return array.copy()
    raise KeyError(target)


def model_to_physical(target: str, values: Any, *, prediction: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if target == "KLOGH":
        if prediction:
            array = np.maximum(array, 0.0)
        return np.expm1(array)
    if target in {"PHIF", "SW"}:
        return np.clip(array, 0.0, 1.0) if prediction else array.copy()
    raise KeyError(target)


def output_domains(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if tuple(raw) != TARGETS:
        raise ValueError(f"raw output keys/order must be {TARGETS}")
    return {target: model_to_physical(target, raw[target], prediction=True) for target in TARGETS}
