"""Frozen TaskSpecs for sweet-spot target 6 porosity label versions."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from ml_framework.contracts import TaskSpec  # noqa: E402


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
FIGURES = (
    "prediction_vs_truth.png",
    "residuals.png",
    "depth_curve.png",
    "distribution_uncertainty.png",
)


def build_phif_task_spec() -> TaskSpec:
    return TaskSpec(
        track_id="sweetspot",
        task_id="target6_porosity_phif",
        task_type="regression",
        input_modalities=("real_ST0202_seismic_patch", "raw_well_log_sequence"),
        targets=("PHIF",),
        units={"PHIF": "fraction"},
        label_version="target6-phif-cpi-v1",
        target_masks={"PHIF": "isfinite(PHIF)"},
        group_keys=("mother_well_family",),
        target_transform={"PHIF": "identity"},
        inverse_transform={"PHIF": "identity"},
        train_loss={"PHIF": "mean_squared_error_on_fold_train_zscore"},
        inference_transform={"PHIF": "clip_identity_prediction_to_[0,1]"},
        threshold_policy={},
        calibration_policy={"uncertainty": "OOF_absolute_residual_q90"},
        primary_metrics=("physical_MAE",),
        secondary_metrics=("physical_RMSE", "physical_R2"),
        guardrail_metrics=("finite_predictions",),
        metric_directions={
            "physical_MAE": "minimize",
            "physical_RMSE": "minimize",
            "physical_R2": "maximize",
            "finite_predictions": "maximize",
        },
        hpo={
            "objective": "development_OOF_physical_MAE",
            "direction": "minimize",
            "test_access": "forbidden",
            "learning_rate": [0.001, 0.01],
            "l2_strength": [0.0, 0.01],
        },
        visualizer_id="target6_porosity_regression",
        required_figures=FIGURES,
        input_whitelist=INPUT_WHITELIST,
        forbidden_inputs=FORBIDDEN_INPUTS,
        metadata={
            "primary_label_version": True,
            "reservoir_interval": "Hugin Top-Base inherited from reservoir real-data build",
            "preferred_frozen_test_family": "15/9-F-15",
            "requested_development_folds": 5,
            "model_plugin_family": "reservoir simple registered models",
        },
    )


def build_phie_task_spec() -> TaskSpec:
    """Independent exact-PHIE case; never aliases PHIF or LFP_PHIE."""
    return TaskSpec(
        track_id="sweetspot",
        task_id="target6_porosity_phie",
        task_type="regression",
        input_modalities=("real_ST0202_seismic_patch", "raw_well_log_sequence"),
        targets=("PHIE",),
        units={"PHIE": "fraction"},
        label_version="target6-exact-phie-v1",
        target_masks={"PHIE": "isfinite(exact_PHIE)"},
        group_keys=("mother_well_family",),
        target_transform={"PHIE": "identity"},
        inverse_transform={"PHIE": "identity"},
        train_loss={"PHIE": "mean_squared_error_on_fold_train_zscore"},
        inference_transform={"PHIE": "clip_identity_prediction_to_[0,1]"},
        threshold_policy={},
        calibration_policy={"uncertainty": "OOF_absolute_residual_q90_if_feasible"},
        primary_metrics=("physical_MAE",),
        secondary_metrics=("physical_RMSE", "physical_R2"),
        guardrail_metrics=("finite_predictions",),
        metric_directions={
            "physical_MAE": "minimize",
            "physical_RMSE": "minimize",
            "physical_R2": "maximize",
            "finite_predictions": "maximize",
        },
        hpo={"objective": "development_OOF_physical_MAE", "direction": "minimize"},
        visualizer_id="target6_porosity_regression",
        required_figures=FIGURES,
        input_whitelist=INPUT_WHITELIST,
        forbidden_inputs=FORBIDDEN_INPUTS,
        metadata={
            "primary_label_version": False,
            "enable_only_if": "exact PHIE covers frozen test plus >=2 development mother-well families",
            "never_alias": ["PHIF", "LFP_PHIE"],
            "mixed_with_PHIF": False,
        },
    )
