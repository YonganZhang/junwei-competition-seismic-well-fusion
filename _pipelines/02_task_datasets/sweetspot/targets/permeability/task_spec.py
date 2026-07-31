"""Frozen TaskSpec for sweet-spot target 7 horizontal permeability."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from ml_framework.contracts import TaskSpec  # noqa: E402


def build_task_spec() -> TaskSpec:
    return TaskSpec(
        track_id="sweetspot",
        task_id="target7_permeability_klogh",
        task_type="regression",
        input_modalities=("real_ST0202_seismic_patch", "raw_well_log_sequence"),
        targets=("KLOGH",),
        units={"KLOGH": "mD"},
        label_version="target7-klogh-cpi-v1",
        target_masks={"KLOGH": "isfinite(KLOGH) and KLOGH>=0"},
        group_keys=("mother_well_family",),
        target_transform={"KLOGH": "log1p(KLOGH_mD)"},
        inverse_transform={"KLOGH": "KLOGH_mD=expm1(log1p_prediction)"},
        train_loss={"KLOGH": "mean_squared_error_on_fold_train_zscore_of_log1p"},
        inference_transform={"KLOGH": "expm1(clamp(log1p_prediction,0))"},
        threshold_policy={},
        calibration_policy={"uncertainty": "OOF_absolute_residual_q90_in_log1p_domain"},
        primary_metrics=("physical_MAE",),
        secondary_metrics=(
            "physical_RMSE",
            "physical_R2",
            "log1p_MAE",
            "log1p_RMSE",
            "log1p_R2",
        ),
        guardrail_metrics=("finite_nonnegative_physical_predictions",),
        metric_directions={
            "physical_MAE": "minimize",
            "physical_RMSE": "minimize",
            "physical_R2": "maximize",
            "log1p_MAE": "minimize",
            "log1p_RMSE": "minimize",
            "log1p_R2": "maximize",
            "finite_nonnegative_physical_predictions": "maximize",
        },
        hpo={
            "objective": "development_OOF_physical_MAE",
            "direction": "minimize",
            "test_access": "forbidden",
            "learning_rate": [0.001, 0.01],
            "l2_strength": [0.0, 0.01],
        },
        visualizer_id="target7_permeability_regression",
        required_figures=(
            "prediction_vs_truth.png",
            "residuals.png",
            "depth_curve.png",
            "distribution_uncertainty.png",
        ),
        input_whitelist=(
            "ST0202_seismic_patch",
            "GR",
            "RT",
            "NPHI",
            "RHOB",
            "GR_observed_mask",
            "RT_observed_mask",
            "NPHI_observed_mask",
            "RHOB_observed_mask",
        ),
        forbidden_inputs=(
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
        ),
        metadata={
            "reservoir_interval": "Hugin Top-Base inherited from reservoir real-data build",
            "preferred_frozen_test_family": "15/9-F-15",
            "requested_development_folds": 5,
            "physical_unit": "millidarcy (mD)",
            "KLOGH_NEW_excluded": True,
            "model_plugin_family": "reservoir simple registered models",
        },
    )
