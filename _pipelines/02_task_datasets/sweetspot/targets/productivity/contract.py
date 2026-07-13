"""Contract for target 3, future 30-day mean oil production."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.splits import build_group_folds

from ..common import build_productivity_dataset


TASK_ID = "sweetspot.productivity.future_30d_oil_v1"
STATUS = "feasible"
TEST_GROUP = "NO 15/9-F-15 D"


def task_spec() -> TaskSpec:
    history_fields = (
        "history_30d_bore_oil_vol", "history_30d_bore_gas_vol", "history_30d_bore_wat_vol",
        "history_30d_on_stream_hrs", "history_30d_avg_downhole_pressure",
        "history_30d_avg_choke_size_p", "history_30d_avg_whp_p",
    )
    return TaskSpec(
        track_id="sweetspot", task_id=TASK_ID, task_type="regression",
        input_modalities=("causal_production_history",),
        targets=("FUTURE_30D_MEAN_OIL",),
        units={"FUTURE_30D_MEAN_OIL": "Sm3/day"},
        label_version="future-30-calendar-row-mean-oil-v1",
        target_masks={"FUTURE_30D_MEAN_OIL": "at_least_21_finite_future_days"},
        group_keys=("well_bore_code",),
        target_transform={"FUTURE_30D_MEAN_OIL": "log1p"},
        inverse_transform={"FUTURE_30D_MEAN_OIL": "expm1"},
        train_loss={"FUTURE_30D_MEAN_OIL": "huber"},
        inference_transform={"FUTURE_30D_MEAN_OIL": "nonnegative_after_expm1"},
        threshold_policy={}, calibration_policy={},
        primary_metrics=("mae", "spearman"),
        metric_directions={"mae": "minimize", "spearman": "maximize", "rmse": "minimize", "topk_hit": "maximize"},
        secondary_metrics=("rmse", "topk_hit"),
        time_cutoff={"history_days": 30, "forecast_days": 30, "features_end_before_cutoff": True},
        hpo={"model_ids": ["robust_linear"], "selection_scope": "development_oof_only"},
        visualizer_id="sweetspot_productivity",
        required_figures=("time_series_forecast", "observed_predicted", "residual_by_well", "topk_ranking"),
        input_whitelist=history_fields,
        forbidden_inputs=("future_BORE_OIL_VOL", "future_BORE_GAS_VOL", "future_BORE_WAT_VOL", "FUTURE_30D_MEAN_OIL"),
        metadata={"status": STATUS, "test_group": TEST_GROUP, "sample_stride_days": 7},
    )


def build_dataset_and_manifest(source_root: Path | None = None) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    dataset, evidence = build_productivity_dataset(source_root)
    manifest = build_group_folds(
        dataset["sample_ids"], dataset["groups"], group_key="well_bore_code",
        test_groups=[TEST_GROUP], requested_n_splits=5, seed=2693,
        metadata={"task_id": TASK_ID, "time_causal": True, "test_selection": "F-15 frozen"},
    )
    evidence.update({
        "status": STATUS, "requested_n_splits": 5,
        "effective_n_splits": manifest.effective_n_splits,
        "downgrade_reason": manifest.downgrade_reason,
    })
    return dataset, manifest, evidence
