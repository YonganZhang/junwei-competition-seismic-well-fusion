"""Contract for target 4, risk of stable reported water within 30 days."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.splits import build_group_folds

from ..common import build_water_breakthrough_dataset


TASK_ID = "sweetspot.water_breakthrough.7day_event_30day_risk_v1"
STATUS = "proxy_feasible"
TEST_GROUP = "NO 15/9-F-15 D"


def task_spec() -> TaskSpec:
    history_fields = (
        "history_7d_bore_oil_vol", "history_7d_bore_gas_vol", "history_7d_bore_wat_vol",
        "history_7d_on_stream_hrs", "history_7d_avg_downhole_pressure",
        "history_7d_avg_choke_size_p", "history_7d_avg_whp_p",
    )
    return TaskSpec(
        track_id="sweetspot", task_id=TASK_ID, task_type="binary",
        input_modalities=("causal_production_history",),
        targets=("WATER_EVENT_WITHIN_30D",), units={"WATER_EVENT_WITHIN_30D": "binary"},
        label_version="first-7-consecutive-positive-water-days-v1",
        target_masks={"WATER_EVENT_WITHIN_30D": "pre_event_history_available_and_not_left_truncated"},
        group_keys=("well_bore_code",),
        target_transform={"WATER_EVENT_WITHIN_30D": "identity"},
        inverse_transform={"WATER_EVENT_WITHIN_30D": "identity"},
        train_loss={"WATER_EVENT_WITHIN_30D": "bce_with_logits"},
        inference_transform={"WATER_EVENT_WITHIN_30D": "sigmoid"},
        threshold_policy={"selection": "OOF_only", "smoke_default": 0.5},
        calibration_policy={"fit_scope": "OOF_only", "method_candidates": ["none", "isotonic"]},
        primary_metrics=("average_precision", "brier"),
        metric_directions={"average_precision": "maximize", "brier": "minimize", "f1": "maximize"},
        secondary_metrics=("f1",),
        time_cutoff={
            "history_days": 7, "risk_horizon_days": 30,
            "event": "first run of 7 consecutive reported days with BORE_WAT_VOL > 0",
            "post_event_inputs_forbidden": True,
        },
        hpo={"model_ids": ["logistic_classifier"], "selection_scope": "development_oof_only"},
        visualizer_id="sweetspot_water_breakthrough",
        required_figures=("risk_timeline", "pr_curve", "calibration", "confusion_by_well"),
        input_whitelist=history_fields,
        forbidden_inputs=("future_BORE_WAT_VOL", "event_date", "days_to_event", "WATER_EVENT_WITHIN_30D"),
        metadata={
            "status": STATUS, "test_group": TEST_GROUP,
            "proxy_warning": "reported positive water event; not a domain-approved water-cut threshold",
        },
    )


def build_dataset_and_manifest(source_root: Path | None = None) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    dataset, evidence = build_water_breakthrough_dataset(source_root)
    manifest = build_group_folds(
        dataset["sample_ids"], dataset["groups"], group_key="well_bore_code",
        test_groups=[TEST_GROUP], requested_n_splits=5, seed=2693,
        max_splits_by_support=3,
        support_reason="only three non-test wells have non-left-truncated histories with both event classes",
        metadata={"task_id": TASK_ID, "time_causal": True, "test_selection": "F-15 frozen"},
    )
    evidence.update({
        "status": STATUS, "requested_n_splits": 5,
        "effective_n_splits": manifest.effective_n_splits,
        "downgrade_reason": manifest.downgrade_reason,
        "proxy_warning": task_spec().metadata["proxy_warning"],
    })
    return dataset, manifest, evidence
