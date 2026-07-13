"""Contract for target 2 using SAND_FLAG as an explicit proxy.

The released interpretation does not provide an approved PAY/RES flag across
the supervised wells.  This task therefore predicts interpreted sand/net
reservoir, not hydrocarbon presence.  The name is kept to preserve the seven-
target product contract while the proxy status remains machine-readable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.splits import build_group_folds

from ..common import RAW_LOG_FEATURES, flatten_petrophysical_target, load_petrophysical_tables


TASK_ID = "sweetspot.hydrocarbon_pay.sand_flag_proxy_v1"
STATUS = "proxy_feasible"
TEST_FAMILY = "15/9-F-15"


def task_spec() -> TaskSpec:
    return TaskSpec(
        track_id="sweetspot",
        task_id=TASK_ID,
        task_type="binary",
        input_modalities=("well_logs",),
        targets=("SAND_FLAG_PROXY",),
        units={"SAND_FLAG_PROXY": "binary"},
        label_version="cpi-sand-flag-near-binary-v1",
        target_masks={"SAND_FLAG_PROXY": "finite_and_within_0.01_of_0_or_1"},
        group_keys=("well_family",),
        target_transform={"SAND_FLAG_PROXY": "identity"},
        inverse_transform={"SAND_FLAG_PROXY": "identity"},
        train_loss={"SAND_FLAG_PROXY": "bce_with_logits"},
        inference_transform={"SAND_FLAG_PROXY": "sigmoid"},
        threshold_policy={"selection": "OOF_only", "smoke_default": 0.5},
        calibration_policy={"fit_scope": "OOF_only", "method_candidates": ["none", "isotonic"]},
        primary_metrics=("average_precision",),
        metric_directions={
            "average_precision": "maximize", "f1": "maximize",
            "net_thickness_mae_m": "minimize", "brier": "minimize",
        },
        secondary_metrics=("f1", "net_thickness_mae_m", "brier"),
        hpo={"model_ids": ["logistic_classifier"], "selection_scope": "development_oof_only"},
        visualizer_id="sweetspot_hydrocarbon_pay_proxy",
        required_figures=("depth_probability_track", "pr_curve", "confusion_matrix", "net_thickness_by_well"),
        input_whitelist=RAW_LOG_FEATURES,
        forbidden_inputs=("SAND_FLAG", "SAND_FLAG_PROXY", "PERF_FLAG", "PHIF", "KLOGH", "SW", "VSH"),
        metadata={
            "status": STATUS,
            "proxy_warning": "SAND_FLAG is a net-reservoir/sand proxy, not direct hydrocarbon-pay truth",
            "test_family": TEST_FAMILY,
        },
    )


def _sand_flag(labels: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.asarray(labels.get("SAND_FLAG", np.array([], dtype=float)), dtype=float)
    if raw.size == 0:
        return raw
    rounded = np.rint(raw)
    valid = np.isfinite(raw) & (np.abs(raw - rounded) <= 0.01) & np.isin(rounded, [0.0, 1.0])
    result = np.full(raw.shape, np.nan, dtype=float)
    result[valid] = rounded[valid]
    return result


def build_dataset_and_manifest(source_root: Path | None = None) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    tables, evidence = load_petrophysical_tables(source_root)
    tables = [table for table in tables if "SAND_FLAG" in table["labels"]]
    dataset = flatten_petrophysical_target(tables, target_name="sand-flag", target_fn=_sand_flag)
    manifest = build_group_folds(
        dataset["sample_ids"], dataset["groups"], group_key="well_family",
        test_groups=[TEST_FAMILY], requested_n_splits=5, seed=2693,
        metadata={"task_id": TASK_ID, "proxy_label": True, "test_selection": "F-15 frozen"},
    )
    positives = int(np.asarray(dataset["target"], dtype=int).sum())
    evidence.update({
        "status": STATUS,
        "valid_samples": len(dataset["sample_ids"]),
        "positive_samples": positives,
        "requested_n_splits": 5,
        "effective_n_splits": manifest.effective_n_splits,
        "downgrade_reason": manifest.downgrade_reason,
        "proxy_warning": task_spec().metadata["proxy_warning"],
    })
    return dataset, manifest, evidence
