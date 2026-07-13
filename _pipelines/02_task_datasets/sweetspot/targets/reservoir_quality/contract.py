"""Contract for target 1, the continuous reservoir quality index (RQI).

RQI is used instead of an invented weighted sweetspot score.  PHIF and KLOGH
construct the label and are therefore forbidden inference inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.splits import build_group_folds

from ..common import RAW_LOG_FEATURES, flatten_petrophysical_target, load_petrophysical_tables


TASK_ID = "sweetspot.reservoir_quality.rqi_v1"
STATUS = "proxy_feasible"
TEST_FAMILY = "15/9-F-15"


def task_spec() -> TaskSpec:
    return TaskSpec(
        track_id="sweetspot",
        task_id=TASK_ID,
        task_type="regression",
        input_modalities=("well_logs",),
        targets=("RQI",),
        units={"RQI": "sqrt(mD/fraction)"},
        label_version="rqi-0.0314-sqrt-klogh-over-phif-v1",
        target_masks={"RQI": "finite_phif_gt_0_and_klogh_ge_0"},
        group_keys=("well_family",),
        target_transform={"RQI": "log1p"},
        inverse_transform={"RQI": "expm1"},
        train_loss={"RQI": "huber"},
        inference_transform={"RQI": "identity_after_expm1"},
        threshold_policy={},
        calibration_policy={},
        primary_metrics=("mae", "spearman"),
        metric_directions={"mae": "minimize", "spearman": "maximize", "rmse": "minimize"},
        secondary_metrics=("rmse",),
        hpo={"model_ids": ["robust_linear"], "selection_scope": "development_oof_only"},
        visualizer_id="sweetspot_reservoir_quality",
        required_figures=("depth_track", "observed_predicted", "residual_by_well", "rank_by_well"),
        input_whitelist=RAW_LOG_FEATURES,
        forbidden_inputs=("PHIF", "PHIE", "KLOGH", "KLOGV", "RQI", "SAND_FLAG", "SW", "VSH"),
        metadata={
            "status": STATUS,
            "label_semantics": "petrophysical reservoir-quality proxy, not field sweetspot truth",
            "formula": "RQI = 0.0314 * sqrt(KLOGH / PHIF)",
            "test_family": TEST_FAMILY,
        },
    )


def _rqi(labels: dict[str, np.ndarray]) -> np.ndarray:
    phif = np.asarray(labels["PHIF"], dtype=float)
    permeability = np.asarray(labels["KLOGH"], dtype=float)
    valid = np.isfinite(phif) & np.isfinite(permeability) & (phif > 0.0) & (permeability >= 0.0)
    result = np.full(phif.shape, np.nan, dtype=float)
    result[valid] = 0.0314 * np.sqrt(permeability[valid] / phif[valid])
    return result


def build_dataset_and_manifest(source_root: Path | None = None) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    tables, evidence = load_petrophysical_tables(source_root)
    dataset = flatten_petrophysical_target(tables, target_name="rqi", target_fn=_rqi)
    manifest = build_group_folds(
        dataset["sample_ids"], dataset["groups"], group_key="well_family",
        test_groups=[TEST_FAMILY], requested_n_splits=5,
        seed=2693,
        metadata={"task_id": TASK_ID, "test_selection": "F-15 frozen before model selection"},
    )
    evidence.update({
        "status": STATUS,
        "valid_samples": len(dataset["sample_ids"]),
        "requested_n_splits": 5,
        "effective_n_splits": manifest.effective_n_splits,
        "downgrade_reason": manifest.downgrade_reason,
    })
    return dataset, manifest, evidence
