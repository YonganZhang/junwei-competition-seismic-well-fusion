from __future__ import annotations

import importlib
import os
import json
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from p4_pipeline import (  # noqa: E402
    POROSITY_PHIF,
    build_independent_manifest,
    load_samples,
    scan_real_sources,
    select_frozen_test_family,
    tiny_development_check,
)
from ml_framework.artifacts import hash_file  # noqa: E402
from ml_framework.checkpoint import load_checkpoint  # noqa: E402
build_phif_task_spec = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.targets.porosity.task_spec"
).build_phif_task_spec


pytestmark = pytest.mark.integration


def test_real_p4_scan_split_and_tiny_development_smoke() -> None:
    processed = os.environ.get("RESERVOIR_PROCESSED_DIR")
    guard = os.environ.get("RESERVOIR_GUARD_PATH")
    if not processed or not guard:
        pytest.skip("set RESERVOIR_PROCESSED_DIR and RESERVOIR_GUARD_PATH for the real P4 smoke")
    spec = build_phif_task_spec()
    indexed = scan_real_sources(
        processed_dir=Path(processed),
        guard_path=Path(guard),
        task_id=spec.task_id,
        definition=POROSITY_PHIF,
    )
    test_family, selection = select_frozen_test_family(indexed)
    manifest = build_independent_manifest(
        indexed,
        test_family=test_family,
        target_name="PHIF",
        label_version=spec.label_version,
        seed=2693,
        selection_record=selection,
    )
    by_id = {record.sample_id: record for record in indexed if record.label_valid}
    development = load_samples([by_id[sid] for sid in manifest.development_sample_ids], POROSITY_PHIF)
    smoke = tiny_development_check(
        development,
        task_spec=spec,
        model_name="reservoir_ridge",
        seed=2693,
    )
    assert test_family == "15/9-F-15"
    assert manifest.effective_n_splits == 4
    assert len(development) == 1216
    assert smoke["status"] == "passed"


def test_archived_target6_and_target7_outputs_are_complete_and_self_consistent() -> None:
    roots = (
        PROJECT_ROOT / "_pipelines/02_task_datasets/sweetspot/targets/porosity/_outputs/phif",
        PROJECT_ROOT / "_pipelines/02_task_datasets/sweetspot/targets/permeability/_outputs/klogh",
    )
    if not all((root / "manifest.json").is_file() for root in roots):
        pytest.skip("run both target P4 baselines before archived-output integration checks")
    for root in roots:
        status = json.loads((root / "status.json").read_text())
        split = json.loads((root / "split_manifest.json").read_text())
        lifecycle = json.loads((root / "lifecycle.json").read_text())
        metrics = json.loads((root / "frozen_test/metrics.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        assert status["status"] == "complete"
        assert status["test_consumed_once"] is True
        assert split["test_groups"] == ["15/9-F-15"]
        assert split["requested_n_splits"] == 5
        assert split["effective_n_splits"] == 4
        assert lifecycle["state"] == "VERIFIED"
        assert lifecycle["test_consumed_at"]
        assert np.isfinite(metrics["physical"]["MAE"])
        assert np.isfinite(metrics["physical"]["RMSE"])
        checkpoint = load_checkpoint(root / "refit/checkpoint_best.pkl")
        assert checkpoint["split_hash"]
        assert checkpoint["config_hash"]
        for name in (
            "prediction_vs_truth.png",
            "residuals.png",
            "depth_curve.png",
            "distribution_uncertainty.png",
        ):
            assert (root / "visualizations" / name).stat().st_size > 10_000
        for relative, record in manifest["artifacts"].items():
            assert hash_file(root / relative) == record["sha256"]
