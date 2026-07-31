from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from data_pipeline import BUILD_REPORT_PATH, GUARD_PATH, load_guard  # noqa: E402
from dataset_io import load_dataset  # noqa: E402


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_real_artifacts() -> None:
    required = [
        PROJECT_ROOT / "_data/processed/reservoir/train.h5",
        PROJECT_ROOT / "_data/processed/reservoir/test.h5",
        GUARD_PATH,
        BUILD_REPORT_PATH,
        HERE / "_outputs/run_manifest.json",
        HERE / "_outputs/normalization.json",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        pytest.skip(
            "integration/data-dependent artifacts missing: "
            + ", ".join(missing)
            + "; run build_dataset.py then train_baseline.py before --run-integration"
        )


def test_real_outputs_are_nonempty_finite_and_family_isolated() -> None:
    parts = {
        "train": list(load_dataset("reservoir", "train")),
        "guard": load_guard(GUARD_PATH),
        "test": list(load_dataset("reservoir", "test")),
    }
    assert all(parts.values())
    families = {
        name: {sample["meta"]["family_id"] for sample in samples}
        for name, samples in parts.items()
    }
    assert not (families["train"] & families["guard"])
    assert not (families["train"] & families["test"])
    assert not (families["guard"] & families["test"])
    for partition, samples in parts.items():
        for sample in samples:
            assert sample["meta"]["partition"] == partition
            assert np.isfinite(sample["seismic_patch"]).all()
            assert np.isfinite(sample["well_log_seq"]).all()
            assert np.isfinite(sample["label"]).all()
            assert sample["label"].shape == (3,)


def test_real_audit_manifests_exclude_test_from_fit_and_training() -> None:
    build = json.loads(BUILD_REPORT_PATH.read_text())
    run = json.loads((HERE / "_outputs/run_manifest.json").read_text())
    normalization = json.loads((HERE / "_outputs/normalization.json").read_text())
    assert build["real_data"] is True
    assert build["family_zero_overlap"] is True
    assert build["test_excluded_from_training_and_statistics"] is True
    assert run["normalization_fit_sources"] == ["train"]
    assert run["guard_used_for_val_loss_only"] is True
    assert run["test_loaded_after_best_checkpoint"] is True
    assert run["test_used_in_training_or_statistics"] is False
    assert normalization["fit_source"] == "train families only"
    assert normalization["max_normalization_roundtrip_error"] <= 1e-8
    assert normalization["max_log1p_roundtrip_error"] <= 1e-8
