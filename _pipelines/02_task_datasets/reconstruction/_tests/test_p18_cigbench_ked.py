from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

SPEC = importlib.util.spec_from_file_location(
    "p18_cigbench_ked", PIPELINE_DIR / "p18_cigbench_ked.py"
)
assert SPEC is not None and SPEC.loader is not None
p18 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p18
SPEC.loader.exec_module(p18)

from _models.reconstruction import cigbench_rgt  # noqa: E402


def test_package_adapter_is_hash_locked_mit_rgt() -> None:
    audit = cigbench_rgt.verify_package()
    caps = cigbench_rgt.capabilities()
    assert audit["version"] == "0.2.0"
    assert audit["license"] == "MIT"
    assert caps["role"].startswith("target-free external drift")
    assert caps["auto_download"] is True


def test_sample_rgt_uses_exact_kji_and_rejects_bounds() -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    indices = np.asarray([[0, 0, 0], [2, 3, 4], [1, 2, 3]])
    np.testing.assert_array_equal(
        p18.sample_rgt(volume, indices), np.asarray([0.0, 59.0, 33.0])
    )
    with pytest.raises(IndexError):
        p18.sample_rgt(volume, np.asarray([[3, 0, 0]]))


def test_ked_specified_drift_smoke_is_finite() -> None:
    rng = np.random.default_rng(2693)
    train_xyz = rng.uniform(size=(24, 3))
    validation_xyz = rng.uniform(size=(7, 3))
    train_rgt = train_xyz[:, 2] ** 2 + 0.1 * train_xyz[:, 0]
    validation_rgt = (
        validation_xyz[:, 2] ** 2 + 0.1 * validation_xyz[:, 0]
    )
    train_target = 0.2 + 0.7 * train_rgt + 0.01 * rng.normal(size=24)
    prediction, audit = p18.fit_predict_ked(
        train_xyz,
        train_target,
        validation_xyz,
        train_rgt,
        validation_rgt,
    )
    assert prediction.shape == (7,)
    assert np.all(np.isfinite(prediction))
    assert audit["drift_terms"] == ["specified:RGT"]
    assert audit["variogram_model"] == "linear"
    assert audit["nlags"] == 4


def test_cli_has_no_holdout_surface_argument() -> None:
    help_text = p18._parser().format_help()  # noqa: SLF001
    source = (PIPELINE_DIR / "p18_cigbench_ked.py").read_text(encoding="utf-8")
    assert "test.h5" not in source.lower()
    assert "--test" not in help_text
    assert p18._shape("128,256,256") == (128, 256, 256)  # noqa: SLF001
