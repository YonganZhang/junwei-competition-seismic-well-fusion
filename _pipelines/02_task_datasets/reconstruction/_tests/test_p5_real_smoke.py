"""Explicit data-dependent P5 Stage-1 smoke; frozen test blocks remain closed."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))



def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


p5 = _load_track_module("reconstruction_p5_stage1_real_smoke", "p5_stage1.py")


class P5RealDevelopmentSmokeTest(unittest.TestCase):
    def test_rbf_both_modes_use_real_development_only(self):
        configured = os.environ.get("RECONSTRUCTION_DATA_DIR")
        if not configured:
            raise unittest.SkipTest("set RECONSTRUCTION_DATA_DIR for the explicit P5 real-data gate")
        data_dir = Path(configured)
        if not all((data_dir / name).is_file() for name in ("train.h5", "test.h5")):
            raise unittest.SkipTest("RECONSTRUCTION_DATA_DIR lacks the legacy HDF5 containers")
        with tempfile.TemporaryDirectory() as directory:
            summary = p5.run_stage1(
                modes=("strict", "conditional"),
                models=("scipy_rbf_neighbors",),
                data_dir=data_dir,
                output_root=Path(directory),
                device="cpu",
                max_train_points=48,
                max_validation_points=24,
            )
        self.assertEqual(summary["counts"], {"passed": 2, "skipped": 0, "failed": 0})
        for mode in ("strict", "conditional"):
            result = summary["results"][mode]["scipy_rbf_neighbors"]
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["evidence_status"], "contract_smoked")
            self.assertEqual(result["firewall"]["frozen_test_i_blocks_loaded"], [])
            self.assertTrue(result["real_development"]["finite_prediction"])


if __name__ == "__main__":
    unittest.main()
