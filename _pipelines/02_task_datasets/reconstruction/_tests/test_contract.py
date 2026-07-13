"""Portable reconstruction contract tests; no HDF5, Layer1, or checkpoint required."""
from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
MODES = ("conditional", "strict")


class PortableContractTest(unittest.TestCase):
    def test_gitignore_excludes_noncanonical_large_assets(self):
        text = (HERE / ".gitignore").read_text()
        for pattern in (
            "_tmp/",
            "**/__pycache__/",
            "*.h5",
            "*.ckpt",
            ".venv/",
            "site-packages/",
            "results.json",
            "_outputs/prediction_visualization.png",
        ):
            self.assertIn(pattern, text)
        self.assertNotIn("results_*.json", text)
        self.assertNotIn("*.png", text)

    def test_canonical_mode_results_preserve_evaluation_contract(self):
        expected = {
            "conditional": {
                "train_i_blocks": [0, 1, 2, 3],
                "guard_i_blocks": [],
                "test_i_blocks": [4, 5],
                "n_train_region_well_constraints": 1,
                "n_guard_region_well_constraints": 0,
                "n_test_region_well_constraints": 90,
                "n_well_constraints_supplied_to_idw": 91,
                "test_well_constraints_supplied_to_idw": True,
                "strict_spatial_holdout_generalization": False,
            },
            "strict": {
                "train_i_blocks": [4, 5],
                "guard_i_blocks": [3],
                "test_i_blocks": [0, 1, 2],
                "n_train_region_well_constraints": 90,
                "n_guard_region_well_constraints": 1,
                "n_test_region_well_constraints": 0,
                "n_well_constraints_supplied_to_idw": 90,
                "test_well_constraints_supplied_to_idw": False,
                "strict_spatial_holdout_generalization": True,
            },
        }
        for mode in MODES:
            results = json.loads((HERE / f"results_{mode}.json").read_text())
            self.assertEqual(results["evaluation_mode"], mode)
            audit = results["leakage_checks"]["protocol"]
            self.assertTrue(audit["spatial_regions_zero_overlap"])
            for key, value in expected[mode].items():
                self.assertEqual(audit[key], value)
            for metrics in results["models"].values():
                for key in ("rmse", "mae", "pearson_r", "r2"):
                    value = metrics[key]
                    if value is None:
                        self.assertFalse(metrics[f"{key}_defined"])
                        self.assertTrue(metrics[f"{key}_reason"])
                    else:
                        self.assertTrue(math.isfinite(value))

    def test_manifest_and_result_paths_are_project_relative(self):
        for mode in MODES:
            manifest = json.loads((HERE / f"run_manifest_{mode}.json").read_text())
            self.assertEqual(manifest["evaluation_mode"], mode)
            for section in ("source_sha256", "input_data_sha256", "artifact_sha256"):
                for path_text in manifest[section]:
                    path = Path(path_text)
                    self.assertFalse(path.is_absolute(), path_text)
                    self.assertNotIn("." + "claude/worktrees", path_text)
            results = json.loads((HERE / f"results_{mode}.json").read_text())
            for variant in ("structural", "seismic"):
                for key in (
                    "best_checkpoint",
                    "last_checkpoint",
                    "history",
                    "loss_curve",
                    "loss_curve_best_epoch_zoom",
                ):
                    self.assertFalse(Path(results["training"][variant][key]).is_absolute())

    def test_executable_sources_and_readmes_have_no_host_paths(self):
        forbidden = re.compile(
            "/" + "mnt/data/|/" + "home/|\\." + "claude/worktrees|file" + "://"
        )
        paths = list(HERE.glob("*.py")) + list(HERE.rglob("*.md")) + list(
            (HERE / "models").glob("*.py")
        )
        for path in paths:
            self.assertIsNone(forbidden.search(path.read_text()), str(path))

    def test_source_uses_unified_dataset_and_shared_framework_contract(self):
        build = (HERE / "build_dataset.py").read_text()
        baseline = (HERE / "baseline.py").read_text()
        integration_test = (HERE / "_tests/test_dual_evaluation.py").read_text()
        self.assertIn('save_split("reconstruction", "train", train)', build)
        self.assertIn('save_split("reconstruction", "test", test)', build)
        self.assertIn('load_dataset("reconstruction", source_split)', baseline)
        self.assertIn("train_batches_fn=lambda:", baseline)
        self.assertIn("val_batches_fn=lambda:", baseline)
        self.assertIn('models_package="models"', baseline)
        self.assertIn("denoise_identity", baseline)
        self.assertIn("plot_loss_curve", baseline)
        self.assertNotIn('["split_strategy"]', baseline)
        self.assertIn("raise unittest.SkipTest", integration_test)
        self.assertIn("data-dependent reconstruction integration gate skipped", integration_test)


if __name__ == "__main__":
    unittest.main()
