"""Regression tests for reconstruction conditional/strict evaluation contracts."""
from __future__ import annotations

import json
import hashlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from dataset_io import load_dataset  # noqa: E402

import baseline  # noqa: E402
import visualize_prediction  # noqa: E402


def _required_integration_assets() -> list[Path]:
    """Assets intentionally excluded from Git but required for real-data checks."""
    required = [
        PROJECT_ROOT / "_data/processed/reconstruction/train.h5",
        PROJECT_ROOT / "_data/processed/reconstruction/test.h5",
        PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz",
        PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/well_tie_weak.npz",
    ]
    for mode in baseline.EVALUATION_MODES:
        results_path = HERE / f"results_{mode}.json"
        if not results_path.exists():
            required.append(results_path)
            continue
        results = json.loads(results_path.read_text())
        for variant in ("structural", "seismic"):
            for key in ("best_checkpoint", "last_checkpoint"):
                required.append(PROJECT_ROOT / results["training"][variant][key])
    return required


class DualEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [path for path in _required_integration_assets() if not path.exists()]
        if missing:
            relative = [
                str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
                for path in missing
            ]
            raise unittest.SkipTest(
                "data-dependent reconstruction integration gate skipped; provision: "
                + ", ".join(relative)
            )
        cls.regions = {
            mode: baseline.load_evaluation_regions(mode) for mode in baseline.EVALUATION_MODES
        }

    def test_unified_dataset_schema_and_spatial_source_blocks(self):
        expected_blocks = {"train": {0, 1, 2, 3}, "test": {4, 5}}
        expected_samples = {"train": 140, "test": 70}
        for split in ("train", "test"):
            samples = list(load_dataset("reconstruction", split))
            self.assertEqual(len(samples), expected_samples[split])
            blocks = set()
            for sample in samples:
                self.assertEqual(
                    set(sample), {"seismic_patch", "well_log_seq", "position", "label", "meta"}
                )
                self.assertEqual(sample["seismic_patch"].shape, (9, 9, 20, 18))
                self.assertEqual(sample["label"].shape, (9, 20, 18))
                self.assertEqual(sample["well_log_seq"].shape, (91, 8))
                blocks.add(int(sample["meta"]["patch_index_kji"][2]))
            self.assertEqual(blocks, expected_blocks[split])

    def test_conditional_and_strict_region_constraint_contracts(self):
        _, conditional_guard, _, conditional_wells, conditional = self.regions["conditional"]
        self.assertIsNone(conditional_guard)
        self.assertEqual(conditional["train_i_blocks"], [0, 1, 2, 3])
        self.assertEqual(conditional["test_i_blocks"], [4, 5])
        self.assertEqual(conditional["n_train_region_well_constraints"], 1)
        self.assertEqual(conditional["n_test_region_well_constraints"], 90)
        self.assertEqual(conditional_wells.shape[0], 91)
        self.assertTrue(conditional["test_well_constraints_supplied_to_idw"])
        self.assertFalse(conditional["strict_spatial_holdout_generalization"])

        strict_train, strict_guard, strict_test, strict_wells, strict = self.regions["strict"]
        self.assertIsNotNone(strict_guard)
        self.assertEqual(strict["train_i_blocks"], [4, 5])
        self.assertEqual(strict["guard_i_blocks"], [3])
        self.assertEqual(strict["test_i_blocks"], [0, 1, 2])
        self.assertEqual(strict["n_train_region_well_constraints"], 90)
        self.assertEqual(strict["n_guard_region_well_constraints"], 1)
        self.assertEqual(strict["n_test_region_well_constraints"], 0)
        self.assertEqual(strict_wells.shape[0], 90)
        self.assertTrue(strict["spatial_regions_zero_overlap"])
        self.assertFalse(strict["guard_well_constraints_supplied_to_idw"])
        self.assertFalse(strict["test_well_constraints_supplied_to_idw"])
        self.assertTrue(strict["strict_spatial_holdout_generalization"])
        self.assertEqual(int(strict_test.observed_mask.sum()), 0)

        allowed_xyz = {tuple(row.tolist()) for row in strict_wells[:, 0:3]}
        train_xyz = {
            tuple(row.tolist())
            for row in strict_train.coordinates[strict_train.observed_mask].astype(np.float32)
        }
        guard_xyz = {
            tuple(row.tolist())
            for row in strict_guard.coordinates[strict_guard.observed_mask].astype(np.float32)
        }
        self.assertEqual(allowed_xyz, train_xyz)
        self.assertTrue(allowed_xyz.isdisjoint(guard_xyz))

    def test_normalization_fits_mode_optimization_train_only(self):
        for mode in baseline.EVALUATION_MODES:
            train, _, _, allowed_wells, _ = self.regions[mode]
            optimization = train.cell_patch_k_block != 3
            idw = baseline._idw_predict(train.coordinates, allowed_wells)
            raw = np.column_stack([idw, train.seismic, train.coordinates])
            report = json.loads(
                (HERE / f"_outputs/ridge_linear/{mode}/preprocess_stats.json").read_text()
            )
            for column, stats in enumerate(report["seismic_features"]["stats"]):
                self.assertAlmostEqual(stats["mean"], float(raw[optimization, column].mean()))
                self.assertAlmostEqual(
                    stats["std"], float(raw[optimization, column].std() + 1e-8)
                )
            target_stats = report["target"]["stats"]
            self.assertAlmostEqual(target_stats["vmin"], float(train.target[optimization].min()))
            self.assertAlmostEqual(target_stats["vmax"], float(train.target[optimization].max()))

    def test_undefined_correlation_is_explicit_not_zero(self):
        metrics = baseline._metrics(np.asarray([1.0, 2.0, 3.0]), np.ones(3))
        self.assertIsNone(metrics["pearson_r"])
        self.assertFalse(metrics["pearson_r_defined"])
        self.assertTrue(metrics["pearson_r_reason"])
        self.assertTrue(math.isfinite(metrics["rmse"]))

    def test_saved_metrics_histories_and_checkpoints(self):
        for mode in baseline.EVALUATION_MODES:
            results = json.loads((HERE / f"results_{mode}.json").read_text())
            self.assertEqual(results["evaluation_mode"], mode)
            for metrics in results["models"].values():
                for key in ("rmse", "mae", "pearson_r", "r2"):
                    value = metrics[key]
                    if value is None:
                        self.assertFalse(metrics[f"{key}_defined"])
                        self.assertTrue(metrics[f"{key}_reason"])
                    else:
                        self.assertTrue(math.isfinite(value))
            for variant in ("structural", "seismic"):
                training = results["training"][variant]
                history_path = PROJECT_ROOT / training["history"]
                history = json.loads(history_path.read_text())
                self.assertEqual(len(history["train_loss"]), 600)
                self.assertEqual(len(history["val_loss"]), 600)
                self.assertTrue(all(math.isfinite(x) for x in history["train_loss"]))
                self.assertTrue(all(math.isfinite(x) for x in history["val_loss"]))
                self.assertEqual(history["best_epoch"], int(np.argmin(history["val_loss"])))
                self.assertEqual(training["best_epoch_1_based"], history["best_epoch"] + 1)
                self.assertTrue((PROJECT_ROOT / training["best_checkpoint"]).exists())
                self.assertTrue((PROJECT_ROOT / training["last_checkpoint"]).exists())

    def test_best_checkpoint_reproduces_mode_metrics(self):
        for mode in baseline.EVALUATION_MODES:
            _, _, _, metrics, results = visualize_prediction._load_real_checkpoint_prediction(mode)
            stored = results["models"][results["primary_baseline"]]
            for key in ("rmse", "mae", "pearson_r", "r2"):
                if metrics[key] is None:
                    self.assertIsNone(stored[key])
                else:
                    self.assertAlmostEqual(metrics[key], stored[key], places=14)

    def test_mode_visualizations_have_matching_metadata(self):
        for mode in baseline.EVALUATION_MODES:
            metadata = json.loads((HERE / f"visualization_metadata_{mode}.json").read_text())
            self.assertEqual(metadata["evaluation_mode"], mode)
            self.assertEqual(metadata["constraint_audit"], self.regions[mode][4])
            image_path = PROJECT_ROOT / metadata["output"]
            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 2000)
                self.assertGreater(image.height, 800)
            self.assertIn(mode.upper(), metadata["caveat"])

    def test_run_manifest_hashes_match_current_artifacts(self):
        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        for mode in baseline.EVALUATION_MODES:
            manifest = json.loads((HERE / f"run_manifest_{mode}.json").read_text())
            self.assertEqual(manifest["evaluation_mode"], mode)
            for section in ("source_sha256", "input_data_sha256", "artifact_sha256"):
                for relative_path, expected_hash in manifest[section].items():
                    self.assertEqual(sha256(PROJECT_ROOT / relative_path), expected_hash)


if __name__ == "__main__":
    unittest.main()
