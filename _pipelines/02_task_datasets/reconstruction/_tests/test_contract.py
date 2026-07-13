"""Portable reconstruction contract tests; no HDF5, Layer1, or checkpoint required."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
MODES = ("conditional", "strict")

sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.model_registry import MODEL_REGISTRY, get_model  # noqa: E402


class PortableContractTest(unittest.TestCase):
    def test_visualization_cli_runs_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indices = np.asarray(
                [(k, j, i) for k in range(2) for j in range(2) for i in range(2)],
                dtype=np.int64,
            )
            truth = np.linspace(0.1, 0.3, indices.shape[0])
            prediction = truth + np.linspace(-0.01, 0.01, indices.shape[0])
            predictions = root / "predictions.npz"
            metrics = root / "metrics.json"
            output = root / "diagnostics.png"
            np.savez_compressed(
                predictions,
                mode=np.asarray("strict"),
                task_id=np.asarray("volve_porosity_strict_spatial_reconstruction"),
                indices_kji=indices,
                volume_shape_kji=np.asarray((2, 2, 2), dtype=np.int64),
                truth=truth,
                prediction=prediction,
                residual=prediction - truth,
                amplitude=np.linspace(-1.0, 1.0, indices.shape[0]),
            )
            metrics.write_text(
                json.dumps(
                    {
                        "evaluation_mode": "strict",
                        "task_id": "volve_porosity_strict_spatial_reconstruction",
                        "strict_rmse": 0.01,
                        "strict_r2": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "p4_visualize.py"),
                    "--predictions",
                    str(predictions),
                    "--metrics",
                    str(metrics),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())

    def test_p4_cli_runs_from_project_root(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(HERE / "p4_reconstruction.py"),
                "task-specs",
                "--mode",
                "strict",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["strict"]["task_id"],
            "volve_porosity_strict_spatial_reconstruction",
        )

    def test_p4_cli_discovers_canonical_model_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "p4_reconstruction.py"),
                    "tiny-smoke",
                    "--mode",
                    "strict",
                    "--model",
                    "ridge_linear",
                    "--output-dir",
                    str(Path(directory) / "tiny"),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["finite_prediction"])
        self.assertEqual(payload["model"], "ridge_linear")

    def test_alternative_models_are_dynamic_and_checkpoint_compatible(self):
        features = np.asarray(
            [
                [0.0, 0.5, -1.0],
                [1.0, -0.5, 0.25],
                [-0.5, 0.25, 0.75],
                [0.25, 1.0, -0.25],
            ],
            dtype=np.float64,
        )
        target = np.asarray([0.2, 0.8, -0.1, 0.4], dtype=np.float64)
        kwargs = {
            "n_features": features.shape[1],
            "learning_rate": 0.01,
            "ridge_alpha": 0.1,
            "n_training_samples": target.size,
        }
        for name in ("reconstruction_linear_sgd", "reconstruction_tiny_mlp"):
            with self.subTest(model=name):
                # Force the registry's same-name dynamic-import path: no models
                # package import list is permitted for swappable alternatives.
                MODEL_REGISTRY.pop(name, None)
                sys.modules.pop(f"models.{name}", None)
                model = get_model(name, models_package="models", **kwargs)
                self.assertEqual(model.__class__.__module__, f"models.{name}")

                prediction_before = model.predict(features)
                self.assertEqual(prediction_before.shape, (target.size,))
                self.assertTrue(np.all(np.isfinite(prediction_before)))
                train_loss = model.train_batch((features, target))
                val_loss = model.validation_loss((features, target))
                self.assertTrue(math.isfinite(train_loss))
                self.assertTrue(math.isfinite(val_loss))
                prediction_after = model.predict(features)
                self.assertEqual(prediction_after.shape, (target.size,))
                self.assertTrue(np.all(np.isfinite(prediction_after)))
                self.assertFalse(np.array_equal(prediction_before, prediction_after))

                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint = Path(temp_dir) / f"{name}.ckpt"
                    model.save_checkpoint(checkpoint)
                    restored = get_model(name, models_package="models", **kwargs)
                    restored.load_checkpoint(checkpoint)
                    np.testing.assert_array_equal(
                        restored.predict(features), prediction_after
                    )

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
        # Inspect portable source/docs only; ignored data-dependent `_tmp`
        # dependency trees are deliberately outside the Git contract.
        paths = (
            list(HERE.glob("*.py"))
            + list(HERE.glob("*.md"))
            + list((HERE / "models").glob("*.py"))
            + list((HERE / "models").glob("*.md"))
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
