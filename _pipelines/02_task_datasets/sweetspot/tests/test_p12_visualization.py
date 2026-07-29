from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image


viz = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p12_visualization")


class SweetspotP12VisualizationTests(unittest.TestCase):
    def test_build_writes_manifest_and_vector_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = viz.build(Path(tmp))
            manifest = Path(result["manifest_path"])
            self.assertTrue(manifest.is_file())
            payload = importlib.import_module("json").loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_count"], 8)
            self.assertEqual(payload["track_id"], "sweetspot")
            self.assertIn("p12_contract", payload)
            contract = payload["p12_contract"]
            self.assertEqual(contract["schema_version"], "scientific-visualization-contract/v1")
            self.assertEqual(contract["profile"], "p12_tracks_1_3_5")
            self.assertEqual(contract["track_id"], "sweetspot")
            self.assertIn("renderer", contract)
            self.assertTrue(contract["renderer"]["path"])
            self.assertTrue(contract["renderer"]["sha256"])
            self.assertFalse(contract["manual_review"]["reviewed"])
            self.assertEqual(contract["manual_review"]["status"], "pending")
            self.assertTrue(contract["inputs"])
            self.assertTrue(contract["outputs"])
            for record in contract["inputs"]:
                for key in ("path", "sha256", "shape_or_row_count", "scientific_role", "split_scope"):
                    self.assertIn(key, record)
            for record in contract["outputs"]:
                for key in ("role", "path", "sha256", "width_px", "height_px", "dpi", "vector_companions"):
                    self.assertIn(key, record)
            self.assertEqual([entry["target_id"] for entry in payload["figures"] if entry.get("target_id") not in {None, "overview"}], ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])
            for entry in payload["figures"]:
                bundle = entry["figure_paths"]
                for ext in ("png", "svg", "pdf"):
                    path = Path(viz.REPO_ROOT) / bundle[ext]
                    self.assertTrue(path.is_file(), path)
                    self.assertGreater(path.stat().st_size, 0, path)
                with Image.open(Path(viz.REPO_ROOT) / bundle["png"]) as image:
                    self.assertGreaterEqual(image.size[0], 1200)
                    self.assertGreaterEqual(image.size[1], 500)
                self.assertTrue(entry["visual_qa"]["no_titles"])
                self.assertTrue(entry["visual_qa"]["fonts_normalized"])
                self.assertEqual(entry["visual_qa"]["palette"], "Akun_UKIYOE_4")
                self.assertEqual(sorted(entry["visual_qa"]["vector_outputs"]), ["pdf", "svg"])
                self.assertIn("dimensions_px", entry)
                self.assertGreater(entry["dimensions_px"]["width_px"], 0)
                self.assertGreater(entry["dimensions_px"]["height_px"], 0)

    def test_two_renders_match_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            result1 = viz.build(Path(tmp1))
            result2 = viz.build(Path(tmp2))
            manifest1 = importlib.import_module("json").loads(Path(result1["manifest_path"]).read_text(encoding="utf-8"))
            manifest2 = importlib.import_module("json").loads(Path(result2["manifest_path"]).read_text(encoding="utf-8"))
            targets1 = [entry for entry in manifest1["figures"] if entry.get("target_id") in {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}]
            targets2 = [entry for entry in manifest2["figures"] if entry.get("target_id") in {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}]
            self.assertEqual(len(targets1), len(targets2))
            for entry1, entry2 in zip(targets1, targets2):
                self.assertEqual(entry1["target_id"], entry2["target_id"])
                for ext in ("png", "svg", "pdf"):
                    path1 = Path(viz.REPO_ROOT) / entry1["figure_paths"][ext]
                    path2 = Path(viz.REPO_ROOT) / entry2["figure_paths"][ext]
                    self.assertEqual(viz._sha256_file(path1), viz._sha256_file(path2), (entry1["target_id"], ext))

    def test_source_inputs_exist_and_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = viz.build(Path(tmp))
            manifest = importlib.import_module("json").loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            for entry in manifest["figures"]:
                for source in entry["source_inputs"]:
                    path = Path(viz.REPO_ROOT) / source["path"]
                    self.assertTrue(path.exists(), path)
                    self.assertEqual(source["sha256"], viz._sha256_file(path))
                self.assertIn(entry["split_scope"], {"mixed seven-target overview", "known-holdout confirmation", "no valid label contract", "no development-only feature source"})
            for source in manifest["p12_contract"]["inputs"]:
                path = Path(viz.REPO_ROOT) / source["path"]
                self.assertTrue(path.exists(), path)
                self.assertEqual(source["sha256"], viz._sha256_file(path))

    def test_metric_direction_contracts_and_guardrails(self) -> None:
        self.assertEqual(viz._metric_direction_text(viz.TARGET_META["T1"]), "↓ MAE")
        self.assertEqual(viz._metric_direction_text(viz.TARGET_META["T3"]), "↓ MAE")
        self.assertEqual(viz._metric_direction_text(viz.TARGET_META["T2"]), "↑ AP / ↓ Brier")
        self.assertEqual(viz._metric_direction_text(viz.TARGET_META["T4"]), "↑ AP / ↓ Brier")
        self.assertEqual(viz._metric_direction_text(viz.TARGET_META["T5"]), "status only")
        fake = {
            "task_type": "regression",
            "primary_metric": "mae",
            "cell_metrics": [],
            "folds": [],
            "models": [],
            "scatter": [],
            "group_error": [],
        }
        with self.assertRaises(ValueError):
            viz._validate_target_payload("T1", fake)
        with self.assertRaises(AssertionError):
            fig = __import__("matplotlib.pyplot").pyplot.figure()
            try:
                fig.suptitle("illegal title")
                viz._assert_no_titles(fig)
            finally:
                __import__("matplotlib.pyplot").pyplot.close(fig)

    def test_status_targets_reject_label_generation(self) -> None:
        fake = {
            "status": "not_feasible",
            "label_generated": True,
            "expected_training_cells": 0,
            "development_feature_source_available": False,
            "task_type": "status",
            "primary_metric": None,
            "cell_metrics": [],
            "folds": [],
            "models": [],
        }
        with self.assertRaises(ValueError):
            viz._validate_target_payload("T5", fake)


if __name__ == "__main__":
    unittest.main()
