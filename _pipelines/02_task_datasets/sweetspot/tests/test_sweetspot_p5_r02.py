"""Fail-closed tests for sweetspot P5.2 / protocol R2 budget sweep."""
from __future__ import annotations

import hashlib
import importlib
import json
import unittest
from pathlib import Path


r02 = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.r02.runner")

OUTPUT_DIR = r02.DEFAULT_OUTPUT_DIR
RESULT_PATH = OUTPUT_DIR / r02.RESULT_FILENAME
SUMMARY_PATH = OUTPUT_DIR / r02.SUMMARY_FILENAME
PLATEAU_PATH = OUTPUT_DIR / r02.PLATEAU_FILENAME
STATUS_PATH = OUTPUT_DIR / r02.STATUS_GATES_FILENAME
VISUALIZATION_PATH = OUTPUT_DIR / r02.VISUALIZATION_FILENAME
ARTIFACT_PATH = OUTPUT_DIR / r02.ARTIFACT_MANIFEST_FILENAME


class SweetspotP5R02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (RESULT_PATH, SUMMARY_PATH, PLATEAU_PATH, STATUS_PATH, VISUALIZATION_PATH, ARTIFACT_PATH)
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("run the sweetspot P5.2/R2 runner to archive portable artifacts")
        cls.results = [json.loads(line) for line in RESULT_PATH.read_text(encoding="utf-8").splitlines()]
        cls.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.plateau = json.loads(PLATEAU_PATH.read_text(encoding="utf-8"))
        cls.status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        cls.visualization = json.loads(VISUALIZATION_PATH.read_text(encoding="utf-8"))
        cls.artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    def test_track_prefixed_module_and_test_basenames(self) -> None:
        self.assertEqual(Path(__file__).name, "test_sweetspot_p5_r02.py")
        self.assertEqual(Path(r02.__file__).name, "runner.py")

    def test_expected_budget_ladder_and_cell_counts(self) -> None:
        self.assertEqual(r02.ROOT_SEED, 2693)
        self.assertEqual(r02.MAIN_BUDGETS, (64, 256, 1024))
        self.assertEqual(r02.ABLATION_BUDGET, 256)
        self.assertEqual(r02.MAIN_EXPECTED_CELLS, 90)
        self.assertEqual(r02.ABLATION_EXPECTED_CELLS, 30)
        self.assertEqual(r02.EXPECTED_CELLS, 120)

    def test_results_cover_main_and_ablation_only(self) -> None:
        self.assertEqual(len(self.results), 120)
        self.assertEqual(self.summary["expected_cells"], 120)
        self.assertEqual(self.summary["attempted_cells"], 120)
        self.assertEqual(sum(row["variant_kind"] == "main" for row in self.results), 90)
        self.assertEqual(sum(row["variant_kind"] == "ablation" for row in self.results), 30)
        self.assertEqual({row["task_id"] for row in self.results}, {"T1", "T2", "T3"})
        self.assertEqual({row["budget"] for row in self.results if row["variant_kind"] == "main"}, {64, 256, 1024})
        self.assertEqual({row["budget"] for row in self.results if row["variant_kind"] == "ablation"}, {256})

    def test_boundary_targets_remain_status_only(self) -> None:
        expected = {"T4": "boundary", "T5": "not_feasible", "T6": "blocked", "T7": "blocked"}
        self.assertEqual(self.summary["target_status"], expected)
        self.assertEqual({entry["task_id"] for entry in self.status["targets"]}, {"T4", "T5", "T6", "T7"})
        self.assertFalse(self.status["test_accessed"])
        self.assertFalse(self.status["historical_test_metrics_read"])
        self.assertFalse(self.status["label_generated"])

    def test_no_test_or_label_generation_claim(self) -> None:
        self.assertFalse(self.summary["test_accessed"])
        self.assertFalse(self.summary["labels_generated"])
        self.assertFalse(self.summary["historical_test_metrics_read"])
        for item in self.results:
            self.assertFalse(item["test_accessed"])
            self.assertFalse(item["label_generated"])
            self.assertIsNotNone(item["status"])
            self.assertIn(item["status"], {"PASS", "SKIP", "FAILED"})

    def test_budget_and_ablation_records_are_present(self) -> None:
        pass_rows = [row for row in self.results if row["status"] == "PASS"]
        self.assertTrue(pass_rows)
        for target_id in ("T1", "T2", "T3"):
            rows = [row for row in self.results if row["task_id"] == target_id]
            self.assertEqual(len(rows), 40)
            self.assertTrue(any(row["variant_kind"] == "main" and row["budget"] == 256 for row in rows))
            self.assertTrue(any(row["variant_kind"] == "ablation" and row["budget"] == 256 for row in rows))

    def test_plateau_gate_has_all_three_targets(self) -> None:
        self.assertEqual(set(self.plateau["targets"]), {"T1", "T2", "T3"})
        self.assertFalse(self.plateau["test_accessed"])
        for target_id in ("T1", "T2", "T3"):
            gate = self.plateau["targets"][target_id]
            self.assertIn(gate["status"], {"pass", "blocked"})
            self.assertIn("budgets", gate)
            self.assertTrue({"64", "256", "1024"} <= set(gate["budgets"]))

    def test_artifacts_are_portable_and_hashed(self) -> None:
        self.assertTrue(self.artifacts["all_paths_portable"])
        for row in self.artifacts["artifacts"]:
            path = OUTPUT_DIR / row["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(row["size_bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        serialized = json.dumps(
            {
                "results": self.results,
                "summary": self.summary,
                "plateau": self.plateau,
                "status": self.status,
                "visualization": self.visualization,
                "artifacts": self.artifacts,
            },
            ensure_ascii=False,
        )
        self.assertNotIn("/mnt/", serialized)
        self.assertNotIn(".claude/worktrees", serialized)

    def test_visualizations_exist(self) -> None:
        figure_names = {Path(row["path"]).name for row in self.visualization["figures"]}
        self.assertTrue({
            "t1_budget_curve.png",
            "t2_budget_curve.png",
            "t3_budget_curve.png",
            "ablation_comparison.png",
            "status_gate.png",
            "plateau_gate.png",
        } <= figure_names)

    def test_summary_hashes_match_archived_files(self) -> None:
        self.assertEqual(self.summary["results_sha256"], hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.summary["plateau_gate_sha256"], hashlib.sha256(PLATEAU_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.summary["status_gate_sha256"], hashlib.sha256(STATUS_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.summary["visualization_manifest_sha256"], hashlib.sha256(VISUALIZATION_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.summary["artifact_manifest_sha256"], hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

