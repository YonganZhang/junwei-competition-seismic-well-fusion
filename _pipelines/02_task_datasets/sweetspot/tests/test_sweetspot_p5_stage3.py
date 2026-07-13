"""Fail-closed tests for sweetspot P5 Stage-3 multiseed development CV."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


stage3 = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage3"
)
data_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_data"
)
labels = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_labels"
)

OUTPUT_DIR = stage3.DEFAULT_OUTPUT_DIR
RESULT_PATH = OUTPUT_DIR / stage3.RESULT_FILENAME
SUMMARY_PATH = OUTPUT_DIR / stage3.SUMMARY_FILENAME
OOF_PATH = OUTPUT_DIR / stage3.OOF_MANIFEST_FILENAME
VISUALIZATION_PATH = OUTPUT_DIR / stage3.VISUALIZATION_MANIFEST_FILENAME


class SweetspotP5Stage3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = labels.validate_label_mapping()

    def test_track_prefixed_module_and_test_basenames(self) -> None:
        self.assertEqual(Path(__file__).name, "test_sweetspot_p5_stage3.py")
        self.assertEqual(Path(stage3.__file__).name, "sweetspot_p5_stage3.py")

    def test_exact_117_unique_preregistered_cells(self) -> None:
        keys = stage3.expected_cell_keys()
        self.assertEqual(stage3.EXPECTED_CELLS, 117)
        self.assertEqual(len(keys), 117)
        self.assertEqual(len(set(keys)), 117)
        expected_by_task = {"T1": 27, "T2": 27, "T3": 36, "T4": 27}
        self.assertEqual(
            {task: sum(key[0] == task for key in keys) for task in expected_by_task},
            expected_by_task,
        )

    def test_repeat_seeds_are_exact_and_not_derived_per_model(self) -> None:
        self.assertEqual(stage3.REPEAT_SEEDS, (1867973658, 2137841944, 3902865753))
        for task_id, model_id, fold_id, repeat_id, seed in stage3.expected_cell_keys():
            self.assertEqual(seed, stage3.REPEAT_SEEDS[repeat_id])

    def test_frozen_candidates_and_folds_match_protocol(self) -> None:
        expected = {
            "T1": (("lightgbm", "catboost", "xgboost"), (0, 1, 2)),
            "T2": (("catboost", "xgboost", "lightgbm"), (0, 1, 2)),
            "T3": (("lightgbm", "inceptiontime", "xgboost"), (0, 1, 2, 3)),
            "T4": (("catboost", "lightgbm", "inceptiontime"), (0, 1, 2)),
        }
        observed = {
            task_id: (tuple(contract["models"]), tuple(contract["folds"]))
            for task_id, contract in stage3.FROZEN_TASKS.items()
        }
        self.assertEqual(observed, expected)

    def test_every_manifest_fold_is_group_and_sample_isolated(self) -> None:
        evidence = stage3.validate_stage3_contract(self.audit)
        self.assertEqual({task: len(row["folds"]) for task, row in evidence.items()}, {
            "T1": 3, "T2": 3, "T3": 4, "T4": 3,
        })
        for task_id, row in evidence.items():
            self.assertFalse(row["test_partition_used"])
            split = self.audit.split_manifest(task_id)
            for fold in split["folds"]:
                self.assertFalse(set(fold["train_groups"]) & set(fold["validation_groups"]))
                self.assertFalse(set(fold["train_sample_ids"]) & set(fold["validation_sample_ids"]))

    def test_budget_and_updates_are_identical_to_stage2(self) -> None:
        self.assertEqual(stage3.TRAIN_SAMPLE_LIMIT, 1024)
        self.assertEqual(stage3.VALIDATION_SAMPLE_LIMIT, 512)
        self.assertEqual(stage3.TREE_UPDATE_LIMIT, 64)
        self.assertEqual(stage3.NEURAL_UPDATE_LIMIT, 64)
        self.assertEqual(stage3.CPU_WALL_LIMIT_SECONDS, 300)
        self.assertEqual(stage3.NEURAL_WALL_LIMIT_SECONDS, 600)

    def test_fold_loader_has_no_random_holdout_api(self) -> None:
        signature = inspect.signature(data_module.load_development_pilot_data)
        self.assertIn("fold_id", signature.parameters)
        self.assertNotIn("test_size", signature.parameters)
        self.assertNotIn("validation_fraction", signature.parameters)
        self.assertNotIn("split_seed", signature.parameters)

    def test_test_firewall_rejects_materialized_test_paths(self) -> None:
        for path in (Path("frozen_test/metrics.json"), Path("test.h5"), Path("x/test.hdf5")):
            self.assertTrue(data_module.forbidden_test_source(path))
        parser_options = {action.dest for action in stage3._parser()._actions}
        self.assertNotIn("test", parser_options)
        self.assertNotIn("frozen_test", parser_options)

    def test_contract_audit_opens_no_frozen_test_or_materialized_test_file(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            candidate = Path(path)
            opened.append(candidate)
            if data_module.forbidden_test_source(candidate):
                raise AssertionError(f"forbidden path opened: {candidate}")
            return original(candidate, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            audit = labels.validate_label_mapping()
            stage3.validate_stage3_contract(audit)
        self.assertTrue(opened)
        self.assertTrue(all(not data_module.forbidden_test_source(path) for path in opened))

    def test_t5_t6_t7_gates_remain_fail_closed_and_independent(self) -> None:
        t5 = stage3._status_payload(self.audit, "T5")
        t6 = stage3._status_payload(self.audit, "T6")
        t7 = stage3._status_payload(self.audit, "T7")
        self.assertEqual(t5["status"], "not_feasible")
        self.assertEqual(t6["status"], "blocked")
        self.assertEqual(t7["status"], "blocked")
        self.assertEqual(t6["target_name"], "PHIF")
        self.assertEqual(t7["target_name"], "KLOGH")
        self.assertNotEqual(t6["label_version"], t7["label_version"])
        self.assertFalse(t6["development_feature_source_available"])
        self.assertFalse(t7["development_feature_source_available"])


class SweetspotP5Stage3ArchivedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (RESULT_PATH, SUMMARY_PATH, OOF_PATH, VISUALIZATION_PATH)
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("run the sweetspot Stage-3 runner to archive portable artifacts")
        cls.results = [json.loads(line) for line in RESULT_PATH.read_text(encoding="utf-8").splitlines()]
        cls.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.oof = json.loads(OOF_PATH.read_text(encoding="utf-8"))
        cls.visualization = json.loads(VISUALIZATION_PATH.read_text(encoding="utf-8"))
        cls.audit = labels.validate_label_mapping()

    def test_results_cover_every_legal_cell_once(self) -> None:
        observed = [
            (row["task_id"], row["model_id"], row["fold_id"], row["repeat_id"], row["seed"])
            for row in self.results
        ]
        self.assertEqual(len(observed), 117)
        self.assertEqual(len(set(observed)), 117)
        self.assertEqual(set(observed), set(stage3.expected_cell_keys()))
        self.assertEqual(self.summary["expected_cells"], 117)
        self.assertEqual(self.summary["attempted_cells"], 117)

    def test_no_cross_lane_candidate_or_artifact_pollution(self) -> None:
        for row in self.results:
            contract = stage3.FROZEN_TASKS[row["task_id"]]
            self.assertIn(row["model_id"], contract["models"])
            self.assertIn(row["fold_id"], contract["folds"])
            self.assertEqual(row["lane"], self.audit.target(row["task_id"])["slug"])
        for entry in self.oof["entries"]:
            if entry.get("prediction_path"):
                self.assertIn(f"/{entry['task_id']}/", entry["prediction_path"])
                self.assertIn(f"/{entry['model_id']}/", entry["prediction_path"])

    def test_same_task_fold_uses_one_frozen_input_budget(self) -> None:
        for task_id, contract in stage3.FROZEN_TASKS.items():
            for fold_id in contract["folds"]:
                hashes = {
                    row["input_budget"]["input_budget_sha256"]
                    for row in self.results
                    if row["task_id"] == task_id and row["fold_id"] == fold_id
                }
                self.assertEqual(len(hashes), 1)
                self.assertNotIn(None, hashes)

    def test_rankability_obeys_eighty_percent_gate(self) -> None:
        for task_id in stage3.FROZEN_TASKS:
            board = json.loads((OUTPUT_DIR / "leaderboards" / f"{task_id}.json").read_text(encoding="utf-8"))
            expected_status = "rankable" if board["completion_rate"] >= 0.80 else "not_rankable"
            self.assertEqual(board["status"], expected_status)
            self.assertEqual(board["expected_cells"], sum(row["task_id"] == task_id for row in self.results))
            self.assertEqual({entry["model_id"] for entry in board["entries"]}, set(stage3.FROZEN_TASKS[task_id]["models"]))

    def test_t5_t6_t7_have_status_only_leaderboards_and_data_gate_figures(self) -> None:
        expected_figures = {"T5_status_gate.png", "T6_data_gate.png", "T7_data_gate.png"}
        figure_names = {Path(entry["figure_path"]).name for entry in self.visualization["entries"]}
        self.assertTrue(expected_figures <= figure_names)
        for task_id in stage3.STATUS_ONLY_TARGETS:
            board = json.loads((OUTPUT_DIR / "leaderboards" / f"{task_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(board["status"], "not_rankable")
            self.assertEqual(board["expected_cells"], 0)

    def test_each_t1_t4_lane_has_separate_required_figures(self) -> None:
        names = {Path(entry["figure_path"]).name for entry in self.visualization["entries"]}
        for task_id in ("T1", "T3"):
            self.assertTrue({
                f"{task_id}_regression_scatter.png",
                f"{task_id}_well_group_error.png",
                f"{task_id}_fold_seed.png",
            } <= names)
        for task_id in ("T2", "T4"):
            self.assertTrue({
                f"{task_id}_pr_calibration.png",
                f"{task_id}_well_group_error.png",
                f"{task_id}_fold_seed.png",
            } <= names)

    def test_figures_rebuild_from_portable_aggregates_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = stage3.rebuild_figures(OUTPUT_DIR / "visualization_data", Path(directory))
            self.assertEqual(len(rebuilt), 15)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in rebuilt))

    def test_oof_manifest_preserves_all_cell_statuses_and_ignored_predictions(self) -> None:
        self.assertEqual(len(self.oof["entries"]), 117)
        self.assertEqual({entry["cell_key"] for entry in self.oof["entries"]}, {row["cell_key"] for row in self.results})
        for entry in self.oof["entries"]:
            self.assertFalse(entry["contains_test"])
            self.assertFalse(entry["tracked"])
            if entry["prediction_path"] is not None:
                self.assertIn("/_private_predictions/", entry["prediction_path"])

    def test_no_test_label_checkpoint_or_historical_metric_claim(self) -> None:
        self.assertFalse(self.summary["test_accessed"])
        self.assertFalse(self.summary["historical_test_metrics_used"])
        self.assertFalse(self.summary["labels_generated"])
        self.assertFalse(self.summary["checkpoints_persisted"])
        self.assertFalse(self.summary["full_predictions_tracked"])
        for row in self.results:
            self.assertFalse(row["test_firewall"]["test_accessed"])
            self.assertFalse(row["test_firewall"]["historical_test_metrics_used"])
            self.assertFalse(row["label_generated"])
            self.assertFalse(row["checkpoint_persisted"])

    def test_hashes_baseline_and_paths_are_portable(self) -> None:
        self.assertEqual(self.summary["baseline_commit"], stage3.BASELINE_COMMIT)
        self.assertEqual(self.summary["results_sha256"], hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.summary["oof_manifest_sha256"], hashlib.sha256(OOF_PATH.read_bytes()).hexdigest())
        serialized = json.dumps({
            "results": self.results,
            "summary": self.summary,
            "oof": self.oof,
            "visualization": self.visualization,
        })
        self.assertNotIn("/mnt/", serialized)
        self.assertNotIn(".claude/worktrees", serialized)


if __name__ == "__main__":
    unittest.main()
