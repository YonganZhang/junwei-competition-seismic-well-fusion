"""Lithofacies-prefixed P5 Stage-2 budget, split, lane, and firewall tests."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pipeline_contract = _load_track_module(
    "lithofacies_pipeline_contract_stage2", "pipeline_contract.py"
)
sys.modules["pipeline_contract"] = _pipeline_contract
_p4_contract = _load_track_module("lithofacies_p4_contract_stage2", "p4_contract.py")
sys.modules["p4_contract"] = _p4_contract
_p5_stage1 = _load_track_module("lithofacies_p5_stage1_for_stage2", "p5_stage1.py")
sys.modules["p5_stage1"] = _p5_stage1
stage2 = _load_track_module("lithofacies_p5_stage2", "lithofacies_p5_stage2.py")


class LithofaciesStage2FrozenContractTests(unittest.TestCase):
    def test_module_and_test_basenames_are_track_prefixed(self) -> None:
        self.assertEqual(Path(stage2.__file__).name, "lithofacies_p5_stage2.py")
        self.assertEqual(Path(__file__).name, "test_lithofacies_p5_stage2.py")
        self.assertEqual(stage2.__name__, "lithofacies_p5_stage2")

    def test_fixed_budget_seed_and_gm09_schema(self) -> None:
        self.assertEqual(stage2.ROOT_SEED, 2693)
        self.assertEqual(stage2.NUM_CLASSES, 9)
        self.assertEqual(len(stage2.CLASS_NAMES), 9)
        self.assertEqual(stage2.FIXED_FOLD_ID, 0)
        self.assertEqual(stage2.P_CONTEXT_LENGTH, 33)
        self.assertLessEqual(stage2.NEURAL_PARAMETER_UPDATE_LIMIT, 200)
        self.assertLessEqual(stage2.NEURAL_WALL_LIMIT_SECONDS, 600)
        self.assertLessEqual(stage2.ESTIMATOR_WALL_LIMIT_SECONDS, 300)
        self.assertLess(stage2.TINY_GATE_UPDATES, stage2.NEURAL_PARAMETER_UPDATE_LIMIT)
        first = stage2.derive_cell_seed(stage2.FIRST_TEN[0], "model")
        self.assertEqual(first, stage2.derive_cell_seed(stage2.FIRST_TEN[0], "model"))
        self.assertNotEqual(first, stage2.derive_cell_seed(stage2.FIRST_TEN[1], "model"))

    def test_first_fold_is_fixed_mother_family_logo(self) -> None:
        arrays = {
            "class_counts": np.ones(9, dtype=np.int64),
            "p_train_well": np.zeros((6, 26, 33), dtype=np.float32),
            "p_train_seismic": np.zeros((6, 3, 3, 33), dtype=np.float32),
            "p_train_labels": np.arange(6, dtype=np.int64),
            "p_train_ids": np.asarray([f"train-{index}" for index in range(6)]),
            "p_validation_well": np.zeros((3, 26, 33), dtype=np.float32),
            "p_validation_seismic": np.zeros((3, 3, 3, 33), dtype=np.float32),
            "p_validation_labels": np.arange(3, dtype=np.int64),
            "p_validation_ids": np.asarray([f"validation-{index}" for index in range(3)]),
        }
        manifest = {
            "stage1_fold_id": 0,
            "stage1_train_groups": ["15/9-F-14", "15/9-F-15", "15/9-F-4"],
            "stage1_validation_groups": ["15/9-19"],
            "frozen_test_accessed": False,
            "s_lane": {"status": "not_feasible"},
        }
        with tempfile.TemporaryDirectory(dir=TRACK_DIR / "_outputs") as directory:
            batch_file = Path(directory) / "batch.npz"
            batch_file.write_bytes(b"development-only-fixture")
            contract = stage2.batch_contract(arrays, manifest, batch_file)
        self.assertEqual(contract["fold_id"], 0)
        self.assertEqual(contract["validation_groups"], ["15/9-19"])
        self.assertFalse(set(contract["train_groups"]) & set(contract["validation_groups"]))
        self.assertNotIn(stage2.TEST_FAMILY, contract["train_groups"] + contract["validation_groups"])

    def test_budget_validator_rejects_excess_updates(self) -> None:
        result = _fake_cell(stage2.FIRST_TEN[0], "P", 0)
        result["parameter_updates"] = stage2.NEURAL_PARAMETER_UPDATE_LIMIT + 1
        with self.assertRaisesRegex(ValueError, "parameter-update"):
            stage2.validate_cell_result(result)

    def test_runner_has_no_frozen_test_entry_or_filename(self) -> None:
        signature = inspect.signature(stage2.run_pilot)
        self.assertEqual(tuple(signature.parameters), ("batch_file", "output", "model_ids", "device"))
        source = (TRACK_DIR / "lithofacies_p5_stage2.py").read_text(encoding="utf-8")
        self.assertNotIn("load_frozen_test", source)
        self.assertNotIn('"test.h5"', source)
        self.assertNotIn("'test.h5'", source)
        self.assertNotIn("test_loader", source)


def _fake_metrics(score: float) -> dict:
    return {
        "supported_class_macro_f1": score,
        "fixed_schema_macro_f1": score * 0.75,
        "worst_family_fixed_schema_macro_f1": score * 0.75,
        "balanced_accuracy": score,
        "negative_log_likelihood": 1.0,
        "expected_calibration_error": 0.1,
    }


def _fake_cell(model_id: str, lane: str, order: int) -> dict:
    is_p = lane == "P"
    return {
        "schema_version": stage2.STAGE2_SCHEMA,
        "track_id": "lithofacies",
        "task_id": stage2.TASK_ID,
        "model_id": model_id,
        "lane": lane,
        "status": "PASS" if is_p else "SKIP",
        "reason": None if is_p else {"code": "lane_not_rankable", "message": "MD absent"},
        "seed": stage2.derive_cell_seed(model_id, "model"),
        "component_seeds": {"model": stage2.derive_cell_seed(model_id, "model")},
        "source_revision": f"revision-{order}",
        "source_lock_sha256": "a" * 64,
        "split_hash": "b" * 64,
        "fold_id": 0,
        "train_groups": ["15/9-F-14", "15/9-F-15", "15/9-F-4"],
        "validation_groups": ["15/9-19"],
        "input_budget": {
            "context_positions": 33,
            "fold_train_samples_used": 10,
            "fold_validation_samples_used": 4,
        },
        "frozen_test_accessed": False,
        "test_metrics_used": False,
        "rank_eligible": is_p,
        "wall_seconds": 1.0 + order,
        "wall_limit_seconds": 600.0,
        "peak_resources": {"process_peak_rss_kib": 1, "peak_vram_bytes": 0},
        "environment": {"device_type": "cpu"},
        "validation_metrics": _fake_metrics(0.1 + order / 100.0) if is_p else None,
        "backend": "torch" if is_p else None,
        "parameter_updates": 4 if is_p else 0,
    }


class LithofaciesStage2LaneAndArtifactTests(unittest.TestCase):
    def test_finalize_keeps_s_out_of_legal_p_leaderboard(self) -> None:
        cells = []
        lock = stage2.load_source_lock()
        lane_by_id = {model["model_id"]: model["leaderboard_lane"] for model in lock["models"]}
        for order, model_id in enumerate(stage2.FIRST_TEN):
            cells.append(_fake_cell(model_id, lane_by_id[model_id], order))
        # A deliberately conflicting diagnostic proves ranking uses fixed-nine
        # Macro-F1 rather than supported-class Macro-F1.
        cells[0]["validation_metrics"].update(
            {
                "supported_class_macro_f1": 0.99,
                "fixed_schema_macro_f1": 0.01,
                "worst_family_fixed_schema_macro_f1": 0.01,
            }
        )
        cells[1]["validation_metrics"].update(
            {
                "supported_class_macro_f1": 0.10,
                "fixed_schema_macro_f1": 0.90,
                "worst_family_fixed_schema_macro_f1": 0.90,
            }
        )
        partial = {
            "schema_version": stage2.PARTIAL_SCHEMA,
            "source_lock_sha256": "a" * 64,
            "batch_sha256": "c" * 64,
            "split_hash": "b" * 64,
            "models": cells,
            "frozen_test_accessed": False,
        }
        with tempfile.TemporaryDirectory(dir=TRACK_DIR / "_outputs") as directory:
            root = Path(directory)
            partial_path = root / "partial.json"
            partial_path.write_text(json.dumps(partial), encoding="utf-8")
            output = root / "canonical"
            summary = stage2.finalize_results([partial_path], output)
            leaderboard = json.loads(
                (output / stage2.P_LEADERBOARD_FILENAME).read_text(encoding="utf-8")
            )
            results = stage2.read_jsonl(output / stage2.RESULTS_FILENAME)
        self.assertEqual(len(results), 10)
        self.assertEqual(summary["s_lane"]["status"], "not_rankable")
        self.assertFalse(summary["s_lane"]["included_in_p_leaderboard"])
        self.assertEqual(len(leaderboard["entries"]), 9)
        self.assertEqual(leaderboard["primary_metric"], "fixed_schema_macro_f1")
        self.assertEqual(leaderboard["entries"][0]["model_id"], stage2.FIRST_TEN[1])
        self.assertIn("supported_class_macro_f1", leaderboard["entries"][0])
        self.assertNotIn(
            "ms_tcn2_dense", {entry["model_id"] for entry in leaderboard["entries"]}
        )

    def test_canonical_results_cover_all_cells_and_are_portable(self) -> None:
        output = TRACK_DIR / "_outputs" / "p5_stage2"
        results_path = output / stage2.RESULTS_FILENAME
        summary_path = output / stage2.SUMMARY_FILENAME
        leaderboard_path = output / stage2.P_LEADERBOARD_FILENAME
        self.assertTrue(results_path.is_file(), results_path)
        self.assertTrue(summary_path.is_file(), summary_path)
        self.assertTrue(leaderboard_path.is_file(), leaderboard_path)
        results = stage2.read_jsonl(results_path)
        self.assertEqual([result["model_id"] for result in results], list(stage2.FIRST_TEN))
        self.assertEqual(sum(result["lane"] == "P" for result in results), 9)
        self.assertEqual(sum(result["lane"] == "S" for result in results), 1)
        for result in results:
            stage2.validate_cell_result(result)
            text = json.dumps(result, ensure_ascii=False)
            self.assertNotIn(".claude/worktrees", text)
            self.assertNotIn("/mnt/data/", text)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(results_path.read_bytes()).hexdigest()
        leaderboard_digest = hashlib.sha256(leaderboard_path.read_bytes()).hexdigest()
        self.assertEqual(summary["results_sha256"], digest)
        self.assertEqual(summary["p_leaderboard_sha256"], leaderboard_digest)
        self.assertEqual(summary["expected_cells"], 10)
        self.assertEqual(summary["passed_cells"], 9)
        self.assertEqual(summary["skipped_cells"], 1)
        self.assertEqual(summary["failed_cells"], 0)
        self.assertEqual(summary["timeout_cells"], 0)
        self.assertEqual(summary["fixed_fold"]["validation_groups"], ["15/9-19"])
        self.assertNotIn(stage2.TEST_FAMILY, summary["fixed_fold"]["train_groups"])
        self.assertEqual(summary["p_lane"]["primary_metric"], "fixed_schema_macro_f1")
        self.assertEqual(leaderboard["primary_metric"], "fixed_schema_macro_f1")
        for entry in leaderboard["entries"]:
            self.assertIn("worst_family_fixed_schema_macro_f1", entry)
        self.assertFalse(summary["frozen_test_accessed"])


if __name__ == "__main__":
    unittest.main()
