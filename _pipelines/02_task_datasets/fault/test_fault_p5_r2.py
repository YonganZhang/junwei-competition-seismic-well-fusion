#!/usr/bin/env python3
"""Fail-closed tests for the fault-prefixed P5.2 R2 acquisition gate."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parent


def _load_fault_p5_r2():
    name = "fault_p5_r2"
    spec = importlib.util.spec_from_file_location(name, TRACK_DIR / "fault_p5_r2.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load fault_p5_r2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fault_p5_r2 = _load_fault_p5_r2()


class FaultP5R2AcquisitionTests(unittest.TestCase):
    def _run(self, root: Path) -> tuple[dict, list[dict], dict, dict]:
        summary = fault_p5_r2.run_r2_acquisition(root)
        results = [
            json.loads(line)
            for line in (root / fault_p5_r2.RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()
        ]
        contract = json.loads((root / fault_p5_r2.CONTRACT_FILENAME).read_text(encoding="utf-8"))
        manifest = json.loads((root / fault_p5_r2.ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        return summary, results, contract, manifest

    def test_summary_rows_and_contract_are_zero_training_and_lane_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, results, contract, manifest = self._run(Path(directory))
        self.assertEqual(summary["baseline_commit"], "af8c066de0c3fc24fce024abb350b4f2c9e82d9b")
        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["ranking_status"], "not_rankable")
        self.assertEqual(summary["reason_code"], "NO_VALID_FAULT_DEVELOPMENT_FOLDS")
        self.assertEqual(summary["model_roster_used"], [])
        self.assertEqual(summary["training_cell_count"], 0)
        self.assertEqual(summary["current_trainable_lanes"], [])
        self.assertFalse(summary["official_learning_curve_generated"])
        self.assertIsNone(summary["winner"])
        self.assertFalse(summary["R3_allowed"])
        self.assertEqual(summary["observed_acquisition_baseline_count"], 3)
        self.assertEqual(summary["planned_null_point_count"], 9)
        self.assertEqual(summary["minimum_unblock_contract_count"], len(contract["minimum_unblock_contract"]))
        self.assertEqual([row["lane_id"] for row in results], list(fault_p5_r2.LANES))
        self.assertEqual(results[1]["data_ready"], True)
        self.assertFalse(results[0]["data_ready"])
        self.assertFalse(results[2]["data_ready"])
        self.assertEqual(results[0]["train_allowed"], False)
        self.assertEqual(results[1]["train_allowed"], False)
        self.assertEqual(results[2]["train_allowed"], False)
        self.assertEqual(results[0]["rank_allowed"], False)
        self.assertEqual(results[1]["rank_allowed"], False)
        self.assertEqual(results[2]["rank_allowed"], False)
        self.assertIn("REGISTERED_DENSE_SYNTHETIC_DATASET_MISSING", results[0]["reason_codes"])
        self.assertIn("MASKED_WEAK_OBJECTIVE_NOT_CONFIGURED", results[1]["reason_codes"])
        self.assertIn("AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING", results[2]["reason_codes"])
        self.assertIn("FEWER_THAN_TWO_LEGAL_BUFFERED_FOLDS", results[2]["reason_codes"])
        self.assertNotIn("observed", summary["stage3_visualization_reuse"]["figures"][0])
        self.assertEqual(manifest["current_trainable_lanes"], [])
        self.assertEqual(manifest["checkpoint_count"], 0)
        self.assertEqual(manifest["prediction_payload_count"], 0)
        self.assertFalse(manifest["refit_executed"])
        self.assertFalse(manifest["holdout_accessed"])

    def test_planned_null_points_are_observed_false_and_null_metric_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, results, contract, _ = self._run(Path(directory))
        for row in results:
            self.assertTrue(row["observed_acquisition_baseline"]["observed"])
            self.assertIsNone(row["observed_acquisition_baseline"]["metric"])
            self.assertIsNone(row["observed_acquisition_baseline"]["value"])
            self.assertTrue(row["planned_null_points"])
            for point in row["planned_null_points"]:
                self.assertFalse(point["observed"])
                self.assertIsNone(point["metric"])
                self.assertIsNone(point["value"])
        self.assertEqual(
            tuple(item["lane_id"] for item in contract["planned_null_points"]),
            fault_p5_r2.LANES,
        )

    def test_formal_lane_stops_before_model_discovery_and_no_test_surface_exists(self) -> None:
        self.assertEqual(
            set(inspect.signature(fault_p5_r2.run_r2_acquisition).parameters),
            {"output_dir"},
        )
        source = Path(fault_p5_r2.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertTrue(
            imported_roots.isdisjoint(
                {"torch", "tensorflow", "monai", "sklearn", "h5py", "segyio", "optuna", "numpy"}
            )
        )
        for option in ("--model", "--fold", "--test", "--holdout", "--hpo", "--checkpoint"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    fault_p5_r2.parse_args([option, "forbidden"])

    def test_portable_artifacts_have_stable_hashes_and_no_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, results, contract, manifest = self._run(root)
            for record in manifest["artifacts"]:
                path = root / record["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
                self.assertEqual(path.stat().st_size, record["bytes"])
            serialized = json.dumps(
                {"summary": summary, "results": results, "contract": contract, "manifest": manifest},
                sort_keys=True,
            )
            self.assertNotIn("/mnt/", serialized)
            self.assertNotIn("/.claude/worktrees/", serialized)
            self.assertEqual(
                summary["stage3_visualization_reuse"]["path"],
                "p5_stage3/p5_stage3_visualization_manifest.json",
            )
            self.assertEqual(len(summary["stage3_visualization_reuse"]["figures"]), 3)


if __name__ == "__main__":
    unittest.main()
