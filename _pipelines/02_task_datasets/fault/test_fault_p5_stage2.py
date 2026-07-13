#!/usr/bin/env python3
"""Contract tests for the fault-prefixed P5 Stage-2 data gate."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parent


def _load_fault_stage2_module():
    module_name = "fault_p5_stage2"
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / "fault_p5_stage2.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load fault_p5_stage2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fault_p5_stage2 = _load_fault_stage2_module()


class FaultP5Stage2DataGateTests(unittest.TestCase):
    def _run(self, root: Path) -> tuple[dict, list[dict]]:
        summary = fault_p5_stage2.run_stage2_data_gate(root)
        results_path = root / fault_p5_stage2.RESULTS_FILENAME
        records = [json.loads(line) for line in results_path.read_text().splitlines()]
        return summary, records

    def test_frozen_ten_cells_are_blocked_and_not_rankable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, records = self._run(Path(directory))
        self.assertEqual(tuple(row["model_id"] for row in records), fault_p5_stage2.FIRST_TEN_MODEL_IDS)
        self.assertEqual(summary["candidate_count"], 10)
        self.assertEqual(summary["status"], "not_rankable")
        self.assertEqual(summary["counts"]["blocked"], 10)
        self.assertEqual(summary["counts"]["attempted"], 0)
        self.assertEqual({row["status"] for row in records}, {"blocked"})
        self.assertEqual({row["ranking_status"] for row in records}, {"not_rankable"})
        self.assertFalse(summary["leaderboard"]["generated"])

    def test_gate_proves_missing_covered_negatives_and_source_unknown_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, records = self._run(Path(directory))
        gate = summary["data_gate"]
        self.assertEqual(gate["status"], "blocked")
        self.assertFalse(gate["formal_training_allowed"])
        self.assertEqual(gate["verified_negative_coverage"]["observed_labels"], 0)
        self.assertEqual(gate["verified_negative_coverage"]["audit_status"], "absent")
        self.assertEqual(gate["unknown_mask"]["observed_unknown_labels"], 2016)
        self.assertTrue(gate["unknown_mask"]["in_memory_unknown_semantics_available"])
        self.assertEqual(gate["unknown_mask"]["source_mask_audit_status"], "absent")
        blocker_codes = {item["code"] for item in gate["blockers"]}
        self.assertIn("AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING", blocker_codes)
        self.assertIn("COVERAGE_AUDITED_UNKNOWN_MASK_MISSING", blocker_codes)
        self.assertIn("DEVELOPMENT_SPLIT_NOT_FEASIBLE", blocker_codes)
        self.assertEqual(records[0]["reason"]["blocking_codes"], [item["code"] for item in gate["blockers"]])

    def test_fixed_budget_is_preregistered_but_nothing_is_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, records = self._run(Path(directory))
        self.assertEqual(summary["fixed_budget"]["max_parameter_updates"], 80)
        self.assertEqual(summary["fixed_budget"]["max_wall_seconds"], 900)
        self.assertEqual(summary["budget_consumed"]["parameter_updates"], 0)
        self.assertEqual(summary["budget_consumed"]["wall_seconds"], 0.0)
        for record in records:
            self.assertEqual(record["input_budget"], summary["fixed_budget"])
            self.assertEqual(record["allocated_input"]["train_samples"], 0)
            self.assertEqual(record["allocated_input"]["validation_samples"], 0)
            self.assertEqual(record["updates_completed"], 0)
            self.assertEqual(record["wall_time_seconds"], 0.0)
            self.assertIsNone(record["validation_metrics"])

    def test_seed_tree_is_stable_unique_and_root_seed_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _, records_a = self._run(Path(first))
            _, records_b = self._run(Path(second))
        trees_a = [row["seed_tree"] for row in records_a]
        trees_b = [row["seed_tree"] for row in records_b]
        self.assertEqual(trees_a, trees_b)
        self.assertEqual({tree["root"] for tree in trees_a}, {2693})
        self.assertEqual(len({tree["model"] for tree in trees_a}), 10)
        self.assertTrue(all(row["seed"] == row["seed_tree"]["model"] for row in records_a))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "frozen at 2693"):
                fault_p5_stage2.run_stage2_data_gate(Path(directory), root_seed=2694)

    def test_split_hash_is_frozen_and_no_fold_is_invented(self) -> None:
        expected = hashlib.sha256(fault_p5_stage2.CV_PLAN.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            summary, records = self._run(Path(directory))
        self.assertEqual(summary["split_hash"], expected)
        self.assertEqual({row["split_hash"] for row in records}, {expected})
        split = summary["data_gate"]["development_split"]
        self.assertEqual(split["plan_status"], "not_feasible")
        self.assertEqual(split["requested_n_splits"], 5)
        self.assertEqual(split["effective_n_splits"], 0)
        self.assertEqual(split["fold_count"], 0)

    def test_stage1_hash_links_bind_every_gate_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = self._run(Path(directory))
        expected = {
            "p5_source_locks.json": fault_p5_stage2.SOURCE_LOCK,
            "p5_stage1/summary.json": fault_p5_stage2.STAGE1_SUMMARY,
            "p4_preflight/buffered_cv_plan.json": fault_p5_stage2.CV_PLAN,
            "p4_preflight/blind_test_not_feasible.json": fault_p5_stage2.BLIND_AUDIT,
        }
        for name, path in expected.items():
            self.assertEqual(summary["source_hashes"][name], hashlib.sha256(path.read_bytes()).hexdigest())
        tampered_stage1 = json.loads(fault_p5_stage2.STAGE1_SUMMARY.read_text())
        tampered_stage1["p4_evidence"]["cv_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(fault_p5_stage2.Stage2EvidenceInvalid, "hash links"):
            fault_p5_stage2._validate_source_hash_links(tampered_stage1, summary["source_hashes"])

    def test_runner_has_no_training_random_negative_or_test_input_surface(self) -> None:
        signature = inspect.signature(fault_p5_stage2.run_stage2_data_gate)
        self.assertFalse(any("test" in name.lower() for name in signature.parameters))
        self.assertEqual(set(signature.parameters), {"output_dir", "root_seed"})
        source = Path(fault_p5_stage2.__file__).read_text(encoding="utf-8")
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
            imported_roots.isdisjoint({"torch", "tensorflow", "monai", "numpy", "h5py", "sklearn"})
        )
        self.assertNotIn("test.h5", source)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                fault_p5_stage2.parse_args(["--test-hdf5", "forbidden.h5"])
        with tempfile.TemporaryDirectory() as directory:
            summary, records = self._run(Path(directory))
        self.assertFalse(summary["prohibitions"]["training_performed"])
        self.assertFalse(summary["prohibitions"]["random_negatives_generated"])
        self.assertFalse(summary["test_firewall"]["frozen_test_accessed"])
        for record in records:
            self.assertFalse(record["operations"]["model_built"])
            self.assertFalse(record["operations"]["training_invoked"])
            self.assertFalse(record["operations"]["random_negative_generation_invoked"])
            self.assertFalse(record["test_firewall"]["runner_accepts_test_inputs"])
            self.assertFalse(record["test_firewall"]["frozen_test_accessed"])

    def test_artifacts_are_portable_complete_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, records = self._run(root)
            results_path = root / fault_p5_stage2.RESULTS_FILENAME
            summary_path = root / fault_p5_stage2.SUMMARY_FILENAME
            persisted = json.loads(summary_path.read_text())
            observed_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
        self.assertEqual(len(records), 10)
        self.assertEqual(persisted, summary)
        self.assertEqual(summary["results_artifact"]["sha256"], observed_hash)
        self.assertEqual(summary["results_artifact"]["line_count"], 10)
        self.assertEqual(summary["results_artifact"]["path"], "p5_stage2_results.jsonl")
        serialized = json.dumps(summary, sort_keys=True) + "\n" + "\n".join(
            json.dumps(row, sort_keys=True) for row in records
        )
        self.assertNotIn(str(Path("/", "mnt")) + "/", serialized)
        self.assertNotIn("/".join((".claude", "worktrees")), serialized)


if __name__ == "__main__":
    unittest.main()
