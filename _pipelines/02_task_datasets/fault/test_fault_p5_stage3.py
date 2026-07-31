#!/usr/bin/env python3
"""Fail-closed tests for the fault-prefixed P5 Stage-3 readiness gate."""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))


def _load_fault_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fault_p5_stage3 = _load_fault_module("fault_p5_stage3", "fault_p5_stage3.py")
fault_p5_stage3_visualize = sys.modules["fault_p5_stage3_visualize"]


class FaultP5Stage3ReadinessTests(unittest.TestCase):
    def _run(self, root: Path) -> tuple[dict, list[dict]]:
        summary = fault_p5_stage3.run_stage3_data_readiness(root)
        rows = [
            json.loads(line)
            for line in (root / fault_p5_stage3.RESULTS_FILENAME).read_text().splitlines()
        ]
        return summary, rows

    def test_zero_fold_protocol_emits_one_gate_record_and_no_training_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = self._run(Path(directory))
        self.assertEqual(summary["protocol_baseline_commit"], "16bebd18a0bc722afcbc4b841610bf76ce9503e4")
        self.assertEqual(summary["repeat_seeds"], [1867973658, 2137841944, 3902865753])
        self.assertEqual(summary["frozen_top_models"], [])
        self.assertEqual(summary["effective_fold_count"], 0)
        self.assertEqual(summary["expected_training_cells"], 0)
        self.assertEqual(summary["counts"]["attempted_training_cells"], 0)
        self.assertEqual(summary["status"], "not_rankable")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["record_type"], "data_readiness_gate")
        self.assertEqual(rows[0]["status"], "blocked")
        self.assertIsNone(rows[0]["model_id"])
        self.assertIsNone(rows[0]["fold_id"])
        self.assertIsNone(rows[0]["repeat_seed"])

    def test_readiness_manifest_quantifies_only_supported_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root)
            manifest = json.loads((root / fault_p5_stage3.DATA_MANIFEST_FILENAME).read_text())
        probe = manifest["coverage"]["voxel_probe"]
        self.assertEqual(probe["total_voxels"], 2048)
        self.assertEqual(probe["positive_labels"], 32)
        self.assertEqual(probe["verified_negative_labels"], 0)
        self.assertEqual(probe["unknown_labels"], 2016)
        self.assertAlmostEqual(probe["unknown_fraction"], 2016 / 2048)
        spatial = manifest["coverage"]["spatial"]
        self.assertEqual(spatial["searched_inline_count"], 385)
        self.assertEqual(spatial["development_unique_inlines"], 177)
        self.assertEqual(spatial["complete_annotation_blocks"], 0)
        self.assertEqual(spatial["coverage_audited_volume_count"], 0)
        self.assertIsNone(spatial["coverage_audited_slice_count"])
        self.assertEqual(spatial["slice_coverage_status"], "not_quantifiable")

    def test_stage2_candidates_configuration_and_budget_are_reused_without_hpo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _ = self._run(root)
            data_manifest = json.loads((root / fault_p5_stage3.DATA_MANIFEST_FILENAME).read_text())
        reuse = data_manifest["stage2_reuse"]
        self.assertEqual(tuple(reuse["candidate_model_ids"]), fault_p5_stage3.FIRST_TEN_MODEL_IDS)
        self.assertEqual(reuse["fixed_budget"], fault_p5_stage3.FIXED_BUDGET)
        self.assertEqual(summary["fixed_budget"], fault_p5_stage3.FIXED_BUDGET)
        self.assertEqual(reuse["candidate_configuration_sha256"], hashlib.sha256(fault_p5_stage3.SOURCE_LOCK.read_bytes()).hexdigest())
        for field in (
            "configuration_changed",
            "preprocessing_changed",
            "loss_changed",
            "update_budget_changed",
            "hpo_performed",
        ):
            self.assertFalse(reuse[field])
        self.assertEqual(summary["budget_consumed"]["parameter_updates"], 0)

    def test_repeat_seed_and_p4_split_contract_cannot_be_overridden(self) -> None:
        expected_split_hash = hashlib.sha256(fault_p5_stage3.CV_PLAN.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = self._run(Path(directory))
        self.assertEqual(tuple(summary["repeat_seeds"]), fault_p5_stage3.REPEAT_SEEDS)
        self.assertEqual(summary["split"]["split_hash"], expected_split_hash)
        self.assertEqual(rows[0]["split_hash"], expected_split_hash)
        self.assertEqual(summary["split"]["requested_n_splits"], 5)
        self.assertEqual(summary["split"]["effective_n_splits"], 0)
        self.assertFalse(summary["split"]["temporary_split_created"])
        self.assertFalse(summary["split"]["temporary_fraction_split_allowed"])

    def test_budget_and_fold_train_fit_operations_remain_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = self._run(Path(directory))
        self.assertEqual(summary["fixed_budget"]["max_parameter_updates"], 80)
        self.assertEqual(summary["fixed_budget"]["max_wall_seconds"], 900)
        self.assertEqual(summary["fold_fit_operations"]["preprocessing"], 0)
        self.assertEqual(summary["fold_fit_operations"]["class_weights"], 0)
        self.assertEqual(summary["fold_fit_operations"]["target_transform"], 0)
        self.assertEqual(summary["fold_fit_operations"]["calibration"], 0)
        self.assertEqual(rows[0]["validation_metrics"], None)
        self.assertFalse(rows[0]["operations"]["model_built"])
        self.assertFalse(rows[0]["operations"]["training_invoked"])
        self.assertFalse(rows[0]["operations"]["prediction_generated"])

    def test_runner_has_no_test_training_hpo_or_temporary_split_input_surface(self) -> None:
        signature = inspect.signature(fault_p5_stage3.run_stage3_data_readiness)
        self.assertEqual(set(signature.parameters), {"output_dir"})
        source = Path(fault_p5_stage3.__file__).read_text(encoding="utf-8")
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
                {"torch", "tensorflow", "monai", "numpy", "h5py", "sklearn", "optuna"}
            )
        )
        for forbidden in (
            "test.h5",
            "baseline_metrics.json",
            "split_manifest.json",
            "train_test_split",
            "random_split",
        ):
            self.assertNotIn(forbidden, source)
        for option in ("--test-hdf5", "--fold", "--model-id", "--repeat-seed", "--hpo"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    fault_p5_stage3.parse_args([option, "forbidden"])
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = self._run(Path(directory))
        self.assertFalse(summary["test_firewall"]["frozen_test_accessed"])
        self.assertFalse(summary["test_firewall"]["test_metrics_accessed"])
        self.assertFalse(rows[0]["test_firewall"]["test_labels_accessed"])

    def test_duplicate_cells_cross_lane_pollution_and_invented_cells_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, rows = self._run(Path(directory))
        with self.assertRaisesRegex(fault_p5_stage3.FaultStage3EvidenceInvalid, "duplicate"):
            fault_p5_stage3.validate_result_records([rows[0], copy.deepcopy(rows[0])])
        polluted = copy.deepcopy(rows[0])
        polluted["lane"] = "another_track_lane"
        with self.assertRaisesRegex(fault_p5_stage3.FaultStage3EvidenceInvalid, "pollution"):
            fault_p5_stage3.validate_result_records([polluted])
        invented = copy.deepcopy(rows[0])
        invented.update(
            {
                "record_type": "training_cell",
                "model_id": "monai_segresnet",
                "fold_id": 0,
                "repeat_seed": 1867973658,
            }
        )
        with self.assertRaisesRegex(fault_p5_stage3.FaultStage3EvidenceInvalid, "zero legal"):
            fault_p5_stage3.validate_result_records([invented])

    def test_completion_threshold_is_fail_closed(self) -> None:
        self.assertEqual(fault_p5_stage3.completion_assessment(10, 7)["status"], "not_rankable")
        self.assertEqual(fault_p5_stage3.completion_assessment(10, 8)["status"], "rankable")
        zero = fault_p5_stage3.completion_assessment(0, 0)
        self.assertEqual(zero["status"], "not_rankable")
        self.assertIsNone(zero["completion_rate"])
        with self.assertRaises(ValueError):
            fault_p5_stage3.completion_assessment(2, 3)

    def test_oof_leaderboard_and_minimum_contract_are_explicitly_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _ = self._run(root)
            oof = json.loads((root / fault_p5_stage3.OOF_MANIFEST_FILENAME).read_text())
            leaderboard = json.loads((root / fault_p5_stage3.LEADERBOARD_PATH).read_text())
            data_manifest = json.loads((root / fault_p5_stage3.DATA_MANIFEST_FILENAME).read_text())
        self.assertEqual(oof["status"], "not_generated")
        self.assertEqual(oof["prediction_count"], 0)
        self.assertEqual(oof["prediction_artifacts"], [])
        self.assertEqual(leaderboard["status"], "not_rankable")
        self.assertEqual(leaderboard["entries"], [])
        self.assertEqual(summary["leaderboards"][0]["status"], "not_rankable")
        contract_ids = {
            item["contract_id"] for item in data_manifest["minimum_data_contract_to_unblock"]
        }
        self.assertEqual(
            contract_ids,
            {
                "coverage_manifest",
                "verified_negative_mask",
                "unknown_and_proxy_masks",
                "buffered_development_folds",
                "fold_train_fit_provenance",
                "test_firewall",
            },
        )

    def test_visualizations_rebuild_from_readiness_manifest_only(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            self._run(first_root)
            data_manifest = first_root / fault_p5_stage3.DATA_MANIFEST_FILENAME
            rebuilt = fault_p5_stage3_visualize.build_figures(data_manifest, Path(second))
            original_manifest = json.loads(
                (first_root / fault_p5_stage3.VISUALIZATION_MANIFEST_FILENAME).read_text()
            )
            for figure in original_manifest["figures"]:
                element_tree.parse(first_root / figure["path"])
        self.assertEqual(
            [item["sha256"] for item in rebuilt],
            [item["sha256"] for item in original_manifest["figures"]],
        )
        self.assertEqual(len(rebuilt), 3)
        self.assertFalse(original_manifest["frozen_test_accessed"])
        self.assertFalse(original_manifest["historical_test_metrics_accessed"])

    def test_artifact_manifest_hashes_every_portable_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, rows = self._run(root)
            manifest = json.loads(
                (root / fault_p5_stage3.ARTIFACT_MANIFEST_FILENAME).read_text()
            )
            for artifact in manifest["artifacts"]:
                path = root / artifact["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
                self.assertEqual(path.stat().st_size, artifact["bytes"])
            serialized = json.dumps(summary, sort_keys=True) + json.dumps(rows, sort_keys=True)
        self.assertEqual(len(manifest["artifacts"]), 9)
        self.assertFalse(manifest["large_artifacts_committed"])
        self.assertEqual(manifest["checkpoint_count"], 0)
        self.assertEqual(manifest["prediction_payload_count"], 0)
        self.assertNotIn(str(Path("/", "mnt")) + "/", serialized)
        self.assertNotIn("/".join((".claude", "worktrees")), serialized)


if __name__ == "__main__":
    unittest.main()
