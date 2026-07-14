#!/usr/bin/env python3
"""Fail-closed tests for the fault-prefixed P5 Stage-4 confirmation gate."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parent


def _load_fault_stage4():
    name = "fault_p5_stage4"
    spec = importlib.util.spec_from_file_location(name, TRACK_DIR / "fault_p5_stage4.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load fault_p5_stage4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fault_p5_stage4 = _load_fault_stage4()


class FaultP5Stage4ConfirmationTests(unittest.TestCase):
    def _run(self, root: Path) -> tuple[dict, dict, dict]:
        confirmation = fault_p5_stage4.run_stage4_confirmation(root)
        reuse = json.loads(
            (root / fault_p5_stage4.VISUALIZATION_REUSE_FILENAME).read_text()
        )
        artifacts = json.loads(
            (root / fault_p5_stage4.STAGE4_ARTIFACT_MANIFEST_FILENAME).read_text()
        )
        return confirmation, reuse, artifacts

    @staticmethod
    def _copy_stage3(root: Path) -> Path:
        copied = root / "p5_stage3"
        shutil.copytree(fault_p5_stage4.STAGE3_DIR, copied)
        return copied

    @staticmethod
    def _rewrite_manifest(stage3: Path, relative: str, mutate) -> None:
        path = stage3 / relative
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact_path = stage3 / fault_p5_stage4.ARTIFACT_MANIFEST_FILENAME
        artifact = json.loads(artifact_path.read_text())
        for record in artifact["artifacts"]:
            if record["path"] == relative:
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                record["bytes"] = path.stat().st_size
                break
        else:
            raise AssertionError(f"artifact record missing for {relative}")
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    def test_confirmation_has_exact_blocked_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, _ = self._run(Path(directory))
        self.assertEqual(confirmation["status"], "blocked")
        self.assertEqual(
            confirmation["baseline_commit"],
            "c9ac3cf8e18191c48cdb1ddfafa34355bf1548c7",
        )
        self.assertEqual(
            confirmation["runner_sha256"],
            hashlib.sha256(Path(fault_p5_stage4.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(confirmation["ranking_status"], "not_rankable")
        self.assertEqual(confirmation["reason"], "NO_VALID_FAULT_DEVELOPMENT_FOLDS")
        self.assertIsNone(confirmation["frozen_winner"])
        self.assertFalse(confirmation["refit_executed"])
        self.assertFalse(confirmation["holdout_accessed"])
        self.assertEqual(confirmation["effective_fold_count"], 0)
        self.assertEqual(confirmation["verified_negative_labels"], 0)

    def test_no_training_refit_prediction_metric_or_holdout_operation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, artifacts = self._run(Path(directory))
        self.assertTrue(all(value is False for value in confirmation["operations"].values()))
        self.assertTrue(all(value is False for value in confirmation["test_firewall"].values()))
        self.assertEqual(artifacts["checkpoint_count"], 0)
        self.assertEqual(artifacts["prediction_payload_count"], 0)
        self.assertFalse(artifacts["refit_executed"])
        self.assertFalse(artifacts["holdout_accessed"])

    def test_minimum_unblock_contract_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, _ = self._run(Path(directory))
        contract_ids = tuple(
            item["contract_id"] for item in confirmation["minimum_unblock_data_contract"]
        )
        self.assertEqual(contract_ids, fault_p5_stage4.EXPECTED_CONTRACT_IDS)
        self.assertTrue(
            all(item["minimum"].strip() for item in confirmation["minimum_unblock_data_contract"])
        )

    def test_prior_test_metadata_exposure_is_honest_and_non_consuming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, _ = self._run(Path(directory))
        exposure = confirmation["prior_test_metadata_exposure"]
        self.assertTrue(exposure["metadata_exposure_present"])
        self.assertEqual(exposure["historical_test_role"], "regression_evidence_only")
        self.assertEqual(exposure["independence_status"], "blocked")
        self.assertFalse(exposure["prior_metric_values_re_emitted"])
        self.assertFalse(exposure["historical_test_artifacts_read_by_stage4"])
        self.assertFalse(exposure["historical_test_metrics_read_by_stage4"])

    def test_visualization_is_hash_reuse_without_prediction_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            confirmation, reuse, _ = self._run(Path(directory))
        self.assertEqual(reuse["status"], "reused_readiness_only")
        self.assertEqual(reuse["reuse_mode"], "hash_reference_only")
        self.assertEqual(len(reuse["figures"]), 3)
        self.assertIsNone(reuse["prediction_source"])
        self.assertEqual(reuse["prediction_artifacts"], [])
        self.assertFalse(reuse["prediction_fabricated"])
        self.assertEqual(confirmation["visualization_reuse"]["prediction_count"], 0)
        for figure in reuse["figures"]:
            source = TRACK_DIR / "_outputs" / figure["path"]
            self.assertTrue(source.is_file())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), figure["sha256"])
            self.assertEqual(source.stat().st_size, figure["bytes"])

    def test_stage4_artifact_manifest_hashes_every_portable_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            confirmation, _, artifacts = self._run(root)
            for record in artifacts["artifacts"]:
                path = root / record["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
                self.assertEqual(path.stat().st_size, record["bytes"])
            serialized = json.dumps(confirmation, sort_keys=True)
        self.assertEqual(len(artifacts["artifacts"]), 2)
        self.assertNotIn(str(Path("/", "mnt")) + "/", serialized)
        self.assertNotIn("/".join((".claude", "worktrees")), serialized)

    def test_modified_stage3_artifact_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage3 = self._copy_stage3(Path(directory))
            figure = stage3 / fault_p5_stage4.EXPECTED_FIGURES[0]
            figure.write_text(figure.read_text() + "\n")
            with self.assertRaisesRegex(
                fault_p5_stage4.FaultStage4ConfirmationError,
                "hash mismatch",
            ):
                fault_p5_stage4.validate_frozen_stage3(stage3)

    def test_self_consistent_but_rewritten_stage3_source_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage3 = self._copy_stage3(Path(directory))
            self._rewrite_manifest(
                stage3,
                fault_p5_stage4.SUMMARY_FILENAME,
                lambda payload: payload.__setitem__("root_seed", 999),
            )
            with self.assertRaisesRegex(
                fault_p5_stage4.FaultStage4ConfirmationError,
                "source lock",
            ):
                fault_p5_stage4.validate_frozen_stage3(stage3)

    def test_winner_fold_or_holdout_tampering_fails_closed(self) -> None:
        mutations = (
            lambda payload: payload.__setitem__("frozen_top_models", ["forbidden_winner"]),
            lambda payload: payload.__setitem__("effective_fold_count", 1),
            lambda payload: payload["test_firewall"].__setitem__("frozen_test_accessed", True),
        )
        patterns = ("winner", "fold", "firewall opened")
        for mutate, pattern in zip(mutations, patterns):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                stage3 = self._copy_stage3(Path(directory))
                self._rewrite_manifest(stage3, fault_p5_stage4.SUMMARY_FILENAME, mutate)
                with self.assertRaisesRegex(
                    fault_p5_stage4.FaultStage4ConfirmationError,
                    pattern,
                ):
                    fault_p5_stage4.validate_frozen_stage3(stage3)

    def test_negative_or_unknown_audit_claim_tampering_fails_closed(self) -> None:
        mutations = (
            lambda payload: payload["coverage"]["negative_provenance"].update(
                {"verified_negative_labels": 1, "audit_status": "complete"}
            ),
            lambda payload: payload["coverage"]["unknown_provenance"].update(
                {"status": "ready", "source_mask_audit_status": "complete"}
            ),
        )
        patterns = ("negative count", "unknown-mask provenance")
        for mutate, pattern in zip(mutations, patterns):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                stage3 = self._copy_stage3(Path(directory))
                self._rewrite_manifest(stage3, fault_p5_stage4.DATA_MANIFEST_FILENAME, mutate)
                with self.assertRaisesRegex(
                    fault_p5_stage4.FaultStage4ConfirmationError,
                    pattern,
                ):
                    fault_p5_stage4.validate_frozen_stage3(stage3)

    def test_cli_and_source_have_no_scientific_override_surface(self) -> None:
        signature = inspect.signature(fault_p5_stage4.run_stage4_confirmation)
        self.assertEqual(set(signature.parameters), {"output_dir"})
        source = Path(fault_p5_stage4.__file__).read_text()
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
        for option in (
            "--winner",
            "--model-id",
            "--fold",
            "--refit",
            "--holdout",
            "--test-hdf5",
            "--prediction",
        ):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    fault_p5_stage4.parse_args([option, "forbidden"])


if __name__ == "__main__":
    unittest.main()
