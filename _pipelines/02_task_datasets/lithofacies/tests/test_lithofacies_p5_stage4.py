"""Contract and canonical-evidence tests for lithofacies P5 Stage-4."""
from __future__ import annotations

import copy
import hashlib
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

import lithofacies_p5_stage4 as stage4  # noqa: E402


class LithofaciesStage4ContractTests(unittest.TestCase):
    def test_stage3_winner_and_frozen_budget_are_exact(self) -> None:
        winner = stage4.verify_stage3_winner()
        self.assertEqual(winner["winner_model_id"], "xgboost_multisoftprob_window")
        self.assertEqual(winner["winner_rank"], 1)
        self.assertEqual(winner["legal_cells"], 12)
        self.assertEqual(winner["primary_metric"], "fixed_schema_macro_f1_mean")
        self.assertEqual(winner["supported_class_metric_role"], "diagnostic_only")
        data = stage4.verify_data_contract_manifest()
        prior = stage4.verify_prior_consumption()
        frozen = stage4._frozen_config(winner, data, prior)
        self.assertEqual(frozen["seed"], 2693)
        self.assertEqual(frozen["model_config"]["rounds"], 40)
        self.assertEqual(frozen["model_config"]["max_depth"], 2)
        self.assertEqual(frozen["budget"]["fold_train_sample_limit"], 320)
        self.assertEqual(frozen["budget"]["fold_validation_sample_limit"], 160)
        self.assertFalse(frozen["budget"]["hpo"])
        self.assertTrue(frozen["prior_holdout_evidence"]["prior_test_consumed"])
        self.assertFalse(frozen["prior_holdout_evidence"]["fresh_blind"])

    def test_tampered_winner_metric_or_budget_fails_closed(self) -> None:
        summary = stage4._read_json(stage4.STAGE3_SUMMARY)
        board = stage4._read_json(stage4.STAGE3_LEADERBOARD)
        results = stage4._read_jsonl(stage4.STAGE3_RESULTS)
        changed = copy.deepcopy(board)
        changed["primary_metric"] = "supported_class_macro_f1_mean"
        with self.assertRaisesRegex(ValueError, "primary metric"):
            stage4.validate_stage3_winner_payloads(summary, changed, results)
        changed = copy.deepcopy(summary)
        changed["budget"]["xgboost"]["rounds"] = 41
        with self.assertRaisesRegex(ValueError, "budget"):
            stage4.validate_stage3_winner_payloads(changed, board, results)
        changed_results = copy.deepcopy(results)
        winner = next(
            result for result in changed_results
            if result["model_id"] == stage4.WINNER_MODEL_ID
        )
        winner["frozen_test_accessed"] = True
        with self.assertRaisesRegex(RuntimeError, "holdout"):
            stage4.validate_stage3_winner_payloads(summary, board, changed_results)

    def test_logo4_f5_and_prior_consumption_contract(self) -> None:
        data = stage4.verify_data_contract_manifest()
        self.assertEqual(set(data["development_families"]), set(stage4.DEVELOPMENT_FAMILIES))
        self.assertEqual(data["holdout_family"], "15/9-F-5")
        self.assertEqual(data["development_samples"], 447)
        self.assertEqual(data["holdout_samples"], 120)
        self.assertEqual(tuple(data["holdout_class_support"]), stage4.EXPECTED_HOLDOUT_SUPPORT)
        prior = stage4.verify_prior_consumption()
        self.assertTrue(prior["prior_test_consumed"])
        self.assertFalse(prior["fresh_blind"])
        self.assertEqual(prior["evidence_class"], "previously_seen_reusable_holdout")

    def test_state_machine_is_ordered_single_use_and_p4_independent(self) -> None:
        source = (TRACK_DIR / "lithofacies_p5_stage4.py").read_text(encoding="utf-8")
        self.assertNotIn("p4_runs", source)
        self.assertNotIn("ExperimentLifecycle", source)
        self.assertEqual(source.count(' / "test.h5"'), 1)
        self.assertNotIn('"test.h5"', inspect.getsource(stage4.prepare_development))
        self.assertNotIn('"test.h5"', inspect.getsource(stage4.refit_winner))
        self.assertLess(source.index("STATE_CONFIG_FROZEN"), source.index("STATE_REFIT_COMPLETE"))
        self.assertLess(source.index("STATE_REFIT_COMPLETE"), source.index("STATE_HOLDOUT_CONSUMED"))
        self.assertLess(source.index("STATE_HOLDOUT_CONSUMED"), source.index("STATE_CONFIRMATION_COMPLETE"))
        with tempfile.TemporaryDirectory(dir=TRACK_DIR) as directory:
            output = Path(directory)
            stage4._write_lifecycle(
                output,
                {
                    "schema_version": stage4.LIFECYCLE_SCHEMA,
                    "state": stage4.STATE_CONFIG_FROZEN,
                },
            )
            stage4.require_state(output, stage4.STATE_CONFIG_FROZEN)
            with self.assertRaisesRegex(RuntimeError, "requires REFIT_COMPLETE"):
                stage4.require_state(output, stage4.STATE_REFIT_COMPLETE)

    def test_no_midpoint_or_test_calibration_policy(self) -> None:
        source = (TRACK_DIR / "lithofacies_p5_stage4.py").read_text(encoding="utf-8")
        self.assertNotIn("top_md_m +", source)
        self.assertNotIn("base_md_m +", source)
        self.assertNotIn("fit_temperature", source)
        self.assertIn("none_test_labels_never_fit", source)
        self.assertIn("midpoint fabrication forbidden", source)


class LithofaciesStage4CanonicalArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = TRACK_DIR / "_outputs" / "p5_stage4_confirmation"
        if not (cls.output / "summary.json").is_file():
            raise unittest.SkipTest("canonical Stage-4 confirmation has not been executed")

    def test_lifecycle_config_refit_and_holdout_evidence(self) -> None:
        lifecycle = stage4._read_json(self.output / "lifecycle.json")
        frozen = stage4._read_json(self.output / "frozen_config.json")
        development = stage4._read_json(self.output / "development_evidence.json")
        refit = stage4._read_json(self.output / "refit_evidence.json")
        holdout = stage4._read_json(self.output / "holdout_data_evidence.json")
        self.assertEqual(lifecycle["state"], stage4.STATE_CONFIRMATION_COMPLETE)
        self.assertEqual(
            [transition["state"] for transition in lifecycle["transitions"]],
            [
                stage4.STATE_CONFIG_FROZEN,
                stage4.STATE_REFIT_COMPLETE,
                stage4.STATE_HOLDOUT_CONSUMED,
                stage4.STATE_CONFIRMATION_COMPLETE,
            ],
        )
        self.assertTrue(lifecycle["prior_test_consumed"])
        self.assertFalse(lifecycle["fresh_blind"])
        self.assertFalse(lifecycle["p4_state_reset"])
        self.assertEqual(frozen["model_id"], stage4.WINNER_MODEL_ID)
        self.assertEqual(frozen["seed"], 2693)
        self.assertEqual(frozen["model_config"]["rounds"], 40)
        self.assertEqual(frozen["model_config"]["max_depth"], 2)
        self.assertEqual(set(development["development_families"]), set(stage4.DEVELOPMENT_FAMILIES))
        self.assertEqual(set(development["selected_training_families"]), set(stage4.DEVELOPMENT_FAMILIES))
        self.assertEqual(development["full_development_samples"], 447)
        self.assertEqual(development["selected_training_samples"], 320)
        self.assertFalse(development["holdout_accessed"])
        self.assertEqual(refit["status"], stage4.STATE_REFIT_COMPLETE)
        self.assertEqual(refit["selected_training_samples"], 320)
        self.assertFalse(refit["holdout_accessed"])
        self.assertEqual(holdout["holdout_family"], stage4.TEST_FAMILY)
        self.assertEqual(holdout["holdout_samples"], 120)
        self.assertEqual(tuple(holdout["class_support"]), stage4.EXPECTED_HOLDOUT_SUPPORT)
        self.assertTrue(holdout["consumed_before_hdf5_open"])

    def test_metrics_predictions_and_fixed_nine_support(self) -> None:
        metrics = stage4._read_json(self.output / "metrics.json")
        predictions = stage4._read_json(self.output / "predictions.json")
        self.assertEqual(metrics["primary_metric"], "fixed_schema_macro_f1")
        self.assertEqual(metrics["supported_class_metric_role"], "diagnostic_only")
        self.assertEqual(metrics["calibration_fit"], "none_test_labels_never_fit")
        self.assertEqual(metrics["fixed_schema_macro_f1"], metrics["macro_f1"])
        self.assertEqual(metrics["evaluated_samples"], 120)
        self.assertEqual(len(metrics["per_class"]), 9)
        self.assertEqual(tuple(row["support"] for row in metrics["per_class"]), stage4.EXPECTED_HOLDOUT_SUPPORT)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (9, 9))
        self.assertEqual(len(predictions["records"]), 120)
        for record in predictions["records"]:
            probability = np.asarray(record["probabilities"], dtype=np.float64)
            self.assertEqual(probability.shape, (9,))
            self.assertTrue(np.isfinite(probability).all())
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=7)
            self.assertEqual(record["family_id"], stage4.TEST_FAMILY)
        for key in (
            "accuracy",
            "fixed_schema_macro_f1",
            "supported_class_macro_f1",
            "negative_log_likelihood",
            "multiclass_brier",
            "expected_calibration_error",
        ):
            self.assertTrue(np.isfinite(float(metrics[key])), key)

    def test_artifact_hashes_visualizations_and_portability(self) -> None:
        manifest = stage4._read_json(self.output / "artifact_manifest.json")
        visualization = stage4._read_json(self.output / "visualization_manifest.json")
        self.assertFalse(manifest["runtime_checkpoint_committed"])
        self.assertFalse(manifest["runtime_batches_committed"])
        for artifact in manifest["artifacts"]:
            path = self.output / artifact["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(artifact["bytes"], path.stat().st_size)
        figure_ids = {figure["figure_id"] for figure in visualization["figures"]}
        self.assertEqual(
            figure_ids,
            {
                "fixed9_confusion",
                "fixed9_per_class_pr_f1_support",
                "calibration_reliability",
                "continuous_depth_facies_track",
            },
        )
        depth = next(
            figure for figure in visualization["figures"]
            if figure["figure_id"] == "continuous_depth_facies_track"
        )
        if depth["status"] == "not_feasible":
            self.assertEqual(depth["finite_md_rows"], 0)
        for path in self.output.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/mnt/data/", text)
            self.assertNotIn(".claude/worktrees", text)


if __name__ == "__main__":
    unittest.main()
