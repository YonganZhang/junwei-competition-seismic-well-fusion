from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "_outputs" / "p19_training_diagnostics"
sys.path.insert(0, str(HERE.parent))

import p17_foundation_geostatistics as p17  # noqa: E402
import p19_meta_purged_geostatistics as p19  # noqa: E402


class P19TrainingDiagnosticsArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        cls.verification = json.loads(
            (OUTPUT / "verification.json").read_text(encoding="utf-8")
        )

    def test_meta_purged_result_is_stable_and_disabled(self) -> None:
        result = self.summary["p19_meta_purged"]
        self.assertLess(result["rmse"], self.summary["baseline"]["rmse"])
        self.assertEqual(result["fold_wins"], 5)
        self.assertEqual(result["fold_losses"], 0)
        self.assertLess(result["bootstrap_ci95"][1], 0.0)
        self.assertFalse(self.summary["decision"]["default_enabled"])
        self.assertFalse(self.summary["decision"]["pretrained_contribution_claimed"])

    def test_cross_fold_meta_overlap_is_explicit(self) -> None:
        overlap = self.summary["cross_fold_meta_overlap"]
        self.assertEqual([row["held_fold"] for row in overlap], list(range(5)))
        self.assertEqual(
            [row["unique_coordinates"] for row in overlap], [58, 42, 24, 27, 45]
        )
        self.assertTrue(all(row["removed_occurrences"] > 0 for row in overlap))

    def test_gradient_diagnosis_records_zero_first_step_and_scale_mismatch(self) -> None:
        diagnosis = self.summary["gradient_diagnosis"]
        self.assertEqual(diagnosis["prefix_batch_shape"], [4, 3, 161, 1200])
        self.assertEqual(diagnosis["tail_input_shape"], [12, 161, 1200])
        self.assertEqual(diagnosis["trainable_encoder_parameters"], 17_298_000)
        self.assertTrue(diagnosis["zero_initialized_output_layer"])
        self.assertTrue(
            all(value == 0.0 for value in diagnosis["encoder_gradient_step1"].values())
        )
        self.assertTrue(
            all(value > 0.0 for value in diagnosis["encoder_gradient_step3"].values())
        )
        self.assertLess(
            max(diagnosis["encoder_relative_update_three_steps"]),
            min(diagnosis["head_relative_update_three_steps"]),
        )

    def test_strict_screens_do_not_beat_accepted_result(self) -> None:
        accepted = self.summary["p19_meta_purged"]["rmse"]
        screens = self.summary["route_screens"]
        self.assertGreater(screens["frozen_mlp_best_relu_rmse"], accepted)
        self.assertGreater(screens["extended_metric_nested_rmse"], accepted)
        self.assertGreater(screens["nested_regression_kriging_rmse"], accepted)
        self.assertGreater(screens["stratigraphic_metric_nested_rmse"], accepted)
        self.assertLess(
            screens["same_oof_regression_kriging_rmse_non_promotable"], accepted
        )

    def test_independent_verification_and_holdout_firewall(self) -> None:
        self.assertEqual(self.verification["status"], "PASSED")
        self.assertEqual(self.verification["rows"], 10_240)
        self.assertEqual(self.verification["fold_wins_recomputed"], 5)
        self.assertTrue(self.verification["meta_purge_checked"])
        self.assertFalse(self.verification["firewall"]["test_h5_opened"])
        self.assertFalse(self.summary["firewall"]["frozen_holdout_opened"])

    def test_meta_purge_helper_removes_only_forbidden_training_rows(self) -> None:
        fold = p17.FoldSamples(
            fold_id=1,
            train_target=np.asarray([0.1, 0.2, 0.3]),
            train_raw_features=np.arange(18, dtype=np.float64).reshape(3, 6),
            validation_raw_features=np.zeros((1, 6), dtype=np.float64),
            train_indices_kji=np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            validation_indices_kji=np.asarray([[9, 9, 9]]),
            source_hashes={},
        )
        purged, removed = p19._without_coordinates(fold, {(1, 1, 1)})  # noqa: SLF001
        self.assertEqual(removed, 1)
        np.testing.assert_array_equal(
            purged.train_indices_kji, np.asarray([[0, 0, 0], [2, 2, 2]])
        )
        np.testing.assert_allclose(purged.train_target, [0.1, 0.3])

    def test_cli_has_no_test_or_holdout_argument(self) -> None:
        help_text = p19._parser().format_help()  # noqa: SLF001
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)


if __name__ == "__main__":
    unittest.main()
