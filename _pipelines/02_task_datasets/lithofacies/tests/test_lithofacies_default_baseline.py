"""Tests for the adopted lithofacies XGBoost default baseline."""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_default_baseline as adoption  # noqa: E402
from _models.lithofacies import xgboost_multisoftprob_window as xgb_model  # noqa: E402


class LithofaciesDefaultBaselineTests(unittest.TestCase):
    def test_adapter_defaults_are_the_adopted_p17_configuration(self) -> None:
        parameters = inspect.signature(xgb_model.XGBoostWindowAdapter).parameters
        self.assertEqual(parameters["max_depth"].default, 3)
        self.assertEqual(parameters["eta"].default, 0.1)
        self.assertEqual(parameters["rounds"].default, 60)

    def test_holdout_and_test_paths_fail_before_open(self) -> None:
        for path in (
            Path("data/test.h5"),
            Path("runtime/frozen_family.npz"),
            Path("known-holdout/results.npz"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "forbidden holdout/test"):
                    adoption.ensure_development_only_paths((path,))
        adoption.ensure_development_only_paths((Path("runtime/development_logo4.npz"),))

    def test_summary_requires_material_gain_and_all_twelve_wins(self) -> None:
        rows = []
        for variant, value in ((adoption.VARIANTS[0], 0.19), (adoption.VARIANTS[1], 0.21)):
            for fold_id in adoption.FOLD_IDS:
                for repeat_id in range(len(adoption.REPEAT_SEEDS)):
                    rows.append(
                        {
                            "variant": variant,
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "metrics": {
                                "fixed_schema_macro_f1": value,
                                "per_class": [
                                    {
                                        "precision": value,
                                        "recall": value,
                                        "f1": value,
                                        "iou": value,
                                    }
                                    for _ in range(9)
                                ],
                            },
                        }
                    )
        summary = adoption.summarize_rows(rows)
        self.assertEqual(summary["comparison"]["default_wins"], 12)
        self.assertAlmostEqual(summary["comparison"]["default_minus_legacy"], 0.02)
        self.assertEqual(summary["decision"]["status"], "ACCEPT_AS_DEFAULT")
        self.assertFalse(summary["decision"]["moment_or_large_model_contribution"])

    def test_registry_declares_xgboost_only_default(self) -> None:
        registry = json.loads((TRACK_DIR / "baseline_registry.json").read_text())
        current = registry["default_baseline"]
        self.assertEqual(
            current["config"], {"eta": 0.1, "max_depth": 3, "rounds": 60}
        )
        self.assertEqual(current["attribution"], "xgboost_hyperparameter_tuning_only")
        self.assertFalse(current["moment_or_large_model_contribution"])

    def test_p17_original_hashes_are_still_exact(self) -> None:
        self.assertEqual(
            adoption.verify_p17_originals(), adoption.P17_ORIGINAL_HASHES
        )


if __name__ == "__main__":
    unittest.main()
