"""Contract and archived-evidence tests for the T3 Chronos-2 lane."""
from __future__ import annotations

import importlib
import inspect
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


adapter = importlib.import_module("_models.sweetspot.p7_chronos2")
runner = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p7.runner"
)


class FakeForecast:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeForecast":
        return self

    def float(self) -> "FakeForecast":
        return self

    def cpu(self) -> "FakeForecast":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakePipeline:
    quantiles = adapter.MEDIAN_QUANTILE + np.arange(-10, 11) * 0.05

    def __init__(self) -> None:
        self.inputs = None
        self.kwargs = None

    def predict(self, inputs, **kwargs):
        self.inputs = inputs
        self.kwargs = kwargs
        outputs = []
        for index in range(len(inputs)):
            values = np.zeros((1, 21, adapter.PREDICTION_LENGTH), dtype=np.float32)
            values[:, 10, :] = np.arange(adapter.PREDICTION_LENGTH) - 5 + index
            outputs.append(FakeForecast(values))
        return outputs


class SweetspotP7Chronos2ContractTests(unittest.TestCase):
    def test_source_lock_is_exact_and_scoped_only_to_t3(self) -> None:
        lock = runner._validate_source_lock()
        self.assertEqual(lock["model_id"], adapter.MODEL_ID)
        self.assertEqual(lock["revision"], adapter.MODEL_REVISION)
        self.assertEqual(lock["license"], "Apache-2.0")
        self.assertEqual(lock["approved_scope"], [
            "T3 future-30-day oil-production forecasting from causal production history"
        ])
        self.assertEqual(len(lock["experimental_scope"]), 1)
        self.assertIn("fault segmentation", lock["excluded_scope"])
        self.assertIn("3D reconstruction", lock["excluded_scope"])

    def test_adapter_builds_past_only_covariates(self) -> None:
        sequence = np.arange(2 * 7 * 30, dtype=np.float32).reshape(2, 7, 30)
        inputs = adapter.build_inputs(sequence, mode="past_covariates")
        self.assertEqual(len(inputs), 2)
        for index, item in enumerate(inputs):
            self.assertEqual(set(item), {"target", "past_covariates"})
            self.assertNotIn("future_covariates", item)
            np.testing.assert_array_equal(item["target"], sequence[index, 0])
            self.assertEqual(set(item["past_covariates"]), set(adapter.PAST_COVARIATE_NAMES))

    def test_adapter_rejects_wrong_shape_infinity_and_all_missing_target(self) -> None:
        with self.assertRaises(ValueError):
            adapter.build_inputs(np.zeros((2, 7, 29), dtype=np.float32))
        invalid = np.zeros((2, 7, 30), dtype=np.float32)
        invalid[0, 1, 0] = np.inf
        with self.assertRaises(ValueError):
            adapter.build_inputs(invalid)
        missing_target = np.zeros((2, 7, 30), dtype=np.float32)
        missing_target[0, 0, :] = np.nan
        with self.assertRaises(ValueError):
            adapter.build_inputs(missing_target)

    def test_adapter_preserves_native_nan_mask_in_past_covariates(self) -> None:
        sequence = np.zeros((1, 7, 30), dtype=np.float32)
        sequence[0, 4, 3] = np.nan
        item = adapter.build_inputs(sequence, mode="past_covariates")[0]
        self.assertTrue(np.isnan(item["past_covariates"]["avg_downhole_pressure"][3]))

    def test_t4_adapter_uses_water_target_and_seven_day_history(self) -> None:
        sequence = np.arange(2 * 7 * 7, dtype=np.float32).reshape(2, 7, 7)
        inputs = adapter.build_water_risk_inputs(sequence)
        self.assertEqual(len(inputs), 2)
        np.testing.assert_array_equal(inputs[0]["target"], sequence[0, 2])
        self.assertEqual(inputs[0]["target"].shape, (7,))
        self.assertNotIn("future_covariates", inputs[0])
        self.assertNotIn("bore_wat_vol", inputs[0]["past_covariates"])

    def test_t4_risk_score_is_derived_from_future_water_forecast(self) -> None:
        pipeline = FakePipeline()
        sequence = np.ones((2, 7, 7), dtype=np.float32)
        scores, quantiles = adapter.forecast_water_risk_scores(pipeline, sequence)
        self.assertEqual(scores.shape, (2, 21))
        self.assertEqual(quantiles.shape, (21,))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertFalse(pipeline.kwargs["cross_learning"])

    def test_forecast_uses_median_nonnegative_daily_mean_and_no_cross_learning(self) -> None:
        pipeline = FakePipeline()
        sequence = np.ones((2, 7, 30), dtype=np.float32)
        daily, point = adapter.forecast_oil(pipeline, sequence, batch_size=8)
        self.assertEqual(daily.shape, (2, 30))
        self.assertTrue(np.all(daily >= 0.0))
        np.testing.assert_allclose(point, daily.mean(axis=1))
        self.assertEqual(pipeline.kwargs["prediction_length"], 30)
        self.assertFalse(pipeline.kwargs["cross_learning"])

    def test_blend_weight_uses_only_three_train_arrays(self) -> None:
        signature = inspect.signature(runner.choose_convex_weight)
        self.assertEqual(
            list(signature.parameters)[:3],
            ["foundation_prediction", "history_prediction", "train_target"],
        )
        self.assertNotIn("validation_target", signature.parameters)
        foundation = np.asarray([1.0, 2.0, 3.0])
        history = np.asarray([3.0, 2.0, 1.0])
        weight, error = runner.choose_convex_weight(
            foundation, history, foundation, grid=(0.0, 0.5, 1.0)
        )
        self.assertEqual(weight, 1.0)
        self.assertEqual(error, 0.0)

    def test_cli_has_no_holdout_or_test_argument(self) -> None:
        options = {action.dest for action in runner._parser()._actions}
        self.assertNotIn("test", options)
        self.assertNotIn("holdout", options)
        self.assertNotIn("frozen_test", options)

    def test_cuda_requires_explicit_gpu_lock(self) -> None:
        with self.assertRaises(ValueError):
            with runner.gpu_lock(None, device="cuda"):
                pass


class SweetspotP7Chronos2ArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = runner.DEFAULT_OUTPUT_DIR / "summary.json"
        if not cls.path.is_file():
            raise unittest.SkipTest("run the P7 Chronos-2 runner to archive evidence")
        cls.summary = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_real_source_locked_weights_were_used(self) -> None:
        foundation = self.summary["foundation"]
        self.assertTrue(foundation["real_pretrained_weights_loaded"])
        self.assertEqual(foundation["model_id"], adapter.MODEL_ID)
        self.assertEqual(foundation["revision"], adapter.MODEL_REVISION)
        lock = runner._validate_source_lock()
        self.assertEqual(
            foundation["artifact"]["weights_sha256"],
            lock["weights_sha256"],
        )

    def test_same_four_folds_and_no_holdout_access(self) -> None:
        evaluation = self.summary["evaluation"]
        self.assertEqual(evaluation["folds"], [0, 1, 2, 3])
        self.assertFalse(evaluation["known_holdout_accessed"])
        self.assertFalse(evaluation["frozen_test_accessed"])
        self.assertFalse(evaluation["validation_labels_used_for_blend_selection"])
        for method in self.summary["methods"].values():
            self.assertEqual(method["fold_count"], 4)

    def test_foundation_hybrid_beats_archived_xgboost_and_naive_control(self) -> None:
        decision = self.summary["decision"]
        self.assertEqual(decision["promotion_status"], "PROMOTE")
        self.assertLess(
            decision["selected_macro_fold_mae"],
            decision["archived_xgboost_macro_fold_mae"],
        )
        self.assertLess(
            decision["selected_macro_fold_mae"],
            decision["causal_history_mean_macro_fold_mae"],
        )
        self.assertGreater(decision["mae_reduction_vs_archived_xgboost_percent"], 0.0)

    def test_non_temporal_tracks_are_explicitly_excluded(self) -> None:
        decision = self.summary["decision"]
        self.assertEqual(decision["non_temporal_tracks_status"], "FOUNDATION_NOT_APPLICABLE")
        self.assertEqual(decision["t4_status"], "REJECTED_NO_GAIN")

    def test_t4_was_evaluated_but_rejected_without_gain(self) -> None:
        t4 = self.summary["t4_experiment"]
        self.assertEqual(t4["promotion_status"], "REJECT")
        self.assertEqual(len(t4["folds"]), 3)
        self.assertFalse(t4["known_holdout_accessed"])
        self.assertFalse(t4["frozen_test_accessed"])
        self.assertLess(
            t4["macro_fold_average_precision"],
            t4["archived_p5_baseline"]["primary_mean"],
        )
        self.assertTrue(
            all(
                not row["validation_labels_used_for_quantile_selection"]
                for row in t4["folds"]
            )
        )

    def test_no_raw_predictions_or_checkpoint_are_persisted(self) -> None:
        boundary = self.summary["runtime_boundary"]
        self.assertFalse(boundary["raw_predictions_persisted"])
        self.assertFalse(boundary["checkpoint_written"])
        serialized = json.dumps(self.summary)
        self.assertNotIn(".claude/worktrees", serialized)
        self.assertNotIn("/mnt/data", serialized)


if __name__ == "__main__":
    unittest.main()
