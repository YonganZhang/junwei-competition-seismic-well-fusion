from __future__ import annotations

import importlib
import json
import unittest

import numpy as np
import pandas as pd


calendar = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p8.calendar_data"
)
adapter = importlib.import_module("_models.sweetspot.p7_chronos2")
runner = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p8.runner"
)


class FakeCalendarPipeline:
    def __init__(self) -> None:
        self.frame = None
        self.kwargs = None

    def predict_df(self, frame, **kwargs):
        self.frame = frame.copy()
        self.kwargs = kwargs
        rows = []
        for item_id, group in frame.groupby("item_id", sort=False):
            start = group["timestamp"].max() + pd.Timedelta(days=1)
            for index, timestamp in enumerate(pd.date_range(start, periods=30, freq="D")):
                rows.append(
                    {
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "predictions": float(index - 5),
                        "0.5": float(index - 5),
                    }
                )
        return pd.DataFrame(rows)


def production_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-04"]
    )
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "WELL_BORE_CODE": "W1",
                "DATEPRD": date,
                "BORE_OIL_VOL": float(index + 1),
                "BORE_GAS_VOL": float(index + 2),
                "BORE_WAT_VOL": float(index + 3),
                "ON_STREAM_HRS": float(index + 4),
                "AVG_DOWNHOLE_PRESSURE": float(index + 5),
                "AVG_CHOKE_SIZE_P": float(index + 6),
                "AVG_WHP_P": float(index + 7),
            }
        )
    return pd.DataFrame(rows)


class CalendarDataTests(unittest.TestCase):
    def test_duplicate_days_aggregate_and_missing_days_remain_missing(self) -> None:
        daily = calendar.calendarize_well(production_frame())
        self.assertEqual(
            list(daily["DATEPRD"]),
            list(pd.date_range("2026-01-01", "2026-01-04", freq="D")),
        )
        first = daily.iloc[0]
        self.assertEqual(first["BORE_OIL_VOL"], 3.0)
        self.assertEqual(first["AVG_WHP_P"], 7.5)
        missing = daily.iloc[1]
        self.assertTrue(np.isnan(missing["BORE_OIL_VOL"]))
        self.assertTrue(np.isnan(missing["AVG_WHP_P"]))

    def test_calendar_frame_requires_exact_gap_free_daily_axis(self) -> None:
        sequence = np.ones((1, 7, 30), dtype=np.float32)
        timestamps = np.asarray(
            [pd.date_range("2026-01-01", periods=30, freq="D")],
            dtype="datetime64[ns]",
        )
        frame = adapter.build_calendar_frame(sequence, timestamps, ("sample",))
        self.assertEqual(len(frame), 30)
        self.assertEqual(frame["timestamp"].diff().dropna().unique()[0], pd.Timedelta(days=1))
        timestamps[0, 10] = timestamps[0, 9]
        with self.assertRaisesRegex(ValueError, "gap-free daily"):
            adapter.build_calendar_frame(sequence, timestamps, ("sample",))

    def test_predict_df_receives_explicit_daily_frequency_and_no_future_frame(self) -> None:
        pipeline = FakeCalendarPipeline()
        sequence = np.ones((2, 7, 30), dtype=np.float32)
        timestamps = np.asarray(
            [
                pd.date_range("2026-01-01", periods=30, freq="D"),
                pd.date_range("2026-02-01", periods=30, freq="D"),
            ],
            dtype="datetime64[ns]",
        )
        daily, point = adapter.forecast_oil_calendar(
            pipeline, sequence, timestamps, ("a", "b"), batch_size=8
        )
        self.assertEqual(daily.shape, (2, 30))
        self.assertTrue(np.all(daily >= 0))
        np.testing.assert_allclose(point, daily.mean(axis=1))
        self.assertTrue(pipeline.kwargs["validate_inputs"])
        self.assertFalse(pipeline.kwargs["cross_learning"])
        self.assertNotIn("freq", pipeline.kwargs)
        self.assertNotIn("future_df", pipeline.kwargs)

    def test_calendar_runner_has_no_holdout_or_test_arguments(self) -> None:
        options = {action.dest for action in runner._parser()._actions}
        self.assertNotIn("test", options)
        self.assertNotIn("holdout", options)
        self.assertNotIn("frozen_test", options)

    def test_calendar_runner_source_lock_separates_code_and_weights(self) -> None:
        lock = runner._validate_source_lock()
        self.assertNotEqual(lock["source_revision"], lock["revision"])
        self.assertEqual(lock["revision"], adapter.MODEL_REVISION)


class CalendarArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = runner.DEFAULT_OUTPUT_DIR / "summary.json"
        if not cls.path.is_file():
            raise unittest.SkipTest("run the P8 calendar Chronos diagnostic")
        cls.summary = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_real_source_locked_foundation_and_exact_daily_contract(self) -> None:
        foundation = self.summary["foundation"]
        self.assertTrue(foundation["real_pretrained_weights_loaded"])
        self.assertEqual(foundation["model_id"], adapter.MODEL_ID)
        self.assertEqual(
            foundation["source_lock_sha256"],
            runner._sha256_file(runner.SOURCE_LOCK_PATH),
        )
        contract = self.summary["calendar_contract"]
        self.assertEqual(contract["frequency"], "D")
        self.assertEqual(contract["history_days"], 30)
        self.assertEqual(contract["forecast_days"], 30)
        self.assertFalse(contract["legacy_p7_observation_index_evidence_promoted"])

    def test_four_group_isolated_development_folds_without_test_access(self) -> None:
        evaluation = self.summary["evaluation"]
        self.assertEqual(evaluation["folds"], [0, 1, 2, 3])
        self.assertFalse(evaluation["known_holdout_accessed"])
        self.assertFalse(evaluation["frozen_test_accessed"])
        for method in self.summary["methods"].values():
            self.assertEqual(method["fold_count"], 4)
            for fold in method["folds"]:
                self.assertFalse(set(fold["train_groups"]) & set(fold["validation_groups"]))

    def test_calendar_foundation_gain_is_diagnostic_not_promotion(self) -> None:
        methods = self.summary["methods"]
        history = methods["B1_calendar_history_mean"]["macro_fold_mean"]["mae"]
        foundation = methods["F0_chronos2_calendar"]["macro_fold_mean"]["mae"]
        blend = methods["F1_chronos2_train_blend_calendar"]["macro_fold_mean"]["mae"]
        self.assertLess(foundation, history)
        self.assertLess(blend, history)
        self.assertEqual(self.summary["decision"]["state"], "CONNECTED_UNVERIFIED")
        self.assertFalse(self.summary["decision"]["default_enabled"])

    def test_no_predictions_or_checkpoint_persisted(self) -> None:
        runtime = self.summary["runtime"]
        self.assertFalse(runtime["raw_predictions_persisted"])
        self.assertFalse(runtime["checkpoint_written"])
        serialized = json.dumps(self.summary)
        self.assertNotIn("/mnt/data", serialized)
        self.assertNotIn(".claude/worktrees", serialized)


if __name__ == "__main__":
    unittest.main()
