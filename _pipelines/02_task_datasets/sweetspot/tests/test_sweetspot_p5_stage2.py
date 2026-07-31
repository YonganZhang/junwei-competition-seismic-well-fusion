"""Sweetspot-unique Stage-2 label, budget, result, and test-firewall tests."""
from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path

matrix_module = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.matrix")
stage2_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2"
)
data_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_data"
)
label_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_labels"
)

MODEL_ORDER = matrix_module.MODEL_ORDER
TARGET_ORDER = matrix_module.TARGET_ORDER
MAPPING_FILENAME = stage2_module.MAPPING_FILENAME
RESULT_FILENAME = stage2_module.RESULT_FILENAME
SUMMARY_FILENAME = stage2_module.SUMMARY_FILENAME
derive_seed = stage2_module.derive_seed
exclusive_gpu_lock = stage2_module.exclusive_gpu_lock
forbidden_test_source = data_module.forbidden_test_source
petrophysical_member_authorized = data_module.petrophysical_member_authorized
production_row_values = data_module.production_row_values
APPROVED_TARGETS = label_module.APPROVED_TARGETS
DEFAULT_MAPPING_PATH = label_module.DEFAULT_MAPPING_PATH
PROJECT_ROOT = label_module.PROJECT_ROOT
build_pilot_task_spec = label_module.build_pilot_task_spec
validate_label_mapping = label_module.validate_label_mapping


OUTPUT_DIR = DEFAULT_MAPPING_PATH.parent / "_outputs" / "stage2_pilot"


class SweetspotP5Stage2LabelMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = validate_label_mapping()

    def test_unique_track_prefixed_basenames(self) -> None:
        self.assertEqual(Path(__file__).name, "test_sweetspot_p5_stage2.py")
        self.assertEqual(
            Path(__import__(
                "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2",
                fromlist=["__file__"],
            ).__file__).name,
            "sweetspot_p5_stage2.py",
        )

    def test_only_existing_p4_labels_are_pilot_approved(self) -> None:
        targets = self.audit.payload["targets"]
        approved = tuple(target_id for target_id in TARGET_ORDER if targets[target_id]["status"] == "approved_for_development_pilot")
        self.assertEqual(approved, APPROVED_TARGETS)
        self.assertEqual(targets["T5"]["status"], "not_feasible")
        self.assertEqual(targets["T5"]["development_rebuild"], "forbidden")

    def test_proxy_semantics_remain_explicit(self) -> None:
        targets = self.audit.payload["targets"]
        self.assertTrue(targets["T1"]["is_proxy"])
        self.assertTrue(targets["T2"]["is_proxy"])
        self.assertTrue(targets["T4"]["is_proxy"])
        self.assertIn("not field sweetspot truth", targets["T1"]["proxy_semantics"])
        self.assertIn("not direct hydrocarbon-pay truth", targets["T2"]["proxy_semantics"])
        self.assertIn("not a domain-approved water-cut threshold", targets["T4"]["proxy_semantics"])

    def test_t6_and_t7_are_independent_estimators_and_heads(self) -> None:
        t6 = build_pilot_task_spec(self.audit, "T6")
        t7 = build_pilot_task_spec(self.audit, "T7")
        self.assertNotEqual(t6.task_id, t7.task_id)
        self.assertNotEqual(t6.targets, t7.targets)
        self.assertNotEqual(t6.label_version, t7.label_version)
        self.assertEqual(t6.targets, ("PHIF",))
        self.assertEqual(t7.targets, ("KLOGH",))

    def test_mapping_never_cites_test_or_historical_metric_artifacts(self) -> None:
        for target in self.audit.payload["targets"].values():
            for key in ("task_spec", "split_manifest", "label_evidence", "not_feasible_evidence"):
                reference = target.get(key)
                if reference is None:
                    continue
                lowered = reference["path"].lower()
                self.assertNotIn("frozen_test", lowered)
                self.assertFalse(lowered.endswith("status.json"))
                self.assertFalse(lowered.endswith("metrics.json"))
                self.assertFalse(Path(reference["path"]).is_absolute())


class SweetspotP5Stage2FirewallAndBudgetTests(unittest.TestCase):
    def test_materialized_test_sources_are_rejected(self) -> None:
        for path in (Path("frozen_test/predictions.csv"), Path("test.h5"), Path("x/test.hdf5")):
            self.assertTrue(forbidden_test_source(path))
        self.assertFalse(forbidden_test_source(Path("_sandbox/volve_data/Volve_Well_logs.zip")))

    def test_petrophysical_member_gate_runs_before_zip_read(self) -> None:
        development = {"15/9-F-1"}
        allowed = "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-1 A/input.LAS"
        forbidden = "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-15 D/input.LAS"
        self.assertTrue(petrophysical_member_authorized(allowed, development))
        self.assertFalse(petrophysical_member_authorized(forbidden, development))

    def test_production_row_authorizes_group_before_numeric_cells(self) -> None:
        class GuardCell:
            def __init__(self, value, *, forbidden=False):
                self._value = value
                self._forbidden = forbidden

            @property
            def value(self):
                if self._forbidden:
                    raise AssertionError("unauthorized numerical cell was accessed")
                return self._value

        row = [GuardCell("NO 15/9-F-15 D"), GuardCell(999.0, forbidden=True)]
        result = production_row_values(
            row,
            {"WELL_BORE_CODE": 0, "DATEPRD": 1, **{name: 1 for name in (
                "ON_STREAM_HRS", "AVG_DOWNHOLE_PRESSURE", "AVG_CHOKE_SIZE_P", "AVG_WHP_P",
                "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
            )}},
            {"NO 15/9-F-1 C"},
        )
        self.assertIsNone(result)

    def test_seed_is_stable_and_cell_specific(self) -> None:
        seeds = {derive_seed(model, target) for model in MODEL_ORDER for target in TARGET_ORDER}
        self.assertEqual(len(seeds), len(MODEL_ORDER) * len(TARGET_ORDER))
        self.assertEqual(derive_seed("xgboost", "T1"), derive_seed("xgboost", "T1"))

    def test_gpu_lock_records_no_machine_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with exclusive_gpu_lock(Path(directory) / "sweetspot.lock") as evidence:
                self.assertTrue(evidence["acquired"])
                self.assertFalse(evidence["path_recorded"])


class SweetspotP5Stage2ArchivedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_path = OUTPUT_DIR / RESULT_FILENAME
        cls.summary_path = OUTPUT_DIR / SUMMARY_FILENAME
        cls.mapping_path = OUTPUT_DIR / MAPPING_FILENAME
        if not all(path.is_file() for path in (cls.result_path, cls.summary_path, cls.mapping_path)):
            raise unittest.SkipTest("run the sweetspot Stage-2 runner to archive portable results")
        cls.results = [json.loads(line) for line in cls.result_path.read_text(encoding="utf-8").splitlines()]
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))
        cls.mapping = json.loads(cls.mapping_path.read_text(encoding="utf-8"))

    def test_full_ten_by_seven_cell_coverage(self) -> None:
        self.assertEqual(len(self.results), 70)
        observed = {(item["model_id"], item["task_id"]) for item in self.results}
        expected = {(model, target) for model in MODEL_ORDER for target in TARGET_ORDER}
        self.assertEqual(observed, expected)
        self.assertEqual(self.summary["expected_cells"], 70)
        self.assertEqual(self.summary["attempted_cells"], 70)

    def test_t5_is_always_not_feasible(self) -> None:
        rows = [item for item in self.results if item["task_id"] == "T5"]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(item["status"] == "SKIP" for item in rows))
        self.assertTrue(all(item["reason"]["code"] == "label_not_feasible" for item in rows))

    def test_same_target_cells_share_input_budget(self) -> None:
        for target_id in ("T1", "T2", "T3", "T4"):
            hashes = {
                item["input_budget"]["input_budget_sha256"]
                for item in self.results if item["task_id"] == target_id
            }
            self.assertEqual(len(hashes), 1)
            self.assertNotIn(None, hashes)

    def test_each_target_has_an_independent_leaderboard(self) -> None:
        self.assertEqual(set(self.summary["leaderboards"]), set(TARGET_ORDER))
        for target_id in ("T1", "T2", "T3", "T4"):
            self.assertEqual(self.summary["leaderboards"][target_id]["status"], "rankable")
        for target_id in ("T5", "T6", "T7"):
            self.assertEqual(self.summary["leaderboards"][target_id]["status"], "not_rankable")

    def test_no_test_or_label_generation_claim(self) -> None:
        self.assertFalse(self.summary["test_accessed"])
        self.assertFalse(self.summary["historical_test_metrics_used"])
        self.assertFalse(self.summary["labels_generated"])
        self.assertFalse(self.mapping["test_accessed"])
        self.assertFalse(self.mapping["labels_generated"])
        for item in self.results:
            self.assertFalse(item["test_firewall"]["test_accessed"])
            self.assertFalse(item["test_firewall"]["historical_test_metrics_used"])
            self.assertFalse(item["label_generated"])

    def test_result_hash_and_portable_paths(self) -> None:
        digest = hashlib.sha256(self.result_path.read_bytes()).hexdigest()
        self.assertEqual(self.summary["results_sha256"], digest)
        serialized = json.dumps(
            {"results": self.results, "summary": self.summary, "mapping": self.mapping},
            ensure_ascii=False,
        )
        self.assertNotIn("/" + "mnt" + "/", serialized)
        self.assertNotIn(".claude" + "/worktrees", serialized)
        self.assertEqual(
            self.summary["portable_output_files"],
            [RESULT_FILENAME, SUMMARY_FILENAME, MAPPING_FILENAME],
        )
        self.assertEqual(
            {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()},
            {RESULT_FILENAME, SUMMARY_FILENAME, MAPPING_FILENAME},
        )


if __name__ == "__main__":
    unittest.main()
