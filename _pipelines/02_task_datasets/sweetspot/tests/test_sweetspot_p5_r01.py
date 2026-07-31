"""Sweetspot-prefixed P5.1 R0/R1 contract, censoring and firewall tests."""
from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


contracts_module = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.r01.contracts")
data_module = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.r01.data")
runner_module = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.r01.runner")

CONTRACT_ORDER = contracts_module.CONTRACT_ORDER
load_contracts = contracts_module.load_contracts
task_spec = contracts_module.task_spec
R01Dataset = data_module.R01Dataset
label_t3_window = data_module.label_t3_window
label_t4_window = data_module.label_t4_window
legal_group_folds = runner_module.legal_group_folds
write_outputs = runner_module.write_outputs


class SweetspotP51R0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = load_contracts()

    def test_track_prefixed_test_and_runner_basenames(self) -> None:
        self.assertEqual(Path(__file__).name, "test_sweetspot_p5_r01.py")
        self.assertEqual(Path(runner_module.__file__).name, "runner.py")
        self.assertIn("sweetspot.p5.1.r01", self.contracts["T1"]["task_id"])

    def test_seven_targets_have_independent_contracts_tasks_and_heads(self) -> None:
        self.assertEqual(tuple(self.contracts), CONTRACT_ORDER)
        self.assertEqual(len({item["contract_sha256"] for item in self.contracts.values()}), 7)
        self.assertEqual(len({item["task_id"] for item in self.contracts.values()}), 7)
        self.assertEqual(len({item["head_name"] for item in self.contracts.values()}), 7)

    def test_proxy_semantics_do_not_overclaim_field_truth(self) -> None:
        for target_id in ("T1", "T2", "T4"):
            self.assertEqual(self.contracts[target_id]["truth_class"], "proxy")
            self.assertFalse(self.contracts[target_id]["field_truth"])
            self.assertTrue(self.contracts[target_id]["warnings"])
        self.assertIsNone(self.contracts["T1"]["label"]["threshold"])
        self.assertNotIn("hydrocarbon", self.contracts["T2"]["semantic_name"])

    def test_t5_is_fail_closed_and_t4_survival_is_unapproved(self) -> None:
        self.assertEqual(self.contracts["T5"]["status"], "not_feasible")
        self.assertFalse(self.contracts["T5"]["approval"]["approved"])
        self.assertEqual(self.contracts["T4"]["blocked_lanes"]["formal_failure_survival"], "unapproved")

    def test_t6_t7_are_independent_and_forbid_test_backfill(self) -> None:
        t6, t7 = self.contracts["T6"], self.contracts["T7"]
        self.assertNotEqual(t6["head_name"], t7["head_name"])
        self.assertEqual(t6["label"]["formula"], "PHIF")
        self.assertEqual(t7["label"]["formula"], "KLOGH")
        self.assertEqual(t7["label"]["model_transform"], "log1p")
        self.assertEqual(t6["split"]["test_access"], "forbidden")
        self.assertEqual(t7["split"]["test_access"], "forbidden")

    def test_task_specs_are_single_head_and_have_no_hpo(self) -> None:
        for target_id in ("T1", "T2", "T3", "T4", "T6", "T7"):
            spec = task_spec(self.contracts[target_id])
            self.assertEqual(len(spec.targets), 1)
            self.assertFalse(spec.hpo["enabled"])
            self.assertTrue(spec.metadata["r1_no_final_ranking"])
            self.assertFalse(set(spec.input_whitelist) & set(spec.forbidden_inputs))


class SweetspotP51CalendarLabelTests(unittest.TestCase):
    def test_t3_exact_calendar_window_retains_zero_and_never_fills_missing(self) -> None:
        future = pd.DataFrame({"BORE_OIL_VOL": [0.0] * 6 + [12.0] * 24})
        value, state = label_t3_window(future, boundary_complete=True)
        self.assertEqual(state, "observed")
        self.assertEqual(value, 9.6)
        future.loc[:6, "BORE_OIL_VOL"] = np.nan
        self.assertEqual(label_t3_window(future, boundary_complete=True), (None, "fewer_than_24_observed_days"))
        self.assertEqual(label_t3_window(future, boundary_complete=False)[1], "right_boundary_incomplete")

    def test_t4_requires_seven_calendar_days_zero_breaks_and_missing_censors(self) -> None:
        values = np.zeros(30)
        values[5:12] = 1.0
        self.assertEqual(label_t4_window(pd.DataFrame({"BORE_WAT_VOL": values}), boundary_complete=True), (1.0, "event"))
        values[8] = 0.0
        self.assertEqual(label_t4_window(pd.DataFrame({"BORE_WAT_VOL": values}), boundary_complete=True), (0.0, "no_event"))
        values[8] = np.nan
        self.assertEqual(label_t4_window(pd.DataFrame({"BORE_WAT_VOL": values}), boundary_complete=True)[1], "missing_makes_nonevent_indeterminate")


class SweetspotP51SplitAndOutputTests(unittest.TestCase):
    def _dataset(self) -> R01Dataset:
        groups = tuple(group for group in ("A", "B", "C") for _ in range(6))
        values = np.arange(len(groups), dtype=float)
        return R01Dataset(
            "T1", "regression", "t1_rqi_proxy", ("x1", "x2"),
            tuple(f"sample-{index}" for index in range(len(groups))), groups,
            tuple(None for _ in groups), np.column_stack([values, values % 3]), values / 10,
            ("A", "B", "C"), {"source": {}, "split": {}}, {"bounded_samples": len(groups)},
        )

    def test_legal_folds_are_group_isolated(self) -> None:
        dataset = self._dataset()
        groups = np.asarray(dataset.groups, dtype=object)
        folds = legal_group_folds(dataset)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(fold["status"], "ready")
            train = fold["train_indices"]
            validation = fold["validation_indices"]
            self.assertFalse(set(groups[train]) & set(groups[validation]))

    def test_fold_preprocessing_evidence_names_only_train_ids(self) -> None:
        dataset = self._dataset()
        contract = load_contracts()["T1"]
        fold = legal_group_folds(dataset)[0]
        _, evidence = runner_module._fit_predict(
            contract, dataset, fold["train_indices"], fold["validation_indices"],
        )
        self.assertEqual(evidence["fit_scope"], "fold_train_only")
        self.assertEqual(evidence["fit_sample_count"], len(fold["train_indices"]))
        self.assertEqual(evidence["target_transform_fit_scope"], "fixed_formula_no_fit")

    def test_cli_has_no_test_or_holdout_argument(self) -> None:
        option_strings = {
            option for action in runner_module.build_parser()._actions for option in action.option_strings
        }
        self.assertNotIn("--test", option_strings)
        self.assertNotIn("--test-h5", option_strings)
        self.assertNotIn("--known-holdout", option_strings)

    def test_portable_outputs_have_seven_boards_and_no_aggregate_score(self) -> None:
        contracts = load_contracts()
        registry = {
            "targets": {target: {"contract_sha256": item["contract_sha256"]} for target, item in contracts.items()}
        }
        audit = {
            "source_manifest_sha256": "a" * 64, "dataset_sample_sha256": {},
            "dataset_split_sha256": {},
        }
        results = [{"target_id": target, "head_name": contracts[target]["head_name"]} for target in CONTRACT_ORDER]
        summary = {
            "target_boards": {target: {} for target in CONTRACT_ORDER},
            "aggregate_sweetspot_score": None,
            "test_firewall": {"physical_test_h5_accessed": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_outputs(Path(directory), (registry, audit, results, summary))
            self.assertEqual(set(manifest["files"]), {
                "r0_contract_registry.json", "r0_data_audit.json", "r1_results.jsonl", "r1_summary.json",
            })
            rows = [json.loads(line) for line in (Path(directory) / "r1_results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 7)
            written = json.loads((Path(directory) / "r1_summary.json").read_text())
            self.assertIsNone(written["aggregate_sweetspot_score"])


class SweetspotP51ArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = Path(runner_module.DEFAULT_OUTPUT_DIR)
        if not all((cls.output_dir / name).is_file() for name in runner_module.OUTPUT_FILES):
            raise unittest.SkipTest("run the bounded R0/R1 development command first")
        cls.rows = [
            json.loads(line)
            for line in (cls.output_dir / "r1_results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        cls.summary = json.loads((cls.output_dir / "r1_summary.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((cls.output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    def test_archive_has_exactly_seven_independent_not_rankable_targets(self) -> None:
        self.assertEqual([row["target_id"] for row in self.rows], list(CONTRACT_ORDER))
        self.assertTrue(all(row["rankability"] == "not_rankable" for row in self.rows))
        self.assertEqual(set(self.summary["target_boards"]), set(CONTRACT_ORDER))
        self.assertIsNone(self.summary["aggregate_sweetspot_board"])
        self.assertIsNone(self.summary["aggregate_sweetspot_score"])
        self.assertEqual(self.summary["ten_model_fair_comparison"], "deferred_to_R2")

    def test_archive_hashes_and_test_firewall_are_self_consistent(self) -> None:
        for name, evidence in self.manifest["files"].items():
            path = self.output_dir / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), evidence["sha256"])
            self.assertEqual(path.stat().st_size, evidence["size_bytes"])
        self.assertEqual(
            set(self.manifest["input_hashes"]),
            {"contracts_sha256", "source_manifest_sha256", "sample_sha256", "split_manifest_sha256", "config_sha256"},
        )
        for row in self.rows:
            self.assertFalse(any(row["test_firewall"].values()))

    def test_archive_is_portable_and_contains_no_model_or_prediction_payload(self) -> None:
        serialized = "\n".join(
            (self.output_dir / name).read_text(encoding="utf-8") for name in runner_module.OUTPUT_FILES
        )
        self.assertNotIn("/mnt/", serialized)
        self.assertNotIn(".claude/worktrees", serialized)
        self.assertEqual({path.name for path in self.output_dir.iterdir()}, set(runner_module.OUTPUT_FILES))


if __name__ == "__main__":
    unittest.main()
