"""Fail-closed tests for sweetspot P5 Stage-4 known-holdout confirmation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


stage4 = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage4"
)
labels = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_labels"
)

OUTPUT_DIR = stage4.DEFAULT_OUTPUT_DIR
SUMMARY_PATH = OUTPUT_DIR / "p5_stage4_summary.json"
RESULT_PATH = OUTPUT_DIR / "p5_stage4_results.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "p5_stage4_manifest.json"


class SweetspotP5Stage4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = labels.validate_label_mapping()
        cls.contract = stage4.validate_stage4_contract(cls.audit)

    def test_track_prefixed_module_and_test_basenames(self) -> None:
        self.assertEqual(Path(__file__).name, "test_sweetspot_p5_stage4.py")
        self.assertEqual(Path(stage4.__file__).name, "sweetspot_p5_stage4.py")

    def test_exact_frozen_winners_seed_and_update_budget(self) -> None:
        self.assertEqual(stage4.ROOT_SEED, 2693)
        expected = {
            "T1": "lightgbm",
            "T2": "catboost",
            "T3": "xgboost",
            "T4": "catboost",
        }
        self.assertEqual(
            {task_id: row["model_id"] for task_id, row in stage4.FROZEN_WINNERS.items()},
            expected,
        )
        self.assertEqual({row["updates"] for row in stage4.FROZEN_WINNERS.values()}, {64})

    def test_stage3_and_p4_hash_contract_is_frozen(self) -> None:
        self.assertEqual(self.contract["stage3_commit"], stage4.STAGE3_COMMIT)
        self.assertEqual(self.contract["stage3_summary_sha256"], stage4.STAGE3_SUMMARY_SHA256)
        self.assertEqual(set(self.contract["winners"]), {"T1", "T2", "T3", "T4"})
        for task_id in stage4.FROZEN_WINNERS:
            split = self.contract["splits"][task_id]
            self.assertGreater(split["development_samples"], 0)
            self.assertEqual(split["known_holdout_samples"], stage4.P4_EXPOSURE[task_id]["test_rows"])
            self.assertFalse(set(split["development_groups"]) & set(split["known_holdout_groups"]))
            self.assertTrue(self.contract["prior_exposure"][task_id]["prior_test_consumed"])

    def test_tampered_stage3_summary_fails_before_other_artifacts(self) -> None:
        parent = stage4.HERE / "_outputs"
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            fake = Path(directory)
            (fake / "p5_stage3_summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary hash changed"):
                stage4.validate_stage4_contract(self.audit, stage3_dir=fake)

    def test_contract_reads_metadata_but_no_historical_metrics_or_predictions(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            candidate = Path(path)
            opened.append(candidate)
            lowered = [part.lower() for part in candidate.parts]
            if "frozen_test" in lowered or candidate.name in {"metrics.json", "predictions.csv"}:
                raise AssertionError(f"historical result opened: {candidate}")
            return original(candidate, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            contract = stage4.validate_stage4_contract(self.audit)
        self.assertTrue(opened)
        self.assertFalse(contract["historical_test_metrics_read"])

    def test_explicit_confirmation_flag_is_required_and_writes_nothing(self) -> None:
        parent = stage4.HERE / "_outputs"
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            output = Path(directory) / "must_not_exist"
            exit_code = stage4.main(["--output-dir", str(output)])
            self.assertEqual(exit_code, 2)
            self.assertFalse(output.exists())

    def test_output_must_stay_in_track_private_directory(self) -> None:
        with self.assertRaises(PermissionError):
            stage4._portable_output_dir(stage4.HERE.parent / "not-sweetspot-stage4")

    def test_no_hpo_or_p4_baseline_execution_surface(self) -> None:
        parser_options = {action.dest for action in stage4._parser()._actions}
        self.assertNotIn("hpo", parser_options)
        self.assertNotIn("trial", parser_options)
        source = inspect.getsource(stage4)
        self.assertNotIn("targets.baseline", source)
        self.assertNotIn("run_target(", source)

    def test_refit_phase_precedes_known_holdout_rebuild_in_source(self) -> None:
        source = inspect.getsource(stage4._run_confirmed_target)
        development = source.index('"development"')
        refit = source.index("_fit_frozen_winner")
        holdout = source.index('"known_holdout"')
        self.assertLess(development, refit)
        self.assertLess(refit, holdout)

    def test_task_specs_are_independent_and_mark_known_holdout_evidence(self) -> None:
        specs = {task_id: stage4._stage4_task_spec(self.audit, task_id) for task_id in stage4.FROZEN_WINNERS}
        self.assertEqual(len({spec.task_id for spec in specs.values()}), 4)
        self.assertEqual(len({spec.targets[0] for spec in specs.values()}), 4)
        for spec in specs.values():
            self.assertEqual(spec.metadata["evidence_class"], stage4.EVIDENCE_CLASS)
            self.assertTrue(spec.metadata["prior_test_consumed"])
            self.assertFalse(spec.metadata["fresh_blind"])
            self.assertEqual(spec.hpo["optimization"], "forbidden")

    def test_t5_t6_t7_stop_without_winner_or_test_access(self) -> None:
        expected = {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"}
        for task_id, status in expected.items():
            row = stage4._blocked_status(self.audit, self.contract, task_id)
            self.assertEqual(row["status"], status)
            self.assertIsNone(row["stage3_winner"])
            self.assertFalse(row["development_feature_source_available"])
            self.assertFalse(row["test_accessed"])
            self.assertFalse(row["labels_generated"])
            self.assertFalse(row["checkpoint_created"])
            self.assertFalse(row["predictions_created"])
        self.assertNotEqual(self.audit.target("T6")["label_version"], self.audit.target("T7")["label_version"])
        self.assertEqual(self.audit.target("T6")["target_name"], "PHIF")
        self.assertEqual(self.audit.target("T7")["target_name"], "KLOGH")

    def test_partition_loader_rejects_status_only_targets_before_data_access(self) -> None:
        for task_id in ("T5", "T6", "T7"):
            with self.assertRaises(PermissionError):
                stage4._rebuild_partition(self.audit, task_id, "development", source_root=None)


class SweetspotP5Stage4ArchivedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not all(path.is_file() for path in (SUMMARY_PATH, RESULT_PATH, MANIFEST_PATH)):
            raise unittest.SkipTest("run sweetspot Stage-4 to archive known-holdout confirmation")
        cls.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.results = [json.loads(line) for line in RESULT_PATH.read_text(encoding="utf-8").splitlines()]
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.by_task = {row["task_id"]: row for row in cls.results}

    def test_all_seven_targets_have_one_independent_record(self) -> None:
        self.assertEqual(len(self.results), 7)
        self.assertEqual(set(self.by_task), {f"T{index}" for index in range(1, 8)})
        self.assertEqual(
            {task_id: self.by_task[task_id]["status"] for task_id in ("T1", "T2", "T3", "T4")},
            {task_id: "confirmed_known_holdout" for task_id in ("T1", "T2", "T3", "T4")},
        )
        self.assertEqual(self.by_task["T5"]["status"], "not_feasible")
        self.assertEqual(self.by_task["T6"]["status"], "blocked")
        self.assertEqual(self.by_task["T7"]["status"], "blocked")

    def test_known_holdout_is_explicitly_not_fresh_blind(self) -> None:
        self.assertEqual(self.summary["evidence_class"], stage4.EVIDENCE_CLASS)
        self.assertTrue(self.summary["prior_test_consumed"])
        self.assertFalse(self.summary["fresh_blind"])
        self.assertFalse(self.summary["selection_feedback_allowed"])
        self.assertFalse(self.summary["historical_p4_metrics_read"])
        self.assertFalse(self.summary["p4_hpo_called"])
        for task_id in ("T1", "T2", "T3", "T4"):
            row = self.by_task[task_id]
            self.assertTrue(row["prior_test_consumed"])
            self.assertFalse(row["fresh_blind"])
            self.assertTrue(row["test_accessed"])

    def test_counts_match_frozen_p4_splits_and_refit_uses_all_development(self) -> None:
        expected = {
            "T1": (35810, 11936),
            "T2": (36122, 12081),
            "T3": (1111, 132),
            "T4": (32, 37),
        }
        for task_id, (development, holdout) in expected.items():
            self.assertEqual(self.by_task[task_id]["development_samples"], development)
            self.assertEqual(self.by_task[task_id]["known_holdout_samples"], holdout)
            refit = json.loads((OUTPUT_DIR / "targets" / task_id / "refit.json").read_text(encoding="utf-8"))
            self.assertEqual(refit["development_samples"], development)
            self.assertFalse(refit["known_holdout_accessed_during_refit"])

    def test_target_specific_metric_contracts_are_preserved(self) -> None:
        t1 = self.by_task["T1"]["metrics"]
        self.assertTrue({"mae", "rmse", "r2", "spearman", "sample_count"} <= set(t1))
        t2 = self.by_task["T2"]["metrics"]
        self.assertTrue({"average_precision", "brier", "f1_at_0_5", "thickness_diagnostic"} <= set(t2))
        t3 = self.by_task["T3"]["metrics"]
        self.assertTrue({"mae", "rmse", "spearman", "topk_diagnostic"} <= set(t3))
        t4 = self.by_task["T4"]["metrics"]
        self.assertTrue({"average_precision", "brier", "f1_at_0_5"} <= set(t4))
        self.assertEqual(t2["threshold"], 0.5)
        self.assertEqual(t4["threshold"], 0.5)

    def test_compact_predictions_checkpoints_configs_refit_and_figures_exist(self) -> None:
        for task_id in ("T1", "T2", "T3", "T4"):
            target = OUTPUT_DIR / "targets" / task_id
            for relative in (
                "config.json", "refit.json", "metrics.json", "predictions.csv.gz",
                "refit/model.pkl.gz", "figure.png",
            ):
                path = target / relative
                self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
            with gzip.open(target / "predictions.csv.gz", "rt", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), self.by_task[task_id]["known_holdout_samples"])
        for task_id in ("T5", "T6", "T7"):
            target = OUTPUT_DIR / "targets" / task_id
            self.assertTrue((target / "status.json").is_file())
            self.assertTrue((target / "figure.png").is_file())
            self.assertFalse((target / "predictions.csv.gz").exists())
            self.assertFalse((target / "refit").exists())

    def test_artifact_manifest_hashes_every_portable_file(self) -> None:
        self.assertTrue(self.manifest["all_paths_portable"])
        self.assertEqual(self.manifest["artifact_count"], len(self.manifest["artifacts"]))
        for row in self.manifest["artifacts"]:
            path = OUTPUT_DIR / row["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, row["size_bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_t1_t2_holdout_provenance_uses_authorized_known_holdout_wording(self) -> None:
        expected = "Volve_Well_logs.zip authorized known-holdout members only"
        for task_id in ("T1", "T2"):
            metrics = json.loads(
                (OUTPUT_DIR / "targets" / task_id / "metrics.json").read_text(encoding="utf-8")
            )
            provenance = metrics["holdout_provenance"]
            self.assertEqual(provenance["source_kind"], expected)
            self.assertEqual(provenance["partition"], "known_holdout")
            self.assertEqual(provenance["authorized_groups"], ["15/9-F-15"])
            self.assertTrue(provenance["test_accessed"])

    def test_execution_scope_is_full_or_audited_targeted_refresh(self) -> None:
        scope = self.summary["execution_scope"]
        self.assertIn(scope, {"full_T1_to_T7", "targeted_T1_T2_provenance_refresh"})
        refresh = self.summary["refresh"]
        if scope == "full_T1_to_T7":
            self.assertIsNone(refresh)
            return
        self.assertEqual(refresh["rerun_target_ids"], ["T1", "T2"])
        self.assertEqual(refresh["preserved_target_ids"], ["T3", "T4", "T5", "T6", "T7"])
        self.assertFalse(refresh["model_config_changed"])
        self.assertFalse(refresh["metric_algorithm_changed"])
        self.assertNotEqual(
            refresh["previous_manifest_sha256"],
            hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        )

    def test_serialized_artifacts_contain_no_machine_or_worktree_paths(self) -> None:
        serialized = json.dumps(
            {"summary": self.summary, "results": self.results, "manifest": self.manifest},
            ensure_ascii=False,
        )
        self.assertNotIn("/mnt/", serialized)
        self.assertNotIn(".claude/worktrees", serialized)

    def test_existing_output_is_never_overwritten(self) -> None:
        before = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            stage4.run_stage4()
        self.assertEqual(before, hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
