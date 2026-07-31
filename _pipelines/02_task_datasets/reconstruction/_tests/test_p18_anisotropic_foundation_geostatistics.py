from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import p18_anisotropic_foundation_geostatistics as p18  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class P18AnisotropicFoundationContractTest(unittest.TestCase):
    def test_search_is_bounded_and_anisotropic(self) -> None:
        candidate_count = (
            len(p18.VERTICAL_WEIGHTS)
            * len(p18.FOUNDATION_WEIGHTS)
            * len(p18.SEISMIC_WEIGHTS)
            * len(p18.NEIGHBOUR_COUNTS)
            * len(p18.DISTANCE_POWERS)
            * len(p18.BLEND_WEIGHTS)
        )
        self.assertEqual(candidate_count, 1215)
        self.assertTrue(all(value > 0.0 for value in p18.FOUNDATION_WEIGHTS))
        self.assertGreater(len(set(p18.VERTICAL_WEIGHTS)), 1)
        self.assertEqual(p18.NESTED_ENSEMBLE_SIZE, 3)

    def test_candidate_name_records_complete_metric(self) -> None:
        self.assertEqual(
            p18._candidate_name(  # noqa: SLF001
                4.0,
                0.10,
                0.20,
                64,
                1.5,
                0.75,
            ),
            "pca_z4_f0.1_s0.2_k64_p1.5_b0.75",
        )

    def test_nested_selection_excludes_held_out_fold_labels(self) -> None:
        fold_ids = np.repeat(np.asarray(p18.p17.base.FOLD_IDS), 3)
        target = np.zeros(len(fold_ids), dtype=np.float64)
        candidates = {
            "a": np.where(fold_ids == 0, 10.0, 0.1),
            "b": np.where(fold_ids == 0, -10.0, 0.2),
        }
        _, original, _ = p18._nested_lofo_select(  # noqa: SLF001
            candidates=candidates,
            target=target,
            fold_ids=fold_ids,
            top_n=1,
        )
        changed_target = target.copy()
        changed_target[fold_ids == 0] = 999.0
        _, changed, _ = p18._nested_lofo_select(  # noqa: SLF001
            candidates=candidates,
            target=changed_target,
            fold_ids=fold_ids,
            top_n=1,
        )
        original_fold_zero = next(row for row in original if row["held_out_fold"] == 0)
        changed_fold_zero = next(row for row in changed if row["held_out_fold"] == 0)
        self.assertEqual(
            original_fold_zero["selected_candidates"],
            changed_fold_zero["selected_candidates"],
        )
        self.assertNotIn(0, original_fold_zero["selection_folds"])
        self.assertFalse(original_fold_zero["held_out_fold_used_for_selection"])

    def test_cli_exposes_no_test_or_holdout_argument(self) -> None:
        parser = p18._parser()  # noqa: SLF001
        choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        help_text = parser.format_help() + "".join(
            subparser.format_help() for subparser in choices.values()
        )
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)


class P18AnisotropicFoundationPortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = (
            HERE / "_outputs" / "p18_anisotropic_foundation_geostatistics"
        )
        cls.summary_path = cls.output_dir / "summary.json"
        cls.verification_path = cls.output_dir / "verification.json"
        if not cls.summary_path.is_file() or not cls.verification_path.is_file():
            raise unittest.SkipTest("real P18 evidence not generated and verified")
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))
        cls.verification = json.loads(
            cls.verification_path.read_text(encoding="utf-8")
        )

    def test_real_evidence_preserves_budget_and_firewall(self) -> None:
        self.assertEqual(self.summary["schema_version"], p18.SCHEMA_VERSION)
        self.assertEqual(self.summary["sample_audit"]["train_labels_per_fold"], 512)
        self.assertEqual(
            self.summary["sample_audit"]["validation_rows_per_fold"], 2048
        )
        self.assertFalse(self.summary["sample_audit"]["label_dataset_read"])
        firewall = self.summary["holdout_firewall"]
        self.assertFalse(firewall["test_h5_opened"])
        self.assertFalse(firewall["frozen_holdout_opened"])
        self.assertFalse(firewall["test_path_argument_exists"])

    def test_real_evidence_corrects_p17_selection_bias(self) -> None:
        audit = self.summary["experiment"]["p17_selection_bias_audit"]
        self.assertLess(audit["same_oof_reported_rmse_delta_vs_pykrige"], 0.0)
        self.assertGreater(audit["honest_nested_top1_rmse_delta_vs_pykrige"], 0.0)
        self.assertGreater(audit["honest_nested_top3_rmse_delta_vs_pykrige"], 0.0)
        self.assertIn("superseded", audit["conclusion"])

    def test_real_evidence_has_robust_nested_improvement(self) -> None:
        experiment = self.summary["experiment"]
        self.assertEqual(experiment["decision"]["state"], "ROBUST_DEVELOPMENT_SIGNAL")
        self.assertFalse(experiment["decision"]["default_enabled"])
        self.assertFalse(experiment["decision"]["pretrained_contribution_claimed"])
        self.assertTrue(experiment["decision"]["ablation_deferred"])
        self.assertLess(experiment["relative_rmse_change_vs_pykrige"], -0.02)
        self.assertEqual(experiment["independent_fold_outcome_counts"]["win"], 5)
        self.assertEqual(experiment["search_space"]["candidate_count"], 1215)
        ci = experiment["whole_fold_bootstrap"][
            "rmse_delta_candidate_minus_baseline"
        ]["ci95"]
        self.assertLess(ci[1], 0.0)
        self.assertIn(
            "five coarse spatial folds",
            experiment["whole_fold_bootstrap"]["interpretation_caveat"],
        )
        self.assertGreaterEqual(len(experiment["limitations"]), 3)
        for row in experiment["nested_selection"]["selections"]:
            self.assertNotIn(row["held_out_fold"], row["selection_folds"])

    def test_prediction_hash_and_independent_verification_match(self) -> None:
        record = self.summary["prediction_error_artifact"]
        artifact = PROJECT_ROOT / record["path"]
        self.assertEqual(_sha256(artifact), record["sha256"])
        self.assertEqual(self.verification["status"], "PASSED")
        self.assertEqual(
            self.verification["summary_sha256"], _sha256(self.summary_path)
        )
        self.assertEqual(
            self.verification["candidate_metrics_recomputed"]["rmse"],
            self.summary["experiment"]["selected_metrics"]["rmse"],
        )
        self.assertEqual(len(self.verification["fold_checks"]), 5)


if __name__ == "__main__":
    unittest.main()
