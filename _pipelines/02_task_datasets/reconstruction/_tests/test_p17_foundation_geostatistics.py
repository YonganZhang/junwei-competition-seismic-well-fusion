from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import p17_foundation_geostatistics as p17  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class P17FoundationGeostatisticsContractTest(unittest.TestCase):
    def test_raw_preprocessing_inverse_is_exact(self) -> None:
        rng = np.random.default_rng(2693)
        raw = rng.normal(size=(12, 7))
        stats = [
            {"mean": float(column), "std": float(column + 1)}
            for column in range(7)
        ]
        preprocessed = np.column_stack(
            [
                (raw[:, column] - stats[column]["mean"])
                / stats[column]["std"]
                for column in range(7)
            ]
        )
        recovered = p17._raw_from_preprocessed(  # noqa: SLF001
            preprocessed,
            stats,
        )
        np.testing.assert_allclose(recovered, raw[:, 1:], rtol=0.0, atol=1e-15)

    def test_coordinate_match_is_exact_and_rejects_drift(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        indices = np.asarray([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
        tree = cKDTree(coordinates)
        matched, distance = p17._match_coordinates(  # noqa: SLF001
            tree,
            indices,
            coordinates[[2, 0]],
        )
        np.testing.assert_array_equal(matched, indices[[2, 0]])
        self.assertEqual(distance, 0.0)
        with self.assertRaises(RuntimeError):
            p17._match_coordinates(  # noqa: SLF001
                tree,
                indices,
                coordinates[[1]] + 1e-4,
            )

    def test_kernel_uses_train_only_metric_fit(self) -> None:
        rng = np.random.default_rng(2693)
        prediction, audit = p17._kernel_prediction(  # noqa: SLF001
            train_target=rng.normal(size=32),
            train_raw=rng.normal(size=(32, 6)),
            validation_raw=rng.normal(size=(11, 6)),
            train_foundation=rng.normal(size=(32, 48)),
            validation_foundation=rng.normal(size=(11, 48)),
            neighbours=8,
            foundation_weight=0.05,
            seismic_weight=0.10,
        )
        self.assertEqual(prediction.shape, (11,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(audit["fit_rows"], 32)
        self.assertTrue(audit["all_transforms_fit_on_outer_train_only"])
        self.assertFalse(audit["target_used_for_metric_fit"])
        self.assertEqual(audit["foundation_weight"], 0.05)
        self.assertEqual(audit["seismic_weight"], 0.10)

    def test_search_is_bounded_and_keeps_foundation_active(self) -> None:
        self.assertEqual(
            len(p17.METRIC_WEIGHT_PAIRS)
            * len(p17.NEIGHBOUR_COUNTS)
            * len(p17.BLEND_WEIGHTS),
            156,
        )
        self.assertTrue(
            all(foundation > 0.0 for foundation, _ in p17.METRIC_WEIGHT_PAIRS)
        )

    def test_cli_exposes_no_test_or_holdout_argument(self) -> None:
        parser = p17._parser()  # noqa: SLF001
        choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        help_text = parser.format_help() + "".join(
            subparser.format_help() for subparser in choices.values()
        )
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)


class P17FoundationGeostatisticsPortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p17_foundation_geostatistics"
        summary_path = cls.output_dir / "summary.json"
        verification_path = cls.output_dir / "verification.json"
        if not summary_path.is_file() or not verification_path.is_file():
            raise unittest.SkipTest("real P17 evidence not generated and verified")
        cls.summary_path = summary_path
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cls.verification = json.loads(
            verification_path.read_text(encoding="utf-8")
        )

    def test_real_evidence_preserves_budget_and_firewall(self) -> None:
        self.assertEqual(self.summary["schema_version"], p17.SCHEMA_VERSION)
        audit = self.summary["sample_audit"]
        self.assertEqual(audit["train_labels_per_fold"], 512)
        self.assertEqual(audit["validation_rows_per_fold"], 2048)
        self.assertFalse(audit["label_dataset_read"])
        firewall = self.summary["holdout_firewall"]
        self.assertFalse(firewall["test_h5_opened"])
        self.assertFalse(firewall["frozen_holdout_opened"])
        self.assertFalse(firewall["test_path_argument_exists"])

    def test_real_evidence_is_development_only_and_noncausal(self) -> None:
        experiment = self.summary["experiment"]
        self.assertEqual(experiment["decision"]["state"], "DEVELOPMENT_SIGNAL")
        self.assertFalse(experiment["decision"]["default_enabled"])
        self.assertFalse(
            experiment["decision"]["pretrained_contribution_claimed"]
        )
        self.assertTrue(experiment["decision"]["ablation_deferred"])
        self.assertLess(experiment["rmse_delta_vs_pykrige"], 0.0)
        self.assertEqual(
            experiment["search_space"]["candidate_count"],
            156,
        )

    def test_prediction_hash_and_independent_verification_match(self) -> None:
        record = self.summary["prediction_error_artifact"]
        artifact = PROJECT_ROOT / record["path"]
        self.assertEqual(_sha256(artifact), record["sha256"])
        self.assertEqual(self.verification["status"], "PASSED")
        self.assertEqual(
            self.verification["summary_sha256"],
            _sha256(self.summary_path),
        )
        self.assertEqual(
            self.verification["candidate_metrics_recomputed"]["rmse"],
            self.summary["experiment"]["selected_metrics"]["rmse"],
        )
        self.assertEqual(len(self.verification["fold_checks"]), 5)


if __name__ == "__main__":
    unittest.main()
