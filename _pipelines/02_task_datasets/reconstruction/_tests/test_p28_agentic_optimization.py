from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


TRACK = Path(__file__).resolve().parents[1]
ROOT = TRACK.parents[2]
sys.path[:0] = [str(TRACK), str(ROOT)]

import p17_foundation_geostatistics as p17  # noqa: E402
import p19_meta_purged_geostatistics as p19  # noqa: E402
import p28_agentic_optimization as p28  # noqa: E402


class P28AgenticOptimizationUnitTest(unittest.TestCase):
    def test_registry_is_frozen_and_single_factor(self) -> None:
        p28._validate_registry()  # noqa: SLF001
        self.assertEqual(len(p28.ACTION_REGISTRY), 9)
        self.assertEqual(len(set(p28.ACTION_BY_ID)), 9)
        for action in p28.ACTION_REGISTRY:
            changed = [
                key
                for key, value in action["parameters"].items()
                if value != p28.A0_PARAMETERS[key]
            ]
            self.assertEqual(changed, [action["changed_factor"]])

    def test_kernel_families_are_finite_and_positive(self) -> None:
        distance = np.asarray([[0.0, 0.5, 2.0]], dtype=np.float64)
        for family in (
            "inverse_distance",
            "matern32",
            "gaussian",
            "rational_quadratic",
        ):
            weights = p28._kernel_weights(  # noqa: SLF001
                distance,
                family=family,
                power=1.5,
                bandwidth=0.75,
            )
            self.assertEqual(weights.shape, distance.shape)
            self.assertTrue(np.all(np.isfinite(weights)))
            self.assertTrue(np.all(weights > 0.0))

    def test_route_gate_schema_is_strict(self) -> None:
        accepted = p28._validate_route_gate(  # noqa: SLF001
            {
                "schema_version": "p28-route-gate/v1",
                "route_id": "rgt_ked",
                "decision": "reject",
                "reason_code": "negative_transfer",
                "stop": True,
            }
        )
        self.assertEqual(accepted["decision"], "reject")
        with self.assertRaises(ValueError):
            p28._validate_route_gate(  # noqa: SLF001
                {
                    "schema_version": "p28-route-gate/v1",
                    "route_id": "rgt_ked",
                    "decision": "reject",
                    "reason_code": "negative_transfer",
                    "stop": True,
                    "extra": "not allowed",
                }
            )

    def test_selection_schema_enforces_four_trial_stop(self) -> None:
        available = {fold: ["kernel_matern32"] for fold in range(5)}
        value = {
            "decisions": [
                {
                    "held_fold": fold,
                    "action_id": "kernel_matern32",
                    "rationale": "bounded final trial",
                }
                for fold in range(5)
            ],
        }
        rows = p28._validate_selection(  # noqa: SLF001
            value, round_id=4, available_by_fold=available
        )
        self.assertEqual(len(rows), 5)
        value["decisions"][0]["extra"] = False
        with self.assertRaises(ValueError):
            p28._validate_selection(  # noqa: SLF001
                value, round_id=4, available_by_fold=available
            )

    def test_random_search_is_reproducible_and_not_random_init(self) -> None:
        first = p28._random_sequences()  # noqa: SLF001
        second = p28._random_sequences()  # noqa: SLF001
        self.assertEqual(first, second)
        for actions in first.values():
            self.assertEqual(len(actions), p28.TRIALS_PER_STRATEGY)
            self.assertEqual(len(set(actions)), p28.TRIALS_PER_STRATEGY)
            self.assertTrue(all(action in p28.ACTION_BY_ID for action in actions))
            self.assertFalse(any("random_init" in action for action in actions))

    def test_p19_coordinate_purge_removes_forbidden_rows(self) -> None:
        fold = p17.FoldSamples(
            fold_id=1,
            train_target=np.asarray([0.1, 0.2, 0.3]),
            train_raw_features=np.arange(18, dtype=np.float64).reshape(3, 6),
            validation_raw_features=np.zeros((1, 6), dtype=np.float64),
            train_indices_kji=np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            validation_indices_kji=np.asarray([[0, 0, 0]]),
            source_hashes={},
        )
        purged, removed = p19._without_coordinates(  # noqa: SLF001
            fold, {(4, 5, 6)}
        )
        self.assertEqual(removed, 1)
        self.assertEqual(len(purged.train_target), 2)
        self.assertNotIn((4, 5, 6), p19._rows(purged.train_indices_kji))  # noqa: SLF001

    def test_strategy_auc_uses_exact_four_round_budget(self) -> None:
        histories = {
            fold: [
                {"feedback": {"relative_gain": gain}}
                for gain in (0.0, 0.001, -0.01, 0.002)
            ]
            for fold in range(5)
        }
        self.assertAlmostEqual(p28._strategy_auc(histories), 0.001)  # noqa: SLF001
        histories[0].pop()
        with self.assertRaises(RuntimeError):
            p28._strategy_auc(histories)  # noqa: SLF001


class P28AgenticOptimizationArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = TRACK / "_outputs" / "p28_agentic_optimization"
        if not (cls.output / "summary.json").is_file():
            raise unittest.SkipTest("P28 execution artifacts have not been run yet")
        cls.summary = json.loads(
            (cls.output / "summary.json").read_text(encoding="utf-8")
        )

    def test_a0_a1_identity_and_hash(self) -> None:
        artifact = ROOT / self.summary["prediction_artifact"]["path"]
        with np.load(artifact, allow_pickle=False) as payload:
            a0 = payload["a0_prediction"]
            a1 = payload["a1_prediction"]
        np.testing.assert_array_equal(a0, a1)
        self.assertEqual(
            p17._array_sha256(a0),  # noqa: SLF001
            p28.EXPECTED_A0_ARRAY_SHA256,
        )
        self.assertEqual(
            self.summary["a0"]["metrics"]["rmse"], p28.EXPECTED_A0_RMSE
        )

    def test_equal_four_trial_budgets_and_real_provider(self) -> None:
        for strategy in ("A2L", "A2D", "A3"):
            row = self.summary["strategies"][strategy]
            self.assertEqual(row["trials"], 20)
            self.assertEqual(row["trials_per_held_fold"], 4)
        self.assertEqual(len(self.summary["a2l_provider_rounds"]), 4)
        self.assertTrue(self.summary["independent_route_gate"]["passed"])
        self.assertTrue(
            all(
                row["provider_audit"]["provider"] == "deepseek"
                for row in self.summary["a2l_provider_rounds"]
            )
        )
        self.assertFalse(
            self.summary["decision"]["random_init_foundation_used_as_a3"]
        )

    def test_firewall_and_purge_audit(self) -> None:
        firewall = self.summary["firewall"]
        self.assertFalse(firewall["test_h5_opened"])
        self.assertFalse(firewall["frozen_holdout_opened"])
        self.assertFalse(firewall["held_fold_metrics_visible_to_strategy"])
        self.assertFalse(firewall["promotion_feedback_visible_to_strategy"])
        self.assertEqual(len(self.summary["purge_audits"]), 5)
        self.assertTrue(
            all(row["removed_occurrences"] > 0 for row in self.summary["purge_audits"])
        )

    def test_independent_verification_and_manifest(self) -> None:
        verification = p28.verify_evidence(self.output)
        self.assertEqual(verification["status"], "PASSED")
        manifest = p28.write_artifact_manifest(self.output)
        self.assertEqual(len(manifest["artifacts"]), 9)
        for row in manifest["artifacts"]:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(p28._sha256(path), row["sha256"])  # noqa: SLF001
            self.assertEqual(path.stat().st_size, row["bytes"])


if __name__ == "__main__":
    unittest.main()
