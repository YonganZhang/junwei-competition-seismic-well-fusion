from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))

import p30_bounded_geostatistics_feasibility as p30  # noqa: E402


class P30BoundedGeostatisticsFeasibilityTest(unittest.TestCase):
    def test_denormalizes_declared_physical_units(self) -> None:
        bounds = np.asarray([[100.0, 200.0], [1_000.0, 1_400.0], [2_000.0, 2_100.0]])
        normalized = np.asarray([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]])
        np.testing.assert_allclose(
            p30.denormalize_coordinates(normalized, bounds),
            np.asarray([[100.0, 1_200.0, 2_100.0], [200.0, 1_000.0, 2_025.0]]),
        )

    def test_directional_variogram_is_train_only_and_finite(self) -> None:
        rng = np.random.default_rng(2693)
        coordinates = rng.uniform(0.0, 100.0, size=(96, 3))
        values = 0.2 + 0.02 * np.sin(coordinates[:, 0] / 20.0) + rng.normal(0.0, 0.005, 96)
        result = p30.fit_directional_variogram(coordinates, values)
        self.assertFalse(result["validation_target_used"])
        self.assertEqual(len(result["effective_ranges_m"]), 3)
        self.assertTrue(np.all(np.asarray(result["effective_ranges_m"]) > 0.0))
        self.assertTrue(all(row["pair_count"] > 0 for row in result["directions"]))

    def test_ordinary_kriging_honours_exact_training_coordinates(self) -> None:
        rng = np.random.default_rng(12)
        coordinates = rng.uniform(0.0, 50.0, size=(48, 3))
        values = rng.uniform(0.1, 0.3, size=48)
        variogram = p30.fit_directional_variogram(coordinates, values)
        prediction, variance, audit = p30.local_ordinary_kriging(
            train_coordinates_m=coordinates,
            train_values=values,
            query_coordinates_m=coordinates[:5],
            variogram=variogram,
            neighbours=16,
        )
        np.testing.assert_allclose(prediction, values[:5], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(variance, 0.0, rtol=0.0, atol=0.0)
        self.assertEqual(audit["exact_conditioned_queries"], 5)

    def test_ordinary_kriging_solves_non_exact_queries(self) -> None:
        rng = np.random.default_rng(13)
        coordinates = rng.uniform(0.0, 50.0, size=(48, 3))
        values = rng.uniform(0.1, 0.3, size=48)
        variogram = p30.fit_directional_variogram(coordinates, values)
        prediction, variance, audit = p30.local_ordinary_kriging(
            train_coordinates_m=coordinates,
            train_values=values,
            query_coordinates_m=np.asarray([[25.1, 24.9, 25.2], [10.1, 9.8, 10.3]]),
            variogram=variogram,
            neighbours=16,
        )
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertEqual(audit["exact_conditioned_queries"], 0)

    def test_physical_porosity_constraint_reports_changes(self) -> None:
        constrained, audit = p30._clip_physical(np.asarray([-0.1, 0.2, 1.1]))  # noqa: SLF001
        np.testing.assert_array_equal(constrained, np.asarray([0.0, 0.2, 1.0]))
        self.assertEqual(audit["violations_before_constraint"], 2)
        self.assertAlmostEqual(audit["maximum_absolute_change"], 0.1)

    def test_future_fusion_contract_forbids_silent_missing_modalities(self) -> None:
        contract = p30._fusion_contract()  # noqa: SLF001
        rendered = json.dumps(contract)
        self.assertIn("well_log_observations", contract["inputs"])
        self.assertIn("seismic_foundation_embedding", contract["inputs"])
        self.assertIn("missing modalities use explicit masks", rendered)
        self.assertIn("porosity_variance", contract["outputs"])

    def test_persisted_evidence_verifies_if_available(self) -> None:
        output = p30.DEFAULT_OUTPUT_DIR
        if not (output / "summary.json").is_file():
            self.skipTest("P30 development pilot has not been run")
        verification = p30.verify_evidence(output)
        self.assertEqual(verification["status"], "PASSED")
        self.assertFalse(verification["firewall"]["test_h5_opened"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(
            summary["conclusions"]["old_p29_outputs"],
            "HISTORICAL_ONLY_NOT_ELIGIBLE_FOR_NEW_PROMOTION",
        )
        self.assertIn("A0_EQUALS_P21", summary["conclusions"]["p29_repair"])
        self.assertTrue(
            summary["default_evidence_audit"]["p29"]
            ["authoritative_output_usable_for_new_promotion"]
        )
        self.assertEqual(summary["conclusions"]["anisotropic_ordinary_kriging"], "NO_PROMOTION")
        self.assertEqual(summary["conclusions"]["regression_kriging_cokriging_proxy"], "NO_PROMOTION")
        self.assertEqual(summary["direction_cone_audit"]["relaxed_fit_count"], 2)
        self.assertTrue(
            all(
                row["low_direction_resolution"]
                for row in summary["direction_cone_audit"]["relaxations"]
            )
        )
        self.assertIn("--verify-only", summary["reproduction"]["verify_only"])
        self.assertTrue((output / "finding.md").is_file())
        manifest = p30.write_artifact_manifest(output)
        self.assertEqual(len(manifest["artifacts"]), 10)
        self.assertEqual(p30.verify_artifact_manifest(output)["status"], "PASSED")


if __name__ == "__main__":
    unittest.main()
