from __future__ import annotations

import json
from pathlib import Path
import unittest


TRACK = Path(__file__).resolve().parents[1]


class P23CheckshotCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary_path = (
            TRACK / "_outputs" / "p23_checkshot_calibration" / "summary.json"
        )
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))

    def test_protocol_keeps_fit_and_validation_wells_disjoint(self) -> None:
        protocol = self.summary["protocol"]
        fit = set(protocol["fit_wells"])
        validation = set(protocol["independent_validation_wells"])
        self.assertEqual(fit, {"19A", "19BT2", "19SR"})
        self.assertEqual(validation, {"F11T2", "F15A"})
        self.assertFalse(fit & validation)
        self.assertFalse(protocol["porosity_labels_used"])
        self.assertEqual(protocol["reconstruction_hdf5_opened"], [])
        self.assertFalse(protocol["holdout_opened"])

    def test_independent_wells_both_improve(self) -> None:
        rows = self.summary["per_validation_well"]
        self.assertEqual(set(rows), {"F11T2", "F15A"})
        for row in rows.values():
            self.assertLess(
                row["checkshot_candidate"]["mae_ms"], row["weak"]["mae_ms"]
            )
        self.assertEqual(self.summary["decision"]["validation_well_wins"], 2)

    def test_pooled_calibration_is_strictly_better(self) -> None:
        pooled = self.summary["pooled"]
        self.assertEqual(pooled["weak"]["rows"], 80)
        self.assertEqual(pooled["checkshot_candidate"]["rows"], 80)
        self.assertAlmostEqual(pooled["weak"]["mae_ms"], 633.1867277468943)
        self.assertAlmostEqual(
            pooled["checkshot_candidate"]["mae_ms"], 8.738925152523326
        )
        self.assertTrue(self.summary["decision"]["strict_pooled_mae_improvement"])
        self.assertTrue(
            self.summary["decision"]["eligible_for_downstream_alignment_evaluation"]
        )

    def test_artifact_claim_boundary_and_source_lock(self) -> None:
        self.assertEqual(
            self.summary["schema_version"],
            "reconstruction-p23-checkshot-calibration/v1",
        )
        self.assertEqual(self.summary["status"], "VALIDATED_CALIBRATION_ONLY")
        self.assertFalse(self.summary["decision"]["porosity_blind_test_claimed"])
        self.assertEqual(
            self.summary["inputs"]["vsp_zip_sha256"],
            "e3c7f0ce7fb2590bc2dc0a24be6df5d90af174c6fb782d95424463e615acc8f4",
        )


if __name__ == "__main__":
    unittest.main()
