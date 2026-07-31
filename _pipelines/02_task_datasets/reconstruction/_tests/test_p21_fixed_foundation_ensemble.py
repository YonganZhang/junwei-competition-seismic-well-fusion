from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


TRACK = Path(__file__).resolve().parents[1]
ROOT = TRACK.parents[2]
sys.path[:0] = [str(TRACK), str(ROOT)]

import p21_fixed_foundation_ensemble as p21  # noqa: E402


class P21FixedFoundationEnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = TRACK / "_outputs" / "p21_fixed_foundation_ensemble"
        cls._temporary_output = tempfile.TemporaryDirectory(
            prefix="p21-verification-", dir=TRACK / "_outputs"
        )
        cls.output = Path(cls._temporary_output.name)
        shutil.copytree(source, cls.output, dirs_exist_ok=True)
        cls.summary = json.loads(
            (cls.output / "summary.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_output.cleanup()

    def test_protocol_and_fixed_candidates(self) -> None:
        protocol = self.summary["protocol"]
        self.assertEqual(protocol["outer_spatial_folds"], [0, 1, 2, 3, 4])
        self.assertEqual(protocol["train_labels_per_fold"], 512)
        self.assertEqual(protocol["validation_rows_per_fold"], 2048)
        self.assertFalse(protocol["candidate_selection_uses_labels"])
        self.assertFalse(protocol["test_h5_opened"])
        self.assertFalse(protocol["holdout_opened"])
        self.assertEqual(len(self.summary["fixed_candidate_names"]), 3)

    def test_candidate_strictly_improves_without_fold_losses(self) -> None:
        comparison = self.summary["comparison"]
        self.assertLess(
            comparison["candidate"]["rmse"], comparison["p19"]["rmse"]
        )
        self.assertEqual(comparison["outcomes_vs_p19"], {"loss": 0, "tie": 4, "win": 1})
        self.assertEqual(self.summary["decision"]["state"], "ACCEPTED_SIMPLICITY_WIN")
        self.assertTrue(self.summary["decision"]["default_enabled"])
        self.assertFalse(self.summary["decision"]["causal_foundation_contribution_claimed"])

    def test_four_folds_are_prediction_equivalent(self) -> None:
        artifact = ROOT / self.summary["prediction_artifact"]["path"]
        with np.load(artifact, allow_pickle=False) as payload:
            fold_ids = payload["fold_ids"]
            p19_prediction = payload["p19_prediction"]
            candidate = payload["candidate_prediction"]
        equivalent = 0
        for fold_id in range(5):
            mask = fold_ids == fold_id
            if np.max(np.abs(candidate[mask] - p19_prediction[mask])) < 1e-12:
                equivalent += 1
        self.assertEqual(equivalent, 4)

    def test_rejected_residual_routes_remain_disabled(self) -> None:
        rows = self.summary["rejected_residual_routes"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["rmse_delta_vs_p19"] > 0.0 for row in rows))
        self.assertFalse(any("accept" in row["verdict"] for row in rows))

    def test_delivery_default_freezes_p21_and_disables_peft_routes(self) -> None:
        config = json.loads(
            (ROOT / "_models" / "reconstruction" / "default_model.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["active_method"], "p21_fixed_foundation_ensemble")
        self.assertTrue(config["default_enabled"])
        self.assertTrue(config["foundation_model"]["backbone_frozen"])
        self.assertFalse(config["fixed_inference"]["per_fold_model_selection"])
        self.assertEqual(config["fixed_inference"]["seismic_weights"], [0.0, 0.1, 0.2])
        self.assertEqual(config["fixed_inference"]["kernel_mean_weight"], 0.75)
        self.assertIn("staged_lora_r4", config["disabled_training_routes"])
        self.assertIn("staged_adapter", config["disabled_training_routes"])

    def test_independent_artifact_verification(self) -> None:
        verification = p21.verify_evidence(self.output)
        self.assertEqual(verification["status"], "PASSED")
        self.assertEqual(verification["fold_losses_vs_p19"], 0)

    def test_artifact_manifest_hashes_portable_outputs(self) -> None:
        manifest = p21.write_artifact_manifest(self.output)
        self.assertEqual(len(manifest["artifacts"]), 5)
        for row in manifest["artifacts"]:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(p21._sha256(path), row["sha256"])  # noqa: SLF001
            self.assertEqual(path.stat().st_size, row["bytes"])


if __name__ == "__main__":
    unittest.main()
