from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _models.gaia_dagt import AgentEvidence, ModelBatch

import facies_p6_gaia_dagt as p6


class FaciesP6GaiaDAGTEvidenceTests(unittest.TestCase):
    def test_private_package_locks_qc_mode_and_weight_gate(self) -> None:
        package = p6.build_package()
        self.assertEqual(package.conclusion["state"], "blocked_by_data")
        self.assertEqual(package.conclusion["agent_mode"], "supervisory_qc_agent")
        self.assertEqual(
            package.conclusion["scientific_claim_boundary"],
            "previously_seen_reusable_holdout",
        )
        self.assertEqual(package.manifest["approved_weight_gate"]["smp_fpn_r18"]["status"], "blocked")
        self.assertEqual(package.manifest["approved_weight_gate"]["smp_deeplabv3plus_r18"]["status"], "blocked")
        self.assertIn("B0", package.manifest["baseline_index"]["facies_f3"])
        self.assertIn("B1", package.manifest["baseline_index"]["facies_penobscot"])

    def test_track_spec_and_model_batch_round_trip(self) -> None:
        spec = p6.build_track_spec("facies_f3")
        restored = type(spec).from_dict(spec.to_dict())
        self.assertEqual(spec, restored)
        sample = {
            "seismic_patch": [[0.0, 1.0], [2.0, 3.0]],
            "label": [[0, 1], [1, 0]],
            "position": {"inline": 489, "crossline": 0},
            "meta": {"split": "development"},
        }
        batch = p6.build_model_batch(sample, "facies_f3")
        self.assertIsInstance(batch, ModelBatch)
        self.assertIsInstance(batch.agent_evidence, AgentEvidence)
        again = ModelBatch.from_dict(batch.to_dict())
        self.assertEqual(batch.track_spec, again.track_spec)
        self.assertEqual(batch.metadata["claim_boundary"], "previously_seen_reusable_holdout")
        self.assertEqual(again.agent_evidence.agent_mode, "supervisory_qc_agent")

    def test_qc_agent_evidence_rejects_predictive_leaks(self) -> None:
        evidence = p6.build_qc_agent_evidence("facies_penobscot")
        self.assertEqual(evidence.agent_mode, "supervisory_qc_agent")
        with self.assertRaises(ValueError):
            AgentEvidence(
                prompt_version=evidence.prompt_version,
                agent_mode="supervisory_qc_agent",
                source_text_hash=evidence.source_text_hash,
                structured_priors={"test_metric": 0.1},
                source_manifest_digest=evidence.source_manifest_digest,
            )

    def test_rendered_svg_and_manifest_are_writeable(self) -> None:
        package = p6.build_package()
        self.assertIn("Facies P6 Gaia/DAGT private evidence", package.figure_svg)
        self.assertIn("previously_seen_reusable_holdout", package.figure_svg)
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            paths = p6.write_package(output_root)
            self.assertTrue(paths["figure"].is_file())
            self.assertTrue(paths["manifest"].is_file())
            manifest = json.loads(paths["manifest"].read_text())
            self.assertEqual(manifest["schema_version"], p6.PIPELINE_VERSION)
            self.assertEqual(manifest["approved_weight_gate"]["smp_fpn_r18"]["status"], "blocked")

    def test_dry_run_serialization_is_lossless(self) -> None:
        package = p6.build_package().to_dict()
        self.assertEqual(package["conclusion"]["state"], "blocked_by_data")
        self.assertEqual(package["resource_log"]["shared_gaia_dagt_contract"]["tests"], 14)
        self.assertEqual(package["failure_log"]["f2"]["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
