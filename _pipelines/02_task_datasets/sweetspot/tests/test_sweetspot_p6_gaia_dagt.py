from __future__ import annotations

import hashlib
import importlib
import json
import unittest
from pathlib import Path

from PIL import Image

from _models.gaia_dagt.contracts import canonical_json

private = importlib.import_module("_models.sweetspot.p6_gaia_dagt")

OUTPUT_DIR = private.P6_OUTPUT_DIR
BUNDLE_PATH = OUTPUT_DIR / "p6_gaia_dagt_bundle.json"
CONCLUSION_PATH = OUTPUT_DIR / "p6_gaia_dagt_conclusion.json"
RESOURCE_PATH = OUTPUT_DIR / "p6_gaia_dagt_resource_log.json"
FAILURE_PATH = OUTPUT_DIR / "p6_gaia_dagt_failure_log.json"
EVIDENCE_PATH = OUTPUT_DIR / "p6_gaia_dagt_evidence_index.json"
MANIFEST_PATH = OUTPUT_DIR / "p6_gaia_dagt_manifest.json"
FIGURE_PATH = OUTPUT_DIR / "figures" / "proxy_only_qc.png"


class SweetspotP6GaiaDAGTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [BUNDLE_PATH, CONCLUSION_PATH, RESOURCE_PATH, FAILURE_PATH, EVIDENCE_PATH, MANIFEST_PATH, FIGURE_PATH]
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("run the sweetspot P6 private evidence runner first")
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        cls.conclusion = json.loads(CONCLUSION_PATH.read_text(encoding="utf-8"))
        cls.resource = json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))
        cls.failure = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_trackspec_and_batch_roundtrip(self) -> None:
        for target_id in ("T1", "T2"):
            target = self.bundle["targets"][target_id]
            self.assertTrue(target["batch_roundtrip_equal"])
            self.assertEqual(target["track_spec"]["prompt_version"], "p6.gaia.dagt.v1")
            self.assertEqual(target["track_spec"]["source_manifest_digest"], self.bundle["source_manifest_digest"])
            self.assertEqual(target["agent_evidence"]["agent_mode"], "supervisory_qc_agent")
            self.assertIn(target["track_spec"]["task_type"], {"classification", "regression"})

    def test_deny_list_and_qc_evidence_present(self) -> None:
        deny = self.failure["deny_list_proof"]
        self.assertTrue(deny["predictive_mode_disabled"])
        self.assertTrue(deny["no_sample_level_text_available"])
        self.assertTrue(deny["no_test_or_holdout_inputs_used"])
        self.assertEqual(private.build_agent_evidence("T1").agent_mode, "supervisory_qc_agent")
        self.assertEqual(private.build_agent_evidence("T2").agent_mode, "supervisory_qc_agent")
        self.assertTrue(private.build_track_spec("T1").provenance["proxy_only"])
        self.assertTrue(private.build_track_spec("T2").provenance["proxy_only"])

    def test_lock_table_and_foundation_status(self) -> None:
        self.assertEqual(self.conclusion["status"], "PARTIAL_READY")
        self.assertEqual(self.conclusion["final_state"], "blocked_by_data")
        self.assertEqual(self.conclusion["target_states"]["T1"], "PARTIAL_READY")
        self.assertEqual(self.conclusion["target_states"]["T2"], "PARTIAL_READY")
        self.assertEqual(self.conclusion["target_states"]["T5"], "NOT_FEASIBLE")
        self.assertEqual(self.conclusion["target_states"]["T6"], "BLOCKED")
        self.assertEqual(self.conclusion["target_states"]["T7"], "BLOCKED")
        self.assertEqual(self.conclusion["target_states"]["F0"], "UNAVAILABLE")
        self.assertEqual(self.conclusion["target_states"]["F1"], "UNAVAILABLE")

    def test_proxy_only_output_and_manifest(self) -> None:
        self.assertTrue(self.manifest["all_paths_portable"])
        self.assertTrue(self.manifest["no_training"])
        self.assertTrue(self.manifest["no_downloads"])
        self.assertTrue(self.manifest["no_test_access"])
        self.assertTrue(FIGURE_PATH.is_file())
        with Image.open(FIGURE_PATH) as image:
            self.assertEqual(image.size, (2200, 1200))
        for row in self.manifest["artifacts"]:
            path = Path(row["path"])
            self.assertFalse(path.is_absolute())
            absolute = Path(private.REPO_ROOT) / path
            self.assertTrue(absolute.is_file())
            self.assertEqual(row["sha256"], hashlib.sha256(absolute.read_bytes()).hexdigest())

    def test_b0_b1_index_and_failure_log(self) -> None:
        self.assertEqual(set(self.evidence), {"B0", "B1"})
        self.assertEqual(self.evidence["B0"]["status"], "PARTIAL_READY")
        self.assertEqual(self.evidence["B1"]["status"], "PARTIAL_READY")
        self.assertEqual(self.resource["shared_contract_tests_required"], 14)
        self.assertTrue(self.resource["shared_contract_tests_ran"])
        self.assertFalse(self.resource["frozen_test_accessed"])
        self.assertTrue(self.resource["no_training"])
        self.assertTrue(self.resource["no_downloads"])
        self.assertTrue(self.resource["no_api_calls"])
        self.assertIn("F0", self.failure["blocked"])
        self.assertIn("F1", self.failure["blocked"])

    def test_bundle_hash_is_stable_and_state_is_allowed(self) -> None:
        self.assertIn(self.bundle["conclusion"]["status"], {"READY", "PARTIAL_READY", "BLOCKED", "NOT_FEASIBLE", "UNAVAILABLE"})
        self.assertIn(self.bundle["conclusion"]["final_state"], {"verified_gain", "foundation_gain_only", "agent_signal_but_no_baseline_win", "no_verified_gain", "blocked_by_data"})
        payload = dict(self.bundle)
        payload.pop("bundle_sha256", None)
        self.assertEqual(
            self.bundle["bundle_sha256"],
            hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
