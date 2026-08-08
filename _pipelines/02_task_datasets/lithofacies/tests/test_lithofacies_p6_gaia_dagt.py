"""Lithofacies P6 Gaia/DAGT private package tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_p6_gaia_dagt as p6  # noqa: E402
from _models.gaia_dagt import ModelBatch, TrackSpec  # noqa: E402


class LithofaciesP6GaiaDAGTTests(unittest.TestCase):
    def test_final_state_is_one_of_the_approved_states(self) -> None:
        bundle = p6.build_bundle()
        self.assertEqual(bundle["final_research_state"], "no_verified_gain")
        self.assertIn(bundle["agent_mode"], {"supervisory_qc_agent", "agent_unavailable"})
        self.assertFalse(bundle["predictive_text_available"])
        self.assertFalse(bundle["f2_c1_c2_available"])
        self.assertEqual(bundle["gaia_contract"]["track_spec"]["prompt_version"], p6.DEFAULT_PROMPT_VERSION)
        self.assertEqual(bundle["gaia_contract"]["track_spec"]["source_manifest_digest"], p6.DEFAULT_SOURCE_MANIFEST.digest())

    def test_track_spec_batch_and_agent_round_trip(self) -> None:
        bundle = p6.build_bundle()
        spec = TrackSpec.from_dict(bundle["gaia_contract"]["track_spec"])
        batch = ModelBatch.from_dict(bundle["gaia_contract"]["dry_run"]["batch"])
        self.assertEqual(spec.track_id, "lithofacies_p6_gaia_dagt")
        self.assertEqual(batch.track_spec, spec)
        self.assertEqual(batch.agent_evidence.agent_mode, bundle["agent_mode"])
        self.assertEqual(tuple(batch.track_spec.target_fields), ("final_research_state",))

    def test_write_and_verify_bundle_are_self_consistent(self) -> None:
        with tempfile.TemporaryDirectory(dir=TRACK_DIR / "_outputs") as directory:
            out = Path(directory)
            payload = p6.write_bundle(output_dir=out)
            report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
            state = json.loads((out / "research_state.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "visualization_manifest.json").read_text(encoding="utf-8"))
            verification = p6.verify_bundle(output_dir=out)
            self.assertTrue(verification["pass"])
            self.assertEqual(payload["final_research_state"], "no_verified_gain")
            self.assertEqual(report["final_research_state"], state["final_research_state"])
            self.assertEqual(provenance["provenance_hashes"]["bundle_sha256"], report["provenance_hashes"]["bundle_sha256"])
            self.assertEqual(manifest["figures"][0]["figure_id"], "development_comparison")
            self.assertIn("unitless", manifest["figures"][0]["labels"]["y"])
            self.assertTrue((out / "figures" / "development_comparison.png").is_file())
            self.assertTrue((out / "figures" / "gaia_dagt_dry_run.svg").is_file())
            self.assertNotIn("/mnt/data/", json.dumps(report))
            self.assertNotIn(".claude/worktrees", json.dumps(report))

    def test_bundle_reports_diagnostic_only_agent_mode(self) -> None:
        bundle = p6.build_bundle()
        self.assertIn(bundle["agent_mode"], {"supervisory_qc_agent", "agent_unavailable"})
        self.assertEqual(bundle["visualization_manifest"]["caveats"][0], "development-only LOGO4 evidence")
        self.assertEqual(bundle["visualization_manifest"]["figures"][0]["figure_id"], "development_comparison")
        self.assertEqual(bundle["final_research_state"], "no_verified_gain")

    def test_direct_cli_build_and_verify_bootstraps_without_test_sys_path_help(self) -> None:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(dir=TRACK_DIR / "_outputs") as directory:
            out = Path(directory)
            build = subprocess.run(
                [
                    sys.executable,
                    str(TRACK_DIR / "lithofacies_p6_gaia_dagt.py"),
                    "build",
                    "--output-dir",
                    str(out),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            verify = subprocess.run(
                [
                    sys.executable,
                    str(TRACK_DIR / "lithofacies_p6_gaia_dagt.py"),
                    "verify",
                    "--output-dir",
                    str(out),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            build_payload = json.loads(build.stdout)
            verify_payload = json.loads(verify.stdout)
            self.assertEqual(build_payload["final_research_state"], "no_verified_gain")
            self.assertTrue(verify_payload["pass"])
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue((out / "provenance.json").is_file())
            self.assertTrue((out / "figures" / "development_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
