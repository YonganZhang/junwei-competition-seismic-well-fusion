from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fault_p18_cigbench import build_report, audit_gate


class FaultP18CIGBenchTests(unittest.TestCase):
    def test_gate_is_blocked_and_names_the_missing_3d_contract(self) -> None:
        status, report = audit_gate()
        self.assertEqual(status, "DATA_GATE_BLOCKED")
        self.assertEqual(report["reason_code"], "NO_VALID_FAULT_3D_DEVELOPMENT_VOLUME")
        codes = {entry["code"] for entry in report["reasons"]}
        self.assertIn("contiguous_3d_development_blocks_missing", codes)
        self.assertIn("coverage_audited_verified_background_missing", codes)
        self.assertIn("explicit_unknown_mask_provenance_missing", codes)
        self.assertIn("group_isolated_development_split_missing", codes)
        self.assertFalse(report["frozen_holdout_accessed"])

    def test_evidence_is_written_and_mentions_install_and_blocking_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = build_report(Path(tmpdir))
            evidence_path = Path(tmpdir) / "evidence.md"
            self.assertTrue(evidence_path.is_file())
            text = evidence_path.read_text(encoding="utf-8")
            self.assertIn("Fault CIG-Bench incremental comparison audit", text)
            self.assertIn("Install and weight proof", text)
            self.assertIn("Minimum unblock contract", text)
            self.assertIn("comparison is therefore blocked", text.lower())
            self.assertEqual(report["status"], "DATA_GATE_BLOCKED")


if __name__ == "__main__":
    unittest.main()
