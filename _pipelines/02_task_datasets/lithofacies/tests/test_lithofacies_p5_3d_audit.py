"""Minimal tests for the lithofacies 3D feasibility audit artifacts."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parents[1]
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))


AUDIT_DIR = TRACK_DIR / "_outputs" / "3d_sci_v1"


class LithofaciesThreeDAuditTests(unittest.TestCase):
    def test_not_feasible_verdict_and_missing_spatial_evidence(self) -> None:
        feasibility = json.loads((AUDIT_DIR / "three_d_feasibility.json").read_text())
        provenance = json.loads((AUDIT_DIR / "provenance.json").read_text())
        caption = (AUDIT_DIR / "caption.md").read_text(encoding="utf-8")

        self.assertEqual(feasibility["verdict"], "not_feasible")
        self.assertTrue(
            any("XYZ" in item or "trajectory" in item for item in feasibility["missing_items"])
        )
        self.assertIn("well trajectory", feasibility["verified_coordinate_evidence"]["missing_spatial_evidence"])
        self.assertEqual(provenance["decision"], "not_feasible")
        self.assertEqual(provenance["evidence_summary"]["stage4_record_count"], 120)
        self.assertEqual(provenance["evidence_summary"]["stage4_finite_center_md_rows"], 0)
        self.assertIn("not_feasible", caption)
        self.assertIn("midpoint fabrication", caption)

    def test_no_fake_3d_products_are_present(self) -> None:
        self.assertTrue(AUDIT_DIR.is_dir())
        self.assertTrue((AUDIT_DIR / "three_d_feasibility.json").is_file())
        self.assertTrue((AUDIT_DIR / "provenance.json").is_file())
        self.assertTrue((AUDIT_DIR / "caption.md").is_file())
        forbidden = list(AUDIT_DIR.glob("*.png")) + list(AUDIT_DIR.glob("*.pdf")) + list(AUDIT_DIR.glob("*.html"))
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
