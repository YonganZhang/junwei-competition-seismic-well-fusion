from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "_pipelines" / "02_task_datasets" / "sweetspot" / "p5" / "_outputs" / "3d_sci_v1"
FEASIBILITY_PATH = OUTPUT_DIR / "three_d_feasibility.json"
PROVENANCE_PATH = OUTPUT_DIR / "provenance.json"
CAPTION_PATH = OUTPUT_DIR / "caption.md"


class SweetspotP5ThreeDSciV1Tests(unittest.TestCase):
    def test_not_feasible_and_no_fake_3d_artifacts(self) -> None:
        self.assertTrue(FEASIBILITY_PATH.is_file())
        self.assertTrue(PROVENANCE_PATH.is_file())
        self.assertTrue(CAPTION_PATH.is_file())

        feasibility = json.loads(FEASIBILITY_PATH.read_text(encoding="utf-8"))
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(feasibility["verdict"], "not_feasible")
        self.assertEqual(provenance["verdict"], "not_feasible")
        self.assertIn("registered 3D grid prediction", feasibility["missing_items"])
        self.assertIn("real X coordinate", feasibility["missing_items"])
        self.assertIn("real Y coordinate", feasibility["missing_items"])
        self.assertIn("real Z coordinate", feasibility["missing_items"])
        self.assertEqual(len(feasibility["prediction_archives"]), 4)

        names = {path.name for path in OUTPUT_DIR.iterdir()}
        self.assertEqual(names, {"three_d_feasibility.json", "provenance.json", "caption.md"})
        self.assertFalse((OUTPUT_DIR / "visualize_3d_sci.py").exists())
        self.assertFalse(any(path.suffix.lower() in {".png", ".pdf", ".html"} for path in OUTPUT_DIR.rglob("*")))


if __name__ == "__main__":
    unittest.main()
