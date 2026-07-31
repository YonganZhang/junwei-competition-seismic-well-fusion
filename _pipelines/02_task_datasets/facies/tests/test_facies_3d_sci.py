from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

OUTPUT_ROOT = TRACK_DIR / "_outputs" / "3d_sci_v1"


class Facies3DSciFeasibilityTests(unittest.TestCase):
    def test_not_feasible_contract_and_no_fake_3d_outputs(self) -> None:
        feasibility = json.loads((OUTPUT_ROOT / "three_d_feasibility.json").read_text())
        provenance = json.loads((OUTPUT_ROOT / "provenance.json").read_text())
        caption = (OUTPUT_ROOT / "caption.md").read_text(encoding="utf-8")

        self.assertEqual(feasibility["verdict"], "not_feasible")
        self.assertIn("independent 2D patches", feasibility["reason"])
        self.assertIn("not_feasible", provenance["verdict"])
        self.assertTrue(provenance["limits"]["no_stack_of_unordered_2d_samples"])
        self.assertIn("not_feasible", caption)

        forbidden = []
        for suffix in ("*.png", "*.pdf", "*.html"):
            forbidden.extend(sorted(str(path) for path in OUTPUT_ROOT.glob(suffix)))
        self.assertEqual(forbidden, [])

        self.assertFalse((TRACK_DIR / "visualize_3d_sci.py").exists())
        self.assertEqual(
            {item["path"] for item in feasibility["checked_files"]},
            {
                "_wiki-methodology/_top/_phases/P5_three_dimensional_sci_visualization_contract.md",
                "_pipelines/02_task_datasets/facies/pipeline_contract.py",
                "_pipelines/02_task_datasets/facies/build_dataset.py",
                "_pipelines/02_task_datasets/facies/visualize_predictions.py",
                "_pipelines/02_task_datasets/facies/_outputs/leakage_fixed_v2/prediction_visualization_evidence.json",
            },
        )
        self.assertEqual(provenance["sampling"]["mode"], "none")
        self.assertTrue(provenance["limits"]["no_png_pdf_html_written"])


if __name__ == "__main__":
    unittest.main()
