from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

from openpyxl import load_workbook

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import p10_model_results


OUTPUT_DIR = TRACK_DIR / "_outputs" / "p10_model_results"


class P10ModelResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputs = p10_model_results.build()

    def test_expected_artifacts_exist(self) -> None:
        expected = {
            "track_model_metrics.xlsx",
            "figures_manifest.csv",
            "tables_manifest.csv",
            "audit_report.md",
            "before_after_primary_metric.png",
        }
        self.assertEqual(set(self.outputs), expected)
        for path in self.outputs.values():
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)

    def test_workbook_reopens_with_single_metrics_sheet(self) -> None:
        workbook = load_workbook(self.outputs["track_model_metrics.xlsx"], read_only=True, data_only=True)
        self.assertEqual(workbook.sheetnames, ["模型指标"])
        sheet = workbook["模型指标"]
        header = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
        self.assertEqual(header[0:6], ["track", "dataset", "task_type", "model_name", "model_family", "is_foundation_model"])
        self.assertIn("checkpoint_path", header)
        self.assertIn("evidence_path", header)

    def test_manifest_paths_exist(self) -> None:
        for manifest_name in ("figures_manifest.csv", "tables_manifest.csv"):
            with self.outputs[manifest_name].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            for row in rows:
                path = TRACK_DIR.parents[2] / row["path"]
                self.assertTrue(path.exists(), path)

    def test_audit_report_mentions_non_beneficial(self) -> None:
        text = self.outputs["audit_report.md"].read_text(encoding="utf-8")
        self.assertIn("non-beneficial SAM2 integration", text)
        self.assertIn("non_beneficial", text)


if __name__ == "__main__":
    unittest.main()
