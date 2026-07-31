from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

import numpy as np

from visualize_3d_sci import FIELDS, load_context, plane_coordinates, render_html, sha256_file


class FaultThreeDSciTests(unittest.TestCase):
    def test_real_spatial_context_contract(self) -> None:
        context = load_context()
        self.assertEqual(FIELDS, ("seismic", "truth", "probability"))
        self.assertEqual(context["patches"].shape, (3, 33, 65))
        self.assertEqual(context["truth"].shape, (3, 33, 65))
        self.assertEqual(context["probability"].shape, (3, 33, 65))
        self.assertEqual(len(set(context["selected_indices"])), 3)
        self.assertTrue(np.isfinite(context["patches"]).all())
        self.assertTrue(np.isfinite(context["probability"]).all())
        self.assertTrue(0.0 < context["time_step_ms"] < 20.0)

        for patch, position in zip(context["patches"], context["positions"]):
            xx, yy, zz = plane_coordinates(position, patch.shape, context["time_step_ms"])
            self.assertEqual(xx.shape, patch.shape)
            self.assertEqual(yy.shape, patch.shape)
            self.assertEqual(zz.shape, patch.shape)
            self.assertTrue(np.all(xx == float(position["inline"])))
            self.assertAlmostEqual(float(np.median(yy)), float(position["crossline"]))
            self.assertAlmostEqual(float(np.median(zz)), float(position["time_ms"]))

    def test_html_explicitly_marks_spatial_context_and_has_no_title(self) -> None:
        context = load_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = render_html(context, Path(tmpdir))
            body = html_path.read_text(encoding="utf-8")
            self.assertIn("spatial context only", body.lower())
            self.assertNotIn("<title>", body.lower())
            self.assertIn("Inline", body)
            self.assertIn("Crossline", body)
            self.assertIn("TWT", body)

    def test_provenance_records_real_source_hashes(self) -> None:
        context = load_context()
        provenance = json.loads((Path(__file__).resolve().parent / "_outputs" / "3d_sci_v1" / "provenance.json").read_text(encoding="utf-8"))
        inputs = provenance["inputs"]
        expected = {
            "checkpoint": Path(__file__).resolve().parent / "_outputs" / "runs" / "audited_v2" / "checkpoints" / "best.ckpt",
            "test_h5": Path(__file__).resolve().parents[3] / "_data" / "processed" / "fault" / "test.h5",
            "visualization_report": Path(__file__).resolve().parent / "_outputs" / "runs" / "audited_v2" / "visualization_report.json",
        }
        for key, path in expected.items():
            self.assertEqual(inputs[key]["sha256"], sha256_file(path))
        self.assertEqual(provenance["code_sha256"], sha256_file(Path(__file__).resolve().parent / "visualize_3d_sci.py"))
        self.assertEqual(context["checkpoint_sha256"], inputs["checkpoint"]["sha256"])


if __name__ == "__main__":
    unittest.main()
