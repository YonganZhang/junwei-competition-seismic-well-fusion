from __future__ import annotations

import importlib.util
import json
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_visualization_package.py"
SPEC = importlib.util.spec_from_file_location("build_visualization_package", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VisualizationPackageTests(unittest.TestCase):
    def test_inputs_have_unique_archive_paths(self) -> None:
        inputs = MODULE.iter_inputs()
        archive_paths = [row[1] for row in inputs]
        self.assertEqual(len(archive_paths), len(set(archive_paths)))
        self.assertGreaterEqual(
            sum(path.endswith(".png") for path in archive_paths),
            13,
        )

    def test_built_archive_matches_manifest(self) -> None:
        MODULE.main()
        manifest = json.loads(MODULE.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        with zipfile.ZipFile(MODULE.OUTPUT_ZIP, "r") as bundle:
            self.assertIsNone(bundle.testzip())
            members = set(bundle.namelist())
            self.assertIn("PACKAGE_MANIFEST.json", members)
            for record in manifest["files"]:
                self.assertIn(record["path"], members)
        self.assertEqual(
            manifest["stable_url"],
            "https://share.yongan.site/junwei-visualizations/"
            "junwei_visualizations_latest.zip",
        )


if __name__ == "__main__":
    unittest.main()
