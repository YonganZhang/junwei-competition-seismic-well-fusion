from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path
import subprocess

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager


TRACK_DIR = Path(__file__).resolve().parent


def _load_fault_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_fault_module("fault_p12_visualization", "p12_visualization.py")


class FaultP12VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tempdir_one = tempfile.TemporaryDirectory()
        cls._tempdir_two = tempfile.TemporaryDirectory()
        cls.output_root_one = Path(cls._tempdir_one.name) / "p12_visualization"
        cls.output_root_two = Path(cls._tempdir_two.name) / "p12_visualization"
        cls.context = renderer._load_reported_context()
        cls.real_panel = renderer.build_real_test_panel(cls.context)
        cls.spatial_figure = renderer.build_spatial_context_figure(cls.context)
        cls.manifest_one = renderer.build_publication(cls.output_root_one)
        cls.manifest_two = renderer.build_publication(cls.output_root_two)
        cls.manifest_disk = json.loads((renderer.PUBLISHED_OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close(cls.real_panel)
        plt.close(cls.spatial_figure)
        cls._tempdir_one.cleanup()
        cls._tempdir_two.cleanup()

    def test_manifest_contract_has_required_fields(self) -> None:
        manifest = self.manifest_disk
        contract = manifest["p12_contract"]
        self.assertEqual(contract["schema_version"], "scientific-visualization-contract/v1")
        self.assertEqual(contract["profile"], "p12_tracks_1_3_5")
        self.assertEqual(contract["track_id"], "fault")
        self.assertEqual(
            contract["renderer"]["path"],
            "_pipelines/02_task_datasets/fault/p12_visualization.py",
        )
        self.assertEqual(contract["renderer"]["sha256"], renderer.sha256_file(TRACK_DIR / "p12_visualization.py"))
        ancestor_check = subprocess.run(
            [
                "git",
                "-C",
                str(renderer.PROJECT_ROOT),
                "merge-base",
                "--is-ancestor",
                contract["source_commit"],
                renderer.git_head(),
            ],
            check=False,
        )
        self.assertEqual(ancestor_check.returncode, 0)
        self.assertEqual(manifest["source_worktree"], ".claude/worktrees/track-fault")
        self.assertEqual(manifest["visual_qa_metadata"]["palette"], "Akun")
        self.assertEqual(manifest["visual_qa_metadata"]["probability_scale"], [0.0, 1.0])
        self.assertTrue(manifest["visual_qa_metadata"]["no_titles"])
        self.assertEqual(manifest["split_scope"]["scope"], "audited_v2 test split only")
        self.assertIn("native 3-D volume reconstruction", manifest["caveat"])
        review = contract["manual_review"]
        self.assertFalse(review["reviewed"])
        self.assertIsNone(review["reviewed_sha256"])
        self.assertIsNone(review["reviewer"])
        self.assertIsNone(review["no_clipping"])
        self.assertIsNone(review["no_overlap"])
        self.assertIsNone(review["labels_legible"])
        self.assertIsNone(review["colors_consistent"])
        self.assertIsNone(review["scientific_boundary_preserved"])

        input_paths = {entry["path"] for entry in contract["inputs"]}
        expected_inputs = {
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/baseline.py",
            ".claude/worktrees/track-fault/_code/dataset_io.py",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/visualization_report.json",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/build_summary.json",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/split_manifest.json",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/checkpoints/best.ckpt",
            ".claude/worktrees/track-fault/_data/processed/fault/test.h5",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/prediction_visualization.png",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/3d_sci_v1/seismic_spatial_context.png",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/3d_sci_v1/truth_spatial_context.png",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/3d_sci_v1/probability_spatial_context.png",
            ".claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/3d_sci_v1/provenance.json",
        }
        self.assertTrue(expected_inputs.issubset(input_paths))
        for record in contract["inputs"]:
            for field in ("path", "sha256", "shape_or_row_count", "scientific_role", "split_scope"):
                self.assertIn(field, record)
            self.assertEqual(len(record["sha256"]), 64)

        output_paths = {entry["path"] for entry in contract["outputs"]}
        expected_output_paths = {
            "_pipelines/02_task_datasets/fault/_outputs/p12_visualization/figures/real_test_panel.png",
            "_pipelines/02_task_datasets/fault/_outputs/p12_visualization/figures/spatial_context.png",
        }
        self.assertEqual(output_paths, expected_output_paths)
        self.assertEqual(len(contract["outputs"]), 2)
        for record in contract["outputs"]:
            for field in ("role", "path", "sha256", "width_px", "height_px", "dpi", "vector_companions"):
                self.assertIn(field, record)
            self.assertEqual(record["dpi"], 300)
            self.assertGreater(record["width_px"], 0)
            self.assertGreater(record["height_px"], 0)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertTrue((renderer.PROJECT_ROOT / record["path"]).exists())
            self.assertEqual(sorted(record["vector_companions"]), ["pdf", "svg"])

    def test_deterministic_two_render_hashes_match(self) -> None:
        first = {Path(record["path"]).name: record["sha256"] for record in self.manifest_one["p12_contract"]["outputs"]}
        second = {Path(record["path"]).name: record["sha256"] for record in self.manifest_two["p12_contract"]["outputs"]}
        self.assertEqual(first, second)

    def test_figures_have_lowercase_labels_and_no_titles(self) -> None:
        selected_font = self.manifest_disk["visual_qa_metadata"]["font_status"]["selected"]
        for fig, labels in ((self.real_panel, {"(a)", "(b)", "(c)", "(d)"}), (self.spatial_figure, {"(a)", "(b)", "(c)"})):
            self.assertTrue(all(not axis.get_title() for axis in fig.axes))
            suptitle = getattr(fig, "_suptitle", None)
            self.assertTrue(suptitle is None or not suptitle.get_text().strip())
            found = {text.get_text() for text in fig.findobj(match=plt.Text) if text.get_text() in labels}
            self.assertEqual(found, labels)
            for text in fig.findobj(match=plt.Text):
                if text.get_text().strip():
                    self.assertEqual(text.get_fontfamily()[0], selected_font)

    def test_title_guard_and_hash_integrity_fail_closed(self) -> None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.set_title("forbidden")
        with self.assertRaisesRegex(RuntimeError, "axis titles"):
            renderer._no_titles(fig)
        plt.close(fig)

        contract = self.manifest_disk["p12_contract"]
        for record in contract["inputs"]:
            path = renderer.SHARED_PROJECT_ROOT / record["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"])

    def test_renderer_is_real_data_only_and_reports_lineage(self) -> None:
        manifest = self.manifest_disk
        contract = manifest["p12_contract"]
        self.assertEqual(self.context.report["selected_sample_indices"], [5, 15, 40])
        self.assertEqual(self.context.representative_index, 15)
        self.assertEqual(self.context.selected_positions[1]["time_index"], 725)
        self.assertEqual(manifest["visual_qa_metadata"]["manual_review"]["reviewed"], False)
        source_test_entry = next(
            record for record in contract["inputs"] if record["scientific_role"] == "canonical audited_v2 test evidence source"
        )
        self.assertEqual(source_test_entry["shape_or_row_count"], 96)
        regression_entry = next(
            record for record in contract["inputs"] if record["scientific_role"] == "archived audited_v2 regression evidence"
        )
        self.assertEqual(regression_entry["shape_or_row_count"]["width_px"], 1910)
        self.assertEqual(contract["outputs"][0]["width_px"], mpimg.imread(self.output_root_one / "figures" / "real_test_panel.png").shape[1])
        self.assertEqual(contract["outputs"][0]["height_px"], mpimg.imread(self.output_root_one / "figures" / "real_test_panel.png").shape[0])

        png = mpimg.imread(self.output_root_one / "figures" / "real_test_panel.png")
        self.assertEqual(png.ndim, 3)
        self.assertGreater(png.shape[0], 100)
        self.assertGreater(png.shape[1], 100)

        for rel in ("real_test_panel.pdf", "spatial_context.pdf"):
            with (self.output_root_one / "figures" / rel).open("rb") as handle:
                self.assertEqual(handle.read(4), b"%PDF")

        for rel in ("real_test_panel.svg", "spatial_context.svg"):
            tree = element_tree.parse(self.output_root_one / "figures" / rel)
            root = tree.getroot()
            self.assertTrue(root.tag.endswith("svg"))
            titles = root.findall(".//{http://www.w3.org/2000/svg}title")
            self.assertEqual(titles, [])


if __name__ == "__main__":
    unittest.main()
