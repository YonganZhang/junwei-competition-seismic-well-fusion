from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from step_01_validate_manifest import (  # noqa: E402
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    ValidationError,
    png_dimensions,
    validate_manifest,
    validate_path_policy,
)
from step_00_discover import PAUSED_TRACKS, TRACKS, discover  # noqa: E402
from step_02_stage_delivery import stage_delivery  # noqa: E402
from step_04_stage_p12_review import (  # noqa: E402
    P12ReviewError,
    _report_path,
    stage_p12_review,
)


class DeliveryPipelineTests(unittest.TestCase):
    def test_p12_profile_is_discoverable_without_reopening_paused_tracks(self) -> None:
        report = discover(PROJECT_ROOT, require_artifacts=False)
        self.assertEqual(report["profile"], "p12_tracks_1_3_5")
        self.assertEqual(tuple(report["included_tracks"]), ("fault", "property", "sweetspot"))
        self.assertEqual(tuple(report["paused_tracks"]), PAUSED_TRACKS)
        self.assertEqual(set(TRACKS), {"fault", "property", "sweetspot"})
        self.assertEqual(
            {item["track_number"] for item in report["tracks"]},
            {"1", "3", "5"},
        )
        self.assertTrue(all(item["actual_branch"] == item["branch"] for item in report["tracks"]))

    def test_p12_artifact_contracts_are_ready(self) -> None:
        report = discover(PROJECT_ROOT, require_artifacts=True)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            {item["track"]: item["status"] for item in report["tracks"]},
            {"fault": "ready", "property": "ready", "sweetspot": "ready"},
        )

    def test_rejects_protocol_status_paths(self) -> None:
        for path in (
            "_outputs/p5_r2_visualization/tracks/track_01_fault.png",
            "_outputs/example/status_gate.png",
            "_outputs/example/continuous_depth_not_feasible.png",
        ):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                validate_path_policy(path)

    def test_manifest_validates_exact_six_real_domain_images(self) -> None:
        report = validate_manifest(PROJECT_ROOT, DEFAULT_MANIFEST)
        self.assertEqual(report["validation_status"], "passed")
        self.assertEqual(report["validated_track_count"], 6)
        self.assertEqual(len({item["track"] for item in report["tracks"]}), 6)
        for item in report["tracks"]:
            self.assertEqual(item["validation_status"], "passed")
            self.assertGreaterEqual(item["resolved"]["png_width"], 500)
            self.assertGreaterEqual(item["resolved"]["png_height"], 300)
            self.assertTrue(item["resolved"]["source_commit_is_ancestor"])

    def test_staging_preserves_hashes(self) -> None:
        report = validate_manifest(PROJECT_ROOT, DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_tmp") as tmp:
            temp_root = Path(tmp)
            validated = temp_root / "validated_manifest.json"
            validated.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staged = stage_delivery(PROJECT_ROOT, validated, temp_root)
            self.assertEqual(staged["status"], "passed")
            self.assertEqual(staged["staged_track_count"], 6)
            for item in staged["tracks"]:
                width, height = png_dimensions(PROJECT_ROOT / item["source_image"])
                self.assertEqual((width, height), (item["png_width"], item["png_height"]))
                self.assertTrue((temp_root / "cards" / Path(item["staged_image"]).name).is_file())

    def test_p12_staging_requires_explicit_visual_review(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_tmp") as tmp:
            with self.assertRaises(P12ReviewError):
                stage_p12_review(
                    PROJECT_ROOT,
                    Path(tmp),
                    "codex-leader",
                    accept_visual_qa=False,
                )

    def test_p12_attestation_uses_project_relative_paths(self) -> None:
        source = (
            PROJECT_ROOT
            / ".claude/worktrees/p12-viz-fault"
            / TRACKS["fault"]["manifest"]
        )
        self.assertEqual(
            _report_path(PROJECT_ROOT, source),
            (
                ".claude/worktrees/p12-viz-fault/"
                "_pipelines/02_task_datasets/fault/"
                "_outputs/p12_visualization/manifest.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
