from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import agent_analysis_chapter as chapter  # noqa: E402


class AgentAnalysisChapterTests(unittest.TestCase):
    def test_gate_reinitialization_changes_only_gate(self) -> None:
        fusion = chapter.p13.CrossAttentionFusion()
        before = {
            name: value.detach().clone()
            for name, value in fusion.state_dict().items()
        }
        chapter._set_fusion_scale_initialization(fusion, 0.5)
        self.assertAlmostEqual(fusion.fusion_scale, 0.5, places=6)
        for name, value in fusion.state_dict().items():
            if name != "fusion_scale_logit":
                torch.testing.assert_close(value, before[name])

    def test_output_boundary_protects_prior_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "overwrite"):
            chapter._validate_output_root(chapter.p13.OUTPUT_ROOT)

    def test_deterministic_execution_is_requested(self) -> None:
        chapter._request_deterministic_execution()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_mixed_task_direction_is_not_promoted(self) -> None:
        verdict = chapter._verdict(
            0.02,
            {
                "F3": {"delta_miou": 0.04},
                "Penobscot": {"delta_miou": -0.002},
            },
        )
        self.assertEqual(verdict, "MIXED_TASK_RESULT")

    def test_generated_evidence_contract(self) -> None:
        verified = chapter.verify()
        self.assertEqual(verified["rows"], 4)
        self.assertEqual(verified["artifacts"], 4)
        self.assertFalse(verified["frozen_test_accessed"])
        self.assertFalse(verified["api_key_persisted"])


if __name__ == "__main__":
    unittest.main()
