from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import p13_cross_attention as p13  # noqa: E402


class P13CrossAttentionUnitTests(unittest.TestCase):
    def test_cross_attention_preserves_cnn_shape_and_backpropagates(
        self,
    ) -> None:
        module = p13.CrossAttentionFusion()
        module.eval()
        cnn = torch.randn(2, 512, 4, 4, requires_grad=True)
        sam = torch.randn(2, 256, 8, 8, requires_grad=True)
        fused, weights = module(cnn, sam)
        self.assertEqual(tuple(fused.shape), tuple(cnn.shape))
        self.assertEqual(tuple(weights.shape), (2, 16, 64))
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2, 16),
            rtol=1e-5,
            atol=1e-5,
        )
        fused.square().mean().backward()
        self.assertIsNotNone(cnn.grad)
        self.assertIsNotNone(sam.grad)
        self.assertGreater(float(sam.grad.abs().sum()), 0.0)

    def test_weight_mode_is_one_parameter_randomization_switch(
        self,
    ) -> None:
        sentinel = object()
        with mock.patch.object(
            p13.p12,
            "_make_sam2_encoder",
            return_value=sentinel,
        ) as builder:
            returned = p13.build_sam2_encoder(
                "facies_f3",
                10,
                "cpu",
                weight_mode="random",
                seed=42,
            )
        self.assertIs(returned, sentinel)
        builder.assert_called_once_with(
            "facies_f3",
            10,
            "cpu",
            randomize=True,
            random_seed=42,
        )
        self.assertEqual(
            p13._validate_sam2_weight_mode("pretrained"),
            "pretrained",
        )
        with self.assertRaisesRegex(ValueError, "weight mode"):
            p13._validate_sam2_weight_mode("unknown")

    def test_contract_and_development_boundary_are_fixed(self) -> None:
        self.assertEqual(p13.FOLDS, (0, 4))
        self.assertEqual(p13.BASELINE_UPDATES, 40)
        self.assertEqual(p13.CANDIDATE_UPDATES, 160)
        self.assertEqual(
            p13.SAM2_WEIGHT_MODES,
            ("pretrained", "random"),
        )
        with self.assertRaisesRegex(ValueError, "overwrite"):
            p13._validate_output_root(p13.p11.OUTPUT_ROOT)
        with self.assertRaisesRegex(
            ValueError,
            "frozen holdout path rejected",
        ):
            p13.p11.validate_development_inputs(
                f3_manifest=Path(
                    "/tmp/frozen_holdout/f3/split_manifest.json"
                ),
                penobscot_manifest=Path(
                    "/tmp/dev/penobscot/split_manifest.json"
                ),
                processed_root=Path("/tmp/dev/processed"),
            )


class P13CrossAttentionEvidenceTests(unittest.TestCase):
    def test_evidence_verifies_without_attribution_claim(self) -> None:
        verified = p13.verify()
        self.assertEqual(verified["rows"], 12)
        self.assertEqual(verified["artifacts"], 3)
        self.assertEqual(
            verified["sam2_weight_mode"],
            "pretrained",
        )
        self.assertFalse(verified["frozen_test_accessed"])


if __name__ == "__main__":
    unittest.main()
