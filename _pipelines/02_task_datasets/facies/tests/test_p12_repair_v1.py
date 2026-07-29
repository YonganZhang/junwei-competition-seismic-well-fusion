from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import p12_repair_v1 as p12  # noqa: E402


class P12RepairUnitTests(unittest.TestCase):
    def test_fixed_development_contract_matches_p11(self) -> None:
        self.assertEqual(p12.FOLDS, (0, 4))
        self.assertEqual(p12.MAX_UPDATES, 40)
        self.assertEqual(p12.BATCH_SIZE, 2)
        self.assertEqual(p12.MIN_PROMOTION_DELTA, 0.005)
        self.assertEqual(p12.VARIANTS, p12.p11.VARIANTS)
        self.assertEqual(p12.TRAINABLE_HIERA_BLOCKS, 2)
        self.assertEqual(p12.NATIVE_INPUT_SIZE, (128, 128))

    def test_native_normalization_preserves_spatial_information(self) -> None:
        inputs = torch.linspace(
            -5.0,
            5.0,
            steps=128 * 128,
        ).reshape(1, 1, 128, 128)
        normalized = p12._sam2_normalize_native(inputs)
        self.assertEqual(tuple(normalized.shape), (1, 3, 128, 128))
        self.assertTrue(bool(torch.isfinite(normalized).all()))
        for channel in range(3):
            self.assertEqual(
                torch.unique(normalized[0, channel]).numel(),
                128 * 128,
            )
        with self.assertRaisesRegex(ValueError, "locked to 128x128"):
            p12._sam2_normalize_native(
                torch.zeros(1, 1, 256, 256)
            )

    def test_only_last_two_hiera_blocks_are_trainable(self) -> None:
        class FakeEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(2, 2)
                self.trunk = torch.nn.Module()
                self.trunk.blocks = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(6)]
                )

        encoder = FakeEncoder()
        indices = p12._set_last_hiera_blocks_trainable(encoder)
        self.assertEqual(indices, (4, 5))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in encoder.stem.parameters()
            )
        )
        for index, block in enumerate(encoder.trunk.blocks):
            self.assertTrue(
                all(
                    parameter.requires_grad == (index >= 4)
                    for parameter in block.parameters()
                )
            )

    def test_p12_cannot_overwrite_p11_or_address_holdout(self) -> None:
        with self.assertRaisesRegex(ValueError, "overwrite"):
            p12._validate_output_root(p12.p11.OUTPUT_ROOT)
        with self.assertRaisesRegex(
            ValueError,
            "frozen holdout path rejected",
        ):
            p12.p11.validate_development_inputs(
                f3_manifest=Path(
                    "/tmp/frozen_holdout/f3/split_manifest.json"
                ),
                penobscot_manifest=Path(
                    "/tmp/dev/penobscot/split_manifest.json"
                ),
                processed_root=Path("/tmp/dev/processed"),
            )


class P12RepairEvidenceTests(unittest.TestCase):
    def test_evidence_verifies_trainable_native_sam2_contract(
        self,
    ) -> None:
        verified = p12.verify()
        self.assertEqual(verified["rows"], 20)
        self.assertEqual(verified["artifacts"], 3)
        self.assertEqual(
            verified["native_input_shape"],
            [3, 128, 128],
        )
        self.assertEqual(
            verified["trainable_hiera_blocks"],
            [22, 23],
        )
        self.assertFalse(verified["frozen_test_accessed"])
        summary = json.loads(
            (
                p12.OUTPUT_ROOT / "p12_repair_v1_summary.json"
            ).read_text(encoding="utf-8")
        )
        for task in ("F3", "Penobscot"):
            self.assertEqual(
                summary["tasks"][task]["variant_means"][
                    "gate_zero"
                ]["miou"],
                summary["tasks"][task]["variant_means"][
                    "strong_small_baseline"
                ]["miou"],
            )


if __name__ == "__main__":
    unittest.main()
