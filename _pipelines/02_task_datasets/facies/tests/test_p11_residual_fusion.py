from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import p11_residual_fusion as p11  # noqa: E402


class P11ResidualFusionUnitTests(unittest.TestCase):
    def test_fixed_development_contract_is_frozen(self) -> None:
        self.assertEqual(p11.FOLDS, (0, 4))
        self.assertEqual(p11.MAX_UPDATES, 40)
        self.assertEqual(p11.BATCH_SIZE, 2)
        self.assertEqual(p11.MIN_PROMOTION_DELTA, 0.005)
        self.assertEqual(
            p11.VARIANTS,
            (
                "strong_small_baseline",
                "direct_sam2",
                "pretrained_residual",
                "random_sam2_residual",
                "gate_zero",
            ),
        )

    def test_development_resolver_rejects_holdout_before_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen holdout path rejected"):
            p11.validate_development_inputs(
                f3_manifest=Path("/tmp/frozen_holdout/f3/split_manifest.json"),
                penobscot_manifest=Path("/tmp/dev/penobscot/split_manifest.json"),
                processed_root=Path("/tmp/dev/processed"),
            )

    def test_development_resolver_requires_separate_manifests_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f3_manifest = root / "f3_split_manifest.json"
            pen_manifest = root / "pen_split_manifest.json"
            f3_manifest.write_text("{}\n", encoding="utf-8")
            pen_manifest.write_text("{}\n", encoding="utf-8")
            for task_id in ("facies_f3", "facies_penobscot"):
                task_root = root / "processed" / task_id
                task_root.mkdir(parents=True)
                (task_root / "train.h5").write_bytes(b"development-only")
            resolved = p11.validate_development_inputs(
                f3_manifest=f3_manifest,
                penobscot_manifest=pen_manifest,
                processed_root=root / "processed",
            )
            self.assertEqual(resolved["facies_f3"], f3_manifest)
            self.assertEqual(resolved["facies_penobscot"], pen_manifest)
            with self.assertRaisesRegex(ValueError, "separate locked manifests"):
                p11.validate_development_inputs(
                    f3_manifest=f3_manifest,
                    penobscot_manifest=f3_manifest,
                    processed_root=root / "processed",
                )

    def test_gate_zero_is_exact_and_correction_is_bounded(self) -> None:
        core = p11.ResidualFusionCore(num_classes=10)
        small_logits = torch.randn(2, 10, 16, 16)
        features = (
            torch.randn(2, 256, 8, 8),
            torch.randn(2, 256, 4, 4),
            torch.randn(2, 256, 2, 2),
        )
        with torch.no_grad():
            core.residual_head[-1].bias.fill_(100.0)
        fused, gate, correction = core(small_logits, features, gate_override=1.0)
        self.assertEqual(tuple(fused.shape), tuple(small_logits.shape))
        self.assertEqual(tuple(gate.shape), (2,))
        self.assertLessEqual(
            float(correction.detach().abs().max()),
            p11.MAX_RESIDUAL_CORRECTION + 1e-7,
        )
        gate_zero_logits, gate_zero, gate_zero_correction = core(
            small_logits,
            features,
            gate_override=0.0,
        )
        torch.testing.assert_close(gate_zero_logits, small_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(gate_zero, torch.zeros_like(gate_zero))
        torch.testing.assert_close(
            gate_zero_correction,
            torch.zeros_like(gate_zero_correction),
        )

    def test_frozen_main_and_encoder_stay_in_eval_during_core_training(self) -> None:
        class Small(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = torch.nn.BatchNorm2d(1)
                self.head = torch.nn.Conv2d(1, 8, kernel_size=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.head(self.norm(inputs))

        class Encoder(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
                batch = len(inputs)
                return [
                    torch.zeros(batch, 256, 8, 8),
                    torch.zeros(batch, 256, 4, 4),
                    torch.zeros(batch, 256, 2, 2),
                ]

        model = p11.ResidualFusionModel(Small(), Encoder(), num_classes=8)
        model.train()
        self.assertFalse(model.small_model.training)
        self.assertFalse(model.sam2_encoder.training)
        self.assertTrue(model.core.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.small_model.parameters())
        )

    def test_small_main_branch_is_seeded_before_construction(self) -> None:
        first = p11._build_seeded_small_model(
            "facies_f3",
            10,
            "cpu",
            seed=p11.ROOT_SEED,
        )
        second = p11._build_seeded_small_model(
            "facies_f3",
            10,
            "cpu",
            seed=p11.ROOT_SEED,
        )
        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())
        for name in first_state:
            torch.testing.assert_close(
                first_state[name],
                second_state[name],
                rtol=0.0,
                atol=0.0,
                msg=lambda message, name=name: f"{name}: {message}",
            )


class P11ResidualFusionEvidenceTests(unittest.TestCase):
    def test_portable_evidence_verifies_and_reports_honest_decisions(self) -> None:
        verified = p11.verify()
        self.assertEqual(verified["rows"], 20)
        self.assertEqual(verified["artifacts"], 5)
        self.assertFalse(verified["frozen_test_accessed"])
        summary = json.loads(
            (
                p11.OUTPUT_ROOT / "p11_residual_fusion_summary.json"
            ).read_text(encoding="utf-8")
        )
        for task in ("F3", "Penobscot"):
            task_summary = summary["tasks"][task]
            self.assertEqual(task_summary["decision"]["state"], "NON_BENEFICIAL")
            self.assertFalse(task_summary["decision"]["default_enabled"])
            self.assertEqual(
                task_summary["variant_means"]["gate_zero"]["miou"],
                task_summary["variant_means"]["strong_small_baseline"]["miou"],
            )


if __name__ == "__main__":
    unittest.main()
