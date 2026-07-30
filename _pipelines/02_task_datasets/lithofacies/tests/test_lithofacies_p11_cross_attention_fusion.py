from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock
import unittest

import numpy as np
import torch


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

fusion = importlib.import_module(
    "_pipelines.02_task_datasets.lithofacies."
    "lithofacies_p11_cross_attention_fusion"
)
p11 = fusion.p11


class P11CrossAttentionFusionContractTest(unittest.TestCase):
    def test_cross_attention_shapes_and_exact_zero_gate(self) -> None:
        torch.manual_seed(2693)
        model = fusion.XGBMomentCrossAttention()
        moment = torch.randn(3, 52, 768)
        leaves = torch.randint(0, 7, (3, 40, 9))
        baseline = torch.randn(3, 9)
        fused, gate, residual, contribution = model(
            moment,
            leaves,
            baseline,
        )
        self.assertEqual(tuple(fused.shape), (3, 9))
        self.assertEqual(tuple(gate.shape), (9,))
        self.assertEqual(tuple(residual.shape), (3, 9))
        self.assertEqual(tuple(contribution.shape), (3, 9))
        gate0, zero_gate, _, zero_contribution = model(
            moment,
            leaves,
            baseline,
            force_gate_zero=True,
        )
        self.assertTrue(torch.equal(gate0, baseline))
        self.assertTrue(torch.equal(zero_gate, torch.zeros(9)))
        self.assertTrue(
            torch.equal(zero_contribution, torch.zeros_like(baseline))
        )
        self.assertLessEqual(
            float(residual.detach().abs().max()),
            fusion.MAX_RESIDUAL_LOGIT,
        )

    def test_encoder_initialization_is_parameterized(self) -> None:
        sentinel = object()
        native_module = importlib.import_module(
            "lithofacies_p11_clean_well_native33"
        )
        with mock.patch.object(
            native_module,
            "_build_native_moment",
            return_value=sentinel,
        ) as builder:
            result = fusion._build_encoder(
                snapshot=Path("/safe/snapshot"),
                device="cpu",
                seed=17,
                encoder_init="pretrained",
            )
            self.assertIs(result, sentinel)
            self.assertFalse(builder.call_args.kwargs["random_init"])
            fusion._build_encoder(
                snapshot=Path("/safe/snapshot"),
                device="cpu",
                seed=18,
                encoder_init="random",
            )
            self.assertTrue(builder.call_args.kwargs["random_init"])
        with self.assertRaisesRegex(ValueError, "encoder_init"):
            fusion._build_encoder(
                snapshot=Path("/safe/snapshot"),
                device="cpu",
                seed=19,
                encoder_init="unknown",
            )

    def test_prior_calibration_uses_only_supplied_train_counts(self) -> None:
        logits = np.zeros((2, 9), dtype=np.float32)
        counts = np.arange(1, 10, dtype=np.int64)
        calibrated, bias = fusion._prior_calibrate(logits, counts)
        expected = fusion.PRIOR_SHRINKAGE * (
            np.log(counts) - np.log(counts).mean()
        )
        self.assertTrue(np.allclose(bias, expected))
        self.assertTrue(np.allclose(calibrated[0], bias))
        self.assertTrue(np.array_equal(calibrated[0], calibrated[1]))
        self.assertAlmostEqual(float(bias.mean()), 0.0, places=7)

    def test_xgboost_feature_order_matches_archived_contract(self) -> None:
        well = np.arange(2 * 26 * 33, dtype=np.float32).reshape(2, 26, 33)
        seismic = (
            np.arange(2 * 3 * 3 * 33, dtype=np.float32).reshape(2, 3, 3, 33)
            + 100000
        )
        features = fusion._xgboost_features(well, seismic)
        expected = np.concatenate(
            (well.reshape(2, -1), seismic.reshape(2, -1)),
            axis=1,
        )
        self.assertEqual(features.shape, (2, 1155))
        self.assertTrue(np.array_equal(features, expected))

    def test_strict_summary_reports_component_deltas(self) -> None:
        rows = []
        values = {
            "baseline": 0.20,
            "prior_calibrated": 0.206,
            "cross_attention": 0.209,
        }
        for fold_id in p11.FOLD_IDS:
            for repeat_id in range(len(p11.REPEAT_SEEDS)):
                for variant in fusion.VARIANTS:
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "variant": variant,
                            "metrics": {
                                metric: values[variant]
                                for metric in p11.PRIMARY_METRICS
                            },
                            "training": {
                                "gate_mean": (
                                    0.02
                                    if variant == "cross_attention"
                                    else None
                                ),
                                "residual_contribution_mean_abs": (
                                    0.01
                                    if variant == "cross_attention"
                                    else None
                                ),
                            },
                        }
                    )
        summary = fusion.summarize_results(rows)
        self.assertEqual(summary["evaluation"]["completed_cells"], 36)
        self.assertAlmostEqual(
            summary["comparison"]["cross_attention_minus_baseline"],
            0.009,
        )
        self.assertAlmostEqual(
            summary["comparison"][
                "cross_attention_minus_prior_calibrated"
            ],
            0.003,
        )
        self.assertEqual(
            summary["decision"]["large_model_contribution_share"],
            "pending_next_pretrained_vs_random_encoder_ablation",
        )
        with self.assertRaisesRegex(ValueError, "complete strict"):
            fusion.summarize_results(rows[:-1])

    def test_holdout_like_paths_remain_forbidden(self) -> None:
        for path in (
            Path("/does/not/exist/test_bundle.npz"),
            Path("/does/not/exist/frozen-holdout"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "forbidden holdout path"):
                    p11.ensure_development_only_paths([path])


if __name__ == "__main__":
    unittest.main()
