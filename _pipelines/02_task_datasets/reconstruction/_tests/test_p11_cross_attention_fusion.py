from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import p11_cross_attention_fusion as fusion  # noqa: E402
import p11_residual_fusion as p11  # noqa: E402


def _synthetic_oof(rows_per_fold: int = 5) -> p11.OOFDevelopment:
    rng = np.random.default_rng(2693)
    fold_ids = np.repeat(np.arange(5), rows_per_fold)
    rows = len(fold_ids)
    structural = rng.normal(size=(rows, 6))
    baseline = 0.24 + 0.01 * structural[:, 0]
    target = baseline + 0.002 * structural[:, 1]
    return p11.OOFDevelopment(
        target=target,
        baseline=baseline,
        indices_kji=np.column_stack(
            [
                np.arange(rows),
                np.zeros(rows, dtype=int),
                np.zeros(rows, dtype=int),
            ]
        ),
        xyz=rng.normal(size=(rows, 3)),
        distance_to_well=np.abs(rng.normal(size=rows)),
        structural_features=structural,
        fold_ids=fold_ids,
        source_records=tuple({"fold_id": fold} for fold in range(5)),
    )


class P11CrossAttentionContractTest(unittest.TestCase):
    def test_balanced_stage_selection_preserves_three_channel_tokens(self) -> None:
        widths = (32, 64, 128, 256, 320, 320)
        features = np.arange(7 * 3 * sum(widths), dtype=np.float64).reshape(
            7,
            3,
            sum(widths),
        )
        tokens, selected = fusion.select_channel_tokens(
            features,
            layer_channels=widths,
            seed=2693,
        )
        self.assertEqual(tokens.shape, (7, 3, 96))
        self.assertEqual(selected.shape, (96,))
        np.testing.assert_array_equal(tokens, features[:, :, selected])

    def test_network_uses_real_cross_attention_over_three_tokens(self) -> None:
        import torch

        config = fusion.CrossAttentionConfig(
            pca_components=4,
            embedding_dim=8,
            attention_heads=2,
            hidden_dim=8,
            dropout=0.0,
            epochs=1,
        )
        model = fusion._make_network(  # noqa: SLF001
            query_width=8,
            token_width=4,
            config=config,
            torch=torch,
        )
        self.assertIsInstance(
            model.cross_attention,
            torch.nn.MultiheadAttention,
        )
        prediction, weights = model(
            torch.zeros((6, 8)),
            torch.zeros((6, 3, 4)),
            return_attention=True,
        )
        self.assertEqual(tuple(prediction.shape), (6,))
        self.assertEqual(tuple(weights.shape), (6, 3))
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.ones(6),
        )

    def test_pca_scalers_and_attention_fit_training_rows_only(self) -> None:
        rng = np.random.default_rng(2693)
        train_queries = rng.normal(size=(24, 8))
        validation_queries = rng.normal(size=(6, 8))
        train_tokens = rng.normal(size=(24, 3, 12))
        validation_tokens = rng.normal(size=(6, 3, 12))
        train_residual = rng.normal(scale=0.01, size=24)
        config = fusion.CrossAttentionConfig(
            pca_components=4,
            embedding_dim=8,
            attention_heads=2,
            hidden_dim=8,
            dropout=0.0,
            epochs=2,
            batch_size=12,
        )
        prediction, audit = fusion._fit_cross_attention(  # noqa: SLF001
            train_queries,
            train_tokens,
            train_residual,
            validation_queries,
            validation_tokens,
            seed=2693,
            config=config,
            device="cpu",
        )
        self.assertEqual(prediction.shape, (6,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(audit["train_rows"], 24)
        self.assertEqual(audit["validation_rows"], 6)
        self.assertEqual(audit["pca_components"], 4)
        self.assertTrue(
            audit["all_preprocessing_fit_on_training_rows_only"]
        )
        self.assertEqual(len(audit["mean_attention_by_channel"]), 3)

    def test_encoder_weight_mode_switches_without_mixing_sources(self) -> None:
        oof = _synthetic_oof()
        inputs = p11.DevInputPaths(
            data_dir=Path("/legal"),
            train_h5=Path("/legal/train.h5"),
        )
        pretrained = np.zeros((len(oof.target), 3, 12), dtype=np.float32)
        random = np.ones((3, len(oof.target), 3, 12), dtype=np.float32)
        common = {
            "inputs": inputs,
            "oof": oof,
            "source_root": Path("/source"),
            "checkpoint": Path("/checkpoint"),
            "dependency_root": None,
            "channel_feature_cache": Path("/pretrained-cache"),
            "random_init_feature_cache": Path("/random-cache"),
            "device": "cpu",
        }
        audit = {"layer_channels": [2, 2, 2, 2, 2, 2]}
        with mock.patch.object(
            fusion.diagnostics,
            "get_openmind_per_channel_features",
            return_value=(pretrained, audit),
        ) as pretrained_loader, mock.patch.object(
            fusion.diagnostics,
            "get_openmind_random_init_features",
            return_value=(
                np.zeros((3, len(oof.target), 12)),
                random,
                audit,
            ),
        ) as random_loader:
            by_seed, resolved = fusion.resolve_encoder_features(
                encoder_weight_mode="pretrained",
                **common,
            )
            pretrained_loader.assert_called_once()
            random_loader.assert_not_called()
            self.assertEqual(resolved["encoder_weight_mode"], "pretrained")
            self.assertFalse(
                resolved["matched_random_init_executed_this_run"]
            )
            self.assertTrue(
                all(
                    np.array_equal(values, pretrained)
                    for values in by_seed.values()
                )
            )

        with mock.patch.object(
            fusion.diagnostics,
            "get_openmind_per_channel_features",
        ) as pretrained_loader, mock.patch.object(
            fusion.diagnostics,
            "get_openmind_random_init_features",
            return_value=(
                np.zeros((3, len(oof.target), 12)),
                random,
                audit,
            ),
        ) as random_loader:
            by_seed, resolved = fusion.resolve_encoder_features(
                encoder_weight_mode="random_init",
                **common,
            )
            pretrained_loader.assert_not_called()
            random_loader.assert_called_once()
            self.assertEqual(resolved["encoder_weight_mode"], "random_init")
            self.assertTrue(
                resolved["matched_random_init_executed_this_run"]
            )
            for seed_id, seed in enumerate(p11.REPEAT_SEEDS):
                np.testing.assert_array_equal(by_seed[seed], random[seed_id])

    def test_cli_has_weight_switch_and_no_holdout_surface(self) -> None:
        parser = fusion._parser()  # noqa: SLF001
        run_parser = parser._subparsers._group_actions[0].choices[  # noqa: SLF001
            "run"
        ]
        help_text = parser.format_help() + run_parser.format_help()
        self.assertIn("--encoder-weight-mode", help_text)
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)


class P11CrossAttentionPortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p11_cross_attention_fusion"
        summary_path = cls.output_dir / "summary.json"
        if not summary_path.is_file():
            raise unittest.SkipTest(
                "real P11 cross-attention evidence not generated"
            )
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def test_evidence_preserves_firewall_and_pretrained_only_scope(self) -> None:
        self.assertEqual(
            self.summary["schema_version"],
            fusion.SCHEMA_VERSION,
        )
        self.assertEqual(
            self.summary["protocol"]["encoder_weight_mode"],
            "pretrained",
        )
        self.assertFalse(
            self.summary["protocol"][
                "random_init_ablation_executed_this_run"
            ]
        )
        firewall = self.summary["holdout_firewall"]
        self.assertEqual(firewall["hdf5_files_opened"], ["train.h5"])
        self.assertFalse(firewall["test_path_argument_exists"])
        self.assertFalse(firewall["test_h5_opened"])
        self.assertFalse(firewall["frozen_holdout_opened"])
        self.assertFalse(firewall["label_dataset_read_by_encoder"])
        self.assertEqual(
            self.summary["implementation"]["script_sha256"],
            p11._sha256(HERE / "p11_cross_attention_fusion.py"),  # noqa: SLF001
        )

    def test_five_spatial_units_and_contribution_boundary_are_explicit(self) -> None:
        experiment = self.summary["experiment"]
        self.assertEqual(
            len(experiment["independent_spatial_folds"]),
            5,
        )
        self.assertEqual(
            sum(experiment["independent_fold_outcome_counts"].values()),
            5,
        )
        self.assertEqual(
            len(experiment["per_fold_seed_pseudo_repeats"]),
            15,
        )
        statement = experiment["comparison"][
            "large_model_contribution_statement"
        ]
        self.assertEqual(statement, "大模型贡献占比待下一轮消融确认")
        evidence = (self.output_dir / "evidence.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("five genuinely independent spatial units", evidence)
        self.assertIn("paired optimization pseudo-repeats", evidence)
        self.assertIn("大模型贡献占比待下一轮消融确认", evidence)

    def test_prediction_errors_are_hash_locked_and_row_aligned(self) -> None:
        artifact = self.summary["prediction_error_artifact"]
        path = PROJECT_ROOT / artifact["path"]
        self.assertEqual(artifact["sha256"], p11._sha256(path))  # noqa: SLF001
        with np.load(path, allow_pickle=False) as payload:
            self.assertEqual(payload["fold_ids"].shape, (10_240,))
            self.assertEqual(payload["indices_kji"].shape, (10_240, 3))
            self.assertEqual(
                set(np.unique(payload["fold_ids"]).tolist()),
                set(p11.FOLD_IDS),
            )
            for seed in p11.REPEAT_SEEDS:
                self.assertEqual(
                    payload[f"gated_prediction__seed_{seed}"].shape,
                    (10_240,),
                )


if __name__ == "__main__":
    unittest.main()
