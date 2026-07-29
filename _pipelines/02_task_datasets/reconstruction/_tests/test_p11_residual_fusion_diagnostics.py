from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

import p11_residual_fusion as p11  # noqa: E402
import p11_residual_fusion_diagnostics as diagnostics  # noqa: E402


def _synthetic_oof(rows_per_fold: int = 8) -> p11.OOFDevelopment:
    rng = np.random.default_rng(2693)
    folds = np.repeat(np.arange(5), rows_per_fold)
    count = len(folds)
    structural = rng.normal(size=(count, 6))
    baseline = 0.22 + 0.01 * structural[:, 0]
    residual = 0.004 * structural[:, 1] - 0.002 * structural[:, 2]
    target = baseline + residual + rng.normal(scale=0.0005, size=count)
    indices = np.column_stack(
        [
            np.arange(count),
            np.zeros(count, dtype=int),
            np.zeros(count, dtype=int),
        ]
    )
    return p11.OOFDevelopment(
        target=target,
        baseline=baseline,
        indices_kji=indices,
        xyz=rng.normal(size=(count, 3)),
        distance_to_well=np.abs(rng.normal(size=count)),
        structural_features=structural,
        fold_ids=folds,
        source_records=tuple({"fold_id": fold} for fold in range(5)),
    )


class P11DiagnosticContractTest(unittest.TestCase):
    def test_feature_variants_preserve_stage_and_channel_widths(self) -> None:
        rows = 12
        layer_channels = (32, 64, 128, 256, 320, 320)
        width = sum(layer_channels)
        mean = np.arange(rows * width, dtype=np.float64).reshape(rows, width)
        channels = np.stack([mean, mean + 1.0, mean + 2.0], axis=1)
        variants, audit = diagnostics.build_feature_variants(
            mean,
            channels,
            layer_channels=layer_channels,
            seed=2693,
        )
        self.assertEqual(tuple(variants), diagnostics.FEATURE_VARIANTS)
        self.assertEqual(variants["mean_mixed16"].shape, (rows, 96))
        self.assertEqual(variants["mean_stage0_all"].shape, (rows, 32))
        self.assertEqual(variants["mean_stage5_all"].shape, (rows, 320))
        self.assertEqual(
            variants["per_channel_mixed16_concat"].shape,
            (rows, 288),
        )
        self.assertEqual(
            variants["per_channel_stage0_all_concat"].shape,
            (rows, 96),
        )
        self.assertEqual(
            variants["per_channel_stage5_all_concat"].shape,
            (rows, 960),
        )
        self.assertEqual(audit["stage0_channel_width"], 32)
        self.assertEqual(audit["stage5_channel_width"], 320)

    def test_fold_outcome_has_explicit_win_loss_tie(self) -> None:
        self.assertEqual(diagnostics.classify_fold_outcome(0.9, 1.0), "win")
        self.assertEqual(diagnostics.classify_fold_outcome(1.1, 1.0), "loss")
        self.assertEqual(diagnostics.classify_fold_outcome(1.0, 1.0), "tie")

    def test_alpha_grid_and_gate_stay_inside_outer_training_folds(self) -> None:
        oof = _synthetic_oof()
        features = np.column_stack(
            [
                oof.structural_features,
                oof.baseline,
                oof.distance_to_well,
            ]
        )
        result = diagnostics.evaluate_adaptive_route(
            route="synthetic",
            features=features,
            oof=oof,
            seed=2693,
        )
        self.assertEqual(len(result["per_fold"]), 10)
        for cell in result["per_fold"]:
            self.assertNotIn(cell["outer_fold"], cell["train_fold_ids"])
            self.assertEqual(cell["validation_fold_ids"], [cell["outer_fold"]])
            self.assertIn(cell["residual_alpha"], diagnostics.RIDGE_ALPHAS)
            self.assertEqual(
                set(cell["train_only_alpha_candidate_rmse"]),
                {f"{alpha:.1f}" for alpha in diagnostics.RIDGE_ALPHAS},
            )
            self.assertGreaterEqual(cell["gate_stats"]["min"], 0.0)
            self.assertLessEqual(cell["gate_stats"]["max"], 1.0)
            self.assertIn(cell["outcome_vs_pykrige"], {"win", "loss", "tie"})

    def test_cli_exposes_no_test_or_holdout_argument(self) -> None:
        parser = diagnostics._parser()  # noqa: SLF001
        help_text = parser.format_help()
        run_help = parser._subparsers._group_actions[0].choices[  # noqa: SLF001
            "run"
        ].format_help()
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--test", run_help)
        self.assertNotIn("--holdout", help_text)
        self.assertNotIn("--holdout", run_help)

    def test_random_init_control_keeps_architecture_and_changes_state(self) -> None:
        import torch

        encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 4, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(4, affine=True),
            torch.nn.GELU(),
            torch.nn.Conv3d(4, 8, kernel_size=3, padding=1),
        )
        architecture_before = diagnostics._encoder_architecture_sha256(  # noqa: SLF001
            encoder
        )
        audit = diagnostics._reset_encoder_same_architecture(  # noqa: SLF001
            encoder,
            seed=2693,
            torch=torch,
        )
        self.assertEqual(audit["architecture_sha256"], architecture_before)
        self.assertTrue(audit["randomized_state_differs_from_pretrained"])
        self.assertTrue(audit["encoder_frozen_after_reset"])
        self.assertNotEqual(
            audit["pretrained_state_sha256_before_reset"],
            audit["random_init_state_sha256"],
        )

    def test_block_bootstrap_resamples_spatial_folds_not_seed_rows(self) -> None:
        oof = _synthetic_oof()
        candidates = {
            seed: oof.baseline + 0.5 * (oof.target - oof.baseline)
            for seed in p11.REPEAT_SEEDS
        }
        result = diagnostics.block_bootstrap_rmse_delta(
            target=oof.target,
            baseline=oof.baseline,
            candidate_predictions_by_seed=candidates,
            fold_ids=oof.fold_ids,
            bootstrap_seed=2693,
            replicates=250,
        )
        self.assertEqual(result["bootstrap_unit"], "locked outer spatial fold")
        self.assertEqual(result["independent_spatial_block_count"], 5)
        self.assertIn("never resampled", result["seed_handling"])
        self.assertLess(result["point_estimate"], 0.0)
        self.assertEqual(len(result["confidence_interval"]), 2)


class P11DiagnosticPortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p11_residual_fusion_diagnostics"
        summary_path = cls.output_dir / "summary.json"
        if not summary_path.is_file():
            raise unittest.SkipTest("real P11 diagnostic evidence not generated")
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def test_evidence_has_complete_controls_and_fold_matrix(self) -> None:
        self.assertEqual(
            self.summary["schema_version"],
            diagnostics.SCHEMA_VERSION,
        )
        experiment = self.summary["experiment"]
        self.assertTrue(
            experiment["baseline"]["gate_zero_bitwise_equal_to_pykrige"]
        )
        self.assertEqual(
            tuple(experiment["variants"]),
            diagnostics.FEATURE_VARIANTS,
        )
        for variant in diagnostics.FEATURE_VARIANTS:
            for mode in ("fixed_ridge10", "train_only_alpha_grid"):
                head = experiment["variants"][variant]["heads"][mode]
                pretrained = head["pretrained_residual"]
                random_init = head["random_init_same_architecture_control"]
                self.assertEqual(pretrained["independent_spatial_units"], 5)
                self.assertEqual(pretrained["seed_pseudo_repeats_per_unit"], 3)
                self.assertEqual(
                    pretrained["seed_level_pseudo_repeat_cells"],
                    15,
                )
                self.assertEqual(random_init["independent_spatial_units"], 5)
                self.assertEqual(
                    sum(pretrained["independent_fold_outcome_counts"].values()),
                    5,
                )
                bootstrap = pretrained[
                    "block_bootstrap_rmse_delta_vs_pykrige"
                ]
                self.assertEqual(
                    bootstrap["bootstrap_unit"],
                    "locked outer spatial fold",
                )
                self.assertEqual(
                    bootstrap["independent_spatial_block_count"],
                    5,
                )
                self.assertTrue(
                    pretrained["promotion"][
                        "random_init_same_architecture_control_present"
                    ]
                )
        self.assertEqual(
            len(experiment["per_fold_seed_pseudo_repeats"]),
            (1 + 3 * len(diagnostics.FEATURE_VARIANTS)) * 2 * 15,
        )
        self.assertTrue((self.output_dir / "evidence.md").is_file())
        evidence = (self.output_dir / "evidence.md").read_text(encoding="utf-8")
        self.assertIn("five genuinely independent spatial units", evidence)
        self.assertIn("paired pseudo-repeats", evidence)
        self.assertNotIn("3/15", evidence)
        self.assertNotIn("6/15", evidence)
        prediction_path = self.output_dir / "prediction_errors.npz"
        self.assertTrue(prediction_path.is_file())
        self.assertEqual(
            self.summary["prediction_error_artifact"]["sha256"],
            p11._sha256(prediction_path),  # noqa: SLF001
        )
        with np.load(prediction_path, allow_pickle=False) as payload:
            self.assertEqual(len(payload.files), 39)
            self.assertEqual(payload["baseline_error"].shape, (10_240,))
            self.assertEqual(payload["fold_ids"].shape, (10_240,))
            self.assertEqual(
                set(np.unique(payload["fold_ids"]).tolist()),
                set(p11.FOLD_IDS),
            )
            error_keys = [
                key for key in payload.files if key.startswith("pretrained__")
            ]
            self.assertEqual(len(error_keys), 36)
            self.assertTrue(
                all(payload[key].shape == (10_240,) for key in error_keys)
            )

    def test_evidence_preserves_genuine_checkpoint_and_firewall(self) -> None:
        self.assertEqual(
            self.summary["openmind_feature_audit"]["checkpoint_sha256"],
            p11.EXPECTED_CHECKPOINT_SHA256,
        )
        self.assertTrue(
            self.summary["openmind_feature_audit"][
                "real_pretrained_weights_loaded"
            ]
        )
        self.assertEqual(
            self.summary["openmind_feature_audit"][
                "seismic_channels_forwarded_separately"
            ],
            3,
        )
        self.assertEqual(
            self.summary["implementation"]["script_sha256"],
            p11._sha256(  # noqa: SLF001
                HERE / "p11_residual_fusion_diagnostics.py"
            ),
        )
        firewall = self.summary["holdout_firewall"]
        self.assertEqual(firewall["hdf5_files_opened"], ["train.h5"])
        self.assertFalse(firewall["label_dataset_read"])
        self.assertFalse(firewall["test_h5_opened"])
        self.assertFalse(firewall["frozen_holdout_opened"])
        random_init = self.summary["random_init_control_audit"]
        self.assertEqual(
            random_init["control_id"],
            "random_init_same_architecture",
        )
        self.assertTrue(random_init["same_architecture_as_pretrained"])
        self.assertFalse(random_init["pretrained_weights_used_for_forward"])
        self.assertEqual(len(random_init["random_init_state_audit"]), 3)
        self.assertEqual(
            len(
                {
                    row["architecture_sha256"]
                    for row in random_init["random_init_state_audit"]
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    row["random_init_state_sha256"]
                    for row in random_init["random_init_state_audit"]
                }
            ),
            3,
        )

    def test_low_win_rate_emits_required_escalation_conclusion(self) -> None:
        decision = self.summary["experiment"]["decision"]
        if (
            decision["highest_independent_fold_wins"]
            < decision["required_independent_fold_wins"]
        ):
            self.assertTrue(
                decision[
                    "adaptation_space_exhausted_under_current_checkpoint"
                ]
            )
            self.assertEqual(
                decision["conclusion"],
                "现有OpenMind checkpoint的适配空间已基本穷尽，"
                "建议更换更贴近地震/地质领域的基础模型。",
            )
            self.assertIn("负责人/军伟", decision["model_replacement_authority"])


if __name__ == "__main__":
    unittest.main()
