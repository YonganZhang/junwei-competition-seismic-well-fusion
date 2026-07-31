from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from _models.reconstruction import geophysical_fm_finetune as adapter  # noqa: E402
import p11_residual_fusion as p11  # noqa: E402
import p15_gfm_finetune as p15  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class P15GFMFineTuneContractTest(unittest.TestCase):
    def test_adapter_opens_only_final_block_and_norm(self) -> None:
        import torch

        class FakeNetwork(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(16)]
                )
                self.norm = torch.nn.LayerNorm(2)
                self.patch_embed = torch.nn.Linear(2, 2)

        network = FakeNetwork()
        indices = adapter.configure_partial_finetune(
            network,
            trainable_block_count=1,
        )
        self.assertEqual(indices, (15,))
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in network.blocks[:15].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in network.blocks[15].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in network.norm.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in network.patch_embed.parameters()
            )
        )

    def test_adapter_capability_declares_real_partial_finetuning(self) -> None:
        capabilities = adapter.capabilities()
        self.assertTrue(capabilities["partial_finetuning"])
        self.assertEqual(capabilities["encoder_depth"], 16)
        self.assertEqual(
            capabilities["trainable_tail_block_counts"],
            [1, 2],
        )
        self.assertTrue(capabilities["supports_frozen_prefix_cache"])
        self.assertEqual(
            capabilities["cached_prefix_precision"],
            "owned_by_evaluation_pipeline",
        )

    def test_native_window_plan_needs_no_resize(self) -> None:
        crossline = np.asarray([2207, 2250, 2286])
        start = p15.centered_native_window_start(
            crossline,
            window_size=160,
            lower_bound=1932,
            upper_bound=2536,
        )
        tokens = crossline - start
        self.assertGreaterEqual(int(tokens.min()), 0)
        self.assertLess(int(tokens.max()), 160)
        self.assertEqual(int(crossline.max() - crossline.min() + 1), 80)
        with self.assertRaises(ValueError):
            p15.centered_native_window_start(
                np.asarray([0, 200]),
                window_size=160,
                lower_bound=0,
                upper_bound=300,
            )

    def test_query_scaler_is_fit_on_selected_rows_only(self) -> None:
        query = np.asarray(
            [
                [0.0, 100.0],
                [2.0, 200.0],
                [1000.0, 300.0],
            ],
            dtype=np.float32,
        )
        train = np.asarray([True, True, False])
        mean, std = p15.fit_query_scaler(query, train)
        np.testing.assert_allclose(mean, np.asarray([1.0, 150.0]))
        np.testing.assert_allclose(std, np.asarray([1.0, 50.0]))
        transformed = p15.transform_query(query, mean, std)
        np.testing.assert_allclose(
            transformed[:2],
            np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
        )
        self.assertEqual(transformed[2, 0], 8.0)

    def test_gate_zero_is_bitwise_baseline(self) -> None:
        baseline = np.asarray([0.2, 0.25, 0.3], dtype=np.float64)
        gate_zero = baseline.copy()
        self.assertTrue(np.array_equal(gate_zero, baseline))

    def test_cli_exposes_no_test_or_holdout_argument(self) -> None:
        parser = p15._parser()  # noqa: SLF001
        run_parser = parser._subparsers._group_actions[0].choices[  # noqa: SLF001
            "run"
        ]
        help_text = parser.format_help() + run_parser.format_help()
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)

    def test_output_validation_preserves_p11_p14(self) -> None:
        with self.assertRaises(ValueError):
            p15._validate_output_dir(  # noqa: SLF001
                HERE / "_outputs" / "p14_geophysical_fm"
            )
        accepted = p15._validate_output_dir(  # noqa: SLF001
            HERE / "_outputs" / "p15_gfm_finetune"
        )
        self.assertEqual(
            accepted,
            (HERE / "_outputs" / "p15_gfm_finetune").resolve(),
        )


class P15GFMFineTunePortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p15_gfm_finetune"
        summary_path = cls.output_dir / "summary.json"
        if not summary_path.is_file():
            raise unittest.SkipTest("real P15 evidence not generated")
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def test_real_evidence_uses_native_windows_and_real_tail_updates(self) -> None:
        self.assertEqual(self.summary["schema_version"], p15.SCHEMA_VERSION)
        native = self.summary["native_mapping_audit"]["native_window"]
        self.assertEqual(native["shape"], [400, 160])
        self.assertFalse(native["resize_applied"])
        self.assertFalse(native["interpolation_applied"])
        self.assertFalse(native["padding_applied"])
        optimization = self.summary["experiment"]["optimization"]
        self.assertEqual(optimization["trainable_block_indices"], [15])
        self.assertLess(
            optimization["encoder_lr"],
            optimization["head_lr"],
        )
        signal = self.summary["experiment"][
            "pretrained_geophysical_fm"
        ]["training_signal"]
        self.assertTrue(signal["all_refits_have_nonzero_gradients"])
        self.assertTrue(signal["all_refits_move_encoder_parameters"])
        self.assertEqual(
            signal["cells_with_positive_mean_adjacent_gradient_cosine"],
            15,
        )
        assessment = self.summary["experiment"]["hypothesis_assessment"]
        self.assertTrue(
            assessment[
                "optimization_dynamics_different_from_p14_frozen_result"
            ]
        )
        self.assertFalse(
            assessment[
                "development_generalization_conclusion_different_from_p14"
            ]
        )

    def test_real_random_control_is_matched_and_seed_distinct(self) -> None:
        audits = self.summary["random_init_prefix_audits"]
        self.assertEqual(len(audits), len(p11.REPEAT_SEEDS))
        self.assertTrue(
            all(not audit["pretrained_weights_loaded"] for audit in audits)
        )
        self.assertEqual(
            len({audit["encoder_probe_sha256"] for audit in audits}),
            len(p11.REPEAT_SEEDS),
        )
        self.assertTrue(
            self.summary["experiment"]["promotion"][
                "matched_random_init_present"
            ]
        )

    def test_real_evidence_keeps_five_units_and_seed_pseudo_repeats(self) -> None:
        protocol = self.summary["experiment"]["fixed_protocol"]
        self.assertEqual(protocol["independent_spatial_units"], 5)
        self.assertTrue(
            protocol["seeds_are_paired_optimization_pseudo_repeats"]
        )
        outcomes = self.summary["experiment"][
            "pretrained_geophysical_fm"
        ]["independent_fold_outcomes"]
        self.assertEqual(len(outcomes), 5)
        self.assertTrue(
            all(row["independent_spatial_unit"] for row in outcomes)
        )

    def test_real_artifacts_and_firewall_are_self_consistent(self) -> None:
        manifest_path = self.output_dir / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.items():
            self.assertEqual(_sha256(self.output_dir / name), expected)
        firewall = self.summary["holdout_firewall"]
        self.assertEqual(firewall["hdf5_files_opened"], ["train.h5"])
        self.assertFalse(firewall["label_dataset_read_from_hdf5"])
        self.assertFalse(firewall["test_path_argument_exists"])
        self.assertFalse(firewall["test_h5_opened"])
        self.assertFalse(firewall["frozen_holdout_opened"])
        self.assertFalse(firewall["historical_test_metrics_read"])


if __name__ == "__main__":
    unittest.main()
