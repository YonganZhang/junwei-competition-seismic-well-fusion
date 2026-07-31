from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from _models.reconstruction import geophysical_fm as adapter  # noqa: E402
import p11_residual_fusion as p11  # noqa: E402
import p14_geophysical_fm as p14  # noqa: E402


def _synthetic_oof(rows_per_fold: int = 8) -> p11.OOFDevelopment:
    rng = np.random.default_rng(2693)
    fold_ids = np.repeat(np.arange(5), rows_per_fold)
    rows = len(fold_ids)
    structural = rng.normal(size=(rows, 6))
    baseline = 0.24 + 0.01 * structural[:, 0]
    residual = 0.002 * structural[:, 1]
    return p11.OOFDevelopment(
        target=baseline + residual,
        baseline=baseline,
        indices_kji=np.column_stack(
            [
                np.arange(rows),
                np.zeros(rows, dtype=np.int64),
                np.zeros(rows, dtype=np.int64),
            ]
        ),
        xyz=rng.normal(size=(rows, 3)),
        distance_to_well=np.abs(rng.normal(size=rows)),
        structural_features=structural,
        fold_ids=fold_ids,
        source_records=tuple({"fold_id": fold} for fold in range(5)),
    )


class P14GeophysicalFMContractTest(unittest.TestCase):
    def test_adapter_declares_domain_model_and_random_init_support(self) -> None:
        capabilities = adapter.capabilities()
        self.assertEqual(adapter.model_id, "geophysical_fm")
        self.assertEqual(
            capabilities["foundation_model"],
            "thinkonward/geophysical-foundation-model",
        )
        self.assertEqual(capabilities["license"], "Apache-2.0")
        self.assertEqual(capabilities["input_shape"], "[B,1,400,160]")
        self.assertEqual(capabilities["output_shape"], "[B,161,1200]")
        self.assertTrue(
            capabilities["supports_same_architecture_random_init"]
        )
        self.assertFalse(capabilities["auto_download"])

    def test_adapter_wrapper_uses_all_trace_tokens_without_masking(self) -> None:
        import torch

        class FakeNetwork(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

            def forward_encoder(self, images, indices, len_keep):
                batch = len(images)
                self.last_indices = indices
                self.last_len_keep = len_keep
                latent = torch.zeros(
                    (batch, 161, 1200),
                    dtype=images.dtype,
                    device=images.device,
                )
                mask = torch.zeros(
                    (batch, 160),
                    dtype=images.dtype,
                    device=images.device,
                )
                restore = torch.arange(
                    160,
                    device=images.device,
                ).unsqueeze(0).expand(batch, -1)
                return latent, mask, restore

        network = FakeNetwork()
        model = adapter._make_wrapper(  # noqa: SLF001
            torch,
            network,
            freeze_encoder=True,
            weight_mode="pretrained",
            asset_audit={},
        )
        output = model(torch.zeros((2, 1, 400, 160)))
        self.assertEqual(tuple(output.shape), (2, 161, 1200))
        self.assertEqual(network.last_len_keep, 160)
        self.assertTrue(
            torch.equal(
                network.last_indices[0],
                torch.arange(160, dtype=torch.float32),
            )
        )
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_trace_mapping_uses_center_aligned_resized_tokens(self) -> None:
        mapped = p14.trace_token_indices(
            np.asarray([0, 50, 99]),
            original_trace_count=100,
        )
        np.testing.assert_array_equal(mapped, np.asarray([0, 80, 159]))
        self.assertTrue(np.all(np.diff(mapped) > 0))

    def test_slice_normalization_uses_active_cells_only(self) -> None:
        values = np.asarray([[1.0, 2.0], [1000.0, 3.0]])
        active = np.asarray([[True, True], [False, True]])
        normalized, audit = p14.normalize_slice(values, active)
        self.assertEqual(normalized[1, 0], 0.0)
        self.assertAlmostEqual(float(np.mean(normalized[active])), 0.0)
        self.assertAlmostEqual(float(np.std(normalized[active])), 1.0, places=6)
        self.assertEqual(audit["active_mean_before"], 2.0)

    def test_seeded_feature_budget_is_six_views_by_sixteen(self) -> None:
        first = p14.selected_embedding_channels(
            embedding_width=1200,
            seed=2693,
        )
        second = p14.selected_embedding_channels(
            embedding_width=1200,
            seed=2693,
        )
        other = p14.selected_embedding_channels(
            embedding_width=1200,
            seed=2694,
        )
        self.assertEqual(first.shape, (6, 16))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(all(len(np.unique(row)) == 16 for row in first))

    def test_reused_p11_harness_keeps_gate_zero_and_controls(self) -> None:
        oof = _synthetic_oof()
        rng = np.random.default_rng(2693)
        shape = (
            len(p11.REPEAT_SEEDS),
            len(oof.target),
            len(p14.VIEW_NAMES) * p14.EMBEDDING_CHANNELS_PER_VIEW,
        )
        pretrained = rng.normal(size=shape)
        random_init = rng.normal(size=shape)
        experiment, payload = p14.evaluate_gfm(
            oof=oof,
            pretrained_features=pretrained,
            random_init_features=random_init,
        )
        self.assertTrue(
            experiment["baseline"]["gate_zero_bitwise_equal_to_pykrige"]
        )
        self.assertEqual(tuple(experiment["heads"]), p14.HEAD_MODES)
        for mode in p14.HEAD_MODES:
            head = experiment["heads"][mode]
            self.assertIn("pretrained_geophysical_fm", head)
            self.assertIn("random_init_same_architecture", head)
            self.assertIn("no_foundation_structural", head)
            self.assertTrue(
                head["promotion"][
                    "random_init_same_architecture_control_present"
                ]
            )
            self.assertEqual(
                head["pretrained_geophysical_fm"][
                    "independent_spatial_units"
                ],
                5,
            )
        self.assertEqual(payload["fold_ids"].shape, (len(oof.target),))

    def test_cli_exposes_no_test_or_holdout_argument(self) -> None:
        parser = p14._parser()  # noqa: SLF001
        run_parser = parser._subparsers._group_actions[0].choices[  # noqa: SLF001
            "run"
        ]
        help_text = parser.format_help() + run_parser.format_help()
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)


class P14GeophysicalFMPortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p14_geophysical_fm"
        summary_path = cls.output_dir / "summary.json"
        if not summary_path.is_file():
            raise unittest.SkipTest("real P14 GFM evidence not generated")
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def test_real_evidence_has_genuine_pretrained_and_random_init(self) -> None:
        self.assertEqual(
            self.summary["schema_version"],
            p14.SCHEMA_VERSION,
        )
        self.assertEqual(
            self.summary["model"]["id"],
            "thinkonward/geophysical-foundation-model",
        )
        self.assertEqual(self.summary["model"]["license"], "Apache-2.0")
        pretrained = self.summary["pretrained_feature_audit"]
        random_init = self.summary["random_init_feature_audit"]
        self.assertTrue(pretrained["pretrained_weights_used_for_forward"])
        self.assertFalse(random_init["pretrained_weights_used_for_forward"])
        self.assertTrue(random_init["seed_distinct_random_states"])
        self.assertEqual(
            pretrained["architecture_sha256"],
            random_init["architecture_sha256"],
        )

    def test_real_evidence_preserves_p11_protocol_and_firewall(self) -> None:
        experiment = self.summary["experiment"]
        self.assertTrue(
            experiment["baseline"]["gate_zero_bitwise_equal_to_pykrige"]
        )
        self.assertEqual(
            experiment["fixed_protocol"]["independent_spatial_units"],
            5,
        )
        self.assertTrue(
            experiment["fixed_protocol"]["seeds_are_paired_pseudo_repeats"]
        )
        self.assertFalse(
            experiment["decision"]["pretrained_contribution_claimed"]
        )
        firewall = self.summary["holdout_firewall"]
        self.assertEqual(firewall["hdf5_files_opened"], ["train.h5"])
        for key in (
            "label_dataset_read_by_encoder",
            "test_path_argument_exists",
            "test_h5_opened",
            "frozen_holdout_opened",
            "historical_test_metrics_read",
        ):
            self.assertFalse(firewall[key])
        evidence = (self.output_dir / "evidence.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("[K,J,I]=[9,20,18]", evidence)
        self.assertIn("KxJ vertical section at fixed I", evidence)
        self.assertIn(
            "five genuinely independent spatial units",
            evidence.lower(),
        )
        self.assertIn("not 15 independent observations", evidence)

    def test_real_artifacts_are_hash_locked_and_row_aligned(self) -> None:
        summary = self.summary
        self.assertEqual(
            summary["implementation"]["script_sha256"],
            p11._sha256(HERE / "p14_geophysical_fm.py"),  # noqa: SLF001
        )
        self.assertEqual(
            summary["implementation"]["adapter_sha256"],
            p11._sha256(  # noqa: SLF001
                PROJECT_ROOT
                / "_models"
                / "reconstruction"
                / "geophysical_fm.py"
            ),
        )
        artifact = summary["prediction_error_artifact"]
        path = PROJECT_ROOT / artifact["path"]
        self.assertEqual(artifact["sha256"], p11._sha256(path))  # noqa: SLF001
        with np.load(path, allow_pickle=False) as payload:
            self.assertEqual(payload["fold_ids"].shape, (10_240,))
            self.assertEqual(payload["indices_kji"].shape, (10_240, 3))


if __name__ == "__main__":
    unittest.main()
