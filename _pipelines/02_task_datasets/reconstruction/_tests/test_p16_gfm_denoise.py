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

from _models.reconstruction import geophysical_fm_denoise as adapter  # noqa: E402
import p11_residual_fusion as p11  # noqa: E402
import p16_gfm_denoise as p16  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_oof(rows_per_fold: int = 8) -> p11.OOFDevelopment:
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


class P16GFMDenoiseContractTest(unittest.TestCase):
    def test_adapter_runs_real_decoder_semantics_and_preserves_visible(self) -> None:
        import torch

        class FakeNetwork(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

            def patchify(self, images):
                return (
                    images[:, 0]
                    .permute(0, 2, 1)
                    .contiguous()
                )

            def unpatchify(self, patches):
                return patches.permute(0, 2, 1).unsqueeze(1).contiguous()

            def forward_encoder(self, images, priorities, len_keep):
                batch = len(images)
                order = torch.argsort(priorities, dim=1)
                restore = torch.argsort(order, dim=1)
                mask = torch.ones(
                    (batch, 160),
                    dtype=images.dtype,
                    device=images.device,
                )
                mask[:, :len_keep] = 0
                mask = torch.gather(mask, 1, restore)
                latent = torch.zeros(
                    (batch, len_keep + 1, 1200),
                    dtype=images.dtype,
                    device=images.device,
                )
                return latent, mask, restore

            def forward_decoder(self, latent, ids_restore):
                return torch.full(
                    (len(latent), 160, 400),
                    7.0,
                    dtype=latent.dtype,
                    device=latent.device,
                )

        network = FakeNetwork()
        model = adapter._make_wrapper(  # noqa: SLF001
            torch,
            network,
            freeze_model=True,
            weight_mode="pretrained",
            asset_audit={},
        )
        image = torch.arange(
            400 * 160,
            dtype=torch.float32,
        ).reshape(1, 1, 400, 160)
        priorities = torch.arange(160, dtype=torch.float32).reshape(1, 160)
        output = model(image, priorities, len_keep=120)
        self.assertEqual(tuple(output.reconstruction.shape), (1, 1, 400, 160))
        self.assertEqual(int(output.mask.sum()), 40)
        np.testing.assert_array_equal(
            output.reconstruction[0, 0, :, :120].numpy(),
            image[0, 0, :, :120].numpy(),
        )
        np.testing.assert_array_equal(
            output.reconstruction[0, 0, :, 120:].numpy(),
            np.full((400, 40), 7.0),
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_adapter_capabilities_describe_masked_reconstruction(self) -> None:
        capabilities = adapter.capabilities()
        self.assertEqual(adapter.model_id, "geophysical_fm_denoise")
        self.assertEqual(capabilities["output_shape"], "[B,1,400,160]")
        self.assertEqual(
            capabilities["pretraining_objective"],
            "trace_masking_vit_mae",
        )
        self.assertTrue(
            capabilities["supports_same_architecture_random_init"]
        )
        self.assertIn("masked traces", capabilities["reconstruction_semantics"])

    def test_mask_schedule_is_exact_deterministic_and_seeded(self) -> None:
        first = p16.trace_mask_priorities(5, seed=2693)
        second = p16.trace_mask_priorities(5, seed=2693)
        other = p16.trace_mask_priorities(5, seed=2694)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        mask = p16.trace_mask_from_priorities(first)
        np.testing.assert_array_equal(mask.sum(axis=1), np.full(5, 40))

    def test_augmented_route_retains_raw_and_adds_reconstruction(self) -> None:
        oof = _synthetic_oof()
        rng = np.random.default_rng(2693)
        reconstructed = rng.normal(
            size=(len(p11.REPEAT_SEEDS), len(oof.target), 3)
        )
        native = rng.normal(size=(len(oof.target), 3))
        denoised = p16.DenoisedPointFeatures(
            reconstructed=reconstructed,
            delta_from_native_input=reconstructed - native[None],
            trace_masked=np.zeros(
                (len(p11.REPEAT_SEEDS), len(oof.target)),
                dtype=bool,
            ),
            native_input=native,
            audit={},
        )
        features = p16.build_augmented_features(
            oof=oof,
            denoised=denoised,
            seed_id=0,
        )
        self.assertEqual(features.shape, (len(oof.target), 15))
        np.testing.assert_array_equal(features[:, :6], oof.structural_features)

    def test_reused_harness_has_gate_zero_raw_and_random_controls(self) -> None:
        oof = _synthetic_oof()
        rng = np.random.default_rng(2693)
        native = rng.normal(size=(len(oof.target), 3))
        masked = rng.random(
            (len(p11.REPEAT_SEEDS), len(oof.target))
        ) < 0.25

        def make_features(offset: float) -> p16.DenoisedPointFeatures:
            reconstructed = np.repeat(
                native[None],
                len(p11.REPEAT_SEEDS),
                axis=0,
            )
            reconstructed = reconstructed + offset * masked[:, :, None]
            return p16.DenoisedPointFeatures(
                reconstructed=reconstructed,
                delta_from_native_input=reconstructed - native[None],
                trace_masked=masked,
                native_input=native,
                audit={},
            )

        experiment, payload = p16.evaluate_p16(
            oof=oof,
            pretrained=make_features(0.01),
            random_init=make_features(0.02),
        )
        self.assertTrue(
            experiment["baseline"]["gate_zero_bitwise_equal_to_pykrige"]
        )
        for mode in p16.HEAD_MODES:
            head = experiment["heads"][mode]
            self.assertIn("pretrained_gfm_reconstruction", head)
            self.assertIn(
                "random_init_same_architecture_reconstruction",
                head,
            )
            self.assertIn("raw_no_foundation_structural", head)
            self.assertTrue(
                head["promotion"][
                    "matched_random_init_reconstruction_control_present"
                ]
            )
        self.assertEqual(payload["fold_ids"].shape, (len(oof.target),))

    def test_cli_exposes_no_test_or_holdout_argument(self) -> None:
        parser = p16._parser()  # noqa: SLF001
        run_parser = parser._subparsers._group_actions[0].choices[  # noqa: SLF001
            "run"
        ]
        help_text = parser.format_help() + run_parser.format_help()
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--holdout", help_text)

    def test_output_validation_preserves_p11_p15(self) -> None:
        for name in ("p11_residual_fusion", "p15_gfm_finetune"):
            with self.assertRaises(ValueError):
                p16._validate_output_dir(  # noqa: SLF001
                    HERE / "_outputs" / name
                )
        accepted = p16._validate_output_dir(  # noqa: SLF001
            HERE / "_outputs" / "p16_gfm_denoise"
        )
        self.assertEqual(
            accepted,
            (HERE / "_outputs" / "p16_gfm_denoise").resolve(),
        )


class P16GFMDenoisePortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p16_gfm_denoise"
        summary_path = cls.output_dir / "summary.json"
        if not summary_path.is_file():
            raise unittest.SkipTest("real P16 evidence not generated")
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def test_real_evidence_uses_decoder_native_windows_and_controls(self) -> None:
        self.assertEqual(self.summary["schema_version"], p16.SCHEMA_VERSION)
        self.assertEqual(self.summary["model"]["decoder_depth"], 12)
        self.assertEqual(self.summary["model"]["len_keep"], 120)
        native = self.summary["native_mapping_audit"]["native_window"]
        self.assertFalse(native["resize_applied"])
        self.assertFalse(native["interpolation_applied"])
        pretrained = self.summary["pretrained_reconstruction_audit"]
        random_init = self.summary["random_init_reconstruction_audit"]
        self.assertTrue(pretrained["pretrained_weights_used_for_forward"])
        self.assertFalse(random_init["pretrained_weights_used_for_forward"])
        self.assertTrue(random_init["seed_distinct_random_states"])
        self.assertEqual(
            pretrained["architecture_sha256"],
            random_init["architecture_sha256"],
        )
        self.assertTrue(self.summary["seismic_alignment_audit"]["passed"])

    def test_real_evidence_preserves_firewall_and_independence(self) -> None:
        experiment = self.summary["experiment"]
        self.assertTrue(
            experiment["baseline"]["gate_zero_bitwise_equal_to_pykrige"]
        )
        self.assertEqual(
            experiment["fixed_protocol"]["independent_spatial_units"],
            5,
        )
        self.assertFalse(
            experiment["decision"]["pretrained_contribution_claimed"]
        )
        firewall = self.summary["holdout_firewall"]
        for key in (
            "label_dataset_read_from_hdf5",
            "test_path_argument_exists",
            "test_h5_opened",
            "frozen_holdout_opened",
            "historical_test_metrics_read",
        ):
            self.assertFalse(firewall[key])
        evidence = (self.output_dir / "evidence.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("not 15 independent observations", evidence)
        self.assertIn("visible traces", evidence)
        self.assertIn("random initialization", evidence)

    def test_real_artifacts_are_hash_locked_and_reconstruction_local(self) -> None:
        implementation = self.summary["implementation"]
        self.assertEqual(
            implementation["script_sha256"],
            _sha256(HERE / "p16_gfm_denoise.py"),
        )
        self.assertEqual(
            implementation["adapter_sha256"],
            _sha256(
                PROJECT_ROOT
                / "_models"
                / "reconstruction"
                / "geophysical_fm_denoise.py"
            ),
        )
        manifest_path = self.output_dir / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.items():
            self.assertEqual(_sha256(self.output_dir / name), expected)
        prediction_path = self.output_dir / "prediction_errors.npz"
        with np.load(prediction_path, allow_pickle=False) as payload:
            self.assertEqual(payload["target"].shape, (10240,))
            self.assertEqual(payload["fold_ids"].shape, (10240,))


if __name__ == "__main__":
    unittest.main()
