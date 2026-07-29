from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

import p11_residual_fusion as p11  # noqa: E402


def _synthetic_oof(rows_per_fold: int = 24) -> p11.OOFDevelopment:
    rng = np.random.default_rng(2693)
    folds = np.repeat(np.arange(5), rows_per_fold)
    count = len(folds)
    structural = rng.normal(size=(count, 6))
    latent = rng.normal(size=(count, 12))
    baseline = 0.22 + 0.01 * structural[:, 0]
    residual = 0.006 * latent[:, 0] - 0.004 * latent[:, 1]
    target = baseline + residual + rng.normal(scale=0.001, size=count)
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


class P11ResidualFusionFirewallTest(unittest.TestCase):
    def test_rejects_holdout_paths_before_any_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden holdout path"):
            p11.ensure_no_holdout_paths(
                [
                    Path("/tmp/whatever/test.h5"),
                    Path("/tmp/known_holdout_bundle"),
                ]
            )

    def test_accepts_only_legal_dev_train_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.h5").write_bytes(b"dev")
            # A sibling test file must not be probed or required.
            resolved = p11.resolve_dev_inputs(root)
            self.assertEqual(resolved.train_h5, root / "train.h5")

    def test_resolver_rejects_any_holdout_like_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "holdout"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "forbidden holdout path"):
                p11.resolve_dev_inputs(root)

    def test_cli_has_no_test_path_argument(self) -> None:
        help_text = p11._parser().format_help()  # noqa: SLF001
        run_help = p11._parser()._subparsers._group_actions[0].choices["run"].format_help()  # noqa: SLF001
        self.assertNotIn("--test", help_text)
        self.assertNotIn("--test", run_help)


class P11ResidualFusionContractTest(unittest.TestCase):
    def test_gate_is_bounded_and_zero_gate_is_exact_baseline(self) -> None:
        oof = _synthetic_oof()
        rng = np.random.default_rng(55)
        latent = rng.normal(size=(len(oof.target), 12))
        # Make two genuine-feature columns predictive in this synthetic seam.
        latent[:, 0] = (oof.target - oof.baseline) / 0.006
        latent[:, 1] = 0.0
        result = p11.evaluate_residual_fusion(
            oof,
            latent,
            layer_channels=(4, 4, 4),
            repeat_seeds=(11,),
        )
        aggregate = result["aggregate_by_seed"][0]
        self.assertTrue(aggregate["gate_zero_bitwise_equal_to_pykrige"])
        self.assertEqual(
            aggregate["gate_zero_exact"]["rmse"],
            aggregate["pykrige_oof"]["rmse"],
        )
        for cell in result["per_fold_seed"]:
            self.assertGreaterEqual(cell["gate_stats"]["min"], 0.0)
            self.assertLessEqual(cell["gate_stats"]["max"], 1.0)
            self.assertNotIn(
                cell["outer_fold"],
                cell["train_fold_ids"],
            )
            self.assertEqual(
                cell["validation_fold_ids"],
                [cell["outer_fold"]],
            )
            self.assertEqual(
                cell["base_predictions_for_training"],
                "P5 Stage-3 OOF only",
            )

    def test_inner_residual_predictions_are_cross_fitted(self) -> None:
        oof = _synthetic_oof(rows_per_fold=10)
        features = np.column_stack(
            [oof.target - oof.baseline, oof.structural_features]
        )
        train = oof.fold_ids != 0
        predicted = p11._inner_oof_residual(  # noqa: SLF001
            features[train],
            (oof.target - oof.baseline)[train],
            oof.fold_ids[train],
        )
        self.assertEqual(predicted.shape, (int(np.sum(train)),))
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_latent_channel_budget_is_stratified_by_stage(self) -> None:
        selected = p11._selected_latent_channels((4, 20, 32), seed=2693)  # noqa: SLF001
        self.assertEqual(len(selected), 4 + 16 + 16)
        self.assertEqual(len(np.unique(selected)), len(selected))
        self.assertTrue(np.all((selected >= 0) & (selected < 56)))

    def test_direct_strict_summary_is_not_misreported_as_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "mode": "strict",
                            "split_hash": "strict-hash",
                        },
                        "runtime": {"raw_predictions_persisted": False},
                        "comparison": {"pretrained_macro_fold_rmse": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            result = p11._direct_foundation_comparison(path)  # noqa: SLF001
            self.assertEqual(result["state"], "not_comparable")
            self.assertIn("strict-lane", result["reason"])


class P11PortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = HERE / "_outputs" / "p11_residual_fusion"
        cls.summary = json.loads(
            (cls.output_dir / "summary.json").read_text(encoding="utf-8")
        )

    def test_real_evidence_preserves_genuine_feature_and_firewall_identity(self) -> None:
        self.assertEqual(self.summary["schema_version"], p11.SCHEMA_VERSION)
        self.assertEqual(
            self.summary["openmind_feature_audit"]["checkpoint_sha256"],
            p11.EXPECTED_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            self.summary["implementation"]["script_sha256"],
            p11._sha256(HERE / "p11_residual_fusion.py"),  # noqa: SLF001
        )
        self.assertTrue(
            self.summary["openmind_feature_audit"]["real_pretrained_weights_loaded"]
        )
        self.assertTrue(
            self.summary["openmind_feature_audit"]["source_revision_verified"]
        )
        self.assertEqual(
            self.summary["holdout_firewall"]["hdf5_files_opened"],
            ["train.h5"],
        )
        self.assertFalse(self.summary["holdout_firewall"]["label_dataset_read"])
        self.assertFalse(self.summary["holdout_firewall"]["test_h5_opened"])
        self.assertFalse(self.summary["holdout_firewall"]["frozen_holdout_opened"])

    def test_real_ablation_matrix_is_complete_and_honest(self) -> None:
        experiment = self.summary["experiment"]
        self.assertEqual(len(experiment["per_fold_seed"]), 2 * 5 * 3)
        self.assertEqual(
            self.summary["pykrige_oof_sources"]["rows"],
            10_240,
        )
        for aggregate in experiment["aggregate_by_seed"]:
            self.assertTrue(aggregate["gate_zero_bitwise_equal_to_pykrige"])
            self.assertEqual(
                aggregate["gate_zero_exact"],
                aggregate["pykrige_oof"],
            )
        self.assertEqual(
            self.summary["ablations"]["original_direct_foundation_path"]["state"],
            "not_comparable",
        )
        self.assertEqual(
            experiment["decision"]["state"],
            "VERIFIED_NO_PROMOTION",
        )
        self.assertFalse(experiment["decision"]["default_enabled"])
        self.assertTrue((self.output_dir / "evidence.md").is_file())


if __name__ == "__main__":
    unittest.main()
