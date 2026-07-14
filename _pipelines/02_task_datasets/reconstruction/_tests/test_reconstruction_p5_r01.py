"""Portable contracts for reconstruction P5.1 R0/R1."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import hash_file  # noqa: E402

import p4_reconstruction as p4  # noqa: E402
import reconstruction_p5_r01 as r01  # noqa: E402


def write_development_hdf5(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shape = (2, 5, 8)
    with h5py.File(root / "train.h5", "w") as handle:
        record_id = 0
        for k_block in range(7):
            for i_block in range(4):
                start = (2 * k_block, 0, 8 * i_block)
                local_k, local_j, local_i = np.indices(shape)
                z = (start[0] + local_k) / 13.0
                y = local_j / 4.0
                x = (start[2] + local_i) / 31.0
                amplitude = np.sin(2.0 * np.pi * x) + 0.2 * z
                rms = np.sqrt(amplitude**2 + 0.05)
                gradient = 0.3 * z - 0.1 * y
                label = (
                    0.12
                    + 0.035 * x
                    + 0.025 * y
                    + 0.030 * z
                    + 0.020 * np.sin(4.0 * np.pi * x) * np.cos(np.pi * y)
                ).astype(np.float32)
                patch = np.stack(
                    [
                        amplitude,
                        rms,
                        gradient,
                        x,
                        y,
                        z,
                        np.zeros(shape),
                        np.zeros(shape),
                        np.ones(shape),
                    ]
                ).astype(np.float32)
                if k_block == 6 and i_block == 0:
                    patch[6, 0, 0, 0] = label[0, 0, 0]
                    patch[7, 0, 0, 0] = 1.0
                group = handle.create_group(f"sample_{record_id:07d}")
                group.create_dataset("seismic_patch", data=patch)
                group.create_dataset("label", data=label)
                # A sentinel global table exists, but the R0/R1 narrow reader is
                # contractually forbidden from opening this dataset.
                group.create_dataset("well_log_seq", data=np.full((91, 8), 9999.0))
                group.attrs["meta"] = json.dumps(
                    {
                        "patch_index_kji": [k_block, 0, i_block],
                        "patch_start_kji": list(start),
                        "patch_shape_kji": list(shape),
                        "task": "reconstruction",
                        "split": "train",
                    }
                )
                record_id += 1
    # Presence of this sentinel proves that success does not depend on opening
    # the physical test container; it deliberately lacks the unified schema.
    with h5py.File(root / "test.h5", "w") as handle:
        handle.create_dataset("must_not_be_read", data=np.asarray([12345]))


class ReconstructionR01ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data_dir = cls.root / "data"
        cls.output_dir = cls.root / "evidence"
        write_development_hdf5(cls.data_dir)
        cls.test_hash_before = hash_file(cls.data_dir / "test.h5")
        cls.h5_paths_opened: list[str] = []
        cls.paths_statted: list[str] = []
        cls.paths_opened: list[str] = []
        original_h5_file = r01.h5py.File
        original_stat = Path.stat
        original_open = Path.open

        def guarded_h5_file(path, *args, **kwargs):
            candidate = Path(path)
            cls.h5_paths_opened.append(candidate.name)
            if candidate.name == "test.h5":
                raise AssertionError("R0/R1 attempted to open physical test.h5")
            return original_h5_file(path, *args, **kwargs)

        def guarded_stat(path, *args, **kwargs):
            cls.paths_statted.append(path.name)
            if path.name == "test.h5":
                raise AssertionError("R0/R1 attempted to stat physical test.h5")
            return original_stat(path, *args, **kwargs)

        def guarded_open(path, *args, **kwargs):
            cls.paths_opened.append(path.name)
            if path.name == "test.h5":
                raise AssertionError("R0/R1 attempted to hash/read physical test.h5")
            return original_open(path, *args, **kwargs)

        with (
            mock.patch.object(r01.h5py, "File", new=guarded_h5_file),
            mock.patch.object(Path, "stat", new=guarded_stat),
            mock.patch.object(Path, "open", new=guarded_open),
        ):
            cls.prepared = r01.prepare_r0(cls.data_dir)
            cls.result = r01.run_r1(cls.prepared, cls.data_dir)
            cls.outputs = r01.write_outputs(cls.prepared, cls.result, cls.output_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_r0_reuses_p4_buffered_development_fold_without_test_access(self) -> None:
        split = self.prepared.split_manifest
        self.assertEqual(split["fold_id"], 2)
        self.assertEqual(split["effective_train_k_blocks"], [0, 1, 2, 6])
        self.assertEqual(split["purged_k_blocks"], [3, 5])
        self.assertEqual(split["pseudo_test_k_blocks"], [4])
        catalog, _ = p4.synthetic_catalog_and_records()
        p4_fold = p4.build_spatial_manifest("conditional", catalog).folds[2]
        self.assertEqual(
            split["effective_train_k_blocks"],
            sorted(
                {
                    int(sample_id[1:3])
                    for sample_id in p4_fold.purge["effective_train_sample_ids"]
                }
            ),
        )
        audit = self.prepared.manifest["access_audit"]
        self.assertEqual(audit["physical_containers_opened"], ["train.h5"])
        self.assertFalse(audit["physical_test_h5_opened"])
        self.assertFalse(audit["well_log_seq_read"])
        self.assertFalse(audit["reference_sparse_poro_channel_6_read"])
        self.assertTrue(
            self.prepared.manifest["rms_cross_check"][
                "exact_nonzero_porosity_multiset_match"
            ]
        )
        self.assertIn(
            "no RMS spatial mapping",
            self.prepared.manifest["rms_cross_check"]["role"],
        )
        self.assertTrue(self.h5_paths_opened)
        self.assertTrue(all(name == "train.h5" for name in self.h5_paths_opened))
        self.assertNotIn("test.h5", self.paths_statted)
        self.assertNotIn("test.h5", self.paths_opened)
        self.assertEqual(hash_file(self.data_dir / "test.h5"), self.test_hash_before)

    def test_cli_has_only_audit_and_development_run_surfaces(self) -> None:
        parser = r01.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(set(subparsers.choices), {"audit", "run"})
        source = inspect.getsource(r01.build_parser)
        self.assertNotIn('add_parser("test"', source)
        self.assertNotIn('add_parser("hpo"', source)

    def test_pseudo_well_selection_is_geometry_only_and_precedes_target(self) -> None:
        selection = self.prepared.manifest["pseudo_well_selection"]
        self.assertTrue(selection["selection_frozen_before_target_values_read"])
        self.assertTrue(selection["target_is_not_function_argument"])
        first = r01.select_spatial_points(
            self.prepared.geometry.coordinates[self.prepared.pseudo_test_mask],
            self.prepared.geometry.indices_kji[self.prepared.pseudo_test_mask],
            r01.PSEUDO_TEST_WELLS,
        )
        second = r01.select_spatial_points(
            self.prepared.geometry.coordinates[self.prepared.pseudo_test_mask],
            self.prepared.geometry.indices_kji[self.prepared.pseudo_test_mask],
            r01.PSEUDO_TEST_WELLS,
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, self.prepared.pseudo_test_indices)

    def test_r1_uses_one_checkpoint_and_one_common_mask_for_three_conditions(self) -> None:
        self.assertIn(self.result["status"], {"passed", "condition_unaware"})
        self.assertTrue(self.result["same_fixed_model_checkpoint_all_conditions"])
        self.assertEqual(self.result["checkpoint"]["update_count"], r01.MODEL_CONFIG["updates"])
        self.assertEqual(
            self.result["common_metric_mask"]["voxel_count"],
            int(self.prepared.common_metric_mask.sum()),
        )
        self.assertEqual(
            self.result["common_metric_mask"][
                "exact_pseudo_test_cells_excluded_from_all_conditions"
            ],
            r01.PSEUDO_TEST_WELLS,
        )
        counts = {item["voxel_count"] for item in self.result["metrics"].values()}
        self.assertEqual(counts, {int(self.prepared.common_metric_mask.sum())})
        self.assertTrue(self.result["conditions"]["shuffled"]["non_identity"])
        self.assertNotEqual(
            self.result["conditions"]["B1"]["values_hash"],
            self.result["conditions"]["shuffled"]["values_hash"],
        )
        expected_gain = bool(
            self.result["metrics"]["B1"]["rmse"] < self.result["metrics"]["B0"]["rmse"]
            and self.result["metrics"]["B1"]["rmse"]
            < self.result["metrics"]["shuffled"]["rmse"]
        )
        self.assertEqual(self.result["well_information_gain_supported"], expected_gain)
        self.assertAlmostEqual(
            self.result["delta_rmse"]["B1_minus_shuffled"],
            self.result["metrics"]["B1"]["rmse"]
            - self.result["metrics"]["shuffled"]["rmse"],
        )

    def test_condition_feature_is_actually_consumed_and_bands_are_supported(self) -> None:
        sensitivity = self.result["condition_sensitivity"]
        self.assertTrue(sensitivity["feature_hash_B0_differs_from_B1"])
        self.assertGreater(abs(sensitivity["conditional_feature_weight"]), 0.0)
        self.assertGreater(sensitivity["B0_B1_changed_voxels"], 0)
        self.assertGreater(sensitivity["B0_B1_max_abs_prediction_difference"], 0.0)
        self.assertEqual(len(self.result["distance_bands"]), 4)
        for band in self.result["distance_bands"]:
            self.assertGreaterEqual(band["voxel_count"], r01.MIN_BAND_SUPPORT)
            self.assertEqual(set(band["conditions"]), {"B0", "B1", "shuffled"})

    def test_portable_artifacts_and_firewall_hashes(self) -> None:
        self.assertFalse(Path(self.outputs["output_dir"]).is_absolute())
        artifact_manifest = json.loads(
            (self.output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(artifact_manifest["artifacts"]),
            {"r0_manifest.json", "r1_results.json", "P5_R01_REPORT.md"},
        )
        for relative_path, record in artifact_manifest["artifacts"].items():
            self.assertEqual(hash_file(self.output_dir / relative_path), record["sha256"])
        self.assertFalse(self.result["fresh_blind"])
        self.assertFalse(self.result["field_generalization"])
        self.assertEqual(self.result["formal_lane_status"], "blocked")
        self.assertFalse(self.result["test_firewall"]["physical_test_h5_opened"])
        self.assertFalse(self.result["test_firewall"]["global_well_log_seq_read"])

    def test_missing_real_asset_is_structured_blocked_not_backfilled(self) -> None:
        blocked_dir = self.root / "blocked"
        blocked = r01.write_data_gate_blocked(blocked_dir)
        self.assertEqual(blocked["r0_status"], "blocked")
        self.assertEqual(blocked["r1_status"], "blocked")
        payload = json.loads((blocked_dir / "r1_results.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["reason"]["code"], "development_train_h5_not_provisioned")
        self.assertIsNone(payload["metrics"])
        self.assertFalse(payload["test_firewall"]["physical_test_h5_opened"])
        self.assertFalse(payload["test_firewall"]["historical_cache_used"])


if __name__ == "__main__":
    unittest.main()
