#!/usr/bin/env python3
"""Unit, gate, adapter, and Stage-1 tests for fault P5 candidates."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (PROJECT_ROOT, TRACK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from _code.ml_framework.artifacts import atomic_write_json, hash_file  # noqa: E402
from _code.ml_framework.contracts import ModelOutput  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.fault.p5_lock import evaluate_runtime_gate, load_source_locks  # noqa: E402


def _load_track_module(module_name: str, filename: str):
    """Load a track-local module without polluting the global short name."""

    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_p4_contract = _load_track_module("fault_p4_contract", "p4_contract.py")
TARGET_NAME = _p4_contract.TARGET_NAME
fault_task_spec = _p4_contract.fault_task_spec
sys.modules["p4_contract"] = _p4_contract
p5_stage1 = _load_track_module("fault_p5_stage1", "p5_stage1.py")


def write_development_hdf5(
    path: Path,
    *,
    split: str = "train",
    include_verified_negative: bool = False,
    audit_negative: bool = False,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["task"] = "fault"
        handle.attrs["split"] = split
        handle.attrs["n_samples"] = 1
        if audit_negative:
            handle.attrs["verified_negative_audit_status"] = "complete"
            handle.attrs["verified_negative_audit_sha256"] = "a" * 64
        group = handle.create_group("sample_0000000")
        patch = np.linspace(-1.0, 1.0, 33 * 65, dtype=np.float32).reshape(1, 33, 65)
        label = np.zeros((33, 65), dtype=np.uint8)
        label[16, 30:35] = 1
        group.create_dataset("seismic_patch", data=patch)
        group.create_dataset("label", data=label)
        if include_verified_negative:
            negative = np.zeros_like(label, dtype=bool)
            negative[8, 8:13] = True
            group.create_dataset("verified_negative_mask", data=negative)
        group.attrs["position"] = json.dumps(
            {"inline": 10100, "crossline": 2200, "time_index": 680}
        )
        group.attrs["meta"] = json.dumps(
            {"sample_kind": "fault", "task": "fault", "split": split}
        )


class FaultP5SourceAndDiscoveryTests(unittest.TestCase):
    def test_source_lock_contains_exact_first_ten_and_complete_primary_evidence(self) -> None:
        locks = load_source_locks()
        self.assertEqual(tuple(locks), p5_stage1.FIRST_TEN_MODEL_IDS)
        self.assertEqual(len(locks), 10)
        for model_id, record in locks.items():
            self.assertEqual(record["model_id"], model_id)
            self.assertRegex(record["source"]["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(record["source"]["url"].startswith("https://github.com/"))
            self.assertTrue(record["source"]["license_spdx"])
            self.assertTrue(record["dependencies"])

    def test_all_ten_are_dynamically_discoverable_without_central_imports(self) -> None:
        locks = load_source_locks()
        for model_id in p5_stage1.FIRST_TEN_MODEL_IDS:
            discovered = discover_model("fault", model_id)
            self.assertEqual(discovered.model_id, model_id)
            self.assertEqual(discovered.capabilities["source_revision"], locks[model_id]["source"]["revision"])
            self.assertEqual(discovered.capabilities["input_modalities"], ["seismic_amplitude"])

    def test_runtime_gates_reflect_the_shared_environment_and_license_policy(self) -> None:
        expected_ready = {
            "monai_segresnet",
            "monai_dynunet",
            "pytorch3dunet_unet3d",
            "monai_vnet",
            "monai_swinunetr",
        }
        observed_ready = {
            model_id
            for model_id in p5_stage1.FIRST_TEN_MODEL_IDS
            if evaluate_runtime_gate(model_id)["status"] == "ready"
        }
        self.assertEqual(observed_ready, expected_ready)
        self.assertEqual(
            evaluate_runtime_gate("faultseg3d_keras")["reason_code"],
            "NONCOMMERCIAL_LICENSE_NOT_APPROVED",
        )
        self.assertEqual(
            evaluate_runtime_gate("faultnet_md")["reason_code"],
            "PRETRAINED_WEIGHT_NOT_APPROVED",
        )
        self.assertEqual(
            evaluate_runtime_gate("nnunet_v2_3d_fullres")["reason_code"],
            "DEPENDENCY_UNAVAILABLE",
        )


class FaultP5AdapterTests(unittest.TestCase):
    def test_masked_loss_uses_only_valid_positive_and_verified_negative_voxels(self) -> None:
        batch = p5_stage1.synthetic_verified_batch(seed=2693)
        adapter = discover_model("fault", "monai_segresnet").build(fault_task_spec(), seed=2693)
        shape = np.asarray(batch.inputs["seismic_amplitude"]).shape
        logits_a = torch.zeros(shape, dtype=torch.float32)
        logits_b = logits_a.clone()
        unknown = torch.as_tensor(np.asarray(batch.input_masks["unknown_mask"], dtype=bool))
        logits_b[unknown] = 100.0
        output_a = ModelOutput(raw={TARGET_NAME: logits_a}, transformed={TARGET_NAME: torch.sigmoid(logits_a)})
        output_b = ModelOutput(raw={TARGET_NAME: logits_b}, transformed={TARGET_NAME: torch.sigmoid(logits_b)})
        self.assertEqual(float(adapter.masked_loss(batch, output_a)), float(adapter.masked_loss(batch, output_b)))

    def test_one_available_adapter_runs_full_cpu_contract_and_checkpoint(self) -> None:
        evidence = p5_stage1.run_trainable_contract(
            "monai_segresnet",
            p5_stage1.synthetic_verified_batch(seed=2693),
            device=torch.device("cpu"),
            seed=2693,
            split_hash="unit-synthetic",
        )
        self.assertEqual(evidence["status"], "passed")
        self.assertTrue(evidence["backward_finite"])
        self.assertTrue(evidence["checkpoint"]["trainer_state_restored"])
        self.assertLessEqual(evidence["checkpoint"]["prediction_max_abs_diff"], 1e-6)
        self.assertFalse(evidence["frozen_test_accessed"])


class FaultP5DevelopmentAndRunnerTests(unittest.TestCase):
    def _summary_for(self, data_path: Path, root: Path) -> Path:
        summary = root / "build_summary.json"
        atomic_write_json(summary, {"dataset_sha256": {"train": hash_file(data_path)}})
        return summary

    def test_real_development_zeros_remain_unknown_and_missing_negatives_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "development.h5"
            write_development_hdf5(data_path)
            with mock.patch.object(p5_stage1, "BUILD_SUMMARY", self._summary_for(data_path, root)):
                probe = p5_stage1.load_real_development_probe(data_path)
            self.assertGreater(probe.evidence["positive_labels"], 0)
            self.assertEqual(probe.evidence["verified_negative_labels"], 0)
            self.assertEqual(probe.evidence["valid_labels"], probe.evidence["positive_labels"])
            self.assertGreater(probe.evidence["unknown_labels"], 0)

    def test_untrusted_negative_mask_and_non_train_split_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            untrusted = root / "untrusted.h5"
            write_development_hdf5(untrusted, include_verified_negative=True)
            with mock.patch.object(p5_stage1, "BUILD_SUMMARY", self._summary_for(untrusted, root)):
                with self.assertRaisesRegex(p5_stage1.DevelopmentDataUnavailable, "AUDIT_INVALID"):
                    p5_stage1.load_real_development_probe(untrusted)

            wrong_role = root / "wrong-role.h5"
            write_development_hdf5(wrong_role, split="test")
            with mock.patch.object(p5_stage1, "BUILD_SUMMARY", self._summary_for(wrong_role, root)):
                with self.assertRaisesRegex(p5_stage1.DevelopmentDataUnavailable, "ROLE_INVALID"):
                    p5_stage1.load_real_development_probe(wrong_role)

    def test_stage1_subset_runs_real_forward_then_structurally_skips_without_negatives(self) -> None:
        self.assertFalse(
            any("test" in name.lower() for name in inspect.signature(p5_stage1.run_stage1).parameters)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "development.h5"
            write_development_hdf5(data_path)
            with mock.patch.object(p5_stage1, "BUILD_SUMMARY", self._summary_for(data_path, root)):
                summary = p5_stage1.run_stage1(
                    root / "evidence",
                    development_hdf5=data_path,
                    model_ids=("monai_segresnet",),
                    device_name="cpu",
                    root_seed=2693,
                )
            self.assertEqual(summary["counts"], {"contract_smoked": 0, "skipped": 1, "failed": 0})
            self.assertEqual(summary["results"][0]["reason_code"], "NO_AUDITED_VERIFIED_NEGATIVES")
            detail = json.loads((root / "evidence" / "models" / "monai_segresnet.json").read_text())
            self.assertEqual(detail["synthetic_contract"]["status"], "passed")
            self.assertEqual(detail["development"]["forward"]["status"], "passed")
            self.assertEqual(detail["development"]["loss_backward_checkpoint"], "skipped")
            self.assertFalse(summary["p4_evidence"]["frozen_test_accessed"])


if __name__ == "__main__":
    unittest.main()
