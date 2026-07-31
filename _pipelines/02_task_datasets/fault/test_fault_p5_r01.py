#!/usr/bin/env python3
"""Fail-closed unit/contract tests for fault-prefixed P5.1 R0/R1."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

import fault_p5_r01 as r01  # noqa: E402


def tiny_development(seed: int = r01.ROOT_SEED) -> r01.DevelopmentSamples:
    rng = np.random.default_rng(seed)
    inlines = np.repeat(np.arange(10000, 10080, 5, dtype=np.int32), 2)
    amplitudes = rng.normal(size=(len(inlines), 1, *r01.PATCH_SHAPE)).astype(np.float32)
    labels = np.zeros((len(inlines), *r01.PATCH_SHAPE), dtype=np.uint8)
    labels[0::2, r01.PATCH_SHAPE[0] // 2, r01.PATCH_SHAPE[1] // 2] = 1
    kinds = tuple("fault_positive" if index % 2 == 0 else "proxy_negative" for index in range(len(inlines)))
    return r01.DevelopmentSamples(
        amplitudes=amplitudes,
        sparse_positive=labels,
        sample_ids=tuple(f"tiny-{index}" for index in range(len(inlines))),
        inlines=inlines,
        kinds=kinds,
        source_hashes={
            "seismic_amplitude": "a" * 64,
            "fault_points": "b" * 64,
            "seismic_index": "c" * 64,
        },
    )


def lock_evidence() -> dict:
    return {
        "input_sha256": {
            "seismic_amplitude": "a" * 64,
            "fault_points": "b" * 64,
            "seismic_index": "c" * 64,
        },
        "development_inline_range": [9985, 10284],
        "lock_sha256": "d" * 64,
    }


class FaultP5R01GateTests(unittest.TestCase):
    def test_three_lanes_are_isolated_and_current_formal_lane_stays_blocked(self) -> None:
        gates = r01.build_r0_gates(lock_evidence=lock_evidence())
        self.assertEqual(tuple(gate["lane_id"] for gate in gates), r01.LANES)
        self.assertEqual(len({gate["hashes"]["config"] for gate in gates}), 3)
        for gate in gates:
            self.assertEqual(set(gate["hashes"]), {"source", "label", "split", "config"})
            self.assertFalse(gate["test_firewall"]["frozen_test_accessed"])
        weak = gates[1]
        self.assertTrue(weak["data_ready"])
        self.assertFalse(weak["train_allowed"])
        self.assertFalse(weak["rank_allowed"])
        self.assertEqual(weak["requirements"]["stick"], "positive")
        self.assertEqual(weak["requirements"]["unlabelled"], "unknown")
        self.assertFalse(weak["requirements"]["valid_label_mask_for_unlabelled"])
        self.assertTrue(weak["requirements"]["proxy_separate"])
        formal = gates[2]
        self.assertFalse(formal["data_ready"])
        self.assertFalse(formal["train_allowed"])
        self.assertFalse(formal["rank_allowed"])
        self.assertIn("AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING", formal["reason_codes"])

    def test_contract_fixture_cannot_unlock_synthetic_lane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "synthetic.json"
            registry = root / "registry.yml"
            payload = {
                "dataset_id": "fault_synthetic_contract_fixture",
                "source_role": "synthetic_verified_batch contract fixture",
                "dense_ground_truth": True,
                "source_sha256": "a" * 64,
                "label_sha256": "b" * 64,
                "train_volume_ids": ["v1"],
                "validation_volume_ids": ["v2"],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            registry.write_text("fault_synthetic_contract_fixture: {}\n", encoding="utf-8")
            gate = r01._synthetic_gate(manifest, registry)
        self.assertFalse(gate["data_ready"])
        self.assertFalse(gate["train_allowed"])
        self.assertFalse(gate["rank_allowed"])
        self.assertFalse(gate["requirements"]["contract_fixture_prohibited"])

    def test_registered_dense_independent_synthetic_contract_is_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "synthetic.json"
            registry = root / "registry.yml"
            payload = {
                "dataset_id": "fault_dense_generator_v1",
                "source_role": "registered_dense_generator",
                "dense_ground_truth": True,
                "source_sha256": "a" * 64,
                "label_sha256": "b" * 64,
                "train_volume_ids": ["generated-1", "generated-2"],
                "validation_volume_ids": ["generated-3"],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            registry.write_text("fault_dense_generator_v1: registered\n", encoding="utf-8")
            gate = r01._synthetic_gate(manifest, registry)
        self.assertTrue(gate["data_ready"])
        self.assertTrue(gate["train_allowed"])
        self.assertTrue(gate["rank_allowed"])
        self.assertEqual(gate["ranking_scope"], "within_synthetic_only_lane")


class FaultP5R01MechanismTests(unittest.TestCase):
    def test_development_centre_selection_never_invents_negative_centres(self) -> None:
        grouped = {
            inline: np.asarray([[2200, 500]], dtype=np.int32)
            for inline in range(10000, 10060)
        }
        index = {
            "xl_min": np.asarray(1900),
            "xl_max": np.asarray(2500),
            "samples_ms": np.arange(1000, dtype=float),
        }
        centres, evidence = r01._choose_development_centres(
            grouped,
            index,
            [10000, 10059],
            root_seed=r01.ROOT_SEED,
        )
        self.assertEqual(len(centres), r01.DEVELOPMENT_INLINE_COUNT)
        self.assertTrue(all(kind == "fault_positive" for *_, kind in centres))
        self.assertFalse(evidence["random_or_annotation_free_negative_centres_generated"])

    def test_random_and_buffered_splits_have_distinct_proximity_evidence(self) -> None:
        samples = tiny_development()
        random_split = r01._random_split(samples, r01.ROOT_SEED)
        spatial_split = r01._spatial_split(samples, r01.BUFFER_INLINES)
        random_audit = r01._split_audit(samples, random_split)
        spatial_audit = r01._split_audit(samples, spatial_split)
        self.assertEqual(random_audit["sample_id_overlap_count"], 0)
        self.assertGreater(random_audit["inline_overlap_count"], 0)
        self.assertEqual(spatial_audit["inline_overlap_count"], 0)
        self.assertGreater(spatial_audit["minimum_train_validation_inline_distance"], r01.BUFFER_INLINES)
        self.assertGreater(spatial_audit["excluded_buffer_samples"], 0)

    def test_invalid_diagnostics_are_fixed_budget_not_rankable_and_fold_train_fit(self) -> None:
        samples = tiny_development()
        split = r01._spatial_split(samples, r01.BUFFER_INLINES)
        result = r01.run_invalid_diagnostic(
            samples,
            protocol_id="unit_spatial",
            split=split,
            root_seed=r01.ROOT_SEED,
        )
        train_values = samples.amplitudes[split["train"]]
        self.assertTrue(result["scientifically_invalid"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["ranking_status"], "not_rankable")
        self.assertEqual(result["model_id"], "fault_local_logistic")
        self.assertEqual(result["fixed_budget"]["parameter_updates"], 4)
        self.assertFalse(result["fixed_budget"]["hpo_performed"])
        self.assertAlmostEqual(result["fold_train_fit"]["preprocessing"]["mean"], float(train_values.mean()))
        self.assertEqual(result["fold_train_fit"]["preprocessing"]["fit_scope"], "fold_train_only")
        self.assertEqual(result["fold_train_fit"]["class_weights"]["fit_scope"], "fold_train_only")
        self.assertEqual(result["fold_train_fit"]["threshold"]["fit_scope"], "fold_train_only")
        self.assertFalse(result["test_firewall"]["frozen_test_accessed"])
        self.assertEqual(set(result["split"]) & {"split_hash"}, {"split_hash"})

    def test_legal_mask_keeps_unknown_and_proxy_separate_and_stops_at_zero_fold(self) -> None:
        samples = tiny_development()
        with mock.patch.object(r01, "discover_model", side_effect=AssertionError("model must not build")):
            result = r01.formal_mask_gate(samples, root_seed=r01.ROOT_SEED)
        counts = result["mask_counts"]
        self.assertEqual(counts["verified_negative"], 0)
        self.assertEqual(counts["valid_label"], counts["positive"])
        self.assertGreater(counts["unknown"], 0)
        self.assertGreater(counts["proxy"], 0)
        self.assertEqual(result["split"]["effective_folds"], 0)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["ranking_status"], "not_rankable")
        self.assertFalse(result["operations"]["model_built"])
        self.assertFalse(result["operations"]["training_invoked"])

    def test_seed_and_test_input_surfaces_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            r01._verify_root_seed(2694)
        required = [
            "--development-segy", "dev.segy",
            "--fault-points", "faults.npz",
            "--seismic-index", "index.npz",
        ]
        for option in ("--test-hdf5", "--holdout", "--test-metrics"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    r01.parse_args(required + [option, "forbidden"])
        source = Path(r01.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        option_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")
        }
        self.assertFalse(any("test" in option or "holdout" in option for option in option_literals))

    def test_label_complement_is_never_promoted_to_verified_negative(self) -> None:
        samples = tiny_development()
        result = r01.formal_mask_gate(samples, root_seed=r01.ROOT_SEED)
        self.assertGreater(result["mask_counts"]["unknown"], 0)
        self.assertEqual(result["mask_counts"]["verified_negative"], 0)
        self.assertIn("AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
