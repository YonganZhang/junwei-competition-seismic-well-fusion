from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    PROJECT_ROOT
    / "_pipelines/05_research_visualization_expansion/render_research_figures.py"
)
SPEC = importlib.util.spec_from_file_location("render_research_figures", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResearchVisualizationExpansionTests(unittest.TestCase):
    def test_declared_sources_exist(self) -> None:
        for path in (
            MODULE.FAULT_POINTS,
            MODULE.HORIZON_POINTS,
            MODULE.F3_INLINES,
            MODULE.F3_MASKS,
            MODULE.PENOBSCOT,
            MODULE.LITHOFACIES_PREDICTIONS,
            MODULE.RECONSTRUCTION_ROOT
            / "reference/reference_property_volume.npz",
            MODULE.RECONSTRUCTION_ROOT
            / "strict/heldout_reconstruction_volume.npz",
            MODULE.RECONSTRUCTION_ROOT
            / "conditional/heldout_reconstruction_volume.npz",
            MODULE.FOUNDATION_MODEL_CONTRACT,
        ):
            self.assertTrue(path.is_file(), path)

    def test_manifest_is_complete_and_hash_consistent(self) -> None:
        manifest_path = MODULE.DEFAULT_OUTPUT / "artifact_manifest.json"
        self.assertTrue(manifest_path.is_file(), manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        MODULE.validate_manifest(manifest)
        self.assertEqual(
            {track["track"] for track in manifest["tracks"]},
            {
                "fault",
                "facies",
                "property",
                "lithofacies",
                "sweetspot",
                "reconstruction",
                "cross_track",
            },
        )

    def test_scientific_boundaries_are_explicit(self) -> None:
        manifest = json.loads(
            (MODULE.DEFAULT_OUTPUT / "artifact_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(manifest["scientific_contract"]["synthetic_geology"])
        self.assertFalse(
            manifest["scientific_contract"]["cross_field_coordinate_fusion"]
        )
        for track in manifest["tracks"]:
            self.assertTrue(track["scientific_boundary"])
            self.assertIn(
                track["evidence_mode"],
                {"native_volume", "spatial_context", "section_only"},
            )

    def test_foundation_model_contract_has_controlled_pretraining_attribution(self) -> None:
        contract = json.loads(
            MODULE.FOUNDATION_MODEL_CONTRACT.read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(contract["tracks"]),
            {
                "fault",
                "facies",
                "property",
                "lithofacies",
                "sweetspot",
                "reconstruction",
            },
        )
        for track in contract["tracks"].values():
            conditions = " ".join(track["controlled_conditions"]).lower()
            if track["foundation_model"] not in {"TabICL", "Chronos-2 for T3 causal-history forecasting"}:
                self.assertIn("random initialization", conditions)
            self.assertTrue(track["promotion_rule"])
            self.assertTrue(track["current_evidence"])


if __name__ == "__main__":
    unittest.main()
