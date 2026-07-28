from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np


RECON_DIR = Path(__file__).resolve().parents[1]
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))

mod = importlib.import_module("visualize_3d_sci")


def test_load_archive_uses_native_voxel_fields_only() -> None:
    archive = mod._load_archive("conditional")
    assert archive["volume_shape_kji"] == (63, 100, 108)
    assert archive["indices_kji"].shape[1] == 3
    assert np.isfinite(archive["truth"]).all()
    assert np.isfinite(archive["prediction"]).all()
    assert np.isfinite(archive["residual"]).all()


def test_render_metadata_keeps_scientific_boundary_text(tmp_path: Path) -> None:
    archive = mod._load_archive("strict")
    sample = mod._lexsort_sample(archive["indices_kji"], max_points=128)
    feasibility = mod._feasibility_record("strict", archive, sample)
    provenance = mod._provenance_record("strict", archive, sample)
    caption = mod._create_caption("conditional", mod._load_archive("conditional"))

    assert feasibility["feasibility"] == "native_volume"
    assert feasibility["native_axes"] == ["K", "J", "I"]
    assert feasibility["boundary"]["strict"] == "no test-region well constraints used"
    assert (
        feasibility["boundary"]["conditional"]
        == "test-region constraints supplied; exact well cells excluded from metrics"
    )
    assert provenance["native_volume"]["coordinates"] == "native voxel K/J/I only"
    assert "conditional reconstruction, not strict holdout" in caption
    assert "native voxel K/J/I only" in caption
    assert "physical XYZ" in caption


def test_visualizer_source_has_no_titles_and_only_expected_outputs() -> None:
    source = (RECON_DIR / "visualize_3d_sci.py").read_text(encoding="utf-8")
    assert "set_title" not in source
    assert "suptitle" not in source
    assert "jet" not in source
    assert "rainbow" not in source
    assert "viridis" not in source.lower()
    assert "coolwarm" not in source.lower()
    assert "OUTPUT_ROOT = TRACK_ROOT / \"_outputs\" / \"3d_sci_v1\"" in source
    assert "_single_panel_figure" in source
    assert "truth.png" in source
    assert "reconstruction.png" in source
    assert "residual.png" in source
