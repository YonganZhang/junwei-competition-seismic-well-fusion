from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


RECON_DIR = Path(__file__).resolve().parents[1]
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))

mod = importlib.import_module("visualize_property_volume")


def test_discovers_and_loads_physical_eclipse_volume() -> None:
    source = mod._discover_eclipse_zip()
    volume = mod._load_eclipse_property_volume(source)
    assert volume.shape_kji == (63, 100, 108)
    assert int(volume.active.sum()) == 183_545
    assert np.isfinite(volume.porosity[volume.active]).all()
    assert np.isnan(volume.porosity[~volume.active]).all()
    assert float(volume.easting_m[volume.active].min()) > 400_000
    assert float(volume.northing_m[volume.active].min()) > 6_000_000
    assert float(volume.depth_m[volume.active].min()) > 2_000
    assert volume.inspection["validation"]["ascii_matches_final_init_on_active_cells"] is True


def test_archived_model_cells_map_to_physical_regional_volumes() -> None:
    volume = mod._load_eclipse_property_volume(mod._discover_eclipse_zip())
    strict = mod._regional_model_volume("strict", volume)
    conditional = mod._regional_model_volume("conditional", volume)
    assert strict.truth.shape == (63, 56, 41)
    assert conditional.truth.shape == (63, 52, 29)
    assert int(np.isfinite(strict.truth).sum()) == 78_949
    assert int(np.isfinite(conditional.truth).sum()) == 49_233
    assert np.allclose(
        strict.residual[np.isfinite(strict.residual)],
        (strict.prediction - strict.truth)[np.isfinite(strict.residual)],
        rtol=0,
        atol=1e-7,
    )
    assert strict.archive["state"]["evidence_class"] == "previously_seen_reusable_holdout"


def test_plotly_reference_uses_volume_and_isosurface_not_scatter(tmp_path: Path) -> None:
    volume = mod._load_eclipse_property_volume(mod._discover_eclipse_zip())
    html = mod._render_reference_interactive(volume, tmp_path)
    text = html.read_text(encoding="utf-8")
    assert '"type":"volume"' in text
    assert '"type":"isosurface"' in text
    source = (RECON_DIR / "visualize_property_volume.py").read_text(encoding="utf-8")
    assert "go.Scatter" not in source
    assert "全场参考属性体" in text


def test_source_preserves_scientific_and_evidence_boundaries() -> None:
    source = (RECON_DIR / "visualize_property_volume.py").read_text(encoding="utf-8")
    assert "Scatter3d" not in source
    assert "set_title" not in source
    assert "suptitle" not in source
    assert "previously_seen_reusable_holdout" in source
    assert "reference property body, not a model prediction" in source
    assert "residual is error, not uncertainty" in source


def test_model_skill_warning_exposes_near_constant_predictions() -> None:
    volume = mod._load_eclipse_property_volume(mod._discover_eclipse_zip())
    strict = mod._regional_model_volume("strict", volume)
    conditional = mod._regional_model_volume("conditional", volume)
    strict_skill = mod._model_skill_interpretation("strict", strict)
    conditional_skill = mod._model_skill_interpretation("conditional", conditional)
    assert strict_skill["r2"] < 0
    assert "constant-mean baseline" in strict_skill["reader_warning_en"]
    assert conditional_skill["prediction_to_truth_std_ratio"] < 0.1
    assert "spatial heterogeneity is essentially not reconstructed" in conditional_skill["reader_warning_en"]
    assert (
        strict_skill["practical_skill_verdict"]
        == "no_practically_useful_spatial_reconstruction_skill"
    )
