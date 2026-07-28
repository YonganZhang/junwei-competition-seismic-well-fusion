from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import visualize_3d_sci as viz  # noqa: E402


pytestmark = pytest.mark.integration


def test_real_spatial_context_artifacts_generate_from_live_inputs(tmp_path: Path) -> None:
    records = viz.load_real_records()
    assert len(records) == 344
    first = records[0]
    assert first.well_id and first.family_id
    assert pytest.approx(first.inline, rel=0.0, abs=1e-6) == 10200.0
    assert pytest.approx(first.crossline, rel=0.0, abs=1e-6) == 2247.0
    assert pytest.approx(first.time_ms, rel=0.0, abs=1e-6) == 2544.4612343864806
    assert all(pytest.approx(first.gt[target], rel=0.0, abs=1e-6) == value for target, value in {"PHIF": 0.1404000073671341, "KLOGH": 4.124691009521484, "SW": 0.23839999735355377}.items())

    output_dir = tmp_path / "3d_sci_v1"
    paths = viz.generate_artifacts(output_dir)
    expected = {
        "phif_png",
        "phif_pdf",
        "klogh_png",
        "klogh_pdf",
        "sw_png",
        "sw_pdf",
        "html",
        "caption",
        "feasibility",
        "provenance",
    }
    assert expected == set(paths)
    for name, path in paths.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    assert (output_dir / "phif_spatial_context.png").exists()
    assert (output_dir / "phif_spatial_context.pdf").exists()
    assert (output_dir / "klogh_spatial_context.png").exists()
    assert (output_dir / "klogh_spatial_context.pdf").exists()
    assert (output_dir / "sw_spatial_context.png").exists()
    assert (output_dir / "sw_spatial_context.pdf").exists()
    for name in ("phif", "klogh", "sw"):
        image = matplotlib.image.imread(output_dir / f"{name}_spatial_context.png")
        assert image.shape[:2] == (2160, 2160)

    feasibility = (output_dir / "three_d_feasibility.json").read_text(encoding="utf-8")
    assert '"mode": "spatial_context"' in feasibility
    assert '"trajectory_used": false' in feasibility
    assert '"volume_used": false' in feasibility
    assert '"interpolation_used": false' in feasibility


def test_source_does_not_use_titles_or_suptitles() -> None:
    source = (HERE / "visualize_3d_sci.py").read_text(encoding="utf-8")
    assert "set_title" not in source
    assert "suptitle" not in source
    assert "plt.title" not in source
    assert 'bbox_inches="tight"' not in source
