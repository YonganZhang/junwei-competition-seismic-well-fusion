from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p12_visualization as viz  # noqa: E402


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_hashes(manifest: dict[str, object]) -> dict[str, tuple[str, str, str]]:
    hashes: dict[str, tuple[str, str, str]] = {}
    for entry in manifest["figures"]:
        hashes[entry["target"]] = (entry["sha256_png"], entry["sha256_pdf"], entry["sha256_svg"])
    return hashes


def test_render_p12_visualization_bundle(tmp_path: Path) -> None:
    manifest = viz.generate_artifacts(tmp_path / "p12_visualization")
    manifest_path = tmp_path / "p12_visualization" / "manifest.json"
    figures_root = tmp_path / "p12_visualization" / "figures"
    assert manifest_path.is_file()
    assert figures_root.is_dir()

    loaded = _load_manifest(manifest_path)
    assert loaded["schema_version"] == 1
    assert loaded["track_id"] == "property"
    assert loaded["renderer"] == "p12_visualization"
    assert loaded["source_commit"]
    assert loaded["resolved_font"] in {"TeX Gyre Termes", "Times New Roman"}
    assert loaded["split_scope"]["known_holdout_family"] == "15/9-F-15"
    assert loaded["split_scope"]["fresh_blind"] is False
    assert loaded["split_scope"]["prior_test_consumed"] is True

    legacy_notes = loaded["lineage_notes"]
    assert any("test_depth_gt_vs_pred.png" in note for note in legacy_notes)
    assert any("log1p(mD)" in note for note in legacy_notes)

    assert len(loaded["figures"]) == 3
    for entry in loaded["figures"]:
        assert entry["kind"] == "heldout_summary"
        for suffix in ("png", "pdf", "svg"):
            path = tmp_path / "p12_visualization" / entry[f"path_{suffix}"]
            assert path.is_file()
            assert path.stat().st_size > 0
        png_path = tmp_path / "p12_visualization" / entry["path_png"]
        image = Image.open(png_path)
        assert image.size == (2160, 2160)
        assert entry["dimensions_png"] == [2160, 2160]
        assert entry["metrics"]["physical_mae"] >= 0.0
        assert entry["metrics"]["physical_rmse"] >= 0.0
        assert 0.0 <= entry["metrics"]["empirical_interval_coverage"] <= 1.0
        if entry["target"] == "KLOGH":
            assert entry["axis_rules"]["interval_display_domain"] == "log1p(mD)"
            assert entry["axis_rules"]["interval_display_transform"] == "log1p(mD)"
        else:
            assert entry["axis_rules"]["interval_display_domain"] == "fraction"
            assert entry["axis_rules"]["interval_display_transform"] == "identity"

    assert all(item["shape"] == [344, 11] for item in loaded["target_sources"].values())
    assert all(item["path"].endswith("predictions.csv") for item in loaded["target_sources"].values())
    assert any(item["path"].endswith("run_manifest.json") for item in loaded["source_inputs"].values())
    assert any(item["path"].endswith("test_depth_gt_vs_pred.png") for item in loaded["source_inputs"].values())
    assert loaded["visual_qa"]["titles_present"] is False
    assert loaded["visual_qa"]["vector_outputs_present"] is True
    assert loaded["visual_qa"]["fonts_normalized"] is True
    assert loaded["visual_qa"]["klogh_interval_display_domain"] == "log1p(mD)"

    contract = loaded["p12_contract"]
    assert contract["schema_version"] == "scientific-visualization-contract/v1"
    assert contract["profile"] == "p12_tracks_1_3_5"
    assert contract["track_id"] == "property"
    assert contract["renderer"]["path"].endswith("p12_visualization.py")
    assert contract["manual_review"]["reviewed"] is False
    assert contract["manual_review"]["status"] == "pending"
    assert len(contract["inputs"]) >= 7
    assert len(contract["outputs"]) == 3
    for item in contract["inputs"]:
        assert "path" in item and "sha256" in item
        assert "shape_or_row_count" in item
        assert "scientific_role" in item
        assert "split_scope" in item
    for item in contract["outputs"]:
        assert item["dpi"] == 300
        assert item["width_px"] == 2160
        assert item["height_px"] == 2160
        assert item["vector_companions"]
        assert all(str(path).endswith(('.pdf', '.svg')) for path in item["vector_companions"])

    # manifest must be reproducible from actual files
    for key, item in loaded["target_sources"].items():
        assert item["sha256"]
        assert item["size_bytes"] > 0
    for entry in loaded["figures"]:
        assert entry["sha256_png"]
        assert entry["sha256_pdf"]
        assert entry["sha256_svg"]

    # Output hashes must be deterministic across two independent renders.
    manifest_2 = viz.generate_artifacts(tmp_path / "p12_visualization_again")
    assert _figure_hashes(loaded) == _figure_hashes(manifest_2)
    for render_root in (tmp_path / "p12_visualization", tmp_path / "p12_visualization_again"):
        for entry in _load_manifest(render_root / "manifest.json")["figures"]:
            pdf_bytes = (render_root / entry["path_pdf"]).read_bytes()
            svg_text = (render_root / entry["path_svg"]).read_text(encoding="utf-8")
            assert b"CreationDate" not in pdf_bytes
            assert b"ModDate" not in pdf_bytes
            assert "dc:date" not in svg_text
            assert "Date" not in svg_text.split("<metadata>", 1)[-1][:4000]

    # Quick sanity on record alignment and target-specific logging.
    target_rows = viz._load_all_targets()
    assert set(target_rows) == {"PHIF", "KLOGH", "SW"}
    assert len(target_rows["PHIF"]) == 344
    assert target_rows["PHIF"][0].family_id == "15/9-F-15"
    assert target_rows["PHIF"][0].well_id == "15/9-F-15 D"


def test_source_has_no_titles_and_uses_vector_outputs() -> None:
    source = (HERE / "p12_visualization.py").read_text(encoding="utf-8")
    assert "set_title" not in source
    assert "suptitle" not in source
    assert "plt.title" not in source
    assert "bbox_inches" not in source
    assert "svg.hashsalt" in source
    assert "CreationDate" in source
    assert "ModDate" in source
    assert "_save_figure_bundle" in source
    for label in ("(a)", "(b)", "(c)", "(d)"):
        assert label in source
    assert "log1p(mD)" in source
    assert "svg" in source
    assert "pdf" in source
