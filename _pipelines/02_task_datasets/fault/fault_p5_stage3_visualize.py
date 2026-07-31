#!/usr/bin/env python3
"""Rebuild fault Stage-3 readiness figures from one portable data manifest."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping


FIGURE_NAMES = (
    "fault_readiness.svg",
    "fault_negative_coverage.svg",
    "fault_unknown_coverage.svg",
)


class FaultStage3VisualizationError(RuntimeError):
    """The archived readiness manifest is missing or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FaultStage3VisualizationError("portable readiness manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != "fault-p5-stage3-data-readiness-v1"
        or payload.get("track_id") != "fault"
    ):
        raise FaultStage3VisualizationError("unexpected fault readiness manifest")
    firewall = payload.get("test_firewall", {})
    if firewall.get("frozen_test_accessed") is not False:
        raise FaultStage3VisualizationError("readiness manifest does not prove a closed test firewall")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _text(x: int, y: int, value: Any, *, size: int = 16, weight: int = 400, fill: str = "#183153") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(str(value))}</text>'
    )


def _base_svg(title: str, subtitle: str, body: str, *, height: int = 520) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{html.escape(title)}">\n'
        '<rect width="960" height="100%" fill="#f7f9fc"/>\n'
        '<rect x="28" y="24" width="904" height="70" rx="12" fill="#17324d"/>\n'
        + _text(52, 57, title, size=24, weight=700, fill="#ffffff")
        + _text(52, 81, subtitle, size=13, fill="#d7e7f5")
        + body
        + _text(52, height - 24, "Portable development audit only · no model score · frozen test untouched", size=12, fill="#60758a")
        + "\n</svg>\n"
    )


def _readiness_svg(payload: Mapping[str, Any]) -> str:
    coverage = payload["coverage"]
    split = payload["split"]
    gates = (
        ("Fault-stick positives", coverage["voxel_probe"]["positive_labels"] > 0, f'{coverage["voxel_probe"]["positive_labels"]} observed'),
        ("Coverage-audited negatives", coverage["voxel_probe"]["verified_negative_labels"] > 0, f'{coverage["voxel_probe"]["verified_negative_labels"]} observed'),
        ("Explicit audited unknown mask", coverage["unknown_provenance"]["status"] == "ready", coverage["unknown_provenance"]["status"]),
        ("Buffered development folds", split["effective_n_splits"] > 0, f'{split["effective_n_splits"]} effective'),
        ("Complete annotation blocks", coverage["spatial"]["complete_annotation_blocks"] > 0, f'{coverage["spatial"]["complete_annotation_blocks"]} complete'),
    )
    rows: list[str] = []
    for index, (label, ready, detail) in enumerate(gates):
        y = 128 + index * 67
        colour = "#218739" if ready else "#c43d3d"
        state = "READY" if ready else "BLOCKED"
        rows.extend(
            [
                f'<rect x="52" y="{y}" width="856" height="50" rx="9" fill="#ffffff" stroke="#d8e1ea"/>',
                f'<circle cx="78" cy="{y + 25}" r="10" fill="{colour}"/>',
                _text(101, y + 22, label, size=16, weight=600),
                _text(101, y + 40, detail, size=12, fill="#60758a"),
                _text(790, y + 31, state, size=14, weight=700, fill=colour),
            ]
        )
    return _base_svg(
        "Fault Stage-3 data readiness",
        "Only positive support is present; formal multiseed CV remains scientifically blocked",
        "".join(rows),
    )


def _negative_svg(payload: Mapping[str, Any]) -> str:
    probe = payload["coverage"]["voxel_probe"]
    total = int(probe["total_voxels"])
    positive = int(probe["positive_labels"])
    negative = int(probe["verified_negative_labels"])
    unknown = int(probe["unknown_labels"])
    scale = 760 / max(total, 1)
    positive_width = positive * scale
    negative_width = negative * scale
    unknown_width = unknown * scale
    x0, y0 = 100, 190
    body = [
        _text(52, 132, f"Real development smoke crop: {total:,} voxels", size=18, weight=700),
        f'<rect x="{x0}" y="{y0}" width="760" height="62" rx="8" fill="#e5eaf0"/>',
        f'<rect x="{x0}" y="{y0}" width="{positive_width:.3f}" height="62" fill="#2c7fb8"/>',
        f'<rect x="{x0 + positive_width:.3f}" y="{y0}" width="{negative_width:.3f}" height="62" fill="#f28e2b"/>',
        f'<rect x="{x0 + positive_width + negative_width:.3f}" y="{y0}" width="{unknown_width:.3f}" height="62" rx="8" fill="#b8c2cc"/>',
        f'<line x1="{x0 + positive_width:.3f}" y1="{y0 - 12}" x2="{x0 + positive_width:.3f}" y2="{y0 + 74}" stroke="#f28e2b" stroke-width="3"/>',
        _text(100, 302, f"Positive: {positive} ({probe['positive_fraction']:.2%})", size=16, weight=700, fill="#2c7fb8"),
        _text(100, 337, f"Verified negative: {negative} ({probe['verified_negative_fraction']:.2%})", size=16, weight=700, fill="#d66c00"),
        _text(100, 372, f"Unknown: {unknown} ({probe['unknown_fraction']:.2%})", size=16, weight=700, fill="#60758a"),
        '<rect x="580" y="292" width="280" height="90" rx="10" fill="#fff4e8" stroke="#f28e2b"/>',
        _text(602, 324, "Formal binary loss unavailable", size=15, weight=700, fill="#a44d00"),
        _text(602, 349, "Unknown cannot become background", size=13, fill="#7b5a3a"),
        _text(602, 371, "No random negatives generated", size=13, fill="#7b5a3a"),
    ]
    return _base_svg(
        "Verified-negative coverage",
        "Counts are archived Stage-1 development evidence, not inferred labels",
        "".join(body),
    )


def _unknown_svg(payload: Mapping[str, Any]) -> str:
    coverage = payload["coverage"]
    probe = coverage["voxel_probe"]
    spatial = coverage["spatial"]
    source_status = coverage["unknown_provenance"]["source_mask_audit_status"]
    audited_slices = spatial["coverage_audited_slice_count"]
    slice_text = "not quantifiable" if audited_slices is None else str(audited_slices)
    unknown_width = 760 * float(probe["unknown_fraction"])
    body = [
        _text(52, 132, "Unknown occupancy in the real smoke crop", size=18, weight=700),
        '<rect x="100" y="170" width="760" height="45" rx="8" fill="#e5eaf0"/>',
        f'<rect x="100" y="170" width="{unknown_width:.3f}" height="45" rx="8" fill="#7d8ea3"/>',
        _text(115, 199, f"{probe['unknown_labels']} / {probe['total_voxels']} unknown ({probe['unknown_fraction']:.2%})", size=15, weight=700, fill="#ffffff"),
        '<rect x="52" y="255" width="410" height="165" rx="10" fill="#ffffff" stroke="#d8e1ea"/>',
        _text(76, 288, "Mask provenance", size=17, weight=700),
        _text(76, 322, "In-memory unknown semantics", size=14),
        _text(340, 322, "available", size=14, weight=700, fill="#218739"),
        _text(76, 354, "Source mask audit", size=14),
        _text(340, 354, source_status, size=14, weight=700, fill="#c43d3d"),
        _text(76, 386, "Audit SHA-256", size=14),
        _text(340, 386, "absent", size=14, weight=700, fill="#c43d3d"),
        '<rect x="486" y="255" width="422" height="165" rx="10" fill="#ffffff" stroke="#d8e1ea"/>',
        _text(510, 288, "Spatial coverage audit", size=17, weight=700),
        _text(510, 322, f"Observed development inlines: {spatial['development_unique_inlines']}", size=14),
        _text(510, 354, f"Complete annotation blocks: {spatial['complete_annotation_blocks']}", size=14),
        _text(510, 386, f"Coverage-audited slices: {slice_text}", size=14),
    ]
    return _base_svg(
        "Unknown-mask and spatial coverage",
        "Unknown semantics exist, but source-mask and per-slice audit provenance are absent",
        "".join(body),
    )


def build_figures(data_manifest_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Generate three deterministic SVGs from the archived readiness manifest only."""

    payload = _read_manifest(data_manifest_path)
    renderers = (_readiness_svg, _negative_svg, _unknown_svg)
    records: list[dict[str, Any]] = []
    figure_dir = output_dir / "figures"
    for name, renderer in zip(FIGURE_NAMES, renderers):
        path = figure_dir / name
        _atomic_write(path, renderer(payload))
        records.append(
            {
                "path": f"figures/{name}",
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "source": "p5_stage3_data_manifest.json",
                "frozen_test_accessed": False,
            }
        )
    return records
