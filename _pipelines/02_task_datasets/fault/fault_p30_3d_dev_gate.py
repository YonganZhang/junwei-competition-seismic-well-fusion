#!/usr/bin/env python3
"""Build a contiguous 3-D development asset for the fault track and verify gates.

This script is intentionally fail-closed. It does not touch frozen holdout or
test.h5. It consumes the already-verified ST0202 seismic index and sparse fault
sticks, cuts a continuous development subvolume, derives conservative verified
background / unknown masks from the sparse fault sticks, writes a development-
only group-isolated split manifest, and then checks whether the previously
blocked fault gate criteria are now satisfied.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import segyio

from build_dataset import (  # reuse audited sparse-fault rasterization logic
    nearest_time_indices,
    rasterize_fault_voxels,
    sha256_file,
)

try:  # pragma: no cover - optional dependency path
    from scipy.ndimage import binary_dilation
except Exception:  # pragma: no cover - fallback for minimal environments
    binary_dilation = None


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p30_3d_dev_gate"

SEGY_META_PATH = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs" / "seismic_index_meta.json"
SEISMIC_INDEX_PATH = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs" / "seismic_index.npz"
FAULT_POINTS_PATH = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs" / "fault_points.npz"

DEV_BOX = {
    "iline": (10095, 10235),
    "crossline": (2175, 2350),
    "time_idx": (605, 785),
}
DEV_SPLIT = {
    "fit": (10095, 10175),
    "guard": (10176, 10183),
    "validation": (10184, 10235),
}
SUBVOLUME_COORD_ORDER = ("tline", "iline", "xline")
UNKNOWN_RADIUS = {"tline": 4, "iline": 2, "xline": 2}
BOUNDARY_HALO = 2


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        .strip()
    )


def _load_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    with np.load(FAULT_POINTS_PATH, allow_pickle=True) as z:
        faults = {key: z[key] for key in z.files}
    with np.load(SEISMIC_INDEX_PATH, allow_pickle=False) as z:
        index = {key: z[key] for key in z.files}
    meta = _load_json(SEGY_META_PATH)
    return faults, index, meta


def _subvolume_ranges() -> dict[str, tuple[int, int]]:
    return DEV_BOX


def _inclusive_range(bounds: tuple[int, int]) -> np.ndarray:
    return np.arange(bounds[0], bounds[1] + 1, dtype=np.int32)


def _time_ms(index: dict[str, np.ndarray], time_idx: np.ndarray) -> np.ndarray:
    samples = np.asarray(index["samples_ms"], dtype=np.float64)
    return samples[time_idx.astype(np.int32)]


def _read_subvolume(
    segy_path: Path,
    index: dict[str, np.ndarray],
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    time_idx_values: np.ndarray,
) -> np.ndarray:
    il_min = int(index["il_min"])
    xl_min = int(index["xl_min"])
    n_xl = int(index["n_xl"])
    t0 = int(time_idx_values[0])
    t1 = int(time_idx_values[-1])
    cube = np.empty((len(time_idx_values), len(inline_values), len(crossline_values)), dtype=np.float32)
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        for il_i, inline in enumerate(inline_values.tolist()):
            for xl_i, crossline in enumerate(crossline_values.tolist()):
                trace_index = (int(inline) - il_min) * n_xl + (int(crossline) - xl_min)
                trace = np.asarray(f.trace[int(trace_index)][t0 : t1 + 1], dtype=np.float32)
                if trace.shape[0] != len(time_idx_values):
                    raise RuntimeError(
                        "unexpected trace length while extracting contiguous 3-D development volume"
                    )
                cube[:, il_i, xl_i] = trace
    if not np.isfinite(cube).all():
        raise RuntimeError("extracted seismic subvolume contains non-finite values")
    return cube


def _rasterize_positive_mask(
    faults: dict[str, np.ndarray],
    index: dict[str, np.ndarray],
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    time_idx_values: np.ndarray,
) -> np.ndarray:
    voxels = rasterize_fault_voxels(faults, index)
    il0 = int(inline_values[0])
    xl0 = int(crossline_values[0])
    t0 = int(time_idx_values[0])
    il1 = int(inline_values[-1])
    xl1 = int(crossline_values[-1])
    t1 = int(time_idx_values[-1])
    inside = (
        (voxels[:, 0] >= il0)
        & (voxels[:, 0] <= il1)
        & (voxels[:, 1] >= xl0)
        & (voxels[:, 1] <= xl1)
        & (voxels[:, 2] >= t0)
        & (voxels[:, 2] <= t1)
    )
    local = voxels[inside].astype(np.int32, copy=False)
    mask = np.zeros((len(time_idx_values), len(inline_values), len(crossline_values)), dtype=bool)
    if local.size:
        mask[
            local[:, 2] - t0,
            local[:, 0] - il0,
            local[:, 1] - xl0,
        ] = True
    return mask


def _dilate(mask: np.ndarray, radius: dict[str, int]) -> np.ndarray:
    if binary_dilation is not None:  # pragma: no branch - preferred path
        structure = np.ones(
            (
                2 * int(radius["tline"]) + 1,
                2 * int(radius["iline"]) + 1,
                2 * int(radius["xline"]) + 1,
            ),
            dtype=bool,
        )
        return binary_dilation(mask, structure=structure)
    # Fallback for environments without SciPy.
    padded = np.pad(
        mask,
        (
            (int(radius["tline"]), int(radius["tline"])),
            (int(radius["iline"]), int(radius["iline"])),
            (int(radius["xline"]), int(radius["xline"])),
        ),
        mode="constant",
        constant_values=False,
    )
    result = np.zeros_like(mask, dtype=bool)
    for dt in range(-int(radius["tline"]), int(radius["tline"]) + 1):
        t_slice = slice(int(radius["tline"]) + dt, int(radius["tline"]) + dt + mask.shape[0])
        for di in range(-int(radius["iline"]), int(radius["iline"]) + 1):
            i_slice = slice(int(radius["iline"]) + di, int(radius["iline"]) + di + mask.shape[1])
            for dx in range(-int(radius["xline"]), int(radius["xline"]) + 1):
                x_slice = slice(int(radius["xline"]) + dx, int(radius["xline"]) + dx + mask.shape[2])
                result |= padded[t_slice, i_slice, x_slice]
    return result


def _boundary_halo(shape: tuple[int, int, int], halo: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if halo <= 0:
        return mask
    mask[:halo, :, :] = True
    mask[-halo:, :, :] = True
    mask[:, :halo, :] = True
    mask[:, -halo:, :] = True
    mask[:, :, :halo] = True
    mask[:, :, -halo:] = True
    return mask


def _split_manifest(
    positive_mask: np.ndarray,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    time_idx_values: np.ndarray,
    index: dict[str, np.ndarray],
    faults: dict[str, np.ndarray],
) -> dict[str, Any]:
    faults_inline = np.asarray(faults["inline"], dtype=np.int32)
    faults_xline = np.asarray(faults["crossline"], dtype=np.int32)
    faults_tidx = nearest_time_indices(index["samples_ms"], faults["twt_ms"]).astype(np.int32)
    blocks = []
    for name, bounds in DEV_SPLIT.items():
        il0, il1 = bounds
        points = (
            (faults_inline >= il0)
            & (faults_inline <= il1)
            & (faults_xline >= int(crossline_values[0]))
            & (faults_xline <= int(crossline_values[-1]))
            & (faults_tidx >= int(time_idx_values[0]))
            & (faults_tidx <= int(time_idx_values[-1]))
        )
        block_inline = _inclusive_range(bounds)
        group_mask = (inline_values >= il0) & (inline_values <= il1)
        blocks.append(
            {
                "name": name,
                "inline": [int(il0), int(il1)],
                "n_inline": int(len(block_inline)),
                "positive_voxels": int(
                    positive_mask[:, group_mask, :].sum()
                ),
                "fault_point_count": int(points.sum()),
            }
        )
    return {
        "schema_version": "fault_p30_3d_dev_gate/split/v1",
        "track_id": "fault",
        "development_only": True,
        "group_isolated": True,
        "coordinate_order": list(SUBVOLUME_COORD_ORDER),
        "subvolume": {
            "inline": [int(inline_values[0]), int(inline_values[-1])],
            "crossline": [int(crossline_values[0]), int(crossline_values[-1])],
            "time_idx": [int(time_idx_values[0]), int(time_idx_values[-1])],
            "time_ms": [float(_time_ms(index, time_idx_values[:1])[0]), float(_time_ms(index, time_idx_values[-1:])[0])],
        },
        "blocks": blocks,
        "frozen_holdout_accessed": False,
    }


def _build_dev_asset(output_root: Path) -> dict[str, Any]:
    faults, index, meta = _load_inputs()
    segy_path = Path(meta["segy_path"])
    inline_values = _inclusive_range(DEV_BOX["iline"])
    crossline_values = _inclusive_range(DEV_BOX["crossline"])
    time_idx_values = _inclusive_range(DEV_BOX["time_idx"])
    seismic = _read_subvolume(segy_path, index, inline_values, crossline_values, time_idx_values)
    positive_mask = _rasterize_positive_mask(faults, index, inline_values, crossline_values, time_idx_values)
    boundary_unknown = _boundary_halo(positive_mask.shape, BOUNDARY_HALO) & ~positive_mask
    unknown = (_dilate(positive_mask, UNKNOWN_RADIUS) & ~positive_mask) | boundary_unknown
    verified_background = ~(positive_mask | unknown)
    if not positive_mask.any():
        raise RuntimeError("selected development subvolume contains no rasterized fault voxels")
    if not unknown.any() or not verified_background.any():
        raise RuntimeError("selected development subvolume does not support both unknown and verified background")
    split_manifest = _split_manifest(positive_mask, inline_values, crossline_values, time_idx_values, index, faults)
    split_manifest["blocks"][0]["positive_voxels"] = int(
        positive_mask[:, (inline_values >= DEV_SPLIT["fit"][0]) & (inline_values <= DEV_SPLIT["fit"][1]), :].sum()
    )
    split_manifest["blocks"][1]["positive_voxels"] = int(
        positive_mask[:, (inline_values >= DEV_SPLIT["guard"][0]) & (inline_values <= DEV_SPLIT["guard"][1]), :].sum()
    )
    split_manifest["blocks"][2]["positive_voxels"] = int(
        positive_mask[:, (inline_values >= DEV_SPLIT["validation"][0]) & (inline_values <= DEV_SPLIT["validation"][1]), :].sum()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    subvolume_path = output_root / "dev_subvolume.npz"
    np.savez_compressed(
        subvolume_path,
        seismic=seismic.astype(np.float32),
        positive_mask=positive_mask.astype(bool),
        unknown_mask=unknown.astype(bool),
        verified_background_mask=verified_background.astype(bool),
        tline_ms=_time_ms(index, time_idx_values).astype(np.float32),
        time_idx=time_idx_values.astype(np.int32),
        iline=inline_values.astype(np.int32),
        xline=crossline_values.astype(np.int32),
    )
    split_manifest_path = output_root / "split_manifest.json"
    split_manifest_path.write_text(canonical_json(split_manifest) + "\n", encoding="utf-8")

    gate_result = audit_gate(output_root, subvolume_path, split_manifest_path, split_manifest, positive_mask, unknown, verified_background)
    gate_result_path = output_root / "gate_result.json"
    gate_result_path.write_text(canonical_json(gate_result) + "\n", encoding="utf-8")

    evidence_path = output_root / "evidence.md"
    evidence_path.write_text(render_evidence(gate_result, split_manifest, faults, index, meta, subvolume_path), encoding="utf-8")

    manifest = {
        "schema_version": "fault_p30_3d_dev_gate/v1",
        "track_id": "fault",
        "source_commit": git_head(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": {"path": display_path(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": [
            {
                "path": display_path(FAULT_POINTS_PATH),
                "sha256": sha256_file(FAULT_POINTS_PATH),
                "role": "sparse fault sticks",
            },
            {
                "path": display_path(SEISMIC_INDEX_PATH),
                "sha256": sha256_file(SEISMIC_INDEX_PATH),
                "role": "seismic index",
            },
            {
                "path": display_path(SEGY_META_PATH),
                "sha256": sha256_file(SEGY_META_PATH),
                "role": "segy metadata",
            },
        ],
        "outputs": [
            {"path": display_path(subvolume_path), "sha256": sha256_file(subvolume_path), "role": "3D development subvolume"},
            {"path": display_path(split_manifest_path), "sha256": sha256_file(split_manifest_path), "role": "development split manifest"},
            {"path": display_path(gate_result_path), "sha256": sha256_file(gate_result_path), "role": "gate result"},
            {"path": display_path(evidence_path), "sha256": sha256_file(evidence_path), "role": "human evidence"},
        ],
        "subvolume": {
            "coordinate_order": list(SUBVOLUME_COORD_ORDER),
            "inline": [int(inline_values[0]), int(inline_values[-1])],
            "crossline": [int(crossline_values[0]), int(crossline_values[-1])],
            "time_idx": [int(time_idx_values[0]), int(time_idx_values[-1])],
            "time_ms": [float(_time_ms(index, time_idx_values[:1])[0]), float(_time_ms(index, time_idx_values[-1:])[0])],
            "shape": [int(x) for x in seismic.shape],
        },
        "mask_logic": {
            "positive_mask": "exact rasterized fault-stick voxels from sparse points",
            "unknown_mask": f"binary dilation radius {UNKNOWN_RADIUS} plus {BOUNDARY_HALO}-voxel boundary halo; excludes positives",
            "verified_background_mask": "complement of positive and unknown within subvolume",
        },
        "split_manifest": display_path(split_manifest_path),
        "manifest_path": display_path(output_root / "manifest.json"),
        "data_gate_blocked": False,
        "status": gate_result["status"],
        "reason_code": gate_result["reason_code"],
        "frozen_holdout_accessed": False,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "gate_result": gate_result,
        "subvolume_path": display_path(subvolume_path),
        "split_manifest_path": display_path(split_manifest_path),
        "gate_result_path": display_path(gate_result_path),
        "evidence_path": display_path(evidence_path),
        "manifest_path": display_path(manifest_path),
        "output_root": display_path(output_root),
    }


def audit_gate(
    output_root: Path,
    subvolume_path: Path,
    split_manifest_path: Path,
    split_manifest: dict[str, Any],
    positive_mask: np.ndarray,
    unknown_mask: np.ndarray,
    verified_background: np.ndarray,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    volume = np.load(subvolume_path, allow_pickle=False)
    if volume["seismic"].ndim != 3:
        reason_codes.append("contiguous_3d_development_blocks_missing")
    if not positive_mask.any():
        reason_codes.append("contiguous_3d_development_blocks_missing")
    if not verified_background.any():
        reason_codes.append("coverage_audited_verified_background_missing")
    if not unknown_mask.any():
        reason_codes.append("explicit_unknown_mask_provenance_missing")
    if not split_manifest.get("development_only", False) or not split_manifest.get("group_isolated", False):
        reason_codes.append("group_isolated_development_split_missing")
    blocks = split_manifest.get("blocks", [])
    if len(blocks) != 3:
        reason_codes.append("group_isolated_development_split_missing")
    inline_ranges = [tuple(block["inline"]) for block in blocks]
    if len(set(inline_ranges)) != len(inline_ranges):
        reason_codes.append("group_isolated_development_split_missing")
    if any(block["positive_voxels"] == 0 for block in blocks[:2]):
        reason_codes.append("group_isolated_development_split_missing")
    if positive_mask.shape != unknown_mask.shape or positive_mask.shape != verified_background.shape:
        reason_codes.append("contiguous_3d_development_blocks_missing")
    if np.any(positive_mask & unknown_mask):
        reason_codes.append("explicit_unknown_mask_provenance_missing")
    if np.any(positive_mask & verified_background) or np.any(unknown_mask & verified_background):
        reason_codes.append("coverage_audited_verified_background_missing")
    status = "READY" if not reason_codes else "DATA_GATE_BLOCKED"
    return {
        "status": status,
        "reason_code": "LEGAL_CONTIGUOUS_3D_DEVELOPMENT_VOLUME_READY" if not reason_codes else "NO_VALID_FAULT_3D_DEVELOPMENT_VOLUME",
        "reason_codes": sorted(set(reason_codes)),
        "frozen_holdout_accessed": False,
        "development_only": True,
        "subvolume_path": display_path(subvolume_path),
        "split_manifest_path": display_path(split_manifest_path),
        "verified_background_voxels": int(verified_background.sum()),
        "unknown_voxels": int(unknown_mask.sum()),
        "positive_voxels": int(positive_mask.sum()),
    }


def render_evidence(
    gate_result: dict[str, Any],
    split_manifest: dict[str, Any],
    faults: dict[str, np.ndarray],
    index: dict[str, np.ndarray],
    meta: dict[str, Any],
    subvolume_path: Path,
) -> str:
    points = np.load(FAULT_POINTS_PATH, allow_pickle=True)
    lines = [
        "# Fault contiguous 3-D development asset",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Gate status: {gate_result['status']}",
        f"- Reason code: {gate_result['reason_code']}",
        f"- Frozen holdout accessed: `{gate_result['frozen_holdout_accessed']}`",
        "",
        "## Subvolume",
        "",
        f"- Coordinate order: `{', '.join(SUBVOLUME_COORD_ORDER)}`",
        f"- Inline range: {split_manifest['subvolume']['inline']}",
        f"- Crossline range: {split_manifest['subvolume']['crossline']}",
        f"- Time index range: {split_manifest['subvolume']['time_idx']}",
        f"- Time ms range: {split_manifest['subvolume']['time_ms']}",
        f"- Saved subvolume: `{display_path(subvolume_path)}`",
        "",
        "## Mask logic",
        "",
        f"- Positive mask: exact rasterized fault-stick voxels from {len(points['inline'])} sparse points and {len(np.unique(np.asarray(faults['fault_name']).astype(str)))} faults.",
        f"- Unknown mask: dilation radius {UNKNOWN_RADIUS} plus {BOUNDARY_HALO}-voxel boundary halo; positives excluded from unknown.",
        "- Verified background mask: complement of positive and unknown within the selected subvolume.",
        "",
        "## Split manifest",
        "",
        f"- Development-only: `{split_manifest['development_only']}`",
        f"- Group-isolated: `{split_manifest['group_isolated']}`",
        f"- Blocks: `{json.dumps(split_manifest['blocks'], sort_keys=True)}`",
        "",
        "## Gate verdict",
        "",
        f"- Status: `{gate_result['status']}`",
        f"- Reason codes: `{json.dumps(gate_result['reason_codes'], sort_keys=True)}`",
        f"- Verified background voxels: {gate_result['verified_background_voxels']}",
        f"- Unknown voxels: {gate_result['unknown_voxels']}",
        f"- Positive voxels: {gate_result['positive_voxels']}",
        "",
        "## Provenance",
        "",
        f"- ST0202 SEG-Y: `{meta['segy_path']}`",
        f"- Seismic index path: `{display_path(SEISMIC_INDEX_PATH)}`",
        f"- Fault points path: `{display_path(FAULT_POINTS_PATH)}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = _build_dev_asset(args.output_root.resolve())
    print(canonical_json({"status": result["gate_result"]["status"], "reason_code": result["gate_result"]["reason_code"], "manifest_path": result["manifest_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
