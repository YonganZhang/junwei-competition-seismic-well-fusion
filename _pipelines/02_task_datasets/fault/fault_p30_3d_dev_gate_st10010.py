#!/usr/bin/env python3
"""Build a contiguous 3-D development asset for the fault track from ST10010.

This is the ST10010 companion to fault_p30_3d_dev_gate.py. It uses the
KIRCH_FULL_T stack from the ST10010 seismic archive, extracts only the needed
stack file if it is not already cached locally, and then builds the same
continuous development subvolume / masks / split manifest on the ST10010 time
grid. The original ST0202 asset is left untouched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import segyio

from build_dataset import nearest_time_indices, rasterize_fault_voxels, sha256_file

try:  # pragma: no cover - optional dependency path
    from scipy.ndimage import binary_dilation
except Exception:  # pragma: no cover - fallback for minimal environments
    binary_dilation = None


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p30_3d_dev_gate_st10010"

ST10010_ZIP_PATH = PROJECT_ROOT / "_sandbox/volve_data/Volve_Seismic_ST10010.zip"
ST10010_STACK_MEMBER = (
    "ST10010/Stacks/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy"
)
EXTRACTED_SEGY_PATH = (
    PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_seismic/ST10010/Stacks/"
    / Path(ST10010_STACK_MEMBER).name
)

SEISMIC_INDEX_NPZ = OUTPUT_ROOT / "seismic_index.npz"
SEISMIC_INDEX_META = OUTPUT_ROOT / "seismic_index_meta.json"
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


def optional_sha256_file(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


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


def ensure_extracted_segy() -> Path:
    if EXTRACTED_SEGY_PATH.exists():
        return EXTRACTED_SEGY_PATH
    if not ST10010_ZIP_PATH.exists():
        raise FileNotFoundError(f"missing ST10010 seismic archive: {ST10010_ZIP_PATH}")
    EXTRACTED_SEGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ST10010_ZIP_PATH) as archive:
        try:
            with archive.open(ST10010_STACK_MEMBER) as src, EXTRACTED_SEGY_PATH.open("wb") as dst:
                dst.write(src.read())
        except KeyError as exc:  # pragma: no cover - archive integrity issue
            raise FileNotFoundError(f"member not found in ST10010 archive: {ST10010_STACK_MEMBER}") from exc
    return EXTRACTED_SEGY_PATH


@dataclass(frozen=True)
class SeismicIndex:
    il_min: int
    il_max: int
    xl_min: int
    xl_max: int
    n_il: int
    n_xl: int
    n_traces: int
    samples_ms: np.ndarray
    trace_map: np.ndarray

    def to_npz_kwargs(self) -> dict[str, np.ndarray]:
        return {
            "il_min": np.asarray(self.il_min, dtype=np.int32),
            "il_max": np.asarray(self.il_max, dtype=np.int32),
            "xl_min": np.asarray(self.xl_min, dtype=np.int32),
            "xl_max": np.asarray(self.xl_max, dtype=np.int32),
            "n_il": np.asarray(self.n_il, dtype=np.int32),
            "n_xl": np.asarray(self.n_xl, dtype=np.int32),
            "n_traces": np.asarray(self.n_traces, dtype=np.int32),
            "samples_ms": np.asarray(self.samples_ms, dtype=np.float64),
            "trace_map": np.asarray(self.trace_map, dtype=np.int32),
        }

    def to_meta(self, segy_path: Path, missing_traces: int) -> dict[str, Any]:
        return {
            "segy_path": str(segy_path),
            "stack_member": ST10010_STACK_MEMBER,
            "il_range": [self.il_min, self.il_max],
            "xl_range": [self.xl_min, self.xl_max],
            "n_traces": self.n_traces,
            "n_samples": len(self.samples_ms),
            "sample_interval_ms": float(self.samples_ms[1] - self.samples_ms[0]),
            "missing_traces": missing_traces,
            "note": "trace_map indexes (inline, crossline) directly into the extracted ST10010 stack",
        }


def build_index(segy_path: Path) -> SeismicIndex:
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        il = f.attributes(segyio.TraceField.INLINE_3D)[:].astype(np.int32)
        xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:].astype(np.int32)
        samples = np.asarray(f.samples, dtype=np.float64)

    il_min, il_max = int(il.min()), int(il.max())
    xl_min, xl_max = int(xl.min()), int(xl.max())
    n_il = il_max - il_min + 1
    n_xl = xl_max - xl_min + 1
    trace_map = np.full((n_il, n_xl), -1, dtype=np.int32)
    il_off = il - il_min
    xl_off = xl - xl_min
    trace_indices = np.arange(len(il), dtype=np.int32)
    trace_map[il_off, xl_off] = trace_indices
    missing = int(np.sum(trace_map < 0))
    return SeismicIndex(
        il_min=il_min,
        il_max=il_max,
        xl_min=xl_min,
        xl_max=xl_max,
        n_il=n_il,
        n_xl=n_xl,
        n_traces=len(il),
        samples_ms=samples,
        trace_map=trace_map,
    ), missing


def load_or_build_index(segy_path: Path) -> tuple[SeismicIndex, dict[str, Any]]:
    if SEISMIC_INDEX_NPZ.exists() and SEISMIC_INDEX_META.exists():
        z = np.load(SEISMIC_INDEX_NPZ, allow_pickle=False)
        index = SeismicIndex(
            il_min=int(z["il_min"]),
            il_max=int(z["il_max"]),
            xl_min=int(z["xl_min"]),
            xl_max=int(z["xl_max"]),
            n_il=int(z["n_il"]),
            n_xl=int(z["n_xl"]),
            n_traces=int(z["n_traces"]),
            samples_ms=np.asarray(z["samples_ms"], dtype=np.float64),
            trace_map=np.asarray(z["trace_map"], dtype=np.int32),
        )
        return index, _load_json(SEISMIC_INDEX_META)
    index, missing = build_index(segy_path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SEISMIC_INDEX_NPZ, **index.to_npz_kwargs())
    meta = index.to_meta(segy_path, missing)
    SEISMIC_INDEX_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index, meta


def inclusive(bounds: tuple[int, int]) -> np.ndarray:
    return np.arange(bounds[0], bounds[1] + 1, dtype=np.int32)


def time_ms(index: SeismicIndex, time_idx: np.ndarray) -> np.ndarray:
    return np.asarray(index.samples_ms, dtype=np.float64)[time_idx.astype(np.int32)]


def read_subvolume(
    segy_path: Path,
    index: SeismicIndex,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    time_idx_values: np.ndarray,
) -> np.ndarray:
    t0 = int(time_idx_values[0])
    t1 = int(time_idx_values[-1])
    cube = np.empty((len(time_idx_values), len(inline_values), len(crossline_values)), dtype=np.float32)
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        for il_i, inline in enumerate(inline_values.tolist()):
            il_off = int(inline) - index.il_min
            for xl_i, crossline in enumerate(crossline_values.tolist()):
                trace_index = int(index.trace_map[il_off, int(crossline) - index.xl_min])
                if trace_index < 0:
                    raise RuntimeError(
                        f"ST10010 missing trace at inline={inline}, crossline={crossline} within the requested box"
                    )
                trace = np.asarray(f.trace[trace_index][t0 : t1 + 1], dtype=np.float32)
                if trace.shape[0] != len(time_idx_values):
                    raise RuntimeError("unexpected trace length while extracting ST10010 development volume")
                cube[:, il_i, xl_i] = trace
    if not np.isfinite(cube).all():
        raise RuntimeError("extracted seismic subvolume contains non-finite values")
    return cube


def dilate(mask: np.ndarray, radius: dict[str, int]) -> np.ndarray:
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


def boundary_halo(shape: tuple[int, int, int], halo: int) -> np.ndarray:
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


def split_manifest(
    positive_mask: np.ndarray,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    time_idx_values: np.ndarray,
    index: SeismicIndex,
    faults: dict[str, np.ndarray],
) -> dict[str, Any]:
    faults_inline = np.asarray(faults["inline"], dtype=np.int32)
    faults_xline = np.asarray(faults["crossline"], dtype=np.int32)
    faults_tidx = nearest_time_indices(index.samples_ms, faults["twt_ms"]).astype(np.int32)
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
        group_mask = (inline_values >= il0) & (inline_values <= il1)
        blocks.append(
            {
                "name": name,
                "inline": [int(il0), int(il1)],
                "n_inline": int(group_mask.sum()),
                "positive_voxels": int(positive_mask[:, group_mask, :].sum()),
                "fault_point_count": int(points.sum()),
            }
        )
    return {
        "schema_version": "fault_p30_3d_dev_gate_st10010/split/v1",
        "track_id": "fault",
        "development_only": True,
        "group_isolated": True,
        "coordinate_order": list(SUBVOLUME_COORD_ORDER),
        "subvolume": {
            "inline": [int(inline_values[0]), int(inline_values[-1])],
            "crossline": [int(crossline_values[0]), int(crossline_values[-1])],
            "time_idx": [int(time_idx_values[0]), int(time_idx_values[-1])],
            "time_ms": [float(time_ms(index, time_idx_values[:1])[0]), float(time_ms(index, time_idx_values[-1:])[0])],
        },
        "blocks": blocks,
        "frozen_holdout_accessed": False,
    }


def gate_result(
    subvolume_path: Path,
    split_manifest_path: Path,
    split_manifest_obj: dict[str, Any],
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
    if not split_manifest_obj.get("development_only", False) or not split_manifest_obj.get("group_isolated", False):
        reason_codes.append("group_isolated_development_split_missing")
    blocks = split_manifest_obj.get("blocks", [])
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
    gate: dict[str, Any],
    split_manifest_obj: dict[str, Any],
    faults: dict[str, np.ndarray],
    index: SeismicIndex,
    source_meta: dict[str, Any],
    subvolume_path: Path,
) -> str:
    points = np.load(FAULT_POINTS_PATH, allow_pickle=True)
    lines = [
        "# Fault contiguous 3-D development asset from ST10010",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Gate status: {gate['status']}",
        f"- Reason code: {gate['reason_code']}",
        f"- Frozen holdout accessed: `{gate['frozen_holdout_accessed']}`",
        "",
        "## Source stack",
        "",
        f"- Seismic archive: `{display_path(ST10010_ZIP_PATH)}`",
        f"- Extracted stack: `{display_path(EXTRACTED_SEGY_PATH)}`",
        f"- Stack member: `{ST10010_STACK_MEMBER}`",
        f"- Input sample interval ms: {float(index.samples_ms[1] - index.samples_ms[0])}",
        f"- Input time range ms: [{float(index.samples_ms[0])}, {float(index.samples_ms[-1])}]",
        "",
        "## Subvolume",
        "",
        f"- Coordinate order: `{', '.join(SUBVOLUME_COORD_ORDER)}`",
        f"- Inline range: {split_manifest_obj['subvolume']['inline']}",
        f"- Crossline range: {split_manifest_obj['subvolume']['crossline']}",
        f"- Time index range: {split_manifest_obj['subvolume']['time_idx']}",
        f"- Time ms range: {split_manifest_obj['subvolume']['time_ms']}",
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
        f"- Development-only: `{split_manifest_obj['development_only']}`",
        f"- Group-isolated: `{split_manifest_obj['group_isolated']}`",
        f"- Blocks: `{json.dumps(split_manifest_obj['blocks'], sort_keys=True)}`",
        "",
        "## Gate verdict",
        "",
        f"- Status: `{gate['status']}`",
        f"- Reason codes: `{json.dumps(gate['reason_codes'], sort_keys=True)}`",
        f"- Verified background voxels: {gate['verified_background_voxels']}",
        f"- Unknown voxels: {gate['unknown_voxels']}",
        f"- Positive voxels: {gate['positive_voxels']}",
    ]
    return "\n".join(lines) + "\n"


def build_dev_asset(output_root: Path) -> dict[str, Any]:
    with np.load(FAULT_POINTS_PATH, allow_pickle=True) as z:
        faults = {key: z[key] for key in z.files}
    segy_path = ensure_extracted_segy()
    index, index_meta = load_or_build_index(segy_path)
    inline_values = inclusive(DEV_BOX["iline"])
    crossline_values = inclusive(DEV_BOX["crossline"])
    time_idx_values = inclusive(DEV_BOX["time_idx"])
    seismic = read_subvolume(segy_path, index, inline_values, crossline_values, time_idx_values)
    voxels = rasterize_fault_voxels(
        faults,
        {
            "samples_ms": index.samples_ms,
            "il_min": index.il_min,
            "il_max": index.il_max,
            "xl_min": index.xl_min,
            "xl_max": index.xl_max,
        },
    )
    # Rasterized voxels are global; clip them to the requested subvolume.
    inside = (
        (voxels[:, 0] >= int(inline_values[0]))
        & (voxels[:, 0] <= int(inline_values[-1]))
        & (voxels[:, 1] >= int(crossline_values[0]))
        & (voxels[:, 1] <= int(crossline_values[-1]))
        & (voxels[:, 2] >= int(time_idx_values[0]))
        & (voxels[:, 2] <= int(time_idx_values[-1]))
    )
    local = voxels[inside].astype(np.int32, copy=False)
    positive = np.zeros((len(time_idx_values), len(inline_values), len(crossline_values)), dtype=bool)
    if local.size:
        positive[local[:, 2] - int(time_idx_values[0]), local[:, 0] - int(inline_values[0]), local[:, 1] - int(crossline_values[0])] = True
    boundary_unknown = boundary_halo(positive.shape, BOUNDARY_HALO) & ~positive
    unknown = (dilate(positive, UNKNOWN_RADIUS) & ~positive) | boundary_unknown
    verified_background = ~(positive | unknown)
    if not positive.any():
        raise RuntimeError("selected ST10010 development subvolume contains no rasterized fault voxels")
    if not unknown.any() or not verified_background.any():
        raise RuntimeError("selected ST10010 development subvolume does not support both unknown and verified background")
    split_manifest_obj = split_manifest(positive, inline_values, crossline_values, time_idx_values, index, faults)
    output_root.mkdir(parents=True, exist_ok=True)
    subvolume_path = output_root / "dev_subvolume.npz"
    np.savez_compressed(
        subvolume_path,
        seismic=seismic.astype(np.float32),
        positive_mask=positive.astype(bool),
        unknown_mask=unknown.astype(bool),
        verified_background_mask=verified_background.astype(bool),
        tline_ms=time_ms(index, time_idx_values).astype(np.float32),
        time_idx=time_idx_values.astype(np.int32),
        iline=inline_values.astype(np.int32),
        xline=crossline_values.astype(np.int32),
    )
    split_manifest_path = output_root / "split_manifest.json"
    split_manifest_path.write_text(canonical_json(split_manifest_obj) + "\n", encoding="utf-8")
    gate = gate_result(subvolume_path, split_manifest_path, split_manifest_obj, positive, unknown, verified_background)
    gate_path = output_root / "gate_result.json"
    gate_path.write_text(canonical_json(gate) + "\n", encoding="utf-8")
    evidence_path = output_root / "evidence.md"
    evidence_path.write_text(render_evidence(gate, split_manifest_obj, faults, index, index_meta, subvolume_path), encoding="utf-8")
    manifest = {
        "schema_version": "fault_p30_3d_dev_gate_st10010/v1",
        "track_id": "fault",
        "source_commit": git_head(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": {"path": display_path(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": [
            {
                "path": display_path(ST10010_ZIP_PATH),
                "sha256": optional_sha256_file(ST10010_ZIP_PATH),
                "role": "canonical ST10010 seismic archive",
                "status": "present_unhashed_large_archive",
            },
            {"path": display_path(EXTRACTED_SEGY_PATH), "sha256": sha256_file(EXTRACTED_SEGY_PATH), "role": "extracted ST10010 stack member"},
            {"path": display_path(FAULT_POINTS_PATH), "sha256": sha256_file(FAULT_POINTS_PATH), "role": "sparse fault sticks"},
        ],
        "outputs": [
            {"path": display_path(SEISMIC_INDEX_NPZ), "sha256": sha256_file(SEISMIC_INDEX_NPZ), "role": "seismic index"},
            {"path": display_path(SEISMIC_INDEX_META), "sha256": sha256_file(SEISMIC_INDEX_META), "role": "seismic index metadata"},
            {"path": display_path(subvolume_path), "sha256": sha256_file(subvolume_path), "role": "3D development subvolume"},
            {"path": display_path(split_manifest_path), "sha256": sha256_file(split_manifest_path), "role": "development split manifest"},
            {"path": display_path(gate_path), "sha256": sha256_file(gate_path), "role": "gate result"},
            {"path": display_path(evidence_path), "sha256": sha256_file(evidence_path), "role": "human evidence"},
        ],
        "subvolume": {
            "coordinate_order": list(SUBVOLUME_COORD_ORDER),
            "inline": [int(inline_values[0]), int(inline_values[-1])],
            "crossline": [int(crossline_values[0]), int(crossline_values[-1])],
            "time_idx": [int(time_idx_values[0]), int(time_idx_values[-1])],
            "time_ms": [float(time_ms(index, time_idx_values[:1])[0]), float(time_ms(index, time_idx_values[-1:])[0])],
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
        "status": gate["status"],
        "reason_code": gate["reason_code"],
        "frozen_holdout_accessed": False,
        "source_stack_member": ST10010_STACK_MEMBER,
        "source_stack_sha256": sha256_file(EXTRACTED_SEGY_PATH),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "gate_result": gate,
        "subvolume_path": display_path(subvolume_path),
        "split_manifest_path": display_path(split_manifest_path),
        "gate_result_path": display_path(gate_path),
        "evidence_path": display_path(evidence_path),
        "manifest_path": display_path(manifest_path),
        "output_root": display_path(output_root),
        "index_path": display_path(SEISMIC_INDEX_NPZ),
        "index_meta_path": display_path(SEISMIC_INDEX_META),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_dev_asset(args.output_root.resolve())
    print(
        canonical_json(
            {
                "status": result["gate_result"]["status"],
                "reason_code": result["gate_result"]["reason_code"],
                "manifest_path": result["manifest_path"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
