#!/usr/bin/env python3
"""P37 Phase-0 audit for real-well/seismic supervision closure.

This entrypoint is deliberately fail-closed.  It reads registered Volve raw
development assets and the target-free coordinate/mask channels of the legal
``train.h5`` container.  It never accepts a test/holdout path and never starts
a model pilot.  A pilot can only be implemented after this audit reports three
independent parent wells with a common native PHIE target and a complete
MD->TVDSS->TWT->UTM/ILXL->legal-development-KJI chain.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

import h5py
import lasio
import numpy as np
from dlisio import dlis
from pypdf import PdfReader
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "_outputs/p37_real_well_seismic_supervision_closure"
DEFAULT_BUILD_SUMMARY = HERE / "build_summary.json"
DEFAULT_P21_SUMMARY = HERE / "_outputs/p21_fixed_foundation_ensemble/summary.json"
DEFAULT_P30_SUMMARY = HERE / "_outputs/p30_bounded_geostatistics_feasibility_v2/summary.json"
TEST_FILE = HERE / "test_p37_real_well_seismic_supervision.py"

SCHEMA = "reconstruction-p37-real-well-seismic-supervision/v1"
DECISION = "BLOCKED_REAL_ALIGNED_SUPERVISION"
P21_RMSE = 0.027734374378067677
P30_ORDINARY_RMSE = 0.030569516403486055
P30_REGRESSION_RMSE = 0.030093884155904194
KJI_SCALES_M = np.asarray([50.0, 50.0, 2.0], dtype=np.float64)
KJI_MAX_SCALED_DISTANCE = 6.0
COMMON_INPUTS = ("GR", "RHOB", "NPHI", "DT", "RT", "CALI")
FORBIDDEN_PATH_TOKENS = ("test.h5", "holdout", "frozen_test")


@dataclass(frozen=True)
class WellSpec:
    key: str
    parent: str
    branch: str
    target_member: str
    target_format: str
    target_curve: str
    input_member: str
    input_format: str
    input_curves: Mapping[str, str]
    survey_member: str
    checkshot_member: str


WELLS = (
    WellSpec(
        key="19A",
        parent="15/9-19",
        branch="15/9-19 A",
        target_member="Well_logs/06.LFP/15_9-19 A/159-19A_LFP.las",
        target_format="las",
        target_curve="LFP_PHIE",
        input_member="Well_logs/06.LFP/15_9-19 A/159-19A_LFP.las",
        input_format="las",
        input_curves={name: f"LFP_{name}" for name in COMMON_INPUTS},
        survey_member=(
            "Well_technical_data/WellWellbore/15_9-19/15_9-19 A/"
            "Standard Survey Report_Volve F_159-19_19 A_19 A_ACTUAL.pdf"
        ),
        checkshot_member="VSP/Checkshots/checkshot_15_9_19A.txt",
    ),
    WellSpec(
        key="F11T2",
        parent="15/9-F-11",
        branch="15/9-F-11 T2",
        target_member=(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-11 T2/"
            "WLC_PETRO_COMPUTED_OUTPUT_1.LAS"
        ),
        target_format="las",
        target_curve="PHIF",
        input_member=(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-11 T2/"
            "WLC_PETRO_COMPUTED_INPUT_1.LAS"
        ),
        input_format="las",
        input_curves={name: name for name in COMMON_INPUTS},
        survey_member=(
            "Well_technical_data/WellWellbore/15_9-F-11/15_9-F-11 T2/"
            "Standard Survey Report_Volve F_F-11_159-F-11 T2_159-F-11 T2_ACTUAL.pdf"
        ),
        checkshot_member="VSP/Checkshots/checkshot_15_9_F_11T2.ASC",
    ),
    WellSpec(
        key="F15A",
        parent="15/9-F-15",
        branch="15/9-F-15 A",
        target_member=(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-15 A/"
            "WLC_PETRO_COMPUTED_OUTPUT_1.DLIS"
        ),
        target_format="dlis",
        target_curve="PHIF",
        input_member=(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-15 A/"
            "WLC_PETRO_COMPUTED_INPUT_1.DLIS"
        ),
        input_format="dlis",
        input_curves={name: name for name in COMMON_INPUTS},
        survey_member=(
            "Well_technical_data/WellWellbore/15_9-F-15/15_9-F-15 A/"
            "Standard Survey Report_Volve F_F-15_F-15A_F-15A_ACTUAL.pdf"
        ),
        checkshot_member="VSP/Checkshots/checkshot_15_9_F_15A.txt",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _assert_development_path(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = str(resolved).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"forbidden holdout/test path: {resolved}")
    if resolved.name != "train.h5" or not resolved.is_file():
        raise RuntimeError(f"legal development train.h5 is required: {resolved}")
    return resolved


def _read_las(value: bytes) -> lasio.LASFile:
    return lasio.read(
        StringIO(value.decode("latin1", errors="replace")),
        engine="normal",
        ignore_header_errors=True,
    )


def _dlis_arrays(value: bytes) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("anonymous in-memory DLIS loading requires os.memfd_create")
    fd = os.memfd_create("p37-dlis-audit", flags=0)
    try:
        os.write(fd, value)
        os.lseek(fd, 0, os.SEEK_SET)
        with dlis.load(f"/proc/self/fd/{fd}") as files:
            logical = next(iter(files))
            if len(logical.frames) != 1:
                raise RuntimeError("P37 expects one DLIS frame for the audited CPI file")
            frame = logical.frames[0]
            curves = frame.curves(strict=False)
            arrays = {
                name: np.asarray(curves[name])
                for name in curves.dtype.names or ()
                if name != "FRAMENO"
            }
            units = {channel.name: str(channel.units or "") for channel in frame.channels}
            return arrays, units
    finally:
        os.close(fd)


def _depth_to_m(values: np.ndarray, unit: str) -> np.ndarray:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {"m", "meter", "metre"}:
        return np.asarray(values, dtype=np.float64)
    if normalized in {"0.1in", "0.1inch"}:
        return np.asarray(values, dtype=np.float64) * 0.1 * 0.0254
    if normalized in {"mm"}:
        return np.asarray(values, dtype=np.float64) / 1000.0
    raise RuntimeError(f"unsupported depth unit: {unit!r}")


def _finite_curve(values: np.ndarray, null_value: float = -999.25) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.isfinite(values) & (~np.isclose(values, null_value))


def _las_curves(value: bytes) -> tuple[dict[str, np.ndarray], dict[str, str], float]:
    las = _read_las(value)
    arrays = {curve.mnemonic: np.asarray(las[curve.mnemonic]) for curve in las.curves}
    units = {curve.mnemonic: str(curve.unit or "") for curve in las.curves}
    null_value = float(las.well.NULL.value) if "NULL" in las.well else -999.25
    return arrays, units, null_value


def _load_curves(
    archive: ZipFile,
    member: str,
    file_format: str,
) -> tuple[dict[str, np.ndarray], dict[str, str], float, str]:
    value = archive.read(member)
    if file_format == "las":
        arrays, units, null_value = _las_curves(value)
    elif file_format == "dlis":
        arrays, units = _dlis_arrays(value)
        null_value = -999.25
    else:
        raise ValueError(f"unsupported well-log format: {file_format}")
    return arrays, units, null_value, sha256_bytes(value)


def _target_profile(
    archive: ZipFile,
    spec: WellSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays, units, null_value, member_hash = _load_curves(
        archive, spec.target_member, spec.target_format
    )
    depth_name = "DEPTH"
    depth = _depth_to_m(arrays[depth_name], units[depth_name])
    target = np.asarray(arrays[spec.target_curve], dtype=np.float64)
    valid = _finite_curve(depth, null_value) & _finite_curve(target, null_value)
    valid &= (target >= 0.0) & (target <= 1.0)
    depth = depth[valid]
    target = target[valid]
    if len(depth) == 0:
        raise RuntimeError(f"no physical target rows for {spec.key}")
    order = np.argsort(depth, kind="stable")
    depth = depth[order]
    target = target[order]
    return depth, target, {
        "member": spec.target_member,
        "member_sha256": member_hash,
        "format": spec.target_format,
        "curve": spec.target_curve,
        "curve_unit": units[spec.target_curve],
        "depth_unit_source": units[depth_name],
        "depth_unit_canonical": "m_MD",
        "physical_rows": int(len(depth)),
        "md_range_m": [float(depth.min()), float(depth.max())],
        "value_range_fraction": [float(target.min()), float(target.max())],
    }


def _input_profile(archive: ZipFile, spec: WellSpec) -> dict[str, Any]:
    arrays, units, null_value, member_hash = _load_curves(
        archive, spec.input_member, spec.input_format
    )
    depth = _depth_to_m(arrays["DEPTH"], units["DEPTH"])
    curves: dict[str, Any] = {}
    for canonical, source in spec.input_curves.items():
        if source not in arrays:
            curves[canonical] = {"source_curve": source, "present": False}
            continue
        valid = _finite_curve(depth, null_value) & _finite_curve(arrays[source], null_value)
        curves[canonical] = {
            "source_curve": source,
            "source_unit": units[source],
            "present": True,
            "finite_rows": int(np.count_nonzero(valid)),
            "finite_md_range_m": (
                [float(depth[valid].min()), float(depth[valid].max())]
                if np.any(valid)
                else None
            ),
        }
    return {
        "member": spec.input_member,
        "member_sha256": member_hash,
        "format": spec.input_format,
        "depth_unit_source": units["DEPTH"],
        "curves": curves,
    }


def parent_from_branch(branch: str) -> str:
    parent = branch.upper().replace("_", "/").strip()
    return re.sub(r"\s+(A|B|C|D|T2|BT2|SR|S)$", "", parent)


def scan_target_tokens(archive: ZipFile) -> dict[str, Any]:
    tokens = (b"PHIE", b"PHIF", b"MPHE", b"EFFECTIVE POROSITY")
    parents: dict[str, dict[str, Any]] = {}
    for info in archive.infolist():
        lowered = info.filename.lower()
        if not any(
            segment in lowered
            for segment in (
                "/04.composite/",
                "/05.petrophysical interpretation/",
                "/06.lfp/",
            )
        ):
            continue
        if not lowered.endswith((".las", ".dlis", ".lis", ".asc", ".txt", ".html")):
            continue
        parts = info.filename.split("/")
        branch = None
        for position, part in enumerate(parts[:-1]):
            if part.lower() in {
                "04.composite",
                "05.petrophysical interpretation",
                "06.lfp",
            }:
                branch = parts[position + 1]
                break
        if not branch:
            continue
        found: set[str] = set()
        tail = b""
        with archive.open(info) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                upper = tail + chunk.upper()
                found.update(token.decode() for token in tokens if token in upper)
                tail = upper[-32:]
        if not found:
            continue
        parent = parent_from_branch(branch)
        record = parents.setdefault(
            parent, {"branches": set(), "tokens": set(), "files": []}
        )
        record["branches"].add(branch)
        record["tokens"].update(found)
        record["files"].append(
            {
                "member": info.filename,
                "size_bytes": int(info.file_size),
                "crc32": f"{info.CRC:08x}",
                "tokens": sorted(found),
            }
        )
    normalized = {
        parent: {
            "branches": sorted(record["branches"]),
            "tokens": sorted(record["tokens"]),
            "files": sorted(record["files"], key=lambda row: row["member"]),
        }
        for parent, record in sorted(parents.items())
    }
    return {
        "parents": normalized,
        "parent_count_with_literal_phie": sum(
            "PHIE" in record["tokens"] for record in normalized.values()
        ),
        "parent_count_with_phif": sum(
            "PHIF" in record["tokens"] for record in normalized.values()
        ),
        "literal_phie_parents": [
            parent for parent, record in normalized.items() if "PHIE" in record["tokens"]
        ],
    }


SURVEY_ROW = re.compile(
    r"^\s*" + r"\s+".join([r"[-+]?[\d,]+\.\d+"] * 10) + r"\s*$"
)
SURVEY_NUMBER = re.compile(r"[-+]?[\d,]+\.\d+")


def parse_survey_text(text: str) -> dict[str, Any]:
    rows = np.asarray(
        [
            [float(value.replace(",", "")) for value in SURVEY_NUMBER.findall(line)]
            for line in text.splitlines()
            if SURVEY_ROW.match(line)
        ],
        dtype=np.float64,
    )
    if rows.ndim != 2 or rows.shape[1] != 10 or len(rows) < 2:
        raise RuntimeError("actual survey station table was not parsed")
    _, positions = np.unique(rows[:, 0], return_index=True)
    rows = rows[np.sort(positions)]
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    if np.any(np.diff(rows[:, 0]) <= 0.0) or np.any(np.diff(rows[:, 3]) < 0.0):
        raise RuntimeError("survey MD/TVD is not monotonic")
    reference = re.search(
        r"@\s*([0-9.]+)m(?:\s*\([^)]*\))?TVD Reference", text
    )
    coordinate_pairs = re.findall(
        r"([0-9],\d{3},\d{3}\.\d+)\n(\d{3},\d{3}\.\d+)", text
    )
    if reference is None or not coordinate_pairs:
        raise RuntimeError("survey datum or site coordinate is missing")
    site_northing = float(coordinate_pairs[0][0].replace(",", ""))
    site_easting = float(coordinate_pairs[0][1].replace(",", ""))
    reference_elevation = float(reference.group(1))
    return {
        "md_m": rows[:, 0],
        "tvd_m_from_reference": rows[:, 3],
        "northing_m": site_northing + rows[:, 4],
        "easting_m": site_easting + rows[:, 5],
        "reference_elevation_m_above_msl": reference_elevation,
        "site_easting_m": site_easting,
        "site_northing_m": site_northing,
    }


def _survey_profile(archive: ZipFile, spec: WellSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    value = archive.read(spec.survey_member)
    reader = PdfReader(BytesIO(value))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    survey = parse_survey_text(text)
    profile = {
        "member": spec.survey_member,
        "member_sha256": sha256_bytes(value),
        "pages": len(reader.pages),
        "stations": int(len(survey["md_m"])),
        "md_range_m": [float(survey["md_m"].min()), float(survey["md_m"].max())],
        "tvdss_range_m": [
            float(
                survey["tvd_m_from_reference"].min()
                - survey["reference_elevation_m_above_msl"]
            ),
            float(
                survey["tvd_m_from_reference"].max()
                - survey["reference_elevation_m_above_msl"]
            ),
        ],
        "reference_elevation_m_above_msl": float(
            survey["reference_elevation_m_above_msl"]
        ),
        "coordinate_reference": "ED50 / UTM zone 31N, parsed from actual survey report",
        "md_monotonic": True,
        "tvd_monotonic": True,
    }
    return survey, profile


def _checkshot_curve(archive: ZipFile, spec: WellSpec) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    value = archive.read(spec.checkshot_member)
    number = re.compile(r"[-+]?\d+(?:\.\d+)?")
    depth: list[float] = []
    twt: list[float] = []
    for line in value.decode("latin1", errors="replace").splitlines():
        values = [float(item) for item in number.findall(line)]
        if spec.key == "19A":
            if not line.lstrip().startswith("TIME-CKS") or len(values) < 4:
                continue
            depth.append(abs(values[2]))
            twt.append(values[3])
        else:
            if len(values) != 3 or values[0] < 500.0:
                continue
            depth.append(values[1])
            twt.append(values[2])
    order = np.argsort(depth, kind="stable")
    depth_array = np.asarray(depth, dtype=np.float64)[order]
    twt_array = np.asarray(twt, dtype=np.float64)[order]
    depth_array, positions = np.unique(depth_array, return_index=True)
    twt_array = twt_array[positions]
    if len(depth_array) < 20 or np.any(np.diff(depth_array) <= 0.0):
        raise RuntimeError(f"invalid checkshot curve for {spec.key}")
    return depth_array, twt_array, {
        "member": spec.checkshot_member,
        "member_sha256": sha256_bytes(value),
        "rows": int(len(depth_array)),
        "tvdss_range_m": [float(depth_array.min()), float(depth_array.max())],
        "twt_range_ms": [float(twt_array.min()), float(twt_array.max())],
        "units": {"depth": "m_TVDSS", "time": "ms_TWT"},
    }


def _active_development_index(
    train_h5: Path,
    coordinate_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coordinates: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    patch_keys: list[str] = []
    with h5py.File(train_h5, "r") as handle:
        for key in sorted(handle):
            group = handle[key]
            metadata = json.loads(group.attrs["meta"])
            start = np.asarray(metadata["patch_start_kji"], dtype=np.int64)
            active = np.asarray(group["seismic_patch"][8], dtype=np.float32) > 0.5
            local = np.argwhere(active)
            normalized = np.asarray(group["seismic_patch"][3:6], dtype=np.float64)[
                :, active
            ].T
            coordinates.append(normalized)
            indices.append(local + start)
            patch_keys.append(key)
    normalized = np.concatenate(coordinates)
    all_indices = np.concatenate(indices)
    physical = normalized * (
        coordinate_bounds[:, 1] - coordinate_bounds[:, 0]
    ) + coordinate_bounds[:, 0]
    if len(np.unique(all_indices, axis=0)) != len(all_indices):
        raise RuntimeError("legal development active KJI index is not unique")
    return physical, all_indices, {
        "active_rows": int(len(all_indices)),
        "patch_count": len(patch_keys),
        "patch_keys_sha256": sha256_bytes("\n".join(patch_keys).encode()),
        "hdf5_files_opened": ["train.h5"],
        "datasets_read": ["seismic_patch[3:6]", "seismic_patch[8]"],
        "label_datasets_read": [],
    }


def _join_profile(
    *,
    target_md: np.ndarray,
    survey: Mapping[str, Any],
    checkshot_depth: np.ndarray,
    checkshot_twt: np.ndarray,
    seismic: Mapping[str, Any],
    coordinate_bounds: np.ndarray,
    active_tree: cKDTree,
    active_indices: np.ndarray,
) -> dict[str, Any]:
    survey_md = np.asarray(survey["md_m"], dtype=np.float64)
    survey_valid = (target_md >= survey_md.min()) & (target_md <= survey_md.max())
    md = target_md[survey_valid]
    tvdss = np.interp(md, survey_md, survey["tvd_m_from_reference"]) - float(
        survey["reference_elevation_m_above_msl"]
    )
    x = np.interp(md, survey_md, survey["easting_m"])
    y = np.interp(md, survey_md, survey["northing_m"])
    checkshot_valid = (tvdss >= checkshot_depth.min()) & (
        tvdss <= checkshot_depth.max()
    )
    tvdss = tvdss[checkshot_valid]
    x = x[checkshot_valid]
    y = y[checkshot_valid]
    twt = np.interp(tvdss, checkshot_depth, checkshot_twt)

    affine = np.asarray(seismic["affine_il_xl_to_xy"], dtype=np.float64)
    inverse = np.linalg.inv(affine[:, :2])
    ilxl = (np.column_stack([x, y]) - affine[:, 2]) @ inverse.T
    il = np.rint(ilxl[:, 0])
    xl = np.rint(ilxl[:, 1])
    seismic_valid = (
        (il >= seismic["il_min"])
        & (il <= seismic["il_max"])
        & (xl >= seismic["xl_min"])
        & (xl <= seismic["xl_max"])
        & (twt >= seismic["sample_min_ms"])
        & (twt <= seismic["sample_max_ms"])
    )
    x = x[seismic_valid]
    y = y[seismic_valid]
    tvdss = tvdss[seismic_valid]
    twt = twt[seismic_valid]
    il = il[seismic_valid]
    xl = xl[seismic_valid]
    within_full_grid = (
        (x >= coordinate_bounds[0, 0])
        & (x <= coordinate_bounds[0, 1])
        & (y >= coordinate_bounds[1, 0])
        & (y <= coordinate_bounds[1, 1])
        & (tvdss >= coordinate_bounds[2, 0])
        & (tvdss <= coordinate_bounds[2, 1])
    )
    query = np.column_stack([x, y, tvdss])
    distance, row = active_tree.query(query / KJI_SCALES_M)
    accepted = distance <= KJI_MAX_SCALED_DISTANCE
    accepted_kji = active_indices[np.asarray(row[accepted], dtype=np.int64)]
    return {
        "target_rows": int(len(target_md)),
        "md_to_tvdss_rows": int(np.count_nonzero(survey_valid)),
        "tvdss_to_twt_rows": int(np.count_nonzero(checkshot_valid)),
        "seismic_ilxl_time_rows": int(np.count_nonzero(seismic_valid)),
        "within_full_eclipse_bounds_rows": int(np.count_nonzero(within_full_grid)),
        "legal_train_kji_rows": int(np.count_nonzero(accepted)),
        "legal_train_unique_kji": int(len(np.unique(accepted_kji, axis=0))),
        "legal_train_kji_gate": {
            "scales_m": KJI_SCALES_M.tolist(),
            "maximum_scaled_distance": KJI_MAX_SCALED_DISTANCE,
        },
        "nearest_legal_train_scaled_distance": {
            "min": float(distance.min()) if len(distance) else None,
            "median": float(np.median(distance)) if len(distance) else None,
            "max": float(distance.max()) if len(distance) else None,
            "max_accepted": float(distance[accepted].max()) if np.any(accepted) else None,
        },
        "aligned_ranges": {
            "tvdss_m": [float(tvdss.min()), float(tvdss.max())] if len(tvdss) else None,
            "twt_ms": [float(twt.min()), float(twt.max())] if len(twt) else None,
            "inline": [int(il.min()), int(il.max())] if len(il) else None,
            "crossline": [int(xl.min()), int(xl.max())] if len(xl) else None,
        },
    }


def curve_equivalence_metrics(
    phie_depth: np.ndarray,
    phie: np.ndarray,
    phif_depth: np.ndarray,
    phif: np.ndarray,
) -> dict[str, Any]:
    phie_depth = np.asarray(phie_depth, dtype=np.float64)
    phie = np.asarray(phie, dtype=np.float64)
    phif_depth = np.asarray(phif_depth, dtype=np.float64)
    phif = np.asarray(phif, dtype=np.float64)
    overlap = (phif_depth >= phie_depth.min()) & (phif_depth <= phie_depth.max())
    aligned_phie = np.interp(phif_depth[overlap], phie_depth, phie)
    aligned_phif = phif[overlap]
    if len(aligned_phie) < 2:
        raise RuntimeError("too few PHIE/PHIF overlap rows")
    error = aligned_phie - aligned_phif
    return {
        "overlap_rows": int(len(error)),
        "mae_fraction": float(np.mean(np.abs(error))),
        "rmse_fraction": float(np.sqrt(np.mean(error**2))),
        "correlation": float(np.corrcoef(aligned_phie, aligned_phif)[0, 1]),
        "maximum_absolute_difference_fraction": float(np.max(np.abs(error))),
        "elementwise_equal": bool(np.array_equal(aligned_phie, aligned_phif)),
    }


def phie_phif_equivalence(archive: ZipFile) -> dict[str, Any]:
    phie_arrays, phie_units, phie_null, _ = _load_curves(
        archive, WELLS[0].target_member, "las"
    )
    phif_member = (
        "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-19 A/CPI/"
        "15_9-19_A_CPI.las"
    )
    phif_arrays, phif_units, phif_null, phif_hash = _load_curves(
        archive, phif_member, "las"
    )
    phie_depth = _depth_to_m(phie_arrays["DEPTH"], phie_units["DEPTH"])
    phie = np.asarray(phie_arrays["LFP_PHIE"], dtype=np.float64)
    phif_depth = _depth_to_m(phif_arrays["DEPTH"], phif_units["DEPTH"])
    phif = np.asarray(phif_arrays["PHIF"], dtype=np.float64)
    phie_valid = _finite_curve(phie, phie_null)
    phif_valid = _finite_curve(phif, phif_null)
    phie_depth = phie_depth[phie_valid]
    phie = phie[phie_valid]
    phif_depth = phif_depth[phif_valid]
    phif = phif[phif_valid]
    metrics = curve_equivalence_metrics(phie_depth, phie, phif_depth, phif)
    return {
        "well": "15/9-19 A",
        "phie_curve": "LFP_PHIE",
        "phif_curve": "PHIF",
        "phif_member": phif_member,
        "phif_member_sha256": phif_hash,
        **metrics,
        "conclusion": "PHIF cannot be silently renamed to PHIE",
    }


def _seismic_profile(index_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(index_path, allow_pickle=False) as payload:
        affine = np.asarray(payload["affine_il_xl_to_xy"], dtype=np.float64)
        samples = np.asarray(payload["samples_ms"], dtype=np.float64)
        values = {
            "affine_il_xl_to_xy": affine,
            "il_min": int(payload["il_min"]),
            "il_max": int(payload["il_max"]),
            "xl_min": int(payload["xl_min"]),
            "xl_max": int(payload["xl_max"]),
            "sample_min_ms": float(samples.min()),
            "sample_max_ms": float(samples.max()),
        }
        profile = {
            "index_path": str(index_path),
            "index_sha256": sha256(index_path),
            "trace_count": int(payload["n_traces"]),
            "inline_range": [values["il_min"], values["il_max"]],
            "crossline_range": [values["xl_min"], values["xl_max"]],
            "sample_rows": int(len(samples)),
            "sample_range_ms": [values["sample_min_ms"], values["sample_max_ms"]],
            "sample_interval_ms": float(np.median(np.diff(samples))),
            "affine_il_xl_to_xy": affine.tolist(),
        }
    return values, profile


def _baseline_audit(p21_path: Path, p30_path: Path) -> dict[str, Any]:
    p21 = json.loads(p21_path.read_text(encoding="utf-8"))
    p30 = json.loads(p30_path.read_text(encoding="utf-8"))
    actual_p21 = float(p21["comparison"]["candidate"]["rmse"])
    actual_ordinary = float(p30["candidates"]["anisotropic_ordinary_kriging"]["metrics"]["rmse"])
    actual_regression = float(p30["candidates"]["regression_kriging_cokriging_proxy"]["metrics"]["rmse"])
    if actual_p21 != P21_RMSE:
        raise RuntimeError("P21 baseline RMSE changed")
    if actual_ordinary != P30_ORDINARY_RMSE or actual_regression != P30_REGRESSION_RMSE:
        raise RuntimeError("P30 proxy baseline changed")
    if p30["decision"]["state"] != "FEASIBLE_NO_PROMOTION":
        raise RuntimeError("P30 decision changed")
    return {
        "p21_rmse": actual_p21,
        "p21_summary_sha256": sha256(p21_path),
        "p30_ordinary_rmse": actual_ordinary,
        "p30_regression_rmse": actual_regression,
        "p30_decision": p30["decision"]["state"],
        "p30_is_sparse_eclipse_proxy_only": True,
    }


def _commands(raw_root: Path, train_h5: Path, output_dir: Path) -> dict[str, str]:
    prefix = "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3"
    script = "_pipelines/02_task_datasets/reconstruction/p37_real_well_seismic_supervision.py"
    arguments = (
        f"--raw-project-root '{raw_root}' --train-h5 '{train_h5}' "
        f"--output-dir '{output_dir}'"
    )
    return {
        "run": f"{prefix} {script} {arguments}",
        "verify_only": f"{prefix} {script} {arguments} --verify-only",
        "py_compile": f"{prefix} -m py_compile {script}",
        "focused_tests": (
            f"{prefix} -m pytest -q "
            "_pipelines/02_task_datasets/reconstruction/_tests/"
            "test_p21_fixed_foundation_ensemble.py "
            "_pipelines/02_task_datasets/reconstruction/_tests/"
            "test_p23_checkshot_calibration.py "
            "_pipelines/02_task_datasets/reconstruction/_tests/"
            "test_p30_bounded_geostatistics_feasibility.py "
            "_pipelines/02_task_datasets/reconstruction/"
            "test_p37_real_well_seismic_supervision.py"
        ),
    }


def build_evidence(
    *,
    raw_project_root: Path,
    train_h5: Path,
    build_summary_path: Path,
    p21_summary_path: Path,
    p30_summary_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_project_root = Path(raw_project_root).expanduser().resolve()
    train_h5 = _assert_development_path(train_h5)
    logs_zip = raw_project_root / "_sandbox/volve_data/Volve_Well_logs.zip"
    technical_zip = raw_project_root / "_sandbox/volve_data/Volve_Well_technical_data.zip"
    vsp_zip = raw_project_root / "_sandbox/volve_data/Volve_Seismic_VSP.zip"
    seismic_index_path = (
        raw_project_root / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
    )
    required = (logs_zip, technical_zip, vsp_zip, seismic_index_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered development assets missing: {missing}")

    build = json.loads(build_summary_path.read_text(encoding="utf-8"))
    bounds = np.asarray(
        [
            build["coordinate_bounds"]["x"],
            build["coordinate_bounds"]["y"],
            build["coordinate_bounds"]["depth"],
        ],
        dtype=np.float64,
    )
    active_coordinates, active_indices, development_audit = _active_development_index(
        train_h5, bounds
    )
    active_tree = cKDTree(active_coordinates / KJI_SCALES_M)
    seismic, seismic_profile = _seismic_profile(seismic_index_path)

    well_profiles: dict[str, Any] = {}
    with ZipFile(logs_zip) as logs, ZipFile(technical_zip) as technical, ZipFile(
        vsp_zip
    ) as vsp:
        token_inventory = scan_target_tokens(logs)
        equivalence = phie_phif_equivalence(logs)
        for spec in WELLS:
            target_md, _, target_profile = _target_profile(logs, spec)
            input_profile = _input_profile(logs, spec)
            survey, survey_profile = _survey_profile(technical, spec)
            checkshot_depth, checkshot_twt, checkshot_profile = _checkshot_curve(vsp, spec)
            join = _join_profile(
                target_md=target_md,
                survey=survey,
                checkshot_depth=checkshot_depth,
                checkshot_twt=checkshot_twt,
                seismic=seismic,
                coordinate_bounds=bounds,
                active_tree=active_tree,
                active_indices=active_indices,
            )
            well_profiles[spec.key] = {
                "parent_well": spec.parent,
                "branch": spec.branch,
                "target": target_profile,
                "inputs": input_profile,
                "actual_survey": survey_profile,
                "checkshot": checkshot_profile,
                "alignment_join": join,
            }

    common_inputs = [
        name
        for name in COMMON_INPUTS
        if all(
            well_profiles[spec.key]["inputs"]["curves"][name]["present"]
            for spec in WELLS
        )
    ]
    parents = sorted({spec.parent for spec in WELLS})
    literal_phie_parents = token_inventory["literal_phie_parents"]
    legal_kji_parents = sorted(
        {
            profile["parent_well"]
            for profile in well_profiles.values()
            if profile["alignment_join"]["legal_train_kji_rows"] > 0
        }
    )
    supervision_gate = {
        "independent_parent_wells_required": 3,
        "independent_parent_wells_audited": len(parents),
        "independent_parent_well_gate_passed": len(parents) >= 3,
        "literal_native_phie_parent_wells_required": 3,
        "literal_native_phie_parent_wells_available": len(literal_phie_parents),
        "literal_native_phie_parents": literal_phie_parents,
        "common_phie_gate_passed": len(literal_phie_parents) >= 3,
        "common_legal_input_curves": common_inputs,
        "common_legal_input_gate_passed": len(common_inputs) == len(COMMON_INPUTS),
        "parents_with_auditable_actual_survey_and_checkshot": parents,
        "parents_with_legal_train_kji": legal_kji_parents,
        "full_alignment_for_all_parent_wells_passed": len(legal_kji_parents) == len(parents),
    }
    supervision_closes = all(
        (
            supervision_gate["independent_parent_well_gate_passed"],
            supervision_gate["common_phie_gate_passed"],
            supervision_gate["common_legal_input_gate_passed"],
            supervision_gate["full_alignment_for_all_parent_wells_passed"],
        )
    )
    if supervision_closes:
        raise RuntimeError(
            "Phase-0 supervision unexpectedly closes; this fail-closed evidence runner "
            "must be reviewed before any pilot implementation"
        )

    baseline = _baseline_audit(p21_summary_path, p30_summary_path)
    provenance = {
        "well_logs_zip_sha256": sha256(logs_zip),
        "well_technical_data_zip_sha256": sha256(technical_zip),
        "seismic_vsp_zip_sha256": sha256(vsp_zip),
        "seismic_index_sha256": sha256(seismic_index_path),
        "train_h5_sha256": sha256(train_h5),
        "build_summary_sha256": sha256(build_summary_path),
        "p21_summary_sha256": sha256(p21_summary_path),
        "p30_summary_sha256": sha256(p30_summary_path),
        "script_sha256": sha256(Path(__file__)),
        "test_sha256": sha256(TEST_FILE),
    }
    blockers = [
        {
            "code": "COMMON_NATIVE_PHIE_PARENT_COUNT_LT_3",
            "evidence": (
                f"Only {len(literal_phie_parents)} independent parent well has native PHIE; "
                "F11T2 and F15A publish PHIF. The same-well PHIE/PHIF audit is non-equivalent."
            ),
        },
        {
            "code": "LEGAL_DEVELOPMENT_KJI_PARENT_COUNT_LT_3",
            "evidence": (
                f"Only {len(legal_kji_parents)} parent wells have any target row within the "
                "existing legal train.h5 active-KJI distance gate; F11T2 has zero."
            ),
        },
    ]
    summary = {
        "schema_version": SCHEMA,
        "phase": "P37_PHASE_0_SUPERVISION_CLOSURE",
        "decision": {
            "state": DECISION,
            "supervision_closes": False,
            "pilot_run": False,
            "promotion_evaluated": False,
            "p21_remains_default": True,
            "eclipse_grid_proxy_substituted": False,
        },
        "supervision_gate": supervision_gate,
        "well_profiles": well_profiles,
        "phie_phif_same_well_audit": equivalence,
        "seismic_index": seismic_profile,
        "legal_development_coordinate_index": development_audit,
        "coordinate_bounds_m": bounds.tolist(),
        "baseline_preservation": baseline,
        "blockers": blockers,
        "firewall": {
            "development_hdf5_opened": ["train.h5"],
            "hdf5_datasets_read": development_audit["datasets_read"],
            "hdf5_label_datasets_read": [],
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "target_statistics_used_for_training": False,
            "normalization_or_encoder_adaptation_run": False,
            "training_run": False,
        },
        "provenance": provenance,
        "rerun_commands_file": "rerun_commands.json",
    }
    inventory = {
        "schema_version": "reconstruction-p37-real-well-asset-inventory/v1",
        "registered_raw_root": str(raw_project_root),
        "target_token_inventory": token_inventory,
        "selected_parent_wells": parents,
        "selected_well_profiles": {
            key: {
                "parent_well": value["parent_well"],
                "branch": value["branch"],
                "target": value["target"],
                "inputs": value["inputs"],
                "actual_survey": value["actual_survey"],
                "checkshot": value["checkshot"],
            }
            for key, value in well_profiles.items()
        },
        "license": "Equinor Open Data Licence",
        "raw_archives_modified": False,
    }
    split_manifest = {
        "schema_version": "reconstruction-p37-parent-well-split/v1",
        "state": "FROZEN_NOT_ACTIVATED",
        "frozen_parent_well_order": parents,
        "planned_leave_one_parent_well_out_folds": [
            {
                "fold_id": fold_id,
                "held_parent_well": parent,
                "training_parent_wells": [item for item in parents if item != parent],
            }
            for fold_id, parent in enumerate(parents)
        ],
        "split_before_normalization": True,
        "split_before_encoder_adaptation": True,
        "split_before_caching": True,
        "held_parent_phie_must_be_fully_masked": True,
        "activation_gate": "three native-PHIE parent wells and full legal alignment",
        "activation_decision": False,
        "reason": DECISION,
        "frozen_test_or_holdout_member": None,
    }
    commands = _commands(raw_project_root, train_h5, Path(output_dir))
    return summary, inventory, split_manifest, commands


def _verification(summary: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "decision_is_blocked_real_aligned_supervision": summary["decision"]["state"] == DECISION,
        "pilot_not_run": summary["decision"]["pilot_run"] is False,
        "p21_remains_default": summary["decision"]["p21_remains_default"] is True,
        "no_eclipse_proxy_substitution": summary["decision"]["eclipse_grid_proxy_substituted"] is False,
        "exactly_three_independent_parents_audited": summary["supervision_gate"]["independent_parent_wells_audited"] == 3,
        "native_phie_parent_count_is_one": summary["supervision_gate"]["literal_native_phie_parent_wells_available"] == 1,
        "phie_phif_not_elementwise_equal": summary["phie_phif_same_well_audit"]["elementwise_equal"] is False,
        "legal_train_kji_parent_count_is_two": len(summary["supervision_gate"]["parents_with_legal_train_kji"]) == 2,
        "f11t2_legal_train_kji_rows_is_zero": summary["well_profiles"]["F11T2"]["alignment_join"]["legal_train_kji_rows"] == 0,
        "p21_rmse_preserved": summary["baseline_preservation"]["p21_rmse"] == P21_RMSE,
        "p30_decision_preserved": summary["baseline_preservation"]["p30_decision"] == "FEASIBLE_NO_PROMOTION",
        "only_train_h5_opened": summary["firewall"]["development_hdf5_opened"] == ["train.h5"],
        "no_hdf5_labels_read": summary["firewall"]["hdf5_label_datasets_read"] == [],
        "frozen_holdout_not_opened": summary["firewall"]["frozen_holdout_opened"] is False,
        "inventory_raw_archives_read_only": inventory["raw_archives_modified"] is False,
    }
    return {
        "schema_version": "reconstruction-p37-supervision-verification/v1",
        "status": "PASS_BLOCKED_EVIDENCE" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def _finding(summary: Mapping[str, Any]) -> str:
    ph = summary["phie_phif_same_well_audit"]
    joins = summary["well_profiles"]
    return "\n".join(
        [
            "# P37 real-well–seismic supervision closure",
            "",
            f"Decision: `{summary['decision']['state']}`. No pilot was run and P21 remains default.",
            "",
            "## Evidence",
            "",
            "- Three independent parent wells were audited: 15/9-19, 15/9-F-11, and 15/9-F-15.",
            "- Only 15/9-19 contains native `PHIE`. F11T2 and F15A publish `PHIF`, so the required common PHIE target is absent.",
            (
                f"- On 15/9-19 A, aligned PHIE versus PHIF has {ph['overlap_rows']} rows, "
                f"MAE {ph['mae_fraction']:.12f}, RMSE {ph['rmse_fraction']:.12f}, "
                f"correlation {ph['correlation']:.12f}, and max absolute difference "
                f"{ph['maximum_absolute_difference_fraction']:.12f}; PHIF is not a legal silent alias for PHIE."
            ),
            (
                "- Actual survey PDFs and direct checkshots close MD→TVDSS→TWT→UTM/ILXL, "
                "but the existing legal train.h5 active-KJI gate retains "
                f"{joins['19A']['alignment_join']['legal_train_kji_rows']} 19A rows, "
                f"{joins['F11T2']['alignment_join']['legal_train_kji_rows']} F11T2 rows, and "
                f"{joins['F15A']['alignment_join']['legal_train_kji_rows']} F15A rows."
            ),
            "- F11T2 has zero legal development KJI rows. Opening frozen test/holdout geometry to recover it is forbidden.",
            "- P30 remains sparse Eclipse-grid proxy history only and was not substituted for missing real-well supervision.",
            "",
            "## Consequence",
            "",
            "The minimum pilot gate fails twice: fewer than three native-PHIE parent wells, and fewer than three parent wells with legal development KJI support. No normalization, encoder adaptation, cache generation, training, agent action, calibration, ablation, or promotion test was run.",
            "",
            "Exact commands are in `rerun_commands.json`; machine checks are in `verification.json` and hashes in `artifact_manifest.json`.",
            "",
        ]
    )


def _artifact_manifest(output_dir: Path, provenance: Mapping[str, str]) -> dict[str, Any]:
    artifact_names = (
        "summary.json",
        "asset_inventory.json",
        "split_manifest.json",
        "verification.json",
        "rerun_commands.json",
        "finding.md",
    )
    return {
        "schema_version": "reconstruction-p37-artifact-manifest/v1",
        "artifacts": [
            {
                "path": name,
                "size_bytes": int((output_dir / name).stat().st_size),
                "sha256": sha256(output_dir / name),
            }
            for name in artifact_names
        ],
        "input_and_source_sha256": dict(sorted(provenance.items())),
    }


def write_evidence(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    inventory: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    commands: Mapping[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = _verification(summary, inventory)
    if verification["status"] != "PASS_BLOCKED_EVIDENCE":
        raise RuntimeError("P37 blocked evidence verification failed")
    payloads = {
        "summary.json": canonical_json(summary),
        "asset_inventory.json": canonical_json(inventory),
        "split_manifest.json": canonical_json(split_manifest),
        "verification.json": canonical_json(verification),
        "rerun_commands.json": canonical_json(commands),
        "finding.md": _finding(summary),
    }
    for name, value in payloads.items():
        (output_dir / name).write_text(value, encoding="utf-8")
    manifest = _artifact_manifest(output_dir, summary["provenance"])
    (output_dir / "artifact_manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )


def verify_existing(
    *,
    output_dir: Path,
    expected: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, str]],
) -> None:
    summary, inventory, split_manifest, commands = expected
    expected_payloads = {
        "summary.json": canonical_json(summary),
        "asset_inventory.json": canonical_json(inventory),
        "split_manifest.json": canonical_json(split_manifest),
        "verification.json": canonical_json(_verification(summary, inventory)),
        "rerun_commands.json": canonical_json(commands),
        "finding.md": _finding(summary),
    }
    for name, value in expected_payloads.items():
        actual = (output_dir / name).read_text(encoding="utf-8")
        if actual != value:
            raise RuntimeError(f"P37 artifact does not reproduce exactly: {name}")
    expected_manifest = _artifact_manifest(output_dir, summary["provenance"])
    actual_manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if actual_manifest != expected_manifest:
        raise RuntimeError("P37 artifact manifest hash verification failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-project-root", type=Path, required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, default=DEFAULT_BUILD_SUMMARY)
    parser.add_argument("--p21-summary", type=Path, default=DEFAULT_P21_SUMMARY)
    parser.add_argument("--p30-summary", type=Path, default=DEFAULT_P30_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence = build_evidence(
        raw_project_root=args.raw_project_root,
        train_h5=args.train_h5,
        build_summary_path=args.build_summary,
        p21_summary_path=args.p21_summary,
        p30_summary_path=args.p30_summary,
        output_dir=args.output_dir,
    )
    if args.verify_only:
        verify_existing(output_dir=args.output_dir, expected=evidence)
        print(canonical_json({"status": "VERIFIED", "decision": DECISION}), end="")
        return 0
    summary, inventory, split_manifest, commands = evidence
    write_evidence(
        output_dir=args.output_dir,
        summary=summary,
        inventory=inventory,
        split_manifest=split_manifest,
        commands=commands,
    )
    print(
        canonical_json(
            {
                "decision": summary["decision"],
                "supervision_gate": summary["supervision_gate"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
