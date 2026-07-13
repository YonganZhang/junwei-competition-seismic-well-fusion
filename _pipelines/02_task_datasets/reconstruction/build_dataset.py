#!/usr/bin/env python3
"""Build the Volve 3-D porosity reconstruction task.

The dense target is the final PORO keyword from the Eclipse simulator INIT file,
mapped through the final GRID file.  The RMS geomodel is used as an independent
reference check: its GEOMATIC ``merge_pp04b_PHIF_NW`` payload contains exactly
the same multiset of non-zero porosity values as the Eclipse ASCII export.

Inputs are actual post-stack seismic attributes sampled at the Eclipse cell
centres plus sparse porosity observations along the three Layer-1 weak well
ties.  The global well table crosses the train/test label boundary, so the
downstream evaluation is conditional reconstruction given test-region well
constraints, not strict spatial holdout generalization.  The weak ties use
measured depth as their depth coordinate, so the depth-to-time conversion is
intentionally labelled approximate throughout.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator
from zipfile import ZipFile

import numpy as np
import segyio
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from ml_framework.preprocess import (  # noqa: E402
    denoise_identity,
    denormalize,
    fit_minmax,
    normalize,
)

VOLVE_DIR = PROJECT_ROOT / "_sandbox" / "volve_data"
ECLIPSE_ZIP = VOLVE_DIR / "Volve_Reservoir_Model-Eclipse_model.zip"
RMS_ZIP = VOLVE_DIR / "Volve_Reservoir_Model-RMS_model.zip"
LAYER1_OUT = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs"
SEISMIC_INDEX = LAYER1_OUT / "seismic_index.npz"
SEISMIC_META = LAYER1_OUT / "seismic_index_meta.json"
WELL_TIE = LAYER1_OUT / "well_tie_weak.npz"
SEGY_PATH = (
    VOLVE_DIR
    / "_extracted_seismic/ST0202/Stacks/"
    "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
)

ECLIPSE_BASE = "Reservoir_Model-Eclipse_model/Volve_sim_model_PPA-Eclipse Res Model/"
ECLIPSE_GRID_MEMBER = ECLIPSE_BASE + "VOLVE_2016.GRID"
ECLIPSE_INIT_MEMBER = ECLIPSE_BASE + "VOLVE_2016.INIT"
ECLIPSE_ASCII_PORO_MEMBER = ECLIPSE_BASE + "PHIF_NW"
RMS_PORO_MEMBER = (
    "Reservoir_Model-RMS_model/"
    "Volve_Full_Field_Geomodel_2014 (pp04xg03gf01sf01.rms2012.0)/"
    "gridmodels/pp04bxg03postf11a/mergepp04bphifnw/realisation"
)

PATCH_SHAPE = (9, 20, 18)  # k, j, i; exactly tiles 63 x 100 x 108
CHANNEL_NAMES = [
    "seismic_amplitude",
    "seismic_local_rms",
    "seismic_vertical_gradient",
    "x_normalized",
    "y_normalized",
    "depth_normalized",
    "sparse_well_porosity",
    "sparse_well_mask",
    "eclipse_active_mask",
]


@dataclass
class ReferenceVolume:
    shape: tuple[int, int, int]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    active: np.ndarray
    porosity: np.ndarray
    inspection: dict


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected EOF while reading {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_fortran_record(stream: BinaryIO) -> bytes | None:
    marker = stream.read(4)
    if not marker:
        return None
    if len(marker) != 4:
        raise ValueError("truncated Eclipse Fortran record marker")
    size = struct.unpack(">i", marker)[0]
    if size < 0:
        raise ValueError(f"negative Eclipse record size: {size}")
    payload = _read_exact(stream, size)
    closing = struct.unpack(">i", _read_exact(stream, 4))[0]
    if closing != size:
        raise ValueError(f"Eclipse record markers disagree: {size} != {closing}")
    return payload


def _keyword_item_size(kind: str) -> int:
    sizes = {"INTE": 4, "REAL": 4, "LOGI": 4, "DOUB": 8, "CHAR": 8, "MESS": 0}
    if kind in sizes:
        return sizes[kind]
    if kind.startswith("C0") and kind[2:].isdigit():
        return int(kind[1:])
    raise ValueError(f"unsupported Eclipse keyword type {kind!r}")


def _decode_keyword_payload(kind: str, payload: bytes) -> np.ndarray | list[str]:
    if kind in ("INTE", "LOGI"):
        return np.frombuffer(payload, dtype=">i4").astype(np.int32)
    if kind == "REAL":
        return np.frombuffer(payload, dtype=">f4").astype(np.float32)
    if kind == "DOUB":
        return np.frombuffer(payload, dtype=">f8").astype(np.float64)
    width = 8 if kind == "CHAR" else int(kind[1:])
    return [payload[i : i + width].decode("latin1").rstrip() for i in range(0, len(payload), width)]


def _iter_eclipse_keywords(
    stream: BinaryIO, capture: set[str]
) -> Iterator[tuple[str, str, int, np.ndarray | list[str] | None]]:
    """Read an Eclipse unformatted file without proprietary libraries."""
    while True:
        header = _read_fortran_record(stream)
        if header is None:
            return
        if len(header) != 16:
            raise ValueError(f"expected a 16-byte Eclipse keyword header, got {len(header)}")
        keyword = header[:8].decode("ascii").strip()
        count = struct.unpack(">i", header[8:12])[0]
        kind = header[12:16].decode("ascii")
        item_size = _keyword_item_size(kind)
        chunks: list[bytes] = []
        consumed = 0
        while consumed < count:
            record = _read_fortran_record(stream)
            if record is None:
                raise EOFError(f"EOF inside Eclipse keyword {keyword}")
            if item_size == 0 or len(record) % item_size:
                raise ValueError(f"invalid payload size for {keyword}/{kind}: {len(record)}")
            consumed += len(record) // item_size
            if keyword in capture:
                chunks.append(record)
        if consumed != count:
            raise ValueError(f"Eclipse keyword {keyword} declares {count} items, read {consumed}")
        value = _decode_keyword_payload(kind, b"".join(chunks)) if chunks else None
        yield keyword, kind, count, value


def _parse_eclipse_grid(zf: ZipFile) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray]:
    """Return (nz, ny, nx), UTM cell centres and final simulator active mask."""
    shape: tuple[int, int, int] | None = None
    mapaxes: np.ndarray | None = None
    centres_local: np.ndarray | None = None
    active: np.ndarray | None = None
    n_coords = 0
    n_corners = 0

    with zf.open(ECLIPSE_GRID_MEMBER) as stream:
        for keyword, _, _, value in _iter_eclipse_keywords(
            stream, {"DIMENS", "MAPAXES", "COORDS", "CORNERS"}
        ):
            if keyword == "DIMENS":
                dims = np.asarray(value, dtype=np.int32)
                if dims.shape != (3,):
                    raise ValueError(f"unexpected DIMENS payload: {dims}")
                nx, ny, nz = map(int, dims)
                shape = (nz, ny, nx)
                n_cells = nx * ny * nz
                centres_local = np.empty((n_cells, 3), dtype=np.float32)
                active = np.zeros(n_cells, dtype=bool)
            elif keyword == "MAPAXES":
                mapaxes = np.asarray(value, dtype=np.float64)
            elif keyword == "COORDS":
                if shape is None or active is None:
                    raise ValueError("COORDS appeared before DIMENS")
                coords = np.asarray(value, dtype=np.int32)
                if coords.shape != (7,):
                    raise ValueError(f"unexpected COORDS payload shape {coords.shape}")
                nx = shape[2]
                ny = shape[1]
                expected_global = (coords[2] - 1) * nx * ny + (coords[1] - 1) * nx + coords[0]
                if coords[3] != expected_global:
                    raise ValueError(f"GRID global index disagrees with IJK: {coords.tolist()}")
                active[coords[3] - 1] = coords[4] == 1
                n_coords += 1
            elif keyword == "CORNERS":
                if centres_local is None:
                    raise ValueError("CORNERS appeared before DIMENS")
                corners = np.asarray(value, dtype=np.float32)
                if corners.shape != (24,):
                    raise ValueError(f"unexpected CORNERS payload shape {corners.shape}")
                centres_local[n_corners] = corners.reshape(8, 3).mean(axis=0)
                n_corners += 1

    if shape is None or mapaxes is None or centres_local is None or active is None:
        raise ValueError("Eclipse GRID is missing DIMENS, MAPAXES, COORDS or CORNERS")
    n_cells = int(np.prod(shape))
    if n_coords != n_cells or n_corners != n_cells:
        raise ValueError(f"GRID cell count mismatch: coords={n_coords}, corners={n_corners}, expected={n_cells}")

    # MAPAXES order is point-on-Y-axis, origin, point-on-X-axis.
    point_y = mapaxes[0:2]
    origin = mapaxes[2:4]
    point_x = mapaxes[4:6]
    unit_x = (point_x - origin) / np.linalg.norm(point_x - origin)
    unit_y = (point_y - origin) / np.linalg.norm(point_y - origin)
    x_local = centres_local[:, 0].astype(np.float64)
    y_local = centres_local[:, 1].astype(np.float64)
    centres_utm = np.column_stack(
        [
            origin[0] + x_local * unit_x[0] + y_local * unit_y[0],
            origin[1] + x_local * unit_x[1] + y_local * unit_y[1],
            centres_local[:, 2],
        ]
    ).astype(np.float32)
    return shape, centres_utm, active


def _parse_eclipse_init_poro(zf: ZipFile) -> np.ndarray:
    porosity: np.ndarray | None = None
    with zf.open(ECLIPSE_INIT_MEMBER) as stream:
        for keyword, _, _, value in _iter_eclipse_keywords(stream, {"PORO"}):
            if keyword == "PORO":
                if porosity is not None:
                    raise ValueError("multiple PORO keywords found in Eclipse INIT")
                porosity = np.asarray(value, dtype=np.float32)
    if porosity is None:
        raise ValueError("PORO keyword not found in Eclipse INIT")
    return porosity


def _parse_ascii_grdecl_property(zf: ZipFile, member: str, keyword: str) -> np.ndarray:
    text = zf.read(member).decode("ascii").strip()
    first, body = text.split(None, 1)
    if first.upper() != keyword.upper():
        raise ValueError(f"{member} starts with {first!r}, expected {keyword!r}")
    values = np.fromstring(body.rsplit("/", 1)[0], sep=" ").astype(np.float32)
    if values.size == 0:
        raise ValueError(f"no numeric values parsed from {member}")
    return values


def _header_value(header: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(.*?)\s*$", header, re.MULTILINE)
    return match.group(1) if match else None


def _parse_rms_poro(zf: ZipFile) -> tuple[np.ndarray, dict]:
    blob = zf.read(RMS_PORO_MEMBER)
    marker = b"End GEOMATIC file header"
    marker_start = blob.find(marker)
    if marker_start < 0:
        raise ValueError("RMS porosity realisation lacks a GEOMATIC header terminator")
    payload_start = marker_start + len(marker)
    while payload_start < len(blob) and blob[payload_start] in b"\r\n":
        payload_start += 1
    header = blob[:payload_start].decode("ascii", errors="strict")
    points_text = _header_value(header, "points")
    if points_text is None:
        raise ValueError("RMS GEOMATIC header has no points field")
    n_points = int(points_text)
    payload = blob[payload_start:]
    if len(payload) != n_points * 4:
        raise ValueError(
            f"RMS porosity payload is {len(payload)} bytes for {n_points} points; expected float32"
        )
    values = np.frombuffer(payload, dtype=">f4").astype(np.float32)
    return values, {
        "member": RMS_PORO_MEMBER,
        "format": "RMS GEOMATIC header + big-endian float32 payload",
        "name": _header_value(header, "name"),
        "points": n_points,
        "unfiltered_min": float(_header_value(header, "unfiltered_min") or "nan"),
        "unfiltered_max": float(_header_value(header, "unfiltered_max") or "nan"),
        "spatial_mapping": (
            "Internal RMS project ordering is proprietary/not reconstructed; "
            "the payload is used only for multiset cross-validation."
        ),
    }


def _zip_inventory(path: Path) -> dict:
    with ZipFile(path) as zf:
        files = [info for info in zf.infolist() if not info.is_dir()]
    return {
        "path": path.name,
        "n_files": len(files),
        "compressed_bytes": int(sum(info.compress_size for info in files)),
        "uncompressed_bytes": int(sum(info.file_size for info in files)),
        "suffix_counts": dict(
            Counter((Path(info.filename).suffix.lower() or "[none]") for info in files).most_common()
        ),
        "largest_members": [
            {"name": info.filename, "bytes": int(info.file_size)}
            for info in sorted(files, key=lambda item: item.file_size, reverse=True)[:8]
        ],
    }


def load_reference_volume() -> ReferenceVolume:
    for path in (ECLIPSE_ZIP, RMS_ZIP):
        if not path.exists():
            raise FileNotFoundError(path)

    eclipse_inventory = _zip_inventory(ECLIPSE_ZIP)
    rms_inventory = _zip_inventory(RMS_ZIP)
    with ZipFile(ECLIPSE_ZIP) as zf:
        shape, centres, active_flat = _parse_eclipse_grid(zf)
        init_poro = _parse_eclipse_init_poro(zf)
        ascii_poro = _parse_ascii_grdecl_property(zf, ECLIPSE_ASCII_PORO_MEMBER, "PORO")
    with ZipFile(RMS_ZIP) as zf:
        rms_poro, rms_info = _parse_rms_poro(zf)

    n_cells = int(np.prod(shape))
    if ascii_poro.size != n_cells:
        raise ValueError(f"Eclipse ASCII PORO has {ascii_poro.size} cells, expected {n_cells}")
    if init_poro.size != int(active_flat.sum()):
        raise ValueError(
            f"Eclipse INIT PORO has {init_poro.size} values, GRID has {active_flat.sum()} active cells"
        )
    if not np.array_equal(ascii_poro[active_flat], init_poro):
        raise ValueError("Eclipse ASCII PORO and final INIT PORO disagree on active cells")

    eclipse_nonzero = ascii_poro[ascii_poro > 0]
    rms_multiset_equal = rms_poro.size == eclipse_nonzero.size and np.array_equal(
        np.sort(rms_poro), np.sort(eclipse_nonzero)
    )
    if not rms_multiset_equal:
        raise ValueError("RMS merge_pp04b_PHIF_NW and Eclipse PHIF_NW porosity multisets disagree")

    porosity_flat = np.zeros(n_cells, dtype=np.float32)
    porosity_flat[active_flat] = init_poro
    x = centres[:, 0].reshape(shape)
    y = centres[:, 1].reshape(shape)
    z = centres[:, 2].reshape(shape)
    active = active_flat.reshape(shape)
    porosity = porosity_flat.reshape(shape)

    inspection = {
        "eclipse_zip": eclipse_inventory,
        "rms_zip": rms_inventory,
        "eclipse_reference": {
            "grid_member": ECLIPSE_GRID_MEMBER,
            "init_member": ECLIPSE_INIT_MEMBER,
            "ascii_porosity_member": ECLIPSE_ASCII_PORO_MEMBER,
            "grid_shape_kji": list(shape),
            "n_cells": n_cells,
            "n_active": int(active.sum()),
            "final_poro_min": float(init_poro.min()),
            "final_poro_max": float(init_poro.max()),
            "final_poro_mean": float(init_poro.mean()),
            "ascii_matches_final_init_on_active_cells": True,
            "binary_reader": "Built-in minimal Eclipse unformatted-record reader",
            "open_tool_validation": {
                "tool": "resdata 6.0.1",
                "status": "validated during implementation",
                "result": "same 108x100x63 grid, 183545 active cells and active PORO values",
            },
            "hdf5_note": "VOLVE_2016.h5 contains simulator summary vectors, not the static grid volume.",
        },
        "rms_reference": {
            **rms_info,
            "porosity_multiset_exactly_matches_eclipse_nonzero_phif_nw": True,
            "rms_rescue_note": (
                "Export_Data/.../Resque.bin.66 is a Rescue Geometry File; it was identified but "
                "not used because its spatial binary layout was not independently validated."
            ),
        },
    }
    return ReferenceVolume(shape, x, y, z, active, porosity, inspection)


def _load_seismic_index() -> dict[str, np.ndarray]:
    if not SEISMIC_INDEX.exists():
        raise FileNotFoundError(SEISMIC_INDEX)
    with np.load(SEISMIC_INDEX, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _xy_from_il_xl(il: np.ndarray, xl: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = affine[0, 0] * il + affine[0, 1] * xl + affine[0, 2]
    y = affine[1, 0] * il + affine[1, 1] * xl + affine[1, 2]
    return x, y


def _xy_to_il_xl(x: np.ndarray, y: np.ndarray, index: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine = np.asarray(index["affine_il_xl_to_xy"], dtype=np.float64)
    matrix = affine[:, :2]
    rhs = np.column_stack([x - affine[0, 2], y - affine[1, 2]])
    il_xl = np.linalg.solve(matrix, rhs.T).T
    il_float, xl_float = il_xl[:, 0], il_xl[:, 1]
    in_bounds = (
        (il_float >= int(index["il_min"]))
        & (il_float <= int(index["il_max"]))
        & (xl_float >= int(index["xl_min"]))
        & (xl_float <= int(index["xl_max"]))
    )
    il = np.clip(np.rint(il_float), int(index["il_min"]), int(index["il_max"])).astype(np.int32)
    xl = np.clip(np.rint(xl_float), int(index["xl_min"]), int(index["xl_max"])).astype(np.int32)
    return il, xl, in_bounds


def _well_names(tie: np.lib.npyio.NpzFile) -> list[str]:
    suffix = "__depth_m"
    return sorted(key[: -len(suffix)] for key in tie.files if key.endswith(suffix))


def _estimate_twt_from_weak_ties(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    seismic_index: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict]:
    if not WELL_TIE.exists():
        raise FileNotFoundError(WELL_TIE)
    affine = np.asarray(seismic_index["affine_il_xl_to_xy"], dtype=np.float64)
    numer = np.zeros(depth.size, dtype=np.float64)
    denom = np.zeros(depth.size, dtype=np.float64)
    well_ranges: dict[str, dict] = {}
    with np.load(WELL_TIE, allow_pickle=False) as tie:
        names = _well_names(tie)
        for name in names:
            well_depth = tie[f"{name}__depth_m"].astype(np.float64)
            well_twt = tie[f"{name}__twt_est_ms"].astype(np.float64)
            well_il = tie[f"{name}__inline"].astype(np.float64)
            well_xl = tie[f"{name}__crossline"].astype(np.float64)
            well_x, well_y = _xy_from_il_xl(well_il, well_xl, affine)
            valid = (depth >= well_depth.min()) & (depth <= well_depth.max())
            twt_here = np.interp(depth, well_depth, well_twt)
            x_here = np.interp(depth, well_depth, well_x)
            y_here = np.interp(depth, well_depth, well_y)
            distance2 = (x - x_here) ** 2 + (y - y_here) ** 2
            weight = valid.astype(np.float64) / (distance2 + 100.0**2)
            numer += weight * twt_here
            denom += weight
            well_ranges[name] = {
                "depth_m": [float(well_depth.min()), float(well_depth.max())],
                "twt_ms": [float(well_twt.min()), float(well_twt.max())],
            }
    if np.any(denom == 0):
        raise ValueError(f"weak well ties do not cover {int(np.count_nonzero(denom == 0))} model cells in depth")
    return (numer / denom).astype(np.float32), {
        "method": "depth-wise inverse-horizontal-distance blend of Layer-1 weak well ties",
        "depth_coordinate_warning": "Layer-1 tie depth is MD while Eclipse centre depth is TVD-like; alignment is weak.",
        "well_ranges": well_ranges,
    }


def _sample_seismic_attributes(
    il: np.ndarray,
    xl: np.ndarray,
    twt_ms: np.ndarray,
    index: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not SEGY_PATH.exists():
        raise FileNotFoundError(SEGY_PATH)
    samples_ms = np.asarray(index["samples_ms"], dtype=np.float64)
    time_index = np.searchsorted(samples_ms, twt_ms)
    time_index = np.clip(time_index, 1, samples_ms.size - 2)
    left = np.abs(samples_ms[time_index - 1] - twt_ms)
    right = np.abs(samples_ms[time_index] - twt_ms)
    time_index = np.where(left < right, time_index - 1, time_index).astype(np.int32)

    n_xl = int(index["n_xl"])
    trace_index = (il - int(index["il_min"])) * n_xl + (xl - int(index["xl_min"]))
    unique_trace, inverse = np.unique(trace_index, return_inverse=True)
    traces = np.empty((unique_trace.size, samples_ms.size), dtype=np.float32)
    with segyio.open(str(SEGY_PATH), "r", ignore_geometry=True) as segy:
        if segy.tracecount != int(index["n_traces"]):
            raise ValueError(f"SEG-Y trace count changed: {segy.tracecount} != {int(index['n_traces'])}")
        for row, trace_id in enumerate(unique_trace):
            traces[row] = np.asarray(segy.trace[int(trace_id)], dtype=np.float32)

    offsets = np.arange(-2, 3, dtype=np.int32)
    window_index = np.clip(time_index[:, None] + offsets, 0, samples_ms.size - 1)
    windows = traces[inverse[:, None], window_index]
    amplitude = traces[inverse, time_index]
    local_rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    gradient = traces[inverse, time_index + 1] - traces[inverse, time_index - 1]
    if not np.isfinite(amplitude).all() or not np.isfinite(local_rms).all() or not np.isfinite(gradient).all():
        raise ValueError("non-finite seismic attribute encountered")
    return amplitude, local_rms, gradient, {
        "segy": SEGY_PATH.name,
        "n_unique_traces_read": int(unique_trace.size),
        "twt_ms_range": [float(twt_ms.min()), float(twt_ms.max())],
        "time_index_range": [int(time_index.min()), int(time_index.max())],
        "attribute_definitions": {
            "amplitude": "nearest 4-ms sample",
            "local_rms": "RMS over +/-2 samples",
            "vertical_gradient": "sample[t+1] - sample[t-1]",
        },
    }


def _normalise_coordinates(
    reference: ReferenceVolume,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    active = reference.active
    bounds = {}
    audit = {}
    normalised = []
    for name, values in (("x", reference.x), ("y", reference.y), ("depth", reference.z)):
        stats = fit_minmax(values[active])
        transformed = normalize(values, stats).astype(np.float32)
        restored = denormalize(transformed, stats).astype(values.dtype)
        max_abs_error = float(
            np.max(np.abs(restored.astype(np.float64) - values.astype(np.float64)))
        )
        tolerance = float(
            4.0 * np.finfo(values.dtype).eps * max(1.0, np.max(np.abs(values)))
        )
        if max_abs_error > tolerance:
            raise ValueError(
                f"shared minmax round-trip failed for {name}: {max_abs_error} > {tolerance}"
            )
        normalised.append(transformed)
        bounds[name] = [float(stats.vmin), float(stats.vmax)]
        audit[name] = {
            "stats": stats.to_dict(),
            "roundtrip_max_abs_error": max_abs_error,
            "roundtrip_tolerance": tolerance,
        }
    return normalised[0], normalised[1], normalised[2], bounds, audit


def _build_sparse_well_constraints(
    reference: ReferenceVolume,
    seismic_index: dict[str, np.ndarray],
    amplitude: np.ndarray,
    local_rms: np.ndarray,
    gradient: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    z_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    active_flat_indices = np.flatnonzero(reference.active.ravel())
    active_xyz = np.column_stack(
        [reference.x.ravel()[active_flat_indices], reference.y.ravel()[active_flat_indices], reference.z.ravel()[active_flat_indices]]
    ).astype(np.float64)
    # One model cell is about 50 m laterally and around 1-2 m vertically.
    scales = np.array([50.0, 50.0, 2.0], dtype=np.float64)
    tree = cKDTree(active_xyz / scales)
    affine = np.asarray(seismic_index["affine_il_xl_to_xy"], dtype=np.float64)
    chosen: list[tuple[int, int, float]] = []
    per_well: dict[str, dict] = {}
    with np.load(WELL_TIE, allow_pickle=False) as tie:
        for well_id, name in enumerate(_well_names(tie)):
            depth = tie[f"{name}__depth_m"].astype(np.float64)
            il = tie[f"{name}__inline"].astype(np.float64)
            xl = tie[f"{name}__crossline"].astype(np.float64)
            x, y = _xy_from_il_xl(il, xl, affine)
            in_depth = (depth >= active_xyz[:, 2].min()) & (depth <= active_xyz[:, 2].max())
            query = np.column_stack([x[in_depth], y[in_depth], depth[in_depth]]) / scales
            if query.size == 0:
                per_well[name] = {"n_constraints": 0, "reason": "no weak-tie MD in model depth range"}
                continue
            distance, active_local = tree.query(query, k=1)
            # Remove ties far outside the grid; retain the closest occurrence of each cell.
            best_by_global: dict[int, float] = {}
            for dist, local_idx in zip(distance, active_local):
                if dist > 6.0:
                    continue
                global_idx = int(active_flat_indices[int(local_idx)])
                if global_idx not in best_by_global or dist < best_by_global[global_idx]:
                    best_by_global[global_idx] = float(dist)
            for global_idx, dist in best_by_global.items():
                chosen.append((global_idx, well_id, dist))
            per_well[name] = {
                "n_constraints": len(best_by_global),
                "min_scaled_nearest_cell_distance": float(distance.min()),
                "max_scaled_nearest_cell_distance": max(best_by_global.values()) if best_by_global else None,
            }

    if not chosen:
        raise ValueError("none of the weak well ties intersects the active Eclipse grid")
    chosen.sort(key=lambda row: (row[1], reference.z.ravel()[row[0]], row[0]))
    sparse_property = np.zeros(reference.porosity.size, dtype=np.float32)
    sparse_mask = np.zeros(reference.porosity.size, dtype=np.float32)
    rows = []
    for global_idx, well_id, _ in chosen:
        sparse_property[global_idx] = reference.porosity.ravel()[global_idx]
        sparse_mask[global_idx] = 1.0
        rows.append(
            [
                x_norm.ravel()[global_idx],
                y_norm.ravel()[global_idx],
                z_norm.ravel()[global_idx],
                reference.porosity.ravel()[global_idx],
                amplitude.ravel()[global_idx],
                local_rms.ravel()[global_idx],
                gradient.ravel()[global_idx],
                float(well_id),
            ]
        )
    well_sequence = np.asarray(rows, dtype=np.float32)
    n_wells_with_constraints = int(np.unique(well_sequence[:, 7]).size)
    return (
        sparse_property.reshape(reference.shape),
        sparse_mask.reshape(reference.shape),
        well_sequence,
        {
            "n_observation_rows": int(well_sequence.shape[0]),
            "n_unique_cells": int(np.count_nonzero(sparse_mask)),
            "n_wells_with_constraints": n_wells_with_constraints,
            "coverage_warning": (
                "Only wells that genuinely intersect the final active simulator cells are retained; "
                "the current data leave one intersecting LFP well."
                if n_wells_with_constraints == 1
                else None
            ),
            "columns": [
                "x_normalized",
                "y_normalized",
                "depth_normalized",
                "porosity",
                "seismic_amplitude",
                "seismic_local_rms",
                "seismic_vertical_gradient",
                "well_id",
            ],
            "mapping": "nearest active Eclipse cell in anisotropic (50m,50m,2m) coordinates",
            "per_well": per_well,
        },
    )


def _make_samples(
    reference: ReferenceVolume,
    channels: np.ndarray,
    well_sequence: np.ndarray,
    il_grid: np.ndarray,
    xl_grid: np.ndarray,
    twt_grid: np.ndarray,
) -> tuple[list[dict], list[dict], dict]:
    pk, pj, pi = PATCH_SHAPE
    nz, ny, nx = reference.shape
    if (nz % pk, ny % pj, nx % pi) != (0, 0, 0):
        raise ValueError(f"patch shape {PATCH_SHAPE} does not tile grid {reference.shape}")
    train: list[dict] = []
    test: list[dict] = []
    active_counts = {"train": 0, "test": 0}
    for kb, k0 in enumerate(range(0, nz, pk)):
        for jb, j0 in enumerate(range(0, ny, pj)):
            for ib, i0 in enumerate(range(0, nx, pi)):
                sl = np.s_[k0 : k0 + pk, j0 : j0 + pj, i0 : i0 + pi]
                active_patch = reference.active[sl]
                # Labels are spatially blocked, but the global well table is supplied to both
                # splits; downstream evaluation is conditional reconstruction, not a pure holdout.
                split = "test" if ib >= 4 else "train"
                active_counts[split] += int(active_patch.sum())
                centre = (k0 + pk // 2, j0 + pj // 2, i0 + pi // 2)
                if active_patch.any():
                    patch_twt = float(twt_grid[sl][active_patch].mean())
                else:
                    patch_twt = float(twt_grid[centre])
                sample = {
                    "seismic_patch": channels[(slice(None),) + sl],
                    "well_log_seq": well_sequence,
                    "position": {
                        "inline": int(il_grid[centre]),
                        "crossline": int(xl_grid[centre]),
                        "time_ms": patch_twt,
                        "well_name": None,
                    },
                    "label": reference.porosity[sl],
                    "meta": {
                        "source": "Volve Eclipse final GRID/INIT; RMS GEOMATIC porosity cross-check",
                        "property": "PORO",
                        "patch_index_kji": [kb, jb, ib],
                        "patch_start_kji": [k0, j0, i0],
                        "patch_shape_kji": list(PATCH_SHAPE),
                        "input_channels": CHANNEL_NAMES,
                        "n_active_cells": int(active_patch.sum()),
                        "n_sparse_constraints_in_patch": int(channels[7][sl].sum()),
                        "split_strategy": (
                            "blocked east labels: i-blocks 0..3 train, 4..5 test; global well "
                            "constraints cross the boundary, so evaluation is conditional "
                            "reconstruction rather than strict spatial holdout generalization"
                        ),
                        "weak_tie_warning": "Layer-1 MD-to-TWT tie is approximate, not a checkshot/VSP tie.",
                    },
                }
                (test if split == "test" else train).append(sample)
    if not train or not test or active_counts["train"] == 0 or active_counts["test"] == 0:
        raise ValueError(f"invalid split: train={len(train)}, test={len(test)}, active={active_counts}")
    return train, test, {
        "strategy": "blocked east label split with global cross-boundary well constraints",
        "evaluation_protocol": (
            "conditional reconstruction given test-region well constraints, "
            "NOT strict spatial holdout generalization"
        ),
        "train_patch_i_blocks": [0, 1, 2, 3],
        "test_patch_i_blocks": [4, 5],
        "n_train_samples": len(train),
        "n_test_samples": len(test),
        "active_cells": active_counts,
    }


def build_dataset() -> dict:
    reference = load_reference_volume()
    seismic_index = _load_seismic_index()
    active_idx = np.flatnonzero(reference.active.ravel())
    active_x = reference.x.ravel()[active_idx].astype(np.float64)
    active_y = reference.y.ravel()[active_idx].astype(np.float64)
    active_z = reference.z.ravel()[active_idx].astype(np.float64)
    active_il, active_xl, in_bounds = _xy_to_il_xl(active_x, active_y, seismic_index)
    if not in_bounds.all():
        raise ValueError(
            f"{int(np.count_nonzero(~in_bounds))} active Eclipse cells lie outside the Layer-1 seismic grid"
        )
    active_twt, tie_info = _estimate_twt_from_weak_ties(
        active_x, active_y, active_z, seismic_index
    )
    active_amp, active_rms, active_grad, seismic_info = _sample_seismic_attributes(
        active_il, active_xl, active_twt, seismic_index
    )

    n_cells = reference.porosity.size
    amplitude = np.zeros(n_cells, dtype=np.float32)
    local_rms = np.zeros(n_cells, dtype=np.float32)
    gradient = np.zeros(n_cells, dtype=np.float32)
    twt = np.zeros(n_cells, dtype=np.float32)
    amplitude[active_idx] = active_amp
    local_rms[active_idx] = active_rms
    gradient[active_idx] = active_grad
    twt[active_idx] = active_twt
    amplitude = amplitude.reshape(reference.shape)
    local_rms = local_rms.reshape(reference.shape)
    gradient = gradient.reshape(reference.shape)
    twt = twt.reshape(reference.shape)

    all_il, all_xl, _ = _xy_to_il_xl(
        reference.x.ravel().astype(np.float64), reference.y.ravel().astype(np.float64), seismic_index
    )
    il_grid = all_il.reshape(reference.shape)
    xl_grid = all_xl.reshape(reference.shape)
    x_norm, y_norm, z_norm, coordinate_bounds, coordinate_normalization_audit = (
        _normalise_coordinates(reference)
    )
    sparse_property, sparse_mask, well_sequence, sparse_info = _build_sparse_well_constraints(
        reference,
        seismic_index,
        amplitude,
        local_rms,
        gradient,
        x_norm,
        y_norm,
        z_norm,
    )
    # Explicit no-op denoising: sharp events can be real thin beds or
    # discontinuities, so smoothing is unsafe by default.  Local RMS remains a
    # separate attribute and never replaces the untouched amplitude channel.
    seismic_channels = denoise_identity(np.stack([amplitude, local_rms, gradient], axis=0))
    channels = np.concatenate(
        [
            seismic_channels,
            np.stack(
                [
                    x_norm,
                    y_norm,
                    z_norm,
                    sparse_property,
                    sparse_mask,
                    reference.active.astype(np.float32),
                ],
                axis=0,
            ),
        ],
        axis=0,
    ).astype(np.float32)
    train, test, split_info = _make_samples(
        reference, channels, well_sequence, il_grid, xl_grid, twt
    )

    from dataset_io import save_split  # noqa: PLC0415

    train_path = save_split("reconstruction", "train", train)
    test_path = save_split("reconstruction", "test", test)
    inspection_path = HERE / "model_inspection.json"
    inspection_path.write_text(
        json.dumps(reference.inspection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "task": "reconstruction",
        "target": "Volve final Eclipse PORO on active cells",
        "grid_shape_kji": list(reference.shape),
        "n_active_cells": int(reference.active.sum()),
        "input_channels": CHANNEL_NAMES,
        "coordinate_bounds": coordinate_bounds,
        "preprocessing": {
            "denoise": "ml_framework.preprocess.denoise_identity",
            "denoise_reason": "sharp seismic/well-log features may be real geology",
            "coordinate_normalization": "ml_framework.preprocess.fit_minmax + normalize",
            "coordinate_roundtrip": coordinate_normalization_audit,
        },
        "weak_tie": tie_info,
        "seismic": seismic_info,
        "sparse_wells": sparse_info,
        "split": split_info,
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "test_path": str(test_path.relative_to(PROJECT_ROOT)),
        "inspection_path": str(inspection_path.relative_to(PROJECT_ROOT)),
    }
    (HERE / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="inspect/cross-check the Eclipse and RMS references without reading seismic or writing HDF5",
    )
    args = parser.parse_args()
    if args.inspect_only:
        reference = load_reference_volume()
        path = HERE / "model_inspection.json"
        path.write_text(json.dumps(reference.inspection, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(reference.inspection, ensure_ascii=False, indent=2))
        return
    print(json.dumps(build_dataset(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
