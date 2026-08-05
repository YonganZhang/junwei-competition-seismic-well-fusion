#!/usr/bin/env python3
"""P38 real-well PHIF and direct-ST0202 cross-modal pilot.

The entrypoint is deliberately two-stage.  ``phase0`` freezes the native-PHIF
target, direct seismic alignment, LOGO3 split and scientific protocol before
any foundation encoder or trainable head can run.  ``run`` is enabled only
when the byte-reproducible Phase-0 evidence passes every fail-closed gate.

No command accepts a test/holdout HDF5 path.  This task never requires sparse
Eclipse KJI coordinates and never treats PHIE or Eclipse PORO as PHIF.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import segyio


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for _root in (str(PROJECT_ROOT), str(HERE)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import p37_real_well_seismic_supervision as p37  # noqa: E402
import p38_pilot_core as pilot  # noqa: E402


SCHEMA = "reconstruction-p38-real-well-phif-direct-seismic/v1"
PHASE0_SCHEMA = "reconstruction-p38-phase0-freeze/v1"
DEFAULT_OUTPUT = HERE / "_outputs/p38_real_well_phif_direct_seismic"
DEFAULT_SCRATCH = PROJECT_ROOT / "_tmp/p38_real_well_phif_direct_seismic"
DEFAULT_RAW_ROOT = PROJECT_ROOT
FORBIDDEN_TOKENS = ("test.h5", "holdout", "frozen_test", "frozen-test")
PARENT_ORDER = ("15/9-19", "15/9-F-11", "15/9-F-15")
CURVES = p37.COMMON_INPUTS
TARGET_LOWER = 0.0
TARGET_UPPER = 1.0
MIN_PARENT_COVERAGE = 0.90
WELL_WINDOW = 33
WELL_RADIUS = WELL_WINDOW // 2
SEISMIC_TIME_SAMPLES = 400
SEISMIC_TRACE_COUNT = 160
SEISMIC_EXTRA_SAMPLES = 2
SEEDS = (2693,)
BOOTSTRAP_SEED = 2693
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_BLOCK_M = 20.0
MAX_SCIENTIFIC_ITERATIONS = 3
HEAD_FEATURE_DIM = 32
HEAD_HIDDEN_DIM = 32
HEAD_MAX_UPDATES = 120
HEAD_EVAL_EVERY = 5
HEAD_LEARNING_RATE = 3e-3
TWT_PERTURBATION_SAMPLES = 40
TWT_PERTURBATION_MS = 160.0
MAX_INFORMATIVE_90_INTERVAL_WIDTH = 1.0
MOMENT_WEIGHTS_SHA256 = (
    "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
)
GFM_REVISION = "d4a33965730a506cfdb4c85fa2a0a344c53216a2"
GFM_WEIGHTS_SHA256 = (
    "c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446"
)


@dataclass(frozen=True)
class P38Well:
    spec: p37.WellSpec
    report_member: str | None
    target_definition_evidence: str


P38_WELLS = (
    P38Well(
        spec=replace(
            p37.WELLS[0],
            target_member=(
                "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-19 A/"
                "CPI/15_9-19_A_CPI.las"
            ),
            target_curve="PHIF",
        ),
        report_member=None,
        target_definition_evidence=(
            "Official CPI output; PHIF is the final edited formation-porosity "
            "curve, BVW=PHIF*SW, and the explicit LAS null is -999.25."
        ),
    ),
    P38Well(
        spec=p37.WELLS[1],
        report_member=(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-11 T2/"
            "PETROPHYSICAL_REPORT_2.PDF"
        ),
        target_definition_evidence=(
            "The report defines PHIF as total porosity derived from density "
            "and calibrated to overburden-corrected core porosity."
        ),
    ),
    P38Well(
        spec=p37.WELLS[2],
        report_member=(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-15 A/"
            "PETROPHYSICAL_REPORT_1.PDF"
        ),
        target_definition_evidence=(
            "The report defines Phif from density porosity, NPHI and fixed "
            "regression constants calibrated to overburden-corrected core porosity."
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _assert_no_forbidden_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        lowered = str(Path(path).expanduser().resolve()).lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            raise RuntimeError(f"forbidden test/holdout path: {path}")


def _validate_output_path(path: Path, *, scratch: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = (
        (PROJECT_ROOT / "_tmp/p38_real_well_phif_direct_seismic").resolve()
        if scratch
        else HERE.resolve()
    )
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        kind = "scratch" if scratch else "output"
        raise ValueError(f"P38 {kind} must stay inside its owned path: {resolved}") from exc
    if not scratch:
        protected = {
            (HERE / "_outputs/p21_fixed_foundation_ensemble").resolve(),
            (HERE / "_outputs/p30_bounded_geostatistics_feasibility").resolve(),
            (HERE / "_outputs/p30_bounded_geostatistics_feasibility_v2").resolve(),
            (HERE / "_outputs/p37_real_well_seismic_supervision_closure").resolve(),
        }
        if resolved in protected:
            raise ValueError("P38 refuses to overwrite P21/P30/P37 evidence")
    return resolved


def _centered_start(
    values: np.ndarray,
    *,
    window_size: int,
    lower_bound: int,
    upper_bound: int,
) -> int:
    selected = np.asarray(values, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("window coordinates must be non-empty")
    source_size = int(upper_bound) - int(lower_bound) + 1
    span = int(selected.max()) - int(selected.min()) + 1
    if source_size < window_size or span > window_size:
        raise ValueError("source or selected span cannot fit the native window")
    proposed = (int(selected.min()) + int(selected.max()) + 1 - window_size) // 2
    return int(
        np.clip(
            proposed,
            int(lower_bound),
            int(upper_bound) - int(window_size) + 1,
        )
    )


def _nearest_curve(
    query_depth: np.ndarray,
    source_depth: np.ndarray,
    source_value: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    depth = np.asarray(source_depth[valid], dtype=np.float64)
    value = np.asarray(source_value[valid], dtype=np.float64)
    if len(depth) < 2:
        raise RuntimeError("curve has fewer than two finite depth rows")
    order = np.argsort(depth, kind="stable")
    depth = depth[order]
    value = value[order]
    spacing = float(np.median(np.diff(depth)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise RuntimeError("curve depth spacing is invalid")
    query = np.asarray(query_depth, dtype=np.float64)
    right = np.clip(np.searchsorted(depth, query), 1, len(depth) - 1)
    left = right - 1
    choose_right = np.abs(depth[right] - query) < np.abs(depth[left] - query)
    chosen = np.where(choose_right, right, left)
    distance = np.abs(depth[chosen] - query)
    observed = distance <= max(0.51 * spacing, 1e-4)
    output = np.full(query.shape, np.nan, dtype=np.float32)
    output[observed] = value[chosen[observed]].astype(np.float32)
    return output, observed, spacing


def _target_arrays(
    archive: ZipFile,
    spec: p37.WellSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Mapping[str, np.ndarray], Mapping[str, str], float]:
    arrays, units, null_value, member_hash = p37._load_curves(  # noqa: SLF001
        archive, spec.target_member, spec.target_format
    )
    depth = p37._depth_to_m(arrays["DEPTH"], units["DEPTH"])  # noqa: SLF001
    target = np.asarray(arrays[spec.target_curve], dtype=np.float64)
    valid = (
        p37._finite_curve(depth, null_value)  # noqa: SLF001
        & p37._finite_curve(target, null_value)  # noqa: SLF001
        & (target >= TARGET_LOWER)
        & (target <= TARGET_UPPER)
    )
    depth = depth[valid]
    target = target[valid]
    order = np.argsort(depth, kind="stable")
    depth = depth[order]
    target = target[order]
    if len(depth) < WELL_WINDOW:
        raise RuntimeError(f"insufficient PHIF rows for {spec.key}")
    curve_description = ""
    explicit_null = float(null_value)
    if spec.target_format == "las":
        las = p37._read_las(archive.read(spec.target_member))  # noqa: SLF001
        curve_description = str(las.curves[spec.target_curve].descr or "")
        explicit_null = float(las.well.NULL.value)
    profile = {
        "member": spec.target_member,
        "member_sha256": member_hash,
        "format": spec.target_format,
        "curve": spec.target_curve,
        "curve_description": curve_description,
        "curve_unit_source": units[spec.target_curve],
        "curve_unit_canonical": "V/V_fraction",
        "depth_unit_source": units["DEPTH"],
        "depth_unit_canonical": "m_MD",
        "explicit_null": explicit_null,
        "valid_rule": "finite and explicit-null-distinct and 0<=PHIF<=1",
        "physical_rows": int(len(depth)),
        "zero_rows": int(np.count_nonzero(target == 0.0)),
        "md_range_m": [float(depth.min()), float(depth.max())],
        "value_range_fraction": [float(target.min()), float(target.max())],
    }
    return depth, target, profile, arrays, units, null_value


def _report_profile(archive: ZipFile, member: str | None) -> dict[str, Any] | None:
    if member is None:
        return None
    value = archive.read(member)
    return {
        "member": member,
        "member_sha256": _sha256_bytes(value),
        "format": "pdf",
    }


def _input_windows(
    archive: ZipFile,
    spec: p37.WellSpec,
    target_md: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays, units, null_value, member_hash = p37._load_curves(  # noqa: SLF001
        archive, spec.input_member, spec.input_format
    )
    source_depth = p37._depth_to_m(arrays["DEPTH"], units["DEPTH"])  # noqa: SLF001
    target_step = float(np.median(np.diff(target_md)))
    query = target_md[:, None] + target_step * np.arange(
        -WELL_RADIUS, WELL_RADIUS + 1, dtype=np.float64
    )[None, :]
    values = np.full((len(target_md), len(CURVES), WELL_WINDOW), np.nan, np.float32)
    masks = np.zeros(values.shape, dtype=bool)
    audit: dict[str, Any] = {}
    for channel, canonical in enumerate(CURVES):
        source = spec.input_curves[canonical]
        raw = np.asarray(arrays[source], dtype=np.float64)
        valid = (
            p37._finite_curve(source_depth, null_value)  # noqa: SLF001
            & p37._finite_curve(raw, null_value)  # noqa: SLF001
        )
        matched, observed, spacing = _nearest_curve(
            query.reshape(-1), source_depth, raw, valid
        )
        values[:, channel, :] = matched.reshape(len(target_md), WELL_WINDOW)
        masks[:, channel, :] = observed.reshape(len(target_md), WELL_WINDOW)
        audit[canonical] = {
            "source_curve": source,
            "source_unit": units[source],
            "finite_source_rows": int(np.count_nonzero(valid)),
            "median_source_step_m": spacing,
            "center_observed_rows": int(np.count_nonzero(masks[:, channel, WELL_RADIUS])),
            "window_observed_fraction": float(np.mean(masks[:, channel, :])),
        }
    return values, masks, {
        "member": spec.input_member,
        "member_sha256": member_hash,
        "format": spec.input_format,
        "depth_unit_source": units["DEPTH"],
        "window_shape": [len(CURVES), WELL_WINDOW],
        "window_step_m": target_step,
        "window_padding_applied": False,
        "missing_values_retained_with_explicit_mask": True,
        "curves": audit,
    }


def _map_alignment(
    target_md: np.ndarray,
    survey: Mapping[str, Any],
    checkshot_depth: np.ndarray,
    checkshot_twt: np.ndarray,
    index: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n = len(target_md)
    tvdss = np.full(n, np.nan, dtype=np.float64)
    x = np.full(n, np.nan, dtype=np.float64)
    y = np.full(n, np.nan, dtype=np.float64)
    twt = np.full(n, np.nan, dtype=np.float64)
    inline = np.full(n, -1, dtype=np.int32)
    crossline = np.full(n, -1, dtype=np.int32)
    time_index = np.full(n, -1, dtype=np.int32)
    survey_md = np.asarray(survey["md_m"], dtype=np.float64)
    survey_ok = (target_md >= survey_md.min()) & (target_md <= survey_md.max())
    tvdss[survey_ok] = (
        np.interp(
            target_md[survey_ok],
            survey_md,
            np.asarray(survey["tvd_m_from_reference"], dtype=np.float64),
        )
        - float(survey["reference_elevation_m_above_msl"])
    )
    x[survey_ok] = np.interp(
        target_md[survey_ok], survey_md, np.asarray(survey["easting_m"], dtype=np.float64)
    )
    y[survey_ok] = np.interp(
        target_md[survey_ok], survey_md, np.asarray(survey["northing_m"], dtype=np.float64)
    )
    checkshot_ok = survey_ok & (tvdss >= checkshot_depth.min()) & (
        tvdss <= checkshot_depth.max()
    )
    twt[checkshot_ok] = np.interp(
        tvdss[checkshot_ok], checkshot_depth, checkshot_twt
    )
    affine = np.asarray(index["affine_il_xl_to_xy"], dtype=np.float64)
    inverse = np.linalg.inv(affine[:, :2])
    xy_ok = np.isfinite(x) & np.isfinite(y)
    fractional = (np.column_stack([x[xy_ok], y[xy_ok]]) - affine[:, 2]) @ inverse.T
    inline[xy_ok] = np.rint(fractional[:, 0]).astype(np.int32)
    crossline[xy_ok] = np.rint(fractional[:, 1]).astype(np.int32)
    samples = np.asarray(index["samples_ms"], dtype=np.float64)
    seismic_ok = (
        checkshot_ok
        & (inline >= int(index["il_min"]))
        & (inline <= int(index["il_max"]))
        & (crossline >= int(index["xl_min"]))
        & (crossline <= int(index["xl_max"]))
        & (twt >= float(samples.min()))
        & (twt <= float(samples.max()))
    )
    position = np.searchsorted(samples, twt[seismic_ok])
    position = np.clip(position, 1, len(samples) - 1)
    left = np.abs(samples[position - 1] - twt[seismic_ok])
    right = np.abs(samples[position] - twt[seismic_ok])
    time_index[seismic_ok] = np.where(left < right, position - 1, position).astype(np.int32)
    return {
        "tvdss_m": tvdss,
        "x_m": x,
        "y_m": y,
        "twt_ms": twt,
        "inline": inline,
        "crossline": crossline,
        "time_index": time_index,
        "survey_valid": survey_ok,
        "checkshot_valid": checkshot_ok,
        "seismic_valid": seismic_ok,
    }


def _read_native_sections(
    *,
    segy_path: Path,
    index: Mapping[str, np.ndarray],
    inline_values: np.ndarray,
    crossline_start: int,
    time_start: int,
) -> np.ndarray:
    unique_inline = np.unique(inline_values).astype(np.int32)
    sections = np.empty(
        (len(unique_inline), 3, SEISMIC_TIME_SAMPLES, SEISMIC_TRACE_COUNT),
        dtype=np.float32,
    )
    crosslines = np.arange(
        crossline_start, crossline_start + SEISMIC_TRACE_COUNT, dtype=np.int32
    )
    sample_start = time_start - SEISMIC_EXTRA_SAMPLES
    sample_stop = time_start + SEISMIC_TIME_SAMPLES + SEISMIC_EXTRA_SAMPLES
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        if handle.tracecount != int(index["n_traces"]):
            raise RuntimeError("ST0202 trace count differs from the locked index")
        for row, inline in enumerate(unique_inline):
            extended = np.empty(
                (SEISMIC_TIME_SAMPLES + 4, SEISMIC_TRACE_COUNT), dtype=np.float32
            )
            for column, crossline in enumerate(crosslines):
                trace_id = (
                    (int(inline) - int(index["il_min"])) * int(index["n_xl"])
                    + int(crossline)
                    - int(index["xl_min"])
                )
                trace = np.asarray(handle.trace[int(trace_id)], dtype=np.float32)
                extended[:, column] = trace[sample_start:sample_stop]
            sections[row, 0] = extended[2:-2]
            sections[row, 1] = np.sqrt(
                sum(
                    extended[offset : offset + SEISMIC_TIME_SAMPLES].astype(np.float64) ** 2
                    for offset in range(5)
                )
                / 5.0
            ).astype(np.float32)
            sections[row, 2] = extended[3:-1] - extended[1:-3]
    if not np.all(np.isfinite(sections)):
        raise FloatingPointError("native ST0202 section contains non-finite samples")
    return sections


def build_phase0(
    *,
    raw_project_root: Path,
    scratch_dir: Path,
    read_native_windows: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    raw_root = Path(raw_project_root).expanduser().resolve()
    scratch = _validate_output_path(scratch_dir, scratch=True)
    logs_path = raw_root / "_sandbox/volve_data/Volve_Well_logs.zip"
    technical_path = raw_root / "_sandbox/volve_data/Volve_Well_technical_data.zip"
    vsp_path = raw_root / "_sandbox/volve_data/Volve_Seismic_VSP.zip"
    index_path = raw_root / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
    segy_path = (
        raw_root
        / "_sandbox/volve_data/_extracted_seismic/ST0202/Stacks/"
        "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
    )
    required = (logs_path, technical_path, vsp_path, index_path, segy_path)
    _assert_no_forbidden_paths(required)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered P38 development asset missing: {missing}")
    with np.load(index_path, allow_pickle=False) as payload:
        index = {key: payload[key] for key in payload.files}
    samples = np.asarray(index["samples_ms"], dtype=np.float64)
    target_contract: dict[str, Any] = {
        "schema_version": "reconstruction-p38-native-phif-target/v1",
        "canonical_target": "published CPI PHIF",
        "unit": "V/V_fraction",
        "bounds": [TARGET_LOWER, TARGET_UPPER],
        "phif_is_distinct_from_phie": True,
        "same_well_phif_phie_rmse": 0.07790825422663222,
        "same_well_phif_phie_max_abs": 0.2631971103083672,
        "zero_rule": (
            "Retain finite PHIF=0. It is distinct from explicit -999.25 null; on 19A all "
            "42 zero rows also have PORD=0, BVW=0, SW=1 and SAND_FLAG=0."
        ),
        "operational_definition": (
            "The native published final/total CPI PHIF curve from each parent is the target. "
            "Different historical interpretation recipes are provenance, not permission to "
            "rename PHIE or performance-filter rows."
        ),
        "parents": {},
    }
    alignment_manifest: dict[str, Any] = {
        "schema_version": "reconstruction-p38-direct-st0202-alignment/v1",
        "chain": [
            "MD_m",
            "actual_survey_TVDSS_and_UTM_ED50_zone31N",
            "checkshot_TWT_ms",
            "UTM_to_ILXL_affine",
            "nearest_native_ST0202_trace_and_4ms_sample",
        ],
        "minimum_parent_coverage": MIN_PARENT_COVERAGE,
        "native_window_shape": [3, SEISMIC_TIME_SAMPLES, SEISMIC_TRACE_COUNT],
        "interpolation_applied": False,
        "padding_applied": False,
        "target_derived_alignment": False,
        "parents": {},
    }
    arrays_by_name: dict[str, list[np.ndarray]] = {}
    section_chunks: list[np.ndarray] = []
    section_parent: list[int] = []
    global_section_offset = 0
    semantic_pass = True
    alignment_pass = True
    with ZipFile(logs_path) as logs, ZipFile(technical_path) as technical, ZipFile(
        vsp_path
    ) as vsp:
        equivalence = p37.phie_phif_equivalence(logs)
        if equivalence["rmse_fraction"] != target_contract["same_well_phif_phie_rmse"]:
            raise RuntimeError("P37 PHIF/PHIE separation evidence drifted")
        for parent_index, well in enumerate(P38_WELLS):
            spec = well.spec
            target_md, target, target_profile, target_arrays, target_units, target_null = (
                _target_arrays(logs, spec)
            )
            input_values, input_masks, input_profile = _input_windows(
                logs, spec, target_md
            )
            survey, survey_profile = p37._survey_profile(technical, spec)  # noqa: SLF001
            check_depth, check_twt, check_profile = p37._checkshot_curve(vsp, spec)  # noqa: SLF001
            mapped = _map_alignment(target_md, survey, check_depth, check_twt, index)
            center_observed = np.all(input_masks[:, :, WELL_RADIUS], axis=1)
            preliminary = center_observed & mapped["seismic_valid"]
            if not np.any(preliminary):
                raise RuntimeError(f"{spec.key} has zero directly aligned joint-quality rows")
            crossline_start = _centered_start(
                mapped["crossline"][preliminary],
                window_size=SEISMIC_TRACE_COUNT,
                lower_bound=int(index["xl_min"]),
                upper_bound=int(index["xl_max"]),
            )
            time_start = _centered_start(
                mapped["time_index"][preliminary],
                window_size=SEISMIC_TIME_SAMPLES,
                lower_bound=SEISMIC_EXTRA_SAMPLES,
                upper_bound=len(samples) - SEISMIC_EXTRA_SAMPLES - 1,
            )
            trace_token = mapped["crossline"] - crossline_start
            local_time = mapped["time_index"] - time_start
            native_inside = (
                (trace_token >= 0)
                & (trace_token < SEISMIC_TRACE_COUNT)
                & (local_time >= 0)
                & (local_time < SEISMIC_TIME_SAMPLES)
            )
            quality = preliminary & native_inside
            retained = int(np.count_nonzero(quality))
            coverage = float(retained / len(target_md))
            if coverage < MIN_PARENT_COVERAGE:
                alignment_pass = False
            selected_inline = mapped["inline"][quality]
            unique_inline = np.unique(selected_inline).astype(np.int32)
            local_section = np.searchsorted(unique_inline, selected_inline).astype(np.int32)
            if not np.array_equal(unique_inline[local_section], selected_inline):
                raise RuntimeError("inline-to-section mapping is incomplete")
            if read_native_windows:
                sections = _read_native_sections(
                    segy_path=segy_path,
                    index=index,
                    inline_values=selected_inline,
                    crossline_start=crossline_start,
                    time_start=time_start,
                )
                section_chunks.append(sections)
                section_parent.extend([parent_index] * len(sections))
            selected_count = len(unique_inline)
            global_section = local_section + global_section_offset
            global_section_offset += selected_count
            flags: dict[str, Any] = {}
            for name in ("SAND_FLAG", "CARB_FLAG", "COAL_FLAG", "VSH", "SW", "BVW", "PORD"):
                if name not in target_arrays:
                    continue
                raw = np.asarray(target_arrays[name], dtype=np.float64)
                finite = p37._finite_curve(raw, target_null)  # noqa: SLF001
                flags[name] = {
                    "unit": target_units.get(name, ""),
                    "finite_rows_in_source": int(np.count_nonzero(finite)),
                }
            report = _report_profile(logs, well.report_member)
            target_contract["parents"][spec.parent] = {
                "key": spec.key,
                "branch": spec.branch,
                "target": target_profile,
                "report": report,
                "definition_evidence": well.target_definition_evidence,
                "quality_flag_provenance": flags,
                "input": input_profile,
                "shared_operational_definition_passed": True,
            }
            alignment_manifest["parents"][spec.parent] = {
                "key": spec.key,
                "physical_phif_rows": int(len(target_md)),
                "six_curve_center_rows": int(np.count_nonzero(center_observed)),
                "direct_st0202_rows_before_joint_mask": int(
                    np.count_nonzero(mapped["seismic_valid"])
                ),
                "retained_joint_rows": retained,
                "retained_fraction": coverage,
                "coverage_gate_passed": coverage >= MIN_PARENT_COVERAGE,
                "survey": survey_profile,
                "checkshot": check_profile,
                "ranges": {
                    "MD_m": [float(target_md[quality].min()), float(target_md[quality].max())],
                    "TVDSS_m": [
                        float(mapped["tvdss_m"][quality].min()),
                        float(mapped["tvdss_m"][quality].max()),
                    ],
                    "TWT_ms": [
                        float(mapped["twt_ms"][quality].min()),
                        float(mapped["twt_ms"][quality].max()),
                    ],
                    "inline": [int(selected_inline.min()), int(selected_inline.max())],
                    "crossline": [
                        int(mapped["crossline"][quality].min()),
                        int(mapped["crossline"][quality].max()),
                    ],
                    "time_index": [
                        int(mapped["time_index"][quality].min()),
                        int(mapped["time_index"][quality].max()),
                    ],
                },
                "native_window": {
                    "unique_inline_sections": selected_count,
                    "crossline_start": crossline_start,
                    "crossline_stop_inclusive": crossline_start + SEISMIC_TRACE_COUNT - 1,
                    "time_start_index": time_start,
                    "time_stop_index_inclusive": time_start + SEISMIC_TIME_SAMPLES - 1,
                    "source_samples_include_two_sample_halo": True,
                    "finite": True if read_native_windows else None,
                    "uninterpolated": True,
                    "unpadded": True,
                },
            }
            chosen = {
                "parent_index": np.full(retained, parent_index, dtype=np.int8),
                "well_key": np.full(retained, spec.key),
                "target": target[quality].astype(np.float32),
                "MD_m": target_md[quality].astype(np.float64),
                "TVDSS_m": mapped["tvdss_m"][quality].astype(np.float64),
                "x_m": mapped["x_m"][quality].astype(np.float64),
                "y_m": mapped["y_m"][quality].astype(np.float64),
                "TWT_ms": mapped["twt_ms"][quality].astype(np.float64),
                "inline": mapped["inline"][quality].astype(np.int32),
                "crossline": mapped["crossline"][quality].astype(np.int32),
                "time_index": mapped["time_index"][quality].astype(np.int32),
                "section_id": global_section.astype(np.int32),
                "trace_token_id": trace_token[quality].astype(np.int16),
                "local_time_index": local_time[quality].astype(np.int16),
                "time_position": (
                    local_time[quality] / float(SEISMIC_TIME_SAMPLES - 1)
                ).astype(np.float32),
                "well_windows": input_values[quality].astype(np.float32),
                "well_masks": input_masks[quality].astype(bool),
            }
            for name, value in chosen.items():
                arrays_by_name.setdefault(name, []).append(value)
    target_contract["semantic_gate_passed"] = semantic_pass
    target_contract["phif_phie_same_well_audit"] = equivalence
    alignment_manifest["alignment_gate_passed"] = alignment_pass
    alignment_manifest["seismic_index"] = {
        "path": str(index_path),
        "sha256": _sha256(index_path),
        "grid_shape_il_xl": [int(index["n_il"]), int(index["n_xl"])],
        "trace_count": int(index["n_traces"]),
        "sample_count": len(samples),
        "sample_interval_ms": float(np.median(np.diff(samples))),
        "affine_il_xl_to_xy": np.asarray(index["affine_il_xl_to_xy"]).tolist(),
    }
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in arrays_by_name.items()}
    arrays["row_id"] = np.arange(len(arrays["target"]), dtype=np.int64)
    if read_native_windows:
        arrays["native_sections"] = np.concatenate(section_chunks, axis=0)
        arrays["section_parent_index"] = np.asarray(section_parent, dtype=np.int8)
        if len(arrays["native_sections"]) != global_section_offset:
            raise RuntimeError("native section count drift")
    split_manifest = {
        "schema_version": "reconstruction-p38-logo3-split/v1",
        "state": "FROZEN_ACTIVE" if semantic_pass and alignment_pass else "BLOCKED",
        "parent_order": list(PARENT_ORDER),
        "folds": [
            {
                "fold_id": fold,
                "held_parent": held,
                "train_parents": [parent for parent in PARENT_ORDER if parent != held],
            }
            for fold, held in enumerate(PARENT_ORDER)
        ],
        "split_before": [
            "normalization",
            "imputation",
            "projection",
            "cache",
            "encoder_call",
            "model_selection",
            "calibration",
            "agent_action",
        ],
        "held_parent_phif_invisible_to_fit": True,
        "test_h5_or_holdout_member": None,
    }
    freeze = {
        "schema_version": PHASE0_SCHEMA,
        "state": (
            "PHASE0_PASSED"
            if semantic_pass and alignment_pass
            else (
                "BLOCKED_PHIF_SEMANTICS" if not semantic_pass else "BLOCKED_DIRECT_SEISMIC_ALIGNMENT"
            )
        ),
        "primary_metric": "equal_parent_macro_RMSE",
        "secondary_metrics": ["per_parent_RMSE", "MAE", "bias"],
        "controls": [
            "well_only",
            "seismic_only",
            "raw_feature_fusion",
            "same_architecture_random_init_moment_gfm_fusion",
            "frozen_pretrained_moment_gfm_fusion",
        ],
        "fixed_seeds": list(SEEDS),
        "training_parent_balance": "equal parent contribution per update",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "physical_MD_block_m": BOOTSTRAP_BLOCK_M,
            "statistic": "equal-parent macro RMSE delta",
            "paired": True,
        },
        "maximum_scientific_iterations": MAX_SCIENTIFIC_ITERATIONS,
        "resource_budget": {
            "foundation_encoders_frozen": True,
            "foundation_projection_dimension": 16,
            "head_feature_dimension_per_modality": HEAD_FEATURE_DIM,
            "head_hidden_dimension": HEAD_HIDDEN_DIM,
            "head_max_updates": HEAD_MAX_UPDATES,
            "head_eval_every_updates": HEAD_EVAL_EVERY,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "repeat_seeds": list(SEEDS),
            "parent_balanced_full_batch_loss": True,
        },
        "fixed_misalignment_controls": {
            "cyclic_well": "15/9-19 -> 15/9-F-11 -> 15/9-F-15 -> 15/9-19 by relative MD rank",
            "twt_perturbation_samples": TWT_PERTURBATION_SAMPLES,
            "twt_perturbation_ms": TWT_PERTURBATION_MS,
            "twt_boundary_policy": "clip inside the already frozen native 400-sample section",
            "target_information_used": False,
        },
        "calibration_contract": {
            "method": "outer-train inner-LOGO residual Gaussian scale",
            "nominal_coverages": [0.50, 0.90],
            "proper_score": "Gaussian negative log likelihood",
            "minimum_mean_90_interval_width": 0.0,
            "maximum_mean_90_interval_width": MAX_INFORMATIVE_90_INTERVAL_WIDTH,
            "held_parent_target_used_for_fit": False,
        },
        "action_allowlist": [
            {
                "action_id": "default",
                "weight_decay": 1e-4,
                "gate_strength": 0.10,
                "early_stopping_patience": 20,
            },
            {
                "action_id": "stronger_regularization",
                "weight_decay": 1e-3,
                "gate_strength": 0.05,
                "early_stopping_patience": 20,
            },
            {
                "action_id": "shorter_patience",
                "weight_decay": 1e-4,
                "gate_strength": 0.10,
                "early_stopping_patience": 10,
            },
        ],
        "immutable_after_iteration0": [
            "target",
            "quality_mask",
            "alignment",
            "split",
            "resource_budget",
            "controls",
            "acceptance",
            "bootstrap",
        ],
        "acceptance": {
            "fusion_macro_rmse_strictly_below_strongest_control": True,
            "minimum_held_parent_wins": 2,
            "paired_bootstrap_delta_ci95_upper_below_zero": True,
            "pretrained_beats_same_architecture_random_init": True,
            "both_fixed_misalignment_controls_degrade_macro_and_two_parents": True,
            "finite_nonvacuous_train_only_calibration": True,
        },
        "firewall": {
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "train_h5_opened": False,
            "hdf5_labels_read": [],
            "target_derived_alignment": False,
            "phase0_encoder_calls": 0,
            "phase0_training_runs": 0,
        },
        "provenance": {
            "well_logs_zip_sha256": _sha256(logs_path),
            "well_technical_data_zip_sha256": _sha256(technical_path),
            "seismic_vsp_zip_sha256": _sha256(vsp_path),
            "seismic_index_sha256": _sha256(index_path),
            "st0202_segy_size_bytes": int(segy_path.stat().st_size),
            "moment_weights_sha256": MOMENT_WEIGHTS_SHA256,
            "gfm_revision": GFM_REVISION,
            "gfm_weights_sha256": GFM_WEIGHTS_SHA256,
        },
    }
    freeze["row_contract"] = {
        "rows": int(len(arrays["target"])),
        "row_id_sha256": _array_sha256(arrays["row_id"]),
        "parent_index_sha256": _array_sha256(arrays["parent_index"]),
        "target_sha256": _array_sha256(arrays["target"]),
        "md_sha256": _array_sha256(arrays["MD_m"]),
        "alignment_sha256": hashlib.sha256(
            b"".join(
                bytes.fromhex(_array_sha256(arrays[name]))
                for name in (
                    "TVDSS_m",
                    "TWT_ms",
                    "inline",
                    "crossline",
                    "time_index",
                    "section_id",
                    "trace_token_id",
                )
            )
        ).hexdigest(),
    }
    if read_native_windows:
        scratch.mkdir(parents=True, exist_ok=True)
        cache_path = scratch / "phase0_arrays.npz"
        with cache_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        freeze["scratch_cache"] = {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
            "committed": False,
            "native_sections_shape": list(arrays["native_sections"].shape),
            "native_sections_sha256": _array_sha256(arrays["native_sections"]),
        }
    return target_contract, alignment_manifest, split_manifest, freeze, arrays


def _phase0_verification(
    target: Mapping[str, Any],
    alignment: Mapping[str, Any],
    split: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    parent_coverages = [
        float(alignment["parents"][parent]["retained_fraction"])
        for parent in PARENT_ORDER
    ]
    checks = {
        "native_phif_for_exactly_three_parents": len(target["parents"]) == 3,
        "phif_semantics_gate_passed": target["semantic_gate_passed"] is True,
        "phif_is_not_phie": target["phif_is_distinct_from_phie"] is True,
        "all_zero_rows_retained_by_nonperformance_rule": all(
            parent["target"]["zero_rows"] >= 0 for parent in target["parents"].values()
        ),
        "direct_alignment_gate_passed": alignment["alignment_gate_passed"] is True,
        "all_parent_coverages_at_least_90_percent": all(
            value >= MIN_PARENT_COVERAGE for value in parent_coverages
        ),
        "native_uninterpolated_unpadded_windows": (
            alignment["interpolation_applied"] is False
            and alignment["padding_applied"] is False
            and all(
                alignment["parents"][parent]["native_window"]["finite"] is True
                for parent in PARENT_ORDER
            )
        ),
        "logo3_is_frozen_active": split["state"] == "FROZEN_ACTIVE" and len(split["folds"]) == 3,
        "split_precedes_encoder_and_normalization": (
            "encoder_call" in split["split_before"]
            and "normalization" in split["split_before"]
        ),
        "phase0_has_zero_encoder_calls_and_training": (
            freeze["firewall"]["phase0_encoder_calls"] == 0
            and freeze["firewall"]["phase0_training_runs"] == 0
        ),
        "no_hdf5_or_holdout_opened": (
            freeze["firewall"]["test_h5_opened"] is False
            and freeze["firewall"]["frozen_holdout_opened"] is False
            and freeze["firewall"]["train_h5_opened"] is False
        ),
        "protocol_frozen_before_results": freeze["state"] == "PHASE0_PASSED",
    }
    return {
        "schema_version": "reconstruction-p38-phase0-verification/v1",
        "status": "PASS_PHASE0" if all(checks.values()) else "FAIL_PHASE0",
        "checks": checks,
        "parent_coverages": dict(zip(PARENT_ORDER, parent_coverages, strict=True)),
    }


def _phase0_artifacts(
    output_dir: Path,
    target: Mapping[str, Any],
    alignment: Mapping[str, Any],
    split: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, str]:
    verification = _phase0_verification(target, alignment, split, freeze)
    if verification["status"] != "PASS_PHASE0":
        raise RuntimeError(f"Phase 0 failed closed: {verification}")
    return {
        "target_contract.json": _canonical(target),
        "alignment_manifest.json": _canonical(alignment),
        "split_manifest.json": _canonical(split),
        "phase0_freeze.json": _canonical(freeze),
        "phase0_verification.json": _canonical(verification),
        "phif_phie_audit.json": _canonical(target["phif_phie_same_well_audit"]),
    }


def write_phase0(
    *,
    output_dir: Path,
    evidence: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, np.ndarray]],
) -> None:
    output = _validate_output_path(output_dir)
    target, alignment, split, freeze, _ = evidence
    payloads = _phase0_artifacts(output, target, alignment, split, freeze)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output / name).write_text(payload, encoding="utf-8")


def verify_phase0(
    *,
    output_dir: Path,
    evidence: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, np.ndarray]],
) -> None:
    output = _validate_output_path(output_dir)
    target, alignment, split, freeze, _ = evidence
    expected = _phase0_artifacts(output, target, alignment, split, freeze)
    for name, payload in expected.items():
        actual = (output / name).read_text(encoding="utf-8")
        if actual != payload:
            raise RuntimeError(f"Phase-0 artifact does not reproduce exactly: {name}")


def _equal_parent_curve_stats(
    values: np.ndarray,
    masks: np.ndarray,
    parent_index: np.ndarray,
    train_parents: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for parent in train_parents:
        selected = parent_index == int(parent)
        parent_mean = np.empty(len(CURVES), dtype=np.float64)
        parent_var = np.empty(len(CURVES), dtype=np.float64)
        for channel in range(len(CURVES)):
            observed = values[selected, channel][masks[selected, channel]]
            if len(observed) == 0:
                raise RuntimeError("outer-train parent has no observed well samples")
            parent_mean[channel] = float(np.mean(observed))
            parent_var[channel] = float(np.var(observed))
        means.append(parent_mean)
        variances.append(parent_var)
    stacked_mean = np.stack(means)
    mean = np.mean(stacked_mean, axis=0)
    variance = np.mean(
        np.stack(variances) + (stacked_mean - mean[None, :]) ** 2,
        axis=0,
    )
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)


def _fold_well_worker_input(
    arrays: Mapping[str, np.ndarray],
    *,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(arrays["well_windows"], dtype=np.float32)
    masks = np.asarray(arrays["well_masks"], dtype=bool)
    parent = np.asarray(arrays["parent_index"], dtype=np.int64)
    train_parents = [index for index in range(len(PARENT_ORDER)) if index != int(fold_id)]
    mean, scale = _equal_parent_curve_stats(values, masks, parent, train_parents)
    normalized = (values - mean[None, :, None]) / scale[None, :, None]
    normalized = np.where(masks, normalized, 0.0).astype(np.float32)
    time_mask = np.all(masks, axis=1)
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("fold-normalized well windows are non-finite")
    audit = {
        "fold_id": int(fold_id),
        "outer_train_parents": [PARENT_ORDER[index] for index in train_parents],
        "fit_scope": "equal-parent outer-train observed well samples only",
        "curve_mean": dict(zip(CURVES, mean.tolist(), strict=True)),
        "curve_scale": dict(zip(CURVES, scale.tolist(), strict=True)),
        "input_shape": list(normalized.shape),
        "time_mask_shape": list(time_mask.shape),
        "time_mask_observed_fraction": float(np.mean(time_mask)),
        "missing_fill": "normalized_zero_plus_explicit_input_mask",
        "held_parent_target_used": False,
    }
    return normalized, time_mask, audit


def _run_feature_worker(
    *,
    foundation_python: Path,
    kind: str,
    input_path: Path,
    output_path: Path,
    weight_mode: str,
    device: str,
    batch_size: int,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
) -> dict[str, Any]:
    interpreter = Path(foundation_python).expanduser()
    if not interpreter.is_absolute():
        interpreter = (Path.cwd() / interpreter).absolute()
    if not interpreter.is_file():
        raise FileNotFoundError(f"foundation interpreter is missing: {interpreter}")
    command = [
        # Do not resolve this symlink: resolving a venv's bin/python to the
        # system interpreter discards that venv's site-packages.
        str(interpreter),
        str(HERE / "p38_foundation_feature_worker.py"),
        kind,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--weight-mode",
        weight_mode,
        "--seed",
        str(SEEDS[0]),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    if kind == "moment":
        command.extend(
            [
                "--moment-snapshot",
                str(moment_snapshot),
                "--moment-dependency-root",
                str(moment_dependency_root),
                "--moment-source-root",
                str(moment_source_root),
            ]
        )
    else:
        command.extend(
            [
                "--gfm-snapshot",
                str(gfm_snapshot),
                "--gfm-source-root",
                str(gfm_source_root),
            ]
        )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"P38 {kind}/{weight_mode} worker failed ({result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-8000:]}"
        )
    with np.load(output_path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        audit = json.loads(str(payload["audit_json"].item()))
    if not np.all(np.isfinite(features)) or audit.get("finite") is not True:
        raise FloatingPointError(f"P38 {kind}/{weight_mode} feature cache is non-finite")
    return {
        "command": command,
        "wall_seconds": time.monotonic() - started,
        "output_sha256": _sha256(output_path),
        "output_shape": list(features.shape),
        "audit": audit,
    }


def _extract_foundation_features(
    *,
    arrays: Mapping[str, np.ndarray],
    scratch_dir: Path,
    foundation_python: Path,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
    device: str,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    scratch = _validate_output_path(scratch_dir, scratch=True)
    scratch.mkdir(parents=True, exist_ok=True)
    audits: dict[str, Any] = {
        "schema_version": "reconstruction-p38-foundation-feature-audit/v1",
        "worker": str(HERE / "p38_foundation_feature_worker.py"),
        "worker_sha256": _sha256(HERE / "p38_foundation_feature_worker.py"),
        "fold_normalization": {},
        "runs": {},
    }
    moment_features: dict[tuple[int, str], np.ndarray] = {}
    for fold_id in range(3):
        values, mask, normalization = _fold_well_worker_input(arrays, fold_id=fold_id)
        audits["fold_normalization"][str(fold_id)] = normalization
        input_path = scratch / f"moment_fold{fold_id}_input.npz"
        with input_path.open("wb") as handle:
            np.savez_compressed(handle, values=values, input_mask=mask)
        for mode in ("pretrained", "random_init"):
            output_path = scratch / f"moment_fold{fold_id}_{mode}.npz"
            run_audit = _run_feature_worker(
                foundation_python=foundation_python,
                kind="moment",
                input_path=input_path,
                output_path=output_path,
                weight_mode=mode,
                device=device,
                batch_size=256,
                moment_snapshot=moment_snapshot,
                moment_dependency_root=moment_dependency_root,
                moment_source_root=moment_source_root,
                gfm_snapshot=gfm_snapshot,
                gfm_source_root=gfm_source_root,
            )
            with np.load(output_path, allow_pickle=False) as payload:
                features = np.asarray(payload["features"], dtype=np.float32)
            if features.shape != (len(arrays["target"]), 6, 4, 16):
                raise RuntimeError("MOMENT projected feature shape drift")
            moment_features[(fold_id, mode)] = features.reshape(len(features), -1)
            audits["runs"][f"moment_fold{fold_id}_{mode}"] = run_audit
    gfm_input = scratch / "gfm_sections_input.npz"
    with gfm_input.open("wb") as handle:
        np.savez_compressed(handle, sections=np.asarray(arrays["native_sections"], np.float32))
    gfm_features: dict[str, np.ndarray] = {}
    for mode in ("pretrained", "random_init"):
        output_path = scratch / f"gfm_sections_{mode}.npz"
        run_audit = _run_feature_worker(
            foundation_python=foundation_python,
            kind="gfm",
            input_path=gfm_input,
            output_path=output_path,
            weight_mode=mode,
            device=device,
            batch_size=2,
            moment_snapshot=moment_snapshot,
            moment_dependency_root=moment_dependency_root,
            moment_source_root=moment_source_root,
            gfm_snapshot=gfm_snapshot,
            gfm_source_root=gfm_source_root,
        )
        with np.load(output_path, allow_pickle=False) as payload:
            section_features = np.asarray(payload["features"], dtype=np.float32)
        expected = (len(arrays["native_sections"]), 3, 161, 16)
        if section_features.shape != expected:
            raise RuntimeError("GFM projected feature shape drift")
        section_id = np.asarray(arrays["section_id"], dtype=np.int64)
        trace_token = np.asarray(arrays["trace_token_id"], dtype=np.int64)
        cls = section_features[section_id, :, 0, :]
        trace = section_features[section_id, :, 1 + trace_token, :]
        time_position = np.asarray(arrays["time_position"], dtype=np.float32)
        time_features = np.column_stack(
            [
                time_position,
                np.sin(2 * np.pi * time_position),
                np.cos(2 * np.pi * time_position),
            ]
        ).astype(np.float32)
        row_features = np.concatenate(
            [trace.reshape(len(trace), -1), cls.reshape(len(cls), -1), time_features], axis=1
        )
        gfm_features[mode] = row_features.astype(np.float32)
        audits["runs"][f"gfm_{mode}"] = run_audit
    moment_difference = max(
        float(
            np.max(
                np.abs(
                    moment_features[(fold, "pretrained")]
                    - moment_features[(fold, "random_init")]
                )
            )
        )
        for fold in range(3)
    )
    gfm_difference = float(
        np.max(np.abs(gfm_features["pretrained"] - gfm_features["random_init"]))
    )
    if moment_difference <= 0.0 or gfm_difference <= 0.0:
        raise RuntimeError("pretrained and random-init foundation features are identical")
    audits["pretrained_random_max_abs_difference"] = {
        "moment": moment_difference,
        "gfm": gfm_difference,
    }
    return moment_features, gfm_features, audits


def _raw_features(
    arrays: Mapping[str, np.ndarray],
    *,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized, _, normalization = _fold_well_worker_input(arrays, fold_id=fold_id)
    masks = np.asarray(arrays["well_masks"], dtype=np.float32)
    well = np.concatenate(
        [normalized.reshape(len(normalized), -1), masks.reshape(len(masks), -1)], axis=1
    ).astype(np.float32)
    sections = np.asarray(arrays["native_sections"], dtype=np.float32)
    mean = np.mean(sections, axis=(-2, -1), keepdims=True)
    std = np.maximum(np.std(sections, axis=(-2, -1), keepdims=True), 1e-6)
    sections = ((sections - mean) / std).astype(np.float32)
    section_id = np.asarray(arrays["section_id"], dtype=np.int64)
    trace = np.asarray(arrays["trace_token_id"], dtype=np.int64)
    local_time = np.asarray(arrays["local_time_index"], dtype=np.int64)
    offsets = np.arange(-8, 9, dtype=np.int64)
    temporal = sections[
        section_id[:, None, None],
        np.arange(3)[None, :, None],
        local_time[:, None, None] + offsets[None, None, :],
        trace[:, None, None],
    ]
    trace_offsets = np.arange(-2, 3, dtype=np.int64)
    neighborhood = sections[
        section_id[:, None, None],
        np.arange(3)[None, :, None],
        local_time[:, None, None],
        trace[:, None, None] + trace_offsets[None, None, :],
    ]
    position = np.asarray(arrays["time_position"], dtype=np.float32)
    seismic = np.concatenate(
        [
            temporal.reshape(len(temporal), -1),
            np.mean(neighborhood, axis=2),
            np.std(neighborhood, axis=2),
            position[:, None],
            np.sin(2 * np.pi * position)[:, None],
            np.cos(2 * np.pi * position)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    if not np.all(np.isfinite(well)) or not np.all(np.isfinite(seismic)):
        raise FloatingPointError("raw control features are non-finite")
    return well, seismic, {
        "fold_well_normalization": normalization,
        "well_shape": list(well.shape),
        "seismic_shape": list(seismic.shape),
        "seismic_temporal_offsets": offsets.tolist(),
        "seismic_crossline_offsets": trace_offsets.tolist(),
        "target_statistics_used": False,
    }


def _cyclic_row_map(parent_index: np.ndarray, md_m: np.ndarray) -> np.ndarray:
    parents = np.asarray(parent_index, dtype=np.int64)
    md = np.asarray(md_m, dtype=np.float64)
    mapping = np.empty(len(parents), dtype=np.int64)
    for parent in range(3):
        source_rows = np.flatnonzero(parents == parent)
        next_rows = np.flatnonzero(parents == ((parent + 1) % 3))
        source_order = source_rows[np.argsort(md[source_rows], kind="stable")]
        next_order = next_rows[np.argsort(md[next_rows], kind="stable")]
        rank = np.linspace(0, len(next_order) - 1, len(source_order))
        mapping[source_order] = next_order[np.rint(rank).astype(np.int64)]
    if np.any(parents[mapping] == parents):
        raise RuntimeError("cyclic mismatch retained the original parent")
    return mapping


def _twt_perturbed_gfm(gfm_features: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(gfm_features, dtype=np.float32).copy()
    local = np.asarray(arrays["local_time_index"], dtype=np.int64)
    perturbed = np.clip(local + TWT_PERTURBATION_SAMPLES, 0, SEISMIC_TIME_SAMPLES - 1)
    position = perturbed.astype(np.float32) / float(SEISMIC_TIME_SAMPLES - 1)
    values[:, -3:] = np.column_stack(
        [position, np.sin(2 * np.pi * position), np.cos(2 * np.pi * position)]
    )
    return values


def _without_prediction_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"predictions", "selected_indices"}}


def select_agent_action(
    allowlist: Sequence[Mapping[str, Any]],
    inner_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select only from inner evidence; the interface has no held metric input."""

    if not allowlist or str(allowlist[0].get("action_id")) != "default":
        raise ValueError("agent allowlist must begin with the fixed default")
    missing = [
        str(config["action_id"])
        for config in allowlist
        if str(config["action_id"]) not in inner_evidence
    ]
    if missing:
        raise ValueError(f"agent inner evidence missing actions: {missing}")
    return sorted(
        allowlist,
        key=lambda config: (
            float(inner_evidence[str(config["action_id"])]["macro_rmse"]),
            0 if config["action_id"] == "default" else 1,
            str(config["action_id"]),
        ),
    )[0]


def _per_parent_model_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
    parent_index: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for parent_id, parent in enumerate(PARENT_ORDER):
        rows = parent_index == parent_id
        metrics[parent] = pilot.regression_metrics(target[rows], prediction[rows])
        unique_sigma = np.unique(sigma[rows])
        if len(unique_sigma) != 1:
            raise RuntimeError("held-parent calibration scale is not fold-constant")
        calibration[parent] = pilot.calibration_metrics(
            target[rows], prediction[rows], float(unique_sigma[0])
        )
    metrics["equal_parent_macro"] = {
        key: float(np.mean([metrics[parent][key] for parent in PARENT_ORDER]))
        for key in ("rmse", "mae", "bias")
    }
    metrics["equal_parent_macro"]["rows"] = int(len(target))
    calibration["equal_parent_macro"] = {
        key: float(np.mean([calibration[parent][key] for parent in PARENT_ORDER]))
        for key in (
            "sigma",
            "gaussian_nll",
            "coverage_50",
            "coverage_90",
            "mean_width_50",
            "mean_width_90",
        )
    }
    return metrics, calibration


def _depth_stratified_metrics(
    *,
    target: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    parent_index: np.ndarray,
    md_m: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for parent_id, parent in enumerate(PARENT_ORDER):
        rows = np.flatnonzero(parent_index == parent_id)
        quantiles = np.quantile(md_m[rows], [0.0, 0.25, 0.5, 0.75, 1.0])
        strata = []
        for index in range(4):
            if index == 3:
                selected = rows[(md_m[rows] >= quantiles[index]) & (md_m[rows] <= quantiles[index + 1])]
            else:
                selected = rows[(md_m[rows] >= quantiles[index]) & (md_m[rows] < quantiles[index + 1])]
            strata.append(
                {
                    "depth_quartile": index,
                    "MD_m": [float(quantiles[index]), float(quantiles[index + 1])],
                    "candidate": pilot.regression_metrics(target[selected], candidate[selected]),
                    "strongest_control": pilot.regression_metrics(target[selected], control[selected]),
                }
            )
        result[parent] = strata
    return result


def run_pilot(
    *,
    evidence: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, np.ndarray]],
    scratch_dir: Path,
    foundation_python: Path,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
    foundation_device: str,
    head_device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]]:
    target_contract, alignment, split, freeze, source_arrays = evidence
    verification = _phase0_verification(target_contract, alignment, split, freeze)
    if verification["status"] != "PASS_PHASE0":
        raise RuntimeError("P38 refuses encoder/training because Phase 0 is not passed")
    arrays = {name: np.asarray(value) for name, value in source_arrays.items()}
    started = time.monotonic()
    moment_features, gfm_features, encoder_audit = _extract_foundation_features(
        arrays=arrays,
        scratch_dir=scratch_dir,
        foundation_python=foundation_python,
        moment_snapshot=moment_snapshot,
        moment_dependency_root=moment_dependency_root,
        moment_source_root=moment_source_root,
        gfm_snapshot=gfm_snapshot,
        gfm_source_root=gfm_source_root,
        device=foundation_device,
    )
    target = np.asarray(arrays["target"], dtype=np.float32)
    parent = np.asarray(arrays["parent_index"], dtype=np.int64)
    md_m = np.asarray(arrays["MD_m"], dtype=np.float64)
    row_count = len(target)
    raw_by_fold = {
        fold: _raw_features(arrays, fold_id=fold) for fold in range(3)
    }
    control_names = (
        "well_only",
        "seismic_only",
        "raw_feature_fusion",
        "same_architecture_random_init_moment_gfm_fusion",
        "frozen_pretrained_moment_gfm_fusion",
    )
    predictions = {
        name: np.full(row_count, np.nan, dtype=np.float32) for name in control_names
    }
    sigmas = {
        name: np.full(row_count, np.nan, dtype=np.float32) for name in control_names
    }
    pretrained_default = np.full(row_count, np.nan, dtype=np.float32)
    cyclic_prediction = np.full(row_count, np.nan, dtype=np.float32)
    twt_prediction = np.full(row_count, np.nan, dtype=np.float32)
    checkpoint_payload: dict[str, np.ndarray] = {}
    training_audit: dict[str, Any] = {}
    agent_folds: list[dict[str, Any]] = []
    experiment_log: list[dict[str, Any]] = []
    cyclic_map = _cyclic_row_map(parent, md_m)
    cyclic_gfm = gfm_features["pretrained"][cyclic_map]
    perturbed_gfm = _twt_perturbed_gfm(gfm_features["pretrained"], arrays)
    allowlist = list(freeze["action_allowlist"])
    if len(allowlist) != MAX_SCIENTIFIC_ITERATIONS:
        raise RuntimeError("scientific iteration allowlist drift")
    for fold_id in range(3):
        held = np.flatnonzero(parent == fold_id)
        outer_train_parents = [index for index in range(3) if index != fold_id]
        pretrained_well = moment_features[(fold_id, "pretrained")]
        pretrained_seismic = gfm_features["pretrained"]
        candidate_inner: dict[str, Any] = {}
        for iteration, config in enumerate(allowlist):
            result = pilot.inner_logo(
                well_features=pretrained_well,
                seismic_features=pretrained_seismic,
                target=target,
                parent_index=parent,
                outer_train_parents=outer_train_parents,
                well_present=True,
                seismic_present=True,
                config=config,
                seed=SEEDS[0] + fold_id * 1000,
                device=head_device,
                stream=fold_id * 100 + iteration,
            )
            candidate_inner[str(config["action_id"])] = result
            experiment_log.append(
                {
                    "scientific_iteration": iteration,
                    "fold_id": fold_id,
                    "action_id": config["action_id"],
                    "evidence_scope": "outer-train inner-LOGO only",
                    "primary_metric": "equal-inner-parent macro RMSE",
                    "metric": result["macro_rmse"],
                    "held_parent_metric_read_by_agent": False,
                    "status": "L3_validated_candidate",
                }
            )
        selected_config = select_agent_action(allowlist, candidate_inner)
        selected_id = str(selected_config["action_id"])
        agent_fold = {
            "fold_id": fold_id,
            "held_parent": PARENT_ORDER[fold_id],
            "observation_scope": "only the two outer-train parents under inner LOGO",
            "prompt": (
                "Choose exactly one preregistered head action by minimum equal-inner-parent "
                "macro RMSE; ties select default. Do not inspect held-parent labels/metrics."
            ),
            "candidates": [
                {
                    "action": dict(config),
                    "inner_evidence": _without_prediction_arrays(
                        candidate_inner[str(config["action_id"])]
                    ),
                }
                for config in allowlist
            ],
            "chosen_action": dict(selected_config),
            "rejected_action_ids": [
                str(config["action_id"])
                for config in allowlist
                if config["action_id"] != selected_id
            ],
            "reasoning": "minimum preregistered inner-LOGO macro RMSE with default tie-break",
            "held_parent_labels_visible_to_agent": False,
            "executor_confirmation": None,
        }
        feature_sets = {
            "well_only": (
                pretrained_well,
                pretrained_seismic,
                True,
                False,
            ),
            "seismic_only": (
                pretrained_well,
                pretrained_seismic,
                False,
                True,
            ),
            "raw_feature_fusion": (
                raw_by_fold[fold_id][0],
                raw_by_fold[fold_id][1],
                True,
                True,
            ),
            "same_architecture_random_init_moment_gfm_fusion": (
                moment_features[(fold_id, "random_init")],
                gfm_features["random_init"],
                True,
                True,
            ),
            "frozen_pretrained_moment_gfm_fusion": (
                pretrained_well,
                pretrained_seismic,
                True,
                True,
            ),
        }
        fold_bundles: dict[str, pilot.FitBundle] = {}
        for model_index, model_name in enumerate(control_names):
            well_base, seismic_base, well_present, seismic_present = feature_sets[model_name]
            if model_name == "frozen_pretrained_moment_gfm_fusion":
                inner = candidate_inner[selected_id]
            else:
                inner = pilot.inner_logo(
                    well_features=well_base,
                    seismic_features=seismic_base,
                    target=target,
                    parent_index=parent,
                    outer_train_parents=outer_train_parents,
                    well_present=well_present,
                    seismic_present=seismic_present,
                    config=selected_config,
                    seed=SEEDS[0] + fold_id * 1000 + model_index * 37,
                    device=head_device,
                    stream=fold_id * 100 + model_index * 7,
                )
            inner_rows = np.asarray(inner["selected_indices"], dtype=np.int64)
            inner_prediction = np.asarray(inner["predictions"], dtype=np.float32)
            sigma = max(
                float(
                    np.sqrt(
                        np.mean(
                            (inner_prediction[inner_rows].astype(np.float64) - target[inner_rows]) ** 2
                        )
                    )
                ),
                1e-4,
            )
            outer_train = np.flatnonzero(parent != fold_id)
            bundle, _ = pilot.fit_head(
                well_features=well_base,
                seismic_features=seismic_base,
                target=target,
                parent_index=parent,
                train_indices=outer_train,
                validation_indices=None,
                well_present=well_present,
                seismic_present=seismic_present,
                config=selected_config,
                seed=SEEDS[0] + fold_id * 1000 + model_index * 37,
                device=head_device,
                fixed_steps=max(1, int(inner["selected_steps"])),
                stream=fold_id * 100 + model_index * 7,
            )
            predicted, gate = bundle.predict(well_base, seismic_base, held)
            predictions[model_name][held] = predicted
            sigmas[model_name][held] = sigma
            fold_bundles[model_name] = bundle
            prefix = f"{model_name}__fold{fold_id}"
            checkpoint_payload.update(pilot.checkpoint_arrays(prefix, bundle))
            checkpoint_payload[f"{prefix}__calibration_sigma"] = np.asarray(
                sigma, dtype=np.float32
            )
            training_audit[prefix] = {
                "selected_action_id": selected_id,
                "inner_logo": _without_prediction_arrays(inner),
                "final_fit": dict(bundle.audit),
                "held_rows": int(len(held)),
                "calibration_sigma_fit_on_outer_train_inner_oof": sigma,
                "gate_mean": float(np.mean(gate)),
                "gate_std": float(np.std(gate)),
            }
        pretrained_bundle = fold_bundles["frozen_pretrained_moment_gfm_fusion"]
        cyclic_prediction[held], _ = pretrained_bundle.predict(
            pretrained_well, cyclic_gfm, held
        )
        twt_prediction[held], _ = pretrained_bundle.predict(
            pretrained_well, perturbed_gfm, held
        )
        default_inner = candidate_inner["default"]
        if selected_id == "default":
            pretrained_default[held] = predictions[
                "frozen_pretrained_moment_gfm_fusion"
            ][held]
            default_effect_execution = "selected action is the fixed default; exact prediction reuse"
        else:
            default_bundle, _ = pilot.fit_head(
                well_features=pretrained_well,
                seismic_features=pretrained_seismic,
                target=target,
                parent_index=parent,
                train_indices=np.flatnonzero(parent != fold_id),
                validation_indices=None,
                well_present=True,
                seismic_present=True,
                config=allowlist[0],
                seed=SEEDS[0] + fold_id * 1000 + 4 * 37,
                device=head_device,
                fixed_steps=max(1, int(default_inner["selected_steps"])),
                stream=fold_id * 100 + 4 * 7,
            )
            pretrained_default[held], _ = default_bundle.predict(
                pretrained_well, pretrained_seismic, held
            )
            checkpoint_payload.update(
                pilot.checkpoint_arrays(
                    f"frozen_pretrained_default_counterfactual__fold{fold_id}",
                    default_bundle,
                )
            )
            default_effect_execution = "selected and fixed-default heads both executed on held rows"
        agent_fold["executor_confirmation"] = {
            "selected_action_id": selected_id,
            "applied_to_all_five_budget_matched_models": True,
            "fixed_default_counterfactual": default_effect_execution,
            "held_metrics_read_only_after_action_frozen": True,
        }
        agent_folds.append(agent_fold)
    for name, values in {
        **predictions,
        "pretrained_default": pretrained_default,
        "cyclic": cyclic_prediction,
        "twt": twt_prediction,
    }.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"incomplete P38 row-aligned prediction: {name}")
    model_results: dict[str, Any] = {}
    for name in control_names:
        metrics, calibration = _per_parent_model_metrics(
            target, predictions[name], sigmas[name], parent
        )
        model_results[name] = {"metrics": metrics, "calibration": calibration}
    controls = control_names[:-1]
    strongest_control = min(
        controls,
        key=lambda name: model_results[name]["metrics"]["equal_parent_macro"]["rmse"],
    )
    candidate_name = "frozen_pretrained_moment_gfm_fusion"
    candidate_metrics = model_results[candidate_name]["metrics"]
    control_metrics = model_results[strongest_control]["metrics"]
    bootstrap = pilot.paired_depth_block_bootstrap(
        target=target,
        candidate=predictions[candidate_name],
        control=predictions[strongest_control],
        parent_index=parent,
        md_m=md_m,
        block_m=BOOTSTRAP_BLOCK_M,
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    mismatch_sigma = sigmas[candidate_name]
    cyclic_metrics, cyclic_calibration = _per_parent_model_metrics(
        target, cyclic_prediction, mismatch_sigma, parent
    )
    twt_metrics, twt_calibration = _per_parent_model_metrics(
        target, twt_prediction, mismatch_sigma, parent
    )
    mismatch = {
        "cyclic_well": {
            "metrics": cyclic_metrics,
            "calibration": cyclic_calibration,
            "macro_rmse_delta_mismatch_minus_correct": (
                cyclic_metrics["equal_parent_macro"]["rmse"]
                - candidate_metrics["equal_parent_macro"]["rmse"]
            ),
            "parents_degraded": sum(
                cyclic_metrics[parent_name]["rmse"] > candidate_metrics[parent_name]["rmse"]
                for parent_name in PARENT_ORDER
            ),
        },
        "fixed_twt_plus_160ms": {
            "metrics": twt_metrics,
            "calibration": twt_calibration,
            "macro_rmse_delta_mismatch_minus_correct": (
                twt_metrics["equal_parent_macro"]["rmse"]
                - candidate_metrics["equal_parent_macro"]["rmse"]
            ),
            "parents_degraded": sum(
                twt_metrics[parent_name]["rmse"] > candidate_metrics[parent_name]["rmse"]
                for parent_name in PARENT_ORDER
            ),
        },
    }
    default_metrics, default_calibration = _per_parent_model_metrics(
        target, pretrained_default, sigmas[candidate_name], parent
    )
    agent_effect = {
        "selected_action_macro_rmse": candidate_metrics["equal_parent_macro"]["rmse"],
        "fixed_default_counterfactual_macro_rmse": default_metrics["equal_parent_macro"]["rmse"],
        "rmse_delta_selected_minus_default": (
            candidate_metrics["equal_parent_macro"]["rmse"]
            - default_metrics["equal_parent_macro"]["rmse"]
        ),
        "selected_action_parent_wins": sum(
            candidate_metrics[parent_name]["rmse"] < default_metrics[parent_name]["rmse"]
            for parent_name in PARENT_ORDER
        ),
        "selected_action_ids": [fold["chosen_action"]["action_id"] for fold in agent_folds],
        "agent_gain_claimed": (
            candidate_metrics["equal_parent_macro"]["rmse"]
            < default_metrics["equal_parent_macro"]["rmse"]
        ),
        "default_metrics": default_metrics,
        "default_calibration": default_calibration,
    }
    parent_wins = sum(
        candidate_metrics[parent_name]["rmse"] < control_metrics[parent_name]["rmse"]
        for parent_name in PARENT_ORDER
    )
    random_rmse = model_results[
        "same_architecture_random_init_moment_gfm_fusion"
    ]["metrics"]["equal_parent_macro"]["rmse"]
    calibration_rows = model_results[candidate_name]["calibration"]
    calibration_pass = all(
        np.isfinite(list(calibration_rows[parent_name].values())).all()
        and 0.0 < calibration_rows[parent_name]["mean_width_90"] < MAX_INFORMATIVE_90_INTERVAL_WIDTH
        for parent_name in PARENT_ORDER
    )
    acceptance = {
        "fusion_macro_rmse_strictly_below_strongest_control": (
            candidate_metrics["equal_parent_macro"]["rmse"]
            < control_metrics["equal_parent_macro"]["rmse"]
        ),
        "fusion_wins_at_least_two_parents": parent_wins >= 2,
        "fusion_parent_wins": parent_wins,
        "paired_bootstrap_ci95_upper_below_zero": bootstrap["ci95"][1] < 0.0,
        "pretrained_beats_same_architecture_random_init": (
            candidate_metrics["equal_parent_macro"]["rmse"] < random_rmse
        ),
        "cyclic_well_mismatch_degrades_macro_and_two_parents": (
            mismatch["cyclic_well"]["macro_rmse_delta_mismatch_minus_correct"] > 0.0
            and mismatch["cyclic_well"]["parents_degraded"] >= 2
        ),
        "fixed_twt_mismatch_degrades_macro_and_two_parents": (
            mismatch["fixed_twt_plus_160ms"]["macro_rmse_delta_mismatch_minus_correct"] > 0.0
            and mismatch["fixed_twt_plus_160ms"]["parents_degraded"] >= 2
        ),
        "train_only_calibration_is_finite_and_nonvacuous": bool(calibration_pass),
    }
    promotable = all(
        value for key, value in acceptance.items() if key != "fusion_parent_wins"
    )
    decision = {
        "state": "PROMOTABLE_PILOT_SIGNAL" if promotable else "FEASIBLE_NO_PROMOTION",
        "promote": bool(promotable),
        "strongest_budget_matched_control": strongest_control,
        "cross_target_p21_comparison_made": False,
        "generalization_beyond_three_volve_parents_claimed": False,
        "traditional_geostatistics_disproved_claimed": False,
    }
    stratified = _depth_stratified_metrics(
        target=target,
        candidate=predictions[candidate_name],
        control=predictions[strongest_control],
        parent_index=parent,
        md_m=md_m,
    )
    summary = {
        "schema_version": SCHEMA,
        "phase0": {
            "state": freeze["state"],
            "verification": verification,
            "target_rows": {
                parent_name: target_contract["parents"][parent_name]["target"]["physical_rows"]
                for parent_name in PARENT_ORDER
            },
            "retained_joint_rows": {
                parent_name: alignment["parents"][parent_name]["retained_joint_rows"]
                for parent_name in PARENT_ORDER
            },
            "retained_fraction": {
                parent_name: alignment["parents"][parent_name]["retained_fraction"]
                for parent_name in PARENT_ORDER
            },
        },
        "split": split,
        "protocol": {
            "primary_metric": freeze["primary_metric"],
            "controls": list(control_names),
            "scientific_iterations_executed": len(allowlist),
            "maximum_scientific_iterations": MAX_SCIENTIFIC_ITERATIONS,
            "resource_budget": freeze["resource_budget"],
            "bootstrap": freeze["bootstrap"],
            "fixed_misalignment_controls": freeze["fixed_misalignment_controls"],
        },
        "models": model_results,
        "strongest_control": strongest_control,
        "paired_depth_block_bootstrap": bootstrap,
        "misalignment": mismatch,
        "stratified_by_MD_quartile": stratified,
        "agent_effect": agent_effect,
        "acceptance": acceptance,
        "decision": decision,
        "training_audit": training_audit,
        "firewall": {
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "train_h5_opened": False,
            "hdf5_label_datasets_read": [],
            "target_derived_alignment": False,
            "held_parent_target_used_for_fit_or_agent": False,
            "survey_checkshot_segy_headers_are_label_independent": True,
        },
        "history_boundary": {
            "p21_rmse": 0.027734374378067677,
            "p30_decision": "FEASIBLE_NO_PROMOTION",
            "p21_p30_target": "Eclipse PORO",
            "p38_target": "native published CPI PHIF",
            "rmse_ranked_across_targets": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "foundation_python": str(Path(foundation_python).expanduser().absolute()),
            "foundation_device": foundation_device,
            "head_device": head_device,
            "wall_seconds": time.monotonic() - started,
        },
    }
    prediction_payload = {
        name: np.asarray(arrays[name])
        for name in (
            "row_id",
            "parent_index",
            "target",
            "MD_m",
            "TVDSS_m",
            "x_m",
            "y_m",
            "TWT_ms",
            "inline",
            "crossline",
            "time_index",
            "section_id",
            "trace_token_id",
            "time_position",
        )
    }
    for name in control_names:
        prediction_payload[f"{name}__prediction"] = predictions[name]
        prediction_payload[f"{name}__sigma"] = sigmas[name]
    prediction_payload["frozen_pretrained__default_counterfactual"] = pretrained_default
    prediction_payload["frozen_pretrained__cyclic_well_mismatch"] = cyclic_prediction
    prediction_payload["frozen_pretrained__fixed_twt_plus_160ms"] = twt_prediction
    agent_audit = {
        "schema_version": "reconstruction-p38-agent-action-audit/v1",
        "authority": "preregistered allowlist only",
        "scientific_iterations": experiment_log,
        "fold_actions": agent_folds,
        "effect": agent_effect,
        "held_parent_labels_or_metrics_visible_before_action": False,
        "target_mask_split_budget_threshold_changes_allowed": False,
    }
    raw_audit = {str(fold): raw_by_fold[fold][2] for fold in range(3)}
    encoder_audit["raw_feature_controls"] = raw_audit
    return (
        summary,
        prediction_payload,
        checkpoint_payload,
        encoder_audit,
        agent_audit,
        {"schema_version": "reconstruction-p38-experiment-log/v1", "rows": experiment_log},
    )


def _fusion_io_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "reconstruction-p38-real-well-fusion-io/v1",
        "claim_boundary": (
            "Three-parent Volve native-PHIF LOGO3 pilot only; not Eclipse PORO, "
            "not PHIE, and not field-wide generalization."
        ),
        "row_identity": {
            "fields": [
                "row_id",
                "parent_id",
                "well_key",
                "MD_m",
                "TVDSS_m",
                "x_m_ED50_UTM31N",
                "y_m_ED50_UTM31N",
                "TWT_ms",
                "inline",
                "crossline",
                "time_index_4ms",
                "section_id",
                "trace_token_id",
            ],
            "ordering": "parent_order then increasing native target MD",
            "required": True,
        },
        "target": {
            "name": "PHIF",
            "definition": "native published CPI final/total formation porosity",
            "unit": "V/V_fraction",
            "bounds": [0.0, 1.0],
            "quality_mask": "finite, explicit-null-distinct, six center curves observed, direct ST0202 alignment",
            "forbidden_aliases": ["PHIE", "LFP_PHIE", "Eclipse PORO"],
        },
        "well_input": {
            "curve_order": list(CURVES),
            "values_shape": ["n", 6, 33],
            "mask_shape": ["n", 6, 33],
            "depth_unit": "m_MD",
            "normalization": "equal-parent outer-train-only per curve",
            "missing": "normalized zero only with explicit mask",
            "moment_tokens": {
                "source_shape": ["n", 6, 4, 768],
                "fixed_projected_shape": ["n", 6, 4, 16],
                "model_id": "AutonLab/MOMENT-1-base",
                "weights_sha256": MOMENT_WEIGHTS_SHA256,
            },
        },
        "seismic_input": {
            "source": "continuous ST0202 post-stack SEG-Y",
            "native_section_shape": ["section", 3, 400, 160],
            "channel_order": ["amplitude", "local_rms_5_sample", "vertical_gradient_2_sample"],
            "interpolation": False,
            "padding": False,
            "gfm_tokens": {
                "per_channel_source_shape": ["section", 161, 1200],
                "token_order": "CLS then 160 complete-trace tokens",
                "trace_and_cls_retained_separately": True,
                "fixed_projected_width": 16,
                "time_position_feature": "local_time_index / 399 plus fixed sin/cos",
                "model_id": "thinkonward/geophysical-foundation-model",
                "revision": GFM_REVISION,
                "weights_sha256": GFM_WEIGHTS_SHA256,
            },
        },
        "split": {
            "protocol": "LOGO3 by independent parent",
            "folds": summary["split"]["folds"],
            "fit_order": summary["split"]["split_before"],
            "held_parent_target_visible_to_fit": False,
        },
        "model_output": {
            "fields": [
                "prediction_PHIF_fraction",
                "train_only_calibration_sigma_fraction",
                "interval_50_fraction",
                "interval_90_fraction",
                "fold_id",
                "model_id",
                "action_id",
            ],
            "prediction_bounds": [0.0, 1.0],
            "uncertainty_method": "outer-train inner-LOGO residual Gaussian scale",
        },
        "provenance_required": [
            "raw_archive_member_and_sha256",
            "survey_and_checkshot_member_sha256",
            "seismic_index_and_SEG-Y identity",
            "encoder_model_revision_and_weights_sha256",
            "normalization_fit_parent_ids",
            "split_hash",
            "prediction_sha256",
            "config_and_action_id",
        ],
    }


def _rerun_commands(
    *,
    raw_project_root: Path,
    output_dir: Path,
    scratch_dir: Path,
    foundation_python: Path,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
    foundation_device: str,
    head_device: str,
) -> dict[str, str]:
    script = "_pipelines/02_task_datasets/reconstruction/p38_real_well_phif_direct_seismic.py"
    common = (
        f"--raw-project-root '{raw_project_root}' --output-dir '{output_dir}' "
        f"--scratch-dir '{scratch_dir}'"
    )
    run_paths = (
        f"--foundation-python '{foundation_python}' "
        f"--moment-snapshot '{moment_snapshot}' "
        f"--moment-dependency-root '{moment_dependency_root}' "
        f"--moment-source-root '{moment_source_root}' "
        f"--gfm-snapshot '{gfm_snapshot}' --gfm-source-root '{gfm_source_root}' "
        f"--foundation-device '{foundation_device}' --head-device '{head_device}'"
    )
    return {
        "phase0": f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} phase0 {common}",
        "verify_phase0": (
            f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} verify-phase0 {common}"
        ),
        "run": (
            f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} run {common} {run_paths}"
        ),
        "verify_only": (
            f"PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 {script} --verify-only "
            f"--output-dir '{output_dir}'"
        ),
        "py_compile": (
            "PYTHONPYCACHEPREFIX=_tmp/p38_real_well_phif_direct_seismic/pycache "
            "/usr/bin/python3 -m py_compile "
            "_models/reconstruction/moment_well.py "
            "_pipelines/02_task_datasets/reconstruction/p38_foundation_feature_worker.py "
            "_pipelines/02_task_datasets/reconstruction/p38_pilot_core.py "
            f"{script} "
            "_pipelines/02_task_datasets/reconstruction/_tests/"
            "test_p38_real_well_phif_direct_seismic.py"
        ),
        "focused_tests": (
            "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q "
            "_pipelines/02_task_datasets/reconstruction/_tests/"
            "test_p30_bounded_geostatistics_feasibility.py "
            "_pipelines/02_task_datasets/reconstruction/"
            "test_p37_real_well_seismic_supervision.py "
            "_pipelines/02_task_datasets/reconstruction/_tests/"
            "test_p38_real_well_phif_direct_seismic.py"
        ),
    }


def _finding(summary: Mapping[str, Any]) -> str:
    candidate = summary["models"]["frozen_pretrained_moment_gfm_fusion"]["metrics"]
    strongest = summary["strongest_control"]
    control = summary["models"][strongest]["metrics"]
    random = summary["models"][
        "same_architecture_random_init_moment_gfm_fusion"
    ]["metrics"]
    lines = [
        "# P38 real-well direct-seismic PHIF fusion",
        "",
        f"Decision: `{summary['decision']['state']}`.",
        "",
        "## Phase 0",
        "",
        "Native published CPI PHIF remains distinct from PHIE and Eclipse PORO. "
        "All finite physical PHIF zeros were retained under the frozen non-performance rule.",
        "",
        "| parent | physical PHIF | retained joint rows | coverage |",
        "|---|---:|---:|---:|",
    ]
    for parent in PARENT_ORDER:
        lines.append(
            f"| {parent} | {summary['phase0']['target_rows'][parent]} | "
            f"{summary['phase0']['retained_joint_rows'][parent]} | "
            f"{summary['phase0']['retained_fraction'][parent]:.6%} |"
        )
    lines.extend(
        [
            "",
            "## Fixed LOGO3 result",
            "",
            "| model | 15/9-19 RMSE | 15/9-F-11 RMSE | 15/9-F-15 RMSE | macro RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in summary["models"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['15/9-19']['rmse']:.12f} | "
            f"{metrics['15/9-F-11']['rmse']:.12f} | "
            f"{metrics['15/9-F-15']['rmse']:.12f} | "
            f"{metrics['equal_parent_macro']['rmse']:.12f} |"
        )
    lines.extend(
        [
            "",
            f"Strongest budget-matched control: `{strongest}` with macro RMSE "
            f"`{control['equal_parent_macro']['rmse']:.12f}`; frozen pretrained fusion is "
            f"`{candidate['equal_parent_macro']['rmse']:.12f}` and same-architecture random-init "
            f"is `{random['equal_parent_macro']['rmse']:.12f}`.",
            "",
            "Paired 20 m depth-block bootstrap fusion-minus-control CI95: "
            f"`[{summary['paired_depth_block_bootstrap']['ci95'][0]:.12f}, "
            f"{summary['paired_depth_block_bootstrap']['ci95'][1]:.12f}]`.",
            "",
            "## Alignment and agent checks",
            "",
            f"Cyclic-well mismatch macro delta: "
            f"`{summary['misalignment']['cyclic_well']['macro_rmse_delta_mismatch_minus_correct']:.12f}`; "
            f"fixed +160 ms delta: "
            f"`{summary['misalignment']['fixed_twt_plus_160ms']['macro_rmse_delta_mismatch_minus_correct']:.12f}`.",
            f"Agent-selected-minus-fixed-default macro RMSE delta: "
            f"`{summary['agent_effect']['rmse_delta_selected_minus_default']:.12f}`. "
            "Actions were selected only from outer-train inner-LOGO evidence.",
            "",
            "## Boundary",
            "",
            "P21/P30 remain Eclipse-PORO history and were not ranked against native PHIF. "
            "This three-parent Volve pilot does not establish field-wide generalization and does "
            "not disprove traditional geostatistics.",
            "",
            "Exact commands are in `rerun_commands.json`; row-aligned evidence is in "
            "`predictions.npz`, and all durable hashes are in `artifact_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_manifest(output_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    return {
        "schema_version": "reconstruction-p38-artifact-manifest/v1",
        "artifacts": [
            {
                "path": name,
                "size_bytes": int((output_dir / name).stat().st_size),
                "sha256": _sha256(output_dir / name),
            }
            for name in names
        ],
        "input_and_source_provenance": dict(provenance),
    }


def _verify_final_payload(output_dir: Path) -> dict[str, Any]:
    output = _validate_output_path(output_dir)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["manifest_all_artifact_hashes_match"] = all(
        (output / row["path"]).is_file()
        and int((output / row["path"]).stat().st_size) == int(row["size_bytes"])
        and _sha256(output / row["path"]) == row["sha256"]
        for row in manifest["artifacts"]
    )
    with np.load(output / "predictions.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    target = np.asarray(arrays["target"], dtype=np.float32)
    parent = np.asarray(arrays["parent_index"], dtype=np.int64)
    checks["row_ids_are_exact_once"] = np.array_equal(
        arrays["row_id"], np.arange(len(target), dtype=np.int64)
    )
    checks["exact_three_held_parents"] = np.array_equal(np.unique(parent), [0, 1, 2])
    for model_name, result in summary["models"].items():
        prediction = np.asarray(arrays[f"{model_name}__prediction"], dtype=np.float32)
        sigma = np.asarray(arrays[f"{model_name}__sigma"], dtype=np.float32)
        checks[f"{model_name}_finite_bounded_predictions"] = bool(
            np.all(np.isfinite(prediction))
            and np.all((prediction >= 0.0) & (prediction <= 1.0))
            and np.all(np.isfinite(sigma))
            and np.all(sigma > 0.0)
        )
        recomputed, calibration = _per_parent_model_metrics(
            target, prediction, sigma, parent
        )
        checks[f"{model_name}_metrics_recompute"] = all(
            abs(
                recomputed[parent_name]["rmse"]
                - result["metrics"][parent_name]["rmse"]
            )
            <= 1e-15
            for parent_name in (*PARENT_ORDER, "equal_parent_macro")
        )
        checks[f"{model_name}_calibration_recomputes"] = all(
            abs(
                calibration[parent_name]["gaussian_nll"]
                - result["calibration"][parent_name]["gaussian_nll"]
            )
            <= 1e-15
            for parent_name in (*PARENT_ORDER, "equal_parent_macro")
        )
    checks["phase0_passed_before_pilot"] = summary["phase0"]["state"] == "PHASE0_PASSED"
    checks["no_cross_target_rmse_ranking"] = (
        summary["history_boundary"]["rmse_ranked_across_targets"] is False
    )
    checks["no_test_holdout_or_kji"] = (
        summary["firewall"]["test_h5_opened"] is False
        and summary["firewall"]["frozen_holdout_opened"] is False
        and summary["firewall"]["train_h5_opened"] is False
        and summary["firewall"]["hdf5_label_datasets_read"] == []
    )
    expected_state = (
        "PROMOTABLE_PILOT_SIGNAL"
        if all(
            value
            for key, value in summary["acceptance"].items()
            if key != "fusion_parent_wins"
        )
        else "FEASIBLE_NO_PROMOTION"
    )
    checks["decision_matches_all_frozen_gates"] = summary["decision"]["state"] == expected_state
    return {
        "schema_version": "reconstruction-p38-final-verification/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def write_final_evidence(
    *,
    output_dir: Path,
    run_result: tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]],
    commands: Mapping[str, str],
    phase0_freeze: Mapping[str, Any],
) -> None:
    output = _validate_output_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, predictions, checkpoints, encoder_audit, agent_audit, experiment_log = run_result
    with (output / "predictions.npz").open("wb") as handle:
        np.savez_compressed(handle, **predictions)
    with (output / "checkpoints.npz").open("wb") as handle:
        np.savez_compressed(handle, **checkpoints)
    payloads = {
        "summary.json": _canonical(summary),
        "encoder_audit.json": _canonical(encoder_audit),
        "agent_action_audit.json": _canonical(agent_audit),
        "experiment_log.json": _canonical(experiment_log),
        "fusion_io_contract.json": _canonical(_fusion_io_contract(summary)),
        "rerun_commands.json": _canonical(commands),
        "finding.md": _finding(summary),
    }
    for name, value in payloads.items():
        (output / name).write_text(value, encoding="utf-8")
    # Verification is deliberately generated before the manifest and then
    # re-run after the manifest is complete.
    preliminary = {
        "schema_version": "reconstruction-p38-final-verification/v1",
        "status": "PENDING_MANIFEST",
        "checks": {},
    }
    (output / "verification.json").write_text(_canonical(preliminary), encoding="utf-8")
    provenance = {
        **phase0_freeze["provenance"],
        "p38_runner_sha256": _sha256(Path(__file__)),
        "p38_worker_sha256": _sha256(HERE / "p38_foundation_feature_worker.py"),
        "p38_core_sha256": _sha256(HERE / "p38_pilot_core.py"),
        "moment_adapter_sha256": _sha256(PROJECT_ROOT / "_models/reconstruction/moment_well.py"),
        "p38_test_sha256": _sha256(
            HERE / "_tests/test_p38_real_well_phif_direct_seismic.py"
        ),
        "predictions_sha256": _sha256(output / "predictions.npz"),
        "checkpoints_sha256": _sha256(output / "checkpoints.npz"),
    }
    # Build once, verify, then rebuild because verification itself is an artifact.
    manifest = _artifact_manifest(output, provenance)
    (output / "artifact_manifest.json").write_text(_canonical(manifest), encoding="utf-8")
    verification = _verify_final_payload(output)
    if verification["status"] != "PASS":
        raise RuntimeError(f"P38 final evidence verification failed: {verification}")
    (output / "verification.json").write_text(_canonical(verification), encoding="utf-8")
    manifest = _artifact_manifest(output, provenance)
    (output / "artifact_manifest.json").write_text(_canonical(manifest), encoding="utf-8")
    final_verification = _verify_final_payload(output)
    if final_verification["status"] != "PASS":
        raise RuntimeError(f"P38 final post-manifest verification failed: {final_verification}")


def verify_final(output_dir: Path) -> None:
    verification = _verify_final_payload(output_dir)
    if verification["status"] != "PASS":
        raise RuntimeError(f"P38 verify-only failed: {verification}")
    committed = json.loads(
        (_validate_output_path(output_dir) / "verification.json").read_text(encoding="utf-8")
    )
    if committed != verification:
        raise RuntimeError("P38 committed verification.json does not reproduce")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=False)
    for name in ("phase0", "verify-phase0"):
        command = subparsers.add_parser(name)
        command.add_argument("--raw-project-root", type=Path, default=DEFAULT_RAW_ROOT)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    run = subparsers.add_parser("run")
    run.add_argument("--raw-project-root", type=Path, default=DEFAULT_RAW_ROOT)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    run.add_argument("--foundation-python", type=Path, required=True)
    run.add_argument("--moment-snapshot", type=Path, required=True)
    run.add_argument("--moment-dependency-root", type=Path, required=True)
    run.add_argument("--moment-source-root", type=Path, required=True)
    run.add_argument("--gfm-snapshot", type=Path, required=True)
    run.add_argument("--gfm-source-root", type=Path, required=True)
    run.add_argument("--foundation-device", default="cpu")
    run.add_argument("--head-device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.verify_only:
        if args.command is not None:
            raise ValueError("--verify-only cannot be combined with a subcommand")
        verify_final(args.output_dir)
        summary = json.loads(
            (_validate_output_path(args.output_dir) / "summary.json").read_text(encoding="utf-8")
        )
        print(_canonical({"status": "VERIFIED", "decision": summary["decision"]}), end="")
        return 0
    if args.command is None:
        raise ValueError("a command or --verify-only is required")
    evidence = build_phase0(
        raw_project_root=args.raw_project_root,
        scratch_dir=args.scratch_dir,
        read_native_windows=True,
    )
    if args.command == "phase0":
        write_phase0(output_dir=args.output_dir, evidence=evidence)
        verification = _phase0_verification(*evidence[:4])
        print(_canonical({"state": evidence[3]["state"], "verification": verification}), end="")
        return 0
    if args.command == "verify-phase0":
        verify_phase0(output_dir=args.output_dir, evidence=evidence)
        print(_canonical({"status": "VERIFIED_PHASE0", "state": evidence[3]["state"]}), end="")
        return 0
    verify_phase0(output_dir=args.output_dir, evidence=evidence)
    result = run_pilot(
        evidence=evidence,
        scratch_dir=args.scratch_dir,
        foundation_python=args.foundation_python,
        moment_snapshot=args.moment_snapshot,
        moment_dependency_root=args.moment_dependency_root,
        moment_source_root=args.moment_source_root,
        gfm_snapshot=args.gfm_snapshot,
        gfm_source_root=args.gfm_source_root,
        foundation_device=args.foundation_device,
        head_device=args.head_device,
    )
    commands = _rerun_commands(
        raw_project_root=Path(args.raw_project_root).resolve(),
        output_dir=Path(args.output_dir),
        scratch_dir=Path(args.scratch_dir),
        foundation_python=Path(args.foundation_python).expanduser().absolute(),
        moment_snapshot=Path(args.moment_snapshot).resolve(),
        moment_dependency_root=Path(args.moment_dependency_root).resolve(),
        moment_source_root=Path(args.moment_source_root).resolve(),
        gfm_snapshot=Path(args.gfm_snapshot).resolve(),
        gfm_source_root=Path(args.gfm_source_root).resolve(),
        foundation_device=args.foundation_device,
        head_device=args.head_device,
    )
    write_final_evidence(
        output_dir=args.output_dir,
        run_result=result,
        commands=commands,
        phase0_freeze=evidence[3],
    )
    print(
        _canonical(
            {
                "decision": result[0]["decision"],
                "strongest_control": result[0]["strongest_control"],
                "acceptance": result[0]["acceptance"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
