"""Real-data assembly for the Volve reservoir-property track.

The split is frozen at the parent-well-family level before any interpolation,
windowing, seismic extraction, or fitted statistics.  Only measured/input log
curves are admitted; interpreted targets and their aliases are explicitly
forbidden from the input tensor.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lasio
import numpy as np
import segyio
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from dataset_io import save_split  # noqa: E402


_LOCAL_VOLVE_DIR = PROJECT_ROOT / "_sandbox" / "volve_data"
_REPO_ROOT_VOLVE_DIR = PROJECT_ROOT.parents[2] / "_sandbox" / "volve_data"
VOLVE_DIR = _LOCAL_VOLVE_DIR if _LOCAL_VOLVE_DIR.is_dir() else _REPO_ROOT_VOLVE_DIR
WELL_LOG_ZIP = VOLVE_DIR / "Volve_Well_logs.zip"
INTERP_DIR = VOLVE_DIR / "_extracted_interp" / "Geophysical_Interpretations"
PICKS_PATH = INTERP_DIR / "Wells" / "Well_picks_Volve_v1.dat"
HORIZON_DIR = INTERP_DIR / "Horizons" / "Horizons_TWT" / "Official_Horizons"
SEGY_PATH = (
    VOLVE_DIR
    / "_extracted_seismic/ST0202/Stacks/"
    "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
)
SEISMIC_INDEX_PATH = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"

OUTPUT_DIR = HERE / "_outputs"
GUARD_PATH = OUTPUT_DIR / "guard.npz"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
BUILD_REPORT_PATH = OUTPUT_DIR / "build_report.json"

INPUT_CHANNELS = ("GR", "RT", "NPHI", "RHOB")
TARGET_NAMES = ("PHIF", "log1p(KLOGH)", "SW")
FORBIDDEN_INPUT_CURVES = {
    "PHIF", "PHIE", "LFP_PHIE", "KLOGH", "KLOGH_NEW", "KLOGV", "SW",
    "BVW", "SWIRR", "VSH", "LFP_VSH", "SAND_FLAG", "COAL_FLAG", "CARB_FLAG",
}
DISCOVERY_ALIASES = {
    "GR": "LFP_GR",
    "RT": "LFP_RT",
    "NPHI": "LFP_NPHI",
    "RHOB": "LFP_RHOB",
}
PRODUCTION_ALIASES = {name: name for name in INPUT_CHANNELS}


@dataclass(frozen=True)
class WellSource:
    well_id: str
    family_id: str
    label_member: str
    input_member: str
    input_aliases: dict[str, str]


@dataclass
class LasTable:
    depth_m: np.ndarray
    curves: dict[str, np.ndarray]


def canonical_well_id(raw: str) -> str:
    """Normalise archive/pick names while preserving the branch identifier."""
    value = raw.strip().replace("NO ", "", 1).replace("_", "/", 1)
    value = re.sub(r"\s+", " ", value)
    return value


def parent_well_family(raw: str) -> str:
    """Group a mother well and every sidetrack/branch as one leakage unit."""
    well = canonical_well_id(raw)
    discovery = re.match(r"^(15/9-19)(?:\s|$)", well)
    if discovery:
        return discovery.group(1)
    production = re.match(r"^(15/9-F-\d+)(?:\s|$)", well)
    if production:
        return production.group(1)
    raise ValueError(f"无法从井眼名推导母井家族: {raw!r}")


def deterministic_family_split(families: Iterable[str]) -> dict[str, str]:
    """Stable 60/20/20-ish train/guard/test assignment by SHA-256 order."""
    unique = sorted(set(families), key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    if len(unique) < 4:
        raise RuntimeError(
            f"只有{len(unique)}个母井家族，无法形成至少2个train+1个guard+1个test家族"
        )
    n_test = max(1, int(round(len(unique) * 0.2)))
    n_guard = max(1, int(round(len(unique) * 0.2)))
    n_train = len(unique) - n_test - n_guard
    if n_train < 2:
        raise RuntimeError(f"确定性划分后train只有{n_train}个母井家族")
    result: dict[str, str] = {}
    for family in unique[:n_train]:
        result[family] = "train"
    for family in unique[n_train:n_train + n_guard]:
        result[family] = "guard"
    for family in unique[n_train + n_guard:]:
        result[family] = "test"
    return result


def _find_unique(paths: list[str], predicate, description: str) -> str:
    hits = [path for path in paths if predicate(path)]
    if len(hits) != 1:
        raise RuntimeError(f"{description} 应唯一命中，实际{len(hits)}个: {hits}")
    return hits[0]


def discover_well_sources(zf: zipfile.ZipFile) -> list[WellSource]:
    """Discover the 12 wells with the approved three interpreted targets."""
    members = zf.namelist()
    label_members = [
        path for path in members
        if path.lower().endswith(".las")
        and "/05.PETROPHYSICAL INTERPRETATION/" in path
        and (
            path.endswith("WLC_PETRO_COMPUTED_OUTPUT_1.LAS")
            or ("/CPI/" in path and path.lower().endswith("_cpi.las"))
        )
    ]
    sources: list[WellSource] = []
    for label_member in sorted(label_members):
        parts = label_member.split("/")
        well_archive = parts[2]
        well_id = canonical_well_id(well_archive)
        if "/CPI/" in label_member:
            input_member = _find_unique(
                members,
                lambda p, prefix=f"Well_logs/06.LFP/{well_archive}/":
                    p.startswith(prefix) and p.lower().endswith("_lfp.las"),
                f"{well_id} LFP输入",
            )
            aliases = DISCOVERY_ALIASES
        else:
            input_member = _find_unique(
                members,
                lambda p, prefix=f"Well_logs/05.PETROPHYSICAL INTERPRETATION/{well_archive}/":
                    p.startswith(prefix) and p.endswith("WLC_PETRO_COMPUTED_INPUT_1.LAS"),
                f"{well_id} WLC输入",
            )
            aliases = PRODUCTION_ALIASES
        sources.append(
            WellSource(
                well_id=well_id,
                family_id=parent_well_family(well_id),
                label_member=label_member,
                input_member=input_member,
                input_aliases=dict(aliases),
            )
        )
    if len(sources) != 12:
        raise RuntimeError(f"预期12个PHIF/KLOGH/SW标签井眼，实际发现{len(sources)}个")
    return sources


def _read_las(zf: zipfile.ZipFile, member: str) -> LasTable:
    text = zf.read(member).decode("latin-1", errors="replace")
    las = lasio.read(io.StringIO(text), engine="normal", ignore_header_errors=True)
    if not las.curves:
        raise RuntimeError(f"LAS无曲线: {member}")
    depth_name = las.curves[0].mnemonic
    depth = np.asarray(las[depth_name], dtype=np.float64)
    null_value = float(las.well.NULL.value) if "NULL" in las.well else -999.25
    curves: dict[str, np.ndarray] = {}
    for curve in las.curves[1:]:
        name = curve.mnemonic.upper()
        values = np.asarray(las[curve.mnemonic], dtype=np.float64)
        values[np.isclose(values, null_value)] = np.nan
        curves[name] = values
    return LasTable(depth_m=depth, curves=curves)


def transform_targets(phif: np.ndarray, klogh: np.ndarray, sw: np.ndarray) -> np.ndarray:
    phif = np.asarray(phif, dtype=np.float64)
    klogh = np.asarray(klogh, dtype=np.float64)
    sw = np.asarray(sw, dtype=np.float64)
    if np.any(klogh < 0):
        raise ValueError("KLOGH含负值，log1p不可按合同执行")
    return np.column_stack([phif, np.log1p(klogh), sw])


def inverse_targets(transformed: np.ndarray) -> np.ndarray:
    transformed = np.asarray(transformed, dtype=np.float64)
    result = transformed.copy()
    result[..., 1] = np.expm1(result[..., 1])
    return result


def parse_well_picks(path: Path = PICKS_PATH) -> dict[str, list[dict[str, float | str]]]:
    """Parse official fixed-width well picks without inventing missing columns."""
    spans: list[tuple[int, int]] | None = None
    wells: dict[str, list[dict[str, float | str]]] = {}
    for line in path.read_text(errors="replace").splitlines(keepends=True):
        if re.match(r"^\s*-{5,}", line):
            spans = [(m.start(), m.end()) for m in re.finditer(r"-+", line)]
            continue
        if spans is None or not line.strip() or line.startswith("Well NO"):
            continue
        if line.lstrip().startswith("Well name"):
            continue
        cols = [line[start:end].strip() for start, end in spans]
        if len(cols) < 12:
            continue
        well, surface, _obs, _qlf, md, _tvd, _tvdss, twt, _dip, _azi, east, north = cols[:12]
        if not well or not md or not twt or not east or not north:
            continue
        try:
            record = {
                "surface": surface,
                "md": float(md),
                "twt_ms": float(twt),
                "easting": float(east),
                "northing": float(north),
            }
        except ValueError:
            continue
        wells.setdefault(canonical_well_id(well), []).append(record)
    return wells


def _deduplicated_tie(picks: list[dict[str, float | str]]) -> dict[str, np.ndarray]:
    ordered = sorted(picks, key=lambda row: float(row["md"]))
    unique: list[dict[str, float | str]] = []
    seen: set[float] = set()
    for row in ordered:
        md = float(row["md"])
        if md not in seen:
            seen.add(md)
            unique.append(row)
    if len(unique) < 2:
        raise RuntimeError(f"有效MD/TWT/XY拾取只有{len(unique)}个，无法建立弱标定")
    return {
        "md": np.asarray([row["md"] for row in unique], dtype=np.float64),
        "twt_ms": np.asarray([row["twt_ms"] for row in unique], dtype=np.float64),
        "easting": np.asarray([row["easting"] for row in unique], dtype=np.float64),
        "northing": np.asarray([row["northing"] for row in unique], dtype=np.float64),
    }


def _interp_extrap(query: np.ndarray, known_x: np.ndarray, known_y: np.ndarray) -> np.ndarray:
    result = np.interp(query, known_x, known_y)
    below = query < known_x[0]
    above = query > known_x[-1]
    if below.any():
        slope = (known_y[1] - known_y[0]) / (known_x[1] - known_x[0])
        result[below] = known_y[0] + slope * (query[below] - known_x[0])
    if above.any():
        slope = (known_y[-1] - known_y[-2]) / (known_x[-1] - known_x[-2])
        result[above] = known_y[-1] + slope * (query[above] - known_x[-1])
    return result


def weak_tie(depth_m: np.ndarray, picks: list[dict[str, float | str]]) -> dict[str, np.ndarray]:
    tie = _deduplicated_tie(picks)
    return {
        key: _interp_extrap(depth_m, tie["md"], tie[key])
        for key in ("twt_ms", "easting", "northing")
    }


class HuginSurfaces:
    def __init__(self, top_path: Path, base_path: Path) -> None:
        self.top_xy, self.top_twt = self._load(top_path)
        self.base_xy, self.base_twt = self._load(base_path)
        self.top_tree = cKDTree(self.top_xy)
        self.base_tree = cKDTree(self.base_xy)

    @staticmethod
    def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("#") or line.count(",") < 4:
                continue
            values = np.fromstring(line, sep=",", dtype=np.float64)
            if values.size >= 5:
                rows.append(values[:5])
        data = np.asarray(rows, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 5:
            raise RuntimeError(f"Hugin层位格式异常: {path} -> {data.shape}")
        return data[:, 2:4], data[:, 4]

    def bounds_at(self, easting: np.ndarray, northing: np.ndarray) -> dict[str, np.ndarray]:
        xy = np.column_stack([easting, northing])
        top_dist, top_idx = self.top_tree.query(xy, k=1)
        base_dist, base_idx = self.base_tree.query(xy, k=1)
        top = self.top_twt[top_idx]
        base = self.base_twt[base_idx]
        return {
            "top_twt_ms": top,
            "base_twt_ms": base,
            "top_distance_m": top_dist,
            "base_distance_m": base_dist,
        }


def _horizon_paths() -> tuple[Path, Path]:
    tops = sorted(HORIZON_DIR.glob("Hugin_Fm_Top+*+TIME.dat"))
    bases = sorted(HORIZON_DIR.glob("Hugin_Fm_Base+*+TIME.dat"))
    if len(tops) != 1 or len(bases) != 1:
        raise RuntimeError(f"Hugin官方TWT面不唯一: top={tops}, base={bases}")
    return tops[0], bases[0]


def _load_seismic_index() -> dict[str, np.ndarray]:
    if not SEISMIC_INDEX_PATH.exists():
        raise FileNotFoundError(f"Layer1地震索引不存在: {SEISMIC_INDEX_PATH}")
    with np.load(SEISMIC_INDEX_PATH, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _utm_to_il_xl(easting: float, northing: float, index: dict[str, np.ndarray]) -> tuple[int, int]:
    affine = index["affine_il_xl_to_xy"]
    matrix = affine[:, :2]
    rhs = np.asarray([easting - affine[0, 2], northing - affine[1, 2]])
    il_f, xl_f = np.linalg.solve(matrix, rhs)
    il = int(np.clip(round(float(il_f)), int(index["il_min"]), int(index["il_max"])))
    xl = int(np.clip(round(float(xl_f)), int(index["xl_min"]), int(index["xl_max"])))
    return il, xl


class SeismicPatchReader:
    """Lazy real SEG-Y trace reader with a small in-process trace cache."""
    def __init__(self, index: dict[str, np.ndarray], spatial_radius: int = 1, time_radius: int = 4):
        self.index = index
        self.spatial_radius = spatial_radius
        self.time_radius = time_radius
        self.handle = segyio.open(str(SEGY_PATH), "r", ignore_geometry=True)
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "SeismicPatchReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _trace(self, il: int, xl: int) -> np.ndarray:
        key = (il, xl)
        if key not in self.cache:
            n_xl = int(self.index["n_xl"])
            trace_index = (il - int(self.index["il_min"])) * n_xl + (xl - int(self.index["xl_min"]))
            self.cache[key] = np.asarray(self.handle.trace[trace_index], dtype=np.float32)
        return self.cache[key]

    def patch(self, il: int, xl: int, twt_ms: float) -> tuple[np.ndarray, int]:
        samples_ms = self.index["samples_ms"]
        time_idx = int(np.clip(np.searchsorted(samples_ms, twt_ms), 0, len(samples_ms) - 1))
        side = 2 * self.spatial_radius + 1
        width = 2 * self.time_radius + 1
        patch = np.zeros((side, side, width), dtype=np.float32)
        for i_offset in range(-self.spatial_radius, self.spatial_radius + 1):
            for x_offset in range(-self.spatial_radius, self.spatial_radius + 1):
                i = il + i_offset
                x = xl + x_offset
                if not (int(self.index["il_min"]) <= i <= int(self.index["il_max"])):
                    continue
                if not (int(self.index["xl_min"]) <= x <= int(self.index["xl_max"])):
                    continue
                trace = self._trace(i, x)
                start = max(0, time_idx - self.time_radius)
                stop = min(len(trace), time_idx + self.time_radius + 1)
                target_start = start - (time_idx - self.time_radius)
                patch[
                    i_offset + self.spatial_radius,
                    x_offset + self.spatial_radius,
                    target_start:target_start + stop - start,
                ] = trace[start:stop]
        return patch, time_idx


def interpolate_log_sequence(
    table: LasTable,
    aliases: dict[str, str],
    query_depths: np.ndarray,
    max_gap_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only across short observed gaps and return an explicit mask."""
    values = np.zeros((len(query_depths), len(INPUT_CHANNELS)), dtype=np.float32)
    mask = np.zeros_like(values)
    for channel_index, channel in enumerate(INPUT_CHANNELS):
        source_name = aliases[channel].upper()
        if source_name in FORBIDDEN_INPUT_CURVES:
            raise RuntimeError(f"输入曲线命中泄漏黑名单: {source_name}")
        source = table.curves.get(source_name)
        if source is None:
            continue
        valid = np.isfinite(table.depth_m) & np.isfinite(source)
        depth = table.depth_m[valid]
        curve = source[valid]
        if len(depth) < 2:
            continue
        order = np.argsort(depth)
        depth = depth[order]
        curve = curve[order]
        right = np.searchsorted(depth, query_depths, side="left")
        for row, position in enumerate(right):
            if position < len(depth) and np.isclose(depth[position], query_depths[row], atol=1e-6):
                values[row, channel_index] = curve[position]
                mask[row, channel_index] = 1.0
                continue
            if position == 0 or position == len(depth):
                continue
            left = position - 1
            gap = depth[position] - depth[left]
            if gap <= 0 or gap > max_gap_m:
                continue
            weight = (query_depths[row] - depth[left]) / gap
            values[row, channel_index] = curve[left] * (1.0 - weight) + curve[position] * weight
            mask[row, channel_index] = 1.0
    return values, mask


def _subsample_depth_indices(depth: np.ndarray, eligible: np.ndarray, step_m: float) -> list[int]:
    selected: list[int] = []
    last_depth = -np.inf
    for index in np.flatnonzero(eligible):
        current = float(depth[index])
        if current >= last_depth + step_m:
            selected.append(int(index))
            last_depth = current
    return selected


def _guard_arrays(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "seismic_patch": np.stack([sample["seismic_patch"] for sample in samples]),
        "well_log_seq": np.stack([sample["well_log_seq"] for sample in samples]),
        "label": np.stack([sample["label"] for sample in samples]),
        "position_json": np.asarray([json.dumps(sample["position"]) for sample in samples]),
        "meta_json": np.asarray([json.dumps(sample["meta"]) for sample in samples]),
    }


def save_guard(samples: list[dict[str, Any]], path: Path = GUARD_PATH) -> Path:
    if not samples:
        raise RuntimeError("guard为空，拒绝生成假验证集")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_guard_arrays(samples))
    return path


def load_guard(path: Path = GUARD_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"guard数据不存在: {path}")
    with np.load(path, allow_pickle=False) as data:
        return [
            {
                "seismic_patch": data["seismic_patch"][i],
                "well_log_seq": data["well_log_seq"][i],
                "label": data["label"][i],
                "position": json.loads(str(data["position_json"][i])),
                "meta": json.loads(str(data["meta_json"][i])),
            }
            for i in range(len(data["label"]))
        ]


def project_relative_path(path: Path) -> str:
    """Return a portable project-relative path or fail if it escapes the repo."""
    resolved_root = PROJECT_ROOT.resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        project_root = PROJECT_ROOT.parents[2].resolve()
        try:
            return resolved_path.relative_to(project_root).as_posix()
        except ValueError as nested_error:
            raise RuntimeError(f"报告路径逃出项目根: {resolved_path}") from nested_error


def build_real_dataset(depth_step_m: float = 2.0, sequence_step_m: float = 0.5) -> dict[str, Any]:
    """Build train/guard/test samples from the real Volve archives and SEG-Y."""
    for required in (WELL_LOG_ZIP, PICKS_PATH, SEGY_PATH, SEISMIC_INDEX_PATH):
        if not required.exists():
            raise FileNotFoundError(required)
    top_path, base_path = _horizon_paths()
    surfaces = HuginSurfaces(top_path, base_path)
    picks_by_well = parse_well_picks()
    seismic_index = _load_seismic_index()
    sequence_offsets = np.arange(-2.0, 2.0 + sequence_step_m / 2.0, sequence_step_m)

    with zipfile.ZipFile(WELL_LOG_ZIP) as zf:
        sources = discover_well_sources(zf)
        family_partitions = deterministic_family_split(source.family_id for source in sources)
        # The split manifest exists before any LAS interpolation/windowing.
        split_manifest = {
            "algorithm": "sha256(parent_family), first train then guard then test",
            "created_before_interpolation": True,
            "family_partition": family_partitions,
            "wells": [
                {
                    "well_id": source.well_id,
                    "family_id": source.family_id,
                    "partition": family_partitions[source.family_id],
                    "label_member": source.label_member,
                    "input_member": source.input_member,
                }
                for source in sources
            ],
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SPLIT_MANIFEST_PATH.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False))

        samples: dict[str, list[dict[str, Any]]] = {"train": [], "guard": [], "test": []}
        coverage: list[dict[str, Any]] = []
        with SeismicPatchReader(seismic_index) as seismic:
            for source in sources:
                partition = family_partitions[source.family_id]
                labels = _read_las(zf, source.label_member)
                inputs = _read_las(zf, source.input_member)
                missing_targets = [name for name in ("PHIF", "KLOGH", "SW") if name not in labels.curves]
                if missing_targets:
                    raise RuntimeError(f"{source.well_id}缺目标曲线: {missing_targets}")
                if source.well_id not in picks_by_well:
                    raise RuntimeError(f"官方井拾取缺少{source.well_id}")

                tie = weak_tie(labels.depth_m, picks_by_well[source.well_id])
                bounds = surfaces.bounds_at(tie["easting"], tie["northing"])
                top = np.minimum(bounds["top_twt_ms"], bounds["base_twt_ms"])
                base = np.maximum(bounds["top_twt_ms"], bounds["base_twt_ms"])
                phif = labels.curves["PHIF"]
                klogh = labels.curves["KLOGH"]
                sw = labels.curves["SW"]
                finite_targets = np.isfinite(phif) & np.isfinite(klogh) & np.isfinite(sw) & (klogh >= 0)
                in_hugin = (tie["twt_ms"] >= top) & (tie["twt_ms"] <= base)
                eligible = finite_targets & in_hugin
                selected = _subsample_depth_indices(labels.depth_m, eligible, depth_step_m)
                no_input = 0
                kept = 0
                mask_values: list[float] = []
                for row_index in selected:
                    depth = float(labels.depth_m[row_index])
                    query_depths = depth + sequence_offsets
                    log_values, log_mask = interpolate_log_sequence(
                        inputs, source.input_aliases, query_depths
                    )
                    if not log_mask.any():
                        no_input += 1
                        continue
                    mask_values.append(float(log_mask.mean()))
                    well_log_seq = np.concatenate([log_values, log_mask], axis=1).astype(np.float32)
                    il, xl = _utm_to_il_xl(
                        float(tie["easting"][row_index]),
                        float(tie["northing"][row_index]),
                        seismic_index,
                    )
                    patch, time_idx = seismic.patch(il, xl, float(tie["twt_ms"][row_index]))
                    label = transform_targets(
                        np.asarray([phif[row_index]]),
                        np.asarray([klogh[row_index]]),
                        np.asarray([sw[row_index]]),
                    )[0].astype(np.float32)
                    sample = {
                        "seismic_patch": patch,
                        "well_log_seq": well_log_seq,
                        "label": label,
                        "position": {
                            "inline": il,
                            "crossline": xl,
                            "time_ms": float(tie["twt_ms"][row_index]),
                            "well_name": source.well_id,
                        },
                        "meta": {
                            "source": "Volve real LAS + ST0202 real SEG-Y",
                            "well_id": source.well_id,
                            "family_id": source.family_id,
                            "partition": partition,
                            "depth_m": depth,
                            "time_idx": time_idx,
                            "target_names": list(TARGET_NAMES),
                            "input_channels": list(INPUT_CHANNELS),
                            "input_layout": "first 4 values, last 4 observed masks",
                            "target_transform": "[PHIF, log1p(KLOGH), SW]",
                            "hugin_top_twt_ms": float(bounds["top_twt_ms"][row_index]),
                            "hugin_base_twt_ms": float(bounds["base_twt_ms"][row_index]),
                        },
                    }
                    samples[partition].append(sample)
                    kept += 1

                coverage.append(
                    {
                        "well_id": source.well_id,
                        "family_id": source.family_id,
                        "partition": partition,
                        "label_rows": int(len(labels.depth_m)),
                        "finite_three_targets": int(finite_targets.sum()),
                        "hugin_and_finite": int(eligible.sum()),
                        "selected_after_depth_stride": int(len(selected)),
                        "kept_samples": kept,
                        "discard_invalid_target": int((~finite_targets).sum()),
                        "discard_outside_hugin_after_valid": int((finite_targets & ~in_hugin).sum()),
                        "discard_no_input": no_input,
                        "mean_input_observed_fraction": float(np.mean(mask_values)) if mask_values else 0.0,
                        "max_horizon_nearest_distance_m": float(max(
                            np.max(bounds["top_distance_m"]), np.max(bounds["base_distance_m"])
                        )),
                        "n_usable_official_picks": len(_deduplicated_tie(picks_by_well[source.well_id])["md"]),
                    }
                )

    partition_families = {
        name: sorted({sample["meta"]["family_id"] for sample in part_samples})
        for name, part_samples in samples.items()
    }
    train_families = set(partition_families["train"])
    guard_families = set(partition_families["guard"])
    test_families = set(partition_families["test"])
    if train_families & guard_families or train_families & test_families or guard_families & test_families:
        raise RuntimeError("train/guard/test母井家族发生交集")
    if len(train_families) < 2 or not guard_families or not test_families:
        raise RuntimeError(f"井族不足: {partition_families}")
    if any(not samples[name] for name in samples):
        raise RuntimeError({name: len(value) for name, value in samples.items()})

    train_path = save_split("reservoir", "train", samples["train"])
    test_path = save_split("reservoir", "test", samples["test"])
    guard_path = save_guard(samples["guard"])
    report = {
        "real_data": True,
        "target_names": list(TARGET_NAMES),
        "excluded_target_aliases": ["KLOGH_NEW", "LFP_PHIE"],
        "input_channels": list(INPUT_CHANNELS),
        "forbidden_input_curves": sorted(FORBIDDEN_INPUT_CURVES),
        "hugin_top_surface": project_relative_path(top_path),
        "hugin_base_surface": project_relative_path(base_path),
        "segy": project_relative_path(SEGY_PATH),
        "depth_step_m": depth_step_m,
        "sequence_offsets_m": sequence_offsets.tolist(),
        "sample_counts": {name: len(value) for name, value in samples.items()},
        "partition_families": partition_families,
        "family_zero_overlap": True,
        "guard_used_only_for_validation": True,
        "test_excluded_from_training_and_statistics": True,
        "coverage": coverage,
        "paths": {
            "train": project_relative_path(train_path),
            "guard": project_relative_path(guard_path),
            "test": project_relative_path(test_path),
        },
    }
    BUILD_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
