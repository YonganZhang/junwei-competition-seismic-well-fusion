"""Development-only data reconstruction for sweetspot P5 Stage-2.

P4 split manifests authorize groups before raw values are read.  Petrophysical
ZIP members outside the development groups are never opened.  For the single
production workbook, only the group-key cell is inspected before a row is
authorized; numerical cells from non-development rows are never accessed.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .sweetspot_p5_stage2_labels import LabelMappingAudit, PROJECT_ROOT, canonical_sha256


TRAIN_SAMPLE_LIMIT = 1024
VALIDATION_SAMPLE_LIMIT = 512
PETRO_SEQUENCE_LENGTH = 32
PETRO_ARCHIVE = Path("_sandbox/volve_data/Volve_Well_logs.zip")
PRODUCTION_ARCHIVE = Path("_sandbox/volve_data/Volve_Production_data.zip")
PRODUCTION_MEMBER = "Production_data/Volve production data.xlsx"
PETRO_PREFIX = "Well_logs/05.PETROPHYSICAL INTERPRETATION/"
RAW_LOG_FEATURES = (
    "GR", "RHOB", "NPHI", "RT", "DT", "DTS", "PEF", "DRHO", "CALI", "BS",
    "RD", "RM", "RACEHM", "RACELM", "RPCEHM", "RPCELM",
)
LABEL_ONLY_FIELDS = (
    "PHIF", "PHIE", "KLOGH", "KLOGV", "SW", "VSH", "SAND_FLAG", "PERF_FLAG",
)
PRODUCTION_COLUMNS = (
    "DATEPRD", "WELL_BORE_CODE", "ON_STREAM_HRS", "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P", "AVG_WHP_P", "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
)
PRODUCTION_SEQUENCE_COLUMNS = (
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE", "AVG_CHOKE_SIZE_P", "AVG_WHP_P",
)


class DevelopmentDataUnavailable(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class LasTable:
    member: str
    curves: Mapping[str, np.ndarray]
    null_value: float


def _resolve_source_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).resolve())
    git_marker = PROJECT_ROOT / ".git"
    if git_marker.is_file():
        line = git_marker.read_text(encoding="utf-8").strip()
        if line.startswith("gitdir:"):
            git_dir = Path(line.split(":", 1)[1].strip()).resolve()
            common_file = git_dir / "commondir"
            common_dir = (
                (git_dir / common_file.read_text(encoding="utf-8").strip()).resolve()
                if common_file.exists() else git_dir
            )
            candidates.append(common_dir.parent)
    candidates.append(PROJECT_ROOT.resolve())
    for candidate in candidates:
        if (candidate / "_sandbox/volve_data").is_dir():
            return candidate
    raise FileNotFoundError("shared Volve source root is unavailable")


def _curve_names_and_null(text: str) -> tuple[list[str], float]:
    curves: list[str] = []
    in_curve = False
    null_value = -999.25
    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("NULL."):
            match = re.search(r"NULL\.\S*\s+([-+0-9.Ee]+)", line, flags=re.IGNORECASE)
            if match:
                null_value = float(match.group(1))
        if upper.startswith("~CURVE"):
            in_curve = True
            continue
        if in_curve and line.startswith("~"):
            break
        if not in_curve or not line or line.startswith("#"):
            continue
        mnemonic = line.split(".", 1)[0].strip().split()[0]
        if mnemonic:
            curves.append(mnemonic.upper())
    return curves, null_value


def parse_las_bytes(payload: bytes, member: str) -> LasTable:
    text = payload.decode("latin-1", errors="replace")
    names, null_value = _curve_names_and_null(text)
    if not names:
        raise ValueError(f"{member}: no LAS curves found")
    data_lines: list[str] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.upper().startswith("~A"):
            in_data = True
            continue
        if in_data and line and not line.startswith("#"):
            data_lines.append(line)
    rows = np.fromstring(" ".join(data_lines), sep=" ", dtype=np.float64)
    if not data_lines or rows.size % len(names):
        raise ValueError(f"{member}: invalid LAS data section")
    matrix = rows.reshape(-1, len(names))
    matrix[np.isclose(matrix, null_value)] = np.nan
    return LasTable(member, {name: matrix[:, index] for index, name in enumerate(names)}, null_value)


def well_family(well: str) -> str:
    normalized = well.upper().replace("_", "/").replace(" ", "")
    if "15/9-19" in normalized:
        return "15/9-19"
    match = re.search(r"15/9-F-(\d+)", normalized)
    if match:
        return f"15/9-F-{int(match.group(1))}"
    raise ValueError(f"cannot derive mother-well family from {well!r}")


def _rqi(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    phif = np.asarray(labels["PHIF"], dtype=float)
    permeability = np.asarray(labels["KLOGH"], dtype=float)
    valid = np.isfinite(phif) & np.isfinite(permeability) & (phif > 0.0) & (permeability >= 0.0)
    result = np.full(phif.shape, np.nan, dtype=float)
    result[valid] = 0.0314 * np.sqrt(permeability[valid] / phif[valid])
    return result


def _sand_flag(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    raw = np.asarray(labels.get("SAND_FLAG", np.array([], dtype=float)), dtype=float)
    if raw.size == 0:
        return raw
    rounded = np.rint(raw)
    valid = np.isfinite(raw) & (np.abs(raw - rounded) <= 0.01) & np.isin(rounded, [0.0, 1.0])
    result = np.full(raw.shape, np.nan, dtype=float)
    result[valid] = rounded[valid]
    return result


def _history_features(history: pd.DataFrame, *, history_days: int) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for column in PRODUCTION_SEQUENCE_COLUMNS:
        series = history[column].astype(float)
        for suffix, value in (
            ("mean", series.mean()), ("std", series.std(ddof=0)), ("last", series.iloc[-1]),
        ):
            names.append(f"history_{history_days}d_{column.lower()}_{suffix}")
            values.append(float(value) if np.isfinite(value) else np.nan)
    return names, values


def _first_sustained_positive(values: np.ndarray, days: int = 7) -> int | None:
    positive = np.isfinite(values) & (values > 0.0)
    if len(positive) < days:
        return None
    run = np.convolve(positive.astype(int), np.ones(days, dtype=int), mode="valid")
    positions = np.flatnonzero(run == days)
    return int(positions[0]) if positions.size else None


@dataclass(frozen=True)
class DevelopmentPilotData:
    target_id: str
    task_type: str
    target_name: str
    feature_names: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_tabular: np.ndarray
    validation_tabular: np.ndarray
    train_sequence: np.ndarray | None
    validation_sequence: np.ndarray | None
    train_target: np.ndarray
    validation_target: np.ndarray
    split_sha256: str
    input_budget_sha256: str
    provenance: Mapping[str, Any]

    @property
    def budget(self) -> dict[str, Any]:
        return {
            "train_sample_limit": TRAIN_SAMPLE_LIMIT,
            "validation_sample_limit": VALIDATION_SAMPLE_LIMIT,
            "train_samples": len(self.train_sample_ids),
            "validation_samples": len(self.validation_sample_ids),
            "feature_count": int(self.train_tabular.shape[1]),
            "sequence_shape": None if self.train_sequence is None else list(self.train_sequence.shape[1:]),
            "input_budget_sha256": self.input_budget_sha256,
        }


def forbidden_test_source(path: Path) -> bool:
    """Reject materialized test artifacts; raw multi-group archives are gated by member/row."""
    for part in Path(path).parts:
        lowered = part.lower()
        if lowered in {"frozen_test", "test.h5", "test.hdf5", "test.npz"}:
            return True
        if lowered.startswith("frozen_test"):
            return True
    return False


def _folder(member: str) -> str:
    return member[len(PETRO_PREFIX):].split("/", 1)[0]


def petrophysical_member_authorized(member: str, development_groups: set[str]) -> bool:
    if not member.startswith(PETRO_PREFIX) or not member.lower().endswith(".las"):
        return False
    try:
        return well_family(_folder(member)) in development_groups
    except ValueError:
        return False


def _member_fingerprint(archive: ZipFile, members: Sequence[str]) -> str:
    records = []
    for member in sorted(set(members)):
        info = archive.getinfo(member)
        records.append({"member": member, "crc32": f"{info.CRC:08x}", "size_bytes": info.file_size})
    return canonical_sha256(records)


def _load_development_petrophysical_tables(
    source_root: Path,
    development_groups: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_path = Path(source_root) / PETRO_ARCHIVE
    if forbidden_test_source(archive_path):
        raise PermissionError("petrophysical source resolves through a forbidden test path")
    records: list[dict[str, Any]] = []
    opened_members: list[str] = []
    with ZipFile(archive_path) as archive:
        authorized_names = [
            name for name in archive.namelist()
            if petrophysical_member_authorized(name, development_groups)
        ]
        by_folder: dict[str, list[str]] = {}
        for name in authorized_names:
            by_folder.setdefault(_folder(name), []).append(name)
        outputs = [
            name for name in authorized_names
            if "COMPUTED_OUTPUT" in name.upper() or "/CPI/" in name.upper()
        ]
        for output_member in sorted(outputs):
            family = well_family(_folder(output_member))
            if family not in development_groups:
                raise PermissionError("petrophysical member authorization changed after selection")
            output = parse_las_bytes(archive.read(output_member), output_member)
            opened_members.append(output_member)
            if not ({"PHIF", "KLOGH"} <= set(output.curves)):
                continue
            folder = _folder(output_member)
            input_candidates = [
                name for name in by_folder.get(folder, ()) if "COMPUTED_INPUT" in name.upper()
            ]
            if not input_candidates:
                continue
            input_member = sorted(input_candidates)[0]
            if not petrophysical_member_authorized(input_member, development_groups):
                raise PermissionError("unauthorized petrophysical input member")
            inputs = parse_las_bytes(archive.read(input_member), input_member)
            opened_members.append(input_member)
            if "DEPTH" not in output.curves or "DEPTH" not in inputs.curves:
                raise ValueError(f"{folder}: DEPTH is required for the P4 join")
            input_index = {
                round(float(depth), 4): index
                for index, depth in enumerate(inputs.curves["DEPTH"])
                if np.isfinite(depth)
            }
            output_indices: list[int] = []
            input_indices: list[int] = []
            for output_index, depth in enumerate(output.curves["DEPTH"]):
                if not np.isfinite(depth):
                    continue
                input_row = input_index.get(round(float(depth), 4))
                if input_row is not None:
                    output_indices.append(output_index)
                    input_indices.append(input_row)
            if not output_indices:
                raise ValueError(f"{folder}: raw and interpreted LAS have no common depths")
            features = {
                field: (
                    np.asarray(inputs.curves[field], dtype=float)[input_indices]
                    if field in inputs.curves else np.full(len(input_indices), np.nan)
                )
                for field in RAW_LOG_FEATURES
            }
            labels = {
                field: np.asarray(output.curves[field], dtype=float)[output_indices]
                for field in LABEL_ONLY_FIELDS if field in output.curves
            }
            if "SAND_FLAG" not in labels and "SAND_FLAG" in inputs.curves:
                labels["SAND_FLAG"] = np.asarray(inputs.curves["SAND_FLAG"], dtype=float)[input_indices]
            records.append({
                "wellbore": folder.replace("_", "/"),
                "well_family": family,
                "depth_m": np.asarray(output.curves["DEPTH"], dtype=float)[output_indices],
                "features": features,
                "labels": labels,
            })
        if any(not petrophysical_member_authorized(member, development_groups) for member in opened_members):
            raise PermissionError("a non-development LAS member was opened")
        fingerprint = _member_fingerprint(archive, opened_members)
    return records, {
        "source_kind": "Volve_Well_logs.zip development members only",
        "archive": PETRO_ARCHIVE.as_posix(),
        "opened_member_count": len(set(opened_members)),
        "opened_member_crc_manifest_sha256": fingerprint,
        "authorized_development_groups": sorted(development_groups),
        "unauthorized_member_reads": 0,
        "test_accessed": False,
    }


def _sequence_window(matrix: np.ndarray, index: int, width: int) -> np.ndarray:
    half = width // 2
    positions = np.arange(index - half, index - half + width)
    positions = np.clip(positions, 0, len(matrix) - 1)
    return matrix[positions].T


def _flatten_petrophysical(
    tables: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
) -> dict[str, Any]:
    target_name = "rqi" if target_id == "T1" else "sand-flag"
    target_fn = _rqi if target_id == "T1" else _sand_flag
    sample_ids: list[str] = []
    groups: list[str] = []
    features: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for table in tables:
        if target_id == "T2" and "SAND_FLAG" not in table["labels"]:
            continue
        label = np.asarray(target_fn(table["labels"]), dtype=float)
        depth = np.asarray(table["depth_m"], dtype=float)
        matrix = np.column_stack([
            np.asarray(table["features"][field], dtype=float) for field in RAW_LOG_FEATURES
        ])
        valid = np.isfinite(label) & np.isfinite(depth) & np.isfinite(matrix).any(axis=1)
        well_token = re.sub(r"[^A-Za-z0-9]+", "-", str(table["wellbore"])).strip("-")
        for index in np.flatnonzero(valid):
            sample_ids.append(f"{target_name}:{well_token}:{depth[index]:.4f}")
            groups.append(str(table["well_family"]))
            features.append(matrix[index])
            sequences.append(_sequence_window(matrix, int(index), PETRO_SEQUENCE_LENGTH))
            targets.append(float(label[index]))
    if not sample_ids:
        raise DevelopmentDataUnavailable("development_rebuild_empty", f"{target_id}: no development samples")
    return {
        "sample_ids": sample_ids,
        "groups": groups,
        "features": np.asarray(features, dtype=float),
        "sequences": np.asarray(sequences, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "feature_names": tuple(RAW_LOG_FEATURES),
    }


def production_row_values(
    row: Sequence[Any],
    header_index: Mapping[str, int],
    development_groups: set[str],
) -> dict[str, Any] | None:
    """Authorize on the group cell before accessing any numerical cell."""
    group = row[header_index["WELL_BORE_CODE"]].value
    if group not in development_groups:
        return None
    return {column: row[header_index[column]].value for column in PRODUCTION_COLUMNS}


_XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _cell_column(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_shared_strings(workbook: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(f"{{{_XLSX_MAIN_NS}}}si"):
        strings.append("".join(node.text or "" for node in item.iter(f"{{{_XLSX_MAIN_NS}}}t")))
    return strings


def _xlsx_sheet_path(workbook: ZipFile, sheet_name: str) -> str:
    root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    relation_id = None
    for sheet in root.iter(f"{{{_XLSX_MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib.get(f"{{{_XLSX_REL_NS}}}id")
            break
    if relation_id is None:
        raise KeyError(f"XLSX sheet not found: {sheet_name}")
    relations = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relation in relations.iter(f"{{{_XLSX_PACKAGE_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib.get("Target")
            break
    if not target:
        raise ValueError(f"XLSX sheet relation is missing for {sheet_name}")
    return posixpath.normpath(posixpath.join("xl", target))


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: Sequence[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{_XLSX_MAIN_NS}}}t"))
    value_node = cell.find(f"{{{_XLSX_MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def _xlsx_authorized_records(payload: bytes, development_groups: set[str]) -> tuple[list[dict[str, Any]], int]:
    """Two-pass reader: decode group keys first, then values of authorized rows only."""
    with ZipFile(BytesIO(payload)) as workbook:
        shared_strings = _xlsx_shared_strings(workbook)
        sheet_path = _xlsx_sheet_path(workbook, "Daily Production Data")
        sheet_payload = workbook.read(sheet_path)

    header: dict[int, str] = {}
    authorized_rows: set[int] = set()
    rejected_rows = 0
    row_tag = f"{{{_XLSX_MAIN_NS}}}row"
    cell_tag = f"{{{_XLSX_MAIN_NS}}}c"
    for _, row in ElementTree.iterparse(BytesIO(sheet_payload), events=("end",)):
        if row.tag != row_tag:
            continue
        row_number = int(row.attrib.get("r", "0"))
        cells = list(row.iter(cell_tag))
        if row_number == 1:
            for cell in cells:
                header[_cell_column(cell.attrib["r"])] = str(_xlsx_cell_value(cell, shared_strings))
            row.clear()
            continue
        group_columns = [column for column, name in header.items() if name == "WELL_BORE_CODE"]
        if len(group_columns) != 1:
            raise ValueError("production XLSX must contain one WELL_BORE_CODE column")
        group_column = group_columns[0]
        group = None
        for cell in cells:
            if _cell_column(cell.attrib["r"]) == group_column:
                group = _xlsx_cell_value(cell, shared_strings)
                break
        if group in development_groups:
            authorized_rows.add(row_number)
        else:
            rejected_rows += 1
        row.clear()

    wanted_columns = {column: name for column, name in header.items() if name in PRODUCTION_COLUMNS}
    if set(wanted_columns.values()) != set(PRODUCTION_COLUMNS):
        missing = sorted(set(PRODUCTION_COLUMNS) - set(wanted_columns.values()))
        raise ValueError(f"production workbook missing columns: {missing}")
    records: list[dict[str, Any]] = []
    for _, row in ElementTree.iterparse(BytesIO(sheet_payload), events=("end",)):
        if row.tag != row_tag:
            continue
        row_number = int(row.attrib.get("r", "0"))
        if row_number not in authorized_rows:
            row.clear()
            continue
        record = {name: None for name in PRODUCTION_COLUMNS}
        for cell in row.iter(cell_tag):
            column = _cell_column(cell.attrib["r"])
            if column in wanted_columns:
                record[wanted_columns[column]] = _xlsx_cell_value(cell, shared_strings)
        records.append(record)
        row.clear()
    return records, rejected_rows


def _load_development_production(
    source_root: Path,
    development_groups: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive_path = Path(source_root) / PRODUCTION_ARCHIVE
    if forbidden_test_source(archive_path):
        raise PermissionError("production source resolves through a forbidden test path")
    with ZipFile(archive_path) as archive:
        info = archive.getinfo(PRODUCTION_MEMBER)
        payload = archive.read(PRODUCTION_MEMBER)
    records, rejected = _xlsx_authorized_records(payload, development_groups)
    frame = pd.DataFrame.from_records(records, columns=PRODUCTION_COLUMNS)
    frame["DATEPRD"] = frame["DATEPRD"].map(
        lambda value: (
            pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
            if isinstance(value, (int, float, np.integer, np.floating))
            else pd.to_datetime(value, errors="coerce")
        )
    )
    for column in PRODUCTION_COLUMNS:
        if column not in {"DATEPRD", "WELL_BORE_CODE"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values(["WELL_BORE_CODE", "DATEPRD"], inplace=True)
    member_fingerprint = canonical_sha256({
        "member": PRODUCTION_MEMBER, "crc32": f"{info.CRC:08x}", "size_bytes": info.file_size,
    })
    return frame, {
        "source_kind": "Volve production workbook with row-level development authorization",
        "archive": PRODUCTION_ARCHIVE.as_posix(),
        "member_crc_manifest_sha256": member_fingerprint,
        "authorized_development_groups": sorted(development_groups),
        "authorized_rows": len(frame),
        "rejected_rows_values_accessed": 0,
        "rejected_row_count": rejected,
        "test_accessed": False,
    }


def _production_sequence(history: pd.DataFrame) -> np.ndarray:
    return np.stack([
        history[column].to_numpy(dtype=float) for column in PRODUCTION_SEQUENCE_COLUMNS
    ])


def _build_productivity(frame: pd.DataFrame) -> dict[str, Any]:
    sample_ids: list[str] = []
    groups: list[str] = []
    features: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    feature_names: list[str] | None = None
    for group, well in frame.groupby("WELL_BORE_CODE", sort=True):
        well = well.sort_values("DATEPRD").reset_index(drop=True)
        for index in range(30, len(well) - 29, 7):
            history = well.iloc[index - 30:index]
            future = well.iloc[index:index + 30]
            if (history["DATEPRD"].iloc[-1] - history["DATEPRD"].iloc[0]).days > 45:
                continue
            if (future["DATEPRD"].iloc[-1] - future["DATEPRD"].iloc[0]).days > 45:
                continue
            target = future["BORE_OIL_VOL"].mean()
            if not np.isfinite(target) or future["BORE_OIL_VOL"].notna().sum() < 21:
                continue
            names, row = _history_features(history, history_days=30)
            if not np.isfinite(row).any():
                continue
            cutoff = well.loc[index, "DATEPRD"]
            sample_ids.append(f"productivity:{group}:{cutoff.date().isoformat()}")
            groups.append(str(group))
            features.append(np.asarray(row, dtype=float))
            sequences.append(_production_sequence(history))
            targets.append(float(target))
            feature_names = names
    return {
        "sample_ids": sample_ids, "groups": groups,
        "features": np.asarray(features, dtype=float),
        "sequences": np.asarray(sequences, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "feature_names": tuple(feature_names or ()),
    }


def _build_water_breakthrough(frame: pd.DataFrame) -> dict[str, Any]:
    sample_ids: list[str] = []
    groups: list[str] = []
    features: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    feature_names: list[str] | None = None
    for group, well in frame.groupby("WELL_BORE_CODE", sort=True):
        well = well.sort_values("DATEPRD").reset_index(drop=True)
        event_index = _first_sustained_positive(well["BORE_WAT_VOL"].to_numpy(float), days=7)
        history_days = 7
        if event_index is not None and event_index < history_days:
            continue
        last_index = len(well) if event_index is None else event_index
        for index in range(history_days, last_index, 7):
            history = well.iloc[index - history_days:index]
            if (history["DATEPRD"].iloc[-1] - history["DATEPRD"].iloc[0]).days > 45:
                continue
            names, row = _history_features(history, history_days=history_days)
            if not np.isfinite(row).any():
                continue
            label = int(event_index is not None and 1 <= event_index - index <= 30)
            cutoff = well.loc[index, "DATEPRD"]
            sample_ids.append(f"water-risk:{group}:{cutoff.date().isoformat()}")
            groups.append(str(group))
            features.append(np.asarray(row, dtype=float))
            sequences.append(_production_sequence(history))
            targets.append(float(label))
            feature_names = names
    return {
        "sample_ids": sample_ids, "groups": groups,
        "features": np.asarray(features, dtype=float),
        "sequences": np.asarray(sequences, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "feature_names": tuple(feature_names or ()),
    }


def _seeded_subset(sample_ids: Sequence[str], limit: int, *, target_id: str, lane: str) -> tuple[str, ...]:
    ordered = sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"2693|{target_id}|{lane}|{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ordered[:limit])


def _take(dataset: Mapping[str, Any], sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    by_id = {sample_id: index for index, sample_id in enumerate(dataset["sample_ids"])}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise DevelopmentDataUnavailable(
            "development_sample_rebuild_mismatch",
            f"rebuild is missing {len(missing)} authorized IDs; first={missing[0]}",
        )
    indices = np.asarray([by_id[sample_id] for sample_id in sample_ids], dtype=int)
    groups = tuple(str(dataset["groups"][index]) for index in indices)
    return (
        np.asarray(dataset["features"])[indices],
        np.asarray(dataset["sequences"])[indices],
        np.asarray(dataset["targets"])[indices],
        groups,
    )


def load_development_pilot_data(
    audit: LabelMappingAudit,
    target_id: str,
    *,
    source_root: Path | None = None,
    fold_id: int = 0,
) -> DevelopmentPilotData:
    target = audit.target(target_id)
    if target.get("status") != "approved_for_development_pilot":
        raise DevelopmentDataUnavailable("label_not_approved", f"{target_id} is not pilot-approved")
    if target_id in {"T6", "T7"}:
        raise DevelopmentDataUnavailable(
            "development_feature_source_unavailable",
            f"{target_id}: P4 labels/split are auditable, but no development-only feature source is present; test.h5 fallback is forbidden",
        )
    split_path = PROJECT_ROOT / target["split_manifest"]["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    matching_folds = [item for item in split["folds"] if int(item["fold_id"]) == int(fold_id)]
    if len(matching_folds) != 1:
        raise DevelopmentDataUnavailable(
            "p4_fold_missing",
            f"{target_id}: P4 manifest does not contain exactly one fold_id={fold_id}",
        )
    fold = matching_folds[0]
    development_groups = set(split["development_groups"])
    resolved_root = _resolve_source_root(source_root)
    if target_id in {"T1", "T2"}:
        tables, provenance = _load_development_petrophysical_tables(resolved_root, development_groups)
        dataset = _flatten_petrophysical(tables, target_id=target_id)
    elif target_id in {"T3", "T4"}:
        frame, provenance = _load_development_production(resolved_root, development_groups)
        dataset = _build_productivity(frame) if target_id == "T3" else _build_water_breakthrough(frame)
    else:
        raise KeyError(target_id)

    rebuilt_ids = set(dataset["sample_ids"])
    expected_ids = set(split["development_sample_ids"])
    if rebuilt_ids != expected_ids:
        missing = sorted(expected_ids - rebuilt_ids)
        unexpected = sorted(rebuilt_ids - expected_ids)
        raise DevelopmentDataUnavailable(
            "development_sample_rebuild_mismatch",
            f"{target_id}: rebuilt/manifest IDs differ; missing={len(missing)}, unexpected={len(unexpected)}",
        )
    train_ids = _seeded_subset(
        fold["train_sample_ids"], TRAIN_SAMPLE_LIMIT, target_id=target_id,
        lane="train" if int(fold_id) == 0 else f"fold-{fold_id}-train",
    )
    validation_ids = _seeded_subset(
        fold["validation_sample_ids"], VALIDATION_SAMPLE_LIMIT, target_id=target_id,
        lane="validation" if int(fold_id) == 0 else f"fold-{fold_id}-validation",
    )
    train_x, train_sequence, train_y, train_groups = _take(dataset, train_ids)
    validation_x, validation_sequence, validation_y, validation_groups = _take(dataset, validation_ids)
    if set(train_groups) & set(validation_groups):
        raise DevelopmentDataUnavailable(
            "split_group_overlap", f"{target_id}: fold_{fold_id} group overlap",
        )
    if target["task_type"] == "binary":
        if set(np.unique(train_y.astype(int))) != {0, 1}:
            raise DevelopmentDataUnavailable(
                "fold_train_class_missing", f"{target_id}: fold_{fold_id} train lacks a class",
            )
    budget_hash = canonical_sha256({
        "target_id": target_id,
        "train_sample_ids": list(train_ids),
        "validation_sample_ids": list(validation_ids),
        "feature_names": list(dataset["feature_names"]),
        "sequence_shape": list(train_sequence.shape[1:]),
    })
    provenance = dict(provenance)
    provenance.update({
        "p4_split_manifest_path": target["split_manifest"]["path"],
        "p4_split_manifest_sha256": target["split_manifest"]["sha256"],
        "p4_fold_id": int(fold["fold_id"]),
        "rebuilt_development_sample_count": len(rebuilt_ids),
        "rebuild_matches_manifest": True,
        "frozen_test_files_opened": 0,
        "historical_test_metrics_read": False,
    })
    return DevelopmentPilotData(
        target_id=target_id,
        task_type=str(target["task_type"]),
        target_name=str(target["target_name"]),
        feature_names=tuple(dataset["feature_names"]),
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        train_groups=train_groups,
        validation_groups=validation_groups,
        train_tabular=train_x,
        validation_tabular=validation_x,
        train_sequence=train_sequence,
        validation_sequence=validation_sequence,
        train_target=train_y,
        validation_target=validation_y,
        split_sha256=str(target["split_manifest"]["sha256"]),
        input_budget_sha256=budget_hash,
        provenance=provenance,
    )
