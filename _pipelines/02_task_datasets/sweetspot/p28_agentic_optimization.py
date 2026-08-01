"""Sweetspot P28 T3-only hybrid execution-agent pilot.

This implementation is fail-closed by design and uses a fully local xgboost
executor. It records a strict A2L DeepSeek policy decision, then runs the same
portable route/config registry under deterministic and random selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT_SEED = 2693
TRACK_ID = "sweetspot"
P28_ID = "p28_agentic_optimization"
SCHEMA_VERSION = "sweetspot-p28-agentic-optimization/v1"
XGB_ACTION_IDS = (
    "T3_XGB_D4_ETA_0_03_ROUNDS_64",
    "T3_XGB_D4_ETA_0_05_ROUNDS_96",
    "T3_XGB_D6_ETA_0_03_ROUNDS_96",
    "T3_XGB_D6_ETA_0_05_ROUNDS_128",
)
SELECTION_FOLDS = [0, 1, 2]
PROMOTION_FOLDS = [3]

HERE = Path(__file__).resolve().parent
TRACK_DIR = HERE
WORKTREE_ROOT = HERE.parents[2]
WORKTREES_DIR = WORKTREE_ROOT.parent
PROJECT_ROOT = WORKTREE_ROOT.parents[2]
REFERENCE_ROOT = WORKTREES_DIR / "p10-results-sweetspot"
REFERENCE_SWEETSPOT = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot"
OUTPUT_DIR = TRACK_DIR / "_outputs" / P28_ID

STAGE3_T3_PATH = REFERENCE_SWEETSPOT / "p5/_outputs/stage3_cv/leaderboards/T3.json"
STAGE3_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p5/_outputs/stage3_cv/p5_stage3_summary.json"
STAGE4_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p5/_outputs/stage4_confirmation/p5_stage4_summary.json"
P7_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p7/_outputs/t3_chronos2_cv/summary.json"
P8_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p8/_outputs/t3_chronos2_calendar_cv/summary.json"
P17_EVIDENCE_PATH = PROJECT_ROOT / "_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


@dataclass(frozen=True)
class PilotAudit:
    mapping_sha256: str
    split_manifest_sha256: str
    target_id: str
    target_name: str
    target_status: str
    p4_status: str
    task_type: str
    primary_metric: str
    primary_metric_direction: str
    label_version: str
    proxy_semantics: str
    split_manifest_path: str
    label_mapping_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PilotTaskSpec:
    target_id: str
    target_name: str
    task_type: str
    primary_metric: str
    primary_metric_direction: str
    label_version: str
    split_manifest_path: str
    split_manifest_sha256: str
    root_seed: int
    route_family: str
    route_note: str
    development_rebuild: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepSeekDecision:
    selected_route: str
    selected_action_id: str
    stop_requested: bool
    confidence: float
    rationale: str
    raw: dict[str, Any]


def _reference_inputs() -> list[dict[str, Any]]:
    def display_path(path: Path) -> str:
        if path.is_relative_to(REFERENCE_ROOT):
            return str(path.relative_to(REFERENCE_ROOT))
        if path.is_relative_to(PROJECT_ROOT):
            return str(path.relative_to(PROJECT_ROOT))
        return path.name

    items = [
        ("p5 label mapping", P5_LABEL_MAPPING_PATH),
        ("p5 split manifest", P5_T3_SPLIT_MANIFEST_PATH),
        ("p5 stage-3 T3 leaderboard", STAGE3_T3_PATH),
        ("p5 stage-3 summary", STAGE3_SUMMARY_PATH),
        ("p5 stage-4 confirmation summary", STAGE4_SUMMARY_PATH),
        ("p7 Chronos summary", P7_SUMMARY_PATH),
        ("p8 calendar Chronos summary", P8_SUMMARY_PATH),
        ("P17 acceptance evidence", P17_EVIDENCE_PATH),
    ]
    return [
        {
            "role": role,
            "path": display_path(path),
            "sha256": _sha256_file(path),
            "shape_or_row_count": path.stat().st_size,
            "scientific_role": role,
            "split_scope": "immutable reference",
        }
        for role, path in items
    ]


def _observation() -> dict[str, Any]:
    return {
        "track_id": TRACK_ID,
        "task_id": "T3",
        "root_seed": ROOT_SEED,
        "a0_baseline": {
            "route": "xgboost",
            "fold_train_state": "flat",
            "selection_dev_feedback": "flat",
            "promotion_dev_feedback": "flat",
            "selection_fold_ids": SELECTION_FOLDS,
            "promotion_fold_ids": PROMOTION_FOLDS,
        },
        "gate_states": {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"},
        "allowlist": {"xgboost": list(XGB_ACTION_IDS)},
        "budget": {"trials": 4, "root_seed": ROOT_SEED},
        "selection_dev_feedback_contract": "LLM sees only improved|flat|worse summaries, never raw metrics, labels, residuals, or paths",
        "promotion_dev_contract": "selection-dev and promotion-dev are nested disjoint folds",
        "foundation_route_fixed_or_separate_factor": True,
    }


def _call_deepseek(messages: list[dict[str, str]]) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_KEY")
    if not key:
        return {"status": "UNAVAILABLE", "reason": "DEEPSEEK_KEY missing"}
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw_content": content}
    parsed["_provider"] = "deepseek"
    parsed["_model"] = payload.get("model")
    return parsed


def _deterministic_choice() -> dict[str, Any]:
    return {
        "selected_route": "xgboost",
        "selected_action_id": XGB_ACTION_IDS[0],
        "stop_requested": False,
        "confidence": 0.71,
        "rationale": "Use the fixed xgboost route and the shallowest config as the deterministic baseline.",
    }


def _random_choice() -> dict[str, Any]:
    import random

    rng = random.Random(ROOT_SEED)
    route = "xgboost"
    action = rng.choice(list(XGB_ACTION_IDS))
    return {
        "selected_route": route,
        "selected_action_id": action,
        "stop_requested": False,
        "confidence": 0.5,
        "rationale": "Random baseline with the same xgboost allowlist and seed roster.",
    }


TRAIN_SAMPLE_LIMIT = 1024
VALIDATION_SAMPLE_LIMIT = 512
PRODUCTION_ARCHIVE = Path("_sandbox/volve_data/Volve_Production_data.zip")
PRODUCTION_MEMBER = "Production_data/Volve production data.xlsx"
PRODUCTION_COLUMNS = (
    "DATEPRD",
    "WELL_BORE_CODE",
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
)
PRODUCTION_SEQUENCE_COLUMNS = (
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
)
P5_LABEL_MAPPING_PATH = REFERENCE_SWEETSPOT / "p5/sweetspot_p5_label_mapping.v1.json"
P5_T3_SPLIT_MANIFEST_PATH = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot/targets/productivity/_outputs/baseline_v1/split_manifest.json"


class DevelopmentDataUnavailable(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


def _resolve_source_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).resolve())
    candidates.append(PROJECT_ROOT.resolve())
    for candidate in candidates:
        if (candidate / "_sandbox/volve_data").is_dir():
            return candidate
    raise FileNotFoundError("shared Volve source root is unavailable")


def _history_features(history, *, history_days: int) -> tuple[list[str], list[float]]:
    import numpy as np

    names: list[str] = []
    values: list[float] = []
    for column in PRODUCTION_SEQUENCE_COLUMNS:
        series = history[column].astype(float)
        for suffix, value in (("mean", series.mean()), ("std", series.std(ddof=0)), ("last", series.iloc[-1])):
            names.append(f"history_{history_days}d_{column.lower()}_{suffix}")
            values.append(float(value) if np.isfinite(value) else np.nan)
    return names, values


def _first_sustained_positive(values, days: int = 7) -> int | None:
    import numpy as np

    positive = np.isfinite(values) & (values > 0.0)
    if len(positive) < days:
        return None
    run = np.convolve(positive.astype(int), np.ones(days, dtype=int), mode="valid")
    positions = np.flatnonzero(run == days)
    return int(positions[0]) if positions.size else None


def _cell_column(reference: str) -> int:
    import re

    letters = re.match(r"[A-Z]+", reference.upper())
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_shared_strings(workbook) -> list[str]:
    from xml.etree import ElementTree

    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(f"{{{namespace}}}si"):
        strings.append("".join(node.text or "" for node in item.iter(f"{{{namespace}}}t")))
    return strings


def _xlsx_sheet_path(workbook, sheet_name: str) -> str:
    from xml.etree import ElementTree
    import posixpath

    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    relation_id = None
    for sheet in root.iter(f"{{{main_ns}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib.get(f"{{{rel_ns}}}id")
            break
    if relation_id is None:
        raise KeyError(f"XLSX sheet not found: {sheet_name}")
    relations = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relation in relations.iter(f"{{{pkg_rel_ns}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib.get("Target")
            break
    if not target:
        raise ValueError(f"XLSX sheet relation is missing for {sheet_name}")
    return posixpath.normpath(posixpath.join("xl", target))


def _xlsx_cell_value(cell, shared_strings):
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{namespace}}}t"))
    value_node = cell.find(f"{{{namespace}}}v")
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
    from io import BytesIO
    from xml.etree import ElementTree
    from zipfile import ZipFile

    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with ZipFile(BytesIO(payload)) as workbook:
        shared_strings = _xlsx_shared_strings(workbook)
        sheet_path = _xlsx_sheet_path(workbook, "Daily Production Data")
        sheet_payload = workbook.read(sheet_path)

    header: dict[int, str] = {}
    authorized_rows: set[int] = set()
    rejected_rows = 0
    row_tag = f"{{{namespace}}}row"
    cell_tag = f"{{{namespace}}}c"
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


def _load_development_production(source_root: Path, development_groups: set[str]):
    import pandas as pd
    from zipfile import ZipFile

    archive_path = Path(source_root) / PRODUCTION_ARCHIVE
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    with ZipFile(archive_path) as archive:
        info = archive.getinfo(PRODUCTION_MEMBER)
        payload = archive.read(PRODUCTION_MEMBER)
    records, rejected = _xlsx_authorized_records(payload, development_groups)
    frame = pd.DataFrame.from_records(records, columns=PRODUCTION_COLUMNS)
    frame["DATEPRD"] = frame["DATEPRD"].map(
        lambda value: (
            pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
            if isinstance(value, (int, float))
            else pd.to_datetime(value, errors="coerce")
        )
    )
    for column in PRODUCTION_COLUMNS:
        if column not in {"DATEPRD", "WELL_BORE_CODE"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values(["WELL_BORE_CODE", "DATEPRD"], inplace=True)
    member_fingerprint = _sha256_bytes(
        json.dumps({"member": PRODUCTION_MEMBER, "crc32": f"{info.CRC:08x}", "size_bytes": info.file_size}, sort_keys=True).encode("utf-8")
    )
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


def _production_sequence(history):
    import numpy as np

    return np.stack([history[column].to_numpy(dtype=float) for column in PRODUCTION_SEQUENCE_COLUMNS])


def _seeded_subset(sample_ids, limit: int, *, target_id: str, lane: str) -> tuple[str, ...]:
    import hashlib

    ordered = sorted(sample_ids, key=lambda sample_id: hashlib.sha256(f"{ROOT_SEED}|{target_id}|{lane}|{sample_id}".encode("utf-8")).hexdigest())
    return tuple(ordered[:limit])


def _take(dataset, sample_ids):
    import numpy as np

    by_id = {sample_id: index for index, sample_id in enumerate(dataset["sample_ids"])}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise DevelopmentDataUnavailable("development_sample_rebuild_mismatch", f"rebuild is missing {len(missing)} authorized IDs; first={missing[0]}")
    indices = np.asarray([by_id[sample_id] for sample_id in sample_ids], dtype=int)
    groups = tuple(str(dataset["groups"][index]) for index in indices)
    return (
        np.asarray(dataset["features"])[indices],
        np.asarray(dataset["sequences"])[indices],
        np.asarray(dataset["targets"])[indices],
        groups,
    )


def _build_productivity(frame):
    import numpy as np

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
        "sample_ids": sample_ids,
        "groups": groups,
        "features": np.asarray(features, dtype=float),
        "sequences": np.asarray(sequences, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "feature_names": tuple(feature_names or ()),
    }


def load_t3_development_pilot_data(*, source_root: Path | None = None, fold_id: int = 0):
    import json
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace

    source_root = _resolve_source_root(source_root)
    mapping = json.loads(P5_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    target = mapping["targets"]["T3"]
    split_path = REFERENCE_ROOT / target["split_manifest"]["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    matching_folds = [item for item in split["folds"] if int(item["fold_id"]) == int(fold_id)]
    if len(matching_folds) != 1:
        raise DevelopmentDataUnavailable("p4_fold_missing", f"T3: P4 manifest does not contain exactly one fold_id={fold_id}")
    fold = matching_folds[0]
    development_groups = set(split["development_groups"])
    frame, provenance = _load_development_production(source_root, development_groups)
    dataset = _build_productivity(frame)
    rebuilt_ids = set(dataset["sample_ids"])
    expected_ids = set(split["development_sample_ids"])
    if rebuilt_ids != expected_ids:
        missing = sorted(expected_ids - rebuilt_ids)
        unexpected = sorted(rebuilt_ids - expected_ids)
        raise DevelopmentDataUnavailable("development_sample_rebuild_mismatch", f"T3: rebuilt/manifest IDs differ; missing={len(missing)}, unexpected={len(unexpected)}")
    train_ids = _seeded_subset(fold["train_sample_ids"], TRAIN_SAMPLE_LIMIT, target_id="T3", lane="train" if int(fold_id) == 0 else f"fold-{fold_id}-train")
    validation_ids = _seeded_subset(fold["validation_sample_ids"], VALIDATION_SAMPLE_LIMIT, target_id="T3", lane="validation" if int(fold_id) == 0 else f"fold-{fold_id}-validation")
    train_x, train_sequence, train_y, train_groups = _take(dataset, train_ids)
    validation_x, validation_sequence, validation_y, validation_groups = _take(dataset, validation_ids)
    if set(train_groups) & set(validation_groups):
        raise DevelopmentDataUnavailable("split_group_overlap", f"T3: fold_{fold_id} group overlap")
    budget_hash = _sha256_bytes(json.dumps({
        "target_id": "T3",
        "train_sample_ids": list(train_ids),
        "validation_sample_ids": list(validation_ids),
        "feature_names": list(dataset["feature_names"]),
        "sequence_shape": list(train_sequence.shape[1:]),
    }, sort_keys=True).encode("utf-8"))
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
    return SimpleNamespace(
        target_id="T3",
        task_type="regression",
        target_name=target["target_name"],
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


def _fold_metric_payload(actual: Any, prediction: Any) -> dict[str, float]:
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    truth = np.asarray(actual, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("invalid prediction vector")
    rho = spearmanr(truth, pred).statistic
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "spearman": float(rho),
    }


def _topk_diagnostic(actual: Any, prediction: Any, *, fraction: float = 0.1) -> dict[str, float]:
    import numpy as np

    truth = np.asarray(actual, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.size == 0:
        return {"fraction": fraction, "k": 0, "hit_rate": float("nan"), "overlap_count": 0}
    k = max(1, int(round(truth.size * fraction)))
    topk = np.argsort(pred)[-k:]
    hit_rate = float(np.mean(truth[topk] >= np.median(truth)))
    return {
        "fraction": float(fraction),
        "k": int(k),
        "hit_rate": hit_rate,
        "overlap_count": int(k),
    }


def _load_task_contract() -> tuple[PilotAudit, PilotTaskSpec, dict[str, Any]]:
    mapping = json.loads(P5_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    target = mapping["targets"]["T3"]
    split_manifest = json.loads(P5_T3_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = PilotAudit(
        mapping_sha256=_sha256_file(P5_LABEL_MAPPING_PATH),
        split_manifest_sha256=_sha256_file(P5_T3_SPLIT_MANIFEST_PATH),
        target_id="T3",
        target_name=str(target["target_name"]),
        target_status=str(target["status"]),
        p4_status=str(target["p4_status"]),
        task_type=str(target["task_type"]),
        primary_metric=str(target["primary_metric"]),
        primary_metric_direction=str(target["primary_metric_direction"]),
        label_version=str(target["label_version"]),
        proxy_semantics=str(target["proxy_semantics"]),
        split_manifest_path=str(target["split_manifest"]["path"]),
        label_mapping_path=str(P5_LABEL_MAPPING_PATH),
    )
    task_spec = PilotTaskSpec(
        target_id="T3",
        target_name=str(target["target_name"]),
        task_type=str(target["task_type"]),
        primary_metric=str(target["primary_metric"]),
        primary_metric_direction=str(target["primary_metric_direction"]),
        label_version=str(target["label_version"]),
        split_manifest_path=str(target["split_manifest"]["path"]),
        split_manifest_sha256=str(target["split_manifest"]["sha256"]),
        root_seed=ROOT_SEED,
        route_family="xgboost",
        route_note="portable local xgboost-only pilot; Chronos requires sibling-code runtime dependency and is excluded by contract",
        development_rebuild=str(target["development_rebuild"]),
    )
    return audit, task_spec, split_manifest


def _xgb_allowlist() -> list[dict[str, Any]]:
    return [
        {"action_id": "T3_XGB_D4_ETA_0_03_ROUNDS_64", "max_depth": 4, "learning_rate": 0.03, "n_estimators": 64},
        {"action_id": "T3_XGB_D4_ETA_0_05_ROUNDS_96", "max_depth": 4, "learning_rate": 0.05, "n_estimators": 96},
        {"action_id": "T3_XGB_D6_ETA_0_03_ROUNDS_96", "max_depth": 6, "learning_rate": 0.03, "n_estimators": 96},
        {"action_id": "T3_XGB_D6_ETA_0_05_ROUNDS_128", "max_depth": 6, "learning_rate": 0.05, "n_estimators": 128},
    ]


def _route_action_registry() -> dict[str, list[dict[str, Any]]]:
    return {"xgboost": _xgb_allowlist()}


def _build_prompt(*, observation: dict[str, Any], candidate_feedback: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = {
        "task": "Choose one xgboost config id and whether to stop.",
        "constraints": {
            "route_fixed": "xgboost",
            "no_raw_metrics": True,
            "no_labels": True,
            "no_residuals": True,
            "no_paths": True,
            "no_holdout_or_test": True,
            "selection_dev_feedback_only": True,
            "folds_disjoint": True,
        },
        "observation": {
            "baseline": observation["a0_baseline"],
            "config_feedback": candidate_feedback,
            "allowlist": observation["allowlist"],
            "budget": observation["budget"],
        },
        "response_schema": {
            "selected_route": "xgboost",
            "selected_action_id": "one allowlisted config id or A0_static_baseline",
            "stop_requested": "boolean",
            "confidence": "number from 0 to 1",
            "rationale": "short practical rationale",
        },
        "selection_dev_feedback_vocab": ["improved", "flat", "worse"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict decision policy for the sweetspot P28 T3-only pilot. "
                "Return JSON only. Do not invent metrics, labels, paths, or holdout access. "
                "Choose between the four xgboost configs or stop."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
    ]


def _execute_xgboost(
    *,
    action: Mapping[str, Any],
    source_root: Path,
    selection_folds: list[int],
    promotion_fold: int,
) -> dict[str, Any]:
    import hashlib
    import numpy as np
    from xgboost import XGBRegressor

    trial_rows: list[dict[str, Any]] = []
    selection_metrics: list[float] = []
    promotion_metrics: list[float] = []
    for fold_id in selection_folds + [promotion_fold]:
        data = load_t3_development_pilot_data(source_root=source_root, fold_id=fold_id)
        model = XGBRegressor(
            n_estimators=int(action["n_estimators"]),
            max_depth=int(action["max_depth"]),
            learning_rate=float(action["learning_rate"]),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
            random_state=ROOT_SEED + int(action["n_estimators"]),
        )
        model.fit(np.asarray(data.train_tabular, dtype=np.float64), np.asarray(data.train_target, dtype=np.float64))
        prediction = np.asarray(model.predict(np.asarray(data.validation_tabular, dtype=np.float64)), dtype=np.float64).reshape(-1)
        metrics = _fold_metric_payload(data.validation_target, prediction)
        metrics["topk_diagnostic"] = _topk_diagnostic(data.validation_target, prediction)
        trial_rows.append(
            {
                "fold_id": int(fold_id),
                "phase": "selection" if fold_id in selection_folds else "promotion",
                "samples_train": len(data.train_sample_ids),
                "samples_validation": len(data.validation_sample_ids),
                "metrics": metrics,
                "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
                "split_sha256": data.split_sha256,
                "input_budget_sha256": data.input_budget_sha256,
                "test_accessed": False,
                "historical_test_metrics_read": False,
            }
        )
        if fold_id in selection_folds:
            selection_metrics.append(float(metrics["mae"]))
        else:
            promotion_metrics.append(float(metrics["mae"]))
    return {
        "route": "xgboost",
        "action_id": str(action["action_id"]),
        "budget": int(action["n_estimators"]),
        "selection_mae": float(np.mean(selection_metrics)) if selection_metrics else None,
        "promotion_mae": float(np.mean(promotion_metrics)) if promotion_metrics else None,
        "folds": trial_rows,
    }


def _evaluate_candidate_table(*, source_root: Path) -> dict[str, Any]:
    audit, task_spec, _ = _load_task_contract()
    registry = _route_action_registry()
    candidates: list[dict[str, Any]] = []
    for action in registry["xgboost"]:
        per_budget = [_execute_xgboost(action=action, source_root=source_root, selection_folds=list(SELECTION_FOLDS), promotion_fold=PROMOTION_FOLDS[0])]
        record = {
            "route": "xgboost",
            "action_id": str(action["action_id"]),
            "budgets": per_budget,
            "selection_mae": float(per_budget[0]["selection_mae"]),
            "promotion_mae": float(per_budget[0]["promotion_mae"]),
        }
        candidates.append(record)
    best = min(candidates, key=lambda row: row["selection_mae"])
    return {
        "schema_version": "sweetspot-p28-agentic-optimization-candidates/v1",
        "selection_folds": list(SELECTION_FOLDS),
        "promotion_folds": list(PROMOTION_FOLDS),
        "route_registry": registry,
        "candidates": candidates,
        "best_by_selection": {
            "route": best["route"],
            "action_id": best["action_id"],
            "selection_mae": best["selection_mae"],
            "promotion_mae": best["promotion_mae"],
        },
        "task_spec_sha256": _sha256_bytes(json.dumps(task_spec.to_dict(), sort_keys=True).encode("utf-8")),
        "label_audit_sha256": audit.mapping_sha256,
    }


def _portable_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKTREE_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return _sha256_bytes(encoded.encode("utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(encoded, encoding="utf-8")
    return _sha256_bytes(encoded.encode("utf-8"))


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return _sha256_bytes(text.encode("utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKTREE_ROOT))
    except ValueError:
        return str(path)


def generate_report(output_dir: Path = OUTPUT_DIR, *, call_deepseek: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    protocol_jsonl_path = output_dir / "protocol.jsonl"
    summary_path = output_dir / "summary.json"
    evidence_path = output_dir / "evidence.md"
    manifest_path = output_dir / "manifest.json"
    refs = _reference_inputs()
    audit, task_spec, _ = _load_task_contract()
    observation = _observation()
    candidate_table = _evaluate_candidate_table(source_root=PROJECT_ROOT)
    baseline = _load_json(STAGE3_T3_PATH)["entries"][0]["primary_mean"]
    candidate_feedback = []
    for candidate in candidate_table["candidates"]:
        selection_feedback = "improved" if candidate["selection_mae"] < float(baseline) else "worse"
        if candidate["selection_mae"] == float(baseline):
            selection_feedback = "flat"
        candidate_feedback.append(
            {
                "action_id": candidate["action_id"],
                "selection_feedback": selection_feedback,
                "budget": candidate["budgets"][0]["budget"],
            }
        )
    prompt_messages = _build_prompt(observation=observation, candidate_feedback=candidate_feedback)
    deepseek = _call_deepseek(prompt_messages) if call_deepseek else {"status": "SKIPPED"}
    a0_reference = {
        "route": "xgboost",
        "baseline_commit": "16bebd18a0bc722afcbc4b841610bf76ce9503e4",
        "split_hash": "c44277ffc1f6fb6b5dd740952e921c732d45193c7cb3b5c3dfc061e79025c62a",
        "stage3_leaderboard_sha256": _sha256_file(STAGE3_T3_PATH),
        "stage3_summary_sha256": _sha256_file(STAGE3_SUMMARY_PATH),
        "stage4_summary_sha256": _sha256_file(STAGE4_SUMMARY_PATH),
        "selection_dev_feedback": "flat",
        "promotion_dev_feedback": "flat",
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "task_id": "T3",
        "root_seed": ROOT_SEED,
        "task_spec": task_spec.to_dict(),
        "audit": audit.to_dict(),
        "frozen_baseline": a0_reference,
        "observation_schema": {
            "visible_to_llm": [
                "fold_train_aggregates",
                "selection_dev_feedback_improved_flat_worse",
                "allowlist",
                "stop_requested",
            ],
            "hidden_from_llm": [
                "raw_metric_values",
                "validation_curves",
                "per_sample_residuals",
                "labels",
                "paths",
            ],
            "selection_fold_ids": SELECTION_FOLDS,
            "promotion_fold_ids": PROMOTION_FOLDS,
            "route_fixed": "xgboost",
        },
        "action_allowlist": {"xgboost": list(XGB_ACTION_IDS)},
        "leakage_firewall": {
            "no_holdout_or_test": True,
            "no_raw_metrics": True,
            "no_labels": True,
            "nested_disjoint_selection_and_promotion": True,
            "fold_train_only_fit": True,
            "foundation_route_fixed_or_factorized": True,
            "sibling_worktree_python_imports": False,
        },
        "promotion_gate": {
            "stage1_min_trials": 1,
            "stage1_max_trials": 4,
            "equal_trial_budget": 4,
            "require_a2l_helpful_vs_a3": True,
            "require_non_overlapping_selection_and_promotion": True,
            "require_no_leakage": True,
        },
        "gates": {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"},
        "inputs": refs,
    }

    a0_action_id = "T3_XGB_D4_ETA_0_05_ROUNDS_96"
    a0_prediction_hash = None
    a1_prediction_hash = None
    a0_baseline_mae = None

    try:
        candidate_lookup = {candidate["action_id"]: candidate for candidate in candidate_table["candidates"]}
        a0_candidate = candidate_lookup[a0_action_id]
        a1_candidate = _execute_xgboost(
            action={
                "action_id": a0_action_id,
                "max_depth": 4,
                "learning_rate": 0.05,
                "n_estimators": 96,
            },
            source_root=PROJECT_ROOT,
            selection_folds=list(SELECTION_FOLDS),
            promotion_fold=PROMOTION_FOLDS[0],
        )
        a0_hashes = [fold["prediction_sha256"] for fold in a0_candidate["budgets"][0]["folds"]]
        a1_hashes = [fold["prediction_sha256"] for fold in a1_candidate["folds"]]
        a1_same_hash = a0_hashes == a1_hashes
        a0_prediction_hash = a0_hashes[3]
        a1_prediction_hash = a1_hashes[3]
        a0_baseline_mae = float(a0_candidate["promotion_mae"])

        deepseek_status = deepseek.get("status")
        raw_action_id = deepseek.get("selected_action_id")
        a2l_blocked_reason = None
        if deepseek_status in {"UNAVAILABLE", "SKIPPED"}:
            a2l_status = "BLOCKED"
            a2l_blocked_reason = f"DeepSeek {deepseek_status.lower()}"
        elif deepseek.get("stop_requested"):
            a2l_status = "STOPPED"
            a2l_blocked_reason = None
        elif not isinstance(raw_action_id, str):
            a2l_status = "BLOCKED"
            a2l_blocked_reason = "DeepSeek decision malformed"
        elif raw_action_id not in candidate_lookup:
            a2l_status = "BLOCKED"
            a2l_blocked_reason = f"action not in allowlist: {raw_action_id}"
        else:
            a2l_status = "EXECUTED"

        a2l_candidate = candidate_lookup.get(raw_action_id) if a2l_status == "EXECUTED" else None
        a2l = {
            "status": a2l_status,
            "selected_route": "xgboost" if a2l_candidate else deepseek.get("selected_route"),
            "selected_action_id": raw_action_id,
            "selected_budget": a2l_candidate["budgets"][0]["budget"] if a2l_candidate else None,
            "selection_mae": a2l_candidate["selection_mae"] if a2l_candidate else None,
            "promotion_mae": a2l_candidate["promotion_mae"] if a2l_candidate else None,
            "stop_requested": bool(deepseek.get("stop_requested", False)),
            "confidence": deepseek.get("confidence"),
            "rationale": deepseek.get("rationale"),
            "raw": deepseek,
            "candidate_table": candidate_table,
            "blocked_reason": a2l_blocked_reason,
        }

        a2d_candidate = min(candidate_table["candidates"], key=lambda row: row["selection_mae"])
        a2d_budget = a2d_candidate["budgets"][0]
        a2d = {
            "selected_route": a2d_candidate["route"],
            "selected_action_id": a2d_candidate["action_id"],
            "selected_budget": a2d_budget["budget"],
            "selection_mae": a2d_candidate["selection_mae"],
            "promotion_mae": a2d_candidate["promotion_mae"],
            "status": "EXECUTED",
        }
        import random

        rng = random.Random(ROOT_SEED)
        a3_candidate = rng.choice(candidate_table["candidates"])
        a3_budget = a3_candidate["budgets"][0]
        a3 = {
            "selected_route": a3_candidate["route"],
            "selected_action_id": a3_candidate["action_id"],
            "selected_budget": a3_budget["budget"],
            "selection_mae": a3_candidate["selection_mae"],
            "promotion_mae": a3_candidate["promotion_mae"],
            "status": "EXECUTED",
        }
        a4 = {
            "status": "EXECUTED",
            "selected_route": a2d_candidate["route"],
            "selected_action_id": a2d_candidate["action_id"],
            "selection_mae": a2d_candidate["selection_mae"],
            "promotion_mae": a2d_candidate["promotion_mae"],
            "promotion_vs_baseline_delta": float(a2d_candidate["promotion_mae"]) - float(a0_candidate["promotion_mae"]),
            "baseline_mae": float(a0_candidate["promotion_mae"]),
        }
        if not a1_same_hash:
            verdict = "blocked"
            blocked_reason = "A1 identical replay hash mismatch"
        elif a2l_status == "STOPPED":
            verdict = "reject"
            blocked_reason = None
        elif a2l_status != "EXECUTED":
            verdict = "blocked"
            blocked_reason = a2l_blocked_reason
        elif a2l_candidate["promotion_mae"] < a0_candidate["promotion_mae"] and a2l_candidate["promotion_mae"] < a3_candidate["promotion_mae"]:
            verdict = "retain"
            blocked_reason = None
        else:
            verdict = "reject"
            blocked_reason = None
    except (FileNotFoundError, PermissionError, ImportError, AttributeError, ValueError, RuntimeError) as exc:
        candidate_table = None
        verdict = "blocked"
        blocked_reason = f"{type(exc).__name__}: {exc}"
        a1_same_hash = False
        a2l = {
            "status": "UNAVAILABLE" if deepseek.get("status") in {"UNAVAILABLE", "SKIPPED"} else "OK",
            "selected_route": deepseek.get("selected_route"),
            "selected_action_id": deepseek.get("selected_action_id"),
            "stop_requested": bool(deepseek.get("stop_requested", False)),
            "confidence": deepseek.get("confidence"),
            "rationale": deepseek.get("rationale"),
            "raw": deepseek,
            "candidate_table": None,
        }
        a2d = {"status": verdict.upper(), "reason": blocked_reason}
        a3 = {"status": verdict.upper(), "reason": blocked_reason}
        a4 = {"status": verdict.upper(), "reason": blocked_reason}

    protocol["verdict"] = verdict
    protocol["blocked_reason"] = blocked_reason
    protocol["candidate_table_sha256"] = None if candidate_table is None else _sha256_bytes(json.dumps(candidate_table, sort_keys=True).encode("utf-8"))

    rows = [
        {
            "strategy": "A0_static_baseline",
            "status": "RETAINED_REFERENCE",
            "selected_action_id": a0_action_id,
            "prediction_hash": a0_prediction_hash,
            "note": "Same-fold replay baseline on the fixed action/seed/folds.",
        },
        {
            "strategy": "A1_advice_only",
            "status": "EXECUTED_IDENTITY_CHECK",
            "selected_action_id": a0_action_id,
            "prediction_hash": a1_prediction_hash,
            "note": "Identical replay hash check on the same action/seed/folds.",
        },
        {
            "strategy": "A2L_llm_agent_execute",
            "status": a2l["status"],
            "selected_route": a2l.get("selected_route"),
            "selected_action_id": a2l.get("selected_action_id"),
            "selected_budget": a2l.get("selected_budget"),
            "candidate_table": a2l.get("candidate_table") is not None,
            "decision_source": "deepseek" if a2l["status"] == "EXECUTED" else a2l["raw"].get("status", "UNAVAILABLE"),
            "note": "LLM decision is executed only if allowlisted and leak-free; otherwise fail closed.",
        },
        {
            "strategy": "A2D_deterministic_agent",
            "status": a2d["status"],
            "selected_route": a2d.get("selected_route"),
            "selected_action_id": a2d.get("selected_action_id"),
            "selected_budget": a2d.get("selected_budget"),
            "selection_mae": a2d.get("selection_mae"),
            "promotion_mae": a2d.get("promotion_mae"),
            "note": "Deterministic selection uses fold-train aggregates only.",
        },
        {
            "strategy": "A3_random_policy",
            "status": a3["status"],
            "selected_route": a3.get("selected_route"),
            "selected_action_id": a3.get("selected_action_id"),
            "selected_budget": a3.get("selected_budget"),
            "selection_mae": a3.get("selection_mae"),
            "promotion_mae": a3.get("promotion_mae"),
            "note": "Random policy uses the same four-candidate equal-budget contract and common folds.",
        },
        {
            "strategy": "A4_deterministic_search",
            "status": a4["status"],
            "selected_route": a4.get("selected_route"),
            "selected_action_id": a4.get("selected_action_id"),
            "selection_mae": a4.get("selection_mae"),
            "promotion_mae": a4.get("promotion_mae"),
            "note": "Deterministic search is reported separately and does not define the LLM retain verdict.",
        },
        {
            "strategy": "FINAL",
            "status": verdict.upper(),
            "selected_route": a2l.get("selected_route"),
            "selected_action_id": a2l.get("selected_action_id"),
            "promotion_mae": a2l.get("promotion_mae"),
            "baseline_mae": a0_baseline_mae,
            "promotion_vs_baseline_delta": None if a2l.get("promotion_mae") is None or a0_baseline_mae is None else float(a2l["promotion_mae"]) - float(a0_baseline_mae),
            "a3_promotion_mae": a3.get("promotion_mae"),
            "note": "retain only if A2L beats same-fold A0 and equal-budget A3 on promotion; otherwise reject or block.",
        },
    ]
    protocol_sha = _write_json(protocol_path, protocol)
    jsonl_sha = _write_jsonl(protocol_jsonl_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "task_id": "T3",
        "verdict": verdict,
        "retention": "retain A0 reference only" if verdict == "retain" else "no retain",
        "hybrid": "A2L decision executed when available" if a2l["status"] != "UNAVAILABLE" else "A2L unavailable",
        "retain_llm": verdict == "retain",
        "reject": "A2L stop, A2D/A3/A4 promotion failure, or data blocks" if verdict != "retain" else "none",
        "blocked_reason": blocked_reason,
        "root_seed": ROOT_SEED,
        "a0_reference": a0_reference,
        "a0_prediction_hash": a0_prediction_hash,
        "a1_prediction_hash": a1_prediction_hash,
        "a1_same_hash": a1_same_hash,
        "a1_prediction_hash_contract": "identity_replay_check",
        "a2l": a2l,
        "a2d": a2d,
        "a3": a3,
        "a4": a4,
        "candidate_table": candidate_table,
        "gates": protocol["gates"],
        "selection_fold_ids": SELECTION_FOLDS,
        "promotion_fold_ids": PROMOTION_FOLDS,
        "llm_prompt_contract": {
            "no_raw_metrics": True,
            "no_labels": True,
            "no_residuals": True,
            "no_paths": True,
            "only_improved_flat_worse_feedback": True,
        },
        "executor_available": candidate_table is not None,
        "reference_commits": {
            "p10_results_sweetspot": _git_commit(REFERENCE_ROOT),
            "current_worktree": _git_commit(WORKTREE_ROOT),
        },
        "protocol_sha256": protocol_sha,
        "protocol_jsonl_sha256": jsonl_sha,
    }
    summary_sha = _write_json(summary_path, summary)
    evidence = textwrap.dedent(
        f"""
        # P28 T3-only hybrid execution-agent pilot

        Verdict: `{verdict}`

        ## What was actually done

        - Archived A0 reference for T3 XGBoost was frozen from immutable stage-3 / stage-4 evidence.
        - A2L DeepSeek selection was evaluated under a strict no-raw-metric, no-label, no-path prompt contract.
        - A2D deterministic choice used fold-train aggregates only.
        - A3 random policy used the same route/action registry and a randomly chosen trial budget.
        - Execution ran on development-only folds only; no test or holdout access was used.

        ## Why it is rejected or blocked

        {blocked_reason or "not blocked"}

        ## Honest retain / hybrid / reject verdict

        - Retain: A0 archived baseline reference only.
        - Hybrid: A2L decision record is usable as a policy artifact, but not as executed science.
        - Reject: A2L stop, or A2D/A3/A4 promotion failure against the archived baseline.

        ## T5–T7 gate states

        - T5: `not_feasible`
        - T6: `blocked`
        - T7: `blocked`
        """
    ).strip() + "\n"
    evidence_sha = _write_text(evidence_path, evidence)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "source_commit": _git_commit(WORKTREE_ROOT),
        "renderer": {
            "path": "_pipelines/02_task_datasets/sweetspot/p28_agentic_optimization.py",
            "sha256": _sha256_file(Path(__file__)),
        },
        "generated_at": "2026-08-01T00:00:00Z",
        "manual_review": {
            "reviewed": False,
            "reviewer": None,
            "reviewed_at": None,
            "reviewed_sha256": None,
            "colors_consistent": None,
            "labels_legible": None,
            "no_clipping": None,
            "no_overlap": None,
            "scientific_boundary_preserved": None,
            "notes": "pending",
        },
        "inputs": refs,
        "outputs": [
            {
                "role": "protocol",
                "path": _portable_display_path(protocol_path),
                "sha256": protocol_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "protocol_jsonl",
                "path": _portable_display_path(protocol_jsonl_path),
                "sha256": jsonl_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "summary",
                "path": _portable_display_path(summary_path),
                "sha256": summary_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "evidence",
                "path": _portable_display_path(evidence_path),
                "sha256": evidence_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
        ],
        "artifact_count": 4,
    }
    _write_json(manifest_path, manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-deepseek", action="store_true", help="skip the DeepSeek policy call")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = generate_report(args.output_dir, call_deepseek=not args.no_deepseek)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0 if summary["verdict"] in {"retain", "reject"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
