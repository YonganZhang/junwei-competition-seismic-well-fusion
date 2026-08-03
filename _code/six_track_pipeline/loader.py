"""Load the six plain-dictionary pipeline adapters without package coupling."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from .contracts import PipelineContractError, TRACKS, TRACK_DIRS


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def adapter_path(project_root: Path, track: str) -> Path:
    return project_root / "_pipelines" / "02_task_datasets" / TRACK_DIRS[track] / "pipeline_adapter.py"


def _load_dict(path: Path) -> dict[str, Any]:
    module_name = "_six_track_adapter_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, "ADAPTER", None)
    if not isinstance(value, dict):
        raise TypeError("ADAPTER must be a plain dict")
    return value


def load_adapter_dicts(project_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Load exactly the canonical six adapters, rejecting extras and omissions."""

    root = project_root.resolve()
    base = root / "_pipelines" / "02_task_datasets"
    expected = {adapter_path(root, track).resolve() for track in TRACKS}
    discovered = {path.resolve() for path in base.glob("*/pipeline_adapter.py")}
    errors: list[str] = []
    for missing in sorted(expected - discovered):
        errors.append(f"missing canonical adapter: {missing.relative_to(root)}")
    for extra in sorted(discovered - expected):
        errors.append(f"unexpected adapter outside the six-track registry: {extra.relative_to(root)}")
    if errors:
        raise PipelineContractError(errors)

    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for track in TRACKS:
        path = adapter_path(root, track)
        try:
            raw = _load_dict(path)
        except Exception as exc:  # the concrete import error is part of the contract report
            errors.append(f"{track}: cannot load {path.relative_to(root)}: {exc}")
            continue
        loaded[track] = (path, raw)
    if errors:
        raise PipelineContractError(errors)
    return loaded
