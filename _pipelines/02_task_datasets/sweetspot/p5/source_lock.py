"""Load and validate the ten-candidate P5 source lock without importing them."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
DEFAULT_LOCK = HERE / "source_lock.v1.json"
REQUIRED_FIELDS = {
    "model_id", "family", "source_url", "documentation_url", "revision", "license",
    "environment_group", "required_module", "distribution", "accepted_version_prefixes",
    "weights", "implementation_mode", "source_equivalence",
}


def load_source_lock(path: Path = DEFAULT_LOCK) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sweetspot-p5-source-lock/v1":
        raise ValueError("unsupported sweetspot P5 source-lock schema")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 10:
        raise ValueError("source lock must contain exactly ten candidates")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("every source-lock entry must be an object")
        missing = sorted(REQUIRED_FIELDS - set(entry))
        if missing:
            raise ValueError(f"source-lock entry missing fields: {missing}")
        model_id = str(entry["model_id"])
        if model_id in result:
            raise ValueError(f"duplicate source lock for {model_id}")
        revision = str(entry["revision"])
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"{model_id}: revision must be a full lowercase Git commit")
        result[model_id] = entry
    return result


def inspect_runtime(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return import/version evidence; never install or download anything."""
    module = entry.get("required_module")
    distribution = entry.get("distribution")
    if not module:
        return {
            "available": False,
            "reason_code": "locked_source_checkout_unavailable",
            "module": None,
            "version": None,
            "version_allowed": False,
        }
    try:
        available = importlib.util.find_spec(str(module)) is not None
    except (ImportError, ModuleNotFoundError):
        available = False
    version = None
    if available and distribution:
        try:
            version = importlib.metadata.version(str(distribution))
        except importlib.metadata.PackageNotFoundError:
            version = None
    prefixes = tuple(str(value) for value in entry.get("accepted_version_prefixes", ()))
    version_allowed = bool(available and (not prefixes or (version and version.startswith(prefixes))))
    return {
        "available": available,
        "reason_code": None if available else "dependency_missing",
        "module": module,
        "version": version,
        "accepted_version_prefixes": list(prefixes),
        "version_allowed": version_allowed,
    }
