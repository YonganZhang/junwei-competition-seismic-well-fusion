"""Runtime guards shared by foundation-model adapters."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .foundation import FoundationModelRef, load_foundation_routes


def route_model(track_id: str) -> FoundationModelRef:
    routes = load_foundation_routes()
    if track_id not in routes:
        raise KeyError(f"no foundation route for {track_id}")
    return FoundationModelRef.from_dict(routes[track_id]["model"])


def verify_git_source(root: Path, expected_revision: str) -> Path:
    resolved = Path(root).resolve()
    if not (resolved / ".git").exists():
        raise FileNotFoundError(f"foundation source checkout is missing: {resolved}")
    actual = subprocess.check_output(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual != expected_revision:
        raise ValueError(
            f"foundation source revision mismatch: expected {expected_revision}, got {actual}"
        )
    return resolved


def verify_checkpoint(track_id: str, checkpoint: Path) -> Path:
    resolved = Path(checkpoint).resolve()
    route_model(track_id).verify_local_artifact(resolved)
    return resolved


def insert_import_root(root: Path, package_name: str) -> None:
    resolved_path = Path(root).resolve()
    resolved = str(resolved_path)
    module = sys.modules.get(package_name)
    if module is not None:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RuntimeError(
                f"{package_name} is already imported without an auditable source file"
            )
        module_path = Path(module_file).resolve()
        try:
            module_path.relative_to(resolved_path)
        except ValueError:
            raise RuntimeError(
                f"{package_name} is already imported from an unapproved source: {module_path}"
            )
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def consume_config(
    config: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    values = dict(config)
    missing = sorted(name for name in required if name not in values)
    if missing:
        raise ValueError(f"missing foundation adapter config: {missing}")
    unknown = sorted(set(values) - set(required) - set(optional))
    if unknown:
        raise ValueError(f"unsupported foundation adapter config: {unknown}")
    return values
