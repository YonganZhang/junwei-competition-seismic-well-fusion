"""Fault-track-only provenance and historical-artifact safeguards."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any


TRACK_DIR = Path(__file__).resolve().parent
RUNS_DIR = TRACK_DIR / "_outputs" / "runs"
HISTORICAL_MANIFEST_PATH = TRACK_DIR / "historical_baseline_manifest.json"
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_run_dir(run_name: str) -> Path:
    if not _SAFE_RUN_NAME.fullmatch(run_name):
        raise ValueError(
            "run-name must contain only letters, numbers, dot, underscore, or hyphen "
            "and may not start with punctuation"
        )
    return RUNS_DIR / run_name


def software_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-learn", "h5py", "joblib", "segyio", "matplotlib")
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def load_historical_manifest() -> dict[str, Any]:
    if not HISTORICAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"historical baseline manifest is required before an audited run: {HISTORICAL_MANIFEST_PATH}"
        )
    return json.loads(HISTORICAL_MANIFEST_PATH.read_text(encoding="utf-8"))


def historical_artifact_paths() -> dict[str, Path]:
    manifest = load_historical_manifest()
    return {relative: TRACK_DIR / relative for relative in manifest["artifacts"]}


def missing_historical_artifacts() -> list[str]:
    return [relative for relative, path in historical_artifact_paths().items() if not path.is_file()]


def verify_historical_artifacts() -> dict[str, str]:
    manifest = load_historical_manifest()
    verified: dict[str, str] = {}
    for relative, expected in manifest["artifacts"].items():
        path = TRACK_DIR / relative
        if not path.exists():
            raise FileNotFoundError(f"historical artifact was removed: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"historical artifact changed: {relative}; expected sha256={expected}, actual={actual}"
            )
        verified[relative] = actual
    return verified


def verify_historical_artifacts_if_present() -> dict[str, str]:
    """Verify a complete local historical bundle, but allow a portable checkout without it.

    Historical checkpoints and images are intentionally local integration assets.
    A checkout with none of them may run the portable pipeline; a partial bundle
    still fails loudly because it is ambiguous and cannot satisfy the manifest.
    """
    paths = historical_artifact_paths()
    present = [relative for relative, path in paths.items() if path.is_file()]
    if not present:
        return {}
    missing = [relative for relative in paths if relative not in present]
    if missing:
        raise FileNotFoundError(
            "partial historical baseline bundle: either provide every manifest artifact "
            f"or none of them; missing={missing}"
        )
    return verify_historical_artifacts()
