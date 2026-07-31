"""Canonical JSON, atomic writes and content-addressed run manifests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(payload: Any) -> bytes:
    if is_dataclass(payload):
        payload = asdict(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


@dataclass
class ArtifactManifest:
    run_id: str
    root: Path
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register(self, relative_path: str, *, role: str, metadata: Mapping[str, Any] | None = None) -> None:
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError("artifact path escapes run root")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        self.artifacts[relative_path] = {
            "role": role,
            "sha256": hash_file(candidate),
            "bytes": candidate.stat().st_size,
            "metadata": dict(metadata or {}),
        }

    def verify(self) -> None:
        for relative_path, record in self.artifacts.items():
            candidate = self.root / relative_path
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            if hash_file(candidate) != record["sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {relative_path}")

    def to_dict(self) -> dict[str, Any]:
        return {"manifest_version": "p4-v1", "run_id": self.run_id, "artifacts": self.artifacts}

    def write(self, relative_path: str = "manifest.json") -> Path:
        return atomic_write_json(self.root / relative_path, self.to_dict())
