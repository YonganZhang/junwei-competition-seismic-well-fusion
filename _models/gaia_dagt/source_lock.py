from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import subprocess
from typing import Any, Mapping

from .contracts import canonical_json, sha256_json

UPSTREAM_REPO_ROOT = Path("/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac")
UPSTREAM_COMMIT = "0987684d6ecb7409bade73a35555593b678d70e1"


@dataclass(frozen=True, slots=True)
class SourceFileRecord:
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SourceManifestStatus:
    path: str
    expected_sha256: str
    actual_sha256: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class SourceManifest:
    upstream_repo_root: str
    commit: str
    files: tuple[SourceFileRecord, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream_repo_root": self.upstream_repo_root,
            "commit": self.commit,
            "files": [record.to_dict() for record in self.files],
            "provenance": dict(sorted(self.provenance.items(), key=lambda item: str(item[0]))),
        }

    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def verify(self) -> tuple[SourceManifestStatus, ...]:
        repo_root = Path(self.upstream_repo_root)
        if not repo_root.exists():
            raise SourceLockError(f"Upstream repo root does not exist: {repo_root}")
        try:
            subprocess.check_output(["git", "-C", str(repo_root), "cat-file", "-e", self.commit], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise SourceLockError(f"Upstream commit is not available: {self.commit}") from exc
        statuses: list[SourceManifestStatus] = []
        for record in self.files:
            path = Path(record.path)
            if not path.exists():
                raise SourceLockError(f"Missing upstream file: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != record.sha256:
                raise SourceLockError(
                    f"Source hash mismatch for {path}: expected {record.sha256}, got {actual}"
                )
            statuses.append(
                SourceManifestStatus(
                    path=record.path,
                    expected_sha256=record.sha256,
                    actual_sha256=actual,
                    status="match",
                )
            )
        return tuple(statuses)


class SourceLockError(RuntimeError):
    pass


DEFAULT_SOURCE_MANIFEST = SourceManifest(
    upstream_repo_root=str(UPSTREAM_REPO_ROOT),
    commit=UPSTREAM_COMMIT,
    files=(
        SourceFileRecord(
            path="/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac/dagt/agents/prompt_templates.py",
            sha256="355513fb7dffb65f90566dee5bfc44d5115305bdd8ab175ff61c1d856d61e1c8",
        ),
        SourceFileRecord(
            path="/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac/dagt/agents/agentize.py",
            sha256="a68f716858d5e147a2ec9889cf0c43fe426ea500eaf0ff6861a5bd2575711296",
        ),
        SourceFileRecord(
            path="/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac/scripts/finetune_chronos2.py",
            sha256="90536b6c47abed5209f112fefcd21ed003a325fc936c776fb1f594fac4f661e6",
        ),
        SourceFileRecord(
            path="/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac/_pipelines/_steps.yml",
            sha256="d7454d75e2a97c5909ac26e30df963822eb4bddcbee899f19c4bb1e8623591d0",
        ),
        SourceFileRecord(
            path="/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac/_pipelines/build_domain.yml",
            sha256="365f57d473c92a73f60a9865e0e4b9ee7cdf77f490da2391c52bd392e509d149",
        ),
        SourceFileRecord(
            path="/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac/_pipelines/predict.yml",
            sha256="b80ffb0b67b4a314aa7d7958ed3f9cc825cf9e32d72db35dc16bf7513bd2c79b",
        ),
    ),
)


def verify_default_source_manifest() -> tuple[SourceManifestStatus, ...]:
    return DEFAULT_SOURCE_MANIFEST.verify()
