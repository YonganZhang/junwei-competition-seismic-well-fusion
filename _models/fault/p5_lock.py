"""Read-only P5 source, dependency, license, and weight gates for fault models."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from packaging.specifiers import SpecifierSet


LOCK_PATH = Path(__file__).with_name("p5_source_locks.json")


class P5ModelUnavailable(RuntimeError):
    """Structured, expected Stage-1 unavailability rather than a fake success."""

    def __init__(self, model_id: str, reason_code: str, detail: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(f"{model_id}: {reason_code}: {detail}")
        self.model_id = model_id
        self.reason_code = reason_code
        self.detail = detail
        self.evidence = dict(evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_locks() -> dict[str, dict[str, Any]]:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol") != "fault-p5-source-lock-v1" or payload.get("candidate_count") != 10:
        raise ValueError("unsupported fault P5 source lock protocol")
    records = [dict(item) for item in payload.get("models", ())]
    identifiers = [record.get("model_id") for record in records]
    if len(records) != 10 or len(set(identifiers)) != 10:
        raise ValueError("fault P5 source lock must contain exactly ten unique model IDs")
    for record in records:
        model_id = str(record.get("model_id", ""))
        source = dict(record.get("source", {}))
        revision = str(source.get("revision", ""))
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError(f"{model_id} must lock an exact 40-character lowercase commit")
        if not str(source.get("url", "")).startswith("https://github.com/"):
            raise ValueError(f"{model_id} source URL must be a primary GitHub URL")
        if not str(source.get("license_spdx", "")).strip():
            raise ValueError(f"{model_id} source license must be explicit")
    return {str(record["model_id"]): record for record in records}


def source_lock(model_id: str) -> dict[str, Any]:
    try:
        return load_source_locks()[model_id]
    except KeyError as exc:
        raise ValueError(f"model_id {model_id!r} is not in the fault P5 source lock") from exc


def lock_file_evidence() -> dict[str, Any]:
    return {"path": "_models/fault/p5_source_locks.json", "sha256": _sha256(LOCK_PATH)}


def _dependency_evidence(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    for raw in record.get("dependencies", ()):
        dependency = dict(raw)
        distribution = str(dependency["distribution"])
        import_name = str(dependency["import_name"])
        specifier = str(dependency["specifier"])
        importable = importlib.util.find_spec(import_name) is not None
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        version_ok = version is not None and version in SpecifierSet(specifier)
        item = {
            "distribution": distribution,
            "import_name": import_name,
            "required": specifier,
            "installed_version": version,
            "importable": importable,
            "version_ok": version_ok,
        }
        evidence.append(item)
        if not importable or not version_ok:
            failures.append(
                f"{distribution}{specifier} (installed={version!r}, importable={importable})"
            )
    return evidence, failures


def _locked_checkout_evidence(source_root: Path | None, revision: str) -> dict[str, Any]:
    if source_root is None:
        return {"status": "missing", "reason": "no locked source checkout supplied"}
    root = source_root.resolve()
    if not root.is_dir():
        return {"status": "missing", "reason": "source checkout directory does not exist"}
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "invalid", "reason": f"cannot audit checkout: {type(exc).__name__}"}
    return {
        "status": "verified" if head == revision and not dirty else "mismatch",
        "head": head,
        "expected_revision": revision,
        "tracked_files_dirty": bool(dirty),
    }


def evaluate_runtime_gate(
    model_id: str,
    *,
    source_root: Path | None = None,
    weight_path: Path | None = None,
) -> dict[str, Any]:
    """Return a stable gate record without importing a model or touching the network."""

    record = source_lock(model_id)
    source = dict(record["source"])
    weights = dict(record["weights"])
    evidence: dict[str, Any] = {
        "model_id": model_id,
        "source_url": source["url"],
        "source_revision": source["revision"],
        "source_tag": source.get("tag"),
        "code_license": source["license_spdx"],
        "commercial_use": source["commercial_use"],
        "weights": weights,
    }
    if source["commercial_use"] != "allowed":
        return {
            "status": "skipped",
            "reason_code": "NONCOMMERCIAL_LICENSE_NOT_APPROVED",
            "detail": f"code license {source['license_spdx']} is not approved for this execution",
            "evidence": evidence,
        }
    if weights.get("required") and (
        not weights.get("approved") or not weights.get("sha256") or weight_path is None
    ):
        return {
            "status": "skipped",
            "reason_code": "PRETRAINED_WEIGHT_NOT_APPROVED",
            "detail": "required upstream weight has no approved license/hash/local path",
            "evidence": evidence,
        }
    if weight_path is not None:
        if not weight_path.is_file():
            return {
                "status": "skipped",
                "reason_code": "WEIGHT_FILE_MISSING",
                "detail": "approved weight path does not exist",
                "evidence": evidence,
            }
        observed_weight_hash = _sha256(weight_path)
        evidence["observed_weight_sha256"] = observed_weight_hash
        if observed_weight_hash != weights.get("sha256"):
            return {
                "status": "skipped",
                "reason_code": "WEIGHT_HASH_MISMATCH",
                "detail": "local weight does not match the frozen source lock",
                "evidence": evidence,
            }

    adapter = dict(record["adapter"])
    if adapter["source_mode"] == "locked_checkout":
        checkout = _locked_checkout_evidence(source_root, source["revision"])
        evidence["source_checkout"] = checkout
        if checkout["status"] != "verified":
            return {
                "status": "skipped",
                "reason_code": "LOCKED_SOURCE_CHECKOUT_UNAVAILABLE",
                "detail": "exact clean upstream checkout is unavailable",
                "evidence": evidence,
            }

    dependencies, failures = _dependency_evidence(record)
    evidence["dependencies"] = dependencies
    if failures:
        return {
            "status": "skipped",
            "reason_code": "DEPENDENCY_UNAVAILABLE",
            "detail": "; ".join(failures),
            "evidence": evidence,
        }
    return {"status": "ready", "reason_code": None, "detail": None, "evidence": evidence}


def require_runtime_ready(
    model_id: str,
    *,
    source_root: Path | None = None,
    weight_path: Path | None = None,
) -> dict[str, Any]:
    result = evaluate_runtime_gate(model_id, source_root=source_root, weight_path=weight_path)
    if result["status"] != "ready":
        raise P5ModelUnavailable(
            model_id,
            str(result["reason_code"]),
            str(result["detail"]),
            result["evidence"],
        )
    return result
