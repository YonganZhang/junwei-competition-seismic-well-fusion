from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    AgentEvidence,
    DEFAULT_PROMPT_VERSION,
    canonical_json,
    sha256_json,
    sha256_text,
)


class AgentUnavailableError(RuntimeError):
    pass


class AgentParseError(RuntimeError):
    pass


class CacheCorruptionError(RuntimeError):
    pass


def require_api_key(api_key: str | None) -> str:
    if not api_key:
        raise AgentUnavailableError("API key missing")
    return api_key


def parse_api_payload(payload: str | bytes) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AgentParseError("Bad JSON payload") from exc
    if not isinstance(parsed, dict):
        raise AgentParseError("API payload must be a JSON object")
    return parsed


def _normalize_report(report: Any) -> str:
    if isinstance(report, (str, bytes)):
        return report.decode("utf-8") if isinstance(report, bytes) else report
    return canonical_json(report)


def _build_cache_key(
    *,
    agent_mode: str,
    source_text_hash: str,
    source_manifest_digest: str,
    prompt_version: str,
    control_mode: str,
    seed: int,
    structured_priors: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "agent_mode": agent_mode,
            "source_text_hash": source_text_hash,
            "source_manifest_digest": source_manifest_digest,
            "prompt_version": prompt_version,
            "control_mode": control_mode,
            "seed": seed,
            "structured_priors": structured_priors,
            "provenance": provenance,
        }
    )


def load_cached_agent_evidence(path: str | Path) -> AgentEvidence:
    cache_path = Path(path)
    if not cache_path.exists():
        raise CacheCorruptionError(f"Missing cache file: {cache_path}")
    try:
        payload = json.loads(cache_path.read_text())
    except json.JSONDecodeError as exc:
        raise CacheCorruptionError(f"Corrupt cache file: {cache_path}") from exc
    if not isinstance(payload, dict):
        raise CacheCorruptionError(f"Cache payload must be a JSON object: {cache_path}")
    try:
        return AgentEvidence.from_dict(payload)
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise CacheCorruptionError(f"Invalid cached evidence: {cache_path}") from exc


def _materialize_evidence(
    *,
    agent_mode: str,
    report_text: str,
    structured_priors: Mapping[str, Any] | None = None,
    confidence: float = 0.0,
    evidence: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
    source_manifest_digest: str = "",
    control_mode: str = "real",
    seed: int = 0,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> AgentEvidence:
    source_text_hash = sha256_text(report_text)
    structured_priors = dict(structured_priors or {})
    provenance = dict(provenance or {})
    cache_key = _build_cache_key(
        agent_mode=agent_mode,
        source_text_hash=source_text_hash,
        source_manifest_digest=source_manifest_digest,
        prompt_version=prompt_version,
        control_mode=control_mode,
        seed=seed,
        structured_priors=structured_priors,
        provenance=provenance,
    )
    return AgentEvidence(
        prompt_version=prompt_version,
        agent_mode=agent_mode,
        source_text_hash=source_text_hash,
        structured_priors=structured_priors,
        confidence=confidence,
        evidence=evidence,
        warnings=warnings,
        provenance=provenance,
        source_manifest_digest=source_manifest_digest,
        control_mode=control_mode,
        seed=seed,
        cache_key=cache_key,
    )


def predictive_text_agent(
    report: Any,
    *,
    client: Any | None = None,
    source_manifest_digest: str = "",
    control_mode: str = "real",
    seed: int = 0,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> AgentEvidence:
    if client is None:
        raise AgentUnavailableError("predictive_text_agent requires a stub or approved client")
    report_text = _normalize_report(report)
    try:
        if hasattr(client, "complete"):
            raw = client.complete(report_text)
        elif hasattr(client, "invoke"):
            raw = client.invoke(report_text)
        elif hasattr(client, "chat"):
            raw = client.chat(report_text)
        else:
            raise AgentUnavailableError("Client does not expose a supported completion method")
    except TimeoutError as exc:
        raise AgentUnavailableError("Agent timeout") from exc
    except AgentUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise AgentUnavailableError("Agent invocation failed") from exc
    payload = parse_api_payload(raw)
    structured_priors = payload.get("structured_priors", {})
    evidence = tuple(payload.get("evidence", ()))
    warnings = tuple(payload.get("warnings", ()))
    confidence = float(payload.get("confidence", 0.0))
    provenance = dict(payload.get("provenance", {}))
    return _materialize_evidence(
        agent_mode="predictive_text_agent",
        report_text=report_text,
        structured_priors=structured_priors,
        confidence=confidence,
        evidence=evidence,
        warnings=warnings,
        provenance=provenance,
        source_manifest_digest=source_manifest_digest,
        control_mode=control_mode,
        seed=seed,
        prompt_version=prompt_version,
    )


def supervisory_qc_agent(
    report: Any,
    *,
    source_manifest_digest: str = "",
    control_mode: str = "real",
    seed: int = 0,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> AgentEvidence:
    report_text = _normalize_report(report)
    structured_priors = {
        "report_length": len(report_text),
        "contains_warning_token": "warning" in report_text.lower(),
        "contains_qc_token": "qc" in report_text.lower(),
    }
    warnings = ("qc-only",) if structured_priors["contains_warning_token"] else ()
    return _materialize_evidence(
        agent_mode="supervisory_qc_agent",
        report_text=report_text,
        structured_priors=structured_priors,
        confidence=0.25,
        evidence=("qc-summary",),
        warnings=warnings,
        provenance={"report_kind": type(report).__name__},
        source_manifest_digest=source_manifest_digest,
        control_mode=control_mode,
        seed=seed,
        prompt_version=prompt_version,
    )


def agent_unavailable(
    report: Any,
    *,
    reason: str = "agent unavailable",
    source_manifest_digest: str = "",
    control_mode: str = "real",
    seed: int = 0,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> AgentEvidence:
    report_text = _normalize_report(report)
    return _materialize_evidence(
        agent_mode="agent_unavailable",
        report_text=report_text,
        structured_priors={"reason": reason},
        confidence=0.0,
        evidence=(reason,),
        warnings=("neutral-condition",),
        provenance={"report_kind": type(report).__name__},
        source_manifest_digest=source_manifest_digest,
        control_mode=control_mode,
        seed=seed,
        prompt_version=prompt_version,
    )
