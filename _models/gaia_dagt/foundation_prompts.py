"""Versioned supervisory-LLM prompt for every foundation-model route.

The LLM is a schema/QC reviewer.  It never supplies sample-level predictive
features and never sees query labels, frozen-test metrics, file paths or
credentials.  Modality conditioning remains in ``FoundationTaskEnvelope``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .contracts import canonical_json
from .foundation import FOUNDATION_PROMPT_VERSION, FoundationTaskEnvelope


SUPERVISORY_TEMPLATE_ID = "gaia.foundation.supervisory-qc.v1"
SUPERVISORY_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "checks", "warnings", "recommended_actions"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "warn", "fail"]},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "status", "evidence"],
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "warn", "fail"]},
                    "evidence": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
}
_FORBIDDEN_SUMMARY_TOKENS = (
    "target",
    "label",
    "truth",
    "groundtruth",
    "gold",
    "answer",
    "classid",
    "foregroundpoint",
    "faultstick",
    "residual",
    "predictionerror",
    "testmetric",
    "holdoutmetric",
    "filepath",
    "checkpointpath",
    "apikey",
    "token",
    "secret",
)


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _validate_safe_summary(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if any(token in normalized for token in _FORBIDDEN_SUMMARY_TOKENS):
                raise ValueError(f"supervisory summary contains forbidden field: {key}")
            _validate_safe_summary(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _validate_safe_summary(nested)
    elif isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith(("/", "file://")) or ":\\" in value:
            raise ValueError("supervisory summary contains a path-like value")


@dataclass(frozen=True, slots=True)
class SupervisoryPrompt:
    agent_model: str
    agent_revision: str
    system: str
    user: str
    response_schema: Mapping[str, Any]
    temperature: float
    max_output_tokens: int
    template_id: str = SUPERVISORY_TEMPLATE_ID
    prompt_version: str = FOUNDATION_PROMPT_VERSION

    def __post_init__(self) -> None:
        if not self.agent_model or not self.agent_revision:
            raise ValueError("supervisory prompt requires a source-locked agent model")
        if self.temperature != 0:
            raise ValueError("supervisory QC temperature must be zero")
        if self.max_output_tokens <= 0:
            raise ValueError("supervisory QC max_output_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "prompt_version": self.prompt_version,
            "agent_model": self.agent_model,
            "agent_revision": self.agent_revision,
            "system": self.system,
            "user": self.user,
            "response_schema": dict(self.response_schema),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    def prompt_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


def build_supervisory_prompt(
    request: FoundationTaskEnvelope,
    safe_summary: Mapping[str, Any],
    *,
    agent_model: str,
    agent_revision: str,
    max_output_tokens: int = 900,
) -> SupervisoryPrompt:
    _validate_safe_summary(safe_summary)
    system = (
        "You are the GAIA foundation-model supervisory QC agent. "
        "Review only schema, units, axes, masks, split isolation, conditioning provenance, "
        "runtime integrity, uncertainty and fallback readiness. "
        "Never infer or reconstruct query labels. Never request paths, credentials, hidden "
        "test statistics, raw predictions, target-derived prompts or post-cutoff values. "
        "A connected foundation model is not promoted unless its same-split gate and required "
        "controls pass. Return JSON matching the supplied schema and no prose outside JSON."
    )
    user_payload = {
        "request_hash": request.request_hash(),
        "track_id": request.track_id,
        "task_type": request.task_type,
        "axis_kind": request.axis_kind,
        "foundation_model": {
            "model_id": request.model.model_id,
            "source_revision": request.model.source_revision,
            "weights_revision": request.model.weights_revision,
            "weights_sha256": request.model.weights_sha256,
            "license_status": request.model.license_status,
        },
        "input_schema": [item.to_dict() for item in request.input_schema],
        "target_schema_names_only": [item.name for item in request.target_schema],
        "visibility": request.visibility.to_dict(),
        "conditioning": request.conditioning.to_dict(),
        "output_schema": dict(request.output_schema),
        "physical_constraints": dict(request.physical_constraints),
        "uncertainty": dict(request.uncertainty),
        "fallback": dict(request.fallback),
        "promotion_gate": request.promotion_gate.to_dict(),
        "current_state": request.state,
        "safe_runtime_summary": dict(safe_summary),
    }
    return SupervisoryPrompt(
        agent_model=agent_model,
        agent_revision=agent_revision,
        system=system,
        user=json.dumps(user_payload, sort_keys=True, ensure_ascii=False),
        response_schema=SUPERVISORY_RESPONSE_SCHEMA,
        temperature=0,
        max_output_tokens=int(max_output_tokens),
    )


def validate_supervisory_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the provider-neutral QC response without permissive coercion."""
    result = dict(payload)
    required = {"status", "checks", "warnings", "recommended_actions"}
    if set(result) != required:
        raise ValueError(
            "supervisory response fields must be exactly "
            f"{sorted(required)}, found {sorted(result)}"
        )
    allowed_status = {"pass", "warn", "fail"}
    if result["status"] not in allowed_status:
        raise ValueError("supervisory response status is invalid")
    checks = result["checks"]
    if not isinstance(checks, list):
        raise ValueError("supervisory response checks must be a list")
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping) or set(check) != {"name", "status", "evidence"}:
            raise ValueError(f"supervisory check[{index}] has invalid fields")
        if check["status"] not in allowed_status:
            raise ValueError(f"supervisory check[{index}] has invalid status")
        if not all(isinstance(check[key], str) for key in ("name", "evidence")):
            raise ValueError(f"supervisory check[{index}] text fields must be strings")
    for name in ("warnings", "recommended_actions"):
        if not isinstance(result[name], list) or not all(
            isinstance(item, str) for item in result[name]
        ):
            raise ValueError(f"supervisory response {name} must be a string list")
    return result


def invoke_supervisory_prompt(
    prompt: SupervisoryPrompt,
    *,
    client: Any,
) -> dict[str, Any]:
    """Invoke an approved client through one explicit, testable boundary.

    The caller owns credentials and transport.  This layer sends only the
    already-sanitized prompt and requires strict JSON on return.
    """
    complete = getattr(client, "complete", None)
    if not callable(complete):
        raise TypeError("supervisory client must expose complete(**kwargs)")
    raw = complete(
        system=prompt.system,
        user=prompt.user,
        response_schema=prompt.response_schema,
        temperature=prompt.temperature,
        max_output_tokens=prompt.max_output_tokens,
        model=prompt.agent_model,
        revision=prompt.agent_revision,
    )
    if isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise TypeError("supervisory client must return JSON text or a mapping")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("supervisory client returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("supervisory client response must be a JSON object")
    return validate_supervisory_response(payload)
