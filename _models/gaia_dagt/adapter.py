from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from .agents import agent_unavailable, supervisory_qc_agent
from .contracts import (
    AgentEvidence,
    ModelBatch,
    ModelOutput,
    TrackSpec,
    infer_shape,
)
from .source_lock import DEFAULT_SOURCE_MANIFEST, SourceManifest


def _stable_score(value: Any) -> int:
    payload = repr(value).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _control_seed(agent_evidence: AgentEvidence | None, fallback_seed: int, fallback_control: str) -> tuple[int, str]:
    if agent_evidence is not None:
        return agent_evidence.seed, agent_evidence.control_mode
    return fallback_seed, fallback_control


def _sequence_signature(
    features: Any,
    *,
    track_id: str,
    seed: int,
    control_mode: str,
    agent_mode: str,
) -> int:
    payload = {
        "features": features,
        "track_id": track_id,
        "seed": seed,
        "control_mode": control_mode,
        "agent_mode": agent_mode,
    }
    return _stable_score(payload)


def _channel_count(track_spec: TrackSpec) -> int:
    return max(2, len(track_spec.target_fields) or len(track_spec.metric_names) or 2)


def _shape_at(batch: Any, index: int, default: int = 0) -> int:
    shape = infer_shape(batch)
    if index < len(shape):
        return shape[index]
    return default


def _sequence_prediction(signature: int, index: int, class_count: int, regression: bool) -> Any:
    local = signature + index * 17
    if regression:
        return ((local % 1000) / 100.0)
    return local % class_count


def _sequence_logits(signature: int, index: int, class_count: int) -> list[float]:
    winner = (signature + index * 17) % class_count
    return [1.0 if cls == winner else 0.0 for cls in range(class_count)]


def _predict_sequence_rank2(features: Sequence[Any], track_spec: TrackSpec, seed: int, control_mode: str, agent_mode: str, regression: bool) -> tuple[list[Any], list[Any]]:
    class_count = _channel_count(track_spec)
    predictions = []
    logits = []
    signature = _sequence_signature(features, track_id=track_spec.track_id, seed=seed, control_mode=control_mode, agent_mode=agent_mode)
    for index, sample in enumerate(features):
        sample_signature = _sequence_signature(sample, track_id=f"{track_spec.track_id}:{index}", seed=seed, control_mode=control_mode, agent_mode=agent_mode)
        value = _sequence_prediction(sample_signature, index, class_count, regression=regression)
        predictions.append(value)
        if regression:
            logits.append(value)
        else:
            logits.append(_sequence_logits(sample_signature, index, class_count))
    return predictions, logits


def _predict_sequence_rank3(features: Sequence[Any], track_spec: TrackSpec, seed: int, control_mode: str, agent_mode: str, regression: bool) -> tuple[list[Any], list[Any]]:
    class_count = _channel_count(track_spec)
    predictions = []
    logits = []
    for batch_index, sample in enumerate(features):
        batch_predictions = []
        batch_logits = []
        for step_index, step in enumerate(sample):
            signature = _sequence_signature(
                step,
                track_id=f"{track_spec.track_id}:{batch_index}:{step_index}",
                seed=seed,
                control_mode=control_mode,
                agent_mode=agent_mode,
            )
            value = _sequence_prediction(signature, step_index, class_count, regression=regression)
            batch_predictions.append(value)
            if regression:
                batch_logits.append(value)
            else:
                batch_logits.append(_sequence_logits(signature, step_index, class_count))
        predictions.append(batch_predictions)
        logits.append(batch_logits)
    return predictions, logits


def _spatial_score(signature: int, batch_index: int, coords: tuple[int, ...], class_index: int, class_count: int) -> float:
    coord_term = sum((idx + 1) * coord for idx, coord in enumerate(coords))
    winner = (signature + batch_index * 13 + coord_term) % max(class_count, 1)
    return 1.0 if class_index == winner else 0.0


def _build_spatial_prediction(signature: int, spatial_shape: tuple[int, ...], class_count: int, batch_index: int) -> Any:
    def _walk(coords: tuple[int, ...], remaining: tuple[int, ...]) -> Any:
        if not remaining:
            coord_term = sum((idx + 1) * coord for idx, coord in enumerate(coords))
            return (signature + batch_index * 19 + coord_term) % class_count
        return [_walk(coords + (i,), remaining[1:]) for i in range(remaining[0])]

    return _walk((), spatial_shape)


def _build_spatial_logits(signature: int, spatial_shape: tuple[int, ...], class_count: int, batch_index: int) -> Any:
    def _channel(channel: int, coords: tuple[int, ...], remaining: tuple[int, ...]) -> Any:
        if not remaining:
            return _spatial_score(signature, batch_index, coords, channel, class_count)
        return [_channel(channel, coords + (i,), remaining[1:]) for i in range(remaining[0])]

    return [_channel(channel, (), spatial_shape) for channel in range(class_count)]


def _truthy_mask(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_truthy_mask(item) for item in value)
    return bool(value)


def _mean_abs_error(target: Any, prediction: Any) -> float:
    totals = []

    def _walk(left: Any, right: Any) -> None:
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
            for item_left, item_right in zip(left, right):
                _walk(item_left, item_right)
        else:
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                totals.append(abs(float(left) - float(right)))

    _walk(target, prediction)
    return sum(totals) / len(totals) if totals else 0.0


def _accuracy(target: Any, prediction: Any) -> float:
    target_shape = infer_shape(target)
    prediction_shape = infer_shape(prediction)
    if target_shape != prediction_shape:
        return 0.0
    matches = 0
    total = 0

    def _walk(left: Any, right: Any) -> None:
        nonlocal matches, total
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
            for item_left, item_right in zip(left, right):
                _walk(item_left, item_right)
        else:
            total += 1
            if left == right:
                matches += 1

    _walk(target, prediction)
    return matches / total if total else 0.0


def _binary_iou(target: Any, prediction: Any) -> tuple[float, float]:
    intersection = 0
    union = 0

    def _walk(left: Any, right: Any) -> None:
        nonlocal intersection, union
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
            for item_left, item_right in zip(left, right):
                _walk(item_left, item_right)
        else:
            left_val = int(bool(left))
            right_val = int(bool(right))
            intersection += left_val & right_val
            union += left_val | right_val

    _walk(target, prediction)
    iou = intersection / union if union else 1.0
    f1 = (2 * intersection) / (union + intersection) if (union + intersection) else 1.0
    return iou, f1


def evaluate_batch(batch: ModelBatch, output: ModelOutput) -> Mapping[str, float]:
    kind = batch.kind()
    if kind == "classification":
        accuracy = _accuracy(batch.target, output.prediction)
        return {"accuracy": accuracy, "loss": 1.0 - accuracy}
    if kind == "regression":
        mae = _mean_abs_error(batch.target, output.prediction)
        return {"mae": mae, "loss": mae}
    if kind in {"segmentation_2d", "volume_3d"}:
        iou, f1 = _binary_iou(batch.target, output.prediction)
        accuracy = _accuracy(batch.target, output.prediction)
        return {"accuracy": accuracy, "miou": iou, "macro_f1": f1, "loss": 1.0 - iou}
    if kind == "multitask":
        report = evaluate_multitask_batch(batch, output)
        return report["aggregate"]
    raise ValueError(f"Unsupported batch kind: {kind}")


def evaluate_multitask_batch(batch: ModelBatch, output: ModelOutput) -> dict[str, Any]:
    task_metrics: dict[str, float] = {}
    skipped: list[str] = []
    if not isinstance(output.prediction, Mapping):
        raise ValueError("Multitask prediction must be a mapping")
    for task_name, target in batch.task_targets.items():
        if not batch.feasibility.get(task_name, True):
            skipped.append(task_name)
            continue
        if not _truthy_mask(batch.task_masks.get(task_name, 1)):
            skipped.append(task_name)
            continue
        if target is None:
            skipped.append(task_name)
            continue
        metric_name = batch.task_metrics.get(task_name, "accuracy")
        prediction = output.prediction.get(task_name)
        if prediction is None:
            skipped.append(task_name)
            continue
        if metric_name in {"accuracy", "f1", "macro_f1"}:
            task_metrics[task_name] = _accuracy(target, prediction)
        elif metric_name in {"mae", "l1"}:
            task_metrics[task_name] = _mean_abs_error(target, prediction)
        else:
            task_metrics[task_name] = _accuracy(target, prediction)
    if task_metrics:
        aggregate = sum(task_metrics.values()) / len(task_metrics)
    else:
        aggregate = 0.0
    return {"task_metrics": task_metrics, "aggregate": {"accuracy": aggregate, "loss": 1.0 - aggregate}, "skipped_tasks": skipped}


@dataclass(frozen=True, slots=True)
class DryRunResult:
    track_spec: TrackSpec
    agent_evidence: AgentEvidence
    batch: ModelBatch
    output: ModelOutput
    metric: Mapping[str, float]
    svg: str


def render_sci_svg(title: str, metric: Mapping[str, float], provenance: Mapping[str, Any]) -> str:
    summary_bits = []
    for key in sorted(metric):
        summary_bits.append(f"{key}={metric[key]:.4f}")
    provenance_bits = []
    for key in sorted(provenance):
        provenance_bits.append(f"{key}:{provenance[key]}")
    body = " | ".join(summary_bits)
    footer = " | ".join(provenance_bits)
    title = escape(str(title))
    body = escape(body)
    footer = escape(footer)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180" viewBox="0 0 720 180">'
        '<rect x="0" y="0" width="720" height="180" fill="#f7f7f9" stroke="#222" stroke-width="1"/>'
        '<text x="24" y="34" font-size="22" font-family="Arial, sans-serif" fill="#111">'
        f"{title}"
        '</text>'
        '<text x="24" y="70" font-size="16" font-family="Arial, sans-serif" fill="#333">'
        f"{body}"
        '</text>'
        '<text x="24" y="104" font-size="13" font-family="Arial, sans-serif" fill="#555">'
        f"{footer}"
        '</text>'
        '<line x1="24" y1="132" x2="696" y2="132" stroke="#999" stroke-width="1"/>'
        '<circle cx="60" cy="132" r="6" fill="#4b7bec"/>'
        '<circle cx="110" cy="132" r="6" fill="#f39c12"/>'
        '<circle cx="160" cy="132" r="6" fill="#27ae60"/>'
        "</svg>"
    )


@dataclass(frozen=True, slots=True)
class GaiaDAGTAdapter:
    track_spec: TrackSpec
    source_manifest: SourceManifest
    control_mode: str = "real"
    seed: int = 2693

    @classmethod
    def from_default_manifest(cls, track_spec: TrackSpec, *, control_mode: str = "real", seed: int = 2693) -> "GaiaDAGTAdapter":
        DEFAULT_SOURCE_MANIFEST.verify()
        return cls(track_spec=track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST, control_mode=control_mode, seed=seed)

    def _manifest_digest(self) -> str:
        return self.source_manifest.digest()

    def build_agent_evidence(self, report: Any, *, mode: str = "agent_unavailable") -> AgentEvidence:
        manifest_digest = self._manifest_digest()
        if mode == "predictive_text_agent":
            from .agents import predictive_text_agent

            return predictive_text_agent(
                report,
                client=self._stub_client(report),
                source_manifest_digest=manifest_digest,
                control_mode=self.control_mode,
                seed=self.seed,
            )
        if mode == "supervisory_qc_agent":
            return supervisory_qc_agent(
                report,
                source_manifest_digest=manifest_digest,
                control_mode=self.control_mode,
                seed=self.seed,
            )
        return agent_unavailable(
            report,
            source_manifest_digest=manifest_digest,
            control_mode=self.control_mode,
            seed=self.seed,
        )

    def _stub_client(self, report: Any) -> Any:
        class _Client:
            def __init__(self, text: str) -> None:
                self.text = text

            def complete(self, _: str) -> str:
                signature = _stable_score(self.text) % 11
                payload = {
                    "structured_priors": {"signature": signature},
                    "confidence": 0.5,
                    "evidence": [f"stub-signature={signature}"],
                    "warnings": ["stub-client"],
                    "provenance": {"client": "stub"},
                }
                return json.dumps(payload, sort_keys=True)

        return _Client(_normalize_report(report))

    def predict(self, batch: ModelBatch) -> ModelOutput:
        if batch.track_spec.cache_key() != self.track_spec.cache_key():
            raise ValueError("TrackSpec mismatch")
        seed, control_mode = _control_seed(batch.agent_evidence, self.seed, self.control_mode)
        agent_mode = batch.agent_evidence.agent_mode if batch.agent_evidence else "agent_unavailable"
        signature = _sequence_signature(
            batch.features,
            track_id=batch.track_spec.track_id,
            seed=seed,
            control_mode=control_mode,
            agent_mode=agent_mode,
        )
        kind = batch.kind()
        if kind == "classification":
            shape = infer_shape(batch.features)
            if len(shape) not in {2, 3}:
                raise ValueError(f"Classification features must be [B,C] or [B,L,C], got {shape}")
            regression = False
            if len(shape) == 2:
                batch_size = shape[0]
                prediction = []
                logits = []
                class_count = _channel_count(batch.track_spec)
                for sample_index in range(batch_size):
                    sample_signature = _sequence_signature(
                        batch.features[sample_index],
                        track_id=f"{batch.track_spec.track_id}:{sample_index}",
                        seed=seed,
                        control_mode=control_mode,
                        agent_mode=agent_mode,
                    )
                    prediction.append(_sequence_prediction(sample_signature, sample_index, class_count, regression))
                    logits.append(_sequence_logits(sample_signature, sample_index, class_count))
            else:
                prediction, logits = _predict_sequence_rank3(batch.features, batch.track_spec, seed, control_mode, agent_mode, regression)
            metric = evaluate_batch(batch, ModelOutput(track_id=batch.track_spec.track_id, prediction=prediction, logits=logits))
        elif kind == "regression":
            shape = infer_shape(batch.features)
            if len(shape) not in {2, 3}:
                raise ValueError(f"Regression features must be [B,C] or [B,L,C], got {shape}")
            regression = True
            if len(shape) == 2:
                batch_size = shape[0]
                prediction = []
                logits = []
                class_count = _channel_count(batch.track_spec)
                for sample_index in range(batch_size):
                    sample_signature = _sequence_signature(
                        batch.features[sample_index],
                        track_id=f"{batch.track_spec.track_id}:{sample_index}",
                        seed=seed,
                        control_mode=control_mode,
                        agent_mode=agent_mode,
                    )
                    value = _sequence_prediction(sample_signature, sample_index, class_count, regression)
                    prediction.append(value)
                    logits.append(value)
            else:
                prediction, logits = _predict_sequence_rank3(batch.features, batch.track_spec, seed, control_mode, agent_mode, regression)
            metric = evaluate_batch(batch, ModelOutput(track_id=batch.track_spec.track_id, prediction=prediction, logits=logits))
        elif kind == "segmentation_2d":
            shape = infer_shape(batch.features)
            if len(shape) != 4:
                raise ValueError(f"2D segmentation features must be [B,C,H,W], got {shape}")
            batch_size, _, height, width = shape
            class_count = _channel_count(batch.track_spec)
            prediction = []
            logits = []
            for batch_index in range(batch_size):
                sample_signature = _sequence_signature(
                    batch.features[batch_index],
                    track_id=f"{batch.track_spec.track_id}:{batch_index}",
                    seed=seed,
                    control_mode=control_mode,
                    agent_mode=agent_mode,
                )
                prediction.append(_build_spatial_prediction(sample_signature, (height, width), class_count, batch_index))
                logits.append(_build_spatial_logits(sample_signature, (height, width), class_count, batch_index))
            metric = evaluate_batch(batch, ModelOutput(track_id=batch.track_spec.track_id, prediction=prediction, logits=logits))
        elif kind == "volume_3d":
            shape = infer_shape(batch.features)
            if len(shape) != 5:
                raise ValueError(f"3D volume features must be [B,C,D,H,W], got {shape}")
            batch_size, _, depth, height, width = shape
            class_count = _channel_count(batch.track_spec)
            prediction = []
            logits = []
            for batch_index in range(batch_size):
                sample_signature = _sequence_signature(
                    batch.features[batch_index],
                    track_id=f"{batch.track_spec.track_id}:{batch_index}",
                    seed=seed,
                    control_mode=control_mode,
                    agent_mode=agent_mode,
                )
                prediction.append(_build_spatial_prediction(sample_signature, (depth, height, width), class_count, batch_index))
                logits.append(_build_spatial_logits(sample_signature, (depth, height, width), class_count, batch_index))
            metric = evaluate_batch(batch, ModelOutput(track_id=batch.track_spec.track_id, prediction=prediction, logits=logits))
        elif kind == "multitask":
            prediction: dict[str, Any] = {}
            logits: dict[str, Any] = {}
            for task_name in batch.track_spec.target_fields:
                task_signature = _sequence_signature(
                    batch.features,
                    track_id=f"{batch.track_spec.track_id}:{task_name}",
                    seed=seed,
                    control_mode=control_mode,
                    agent_mode=agent_mode,
                )
                metric_name = batch.task_metrics.get(task_name, "accuracy")
                if metric_name in {"mae", "l1", "regression"}:
                    prediction[task_name] = (task_signature % 1000) / 100.0
                    logits[task_name] = prediction[task_name]
                else:
                    class_count = _channel_count(batch.track_spec)
                    prediction[task_name] = task_signature % class_count
                    logits[task_name] = [1.0 if cls == prediction[task_name] else 0.0 for cls in range(class_count)]
            metric_report = evaluate_multitask_batch(batch, ModelOutput(track_id=batch.track_spec.track_id, prediction=prediction, logits=logits))
            metric = metric_report["aggregate"]
        else:
            raise ValueError(f"Unsupported task kind: {kind}")
        output = ModelOutput(
            track_id=batch.track_spec.track_id,
            prediction=prediction,
            logits=logits,
            metric=metric,
            uncertainty=None,
            agent_mode=agent_mode,
            provenance={
                "control_mode": control_mode,
                "seed": seed,
                "signature": signature,
                "source_manifest_digest": self._manifest_digest(),
            },
            diagnostics={
                "kind": kind,
                "shape": batch.shape(),
                "voxel_index_map": batch.voxel_index_map if kind == "volume_3d" else None,
            },
        )
        return output

    def dry_run(self, report: Any, batch: ModelBatch, *, mode: str = "agent_unavailable") -> DryRunResult:
        evidence = self.build_agent_evidence(report, mode=mode)
        dry_batch = ModelBatch(
            track_spec=batch.track_spec,
            features=batch.features,
            target=batch.target,
            mask=batch.mask,
            task_targets=batch.task_targets,
            task_metrics=batch.task_metrics,
            task_masks=batch.task_masks,
            feasibility=batch.feasibility,
            metadata=batch.metadata,
            agent_evidence=evidence,
            voxel_index_map=batch.voxel_index_map,
        )
        output = self.predict(dry_batch)
        svg = render_sci_svg(
            title=f"{self.track_spec.track_id} offline dry run",
            metric={"dry_run": 1.0, "metric_count": float(len(output.metric))},
            provenance={"agent_mode": evidence.agent_mode, "cache_key": evidence.cache_key},
        )
        return DryRunResult(
            track_spec=self.track_spec,
            agent_evidence=evidence,
            batch=dry_batch,
            output=output,
            metric=output.metric,
            svg=svg,
        )
