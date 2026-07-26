from __future__ import annotations

import ast
import inspect
import json
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path

from _models.gaia_dagt import (
    AgentEvidence,
    AgentParseError,
    AgentUnavailableError,
    CacheCorruptionError,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SOURCE_MANIFEST,
    DryRunResult,
    GaiaDAGTAdapter,
    ModelBatch,
    ModelOutput,
    SourceLockError,
    TrackSpec,
    agent_unavailable,
    apply_control,
    counterfactual_control,
    load_cached_agent_evidence,
    parse_api_payload,
    predictive_text_agent,
    random_control,
    real_control,
    render_sci_svg,
    require_api_key,
    shuffle_control,
    supervisory_qc_agent,
    verify_default_source_manifest,
)
from _models.gaia_dagt.adapter import evaluate_multitask_batch
from _models.gaia_dagt.contracts import infer_shape


def make_features(shape: tuple[int, ...], base: float = 0.1) -> object:
    counter = {"value": base}

    def _build(remaining: tuple[int, ...]) -> object:
        if not remaining:
            value = round(counter["value"], 3)
            counter["value"] += 0.1
            return value
        return [_build(remaining[1:]) for _ in range(remaining[0])]

    return _build(shape)


def make_track_spec(task_type: str, target_fields: tuple[str, ...], metric_names: tuple[str, ...] = ("accuracy",)) -> TrackSpec:
    return TrackSpec(
        track_id=f"gaia-dagt-{task_type}",
        task_type=task_type,
        modality="offline-dummy",
        input_fields=("report", "features"),
        target_fields=target_fields,
        allowed_paths=("/tmp/allowed",),
        forbidden_paths=("/tmp/forbidden",),
        metric_names=metric_names,
        base_seed=2693,
        source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        provenance={"source": "unit-test"},
    )


def shape_of(value: object) -> tuple[int, ...]:
    return infer_shape(value)


def make_classification_batch() -> ModelBatch:
    spec = make_track_spec("classification", ("c0", "c1", "c2"))
    return ModelBatch(
        track_spec=spec,
        features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        target=[1, 0],
        mask=[1, 1],
        metadata={"task_type": "classification"},
    )


def make_regression_batch() -> ModelBatch:
    spec = make_track_spec("regression", ("r0", "r1"))
    return ModelBatch(
        track_spec=spec,
        features=[[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], [[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]]],
        target=[[0.5, 0.6], [1.5, 1.6]],
        mask=[1, 1],
        metadata={"task_type": "regression"},
    )


def make_segmentation_2d_batch() -> ModelBatch:
    spec = make_track_spec("segmentation_2d", ("f0", "f1", "f2", "f3"))
    features = make_features((2, 3, 4, 5), base=0.1)
    target = make_features((2, 4, 5), base=0.0)
    return ModelBatch(
        track_spec=spec,
        features=features,
        target=target,
        mask=make_features((2, 4, 5), base=1.0),
        metadata={"task_type": "segmentation_2d"},
    )


def make_volume_3d_batch() -> ModelBatch:
    spec = make_track_spec("volume_3d", ("v0", "v1", "v2", "v3"))
    features = make_features((1, 2, 2, 3, 4), base=0.2)
    target = make_features((1, 2, 3, 4), base=0.0)
    voxel_index_map = make_features((1, 2, 3, 4, 3), base=0.0)
    return ModelBatch(
        track_spec=spec,
        features=features,
        target=target,
        mask=make_features((1, 2, 3, 4), base=1.0),
        voxel_index_map=voxel_index_map,
        metadata={"task_type": "volume_3d"},
    )


def make_multitask_batch() -> ModelBatch:
    spec = make_track_spec("multitask", ("task_cls", "task_reg", "task_skip"))
    return ModelBatch(
        track_spec=spec,
        features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        task_targets={"task_cls": 1, "task_reg": 2.5, "task_skip": None},
        task_metrics={"task_cls": "accuracy", "task_reg": "mae", "task_skip": "accuracy"},
        task_masks={"task_cls": True, "task_reg": [1, 1], "task_skip": False},
        feasibility={"task_cls": True, "task_reg": True, "task_skip": False},
        metadata={"task_type": "multitask"},
    )


class GaiaDAGTContractTests(unittest.TestCase):
    def test_01_source_lock(self) -> None:
        statuses = verify_default_source_manifest()
        self.assertEqual(len(statuses), 6)
        self.assertTrue(all(status.status == "match" for status in statuses))

    def test_02_track_spec_round_trip(self) -> None:
        spec = make_track_spec("classification", ("c0", "c1", "c2"))
        restored = TrackSpec.from_dict(spec.to_dict())
        self.assertEqual(spec, restored)
        self.assertEqual(spec.cache_key(), restored.cache_key())

    def test_03_batch_round_trip_includes_task_fields(self) -> None:
        batch = make_multitask_batch()
        restored = ModelBatch.from_dict(batch.to_dict())
        self.assertEqual(batch, restored)
        self.assertEqual(batch.task_targets, restored.task_targets)
        self.assertEqual(batch.task_metrics, restored.task_metrics)
        self.assertEqual(batch.task_masks, restored.task_masks)
        self.assertEqual(batch.feasibility, restored.feasibility)

    def test_04_agent_evidence_deny_list_and_path_rejection(self) -> None:
        evidence = AgentEvidence(
            prompt_version=DEFAULT_PROMPT_VERSION,
            agent_mode="agent_unavailable",
            source_text_hash="abc123",
            structured_priors={"signal": 1.0},
            confidence=0.0,
            evidence=("neutral",),
            warnings=("none",),
            provenance={"report_kind": "unit"},
            source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        )
        again = AgentEvidence.from_dict(evidence.to_dict())
        self.assertEqual(evidence, again)
        for payload in (
            {"ground_truth_label": 7},
            {"test_fold_score": 0.2},
            {"nested": {"ground_truth_label": 1}},
        ):
            with self.assertRaises(ValueError):
                AgentEvidence(
                    prompt_version=DEFAULT_PROMPT_VERSION,
                    agent_mode="agent_unavailable",
                    source_text_hash="abc123",
                    structured_priors=payload,
                    source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
                )
        for payload in (
            {"path": "/mnt/data/secret/file.txt"},
            {"path": "file:///mnt/data/secret/file.txt"},
            {"path": r"C:\\Users\\x\\secret.txt"},
            {"nested": {"uri": "/mnt/data/secret/file.txt"}},
        ):
            with self.assertRaises(ValueError):
                AgentEvidence(
                    prompt_version=DEFAULT_PROMPT_VERSION,
                    agent_mode="agent_unavailable",
                    source_text_hash="abc123",
                    structured_priors=payload,
                    source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
                )

    def test_05_predict_ignores_target_mask_and_task_values(self) -> None:
        adapter = GaiaDAGTAdapter(track_spec=make_segmentation_2d_batch().track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST, seed=2693)
        base = make_segmentation_2d_batch()
        mutated = replace(
            base,
            target=make_features((2, 4, 5), base=9.0),
            mask=make_features((2, 4, 5), base=7.0),
            task_targets={"task_a": 999},
            task_masks={"task_a": False},
        )
        first = adapter.predict(base)
        second = adapter.predict(mutated)
        self.assertEqual(first.prediction, second.prediction)
        self.assertEqual(first.logits, second.logits)
        self.assertEqual(first.provenance["signature"], second.provenance["signature"])
        changed = replace(base, features=make_features((2, 3, 4, 5), base=0.9))
        third = adapter.predict(changed)
        self.assertNotEqual(first.prediction, third.prediction)
        self.assertNotEqual(first.logits, third.logits)

    def test_06_classification_and_regression_shapes(self) -> None:
        cls_batch = make_classification_batch()
        reg_batch = make_regression_batch()
        adapter_cls = GaiaDAGTAdapter(track_spec=cls_batch.track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST)
        adapter_reg = GaiaDAGTAdapter(track_spec=reg_batch.track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST)
        cls_out = adapter_cls.predict(cls_batch)
        reg_out = adapter_reg.predict(reg_batch)
        self.assertEqual(shape_of(cls_out.prediction), (2,))
        self.assertEqual(shape_of(cls_out.logits), (2, 3))
        self.assertEqual(shape_of(reg_out.prediction), (2, 2))
        self.assertEqual(shape_of(reg_out.logits), (2, 2))

    def test_07_segmentation_2d_and_volume_3d_shapes(self) -> None:
        seg_batch = make_segmentation_2d_batch()
        vol_batch = make_volume_3d_batch()
        seg_out = GaiaDAGTAdapter(track_spec=seg_batch.track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST).predict(seg_batch)
        vol_out = GaiaDAGTAdapter(track_spec=vol_batch.track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST).predict(vol_batch)
        self.assertEqual(shape_of(seg_out.prediction), (2, 4, 5))
        self.assertEqual(shape_of(seg_out.logits), (2, 4, 4, 5))
        self.assertEqual(shape_of(vol_out.prediction), (1, 2, 3, 4))
        self.assertEqual(shape_of(vol_out.logits), (1, 4, 2, 3, 4))
        self.assertEqual(vol_out.diagnostics["voxel_index_map"], vol_batch.voxel_index_map)

    def test_08_multitask_skip_and_report(self) -> None:
        batch = make_multitask_batch()
        adapter = GaiaDAGTAdapter(track_spec=batch.track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST)
        out = adapter.predict(batch)
        self.assertIsInstance(out.prediction, dict)
        self.assertIn("task_cls", out.prediction)
        self.assertIn("task_reg", out.prediction)
        self.assertIn("task_skip", out.prediction)
        report = evaluate_multitask_batch(batch, out)
        self.assertIn("task_cls", report["task_metrics"])
        self.assertIn("task_reg", report["task_metrics"])
        self.assertNotIn("task_skip", report["task_metrics"])
        self.assertIn("task_skip", report["skipped_tasks"])
        self.assertGreaterEqual(report["aggregate"]["accuracy"], 0.0)

    def test_09_agent_modes_and_offline_predictive_stub(self) -> None:
        class _StubClient:
            def complete(self, _: str) -> str:
                payload = {
                    "structured_priors": {"signal": 3},
                    "confidence": 0.5,
                    "evidence": ["stub"],
                    "warnings": ["offline"],
                    "provenance": {"source": "stub"},
                }
                return json.dumps(payload, sort_keys=True)

        manifest_digest = DEFAULT_SOURCE_MANIFEST.digest()
        predictive = predictive_text_agent(
            "structured report",
            client=_StubClient(),
            source_manifest_digest=manifest_digest,
            seed=7,
        )
        self.assertEqual(predictive.agent_mode, "predictive_text_agent")
        self.assertTrue(predictive.cache_key)
        qc = supervisory_qc_agent("qc report", source_manifest_digest=manifest_digest)
        self.assertEqual(qc.agent_mode, "supervisory_qc_agent")
        neutral = agent_unavailable("raw report", source_manifest_digest=manifest_digest)
        self.assertEqual(neutral.agent_mode, "agent_unavailable")
        self.assertEqual(neutral.confidence, 0.0)

    def test_10_controls_are_deterministic(self) -> None:
        items = ("a", "b", "c", "d")
        self.assertEqual(real_control(items), items)
        self.assertEqual(shuffle_control(items, seed=11), shuffle_control(items, seed=11))
        self.assertEqual(random_control(items, seed=11), random_control(items, seed=11))
        self.assertEqual(counterfactual_control(items, seed=11), counterfactual_control(items, seed=11))
        self.assertEqual(apply_control(items, "shuffle", 11), shuffle_control(items, 11))

    def test_11_api_and_cache_fail_loud(self) -> None:
        with self.assertRaises(AgentUnavailableError):
            require_api_key(None)

        class _TimeoutClient:
            def complete(self, _: str) -> str:
                raise TimeoutError("timeout")

        with self.assertRaises(AgentUnavailableError):
            predictive_text_agent(
                "report text",
                client=_TimeoutClient(),
                source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
            )
        with self.assertRaises(AgentParseError):
            parse_api_payload("{not json")
        with self.assertRaises(AgentParseError):
            parse_api_payload(json.dumps([1, 2, 3]))
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text("[]")
            with self.assertRaises(CacheCorruptionError):
                load_cached_agent_evidence(cache)

    def test_12_source_lock_mismatch_is_fail_loud(self) -> None:
        from _models.gaia_dagt.source_lock import SourceFileRecord, SourceManifest

        bad_manifest = SourceManifest(
            upstream_repo_root=DEFAULT_SOURCE_MANIFEST.upstream_repo_root,
            commit=DEFAULT_SOURCE_MANIFEST.commit,
            files=(SourceFileRecord(path=DEFAULT_SOURCE_MANIFEST.files[0].path, sha256="0" * 64),),
        )
        with self.assertRaises(SourceLockError):
            bad_manifest.verify()

    def test_13_dry_run_is_escaped_and_neutral(self) -> None:
        batch = make_segmentation_2d_batch()
        adapter = GaiaDAGTAdapter(track_spec=batch.track_spec, source_manifest=DEFAULT_SOURCE_MANIFEST)
        result = adapter.dry_run(
            {"report": "<qc & notes>", "flags": ["stable"], "notes": "synthetic"},
            batch,
            mode="supervisory_qc_agent",
        )
        self.assertIsInstance(result, DryRunResult)
        self.assertTrue(result.svg.startswith("<svg"))
        self.assertNotIn("accuracy", result.svg.lower())
        svg = render_sci_svg(
            "A&B <test>",
            {"dry_run": 1.0},
            {"note": "<raw>&", "path": "neutral"},
        )
        self.assertIn("A&amp;B &lt;test&gt;", svg)
        self.assertIn("&lt;raw&gt;&amp;", svg)

        multitask_batch = make_multitask_batch()
        multitask_result = GaiaDAGTAdapter(
            track_spec=multitask_batch.track_spec,
            source_manifest=DEFAULT_SOURCE_MANIFEST,
        ).dry_run("multitask qc summary", multitask_batch)
        self.assertEqual(multitask_result.batch.task_targets, multitask_batch.task_targets)
        self.assertEqual(multitask_result.batch.task_metrics, multitask_batch.task_metrics)
        self.assertEqual(multitask_result.batch.task_masks, multitask_batch.task_masks)
        self.assertEqual(multitask_result.batch.feasibility, multitask_batch.feasibility)

    def test_14_static_no_network_secret_or_target_flow(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        banned_modules = {
            "requests",
            "urllib",
            "urllib3",
            "socket",
            "httpx",
            "aiohttp",
            "openai",
            "smtplib",
            "paramiko",
        }
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name.split(".")[0], banned_modules, path)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(node.module.split(".")[0], banned_modules, path)

        source = textwrap.dedent(inspect.getsource(GaiaDAGTAdapter.predict))
        tree = ast.parse(source)
        forbidden = {"target", "mask", "task_targets", "task_masks"}
        chain_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names = []
                current = node
                while isinstance(current, ast.Attribute):
                    names.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name) and current.id == "batch":
                    names.append(current.id)
                    chain_names.add(".".join(reversed(names)))
        self.assertFalse(any(name in forbidden for name in chain_names), chain_names)
        self.assertNotIn("batch.target", source)
        self.assertNotIn("batch.mask", source)
        self.assertNotIn("batch.task_targets", source)
        self.assertNotIn("batch.task_masks", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
