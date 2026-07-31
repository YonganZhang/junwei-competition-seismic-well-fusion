from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

from _code.ml_framework.model_discovery import discover_model
from _models.gaia_dagt.foundation import (
    FOUNDATION_PROMPT_VERSION,
    ConditioningSpec,
    FoundationModelRef,
    FoundationTaskEnvelope,
    PromotionGate,
    TensorFieldSpec,
    VisibilityPolicy,
    load_foundation_routes,
)
from _models.gaia_dagt.foundation_prompts import (
    build_supervisory_prompt,
    invoke_supervisory_prompt,
)


SPLIT_HASH = "a" * 64
SAMPLE_HASH = "b" * 64


def model_ref() -> FoundationModelRef:
    return FoundationModelRef(
        model_id="example/foundation",
        family="test_foundation",
        source_url="https://example.invalid/source",
        source_revision="c" * 40,
        code_license="Apache-2.0",
        weights_uri="https://example.invalid/weights",
        weights_revision="d" * 40,
        weight_license="Apache-2.0",
        artifact_state="not_cached",
        weights_sha256="e" * 64,
        weights_size_bytes=10,
    )


def gate() -> PromotionGate:
    return PromotionGate(
        metric="mae",
        direction="minimize",
        baseline_id="same_split_baseline",
        required_relative_improvement=0.01,
        minimum_winning_folds=3,
        minimum_completed_fraction=0.8,
        controls=("random_init_same_architecture", "label_shuffle"),
    )


def field(name: str, unit: str = "unitless") -> TensorFieldSpec:
    return TensorFieldSpec(
        name=name,
        unit=unit,
        dtype="float32",
        shape=("B", 1, "T"),
        mask_semantics="true=observed",
        coordinate_frame="well_time",
    )


def time_envelope() -> FoundationTaskEnvelope:
    return FoundationTaskEnvelope(
        track_id="sweetspot",
        task_type="time_forecasting",
        axis_kind="time",
        model=model_ref(),
        split_hash=SPLIT_HASH,
        sample_ids_hash=SAMPLE_HASH,
        input_schema=(field("oil_rate", "m3/day"),),
        target_schema=(field("future_oil_rate", "m3/day"),),
        visibility=VisibilityPolicy(
            allowed_fields=("oil_rate", "timestamp"),
            forbidden_fields=("future_oil_rate",),
            cutoff="2026-01-31",
        ),
        conditioning=ConditioningSpec(
            kind="time_window",
            payload={
                "timestamps": ["2026-01-29", "2026-01-30", "2026-01-31"],
                "frequency": "D",
                "history_length": 3,
                "prediction_length": 2,
            },
        ),
        output_schema={"daily_quantiles": ["B", "Q", "H"]},
        physical_constraints={"non_negative": True},
        uncertainty={"kind": "quantiles"},
        fallback={"model_id": "same_split_xgboost"},
        promotion_gate=gate(),
    )


class FoundationContractTests(unittest.TestCase):
    def test_roundtrip_and_hash_are_deterministic(self) -> None:
        request = time_envelope()
        rebuilt = FoundationTaskEnvelope.from_dict(request.to_dict())
        self.assertEqual(rebuilt.to_dict(), request.to_dict())
        self.assertEqual(rebuilt.request_hash(), request.request_hash())
        self.assertEqual(rebuilt.prompt_version, FOUNDATION_PROMPT_VERSION)

    def test_hash_includes_units_frequency_and_mask_semantics(self) -> None:
        request = time_envelope()
        changed_unit = replace(
            request,
            input_schema=(replace(request.input_schema[0], unit="stb/day"),),
        )
        changed_frequency = replace(
            request,
            conditioning=ConditioningSpec(
                kind="time_window",
                payload={
                    **request.conditioning.payload,
                    "frequency": "h",
                },
            ),
        )
        changed_mask = replace(
            request,
            input_schema=(
                replace(request.input_schema[0], mask_semantics="true=imputed"),
            ),
        )
        self.assertNotEqual(request.request_hash(), changed_unit.request_hash())
        self.assertNotEqual(request.request_hash(), changed_frequency.request_hash())
        self.assertNotEqual(request.request_hash(), changed_mask.request_hash())

    def test_time_window_rejects_future_target_and_non_monotonic_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "future target"):
            ConditioningSpec(
                kind="time_window",
                payload={
                    "timestamps": [1, 2],
                    "frequency": "D",
                    "history_length": 2,
                    "prediction_length": 1,
                    "future_target": [9],
                },
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ConditioningSpec(
                kind="time_window",
                payload={
                    "timestamps": [2, 1],
                    "frequency": "D",
                    "history_length": 2,
                    "prediction_length": 1,
                },
            )

    def test_spatial_prompt_rejects_validation_gt_aliases(self) -> None:
        for source in ("ground_truth_clicks", "fault_stick_points", "label_sampler"):
            with self.subTest(source=source), self.assertRaisesRegex(
                ValueError, "target-derived"
            ):
                ConditioningSpec(
                    kind="spatial_prompt",
                    payload={
                        "prompt_kind": "points_3d",
                        "prompt_source": source,
                        "coordinate_frame": "resampled_kji",
                        "split_role": "validation",
                    },
                )

    def test_support_set_and_strict_volume_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be isolated"):
            ConditioningSpec(
                kind="support_set",
                payload={
                    "context_group_hash": "same",
                    "query_group_hash": "same",
                    "feature_names": ["x"],
                    "label_names": ["y"],
                },
            )
        with self.assertRaisesRegex(ValueError, "target-derived"):
            ConditioningSpec(
                kind="masked_volume",
                payload={
                    "axis_order": ["K", "J", "I"],
                    "spacing": [1.0, 1.0, 1.0],
                    "active_mask_semantics": "true=score",
                    "observation_visibility": "ECLIPSE_PORO_target_values",
                    "mode": "strict",
                },
            )

    def test_visibility_rejects_query_target_and_frozen_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "query target"):
            VisibilityPolicy(
                allowed_fields=("x",),
                forbidden_fields=("y",),
                query_target_visible=True,
            )
        with self.assertRaisesRegex(ValueError, "frozen-test"):
            VisibilityPolicy(
                allowed_fields=("x",),
                forbidden_fields=("y",),
                frozen_test_accessed=True,
            )

    def test_state_machine_forbids_unverified_to_holdout(self) -> None:
        request = time_envelope()
        with self.assertRaisesRegex(ValueError, "invalid foundation state transition"):
            request.transition("CONFIRMED_HOLDOUT")
        promoted = request.transition("PROMOTED_DEV")
        confirmed = promoted.transition("CONFIRMED_HOLDOUT")
        self.assertEqual(confirmed.state, "CONFIRMED_HOLDOUT")

    def test_local_artifact_check_is_fail_loud(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                model_ref().verify_local_artifact(path)

    def test_import_guard_rejects_prefix_collision(self) -> None:
        from _models.gaia_dagt.foundation_runtime import insert_import_root

        with tempfile.TemporaryDirectory() as directory:
            approved = Path(directory) / "source"
            collision = Path(directory) / "source_unapproved" / "package.py"
            approved.mkdir()
            collision.parent.mkdir()
            collision.write_text("", encoding="utf-8")
            package_name = "_p8_prefix_collision_package"
            module = types.ModuleType(package_name)
            module.__file__ = str(collision)
            sys.modules[package_name] = module
            try:
                with self.assertRaisesRegex(RuntimeError, "unapproved source"):
                    insert_import_root(approved, package_name)
            finally:
                sys.modules.pop(package_name, None)

    def test_supervisory_prompt_is_versioned_deterministic_and_qc_only(self) -> None:
        request = time_envelope()
        prompt = build_supervisory_prompt(
            request,
            {"input_rows": 90, "regular_frequency": True, "runtime_seconds": 2.5},
            agent_model="approved/qc-model",
            agent_revision="2026-07-28",
        )
        rebuilt = build_supervisory_prompt(
            request,
            {"input_rows": 90, "regular_frequency": True, "runtime_seconds": 2.5},
            agent_model="approved/qc-model",
            agent_revision="2026-07-28",
        )
        self.assertEqual(prompt.prompt_hash(), rebuilt.prompt_hash())
        self.assertEqual(prompt.temperature, 0)
        self.assertIn("same-split", prompt.system)
        self.assertIn("Return JSON", prompt.system)
        self.assertIn('"target_schema_names_only"', prompt.user)
        self.assertNotIn('"target_values"', prompt.user)

    def test_supervisory_prompt_rejects_labels_paths_and_secrets(self) -> None:
        request = time_envelope()
        for unsafe in (
            {"validation_labels": [1, 0]},
            {"checkpoint_path": "/tmp/model.bin"},
            {"api_key": "secret"},
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                build_supervisory_prompt(
                    request,
                    unsafe,
                    agent_model="approved/qc-model",
                    agent_revision="2026-07-28",
                )

    def test_supervisory_client_boundary_sends_locked_prompt_and_validates_json(self) -> None:
        prompt = build_supervisory_prompt(
            time_envelope(),
            {"input_rows": 90, "regular_frequency": True},
            agent_model="approved/qc-model",
            agent_revision="2026-07-28",
        )

        class StubClient:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def complete(self, **kwargs: object) -> str:
                self.kwargs = kwargs
                return json.dumps(
                    {
                        "status": "pass",
                        "checks": [
                            {"name": "split", "status": "pass", "evidence": "isolated"}
                        ],
                        "warnings": [],
                        "recommended_actions": [],
                    }
                )

        client = StubClient()
        response = invoke_supervisory_prompt(prompt, client=client)
        self.assertEqual(response["status"], "pass")
        self.assertEqual(client.kwargs["model"], "approved/qc-model")
        self.assertEqual(client.kwargs["temperature"], 0)

    def test_supervisory_client_boundary_rejects_extra_or_malformed_fields(self) -> None:
        prompt = build_supervisory_prompt(
            time_envelope(),
            {"input_rows": 90},
            agent_model="approved/qc-model",
            agent_revision="2026-07-28",
        )

        class BadClient:
            def complete(self, **_: object) -> dict[str, object]:
                return {
                    "status": "pass",
                    "checks": [],
                    "warnings": [],
                    "recommended_actions": [],
                    "raw_predictions": [],
                }

        with self.assertRaisesRegex(ValueError, "fields must be exactly"):
            invoke_supervisory_prompt(prompt, client=BadClient())


class FoundationRouteTests(unittest.TestCase):
    def test_every_track_has_one_real_modality_appropriate_route(self) -> None:
        routes = load_foundation_routes()
        self.assertEqual(
            set(routes),
            {"fault", "facies", "property", "lithofacies", "sweetspot", "reconstruction"},
        )
        expected = {
            "fault": ("segmentation_3d", "spatial_prompt"),
            "facies": ("segmentation_2d", "spatial_prompt"),
            "property": ("tabular_regression", "support_set"),
            "lithofacies": ("depth_classification", "depth_window"),
            "sweetspot": ("time_forecasting", "time_window"),
            "reconstruction": ("volume_regression_3d", "masked_volume"),
        }
        for track_id, route in routes.items():
            with self.subTest(track_id=track_id):
                self.assertEqual(
                    (route["task_type"], route["conditioning_kind"]), expected[track_id]
                )
                model = FoundationModelRef.from_dict(route["model"])
                self.assertEqual(len(model.weights_sha256), 64)
                self.assertGreater(model.weights_size_bytes, 0)
                self.assertNotIn("placeholder", model.family)
                self.assertFalse(route["default_enabled"])

    def test_adapter_modules_import_without_loading_weights(self) -> None:
        routes = load_foundation_routes()
        expected_module_ids = {
            "fault": "sam_med3d_semantic",
            "facies": "sam2_semantic",
            "property": "tabiclv2_regressor",
            "lithofacies": "moment_depth",
            "sweetspot": "p7_chronos2",
            "reconstruction": "openmind_mae",
        }
        for track_id, expected_model_id in expected_module_ids.items():
            with self.subTest(track_id=track_id):
                module = importlib.import_module(str(routes[track_id]["adapter"]))
                self.assertEqual(module.model_id, expected_model_id)
                self.assertTrue(module.capabilities()["requires_pretrained_weight"])
                self.assertFalse(module.capabilities()["auto_download"])
                discovered = discover_model(track_id, expected_model_id)
                self.assertEqual(discovered.model_id, expected_model_id)

    def test_runtime_smoke_evidence_matches_every_locked_route(self) -> None:
        evidence_path = Path(__file__).parents[1] / "foundation_runtime_smoke.v1.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        routes = load_foundation_routes()
        self.assertTrue(evidence["policy"]["connection_is_not_promotion"])
        self.assertFalse(evidence["policy"]["frozen_test_accessed"])
        self.assertEqual(set(evidence["tracks"]), set(routes))
        for track_id, route in routes.items():
            with self.subTest(track_id=track_id):
                record = evidence["tracks"][track_id]
                self.assertEqual(record["adapter"], route["adapter"])
                self.assertEqual(
                    record["checkpoint_sha256"], route["model"]["weights_sha256"]
                )
                self.assertEqual(
                    record["source_revision"], route["model"]["source_revision"]
                )
                self.assertTrue(record["finite"])
                self.assertFalse(route["default_enabled"])

    def test_openmind_small_volume_is_safely_padded_then_cropped(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch runtime is unavailable")
        from _models.reconstruction.openmind_mae import _make_network

        class FakeNetwork(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen_shape: tuple[int, ...] = ()

            def forward(self, value: object) -> object:
                self.seen_shape = tuple(value.shape)
                return value

        fake = FakeNetwork()
        model = _make_network(torch, fake, freeze_encoder=False)
        output = model(torch.zeros(1, 3, 32, 40, 48))
        self.assertEqual(fake.seen_shape, (1, 1, 64, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 1, 32, 40, 48))

    def test_moment_default_mask_is_interpolated_as_float(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch runtime is unavailable")
        from _models.lithofacies.moment_depth import _make_network

        class Output:
            logits = torch.zeros(2, 9)

        class FakePipeline:
            def __init__(self) -> None:
                self.mask_dtype = None
                self.mask_shape: tuple[int, ...] = ()

            def __call__(self, *, x_enc: object, input_mask: object) -> Output:
                self.mask_dtype = input_mask.dtype
                self.mask_shape = tuple(input_mask.shape)
                return Output()

        fake = FakePipeline()
        output = _make_network(torch, fake)(torch.zeros(2, 35, 33))
        self.assertEqual(tuple(output.shape), (2, 9))
        self.assertEqual(fake.mask_dtype, torch.long)
        self.assertEqual(fake.mask_shape, (2, 512))


if __name__ == "__main__":
    unittest.main()
