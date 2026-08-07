"""Declarative, fail-closed adapter for the sweetspot pipeline."""

ADAPTER = {
    "schema_version": "six_track_adapter/v1",
    "track": "sweetspot",
    "task_dir": "_pipelines/02_task_datasets/sweetspot",
    "manifest": "_pipelines/sweetspot_agentic_optimization.yml",
    "stage_order": ["validate", "prepare", "baseline", "optimize", "promote", "refit", "verify"],
    "stages": {
        "validate": {
            "id": "validate",
            "needs": [],
            "execution": "evidence",
            "entrypoint": "_pipelines/02_task_datasets/track_lifecycle.py",
            "argv": ["{python}", "{project_root}/_pipelines/02_task_datasets/track_lifecycle.py", "--track", "sweetspot", "--stage", "validate"],
            "required_parameters": [],
            "required_inputs": ["_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json"],
            "expected_outputs": [],
            "description": "Validate the archived action-effect evidence without claiming that a canonical sweetspot truth dataset exists.",
        },
        "prepare": {
            "id": "prepare",
            "needs": ["validate"],
            "execution": "manual",
            "entrypoint": "_pipelines/02_task_datasets/sweetspot/build_dataset.py",
            "argv": [
                "{python}",
                "{project_root}/_pipelines/02_task_datasets/sweetspot/build_dataset.py",
                "--mode", "validate-only",
                "--spec", "{approved_label_spec}",
            ],
            "required_parameters": ["approved_label_spec"],
            "required_inputs": [
                "{approved_label_spec}",
                "_pipelines/02_task_datasets/sweetspot/label_spec.schema.v1.json",
                "_sandbox/volve_data",
            ],
            "expected_outputs": [
                "_pipelines/02_task_datasets/sweetspot/audit/data_availability.json",
                "_pipelines/02_task_datasets/sweetspot/audit/data_readiness.md",
                "_pipelines/02_task_datasets/sweetspot/audit/contract_validation.json",
            ],
            "description": "Validate a domain-approved label contract against real fields; the current entrypoint deliberately cannot write train/test HDF5.",
            "block_reason": "No approved sweetspot label specification or authorized dataset-building implementation exists. build_dataset.py supports audit/validate-only and must not be mistaken for preprocessing.",
        },
        "baseline": {
            "id": "baseline",
            "needs": ["prepare"],
            "execution": "manual",
            "entrypoint": "_pipelines/02_task_datasets/sweetspot/sweetspot_incumbent.py",
            "argv": [
                "{python}",
                "{project_root}/_pipelines/02_task_datasets/sweetspot/sweetspot_incumbent.py",
            ],
            "required_parameters": [],
            "required_inputs": [
                "_pipelines/02_task_datasets/sweetspot/audit/contract_validation.json",
                "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards/T3.json",
                "_pipelines/02_task_datasets/sweetspot/p7/_outputs/t3_chronos2_cv/summary.json",
                "_pipelines/02_task_datasets/sweetspot/p8/_outputs/t3_chronos2_calendar_cv/summary.json",
            ],
            "expected_outputs": [
                "_pipelines/02_task_datasets/sweetspot/_outputs/incumbent/incumbent.json",
            ],
            "description": (
                "Resolve the current incumbent across both baseline layers and record it in a single file. "
                "Small-model layer: P5 Stage-3 multi-model CV (LightGBM/XGBoost/CatBoost/InceptionTime) over "
                "the rankable targets. Foundation layer: P7/P8 Chronos-2, which formally PROMOTEd on T3 with "
                "macro-fold MAE 186.572 against the archived XGBoost 267.118 (-30.15%) and the causal "
                "history-mean control 204.637 (-8.83%). incumbent.json also carries rejected_routes and "
                "open_work so a fresh session does not restart a refuted path."
            ),
            "block_reason": (
                "Retraining either baseline layer from scratch still requires an approved label contract, so "
                "this stage stays manual. The runner itself only reads archived evidence and trains nothing."
            ),
            "layers": {
                "small_model": "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards",
                "foundation_model": "_pipelines/02_task_datasets/sweetspot/p7/_outputs/t3_chronos2_cv/summary.json",
            },
            # The stage entrypoint resolves incumbent identity, which is this stage's contractual
            # responsibility. Actually retraining the provisional per-target baselines is a separate,
            # still-gated action; keeping the pointer here so the trainer is not lost from the adapter.
            "training_entrypoint": {
                "path": "_pipelines/02_task_datasets/sweetspot/targets/baseline.py",
                "argv": [
                    "{python}",
                    "-m", "_pipelines.02_task_datasets.sweetspot.targets.baseline",
                    "--target", "all",
                    "--source-root", "{source_root}",
                ],
                "required_parameters": ["source_root"],
                "outputs": [
                    "_pipelines/02_task_datasets/sweetspot/targets/reservoir_quality/_outputs/baseline_v1/status.json",
                    "_pipelines/02_task_datasets/sweetspot/targets/hydrocarbon_pay/_outputs/baseline_v1/status.json",
                    "_pipelines/02_task_datasets/sweetspot/targets/productivity/_outputs/baseline_v1/status.json",
                    "_pipelines/02_task_datasets/sweetspot/targets/water_breakthrough/_outputs/baseline_v1/status.json",
                ],
                "gated_by": "No approved canonical label contract; running it would bypass the label gate.",
            },
        },
        "optimize": {
            "id": "optimize",
            "needs": ["baseline"],
            "execution": "manual",
            "entrypoint": "_pipelines/02_task_datasets/sweetspot/p29_agent_action_effect.py",
            "argv": ["{python}", "{project_root}/_pipelines/02_task_datasets/sweetspot/p29_agent_action_effect.py"],
            "required_parameters": [],
            "required_inputs": [
                "_pipelines/02_task_datasets/sweetspot/p5/sweetspot_p5_label_mapping.v1.json",
                "_pipelines/02_task_datasets/sweetspot/targets/productivity/_outputs/baseline_v1/split_manifest.json",
                "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards/T3.json",
                "_pipelines/02_task_datasets/sweetspot/_outputs/incumbent/incumbent.json",
            ],
            "expected_outputs": [
                "_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json",
                "_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/action_effects.json",
                "_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/manifest.json",
            ],
            "description": (
                "Third layer: the DeepSeek action selector proposes one allowlisted T3 action, a deterministic "
                "executor trains it for real, and an independent promotion fold decides. Both archived runs "
                "returned a legal stop and REJECT_AGENT."
            ),
            "block_reason": (
                "The agent's A0 is still the archived XGBoost (MAE 267.118), which the foundation layer "
                "superseded at 186.572. Rerunning would measure candidates against a stale frontier, so the "
                "stage stays manual until A0 is switched to the incumbent recorded in incumbent.json. "
                "The canonical label gate also remains unapproved. (The earlier portability blocker — evidence "
                "resolved through an implicit p10-results-sweetspot worktree — was fixed in c9a85f4.)"
            ),
            "agent": {
                "enabled": True,
                "mode": "action_selector",
                "role": "Select one allowlisted T3 XGBoost action or request a legal stop from signed normalized development feedback.",
                "decision_owner": "Deterministic allowlist validation and promotion comparison against same-fold A0 and independent A3",
                "candidate_source": "Frozen T3 action registry plus a live DeepSeek selected_action_id",
                "promotion_guard": "Selection/promotion fold disjointness and lower promotion MAE than both A0 and independent random control",
                "fallback": "DATA_GATE_BLOCKED or REJECT_AGENT; retain frozen A0 XGBoost",
                "known_misconfiguration": (
                    "A0 points at the superseded XGBoost rather than the current incumbent; see open_work in "
                    "incumbent.json. Archived REJECT_AGENT verdicts must be read with this in mind."
                ),
            },
        },
        "promote": {
            "id": "promote",
            "needs": ["optimize"],
            "execution": "included",
            "entrypoint": "_pipelines/02_task_datasets/sweetspot/p29_agent_action_effect.py",
            "argv": [],
            "required_parameters": [],
            "required_inputs": ["_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json"],
            "expected_outputs": ["_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json"],
            "description": "The P29 composite run evaluates the selected action on the promotion fold and records RETAIN_AGENT/REJECT_AGENT/DATA_GATE_BLOCKED.",
            "included_in": "optimize",
        },
        "refit": {
            "id": "refit",
            "needs": ["promote"],
            "execution": "included",
            "entrypoint": "_pipelines/02_task_datasets/sweetspot/p29_agent_action_effect.py",
            "argv": [],
            "required_parameters": [],
            "required_inputs": ["_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json"],
            "expected_outputs": ["_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json"],
            "description": "The current decision retains frozen A0; no rejected agent candidate is refit and the no-op retention is recorded by the composite optimizer.",
            "included_in": "optimize",
        },
        "verify": {
            "id": "verify",
            "needs": ["refit"],
            "execution": "evidence",
            "entrypoint": "_pipelines/02_task_datasets/track_lifecycle.py",
            "argv": ["{python}", "{project_root}/_pipelines/02_task_datasets/track_lifecycle.py", "--track", "sweetspot", "--stage", "verify"],
            "required_parameters": [],
            "required_inputs": [
                "_pipelines/02_task_datasets/sweetspot/_outputs/incumbent/incumbent.json",
                "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards/T3.json",
                "_pipelines/02_task_datasets/sweetspot/p7/_outputs/t3_chronos2_cv/summary.json",
                "_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json",
                "_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/action_effects.json",
            ],
            "expected_outputs": [],
            "description": (
                "Verify all three layers together: the small-model leaderboard, the promoted Chronos-2 "
                "foundation result, and the agent's legal stop / rejection with frozen-A0 retention. Before "
                "this stage listed the foundation evidence it could pass green while the track's best result "
                "sat outside the pipeline."
            ),
        },
    },
}
