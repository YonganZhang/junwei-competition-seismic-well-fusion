# Fault P29 action-effect repair

- Generated at: 2026-08-01T13:10:12.717829+00:00
- Source commit: `76c4464d01f6a4372fc1b48b75e4765f66c6b2ee`
- data_gate_blocked: `True`
- selection/promotion intersection: `[]`

## Root cause

P28 proved the gate controller could stop or proceed, but it did not prove that PROCEED actually invoked a registered predictor exactly once or that all other gate actions were no-ops.

## Repairs

- Evidence tokens are now validated against a per-scenario allowlist.
- stop_requested is validated against the selected action.
- decision hashes use a canonical decision payload and are replay-checked for A1.
- PROCEED dispatches exactly one registered predictor; all other gate actions dispatch none.

## Sources

- `fault_p28_agentic_ablation.py`: `_pipelines/02_task_datasets/fault/fault_p28_agentic_ablation.py` sha256=`41756a72e9db7c83d157753272656f07adad304fa1cc79c31383cc65e3b1291c`
- `tests/test_fault_p29_agent_action_effect.py`: `_pipelines/02_task_datasets/fault/tests/test_fault_p29_agent_action_effect.py` sha256=`cf849bf932cec9f5f15fa65e155558976180c3d2e180b40df87affc44ec07305`
- `p28_summary.json`: `_pipelines/02_task_datasets/fault/_outputs/p28_agentic_ablation/summary.json` sha256=`ff72a0cb53c2c3e00f901018f2eaeba8ce092b9044ed75a15cf885f22f6734cd`
- `p28_manifest.json`: `_pipelines/02_task_datasets/fault/_outputs/p28_agentic_ablation/manifest.json` sha256=`8d977285ec2364b91d2c6cc60ef80a5bae89dcb613b1c003437d39bde0dad6dd`
- `p18_evidence.md`: `_pipelines/02_task_datasets/fault/_outputs/p18_cigbench_fault/evidence.md` sha256=`b7057d3b24671fd24df667dfaddca43e6af2b5f5aaacb70400cc839fc1887133`
- `baseline_metrics.json`: `_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json` sha256=`0d0e3093da01eee7203c7afcd9b49667c25ebef49b40f6e7ef554c423768c631`
- `build_summary.json`: `_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/build_summary.json` sha256=`decd62f85296667e816d25a87ecb103a5385de8001557dac82c3c99ab189df64`
- `historical_baseline_manifest.json`: `_pipelines/02_task_datasets/fault/historical_baseline_manifest.json` sha256=`c2524a2a80d3debf424e9b9933f143082641ad90803e880580e5e41368ef439c`
