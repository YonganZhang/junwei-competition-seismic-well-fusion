# Fault P28 agentic ablation preflight

- Generated at: 2026-08-01T07:49:58.356989+00:00
- Source commit: `28a328f9ea10bc4999f124121ad446f7a04595e1`
- Trial budget: 4
- Selection scenarios: `observed_blocked_current, packet_hash_missing`
- Promotion scenarios: `packet_hash_mismatch, counterfactual_contract_green`
- Action registry: `STOP_DATA_GATE, REQUEST_EVIDENCE, VERIFY_HASHES, PROCEED`

## Contract evidence bundle

- `fault_p18_cigbench.py`: `_pipelines/02_task_datasets/fault/fault_p18_cigbench.py` sha256=`1f2ff3bb6d727f22340e37baa9816a2f244841f2fb7453a22bce232eda496aa7`
- `test_fault_p18_cigbench.py`: `_pipelines/02_task_datasets/fault/test_fault_p18_cigbench.py` sha256=`53a47406fb8f8b2ea7f5a67e0246c42682f31bb42294d4f561e86ad5302d1c7f`
- `tests/test_fault_p28_agentic_ablation.py`: `_pipelines/02_task_datasets/fault/tests/test_fault_p28_agentic_ablation.py` sha256=`500c3cfd9637d8e157b652ac2ea622a06c64694a064cfb2d2a1d07525d1c633a`
- `baseline_metrics.json`: `_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json` sha256=`0d0e3093da01eee7203c7afcd9b49667c25ebef49b40f6e7ef554c423768c631`
- `build_summary.json`: `_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/build_summary.json` sha256=`decd62f85296667e816d25a87ecb103a5385de8001557dac82c3c99ab189df64`
- `p18_evidence.md`: `_pipelines/02_task_datasets/fault/_outputs/p18_cigbench_fault/evidence.md` sha256=`b7057d3b24671fd24df667dfaddca43e6af2b5f5aaacb70400cc839fc1887133`
- `historical_baseline_manifest.json`: `_pipelines/02_task_datasets/fault/historical_baseline_manifest.json` sha256=`c2524a2a80d3debf424e9b9933f143082641ad90803e880580e5e41368ef439c`

## Scenario design

| scenario | split | truth | packet hash state | notes |
| --- | --- | --- | --- | --- |
| `observed_blocked_current` | selection | `STOP_DATA_GATE` | `verified` | Directly derived from the current P18 blocked evidence. |
| `packet_hash_missing` | selection | `REQUEST_EVIDENCE` | `unknown` | The packet omits two hash records; the agent should request them, not pretend the packet is complete. |
| `packet_hash_mismatch` | promotion | `VERIFY_HASHES` | `mismatch` | The reported hash for baseline_metrics.json disagrees with the verified artifact hash. |
| `counterfactual_contract_green` | promotion | `PROCEED` | `verified` | Protocol fixture only; it exercises the proceed branch without opening frozen test or claiming current data are green. |

## Policy results

| policy | selection accuracy | selection evidence F1 | promotion accuracy | promotion evidence F1 | dangerous false release | blocked provider |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_static_baseline` | 0.500 | 0.571 | 0.500 | 0.000 | 0.000 | 0 |
| `A1_advice_only` | 0.500 | 0.571 | 0.500 | 0.000 | 0.000 | 0 |
| `A2L_llm_agent_execute` | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| `A2D_deterministic_agent` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| `A3_random_policy` | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0 |

## Retain / reject

- Retain: `A2D_deterministic_agent`
- Reject: `A2L_llm_agent_execute`
- A2L blocked provider: `False`
- Frozen test accessed: `False`

## Notes

- The real formal fault training lane remains DATA_GATE_BLOCKED.
- `counterfactual_contract_green` is a protocol fixture that exercises the proceed branch without opening frozen test or claiming current data are green.
- A1 keeps its final decision hash identical to A0 because the hash excludes advice-only text.
