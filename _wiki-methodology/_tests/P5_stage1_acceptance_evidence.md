# P5 open-model Stage-1 acceptance evidence

> Date: 2026-07-14
> Scope: isolated branch `p5-model-benchmark-integration`; no merge to `master`, no push
> Interpretation: every frozen first-batch candidate has an auditable contract attempt. This does not mean 60 successful trainings or a performance ranking.

## Three evidence layers

| Evidence layer | Result | Boundary |
|---|---|---|
| Command gate | `torch-common` full suite: 53 passed, 6 skipped, 77 subtests passed. `tabular-cpu`: 31 passed, 2 skipped, 20 subtests passed. | These are cross-track unit/adapter/contract checks. Structured skips remain skips. |
| Live/user journey | Six Stage-1 runners were exercised against real development inputs where an approved task contract exists; fixture-only adapter checks were used only as engineering evidence. | No approved real `label_spec` exists for the seven sweetspot targets in this Stage-1 runner. Fault lacks audited negatives. Neither track may publish a ranking. |
| Trace/SSDO audit | Source locks, adapter registries, structured status/reason records, seed/environment evidence and per-track summaries were read back. | There is no online trace service for these offline jobs; persisted result JSON plus immutable commits are the declared SSDO downgrade. |

## Integrated commits

| Order | Track | Integrated commit |
|---:|---|---|
| 1 | fault | `53daaa7` |
| 2 | facies | `26c4250` |
| 3 | property | `6ebea6f` |
| 4 | lithofacies | `0ce3bf1` |
| 5 | sweetspot | `2559b7b` |
| 6 | reconstruction | `fabe99b` |

The six branches were integrated from the clean P4 ancestor in deterministic track order. The dirty `master`
working tree was not modified. A historical property worktree contains an abandoned, uncommitted first attempt and
is explicitly excluded from integration; only the clean P5 property commit above is accepted.

## Stage-1 result matrix

| Track | Audited result | Test firewall / stop reason |
|---|---|---|
| fault | Ten frozen candidates received an engineering contract attempt; ready adapters completed development forward/contract checks. | Formal pilot/ranking stops because no audited negative mask exists. `frozen_test_accessed=false`. |
| facies | F3: 6 smoked + 4 structured skips. Penobscot: 6 smoked + 4 structured skips. | Datasets retain separate heads, label spaces and boards. No test archive, labels or metrics were accessed. |
| property | 9 smoked + 1 structured skip; the MONAI 3D adapter passed an independent deterministic GPU replay. | TabICLv2 stopped at the unapproved checkpoint-license gate. PHIF/KLOGH/SW masks stay independent. |
| lithofacies | P lane: 9 passes. S lane: 1 structured skip in the fixed real batch. | The S-lane batch lacked a continuous same-well MD sequence; results were not moved across lanes. No test access. |
| sweetspot | 10 adapters passed fixture contract checks; all 10×7 real task cells emitted a structured label-gated skip. | All seven `label_spec` records are missing/unapproved for this runner. No label was generated and no frozen test was accessed. |
| reconstruction | strict: 8 passes + 2 skips; conditional: 8 passes + 2 skips. | Task modes and boards stay separate; frozen-test block list is empty. |

## Cross-track integration fix

Independent suites originally reused bare module names such as `p5_stage1`/`p4_contract` and several identical
`test_p5_stage1.py` basenames. They passed alone but collided during one-process pytest collection. The integration
layer now uses track-unique test basenames and explicit file-path imports, while preserving the track implementations.

## Decision for Stage 2

- Proceed only with candidates that have approved development labels, source/license clearance and a real input lane.
- Keep fault at the data-gate audit until audited negatives exist.
- Keep sweetspot at label-contract work until each intended target has an approved real `label_spec`.
- Do not open frozen tests. Stage 2 is a fixed-budget development comparison, not final evaluation.
