# Test Run Ledger - 军伟的比赛

> Historical run evidence. Do not rewrite old command contexts to look cleaner; append new runs with cwd and exact command shape.

## 2026-07-25 — six-track domain-visualization delivery v1

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5`
- command: `python3 -m unittest discover -s _pipelines/03_domain_visualization_delivery/tests -p 'test_*.py' -v`
- result: PASS, 3/3 tests (reject status/protocol/placeholder-like paths; validate the exact six live figures; preserve hashes during staging).
- validation: `step_01_validate_manifest.py` passed all six source/provenance/hash/human-review gates.
- staging: `step_02_stage_delivery.py` copied all six figures to `_outputs/domain_visualization_delivery/v1/cards/` with unchanged SHA-256.
- publication: `step_03_publish_cards.py --yes-public` published six permanent `share.yongan.site` URLs; every URL returned HTTP 200.

## 2026-07-30 — P12 visualization standardization for tracks 1 / 3 / 5

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5`
- scope: fault / property / sweetspot only; facies / lithofacies / reconstruction remained paused.
- discovery: `python3 _pipelines/03_domain_visualization_delivery/step_00_discover.py --check` → all three `ready`.
- central tests: `python3 -m unittest discover -s _pipelines/03_domain_visualization_delivery/tests -p 'test_*.py' -v` → PASS, 7/7.
- track tests: fault 5/5, property 2/2, sweetspot 5/5; deterministic rendering and manifest hash checks passed.
- visual QA: 13 PNGs opened at original resolution; clipping, overlap, labels, units, split scope and scientific caveats checked.
- staging: `step_04_stage_p12_review.py --reviewer codex-leader --accept-visual-qa` copied 39 PNG/PDF/SVG files with unchanged SHA-256.
- evidence: `_outputs/domain_visualization_delivery/p12/review_attestation.json`; source heads fault `5d22c9a`, property `c914bd6`, sweetspot `39e0e97`.

## 2026-07-13 P4 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p4-training-integration`
- branch: `p4-training-integration`
- root seed: `2693`
- portable suites: 按 `_gates.yml` 的 `p4-shared` 至 `p4-reconstruction` 逐一执行；可选真实产物缺失时为可审计 skip，非静默通过。
- explicit real smoke: fault coordinate/index preflight、F3/Penobscot 各 1 CPU epoch、property P4 integration、lithofacies HDF5、reconstruction HDF5 split/preprocess 均 exit 0。
- facies campaigns: `p4_workflow.py` 在 `_tmp/p4-acceptance/facies_f3` 与 `facies_penobscot` 完成 `prepare -> smoke -> cv -> freeze -> refit -> test -> visualize`。
- reconstruction campaigns: `p4_reconstruction.py` 在 `_tmp/p4-acceptance/reconstruction_strict_v2` 与 `reconstruction_conditional_v2` 完成 `prepare -> cv -> refit -> test`；`p4_visualize.py` 从归档 predictions/metrics 生成 diagnostics。
- lithofacies campaign: `p4_runner.py freeze/refit/test --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/integration_cv_smoke --dataset-root <track-lithofacies real HDF5>`；`visualize_p4.py` 读归档预测生成分类/校准图。
- artifact audit: 7 manifests / 144 manifest artifacts verified；seven-target registry 7 cases / 50 registry artifacts verified；`all_targets_independent=true`。
- 结果与 blocker: `P4_acceptance_evidence.md`。

## 2026-07-14 P5 Stage-1 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-model-benchmark-integration`
- branch: `p5-model-benchmark-integration`
- root seed: `2693`
- source branches: six clean P5 track commits, integrated in fault/facies/property/lithofacies/sweetspot/reconstruction order.
- `torch-common` full combined gate: 53 passed, 6 skipped, 77 subtests passed.
- `tabular-cpu` combined gate: 31 passed, 2 skipped, 20 subtests passed.
- live development inputs: used where task labels/contracts were approved; fault formal comparison and all sweetspot real task cells stopped fail-closed at their scientific gates.
- frozen test: not accessed by Stage-1 runners.
- results, structured skips and SSDO downgrade: `P5_stage1_acceptance_evidence.md`.

## 2026-07-14 P5 Stage-2 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-model-benchmark-integration`
- branch: `p5-model-benchmark-integration`
- integrated HEAD: `d46a7b5`
- root seed: `2693`
- source commits: fault `53db563`, facies `3c6c3b8`, property `cab82e5`, lithofacies `5302954`, sweetspot `ad5fde8`, reconstruction `15c4ae1`; cherry-picked in fixed track order.
- Stage-2 matrix: 140 preregistered cells; 53 development pilots, 87 structured skip/blocked, 0 failed, 0 timeout.
- `torch-common` Stage-2 combined gate: 59 passed, 22 subtests passed.
- `tabular-cpu` applicable four-track gate: 40 passed. Reconstruction is intentionally excluded from this environment because it lacks `h5py`; its 12 Stage-2 tests passed in `torch-common`.
- Stage-1 regression after integration: 53 passed, 6 skipped, 77 subtests passed.
- frozen test: not accessed by any Stage-2 runner; all leaderboards are fixed-fold development evidence only.
- results and scientific stop lines: `P5_stage2_acceptance_evidence.md`.

## 2026-07-14 P5 Stage-3 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-model-benchmark-integration`
- branch: `p5-model-benchmark-integration`; integrated result HEAD before documentation: `9e5f501`.
- protocol commit: `16bebd1`; root seed `2693`; repeat seeds `1867973658`, `2137841944`, `3902865753`.
- development matrix: five runnable tracks, 441 attempted cells; 437 pass, 3 fail, 1 timeout. Fault had zero scientifically legal training cells and remained `not_rankable`.
- independent full regressions: fault `41 passed` plus `15 passed/7 skipped/2 subtests`; facies `47 passed/1 skipped/43 subtests`; property `37 passed/5 skipped`; lithofacies P5 `25 passed/1 skipped/26 subtests`, data contract `8 passed/4 skipped`, mixed legacy `5 passed/6 skipped/3 subtests`; sweetspot data `21 passed`, models `51 passed/1 skipped`; reconstruction `55 passed/10 skipped/53 subtests`.
- dependency policy: `torch-common` and `tabular-cpu` were not modified. The legacy lithofacies module that imports Torch plus HDF5/LAS/SEG-Y readers was rerun in a one-shot `uv run --isolated` group.
- frozen test: no Stage-3 runner opened frozen arrays, labels, predictions or historical metrics.
- results, winners, figures and boundaries: `P5_stage3_acceptance_evidence.md`.

## 2026-07-28 P8 multimodal foundation-model routing acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p8-multimodal-foundation`
- branch: `p8-multimodal-foundation`; base HEAD `1bb0595`.
- six routes: fault/SAM-Med3D, facies/SAM 2.1 + semantic head, property/TabICLv2, lithofacies/MOMENT depth windows, sweetspot/Chronos-2 calendar windows, reconstruction/OpenMind ResEnc-L 3D MAE.
- source and weights: every route pinned to an exact source commit and weight revision; all six local checkpoint sizes and SHA-256 values matched `foundation_routes.v1.json`.
- real runtime evidence: all six adapters loaded their real pinned weights and returned finite outputs. Fault evidence remains synthetic-only because audited negatives and contiguous 3D development blocks are unavailable.
- bugs caught by real forward passes: MOMENT default mask required float interpolation before restoring integer dtype; OpenMind ResEnc-L requires padding a 32-voxel input to at least 64 voxels before its fifth downsample. Both have regression coverage.
- combined portable regression: `python3 -m pytest _code/ml_framework/tests _models/gaia_dagt/tests _pipelines/02_task_datasets/sweetspot/tests -q` -> `197 passed, 7 skipped, 24 subtests passed`.
- explicit Torch adapter regression: `PYTHONPATH=. /mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python -m pytest _models/gaia_dagt/tests/test_foundation_contract.py -q` -> `18 passed, 24 subtests passed`.
- property source-lock regression: `python3 -m pytest _pipelines/02_task_datasets/reservoir/tests/test_p5_stage1_contracts.py -q` -> `6 passed, 1 skipped`.
- Chronos-2 real development run: 4 group-isolated folds, 196 train / 84 validation rows per fold; history-mean MAE `184.6686`, Chronos MAE `172.3162`, train-only blend MAE `166.3343`. This is development evidence only, not frozen-test confirmation.
- lifecycle: all six routes remain `CONNECTED_UNVERIFIED`, `default_enabled=false`; no route may be promoted without same-split baseline, random-init same-architecture control, leakage checks and material gain.
- supervisory LLM: deterministic prompt template, provider-neutral client boundary and strict JSON response validation are tested with a stub. No external LLM API was invoked because provider/model/revision approval is absent.
- frozen test: not accessed.
- detailed evidence and blockers: `P8_multimodal_foundation_acceptance_evidence.md`.

## 2026-07-31 P17 reconstruction foundation-geostatistics acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction`
- branch: `p11-residual-reconstruction`; root seed `2693`.
- model: frozen genuine `thinkonward/geophysical-foundation-model`; cached seismic representations were reused only after source/snapshot/cache hashes were verified.
- data budget: five locked spatial folds; exactly 512 `point_train` labels and 2,048 validation rows per fold; no frozen-test path or HDF5 was opened.
- bounded search: 13 positive-foundation metric pairs × 3 neighbour counts × 4 PyKrige blends = 156 candidates.
- selected result: `gfm_metric_f0.05_s0.10_k128_blend_0.75`; pooled OOF RMSE `0.028319907650` versus PyKrige `0.028449728170`; 3 wins / 2 losses across the five folds.
- uncertainty: 20,000 whole-fold bootstrap draws; P(candidate better)=`0.7668`, RMSE-delta CI95 `[-0.000589605334, +0.000128806422]`.
- regression: P14–P17 combined suite `39 tests`, `OK`; P17 portable suite `8 tests`, `OK`.
- independent verifier: `p17_foundation_geostatistics.py verify` -> `PASSED`; 10,240 rows, per-fold metrics, prediction hash and summary hash matched.
- lifecycle: `DEVELOPMENT_SIGNAL`, `default_enabled=false`; ablation deferred by user instruction; frozen holdout remains sealed.
- detailed evidence: `P17_reconstruction_foundation_acceptance_evidence.md`.

## 2026-07-31 P18 reconstruction anisotropic foundation-geostatistics acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction`
- branch: `p11-residual-reconstruction`; root seed `2693`.
- external review: Claude completed a read-only audit and identified same-OOF winner's curse plus missing geological anisotropy; the accepted review is `_top/_external_reviews/P18_claude_p17_metric_review_20260731.md`.
- P17 correction: the original 156-candidate family under nested LOFO top-3 gives RMSE `0.028534404074`, worse than PyKrige `0.028449728170`; the old same-OOF `-0.4563%` result is superseded.
- P18 protocol: 1,215 bounded anisotropic candidates; each held-out spatial fold is predicted using top-3 candidates ranked only on the other four folds' metric rows. This did not purge the held fold's coordinates from other-fold meta-fits and was later superseded by P19. Scaling/PCA remains fit on the current outer fold's 512 legal labels.
- selected result: nested P18 RMSE `0.027752680679`, MAE `0.020830115995`, relative RMSE change `-2.4501%`, with 5 wins / 0 losses across spatial folds.
- uncertainty: 20,000 whole-fold bootstrap draws; P(candidate better)=`1.0`, RMSE-delta CI95 `[-0.001140994782, -0.000353924655]`.
- alternatives: PLS nested route RMSE `0.027811249487` was weaker; aggressive refinement RMSE `0.027732695244` had only 4/5 fold wins and was rejected.
- regression: P17+P18 combined suite `16 tests`, `OK`; `git diff --check` passed.
- independent verifier: `p18_anisotropic_foundation_geostatistics.py verify` -> `PASSED`; 10,240 rows, per-fold metrics, legacy correction, selection firewall and artifact hashes matched.
- Claude final read-only review: `PASS`, no P0/P1; independently reproduced metrics and bootstrap bit-exactly. The P2 requests for nested top-1 parity and five-fold caveat were incorporated; grid-edge sensitivity remains documented for preregistered external validation.
- firewall/lifecycle: 512 labels and 2,048 validation rows per fold; no frozen holdout path or HDF5 opened; `ROBUST_DEVELOPMENT_SIGNAL`, `default_enabled=false`, ablation deferred and no causal pretraining claim.
- detailed evidence: `P18_reconstruction_anisotropic_acceptance_evidence.md`.

## 2026-07-31 P19 reconstruction meta-purge and training-dynamics diagnosis

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction`
- branch: `p11-residual-reconstruction`; root seed `2693`.
- protocol audit: every fold's own 512 training labels remain disjoint from its 2,048 validation rows, but the current validation coordinates occurred in other-fold training subsets for folds 0--4 at `58/44/24/27/45` row occurrences. P19 removes those coordinates before every meta-selection refit.
- selected result: meta-purged nested top-3 RMSE `0.027751397628`, MAE `0.020825760745`, bias `-0.000417160127`, relative RMSE change `-2.4546%`, with 5 wins / 0 losses against PyKrige.
- uncertainty: 20,000 whole-fold bootstrap draws; P(candidate better)=`1.0`, RMSE-delta CI95 `[-0.001143968280, -0.000353924655]`.
- real-tail probe: prefix `[4,3,161,1200]`, tail `[12,161,1200]`, query `[150,12]`, output `[150]`; 17,298,000 trainable encoder parameters and 122,553 head parameters. Zero-initialized final output makes encoder gradient exactly zero at step 1; by step 3 encoder relative update is about `1.9e-5` versus head update about `9.4e-3`.
- activation/route screens: GELU, SiLU and ReLU were probed; frozen-GFM MLP, extended anisotropy, strictly nested regression kriging, and K/J/I hybrid metrics did not beat P19. Same-OOF regression-kriging RMSE `0.027720062019` was rejected because strict nesting regressed to `0.027830073069`.
- independent verifier: recomputed all 10,240 predictions, five fold metrics, purge declarations, gradient invariants and rejected-route gates; status `PASSED`.
- regression: P17+P18+P19 focused suites passed; Python compile, JSON/YAML parse and `git diff --check` passed.
- firewall/lifecycle: no frozen holdout path or `test.h5` opened; no random-init/no-foundation ablation; `L3_VALIDATED_KEEP`, `default_enabled=false`, no causal pretraining claim.
- detailed evidence: `P19_reconstruction_training_diagnostics_acceptance_evidence.md`.

## 2026-08-01 P24 historical transfer and P25 integration acceptance

- integration target: `final-integration`; merge commit `707cff2` contains the local `master` line, the cumulative P4--P24 research line and the public-repository redaction commit.
- reconstruction gate: `python3 -m unittest -v ...test_p24_historical_transfer.py ...test_p21_fixed_foundation_ensemble.py` -> 10/10 passed with system Python.
- test hygiene: P21 verification/manifest generation now runs against a temporary copied bundle, so changing Python/NumPy float formatting cannot rewrite the tracked scientific evidence during a read-only gate.
- read-only verifier: P24 recomputed RMSE `0.027825182663` versus PyKrige `0.028235410003`, relative improvement `1.4529%`, 4 wins / 1 loss, and preserved the same-field/non-blind claim boundary.
- integration portability fix: domain-visualization steps now resolve the shared Git common directory, so a linked worktree no longer constructs a nested `.claude/worktrees/<integration>/.claude/worktrees/...` path.
- P12 central gate: 6/7 tests passed. The remaining fail-closed assertion detected a real uncommitted hash drift in `track-fault` (`3d_sci_v1/provenance.json` plus its generator/test); those user-owned changes were not overwritten, staged or committed.
- lifecycle: user accepted P24 as the final transfer-evidence tier for this round; no cross-field test is scheduled. No push was performed.

## 2026-08-01 P26 master merge and report delivery

- Command gate: P12 central visualization `7/7`, P21/P24 reconstruction `11/11`, and research-visualization expansion `4/4` all passed.
- Live/user journey: the 45-page A4 PDF and 140-entry LaTeX source package were published through stable `share.yongan.site` topics; both latest URLs returned HTTP 200 with the expected content types and sizes.
- Trace/SSDO audit: full hashes, version URLs, merge-preservation sequence and scientific claim boundaries are recorded in `P26_master_merge_and_report_delivery_evidence.md`.
- integration: cumulative research line merged to `master@8375b97`; P21 remains the delivery default, P20 PEFT routes remain disabled, and no remote push was performed.
