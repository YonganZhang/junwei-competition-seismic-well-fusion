# Test Run Ledger - 军伟的比赛

> Historical run evidence. Do not rewrite old command contexts to look cleaner; append new runs with cwd and exact command shape.

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
