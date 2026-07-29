# Test Run Ledger - <PROJECT_NAME>

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
