# Test Coverage - <PROJECT_NAME>

> COL3 test map. Register reusable tests, coverage purpose, runnable entrypoints and gaps here. Long run evidence can go in `_run_ledger.md`.
> For software projects with pipeline/service/log/status/report evidence, also register the SSDO/audit-first entrypoint here or record it as a current gap.

## Audit-First / SSDO

| Gate | Entrypoint | Coverage Purpose |
|---|---|---|
| domain visualization audit | `python3 _pipelines/03_domain_visualization_delivery/step_00_discover.py --check` | 固定发现 P12 赛道1/3/5的渲染器、测试、manifest、输入输出哈希和 paused 边界；失败时不得进入人工验收 |
| legacy six-track audit | `python3 _pipelines/03_domain_visualization_delivery/step_01_validate_manifest.py --check-only` | 读取旧六赛道白名单的HEAD、哈希、脚本、证据、尺寸与人工复核状态；失败时不生成交付清单 |

## Trunk

| Gate | Entrypoint | Coverage Purpose |
|---|---|---|
| `domain-visualization-delivery` | `python3 -m unittest discover -s _pipelines/03_domain_visualization_delivery/tests -p 'test_*.py' -v` | 证明旧状态图被拒绝、六赛道白名单继续有效、P12赛道1/3/5可发现、人工验收必须显式接受且复制后哈希保持一致 |

## Branch

| Test | Covers |
|---|---|
| `_pipelines/03_domain_visualization_delivery/tests/test_delivery_pipeline.py` | 路径分类、真实六赛道清单、P12合同、PNG尺寸、独立人工验收与复制哈希 |

## Leaves

| Unit / Sample | Purpose |
|---|---|
| `_outputs/domain_visualization_delivery/v1/validation_report.json` | 结构化 anti-fake-completion 断言 |
| `_outputs/domain_visualization_delivery/v1/published_manifest.json` | 六个永久URL及HTTP 200发布证据 |
| `_outputs/domain_visualization_delivery/p12/review_attestation.json` | 赛道1/3/5的负责人逐图验收、来源/稳定副本路径与39个文件哈希 |

## Current Gaps

- 卡片公网内容本身没有视觉像素回归；当前以源图SHA-256、确定性双渲染测试与发布前人工逐张预览锁定内容。
