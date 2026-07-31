# P8 多模态基础模型接线验收证据

日期：2026-07-28

## 验收范围

- 分支/工作树：`p8-multimodal-foundation`
- 六条 foundation route、统一 conditioning envelope、监督 LLM prompt/client boundary。
- 固定 source revision、weights revision、size、SHA-256 和许可证状态。
- 无 frozen test、known holdout 或 fresh blind 访问；所有路线保持 `default_enabled=false`。

## 权重与真实运行

| 赛道 | 模型 | 真实运行 | 结果 |
|---|---|---|---|
| fault | SAM-Med3D turbo | `[1,1,24,32,40] -> [1,1,24,32,40]` | finite；synthetic-only |
| facies | SAM 2.1 Hiera B+ | `[1,1,64,64] -> [1,10,64,64]` | finite |
| property | TabICLv2 | 24 行×6 特征，三目标 fit/predict | finite |
| lithofacies | MOMENT-1-base | `[1,35,33] -> [1,9]` | finite；head 待微调 |
| sweetspot | Chronos-2 | 4 折真实日历 development CV | finite；MAE 172.3162 |
| reconstruction | OpenMind ResEnc-L MAE | `[1,3,32,32,32] -> [1,1,32,32,32]` | finite |

机读证据：`_models/gaia_dagt/foundation_runtime_smoke.v1.json`。甜点详细折级证据：
`_pipelines/02_task_datasets/sweetspot/p8/_outputs/t3_chronos2_calendar_cv/summary.json`。

## 测试门

宽范围回归命令：

`python3 -m pytest _code/ml_framework/tests _models/gaia_dagt/tests _pipelines/02_task_datasets/sweetspot/tests -q`

结果：`197 passed, 7 skipped, 24 subtests passed`。skip 均为既有可选依赖/数据门。

Torch 专项合同：

`PYTHONPATH=. /mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python -m pytest _models/gaia_dagt/tests/test_foundation_contract.py -q`

结果：`18 passed, 24 subtests passed`。

## 证据三分法

### Command gate

上述 portable、Torch 和 property source-lock 回归均 exit 0；JSON/YAML可解析、`compileall` 与
`git diff --check` 通过。

### Live/user journey

六条路线均通过真实固定权重加载和一次有限值前向；Chronos-2另外完成真实development四折日历运行。
本工作不含网页或人工交互流程，因此live证据采用“实际模型加载→条件校验→前向→机读证据落盘”的
最短模型用户旅程。

### Trace/SSDO audit

`foundation_runtime_smoke.v1.json` 保存每条路线的source/weight hash、输入输出shape、有限值状态与
科学边界；Chronos折级summary保存development指标。未部署独立trace collector，因此本轮SSDO降级为
机读运行证据、命令台账和回归测试三者互相核对。

## 运行中发现并修复

1. OpenMind 32 边长在最深层退化成单 voxel，修为至少补齐 64。
2. MOMENT 默认 Long mask 无法 nearest interpolate，修为 float 插值后转 Long。
3. HF snapshot symlink 解析到 blob 后被错误父目录检查拒绝，改由固定 SHA 验证。
4. TabICL source lock 版本与固定 commit 不符，修为真实 `2.1.1` 并验证三目标 fit/predict。
5. 四个新模型和 Chronos 原先未全部满足动态发现合同，现六条逐一可发现。

## 科学结论

验收证明六条大模型都不是占位，并且统一接口能 fail closed；它不证明五条非时序赛道已经提升。
Chronos 在严格日历 development 诊断上优于历史均值，但还缺同网格树基线及随机/因果控制。
六条路线因此都保持 `CONNECTED_UNVERIFIED`。

监督 LLM 已有统一提示模板、provider-neutral client 和严格响应 schema；由于尚未批准
provider/model/revision，本轮只做 stub 合同测试，没有真实调用外部 LLM API，也没有向 LLM 暴露
标签、原始预测或冻结测试指标。
