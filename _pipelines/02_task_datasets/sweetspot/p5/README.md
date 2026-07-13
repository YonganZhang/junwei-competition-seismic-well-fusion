# P5 sweetspot：十模型 Stage-1 合同 smoke

本目录实现 P5 首批十个模型家族的 source lock、T1–T7 独立 TaskSpec 构造、适配矩阵 gate 与 Stage-1 runner。它不定义标签，也不复用 P4 代理标签作为 P5 真值。

## fail-closed 顺序

每个“模型 × 目标”严格按以下顺序通过：

1. 冻结适配矩阵允许该组合；
2. 该目标有独立 `status=approved, approval.approved=true` 的标签合同，且通过现有真实字段、空间尺度、正负/未标注、split 与 test-statistics 审计；
3. source lock、许可证、运行包版本和依赖可核验；
4. 提供内容寻址的 `split=development, contains_test=false, test_accessed=false` 小批次 manifest；
5. 分别用全新 estimator/head 跑合成 batch 与真实 development batch，检查 finite、shape、一步 backward（神经模型）和内存 checkpoint round-trip。

任何一关失败都输出结构化 `SKIP`。runner 没有 test 参数，不调用 `dataset_io`，不创建标签、HDF5、checkpoint、榜单或科学指标。T6 `porosity` 与 T7 `permeability` 拥有不同 head、label hash、manifest 和结果单元。

## 当前安全命令

当前没有七目标专属批准合同，以下命令只打印 10×7 gate 审计；不会读取 development 或 test 数据：

```bash
python -m _pipelines.02_task_datasets.sweetspot.p5.runner
```

批准某目标并由未来 builder 生成 development-only manifest 后，准确接口为：

```bash
python3 -m _pipelines.02_task_datasets.sweetspot.p5.runner \
  --target T6 \
  --label-spec T6=<APPROVED_T6_LABEL_SPEC.yml> \
  --development-manifest T6=<T6_DEVELOPMENT_BATCH_MANIFEST.json>
```

树模型使用共享 `tabular-cpu` 环境；MONAI 使用共享 `torch-common` 环境。InceptionTime 与 TFT 当前也在 `tabular-cpu` 中。命令中的 `python` 应由调用方指向对应共享环境，不在仓库固化机器路径。不得为缺失的 PatchTST、SEG、PyG 或 AutoGluon 自行安装依赖，也不得用同名第三方实现替换 source lock。

## Stage-2：P4 development 标签映射与固定预算 pilot

Stage-2 的版本化映射是
[`sweetspot_p5_label_mapping.v1.json`](sweetspot_p5_label_mapping.v1.json)。它只批准 P4
已有标签进入单折 development pilot，并不批准一个新的“综合甜点真值”：

- T1、T2、T4 保留 P4 的 `proxy_feasible` 语义；T3 保留未来 30 日产油定义；
- T6 PHIF 与 T7 KLOGH 使用不同 TaskSpec、estimator/head、label version 和榜单；
- T5 维持 `not_feasible`，runner 不提供任何代理或合成标签降级；
- 当前 T6/T7 没有可在不打开 `test.h5` 的条件下复建的 development 特征源，因此标签映射可审计，但所有 pilot cell 结构化 `SKIP`。

runner 固定使用各 P4 manifest 的 fold 0、`seed=2693`、每目标相同的样本 ID/输入预算。
树模型最多 64 个 boosting updates；InceptionTime 最多 64 个 AdamW updates。所有预处理只在
fold-train 拟合。GPU 任务必须由调用方显式传入协议锁路径；代码和结果不会固化机器路径。

```bash
TABULAR_PYTHON="${VOLVE_P5_TABULAR_PYTHON:?shared tabular-cpu interpreter}"
GPU_LOCK="${VOLVE_P5_GPU_LOCK:?shared GPU lock}"
PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" \
  -m _pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2 \
  --device cuda --gpu-lock "$GPU_LOCK"
```

便携小型结果仅写入本赛道私有目录 `p5/_outputs/stage2_pilot/`：

- `p5_stage2_results.jsonl`：冻结的 10×7 cell，每行一个真实 pilot 或结构化 skip；
- `p5_stage2_summary.json`：每目标独立榜单、预算、资源和 test-firewall 汇总；
- `p5_stage2_label_mapping.json`：带 P4 provenance/hash/status 的映射审计副本。

准确测试命令（唯一赛道前缀 basename）为：

```bash
PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" -m unittest \
  _pipelines/02_task_datasets/sweetspot/tests/test_sweetspot_p5_stage2.py -v
```

Stage-2 CLI 没有 test 参数，不读取历史 test 指标，不持久化标签、checkpoint 或模型。
