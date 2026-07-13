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
