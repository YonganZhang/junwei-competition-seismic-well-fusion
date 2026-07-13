# 甜点评价：七个独立任务案例

本目录把“甜点”实现为一个赛道、七个可独立运行和验收的目标。统一的是模型发现、随机种子、分组交叉验证、HPO、checkpoint 和证据格式；不同粒度的标签不会被强塞进同一个样本表或一个综合分数。

## 七目标

1. `reservoir_quality`：RQI 储层品质代理回归；PHIF/KLOGH 只造标签，不进输入。
2. `hydrocarbon_pay`：当前为 CPI `SAND_FLAG` 砂/净储层代理分类，不冒充直接含油气真值。
3. `productivity`：用截止日前 30 天历史预测未来 30 天平均油量。
4. `water_breakthrough`：用事件前 7 天历史预测未来 30 天首次稳定见水风险；当前是报表正水量代理事件。
5. `remaining_oil_infill`：缺少冻结模拟时刻、候选单元和经济标签，按 `not_feasible` 关闭，禁止合成标签。
6. `porosity`：PHIF 独立回归；精确 PHIE 作为额外独立 case，当前不具备多母井支持。
7. `permeability`：KLOGH 的 `log1p` 空间回归，同时报告原始 mD 指标。

目标 1–4 的真实简单模型由 `baseline.py` 运行。默认每个目标执行 8 个 sanity 配置和 20 个 pilot 配置，只用 development OOF 选模型/阈值；随后冻结配置、全 development refit，并通过单次门消费预先冻结的测试井。目标 6/7 由 property adapter 运行。所有结果由 `registry.py` 汇总成 `_outputs/registry_targets_1_to_7.json`。

```bash
python3 -m _pipelines.02_task_datasets.sweetspot.targets.baseline \
  --target all \
  --output-root _tmp/sweetspot-rerun-$(date +%Y%m%d-%H%M%S)
python3 -m _pipelines.02_task_datasets.sweetspot.targets.registry
```

每个可行 case 至少保存 TaskSpec、split manifest、HPO 记录、OOF 指标、冻结配置、refit checkpoint、单次测试指标/预测、专属图和内容哈希 manifest。正式状态与限制以七目标注册表和各 case 的 `status.json` 为准。
