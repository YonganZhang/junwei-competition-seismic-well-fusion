# P4 甜点目标网络调研与本地数据可行性

> 日期：2026-07-14
> 状态：目标 1–4、6、7 的操作定义已冻结；目标 5 fail-closed
> 原则：论文证明“值得做”，本地字段、分组和时间支持才证明“现在能做”。

## 外部依据

- Equinor 的 [Volve 官方数据页](https://www.equinor.com/energy/volve-data-sharing) 说明该开放集包含完整地下、生产和运营数据，约 40,000 个文件，Volve 生产期为 2008–2016。因此可以研究井日志、地震、生产历史和 Eclipse 仿真，但“完整开放”不等于每一个目标都有现场真值。
- 已发表的 [Volve 井产量预测研究](https://doi.org/10.1016/j.petrol.2021.109468) 使用时间、开井时数、压力、温度、油嘴、日产气和日产水等变量预测日产油，支持将产能做成独立时序目标；本项目采用更严格的因果滞后输入和按井隔离。
- 2024 年的 [Volve 产量与见水分析](https://doi.org/10.1016/j.ptlrs.2024.03.001) 说明见水及注采连通是有实际关注度的开发目标。论文中的水驱解释不能直接充当逐日监督标签，因此本项目冻结了可审计的“连续 7 天报告产水”代理事件并显式标注 proxy。
- RQI 文献定义为 `0.0314*sqrt(k/phi)`；[相关储层表征研究](https://www.sciencedirect.com/science/article/pii/S1110062116300319) 给出单位与公式。[动态储层质量研究](https://www.sciencedirect.com/science/article/abs/pii/S0920410522009019) 同时提醒 RQI 来自单相孔渗性质，不能冒充多相动态品质。因此目标 1 命名为 petrophysical proxy，不再发明 PHIF/KLOGH/SW 权重。
- [剩余油与加密井研究](https://doi.org/10.3390/en17143492) 指出只用剩余油饱和度等单一指标并不足以确定井位，需结合可采潜力、约束和仿真。因此目标 5 必须固定 realization、预测/评价时刻、候选井位和经济/井距约束；当前不能用一个静态指数伪装成井位真值。

## 本地真实数据审计

| 目标 | 本地标签/事件 | 开发组与冻结 test | 结论 |
|---|---|---|---|
| 1 储层品质 | CPI `PHIF`、`KLOGH` 构造 RQI；二者只作 label | F-1/F-11/F-12 开发，F-15 test；requested 5 → effective 3 | `proxy_feasible` |
| 2 有效储层/含油气 | CPI `SAND_FLAG`，近 0/1 值才清理为二值；不等同 PAY | F-1/F-11/F-12 开发，F-15 test；5 → 3 | `proxy_feasible`，名称保留但必须显示 sand/net-reservoir proxy |
| 3 产能 | cutoff 前 30 天历史 → 未来 30 天平均日产油 | F-1C/F-11/F-12/F-14 开发，F-15D test；5 → 4 | `feasible` |
| 4 见水风险 | 首个连续 7 天 `BORE_WAT_VOL>0` 的起始日；7 天历史预测未来 30 天 | F-1C/F-11/F-14 开发，F-15D test；F-12 左截断；5 → 3 | `proxy_feasible`；每个 CV/test 井均同时有正负样本 |
| 5 剩余油/井位 | Eclipse `UNRST/GRID` 存在，但无已验证 cell-state parser，时刻/候选井/经济约束未冻结 | 不允许先拆分或训练 | `not_feasible`，禁止 synthetic fallback |
| 6 孔隙度 | CPI `PHIF` 主版本；`PHIE` 单独审计 | 由 property adapter 固定母井族 test/LOGO | 独立任务，不藏入 RQI |
| 7 渗透率 | CPI `log1p(KLOGH)`，最终回到 mD 报告 | 由 property adapter 固定母井族 test/LOGO | 独立任务，不藏入 RQI |

真实审计入口：

```bash
python3 -m _pipelines.02_task_datasets.sweetspot.targets.audit \
  --output-dir _artifacts/p4/sweetspot_contract_audit
```

该命令只生成 TaskSpec、真实样本摘要、split manifest、CRC/哈希证据和目标 5 的 `not_feasible.json`，不执行 HPO，也不读取 test 指标做选择。

## 冻结决定

1. 七个目标保留独立 `task_id`、标签版本、mask、split、模型、指标、checkpoint 和图件；共享外层合同，不共享假定一致的张量头。
2. 目标 1、2 明确是代理任务；网页或报告不得去掉 proxy 警告。
3. 目标 3、4 的样本 cutoff 之前只读历史，cutoff 之后只生成标签；未来列列入 `forbidden_inputs`。
4. 五折是请求值，不是政绩值。独立井族不足时降级到 3/4 折，不拆同井凑数。
5. 目标 5 在 parser 和域定义补齐前没有 baseline、metric 或 figure；这是合格的 fail-closed 状态，不是缺一张假图。
