# P5 Stage-2 固定预算 development pilot

> 冻结日期：2026-07-14
> 基线：`p5-model-benchmark-integration@85727fd`
> 状态：已完成并独立验收
> 根随机种子：`2693`

## 1. 本阶段回答什么

Stage-2只回答：在同一赛道、同一任务通道、同一development数据和同一小预算下，哪些Stage-1候选值得
进入完整CV。它不是最终性能报告，不打开frozen test，也不把不同标签空间、模态预算或任务定义混成一张榜。

## 2. 公共执行合同

- 使用P4已锁定split manifest；pilot固定使用第一个有效development fold，禁止重切数据。
- `root_seed=2693`；每个候选的模型/loader/sampler seed由稳定哈希派生并写入结果。
- 所有预处理、target transform、类别权重和校准只在该fold的train部分拟合。
- 每个候选先跑tiny-overfit/finite/shape/checkpoint门，再进入pilot；失败或超时写结构化状态。
- frozen test loader、路径、标签、指标和历史test产物均不得被runner接收。
- 传统CPU模型每个cell最多300秒；1D/2D神经模型最多200次参数更新且最多600秒；3D神经/算子模型
  最多80次参数更新且最多900秒。更早达到共同early-stop条件可以停止，但不得给单一模型追加预算。
- 同一lane固定train/validation样本上限；若原始fold更小则使用全部，不能为某候选单独扩样。
- GPU任务必须持有 `/mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock` 的排他锁；一次只跑一个
  可比GPU cell。等待锁的时间不计入模型wall time。
- 禁止长HPO、大权重下载、修改P4 split/test firewall或共享 `_code/ml_framework`。

## 3. 六赛道执行矩阵

| 赛道 | Stage-2纳入对象 | 独立lane / 主指标 | 当前停止线 |
|---|---|---|---|
| ①断层 | 暂不训练；只生成10候选统一data-gate结果 | 工程contract证据，不建性能榜 | 无覆盖审核负例与unknown mask |
| ②地震相 | Stage-1通过的6个候选 | F3与Penobscot分别mIoU/Macro-F1 | 4个依赖/来源skip不替换同名实现 |
| ③储层物性 | Stage-1通过的9个候选 | PHIF、KLOGH、SW分别RMSE/R²/worst-family | TabICLv2权重许可未批准 |
| ④岩相 | P通道可运行的9个候选 | GM09固定九类Macro-F1与worst-family | S通道无连续同井MD批次时不混入P榜 |
| ⑤甜点 | 先审计P4七目标registry并生成P5 label-spec映射；仅批准的目标×候选进入pilot | T1–T7各自指标/榜单 | T5维持`not_feasible`；proxy目标必须显式写proxy |
| ⑥三维重建 | Stage-1通过的8个候选 | strict与conditional分别RMSE/MAE/频谱误差 | 2个依赖/数据门skip；两模式禁止混报 |

## 4. 最小输出

每赛道必须生成：

1. `p5_stage2_results.jsonl`：每个预注册cell一行，含`model_id/task_id/lane/status/reason/seed/split_hash`、
   输入预算、更新步数、wall time、峰值资源、validation指标和test-firewall证明；
2. `p5_stage2_summary.json`：预期/尝试/通过/skip/failed/timeout计数与source/result哈希；
3. 仅当同一lane至少两个候选产生合法validation指标时生成该lane leaderboard；否则输出`not_rankable`；
4. 新增unit/contract测试，证明预算、split、seed、结构化失败和test firewall；
5. 一个干净赛道提交。runner结果若过大或含机器绝对路径，应保存便携摘要而非提交原始缓存。

## 5. 验收与Stage-3准入

- 负责人独立复跑测试、读取结果、核对预注册cell覆盖和工作树状态。
- Stage-2通过不等于模型优胜；只有同lane合法结果才能按预注册主指标、worst-group、稳定性和资源做Pareto。
- 每赛道最多3个候选进入Stage-3；没有科学可比结果的赛道保持数据/标签阻塞，不强凑top-3。

## 6. 实际验收结果

| 赛道 | 预注册cell | 真实pilot | 结构化停止 | Stage-3边界 |
|---|---:|---:|---:|---|
| ①断层 | 10 | 0 | 10 | 缺审核负例与unknown覆盖，不排名 |
| ②地震相 | 20 | 12 | 8 | F3、Penobscot各6个可比候选，榜单独立 |
| ③储层物性 | 10 | 9 | 1 | tabular 8候选可排名；MONAI 3D单候选`not_rankable` |
| ④岩相 | 10 | 9 | 1 | 仅P通道固定九类榜；S通道不混入 |
| ⑤甜点 | 70 | 16 | 54 | T1–T4各4个可比；T5不可行；T6/T7缺development-only特征源 |
| ⑥三维重建 | 20 | 16 | 4 | strict/conditional各8个可比，榜单独立 |
| **合计** | **140** | **53** | **87** | **0 failed / 0 timeout；frozen test未访问** |

六轨验收提交进入集成分支后的SHA依次为`dd44c0a`、`e98a870`、`df7f3a7`、`744c564`、
`a359ad3`、`d46a7b5`。跨轨Stage-2联合门为`torch-common: 59 passed + 22 subtests`；
`tabular-cpu`适用四轨为`40 passed`。完整证据见`../../_tests/P5_stage2_acceptance_evidence.md`。
