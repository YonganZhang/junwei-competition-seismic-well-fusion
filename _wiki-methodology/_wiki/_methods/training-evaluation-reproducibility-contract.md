# 训练、评价与复现 SOP

> 适用：本项目六赛道与⑤甜点七任务  
> 决策真源：`_top/_decisions/P4.1_training_validation_reproducibility_architecture.md`

## 一次实验的标准状态机

```text
DRAFT
  -> SPLIT_LOCKED
  -> SMOKE_PASSED
  -> CV_COMPLETE
  -> CONFIG_FROZEN
  -> REFIT_COMPLETE
  -> TEST_CONSUMED
  -> VERIFIED
```

- `DRAFT`：任务语义、标签、输入白名单、loss/metric/search space 可修改。
- `SPLIT_LOCKED`：test group/block/time window 与 development folds 已写入 manifest；之后修改需要新版本。
- `SMOKE_PASSED`：unit、contract、tiny-overfit、real-data smoke 全通过。
- `CV_COMPLETE`：所有有效 folds 结束，OOF/逐折预处理/指标完整。
- `CONFIG_FROZEN`：依据 OOF 选择配置、阈值、校准和最终 epoch 规则，生成只读配置哈希。
- `REFIT_COMPLETE`：只用 development 全量数据重训，checkpoint 与 config hash 对应。
- `TEST_CONSUMED`：单独命令运行 frozen test；写入消费时间、checkpoint、config、split hash。
- `VERIFIED`：独立窗口核对证据、脏状态、指标口径和图件。

任何 test 结果导致的模型修改都必须新开 experiment version，不能回写同一次 `TEST_CONSUMED` run。

## 标准命令语义

实际 CLI 名可在实施时确定，但必须保留以下职责隔离：

```text
prepare      构建/核验 TaskSpec、原始数据索引、标签与 split manifest
smoke        只用 development 做 unit/contract/tiny-overfit/真实小跑
cv           按锁定 folds 训练并生成 OOF；不可加载 test
tune         调用 cv objective；不可加载 test
freeze       从 OOF/HPO 生成冻结配置、阈值、校准和 epoch 规则
refit        用全部 development 重训
test         只允许冻结配置 + refit checkpoint + frozen test
visualize    从保存的预测和指标画图，不重新训练或重新选择阈值
reproduce    同环境同 seed 回归，或预注册多 seed 稳健性复验
```

## Split SOP

1. 定义不可泄漏单位：母井家族、地震体、连续空间块、模拟 realization、生产井+因果时间窗。
2. 先锁定 test，再对 development 划 folds。
3. 生成 group/block/time 边界、空间 buffer/purge、类别/标签支持表。
4. 验证任何原始实体及其派生样本只属于一个集合。
5. `requested_n_splits=5`；按独立 group 与有效类别/正例块诚实降级。
6. 每折只用 fold-train 拟合预处理和采样/类别权重。
7. OOF prediction 每个 development 样本恰好一份；重叠滑窗须先还原到物理 voxel 再计分。

## Seed SOP

1. `root_seed=2693` 是默认，不是硬编码；CLI/config 可覆盖。
2. 用稳定哈希派生 split/cv/model/loader/sampler/augmentation/HPO/diagnostic seed。
3. 设置 Python、NumPy、框架 CPU/CUDA、DataLoader worker/generator。
4. 严格 deterministic 优先；无法严格执行时记录具体 warning 和算子。
5. HPO 确定性验收使用单进程顺序 sampler；分布式加速运行必须标记为非 bitwise reproducible。
6. 保存 RNG state 到可恢复 checkpoint。

## Loss 与输出 SOP

1. `TaskSpec` 先写任务语义和物理边界，再选 loss/输出；模型文件不得隐式覆盖。
2. 二分类/多分类训练返回 logits，使用 logits loss；sigmoid/softmax 在推理 adapter 中显式执行。
3. 回归同时声明 train target transform、inverse transform、raw output、物理约束和越界报告。
4. 候选 loss 只在 development CV 中比较；test 不参与。
5. 多任务 loss 必须按 target mask 聚合；每个目标单独保留 loss 与 metric。
6. reducer 按样本数/有效标签数加权，禁止 batch mean 等权平均。

## HPO SOP

1. 无 HPO 的固定 baseline 始终可运行。
2. 先 8–12 个 sanity/random trials，再 20–30 个 TPE pilot。
3. 小数据/噪声大时先 `NopPruner`；有足够可比 trial 后再保守 pruning。
4. objective 只返回 development fold primary score；另存 std、worst fold、guardrail 和成本。
5. top 3 配置至少做 3 seeds 确认。
6. 搜索结束保存 study DB、trial 表、失败原因、搜索空间、sampler/pruner seed 和最佳配置哈希。

## Metric SOP

- 所有指标保存 sample count、valid-label count、group/fold/seed、平均方式和阈值来源。
- 分类同时保存 per-class、macro、micro/weighted（如适用）与 confusion。
- 极不平衡任务以 AP/PR、Dice/IoU、precision/recall 为主，不用 accuracy 掩盖失败。
- 回归保存 MAE/RMSE/bias/R²；物性保存 log/raw 双空间、逐井/逐层段和越界率。
- 三维任务除体素指标外保存结构/边界/连通/频谱或变差函数诊断。
- 报告 mean/std/worst fold/seed，不只报总平均。

## 可视化 SOP

1. 图件只能读取归档 prediction/metric，不能边画边重新调阈值。
2. 抽样规则预先固定，禁止只挑最好看的测试样本。
3. 每张图标题或 sidecar 至少含 track/task/run/config/split/checkpoint hash。
4. 统一图与赛道图分开：统一图展示训练/CV/HPO；赛道图回答空间、井深、类别、物理或结构问题。
5. 图对应的原始数值、坐标、色标范围和 mask 必须可追溯。

## 七目标甜点 SOP

- 七个目标各自拥有 `task_id`、label mask、split manifest、baseline、metric、图件和可行性状态。
- 目标 6 `porosity` 和目标 7 `permeability` 不能藏入目标 1 的综合评分。
- 构造标签的 PHIF/KLOGH/SW/pay/生产未来字段默认不得作为同任务推理输入。
- 目标 3/4 遵守因果 cutoff；目标 5 标明 simulation truth 或 field truth。
- 数据不足时生成 `not_feasible.json`：缺字段、缺 group、缺时间跨度、需要的数据、验证命令；不伪造标签。

## 最低归档清单

- `task_spec.json`
- `run_config.json`
- `seed_report.json`
- `environment.json`
- `split_manifest.json`
- `folds/*/preprocess_stats.json`
- `folds/*/checkpoint_best.*`
- `oof/predictions.*`
- `hpo/study.*` 与 `trials.*`
- `refit/checkpoint.*`
- `frozen_test/predictions.*` 与 `metrics.json`
- `visualizations/*` 与 sidecar
- `manifest.json`

## 拒收清单

- HPO、early stop、阈值或 loss 读取 test。
- 同井、相邻 patch 或同一未来事件跨集合。
- 只有训练损失或单一 seed。
- 静默 clip 后不报告 raw 预测和越界率。
- 因为不足五组而随机拆样本凑五折。
- 只保存 state_dict，缺 config/split/seed/environment。
- 图件无法追溯到 prediction/checkpoint/config。
- sweetspot 目标用无证据阈值伪造“真实标签”。
