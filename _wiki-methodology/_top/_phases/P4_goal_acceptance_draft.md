# P4 实施 Goal 验收合同

> 状态：已由五窗口调研和负责人网络复核定稿。  
> 实施命令见 `P4_goal_prompt.md`；本文件是不可绕过的验收边界。

## 拟实施分批

1. **集成审计批**：核对五个已验收 worktree commit、主仓脏改动和安全合并顺序；不覆盖用户改动。
2. **公共合同批**：统一配置、seed、split manifest、训练事件、checkpoint、指标与可视化产物 envelope。
3. **赛道插件批**：分别实现 dataset adapter、loss/activation 候选、metric、visualizer 和搜索空间。
4. **⑤七目标批**：扩展 Volve 预处理/连接，逐目标确认可行性并建立独立 baseline；孔隙度和渗透率单列。
5. **验证批**：小批过拟合、真实 smoke、训练/验证 CV、冻结测试集一次性评估、跨随机种子复验。
6. **独立验收批**：负责人按产物和命令重跑，确认没有数据泄漏、伪指标、绝对路径或未登记模型文件。

## 拟共同验收门

- 所有模型通过统一 `ModelBatch -> ModelOutput` 外层 envelope；任务内部张量、head 和 mask 允许不同。
- split manifest 可复用且包含不可泄漏 group/spatial/time 键；测试集从 HPO 和阈值选择中隔离。
- 默认争取 5-fold CV，但仅在训练+验证部分且独立 group 足够时执行；不足时诚实降级并记录，不拆同井/相邻块凑五折；固定配置重训后才评估 frozen test。
- 全局 seed 可从单一配置覆盖 Python/NumPy/框架/DataLoader/采样器/划分器，并保存确定性报告。
- 自动调参是可选模块；简单 baseline 在无调参器时仍能运行。
- 损失、推理激活、阈值、target transform 和 metric 由赛道配置显式声明，不能隐藏在模型文件中。
- 每条赛道有独立可视化入口，⑤的七个目标各有任务适配图；所有图由机器可追溯指标/预测产物生成。
- 每条赛道至少具备 unit、contract、tiny-overfit、real-data smoke 和 frozen-test inference；集成测试缺数据时明确 skip reason。
- 训练、CV、最终测试产物分目录保存，防止把训练指标误报为测试指标。
- 环境、数据哈希、代码 SHA、配置、split、seed、checkpoint、指标、图和运行日志均进入 manifest。
- loss reducer 按样本数或有效标签数加权，不按 batch mean 等权平均。
- 规范模型真源位于 `_models/<track>/<model_id>.py`；旧目录仅可留兼容 shim，不保留重复实现。
- ⑤七个目标各自有任务 ID、标签状态、split、baseline/`not_feasible`、指标与图；孔隙度和渗透率独立验收。

## 拟拒收条件

- 随机 patch/样本切分让同井或相邻空间数据跨集合。
- 用测试集挑 loss、epoch、阈值或超参数。
- 只报告训练损失、单次随机种子或单一平均指标。
- 用构造标签的字段直接作为该任务推理输入而未做泄漏声明。
- 把不存在/数据不足的⑤目标用随意阈值伪造成真实标签。
- 强行给所有任务使用同一个 loss、最后激活或可视化模板。
- 为跑通而改写/删除用户现有脏工作树内容。
- 为满足形式上的五折而拆同一母井、随机 patch、取消 buffer 或混入 test。
- 静默裁剪回归输出、只报告约束后指标而不报告 raw 指标和越界率。
