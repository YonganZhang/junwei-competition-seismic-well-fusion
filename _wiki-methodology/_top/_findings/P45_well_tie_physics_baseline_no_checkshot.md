---
phase_id: P45
status: accepted
severity: minor
owner_col: COL2
source: experiment
created_at: 2026-08-07
---

# P45 I0：无checkshot物理baseline井震标定——真实可测但离checkshot精度尚远，且发现F3暂不能做跨工区验证

## Local Case

军伟在GitHub issue #1提出：现有井震标定（P23）依赖真实checkshot，未覆盖"未知工区无时深表/checkshot"场景，要求探索AI能否在物理方法基础上做增量。按预注册的"先物理baseline、再判断AI是否有增量"路线，本轮（I0）用`welly`+`bruges`+`dtaidistance`实现了一条完全不读取checkshot的物理管线：

1. 从`well_logs_clean.h5`取声波(DT)+密度(RHOB)，只用非NaN的最长连续区间。
2. 用官方`Well_picks_Volve_v1.dat`的MD/TVDSS对做深度类型转换（不是时间信息，非checkshot泄漏）。
3. 声波积分得到**相对**单程时（绝对起始时间未知，这是声波积分的已知物理局限，issue里也提到过）。
4. `bruges.reflection.acoustic_reflectivity`算反射系数，配合从真实地震道频谱估计的Ricker子波做褶积，得到合成地震记录。
5. 对候选静态时移(t0)做粗相关搜索，取前3个非极大值抑制后的相关峰，峰间差距<0.05时标记`reject_ambiguous=true`（对应军伟验收标准"后验不明确时自动拒绝结果"）。
6. 在最优t0附近±200ms窗口内做**斜率约束**DTW精细对齐（局部最大拉伸20%，避免军伟提到的"无约束局部拉伸获得高相关系数却产生不合理速度"）。

在Volve唯一有DT+RHOB覆盖的3口井（19A/19BT2/19SR，同属15/9-19井槽）上独立测试，用各井**自己的真实checkshot**（VSP归档）做事后评分（checkshot只用于评估，不进入预测流程）：

| 井 | 粗搜索t0 | xcorr | 候选间差距 | 是否拒绝 | 评分点数 | MAE(ms) | bias(ms) |
|---|---:|---:|---:|---|---:|---:|---:|
| 19A | 2740ms | 0.553 | 0.018 | 是(歧义) | 39 | 265.9 | +265.9 |
| 19BT2 | 2560ms | 0.430 | 0.066 | 否 | 71 | 183.8 | +183.8 |
| 19SR | 612ms | 0.547 | 0.022 | 是(歧义) | 20 | 1863.8 | -1863.8 |

对比P23的真实checkshot锚定结果（目标储层MAE 8.7ms）和更早的纯官方分层弱标定（MAE 633ms）：本轮的"无checkshot物理baseline"精度**介于两者之间但明显更接近弱标定那一端**——比633ms好，但离8.7ms差一到两个数量级。

## Class Pattern

物理baseline的表现符合井震标定领域的既有认知：声波积分能提供正确的**相对**时深关系（斜率/形状），但绝对起始时间和子波相位这两个自由度必须靠某种外部约束（checkshot、或跟真实地震的相关搜索）来锁定，而全局盲搜相关峰在信号弱/覆盖窄的区间容易锁到错误峰（周期跳跃），这正是19SR的情况：它唯一的DT+RHOB连续区间(MD 4175–4596.5m)是井底附近一段很窄的区间，跟checkshot覆盖范围([0, 3064.5m] TVDSS)只有边缘重叠，导致合成记录信息量不足、相关峰高度模糊——但**歧义检测机制成功识别了这一点**(reject_ambiguous=true)，说明"报告不确定性、歧义时拒绝"这个工程约束是真实有效的，不是摆设。19A同样被标记为歧义（差距0.018），其265.9ms的误差也印证了标记的合理性；唯一通过歧义门槛的19BT2误差(183.8ms)也仍然不小，说明即使"看起来不歧义"，物理baseline本身的精度上限也有限。

## Evidence

- 脚本：`_pipelines/02_task_datasets/reconstruction/p45_well_tie_physics_baseline.py`
- 测试：`_pipelines/02_task_datasets/reconstruction/_tests/test_p45_well_tie_physics_baseline.py`，4/4通过（DT/RHOB物理量纲检查、MD→TVDSS单调性、声波积分单调性、歧义门锁定19SR）
- 产出：`_pipelines/02_task_datasets/reconstruction/_outputs/p45_well_tie_physics_baseline/summary.json`（含每口井的粗搜索候选峰、DTW距离、子波频率、评分明细）
- 依赖：新增`welly`(0.5.2, Apache 2.0)、`bruges`(0.5.4)、`dtaidistance`(2.3.13)，均`pip install --user`装在本机用户目录，未sudo污染系统。

## Impact

1. **物理baseline本身可用但不够精确**：作为"完全没有checkshot时的应急/初筛手段"，比纯官方分层插值(633ms)有意义的提升，可以给出粗略但比瞎猜好得多的时深关系；但不能替代真实checkshot(8.7ms)，也不建议直接喂给下游③④⑥的井震融合任务当作精确对齐——现在这个精度级别的误差会导致配对到错误的地震道/时间窗口。
2. **歧义检测机制值得保留**：这是本轮除精度数字外最有工程价值的产出——它把"看起来算出来了"和"算出来的东西可信"分开，直接落实了军伟验收标准里的"不确定性/自动拒绝"要求，后续I1如果要在这个baseline上加AI组件，应该保留、而不是绕过这个门。
3. **F3暂时不能做跨工区验证**（重要的范围收窄，纠正此前的错误假设）：项目里已下载的F3数据（Zenodo `1471548`）核实后是inline/crossline图块+层位掩膜的解释数据集，**不含任何LAS测井曲线或原始SEG-Y振幅道**——数据注册表里"4口井测井"的表述描述的是F3原始数据集页面的介绍内容，不是我们实际下载到的文件内容。这意味着I0原计划的"F3端到端冒烟测试"目前做不了，不是算法问题，是数据缺口。若要真正验证跨工区泛化，需要额外下载官方F3-Demo的4口井LAS文件（dGB/OpendTect学术数据集，与当前Zenodo ML-slice是不同的下载源）。在此之前，跨"地震采集/处理批次"的鲁棒性测试可以先用Volve自己的ST0202 vs ST10010两期数据做部分替代（真实数据、不同处理批次），但这不能完全代替真正的跨工区测试。

## Prevention Rule (candidate)

全局盲搜时深关系的粗相关峰前，先检查候选井的物性曲线连续覆盖区间跟评估/目标区间的重叠比例——重叠过窄或过偏时，相关搜索的信噪比不足以可靠定位，应该在算法层面强制走"歧义/拒绝"路径而不是相信单一最优分数；同时任何"跨工区验证"的立项前，必须先核实目标工区已下载的文件里是否真的包含所需模态（本例中是LAS曲线），不能只看数据集页面描述或早期registry note里的概述性文字。

## Links

- task_plan: ../_task_plan.md
- GitHub issue: https://github.com/YonganZhang/junwei-competition-seismic-well-fusion/issues/1
- 对照基线: `P23_reconstruction_checkshot_calibration.md`（同一批井，checkshot锚定，MAE 8.7ms）
- 数据缺口: `_meta/_data_registry.yml` 的 F3 条目需要后续补充"实际下载内容不含LAS"的澄清
