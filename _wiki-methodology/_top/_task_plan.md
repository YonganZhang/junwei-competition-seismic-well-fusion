# 军伟的比赛（地震+测井多模态融合识别有利油气目标） — 任务计划

> 创建: 2026-07-08
> **🧭 当前: [P45 无checkshot井震标定 I0] 军伟 GitHub issue #1 驱动，物理baseline(声波积分+bruges反射系数+相关搜索+斜率约束DTW)已在Volve 3口有DT+RHOB的井上跑通，19BT2通过歧义门MAE=183.8ms，19A/19SR被正确拒绝为歧义。F3当前下载内容不含LAS测井，跨工区冒烟测试暂被数据缺口阻塞。下一步：I1(小模型探索)或先解决F3井数据缺口。**
>   2026-08-06 增量：⑥P39 固定共同 well-only base 后，双预训练融合宏平均 RMSE=`0.075908484`，仍弱于 well-only `0.075314433`；④P40 双基础融合 Macro-F1=`0.161312`，弱于 B0 `0.213349`；③P41 双基础融合相对 B0 仅改善 `0.1703%`，2/4 外层胜出且 bootstrap 区间跨零。三条均不晋级，实验代码、轻量证据和结论边界已归档。
>   2026-08-06 结论收窄：用户质疑触发 Claude 子智能体 + Codex 双独立复核（不预设立场，重新从数据侧核查）。⑥P38/P39 负结果确认可信、无 bug。但④P40、③P41 发现同一类结构性接口缺陷——GFM 地震侧只保留整道 CLS token、未传入查询点 time_idx，导致 P40 的 447 个样本只对应 69 个不同地震表示（443/447 行落在同道重复组，39 组内部混着不同岩相标签）、P41 的 1216 个样本只对应 211 个不同地震表示（1201/1216 行同道重复但组内真实物性值明显不同）。这不是数据/坐标/标签 bug，是"整道级表征当作深度局部表征使用"的接口设计缺陷。P40、P41 的结论已从 `R0_STOP_NO_ATTRIBUTABLE_SIGNAL` 收窄为 `R0_INCONCLUSIVE_DEPTH_BLIND_SEISMIC_INTERFACE`（详见各自 finding），下一步应换用 P39 已验证过的 query-local 波形 token 设计重新测试，而非直接判定井震融合在④③无效。
>   2026-08-03 增量：① P30 连续三维开发体已集成，CIG-Bench guard 逐体素 F1=`0.003555` 低于 baseline `0.017641`，不晋级；忽略目录中的 joblib 已转为哈希绑定的可移植系数检查点，在完整 P30 体上复算指标至 12 位小数一致。④默认 XGBoost 已升级为 `depth=3/eta=0.1/rounds=60`，Macro-F1=`0.213349`，与 MOMENT/LLM 无关。六赛道聚焦回归共 61 项通过。
>   P21 在开发 OOF 上相对 PyKrige RMSE 改善 `2.5144%`，相对 P19 改善 `0.0613%`；P24 同场区历史版本迁移中相对重拟合 PyKrige 改善 `1.4529%`，但不声称跨场区、fresh blind 或 LoRA 因果贡献。
>   六赛道 baseline pipeline 候选已完成真实数据端到端验收和可移植性收口；
>   ①②③④⑥又各新增2个已通过动态发现/小批次训练/检查点契约的简单备选模型（每条现有3个模型，SHA见下方），
>   ⑤已由军伟拍板为一个赛道七个任务目标（储层品质、含油气/有效厚度、产能、见水风险、剩余油/加密井潜力、
>   孔隙度、渗透率）；五窗口只读调研、统一训练/验证/复现/可视化 SOP 与分批实施 Goal 均已收敛。
>
> **P25 历史状态：累计研究主线已合并至安全集成分支，P21 为赛道⑥默认模型。**
>   2026-08-01 P24 在读取指标前预注册并冻结 P21，在未被现有管线使用的
>   同场区历史 `pp04phif/realisation.1` 上完成一次性迁移测试。保持 5 个空间折、
>   每折 512/2,048 标签预算和 PyKrige 1.7.3 不变后，RMSE 从
>   `0.028235410003` 降至 `0.027825182663`（相对改善 `1.4529%`），MAE 从
>   `0.021293578592` 降至 `0.020826337775`，4 折胜/1 折负，通过预注册门槛。
>   该证据只属于同场区历史版本迁移，不是跨场区或首次盲测；整折 bootstrap
>   95% 区间仍跨 0。用户已明确接受该证据等级作为本轮闭环，不再追加跨场区测试。
>   P4--P24 累计研究线、P12 可视化线与公开仓库脱敏提交已合并到 `final-integration`；
>   原 `master` 工作树含用户未提交报告/可视化改动，暂不移动其分支指针。
>   证据见 `_findings/P24_reconstruction_historical_transfer.md`。
>   2026-08-01 P23 复核发现本地 Volve VSP 归档含 5 口井
>   checkshot，改正了“项目没有 VSP/时深表”的旧判断。使用
>   19A/19BT2/19SR 拟合、F11T2/F15A 独立校验后，目标储层
>   TWT MAE 从 `633.1867 ms` 降至 `8.7389 ms`，2/2 独立井均改善。
>   但全体积替换和直接观测支持门控的下游 RMSE 分别为
>   `0.027768546911` 和 `0.027790989240`，均未超过 P21 的
>   `0.027734374378`。因此标定资产保留，模型候选拒绝；不声称
>   孔隙度提升。证据见 `_findings/P23_reconstruction_checkshot_calibration.md`。
>   2026-08-01 P21 将学习信号改为无标签多视图地震一致性适配，再学习 512 标签
>   内部五折 P19 同构基准的样本外残差。LoRA 与头部梯度均真实非零，但神经头、
>   单次校准 Ridge、完整五折 Ridge 的 RMSE 分别为 `0.028278976997`、
>   `0.029618418227`、`0.028080039761`，均未超过 P19。去掉逐折元选择、统一平均
>   三个固定 foundation 核后 RMSE 为 `0.027734374378`，相对 P19 的
>   `0.027751397628` 小幅改善，外折 1 胜/4 平/0 负。该路线按简化胜出启用，
>   不声称广泛统计效应或 LoRA 因果贡献；后续应增加独立地质监督，而不是继续扩大
>   PEFT 容量。证据见 `_findings/P21_reconstruction_contrastive_residual.md`。
>   2026-07-31 P19 发现：P18 每折自身训练/验证虽零重叠，但其余折的 512 点
>   训练子集与当前验证折仍有 24--58 个标签坐标重叠，可能间接影响元选择。
>   删除这些坐标、重新拟合全部候选后，10,240 条 OOF RMSE 为
>   `0.027751397628`，相对 PyKrige `0.028449728170` 改善 `2.4546%`，仍为
>   5/5 折改善；整折 bootstrap 95% 区间
>   `[-0.001143968280, -0.000353924655]`。P18 主信号没有由该漏洞制造，但
>   协议和数值已由 P19 取代。真实 P15 动力学探针同时确认：17,298,000 个
>   可训练尾块参数因零初始化输出层首步梯度为零，三步相对更新约 `2e-5`，
>   比头部约 `9.4e-3` 小约 470 倍。GELU/SiLU/ReLU、冻结小 MLP、扩展网格、
>   回归克里金和 K/J/I 层序距离均未在严格口径下超过 P19。下一步应预注册
>   参数高效适配器与分阶段解冻；按用户要求，消融仍后置，默认仍关闭。证据见
>   `_wiki-methodology/_tests/P19_reconstruction_training_diagnostics_acceptance_evidence.md`。
>   P18 历史记录（已由上方 P19 取代）：2026-07-31 Claude 独立审查发现 P17
>   在同一 OOF 上从 156 个候选中选优；
>   对原候选族改用嵌套留一空间折后 RMSE 为 `0.028534404074`，原 `-0.4563%`
>   已被取代。P18 增加垂向各向异性，并对每个报告折只用其余四折选择 top-3。
>   在不扩大每折 512 条训练标签、不打开冻结 holdout 的条件下，10,240 条 OOF
>   RMSE 从 `0.028449728170` 降至 `0.027752680679`（-2.4501%），5/5 空间折
>   全部改善；完整空间折 bootstrap 95% 区间
>   `[-0.001140994782, -0.000353924655]`。状态为
>   `ROBUST_DEVELOPMENT_SIGNAL`、`default_enabled=false`。本阶段按用户要求不做
>   消融，只报告含真实预训练 GFM 的完整方案，不声称具体因果来源。证据见
>   `_wiki-methodology/_tests/P18_reconstruction_anisotropic_acceptance_evidence.md`。
>   2026-07-28 已修正旧 P6 Gaia/DAGT“只有合同/QC、没有大模型参与预测”的边界：
>   断层 SAM-Med3D、地震相 SAM 2.1、物性 TabICLv2、岩相 MOMENT、甜点 Chronos-2、
>   三维重建 OpenMind ResEnc-L MAE 均已固定 source/weights revision 与 SHA，并完成真实加载计算。
>   统一 `FoundationTaskEnvelope` 覆盖 time/support/depth/spatial/masked-volume 条件；
>   统一监督 LLM 模板与严格 JSON client boundary 已实现，但无批准 provider 时不发 API。
>   六条当前都为 `CONNECTED_UNVERIFIED`、`default_enabled=false`：接线成功不等于提升成立。
>   Chronos 严格 30 日历天真实 development 4 折诊断 MAE `172.3162`，优于历史均值 `184.6686`，
>   但同网格树基线和随机/因果控制未完成，不能晋级。断层仍缺审核负例、连续 3D block，
>   且 checkpoint 条款待确认。下一步按赛道完成 pretrained/random-init/强基线同 split 对照，
>   只允许通过晋级门的路线成为默认。证据见
>   `_wiki-methodology/_tests/P8_multimodal_foundation_acceptance_evidence.md`；
>   相关实现和证据已进入本次累计研究线集成，未 push。
> Done Criteria: Volve/F3-Demo/Penobscot 三批数据下载完整校验通过(已达成)；六赛道baseline pipeline候选
>   各自端到端验收通过(已达成，见下方P3/P4)；③④标签/目标定义已落地；
>   ⑤七目标均有可审计状态；累计研究线已在安全集成分支完成合并；正式交付仍需同步主工作树与报告。
> 目标交付物: 模型 + 模型参数 + 数据预处理代码 + 模型运行代码（无文字报告要求）
> 距交付: TBD（官方未给最终截止时间，待军伟电话确认）

| Phase | 状态 | 进度 | 关键 finding |
|---|---|---|---|
| P0 问题界定 | ✅ | 1/1 | 赛题核心=地震+测井多模态融合预测有利油气目标，见下方赛题信息 |
| P1 方法论 / 方案 | 🔄 | 0/1 | pipeline 骨架已定（5阶段），具体算法待补 |
| P2 数据 / 实现准备 | ✅ | 3/3 | Volve全套(含ST10010)+F3+Penobscot下载完成；③④赛道数据量核实足够(纠正见P2.2) |
| P3 探索 / 实现 | ✅ | 6/6 | 六赛道 baseline pipeline 候选均已完成端到端验收、可移植性收口和 `master` 累计合并，见 P2.6 与各阶段 SHA 登记。 |
| P4 验证 | ✅ | 2/2 | 公共合同+五赛道插件+⑤七目标已在隔离集成分支实施；便携测试、真实 smoke、可行任务 CV/refit/frozen-test 与产物哈希已验收，证据见 `_tests/P4_acceptance_evidence.md` |
| P5 合成 / 交付 | ✅ | 1/1 | 六赛道累计研究线已合并；P21 默认配置、复现入口、45 页技术报告与固定下载链接已同步。 |
| P6/P7 Gaia+时序 | ✅ | 2/2 | 旧 Gaia/DAGT 边界已审计；Chronos-2 真实预训练权重与因果预测通路已建立。 |
| P8 多模态基础模型 | 🔄 | 0.65/1 | 六赛道权重/接口/真实计算已通过；同 split 性能、random-init、shuffle/causal 控制与断层数据门待完成。 |
| P12 可视化标准化 | ✅ | 1/1 | 赛道1/3/5出版级图组、确定性渲染、人工复核和稳定交付入口保留。 |
| P13 科研报告 | ✅ | 1/1 | 六赛道统一研究合同、结构诊断、架构图和45页 LaTeX 技术报告 v1.4 已完成。 |
| P17 重建基础模型 | ⏭️ | 1/1 | 同 OOF 选优的 -0.4563% 已由 P18 嵌套复核取代；保留为方法角色转换历史。 |
| P18 各向异性重建 | ⏭️ | 1/1 | 历史方向有效，但元选择训练坐标未去重；协议和数值已由 P19 取代。 |
| P19 重建训练诊断 | ✅ | 1/1 | 元选择训练坐标去重后 RMSE -2.4546%、5/5 折改善；定位 P15 全尾块微调梯度失配，五条替代路线严格复核未晋级。 |
| P21 固定基础核 | ✅ | 1/1 | 固定三核平均 RMSE `0.027734374378`，简化胜出；LoRA 残差路线未超过强基线。 |
| P23 Checkshot 标定 | ✅ | 1/1 | 3 拟合井/2 独立校验井将 TWT MAE 降至 `8.7389 ms`；下游候选未超过 P21，仅保留标定结论。 |
| P24 历史版本迁移 | ✅ | 1/1 | 冻结 P21 在未使用的同场区历史属性版本上相对 PyKrige RMSE 改善 `1.4529%`、4/5 折获胜；用户接受同场区迁移证据，本轮停止跨场区扩展。 |
| P25 安全集成 | ✅ | 1/1 | P4--P24 累计研究线、P12 可视化线与脱敏提交已先合并到 `final-integration`。 |
| P26 主线收口 | ✅ | 1/1 | `final-integration` 已无损合并到 `master@8375b97`；可视化 7/7、重建 11/11 回归通过，PDF 与 LaTeX 源码包固定链接已更新。 |
| P27 报告终审 | ✅ | 1/1 | 修正 P21 跨章节旧结论与科研总览图，六赛道目录逐项核对；45 页 PDF、143 项源码包和固定链接通过交付检查。 |
| P28 执行型智能体消融 | ✅ | 1/1 | 六赛道 Stage 1 已完成：智能体均能真实选动作并执行，但 A2L 均未获得可归因的 prediction endpoint 增益，因此不进入直接 LLM 数值优化 Stage 2；保留确定性执行器及有限的 LLM 路线/诊断/停止角色。Claude 最终复核无 blocker/major。 |
| P29 动作效应修复 | ✅ | 1/1 | 六赛道动作链、对照和晋级口径完成修复与独立验收：②仅保留确定性 hybrid（promotion mean mIoU 相对 A0 `+0.030481`），③仅保留 A2D；直接 LLM 在六赛道均未取得可归因的稳定 endpoint 优势。详见 `_phases/P29_agent_action_effect_repair.md`。 |
| P30 断层连续三维评测 | ✅ | 1/1 | 连续 3D development 体、verified background/unknown mask、group-isolated split 和 CIG 对比已集成到 P31 分支；CIG 不晋级，可移植 baseline 复算通过。 |
| P31 智能体优化与管线注册 | ✅ | 1/1 | 六赛道 P29/P30/默认配置、六个独立 manifest、七段 lifecycle CLI、registry、gate 与机器 stamp 已合入主仓；doctor 显示 8/8 pipeline fresh。详见 `_findings/P31_agent_optimizer_and_six_pipeline_registration_audit.md`。 |
| P32 混合智能体优化 | ✅ | 1/1 | ②③完成“LLM 候选生成 + 确定性预算调度”真实 development pilot 与独立复跑。③相对确定性策略主指标改善 `4.2696%`、3/3 seed 全胜；②等均 mIoU 改善 `+0.024699`，F3 `+0.049397`、Penobscot 不降。完整候选池不完全稳定，但最终可执行决策与指标稳定；frozen test 未读取。详见 `_findings/P32_hybrid_agent_optimizer_results.md`。 |
| P33 四赛道混合扩展 | 🔄 | 1/4 | ④完成真实 matched-budget pilot 与独立复跑：智能体相对确定性端点 `+0.027373`，但相对现默认 A0 `-0.013750`，共同护栏拒绝假晋级并保留 `depth=3/eta=0.1/rounds=60`。下一步推进⑥重建。详见 `_findings/P33_lithofacies_incumbent_guard_prevents_false_promotion.md`。 |
| P34 Pipeline 模块化 | ✅ | 4/4 | 六条独立 Pipeline、共享运行时、赛道 adapter、CodeBook、registry 与防漏门禁已完成；14 项运行时测试、3 项 lifecycle 回归、42 阶段核验和 Claude 独立终审通过，六条 manifest 验签均已刷新。详见 `_phases/P34_six_track_pipeline_modularization.md`。 |
| P35 接口与证据收口 | ✅ | 4/4 | ①最终 ST10010 评估原子接入 Pipeline；⑥P29 v2 显式注入 feature cache 和查询侧模态，真实五折重算；P30 v2 复验地质统计候选并产出井震跨模态 I/O 合同；旧 P29 v1 退出晋级证据。详见 `_findings/P35_fault_reconstruction_interface_closeout.md`。 |
| P36 重建方差修复 | ✅ | 1/1 | 修正 P30 covariance-form ordinary kriging 方差的 Lagrange multiplier 符号；权重、均值预测、RMSE 和晋级决定不变。 |
| P37 真实井监督门 | ✅ | 1/1 | 三口父井无法同时满足原生 PHIE 与合法 development KJI 支持，状态为 `BLOCKED_REAL_ALIGNED_SUPERVISION`；未启动模型小试。 |
| P38 真实井 PHIF 融合 | ✅ | 1/1 | 完成三父井 LOGO3、真实测井 PHIF 与原生地震对齐；双预训练融合 RMSE=`0.079781229`，未超过 well-only `0.075314433`。 |
| P39 查询侧井震融合 | ✅ | 1/1 | 修复所有控制共享同一锁定 P38 well-only base；双预训练 RMSE=`0.075908484`，置信区间和错位门均未通过，状态为 `FEASIBLE_NO_PROMOTION`。 |
| P40 岩相双基础资格门 | ⚠️ | 1/1 | MOMENT+GFM 真实进入门控残差头，Macro-F1=`0.161312`低于B0`0.213349`；**2026-08-06收窄**：GFM整道CLS token丢失深度信息(447样本仅69个不同表示)，结论改为`R0_INCONCLUSIVE_DEPTH_BLIND_SEISMIC_INTERFACE`，非"融合无效"，待depth-local token重测。 |
| P41 物性双基础资格门 | ⚠️ | 1/1 | 复合 RMSE 从 `0.427225122` 降至 `0.426497478`，仅改善`0.1703%`；**2026-08-06收窄**：同一类GFM深度盲缺陷(1216样本仅211个不同表示)，结论改为`R0_INCONCLUSIVE_DEPTH_BLIND_SEISMIC_INTERFACE`，待depth-local token重测。 |
| P42 主仓与交接收口 | ✅ | 1/1 | 集成 P36--P41 的可复现增量，保留六 Pipeline 默认与拒绝结论，归档 Claude 接手文档、轻量证据和验证记录。详见 `_findings/P42_six_track_progress_and_claude_handoff.md`。 |
| P43 甜点门限根因 | ✅ | 1/1 | T6/T7 的 `no development-only feature source` 实为样本身份不可逆（P4 冻结 1216 个内容哈希 ID，物化 h5 已不存在），标签源 `PHIF`/`KLOGH` 各 35810 条一直在被读取；T2 的 AP `0.9847` 非泄漏而是 `SAND_FLAG` 代理任务。另更正：T6/T7 P4 阶段已完整完成并过 frozen test（T6 R²=`0.93411`）。详见 `_findings/P43_sweetspot_seven_target_gate_root_cause.md`。 |
| P44 甜点标签溯源 | ✅ | 1/1 | 特征消融显示 T6 上单条 `RHOB` 即达 R²=`0.9696`、全 16 条仅 `0.9709`；T1/T2/T6/T7 的标签均为 CPI 解释产物，模型在复现解析式而非预测地质，其中 T6/T7 的 `is_proxy=False` 标注与证据不符。七目标中仅 T3/T4/T5 具备真实预测意义。详见 `_findings/P44_sweetspot_label_provenance_collapse.md`。 |
| P45 无checkshot井震标定 | 🔄 | 1/3(I0) | 军伟issue #1驱动。I0物理baseline(声波积分+bruges反射系数+Ricker子波+相关搜索+斜率约束DTW)在Volve 3口有DT+RHOB的井上测试：19BT2通过歧义门MAE=183.8ms，19A/19SR被歧义检测正确拒绝(19SR因DT+RHOB覆盖区间跟checkshot范围只有边缘重叠导致周期跳跃，MAE若不拒绝会是1863.8ms)。对比P23 checkshot锚定(8.7ms)仍差1-2个数量级，但比纯官方分层弱标定(633ms)有意义提升。**重要发现**：项目已下载的F3数据是图块+层位解释集，不含LAS测井曲线，原计划的跨工区冒烟测试暂不可行，需额外下载官方F3-Demo井数据才能做真正跨工区验证。详见`_findings/P45_well_tie_physics_baseline_no_checkshot.md`。I1(AI小模型探索)、I2(基础模型接入)待续。 |
| P46 甜点三层技术流固化 | ✅ | 1/1 | ⑤甜点的小模型层（P5）、大模型层（P7/P8 Chronos-2）与智能体层（P28/P29）已写进 adapter：`baseline` 解析 incumbent、`optimize` 承载智能体、`verify` 三层同验。此前 P7 Chronos-2 的晋级结果（T3 MAE `186.572`，较归档 XGBoost 降 `30.15%`）完全游离于 pipeline 之外，verify 可以全绿而验不到真正的冠军。新增单一真源 `sweetspot/_outputs/incumbent/incumbent.json`（incumbents / rejected_routes / open_work）。六赛道 verify 仍 PASS。 |

## 协作与决策权边界

- **军伟**：顶层设计决策者 + 技术路线（多模态算法/最新AI模型魔改，军伟自己负责，不外包）。
- **永安（本人）**：数据获取、pipeline 骨架梳理、文献调研协调。
- **孟洋师兄等其他人**：建议权，非决策权——军伟判断有用才采纳。
- **原则**：不闭门造车，尽量复用现成轮子（开源代码/预训练模型/已发表pipeline）。

## 赛题信息（官方，摘要不含具体建模方法）

### 核心任务
利用地震数据和测井数据做多模态融合，识别地下有利油气目标；重点证明"地震+测井融合"相比"仅地震"有指标提升。

### 输入
1. **地震数据**：三维叠后地震体/图像（SEG-Y / numpy / inline-crossline-time 切片等格式）
2. **测井数据**：LAS格式，常规九线（岩性：SP/GR/CAL；三孔隙度：AC/CNL/DEN；电阻率：MSFL/LLS/LLD）
3. **井位与井轨迹**：井口坐标、井轨迹、MD/TVD/TVDSS、补心海拔、地面海拔（单独文件）。比赛输入本身不含时深表/校验炮/VSP/合成记录/速度模型；但项目后续下载的 Volve VSP 归档已确认包含 5 口井 checkshot，可作为研究阶段的外部标定资料，不应与比赛可用输入混为一谈。
4. **专家标签**：断层/有利储层等目标标签，三维网格对齐优先，允许二维切片/井点/弱标签

### 可选辅助数据（仅可用于训练增强，推理阶段不可强依赖）
时深表、速度模型、合成记录、地层分层、岩性解释、储层参数解释、地震属性体

### 输出
断层概率图 / 有利储层概率图 / 有利油气目标概率体 或 二值分割结果（具体类型待与官方确认）

### 评估指标
- 分割类：Dice / IoU / Precision / Recall / F1
- 概率类：ROC-AUC / PR-AUC / Top-K命中率
- 断层专项：连续性 / 边界F1 / 骨架匹配
- **核心对比实验**：地震单模态 baseline vs 地震+测井融合模型，看指标是否提升——这是证明多模态融合有效性的关键，比赛评判重点

### 提交要求
模型 + 模型参数 + 数据预处理代码 + 模型运行代码；需在多个开源下游分割任务数据集上测试。**不要求文字报告**。

### 已知信息缺口
- 比赛主办方"勘探院"流程仓促，缺测试数据提交样例（军伟原话）
- 具体输出类型、最终截止时间待军伟电话确认后更新

## 当前门槛（P2 阶段）

- [x] Layer1公共预处理(`_pipelines/01_common_preprocess`)4个step已用真实Volve数据(ST0202地震体
      +06.LFP测井LAS+Official_Faults.dat+Horizons_TWT层位+Well_picks井分层点)跑通验证，
      2026-07-10。细节见该目录`_readme.md`与`_meta/_registry.yml`的`common_preprocess`条目。
- [ ] Volve 数据（~5TB全套，含2套3D地震+4D地震+测井+生产+钻井）下载完整校验通过（进行中，见 `data:volve-north-sea`；`GeoScience_OW_Archive.zip`此前截断827MB已修复）
- [ ] F3 Demo（荷兰北海，~8GB）下载 + 对应 Zenodo ML切片包（下载中，见 `data:f3-demo-netherlands`）
- [ ] Penobscot（加拿大Nova Scotia，~13GB）下载 + 对应 Zenodo ML切片包（下载中，见 `data:penobscot-canada`）
- [ ] 军伟周五前完成文献综述（1~5号pipeline阶段对应文献 + 其他领域可迁移方法文献包）
- [ ] 传统石油行业标准pipeline调研（军伟负责问询），确认能否补全"传统方法先后顺序"作为本项目pipeline骨架的参照
- [x] 六赛道数据可行度盘点（断层/相分类/储层物性/岩相/甜点/三维重建），Excel已发用户；③④两条经核实岩心样本量不足，
      需军伟决策是否降级为"仅校准用途"，见 `_findings/P2.1_volve_core_data_scale_gap.md`
- [x] 2026-07-11 **纠正上条结论**：之前只查了`06.LFP`/`09.CORE`(3口井)就判定"数据不够"，漏查了同一
      压缩包里`02.LWD_EWL`(24口井全覆盖原始测井)和`05.PETROPHYSICAL INTERPRETATION`(21口井CPI)。
      Volve历史钻井24口，测井曲线覆盖21-24口井，样本量足够支撑③④赛道，**不再需要军伟决策降级**，
      可直接开工；岩心数据(30-47样本)改作校准/精度验证用途。见 `_findings/P2.2_volve_well_log_scale_correction.md`
- [x] 2026-07-11 用户拍板：①②⑥三条赛道用git worktree隔离，各开一个Codex worker(leader mode由本Claude
      会话主线负责)。①⑥已用**简单baseline模型**(逻辑回归/岭回归，非深度学习)跑通完整pipeline验证数据管道，
      ②因F3/Penobscot标签空间不兼容(军伟决策：不强行统一，同一模型架构分开训练/测试)重新派活中。
      **🔴 当前baseline模型仅用于验证数据管道，不是最终模型**——用户已明确后续要换成深度学习/大模型，
      模型架构选型仍是军伟的决策权，worker不得擅自升级模型架构。见 `.claude/rules/leader-mode.md`
- [x] 2026-07-12 Volve/F3/Penobscot数据全貌整合为统一权威文档，取代逐个翻查P2.1~P2.4：
      见 `_wiki-methodology/_wiki/_entities/volve-dataset.md`（9大类→14zip映射、24井×数据源
      覆盖矩阵、井名命名陷阱、内容级程序化验证结果、③④⑤赛道数据可行性结论）。
- [x] 2026-07-13 Codex(直连Databricks远端字节级比对)+Claude Workflow(8路并行网络搜索)联合审查：
      **Volve/F3/Penobscot三批数据确认完整无遗漏**(与官方远端逐文件同名同字节数)；同时修正
      registry里4处文字漂移(总量7.7TB→实测4.566TB压缩/5.437TB标称、"9大类"→官方11类文件夹/
      本项目归并9组、F3标注类别数、Penobscot真实num_classes=8而非页面写的7)。产出①②③④⑥赛道
      均有权威公开代码仓库(15个repo均`git ls-remote`验证可访问)，**⑤甜点预测仅有论文级方法，
      无作者公开仓库，且Volve无现成甜点真值**——加固了⑤仍需军伟定义代理标签的既有结论。
      见 `_findings/P2.5_joint_data_and_algorithm_source_audit.md`、
      `_wiki-methodology/_wiki/_entities/algorithm-baselines-6tracks.md`。
- [x] 2026-07-13 军伟决策落地(通过tmux直接下发给③④worker)：③储层物性目标定为PHIF/log1p(KLOGH)/SW
      三输出+地震测井多模态+Hugin储层段+母井家族隔离；④岩相定为9类GENETIC FACIES(非36码LITH)+
      多模态+母井家族隔离；⑤仍未定标签，worker只交付了validate-only审计基础设施(未生成任何标签/
      数据集)。①-⑥六条赛道baseline pipeline全部完成并独立verify通过(独立重跑测试+dataset_io
      stats+哈希核验，非只信worker自述)。
- [x] 2026-07-13 六赛道worktree只读合并就绪审计：**全部MERGE_READY=NO**，核心共性阻塞是6个track
      分支目前都是0 commits(HEAD与master相同)，直接`git merge`会合入零内容；次要问题是集成测试
      无条件依赖被gitignore的产物(需skip guard)、各赛道独立.venv体积6.5-6.9GB级别、部分JSON残留
      worktree绝对路径。见 `_findings/P2.6_six_tracks_merge_readiness_audit.md`。
- [x] 2026-07-13 六赛道可移植性修复：按P2.6审计的REQUIRED_PRE_MERGE_ACTIONS逐条修复，范围严格限定
      README/测试门控/路径序列化/.gitignore(禁止改模型/标签/split/指标)。全部独立verify通过：
      ①fault候选收敛21文件/378KB；②facies候选从6.9GB(.venv)压到521KB/23文件；③reservoir新增
      `--run-integration`显式门控(默认7 passed 2 skipped，完整9 passed)；④lithofacies同样门控
      模式(默认5 passed 6 skipped，integration 11/11)；⑤sweetspot清理绝对路径+扩充.gitignore，
      标签仍是draft未approve；⑥reconstruction候选从~394MB(含_tmp 392MB)压到34文件/1.07MB。
      所有指标/split/模型值均确认未变。**下一步（未执行，需先确认方向）**：各worktree按各自
      INCLUDE清单做选择性`git add`+commit，再评估合并进主仓策略。
- [x] 2026-07-13 六赛道各自完成1个已验收commit(按各自INCLUDE清单选择性提交，非全量`git add -A`)，
      仍未合并进主仓master(各分支ahead=1/behind=0)。真实SHA：
      ①fault `09abe70905f988f4585debbd2216a6a2542e5dfd`；②facies `5f70a33e23227f45bb401d077630ff594d477b14`；
      ③reservoir `ea3b0f35c0dd05173b38be83834f87b82315f875`；④lithofacies `84839c2421143804e7c923c30500cf03eefc7078`；
      ⑤sweetspot `b13d876349c55834abc36550864d6b6f19bda9cd`；⑥reconstruction `1a142f3787b18999123a9750fcd8bea30001d822`。
      **下一步（未执行，需先确认方向）**：待军伟/负责人拍板合并策略后再执行merge进master。
- [x] 2026-07-13 按军伟“先做数个简单模型”的指令，①②③④⑥各新增2个同名文件动态注册的备选模型，
      未改共享`_code/ml_framework`、数据、标签、split、既有baseline或正式指标。新增commit：
      ①fault `db84205163e2954060cf6905fe88fe56ecbc0657`（raw logistic/local Huber，独立复测15/15）；
      ②facies `25edb4097fab36bd10663dda93791c606a510de0`（pixel linear/tiny FCN，独立复测9/9）；
      ③reservoir `d35bbf0b4b85a1cc88149c5515d5f3322c959fb2`（linear/ridge，独立复测7 passed、2 integration skipped）；
      ④lithofacies `86281c6c5002c2302e9b58bacfba169de59c3218`（concat linear/late fusion，独立复测5 passed、6 integration skipped）；
      ⑥reconstruction `7e97f33947dc329acfcddb4e5d2113af20334e21`（linear SGD/tiny MLP，独立复测14/14）。
      每个新模型均位于本赛道`models/<registered_name>.py`并导出`build_model()`，可由既有`get_model`动态发现，
      不需改训练主循环。**边界**：目前固定的是“各赛道内部”的模型输入输出适配；五条赛道的张量/方法签名仍因任务类型
      （分割、分类、回归、重建）不同而异，尚没有跨赛道统一`ModelBatch -> ModelOutput`合同，不能误报为全项目统一I/O已完成。
      **下一步（未执行）**：待军伟确认⑤目标语义，并另行决定是否设计跨赛道统一I/O合同与五个分支的merge顺序。
- [x] 2026-07-13 军伟进一步拍板：⑤甜点采用“一赛道七任务”，在原拟五个目标外独立增加⑥孔隙度、⑦渗透率；
      同时要求六赛道补齐独立测试、合理train/validation/test划分、训练+验证内5-fold、全局随机种子、自动调参、
      loss/输出激活比较、赛道专属可视化及分层自测。当前先执行五窗口**只读调研批**，不改代码、不跑长训练；
      五窗口已通过 Secretary Bus 完成并由负责人逐一verify，leader `junwei-p4-training-research-20260713` 已关闭为completed。
      研究合同见 `_phases/P4_training_validation_system_research.md`，合并调研见
      `_external_reviews/P4_five_track_training_research_20260713.md`，七目标合同见
      `_phases/P4_sweetspot_seven_target_contract.md`，架构决策见
      `_decisions/P4.1_training_validation_reproducibility_architecture.md`，统一SOP见
      `../_wiki/_methods/training-evaluation-reproducibility-contract.md`，实施路线见
      `_phases/P4_implementation_roadmap.md`，最终可复制 `/goal` 见 `_phases/P4_goal_prompt.md`。本轮只做调研与文档，未实施模型/CV/HPO代码。
- [x] 2026-07-25 可视化交付纠错与门禁固化：确认旧 `_outputs/p5_r2_visualization/track_*.png`
      实际来自R2汇总JSON，属于协议覆盖率/预算/性能图，不是领域图；已明确禁止把它们用于领域图卡片渲染。
      新增 `_pipelines/03_domain_visualization_delivery/`，以六赛道白名单锁定真实图的worktree HEAD、
      图片SHA-256、来源脚本、证据文件、PNG尺寸和逐张人工复核；复制后再次验哈希，公开发布必须显式
      `--yes-public`且每个永久URL返回HTTP 200。当前6张真实领域图已通过3/3回归测试并发布，
      证据见 `_outputs/domain_visualization_delivery/v1/published_manifest.json`。科学边界继续保留：
      ①低精度baseline、④known F-5确认、⑤development OOF、⑥conditional development-only。
- [x] 2026-07-25 六赛道真实三维成像与 SCI Plot：逐赛道先审计 `native_volume / spatial_context /
      not_feasible`，只对有真实三维网格、空间坐标或井轨迹的结果生成三维静态出版图和交互 HTML；
      禁止堆叠无序二维样本、任意插值或示意点云冒充三维成果。共同合同见
      `_phases/P5_three_dimensional_sci_visualization_contract.md`，由六个可见赛道窗口各自在隔离写域实施，
      主控逐图、逐 HTML、逐血缘独立验收后再进入卡片渲染。最终判定：
      ①③=`spatial_context`、⑥=`native_volume`，②④⑤=`not_feasible`；可行赛道共交付12张
      2160×2160/300 DPI静态图、12份PDF、4个交互HTML，测试、Chromium拖拽、Gemini视觉二审及
      16个永久URL HTTP 200均通过。六赛道本地验收commit与公共链接见合同“完成记录”。
- [x] 2026-07-30 P12 可视化标准化：按用户要求只处理赛道1/3/5，赛道2/4/6继续暂停。
      三条线均建立固定 `p12_visualization.py`、赛道测试、`manifest.json` 与 `figures/`，共交付
      13张PNG及其PDF/SVG同伴（39个稳定文件）；统一 TNR/TGT 字体、Akun配色、无图内总标题、
      `(a)` panel label、确定性输出和输入/输出哈希，但保留各赛道不同的科学诊断结构与真实负面结果。
      负责人逐张原分辨率验收后由 `step_04_stage_p12_review.py` 写入独立
      `_outputs/domain_visualization_delivery/p12/review_attestation.json`，中央门禁7/7、
      fault 5/5、property 2/2、sweetspot 5/5通过；共享冷启动入口为
      `_pipelines/03_domain_visualization_delivery/step_00_discover.py --check`。
- [x] 2026-07-31 P13 科研深度扩展：六赛道补齐统一基础模型实验合同与真实领域结构诊断，
      包括断层 SEG-Y 解释叠合、F3 连续相切片、井深物性与岩相序列、生产预测残差诊断，
      以及 Volve MAPAXES 三维体、正交切片和方向变差函数。12组静态图、3个交互 HTML、
      输入输出 SHA-256 清单与4项自动测试均已落地；六张架构图使用 Nano Banana 2 绘制，
      但只作为方法解释，不作为性能证据。45页 LaTeX 技术报告按六个固定四字节标题组织，
      详见 `_phases/P13_six_track_research_depth_expansion.md`。
- [x] 2026-07-31 P16.1 基础模型迁移诊断：复核赛道② SAM2、赛道④ MOMENT 与赛道⑥ GFM
      的真实权重加载、训练配方和强基线对照，确认②、④在引入基础模型后的完整方案均取得开发指标
      提升，具体来源留待后续消融。⑥只穷尽了当前 GFM、数据与三种直接桥接协议，下一步优先让
      基础模型提供结构分区、各向异性、边界与不确定性，再与非平稳地质统计联合建模。详见
      `_findings/P16.1_foundation_model_transfer_diagnosis_and_recovery.md` 与
      `../../_reports/foundation_model_diagnosis/20260731_foundation_model_transfer_diagnosis_and_recovery.md`。
- [x] 2026-07-13 P4 Goal 已在隔离 `p4-training-integration` 分支实施：安全集成五赛道候选，
      完成 `_code/ml_framework` 公共合同、`_models/<track>/` 唯一模型真源、全局 seed=2693、
      group/spatial split、加权 reducer、checkpoint/resume、HPO 接口、artifact manifest 与一次性冻结测试门。
      ⑤七目标中 1–4、6、7 已训练并消费冻结测试，5 因缺 Eclipse 状态解析器与冻结时点/候选井/经济约束而
      `not_feasible`；精确 PHIE 也独立 `not_feasible`，没有用 LFP_PHIE 偷换。断层真实数据有 3998 个正例点，
      但缺覆盖已审核负例，因此正式 blind/CV 也诚实 `not_feasible`。可行赛道已补齐真实 smoke；
      F3/Penobscot、岩相、strict/conditional 重建已走通 CV→freeze→refit→single-use test→archived visualization。
      验收证据见 `_wiki-methodology/_tests/P4_acceptance_evidence.md`。**未 merge 回 master，未 push**。
- [x] 2026-07-14 P5 开源模型调研批完成：六个赛道分别联网核验 GitHub、Kaggle、论文官方实现与其他
      开源社区，共审计127个去重候选，筛出92个L2可实测候选，并为每个赛道冻结首批10个模型。
      六份报告均包含精确revision、许可证/权重边界、I/O适配、数据泄漏风险、最小smoke、资源预算和
      fail-closed降级，且由负责人独立verify；报告位于主工作区 `_tmp/model_scout_20260714/`。
      新建干净分支/工作树 `p5-model-benchmark-integration@2d128b0`，统一执行合同见
      `_phases/P5_open_model_benchmark_protocol.md`。**边界**：目前完成的是调研和SOP，不是60个模型已实测；
      master仍有未归属脏改动，因此没有merge/push。
- [x] 2026-07-14 P5 Stage-1 完成并集成验收：六赛道各10个候选均有独立、可审计的contract-smoke
      尝试记录；状态不等于“60个全部成功”。①断层在无审核负例时仅做工程forward/contract检查并停止
      正式排名；②F3与Penobscot各6个smoke通过、4个依赖/来源硬门skip；③9个通过、TabICLv2因未批准
      checkpoint许可skip；④P通道9个通过、S通道因小批次缺连续同井MD样本而1个skip；⑤10模型×7目标
      共70个真实运行格全部因`label_spec`未批准而fail-closed skip，但adapter fixture合同测试通过；⑥strict与
      conditional各8个通过、2个硬门skip。六个赛道提交按①→⑥顺序集成；随后修复测试文件同名和裸模块名
      在联合收集时的跨赛道碰撞。`torch-common`全量联合验收为53 passed、6 skipped、77 subtests passed；
      `tabular-cpu`联合验收为31 passed、2 skipped、20 subtests passed。frozen test未被访问，master未merge，
      未push。完整验收证据见 `_wiki-methodology/_tests/P5_stage1_acceptance_evidence.md`。下一步只对科学可行
      候选执行Stage-2固定development pilot；⑤先补真实标签合同，①先补审核负例。
- [x] 2026-07-14 P5 Stage-2完成并集成验收：固定预算与停止线见
      `_phases/P5_stage2_fixed_budget_pilot.md`。六个隔离工作树均从Stage-1集成验收提交`85727fd`创建；
      同赛道/同lane固定development fold、样本/更新/墙钟预算，全局seed=2693，GPU通过排他锁串行；
      ①只做负例/unknown data-gate，⑤先把P4七目标registry审计映射为P5 label-spec，其他可行候选执行pilot。
      六轨提交按①→⑥顺序集成到`d46a7b5`：①10个blocked/not_rankable；②F3/Penobscot各6 pilot+4 skip；
      ③9 pilot+1许可证skip（表格与3D模态分榜）；④9 pilot+1 S通道skip；⑤70 cell中16 pilot+54 skip，
      T6 PHIF/T7 KLOGH因缺development-only特征源阻断；⑥strict/conditional各8 pilot+2 skip。合计140 cell、
      53 pilot、87结构化停止、0 failed/timeout。`torch-common` Stage-2联合门59 passed+22 subtests，
      `tabular-cpu`适用四轨40 passed；frozen test继续封锁，未merge master，未push。完整证据见
      `_wiki-methodology/_tests/P5_stage2_acceptance_evidence.md`。
- [x] 2026-07-14 P5 Stage-3准入合同冻结：严格从Stage-2同任务/同lane榜单锁定top-3，三个重复seed为
      `1867973658/2137841944/3902865753`，使用P4全部科学有效development folds；不同数据集、目标、模态与
      strict/conditional继续分榜。断层、物性3D单候选、甜点T5/T6/T7不强凑准入。统一预算、cell矩阵、
      OOF专属可视化和test firewall见`_phases/P5_stage3_multiseed_cv.md`；frozen test仍封锁。
- [x] 2026-07-14 P5 Stage-3执行与集成验收完成：①断层零合法fold、只交data-gate图；②F3/Penobscot
      90/90 cell；③PHIF/KLOGH/SW 108/108；④GM09 P通道33/36，三次CatBoost NaN/Inf失败保留；
      ⑤T1–T4 117/117，T5–T7继续诚实停止；⑥strict/conditional 89/90，一次PyKrige 300秒超时保留。
      合计441个可运行cell、437 pass、3 fail、1 timeout；各赛道OOF manifest、专属图、bootstrap CI、
      worst-fold、seed稳定性、资源和test firewall均独立核验。集成HEAD为`9e5f501`，完整证据见
      `_wiki-methodology/_tests/P5_stage3_acceptance_evidence.md`。master仍脏，未merge、未push。
- [x] 2026-07-14 P5 Stage-4执行与集成验收完成：严格冻结Stage-3唯一胜者、配置、预算和seed，在全部合法
      development上refit；五个可运行赛道对历史已见holdout生成真实指标、预测、哈希manifest和赛道专属图，
      ①断层继续`blocked/not_rankable`，⑤T5`not_feasible`、T6/T7`blocked`。六轨提交按①→⑥集成到
      `5af968c`；集成后`178 passed`、`6 subtests passed`、`1 skipped`。所有结果明确为
      `previously_seen_reusable_holdout`、`prior_test_consumed=true`、`fresh_blind=false`，不能称首次盲测。
      完整指标、图件目检和facies单次访问恢复记录见`_wiki-methodology/_tests/P5_stage4_acceptance_evidence.md`。
      master仍脏，未merge、未push；下一步必须获取新的外部/赛事隐藏测试集，或先补齐被阻塞的数据合同。
- [x] 2026-07-14 P5 六赛道直接基准协议复核完成：六个赛道窗口只纳入同物理任务、同输入/标签语义的
      原始论文、官方竞赛/数据页和作者仓库，分别形成13/10/10/10/14/10项直接证据报告；六份提交均由
      负责人核对提交范围、工作树、来源可达性和科学边界后verify。统一结论见
      `_decisions/P5.1_direct_benchmark_protocol_revision.md`：现有Pipeline工程骨架保留，但短预算pilot不得
      冒充能力榜，已见holdout不得冒充fresh blind；断层先补审计负例，地震相补full-volume与学习曲线，
      物性/岩相/甜点/重建按任务与模态分lane并修正各自split/标签/成对对照。本批只做报告和决策，
      尚未引入六份报告提交、未改训练代码、未merge master、未push。
- [x] 2026-07-14 P5.1 R0/R1实施与集成验收完成：六赛道从共同基线`d1bf52a`建立干净实现提交，按
      ①断层→②地震相→③物性→④岩相→⑤甜点→⑥重建顺序集成到`p5-r01-integration@5d4a917`。
      六个worker均已独立collect/verify；跨赛道联合回归290 passed、6个可解释数据门skip。首次联合跑的
      facies失败复现为根分区`/tmp`仅余约71MB导致第二个57.4MB临时checkpoint写满；把`TMPDIR`切到
      `/mnt/data`后原测试通过，未改算法或产物。全局seed=2693，frozen/physical test与known-holdout指标
      均未访问，所有输出仍是protocol mechanism evidence而非性能榜。验收见
      `_wiki-methodology/_tests/P5.1_r01_integration_acceptance_evidence.md`。下一步统一称P5.2/protocol R2学习曲线；达到停止线后
      才启动P5.3/protocol R3每合法赛道/lane至少10候选正式grouped/spatial/temporal CV。master未merge、未push。

- [x] 2026-07-31 赛道⑥ P20 完成非零初始化、LoRA r4、Adapter 和分阶段解冻的严格五折复测。
      四路线均为 5/5 折优于 PyKrige，最佳 staged-LoRA RMSE=`0.027789615700`，但仍差于 P19 的
      `0.027751397628`；80 步扩展降为 `0.027791517166`，P19/P20 误差相关=`0.9992037`，固定融合
      最优 P20 权重为 0。梯度、参数更新、原生窗口和 holdout firewall 均独立复核通过，结论为
      `VERIFIED_NO_PROMOTION`、默认关闭。见 `_findings/P20_reconstruction_peft_staged_unfreeze.md`。
- [x] 2026-08-01 赛道⑥ P21 完成多视图无标签 LoRA 与交叉拟合残差复测。三条神经残差路线均未
      超过 P19；固定 `z4_f0.1_s{0,0.1,0.2}_k64_p1.5_b0.75` foundation 三核平均达到
      RMSE=`0.027734374378`，相对 P19 为 1 胜/4 平/0 负，验收为
      `ACCEPTED_SIMPLICITY_WIN`。见 `_findings/P21_reconstruction_contrastive_residual.md`。
- [x] 2026-08-01 赛道⑥ P24 完成预注册的同场区历史属性版本迁移测试。冻结 P21 相对重新拟合的
      PyKrige 将 RMSE 从 `0.028235410003` 降至 `0.027825182663`（`1.4529%`），4/5 折改善；
      开目标后未调参。证据等级不等于 fresh blind/cross-field；用户已明确本轮不再追加跨场区测试。
      见 `_findings/P24_reconstruction_historical_transfer.md`。
- [x] 2026-08-01 P27 完成技术报告 1.4 版终审。修正“三维重建仍以普通克里金最优”的过期表述，
      同步 P21 固定基础核的开发折和同场区历史版本结果；重绘六赛道科研总览并修复缺失表头，
      核对六个赛道各六节目录。45 页 PDF 无未定义引用、无越界盒子，研究可视化测试 4/4 通过；
      本轮不追加跨数据集测试。证据见 `_wiki-methodology/_tests/P27_report_v14_audit_and_delivery_evidence.md`。

## 数据资产索引

详见 `_meta/_data_registry.yml`（单一真源，本文件不重复数据字段细节）。

## Pipeline 骨架

详见 `_pipelines/_readme.md`（元思维层：5阶段骨架 + 接口关系；不含具体算法/模型选型——由军伟后续用最新AI论文模型替换填充）。

## 决策入口

- 关键决策放 `_decisions/`；这里只留当前阶段必须看的链接。
- P5六赛道训练/测试协议统一裁决：`_decisions/P5.1_direct_benchmark_protocol_revision.md`。

## 证据入口

- 结论 / 类层模式放 `_findings/`。
- 操作过程放 `_logs/`。
- 外援意见放 `_external_reviews/`。
