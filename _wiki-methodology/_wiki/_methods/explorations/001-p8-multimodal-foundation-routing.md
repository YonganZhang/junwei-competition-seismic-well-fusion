# P8 六赛道多模态基础模型路由探索

日期：2026-07-28

## 问题

旧 P6 “Gaia/DAGT”主要是合同、哈希和 QC 证据打包，`no_api_calls=true`、`no_training=true`，
不能称为大模型已经参与预测。用户要求每个赛道都接入与数据模态匹配的开源基础模型，并检查接口、
提示模板和整个流程是否科学正确。

## 核心判断

“每个赛道用大模型”不等于“把同一个聊天 LLM 塞进所有赛道”。预测主干必须匹配物理轴：

| 赛道 | 预测主干 | 条件类型 | 当前证据 |
|---|---|---|---|
| 断层 | SAM-Med3D turbo | 无标签派生的 3D spatial prompt | 权重+真实前向通过；真实任务仍被审核负例和连续 3D block 阻塞 |
| 地震相 | SAM 2.1 Hiera B+ 编码器 + 闭集语义头 | 2D spatial prompt=`none` | 10 类语义输出真实前向通过 |
| 储层物性 | TabICLv2 regressor | group-isolated support set | 三目标 synthetic fit/predict 通过 |
| 岩相 | MOMENT-1-base + 9 类头 | measured-depth window | 真实预训练权重前向通过；分类头必须在 development 内微调 |
| 甜点产能 | Chronos-2 | exact calendar time window | 4 折真实 development 诊断通过 |
| 三维重建 | OpenMind ResEnc-L MAE + 连续回归解码 | masked 3D volume | 真实预训练编码器前向通过 |

六条权重都固定 repository revision、source revision、字节数和 SHA-256，运行时禁止自动下载。
SAM 3D Objects 被排除：它是单张 RGB 物体三维生成，不是地震体语义分割/连续属性重建。

## 统一条件模板

统一的是 envelope，而不是把所有输入转成自然语言：

- `time_window`：严格日历时间、历史截止点和预测长度；
- `support_set`：上下文井组与查询井组必须隔离；
- `depth_window`：真实 MD 坐标、中心点、长度和曲线名；
- `spatial_prompt`：提示来源、坐标系和 split role；验证/推理禁止 GT 派生点；
- `masked_volume`：K/J/I 轴、spacing、active mask 和可见观测；
- `language_prompt`：只给监督 QC agent，不进入样本级预测张量。

自然语言 QC 模板 `gaia.foundation.supervisory-qc.v1` 只检查 schema、单位、轴、mask、split、
条件来源、运行完整性、许可证、fallback 和晋级门。它禁止接收标签、raw prediction、测试指标、
文件路径、凭证和 post-cutoff 值；返回必须匹配严格 JSON schema。

## 晋级状态机

1. `CONNECTED_UNVERIFIED`：权重与接口真实可运行，但还不能替换默认模型；
2. `VERIFIED_NO_GAIN`：同 split 验证未证明提升；
3. `PROMOTED_DEV`：同 split 基线、random-init 同架构、shuffle/causal 控制和最少胜出折通过；
4. `CONFIRMED_HOLDOUT`：配置冻结后才允许一次外部/holdout 确认。

当前六条全部停在 `CONNECTED_UNVERIFIED`，`default_enabled=false`。这是为了防止“接上了”
被误写成“提升已经成立”。

## 真实运行发现

- Chronos 旧 P7 用的是 30 个观测行，不是严格 30 个日历日。P8 已改为缺日显式重建且不以零填补，
  30 天历史预测未来 30 天。
- Chronos 4 折 development 宏平均 MAE：历史均值 `184.6686`，Chronos `172.3162`，
  train-only blend `166.3343`；因同网格树模型与随机/因果扰动控制未完成，未晋级。
- OpenMind 最小边长 32 经五次下采样会退化到 `1×1×1` 并使 InstanceNorm3d 失败；
  适配器现补齐至至少 64 再裁回原大小。
- MOMENT 默认 mask 原为 Long tensor，nearest interpolate 在当前 Torch 不支持；现先转 float，
  插值后再转 Long。
- TabICL 固定 source commit 的实际 package version 是 `2.1.1`，旧 source lock 误写 `2.0.0`；
  已按 commit 真值修正，并限制合法 offload 配置。

## 仍然阻塞

- 断层：现有正例不是完整体 mask，缺审核负例和至少两个合法连续 3D development block；
  SAM-Med3D checkpoint 条款还需独立许可证确认。
- 其他非时序赛道：已过权重和前向门，但尚未完成相同 split、相同数据预算的 pretrained /
  random-init / 传统强基线对照。
- 监督语言模型：统一调用边界已实现并用 stub 验证；没有批准的 provider/model/revision 时不发 API，
  不能声称线上 LLM 已调用。

## 结论

接受“模态匹配基础模型 + 统一条件 envelope + 独立监督 LLM QC”的架构。拒绝一个聊天 LLM
直接替代所有数值预测，也拒绝把连接成功当作性能晋级。
