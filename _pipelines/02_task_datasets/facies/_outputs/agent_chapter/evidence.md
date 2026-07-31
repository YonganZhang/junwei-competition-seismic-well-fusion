# ②地震相分类：智能体分析章节

## 证据上下文

本章以已提交的 P13 cross-attention 固定 development 结果为模型上下文，同时保留 small-CNN baseline 和 continued-CNN control，避免把额外训练收益误写成大模型贡献。

| 数据集 | development patch | inline组 | 类别数 | fold 0/4 最大类÷最小正类 | 固定实验样本/折 |
|---|---:|---:|---:|---:|---:|
| F3 | 1548 | 387 | 10 | 82.67/72.29 | 32 train / 16 validation |
| Penobscot | 1398 | 357 | 8 | 54.48/58.73 | 32 train / 16 validation |

P13 mIoU：F3 `0.136263 baseline → 0.206862 continued CNN → 0.292153 cross-attention`；Penobscot `0.129101 → 0.189564 → 0.205270`。P13 的 fusion scale 从 0.2 初始化后仅到 F3 0.200461、Penobscot 0.200573，attention entropy 分别为 0.891912、0.951056。

## 发给 DeepSeek 的结构化 prompt

请求模型：`deepseek-chat`；服务端返回模型：`deepseek-v4-flash`。

### System

~~~text
你是一名独立的地震解释与小样本语义分割顾问。只能基于给出的development证据做常识性诊断；区分已观察事实、技术假设与待验证建议；禁止声称SAM2造成了提升，因为随机权重消融尚未执行。建议须尽量给出可执行参数，并标注LOW_COST/MEDIUM/HIGH成本。
~~~

### User

~~~text
# 任务
②地震相分类赛道：输入单通道128×128地震振幅patch，逐像素预测facies。F3为10类，Penobscot为8类。主指标是固定实现的mIoU，另看macro-F1/accuracy/NLL；类别权重只由各fold训练集拟合。

# 数据与评测边界
- F3 development：1548个patch、387个inline组；Penobscot development：1398个patch、357个inline组。
- 锁定空间隔离5折中的fold 0和4；实际固定开发预算每折仅32个训练patch、16个验证patch，batch=2。不得使用frozen holdout或test.h5。
- F3 full fold-train像素占比在fold 0/4中约为：[0.26/0.31, 1.06/1.21, 10.38/10.74, 6.71/6.89, 11.64/12.48, 21.58/22.36, 19.50/17.10, 11.46/10.03, 13.33/13.47, 4.08/5.40]%，最大类/最小正类约82.67/72.29倍。
- Penobscot full fold-train像素占比约：[1.10/1.03, 4.75/4.47, 7.67/7.21, 1.45/1.37, 1.71/1.71, 14.96/15.12, 8.26/8.65, 60.09/60.44]%，最大类/最小类约54.48/58.73倍。

# 当前模型（P13）
- strong_small_baseline：40 updates的小CNN。
- continued_cnn_control：从同一baseline继续160 updates，只训练CNN decoder/head；AdamW，CNN lr=5e-5，weight_decay=1e-4，cosine，CE+0.25 Dice，水平翻转p=0.5、强度缩放和Gaussian noise std=0.03。
- cross_attention_fusion：CNN最深层特征作query，SAM2 native-128特征作key/value，4 heads、dim=128，feature residual写回CNN decoder；SAM2最后Hiera blocks 22/23解冻。lr：CNN 5e-5、fusion 2e-4、SAM2 1e-5；其余训练口径同continued control。当前用pretrained权重，但贡献占比待random-weight消融。

# 固定development结果（fold 0/4等权均值）
- F3：baseline mIoU 0.136263，continued CNN 0.206862，cross-attention 0.292153；cross比continued +0.085291。cross macro-F1 0.416360。
- Penobscot：baseline 0.129101，continued CNN 0.189564，cross-attention 0.205270；cross比continued +0.015707。cross macro-F1 0.282640。
- 两任务等权：baseline 0.132682 → cross 0.248712（整体+0.116029），但不能归因于SAM2。

# 已知限制/诊断
- fusion_scale=sigmoid(logit)，初始化0.2；训练后F3均值0.200461、Penobscot 0.200573，几乎钳在初始化附近，门控学习可能不充分。
- 归一化attention entropy偏高：F3 0.891912、Penobscot 0.951056，尤其Penobscot注意力接近均匀。
- 训练样本极少、类别极不均衡；P13只跑两个固定fold，尚无随机SAM2消融，不能做来源归因。

# 请分析并回答
1. 先列3–6条最可能的技术瓶颈，逐条说明依据，不要把相关性写成因果。
2. 给出按优先级排序的参数调整、特征设计、训练策略建议；每条标注[LOW_COST]/[MEDIUM]/[HIGH]、精确改动值或范围、预期观察信号与失败判据。
3. 至少给出一个可在上述相同development口径下立刻验证、只改学习率/正则/损失权重之一的低成本实验，最好针对fusion_scale不动或高attention entropy；一次只改一个主要因素。
4. 明确列出“未验证建议”，并提醒大模型贡献需后续pretrained/random对照确认。
5. 不得给出虚构提升数字；可以给假设方向，但不要承诺提升。
~~~

## DeepSeek 原始分析文本

# 地震相分类P13模型诊断与建议

## 1. 最可能的技术瓶颈（按可能性排序）

### 瓶颈1：门控机制失效——fusion_scale钳制在初始化值
**依据（观察事实）**：训练后fusion_scale均值F3为0.200461、Penobscot为0.200573，与初始化0.2几乎无差异。这表明sigmoid(logit)门控在训练中几乎没有更新，融合路径可能被有效关闭或仅以固定比例叠加。

**技术假设**：门控梯度可能因以下原因消失：(a) logit初始化值对应sigmoid输出0.2，若logit初始为负较大值，梯度饱和；(b) 融合特征与CNN decoder特征尺度不匹配，导致门控梯度被主路径梯度淹没；(c) 学习率1e-5对SAM2侧过低，但fusion层本身lr=2e-4，若门控参数在fusion层内，则更可能是初始化或梯度流问题。

**注意**：这是相关性观察，不能断定门控失效是性能瓶颈的唯一原因，但门控值几乎不动是明确的异常信号。

---

### 瓶颈2：注意力熵过高——Penobscot注意力接近均匀分布
**依据（观察事实）**：归一化attention entropy F3为0.891912、Penobscot为0.951056（接近最大值1.0）。Penobscot的注意力几乎均匀，意味着cross-attention没有有效聚焦到判别性空间位置。

**技术假设**：可能原因包括：(a) key/value特征（SAM2 Hiera blocks 22/23）与query特征（CNN最深特征）域差异过大，导致相似度分布平坦；(b) 训练样本极少（每折32 patch），注意力机制难以学到锐利分布；(c) 4 heads、dim=128的配置可能容量不足或过大，导致退化到均匀分布。

**注意**：高熵是观察事实，但“高熵导致性能受限”是假设——需要验证降低熵是否带来提升。

---

### 瓶颈3：训练样本极度稀缺与类别极端不均衡
**依据（观察事实）**：每折仅32个训练patch（batch=2，即16次更新/epoch），F3最大/最小正类比约82.67倍，Penobscot约54.48倍。P13的continued CNN仅160 updates，cross-attention模型总训练量也极小。

**技术假设**：在如此少的数据下，模型可能欠拟合（而非过拟合），尤其对尾部类别。CE+0.25 Dice的损失组合可能不足以应对极端不均衡——Dice权重偏低，且Dice对极小类梯度不稳定。

**注意**：样本少是事实，但“欠拟合”是假设——需观察训练/验证曲线确认。

---

### 瓶颈4：SAM2特征贡献未验证——可能仅提供有限增量
**依据（观察事实）**：cross-attention相比continued CNN在F3提升+0.085291，Penobscot仅+0.015707。Penobscot提升微弱，且attention熵接近均匀，提示SAM2特征可能未提供有效判别信息。

**技术假设**：SAM2的Hiera特征可能对地震相判别不够敏感（预训练于自然图像/视频），或blocks 22/23特征过于高层、空间分辨率不足。Penobscot提升微弱可能反映特征不匹配。

**注意**：这是假设，必须通过random-weight消融验证。当前不能归因于SAM2。

---

### 瓶颈5：训练策略可能不匹配——cosine调度与极短训练周期
**依据（观察事实）**：continued CNN仅160 updates，cross-attention模型总训练量未明确但推测类似量级。cosine调度在极短周期内可能未充分降低学习率，导致后期震荡。

**技术假设**：在<200 updates的规模下，cosine可能不如固定小学习率或线性衰减稳定。但这是推测，无直接证据。

---

## 2. 按优先级排序的建议

### 建议A：修正fusion_scale初始化与门控梯度流 [LOW_COST]
- **改动**：将fusion_scale的sigmoid logit初始化从对应0.2改为对应0.5（即logit=0），或直接移除门控（固定为1.0）作为对照。
- **精确参数**：若保留门控，初始化logit=0.0（sigmoid输出0.5）；若移除，删除fusion_scale乘法，直接相加。
- **预期观察信号**：训练后fusion_scale均值应偏离0.5（若保留门控），或模型mIoU变化。
- **失败判据**：若fusion_scale仍钳制在0.5附近且mIoU无变化，说明门控不是瓶颈，应转向其他因素。

---

### 建议B：降低attention熵——温度缩放或更少heads [LOW_COST]
- **改动**：在attention softmax前除以温度T，初始T=0.5（即logits除以0.5，使分布更锐利）；或将heads从4减至2。
- **精确参数**：T=0.5，或heads=2，dim保持128。
- **预期观察信号**：attention entropy应下降（例如Penobscot从0.95降至<0.8），mIoU可能变化。
- **失败判据**：若entropy下降但mIoU无提升或下降，说明均匀注意力并非主要瓶颈。

---

### 建议C：调整损失权重——提高Dice权重或改用Focal [MEDIUM]
- **改动**：将CE+0.25 Dice改为CE+0.5 Dice，或替换为CE+0.5 Focal（gamma=2.0）。
- **精确参数**：Dice权重0.5；若用Focal，alpha按类别频率倒数设置，gamma=2.0。
- **预期观察信号**：尾部类别（如F3的class 0/1）的IoU应提升，整体mIoU可能变化。
- **失败判据**：若尾部类别IoU无改善且整体mIoU下降，说明损失权重不是主要瓶颈。

---

### 建议D：延长训练周期并调整调度 [MEDIUM]
- **改动**：将总updates从~200增至400-600，cosine周期相应延长；或改用固定lr=1e-4（CNN）无调度。
- **精确参数**：总updates=400，cosine T_max=400；或固定lr=1e-4，无decay。
- **预期观察信号**：训练loss应持续下降，验证mIoU应随训练上升（若欠拟合）。
- **失败判据**：若验证mIoU在200 updates后不再上升，说明延长训练无益，可能已过拟合。

---

### 建议E：特征对齐——对SAM2特征做投影或归一化 [MEDIUM]
- **改动**：在cross-attention前对SAM2 key/value特征加一层LayerNorm+线性投影（dim=128），或对CNN query特征做L2归一化。
- **精确参数**：key/value投影层：Linear(128→128)+LayerNorm；query L2归一化。
- **预期观察信号**：attention entropy应下降，Penobscot mIoU可能提升。
- **失败判据**：若entropy无变化且mIoU无提升，说明域差异不是主因。

---

### 建议F：数据增强强化——针对空间不变性 [LOW_COST]
- **改动**：在现有增强基础上增加随机旋转90°（概率0.5）和垂直翻转（概率0.5）。
- **精确参数**：rotation 90° p=0.5，vertical flip p=0.5。
- **预期观察信号**：训练loss下降更平稳，验证mIoU可能提升（若增强有效）。
- **失败判据**：若mIoU无变化或下降，说明增强过度或与任务不匹配。

---

## 3. 立即可验证的低成本实验（一次只改一个因素）

**实验：调整fusion_scale初始化（针对门控不动问题）**

- **改动**：将fusion_scale的logit初始化从当前值改为0.0（sigmoid输出0.5），其余完全不变（lr、损失、增强、训练周期均同P13）。
- **成本**：[LOW_COST]——仅改一行初始化代码，训练时间同P13。
- **预期观察信号**：
  - 训练后fusion_scale均值应偏离0.5（若门控可学习），或仍钳制在0.5附近（若梯度问题依旧）。
  - 若mIoU相比P13的cross-attention结果（F3 0.292153, Penobscot 0.205270）有变化，说明门控初始化是瓶颈之一。
- **失败判据**：若fusion_scale仍不动且mIoU无变化，则门控初始化不是主因，应转向建议B或E。

---

## 4. 未验证建议与归因提醒

**未验证建议**（需后续实验确认）：
- 温度缩放降低attention熵（建议B）——未测试，可能有效也可能无效。
- 损失权重调整（建议C）——未测试，对尾部类别影响未知。
- 延长训练周期（建议D）——未测试，可能欠拟合也可能过拟合。
- 特征投影对齐（建议E）——未测试，域差异假设未验证。
- 增强强化（建议F）——未测试，可能引入噪声。

**归因提醒**：
- **当前所有cross-attention相比continued CNN的提升（F3 +0.085291, Penobscot +0.015707）不能归因于SAM2**，因为未执行pretrained/random权重消融。
- 必须运行random-weight SAM2（冻结或解冻）对照实验，才能区分提升来自SAM2预训练特征还是融合架构本身。
- 在消融完成前，任何关于“SAM2带来提升”的表述均为未验证假设。

---

## 5. 假设方向（不承诺提升）

- 若门控初始化修正后mIoU提升，则门控是瓶颈之一。
- 若温度缩放降低熵后Penobscot mIoU提升，则注意力均匀性限制性能。
- 若Dice权重提高后尾部类别IoU改善，则损失权重需调整。
- 若random-weight SAM2消融显示与pretrained结果无显著差异，则SAM2预训练贡献可忽略，应聚焦融合架构本身。

以上所有方向均为假设，需实验验证，不提供虚构数字。

## 低成本 development 验证

DeepSeek 提议把 `fusion_scale` 初值由 0.2 改为 0.5。P13 既有的全参数 `max_norm=1.0` 梯度裁剪保持不变，因此本次新增的唯一主要变量是门控初值 `0.2 → 0.5`；学习率、权重衰减、损失、增强、更新数、SAM2 解冻层、样本、fold 和 seed 均保持不变。考虑到 CUDA 复跑有非严格确定性，每个 cell 都从同一个本次重训 baseline 分叉并重新训练 gate=0.2 control 与 gate=0.5 repair；P13 同 cell 数字只作历史参照。

| 数据集 | fold | P13参照 | 新鲜0.2 control | 0.5 repair | Δ | 后 fusion scale | 后 attention entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| F3 | 0 | 0.273459 | 0.265953 | 0.293468 | +0.027515 | 0.500228 | 0.885546 |
| F3 | 4 | 0.310847 | 0.306570 | 0.367532 | +0.060962 | 0.499077 | 0.925887 |
| Penobscot | 0 | 0.192684 | 0.192062 | 0.195584 | +0.003522 | 0.500174 | 0.913457 |
| Penobscot | 4 | 0.217857 | 0.234513 | 0.226004 | -0.008510 | 0.499436 | 0.952694 |

| 数据集 | 前均值 mIoU | 后均值 mIoU | Δ | 前/后 macro-F1 | 前/后 attention entropy |
|---|---:|---:|---:|---:|---:|
| F3 | 0.286262 | 0.330500 | +0.044238 | 0.408187/0.454899 | 0.891023/0.905716 |
| Penobscot | 0.213288 | 0.210794 | -0.002494 | 0.291376/0.291955 | 0.942898/0.933075 |

两任务等权 mean mIoU 从 `0.249775` 变为 `0.270647` (`+0.020872`)。按本 runner 预注册的 `> 0.005` materiality 阈值并同时检查任务方向一致性，本次判定为 `MIXED_TASK_RESULT`。该判定只覆盖这两个固定 development folds，不外推到 frozen holdout。
F3 为 `+0.044238`，Penobscot 为 `-0.002494`，方向不一致；两任务最终 fusion scale 也仍紧贴新的 0.5 初值。因此实验只证明固定融合强度变化会改变指标，未证明门控已经学会自适应。

复跑 baseline 相对 P13 历史 baseline 的单 cell 漂移已记录在 `low_cost_results.jsonl`；正式 Δ 始终由同次新鲜 gate=0.2 control 与 gate=0.5 repair 相减，不混用历史运行。
历史 P13 harness 只固定随机种子、未启用严格确定性 CUDA 算法，因此不同进程的绝对分数会漂移；本次成对运行额外请求 deterministic algorithms 并关闭 cuDNN benchmark，两支设置相同。本章不把历史 P13 与本次运行的差异解释成超参数效果。

## 未验证建议

以下 DeepSeek 建议本轮未验证，原文已完整保留在上文：

- key/value LayerNorm 与可学习 attention temperature：未验证。
- warmup、缩短 cosine 周期或 Penobscot 单独调 fusion LR：未验证。
- Dice 权重 0.1/0.5、Focal loss：未验证。
- 改 attention dim/heads、融合层位或多尺度特征：未验证。
- 更大规模预训练或更多 development 样本：未验证。

**大模型贡献占比仍需下一轮 pretrained/random 权重对照确认。** 本章只报告整体数值变化，不作 SAM2 因果归因。

## 数据与密钥边界

- DeepSeek 密钥由调用进程的临时环境注入，未写入任何产物。
- 只读取锁定 split manifest、development `train.h5` 和既有 P13 证据；未读取 frozen holdout/`test.h5`。
- mIoU 仍调用 P11/P13 的同一 probability evaluator，fold 0/4 和每折 32/16 样本不变。
- P11、P12、P13 产物在运行前后以 manifest SHA-256 保护，未删除或覆盖。
