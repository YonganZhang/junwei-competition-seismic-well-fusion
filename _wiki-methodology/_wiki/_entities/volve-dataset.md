# Volve数据集 — 完整档案

> 更新: 2026-07-13
> 定位：这是查Volve数据"到底有什么、在哪、能不能用"的**权威入口**。原始细节调查过程见 `_wiki-methodology/_top/_findings/P2.1~P2.5`，本文件是那几份finding的整合结论，供后续开发直接查阅，不用再翻散落的finding。
> 数据本体位置：`_sandbox/volve_data/`；数据资产登记：`_meta/_data_registry.yml` id=`volve-north-sea`
> 算法源头见姊妹文档：`_wiki-methodology/_wiki/_entities/algorithm-baselines-6tracks.md`

## 一、数据集身份

- **来源**：Equinor官方开放数据（挪威北海Volve油田，2008-2016生产，2016退役后公开）
- **官方渠道**：Databricks Marketplace「Equinor ASA_Volve Data Village」（旧`data.equinor.com`门户已下线，Equinor自己迁移到这里，内容同源，不是裁剪版）
- **许可证**：Equinor Open Data Licence（类CC BY，可复用署名，禁止转售；不是CC BY-NC-SA）
- **规模**：14个zip，压缩后实测**4.566TB**(`du -sh`=4.2T)，`unzip -l`汇总的包内标称未压缩大小约**5.437TB**，官方口径约40,000文件（2026-07-13 Codex审查更正：此前记录的"~7.7TB"是历史遗留粗估，两个真实数字都不是7.7TB）
- **历史钻井数**：**24口井**（这是所有"井×数据源"覆盖判断的基准数字）

## 二、官方11个文件夹类别（业务归并9组）→ 我们的14个zip（完整性核查结论：无遗漏）

> 2026-07-13更正：Equinor Open Data Licence原始条款(Extract 1)列出的是**11个官方文件夹类别**（Seismic细分为ST0202/ST10010/4D/VSP共4项），此前"9大类"是本项目自己的业务归并口径，不是官方原始类目数，两种说法都对但要分清楚层级。下表按业务归并口径展示：

| 官方类别 | 对应zip | 大小 | 状态 |
|---|---|---|---|
| 生产数据 | Volve_Production_data.zip | 2MB | ✅完整 |
| 测井数据 | Volve_Well_logs.zip（按类型分14子目录）+ Volve_Well_logs_pr_WELL.zip（按24井分目录） | 7GB+7GB | ✅完整，内容级验证见下 |
| 储层建模 | Volve_Reservoir_Model-Eclipse_model.zip + -RMS_model.zip | 391MB+2.2GB | ✅完整 |
| GeoScience OW Archive | Volve_GeoScience_OW_Archive.zip（5个顶层条目：license/VOLVE_PUBLIC.bck/Externalfiles.dszip/Seismic.dszip） | 58.7GB | ✅完整（此前发现截断827MB已修复） |
| 地球物理解释 | Volve_Geophysical_Interpretations.zip（含Well_picks/Well_perforations/Horizons三套Depth-TWT-Timelapse层位） | 100MB | ✅完整 |
| 报告 | Volve_Reports.zip（只有2份报告+license：Discovery report + Volve PUD，官方原样如此，不是渠道精简；日常作业文档在Well_technical_data/WITSML里） | 162MB | ✅完整 |
| 井筒技术数据 | Volve_Well_technical_data.zip | 212MB | ✅完整 |
| 地震 | Volve_Seismic_VSP.zip + ST0202.zip(1.17TB) + ST10010.zip(2.59TB) + 4D.zip(330GB)（各自内部含主体Stacks + Velocities速度模型 + Raw_data + Prestack_data + Documentation + Other_data） | ~4.2TB | ✅完整，含辅助数据非仅主体地震体 |
| 钻井数据(WITSML) | Volve_WITSML Realtime drilling data.zip（约18,814个XML文件，26个顶层目录，仅覆盖F系列生产井，不含19号发现井群） | 2.3GB | ✅完整（数量与公开Azure枚举记录20,087精确吻合） |

**结论**：9类对9类，没有第10类缺失。总体积差异（宣传"5TB" vs 实测4.08TB压缩后）是口径问题，不是缺数据。

## 三、24口井 × 数据源覆盖（避坑用）

**核心事实：24口井在Well_logs / Well_logs_pr_WELL / Well_technical_data / LAS测井格式上全覆盖，无遗漏井。**

例外/需注意的子集覆盖：
- **RMS模型的`wells/`目录缺4口井**：F-1C、F-7、F-9、F-9A（这4口井在原始测井数据里是完整的，只是RMS建模时没导入，不是数据缺失）
- **WITSML只覆盖F系列生产井（约17口）**，完全不含19号发现井群（19A/19BT2/19S&SR），因为早期发现井没有WITSML标准
- **岩心(core)数据只在发现井群**：19A、19BT2、19S&SR三个井眼都有（`Well_logs.zip/09.CORE/`），其中19A、19BT2在RMS模型里还有孔渗塞子数值表(`coreplugdata19a`/`coreplugdata19bt2`)；F系列生产井完全没有岩心数据。**之前误判"只有1口井有岩心"是错的，实际是3个井眼**，具体样本总数待精确统计（此前粗估30-47个，已知不完整）

详细矩阵见 `_sandbox/volve_data/_full_inventory/well_zip_matrix.tsv`

## 四、井名命名不统一（查数据前必读，否则容易漏看）

同一口物理井在不同zip里可能写成完全不同的字符串，简单grep一种格式会漏收：

| 陷阱 | 示例 | 出现位置 |
|---|---|---|
| URL编码斜杠 | `15_$47$_9-F-15A` 实际是 `15/9-F-15A` | WITSML |
| 无分隔符粘连 | `15919a`、`159f11t2` | RMS_model |
| 同井多机构前缀重复 | 同一口井在`NA-NA-`、`Norway-StatoilHydro-`、`Norway-Statoil-`三个前缀下各建一份目录 | WITSML |
| 分支合并写法 | `15_9-19 B&BT2`（用`&`把主井眼和侧钻合并命名） | Well_logs_pr_WELL |
| 未解疑点 | WITSML出现`F-15S`编号，不在官方24口名单（只有F-15/A/B/C/D）里，未强行归并 | WITSML |

## 五、内容级验证结果（不是抽查，是真实程序化验证）

| 格式 | 总数 | 验证方式 | 结果 |
|---|---|---|---|
| DLIS | 607 | 真100%（dlisio逐个打开） | 607成功，0失败 |
| PDF井技术数据 | 1831 | 真100%（pdfplumber） | 全部成功 |
| PDF测井报告 | 638 | 真100%（pdfplumber） | 全部成功 |
| LAS | 301 | 真100%（lasio） | 251成功 / 2真失败 / **48超时（全部集中在`03.PRESSURE`目录）** |
| WITSML | 18,814 | 分层抽样1.06%（199个） | 抽样全部合法，未做全量 |
| OpenWorks `.bri`/`.hts` | 未知 | **硬限制**，无开源工具可解析内容 | 仅确认文件存在 |

**⚠️ 开发时必须注意**：
1. **`Well_logs.zip/03.PRESSURE/`目录（压力测试MWD原始数据）用常规`lasio.read()`会大概率超时**，需要专门的流式/分块解析策略，不要直接批量加载
2. 2个已知真实损坏的LAS文件（不要盲目信任，用前先检查）：
   - `Well_logs/02.LWD_EWL/15_9-F-12/WL_RAW_BHPR-GR-MECH_TIME_MWD_2.LAS`（时间字符串混入数值列）
   - `Well_logs/10.PRODUCTION LOGS/15_9-F-12/RAW/WL_RAW_PROD_CCL-PERF_2014-11-26_1.LAS`（行列数不匹配）
3. `.bri`/`.hts`（Landmark OpenWorks专有格式，在`GeoScience_OW_Archive.zip`的`VOLVE_PUBLIC_Seismic.dszip`内层）无法用开源工具验证内容，如果需要用这部分数据，要么找商业软件（Landmark OpenWorks）要么放弃这部分

## 六、给③④⑤赛道的数据可行性结论（取代之前过时的判断）

- **③储层物性预测 / ④岩相预测**：**数据量足够，可以做，不需要降级为仅校准用途**。主训练集用21-24口井的原始测井曲线（`02.LWD_EWL`）+ 计算解释测井（`05.PETROPHYSICAL INTERPRETATION`，21口井），单井数千到上万深度点，24口井合计数万到十几万行样本。岩心塞数据（3个井眼）角色是**校准/精度验证**，不是训练主体。
- **⑤甜点预测**：数据可行性不受本轮扫描影响，卡点仍是"没有现成标签"，需要军伟决定怎么定义代理标签（断层+储层物性+已知产油井位置的组合方案），这是方法论决策不是数据缺失问题。

## 相关文档

- 完整调查过程：`_wiki-methodology/_top/_findings/P2.1_volve_core_data_scale_gap.md`（初版，已被后续finding修正，superseded）、`P2.2`（测井覆盖修正）、`P2.3`（完整清单扫描+井名映射）、`P2.4`（内容级程序化验证）、`P2.5`（2026-07-13 Codex+Claude Workflow联合审查，直连Databricks远端字节级比对确认无遗漏，并修正本文档的规模/类别数字描述）
- Codex完整审查报告：`_wiki-methodology/_top/_external_reviews/codex_data_algorithm_audit_20260713.md`
- 算法源头档案：`_wiki-methodology/_wiki/_entities/algorithm-baselines-6tracks.md`
- 原始扫描产出：`_sandbox/volve_data/_full_inventory/`（含每个zip完整清单、井×zip矩阵、扩展名统计、内容验证JSON）
- 数据资产登记：`_meta/_data_registry.yml` id=`volve-north-sea`
