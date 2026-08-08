# P45 调研笔记：F3-Demo 官方真实测井 LAS 是否可下载

日期：2026-08-08　类型：只读调研，未下载数据到项目主流程，未落代码

## 结论
可行，但本机沙箱当前无法直接验证下载。dGB/TerraNubis 官方 F3-Demo 渠道**不提供独立小体积 LAS**，测井曲线被打包在 4.5GB 的 OpendTect 二进制工程（.cbvs）里，需登录下载整包——这与项目之前的调查结论一致。但存在第二条更有希望的官方路径：**NLOG.nl（荷兰政府官方油气数据门户）**，F02-1/F03-2/F03-4/F06-1 四口井本身就登记在 NLOG，其行政数据默认公开，测井数据的 5 年保密期早已过期（这些井是上世纪的老井），第三方证据（一篇 MDPI 论文直接引用了同区块 F02-05 井的 RHOB/DT/GR/NPHI 曲线；知名测井教育者 Andy McDonald 的 GitHub 仓库里有一份明确标注来自 NLOG 的荷兰井 LAS 文件 P11-A-02_Composite...las）表明 NLOG 上确实公开分发同区块井的真实 LAS/DLIS 文件，且该门户明确声明数据可自由复制、下载、传播，无需事先书面许可。

## 证据
1. TerraNubis「F3 Demo 2023」`https://terranubis.com/datainfo/F3-Demo-2023`：4 口井（F02-1/F03-2/F03-4/F06-1），含 sonic/GR/density/porosity/impedance 曲线，但"integrated within the OpendTect project — not separately distributed as LAS files"；许可 CC BY-SA 3.0；下载需注册登录；直链 `https://terranubis.com/download/F3_Demo_2023.zip/2`（压缩包 4.5GB，未下载，符合"不下载 GB 级"约束）。
2. `www.opendtect.org/osr/...` 会 301 跳转到上面同一 TerraNubis 页面，确认官方唯一渠道就是整包。
3. dGB 官方数据集页 `https://www.dgbes.com/resources/data-sets`：另有约 2.5GB 版本需邮件 `info@dgbes.com` 申请访问，同样是整包，非单井 LAS。
4. NLOG.nl（`https://www.nlog.nl/en/boreholes`、`/en/data`）：荷兰政府官方油气勘探数据门户，声明数据免费自由使用；四口目标井均属该门户登记范围；同区块 F02-05 井曲线被学术论文直接引用，另有第三方 GitHub 仓库托管来自 NLOG 的荷兰井 LAS 文件，佐证该门户确实分发单井 LAS/DLIS。
5. `tecgraf.puc-rio.br/welllogs/`（巴西 PUC-Rio 北海井库）提供 11 口 F 区块井的 ASCII 曲线，但**不含**目标四口井，且格式非标准 LAS。

## 未完成 / 风险
- 本机沙箱经 mihomo 代理访问 `nlog.nl` 时 TLS 握手失败（WebFetch 报 socket closed，curl 报 SSL unexpected eof），无法确认目标四口井在 NLOG 数据中心的确切详情页/下载链接，也**未能下载任何 LAS 文件做 lasio 真实性验证**——这是本轮任务的主要缺口，需要在无代理限制的网络环境下手动到 `nlog.nl` 数据中心/交互地图按井名检索。
- NLOG 数据虽声明可自由复制传播，但未逐字核对其许可证是否明确允许"用于竞赛项目"这类场景，需实际打开条款页确认。
- 无法 100% 确认 NLOG 收录的是这四口井的**完整**曲线集（尤其 DT/RHOB 是否都全），只能确认同区块相邻井（F02-05）有类似曲线，需要实拿到文件核实。

## 工作量估计
若网络可达 NLOG：人工在数据中心/地图按井名搜索 + 下载 4 个 LAS，预计 30–60 分钟，免费、无需付费；lasio 解析验证再加 10–15 分钟。若 NLOG 也不可行，退回官方整包（4.5GB，需注册）是唯一确定能拿到这四口井曲线的路径，但违反"不下载 GB 级"的任务约束，需另行请示。
