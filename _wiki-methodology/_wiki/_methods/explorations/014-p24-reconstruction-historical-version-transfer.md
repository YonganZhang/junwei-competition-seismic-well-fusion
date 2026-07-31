# 014: P24 同场区历史属性版本迁移

status: L3_validated_keep_transfer | created: 2026-08-01 | updated: 2026-08-01

## Hypothesis

若 P21 的真实预训练 GFM 表征确实改善了空间邻域，而非只适配最终
`merge_pp04b_PHIF_NW` 目标版本，则冻结模型在同一网格、未参与开发的
历史 `pp04phif/realisation.1` 上仍应优于重新拟合的 PyKrige。

## Frozen protocol

- 在读取历史目标指标前提交
  `p24_historical_transfer_preregistration.json`，固定数据哈希、空间映射、
  5 折 KJI、512/2,048 标签预算、PyKrige 1.7.3 与 P21 参数。
- 最终 RMS 属性必须与 Eclipse KJI→IJK 正值序列逐元素完全相等，历史属性
  才可沿相同 RMS 顺序映射回 KJI。
- 每折只用该历史版本的 512 个训练目标重拟合 PyKrige；P21 的地震输入、
  真实预训练 GFM 特征、三个固定核和融合权重不变。
- 开目标后不做 HPO，不修改映射或成功门槛。

## Acceptance

候选 pooled RMSE 相对 PyKrige 至少改善 1%，且五个空间折最多一个退步。

## Result

PyKrige RMSE 为 `0.028235410003`，冻结 P21 为 `0.027825182663`，相对改善
`1.4529%`；4 折改善、1 折退步。MAE 从 `0.021293578592` 降至
`0.020826337775`。预注册门槛通过。

## Verdict + Reason

`L3_validated_keep_transfer`。冻结 P21 对未使用的同场区历史目标版本保持收益，
说明其邻域表征具有一定目标版本迁移性。由于数据仍来自同一 Volve 场区、同一
geomodel，且整折 bootstrap 95% 区间跨 0，本轮不升级为跨场区或首次盲测结论；
外部场区/官方隐藏测试仍是最终缺口。
