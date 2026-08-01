# P29 lithofacies root-cause audit

## Conclusion

P28 connected actions to XGBoost predictions and the fixed-schema metric, but its categorical-only observation discarded effect magnitude and uncertainty. Its three model seeds were also deterministic duplicates. P29 repairs both defects without changing the split or primary metric.

## Causal-chain audit

| stage | P28 finding | P29 evidence |
|---|---|---|
| observation | Only support buckets and categorical feedback were visible. | Bounded normalized deltas, anonymous class-support shares, and three-inner-fold uncertainty are visible. |
| prompt | Numeric effect size and confidence were removed. | The live enhanced prompt receives effect units and an equal-budget categorical ablation is retained. |
| selected action | Live strict-JSON actions were valid. | Every live decision remains strict JSON and without replacement. |
| executor | Actions changed XGBoost configuration. | All non-A0 config hashes differ: `True`. |
| prediction | P28 actions changed predictions, but duplicated seed hashes were counted as repeats. | One seed is declared; every non-A0 action changes an inner or outer prediction hash: `True`. |
| metric | Fixed-nine Macro-F1 was correctly computed. | The same primary metric is preserved. |
| promotion | Inner LOGO3 selected actions for a disjoint outer fold. | The same nested split is preserved; the robust inner rule is frozen before evaluation. |
| endpoint | P28 outer improvement failed. | Post-policy transfer correlation is `0.19584841847480586` and is diagnostic only. |

## Leakage firewall

The live policy sees no raw metric, row-level target, class name, group identity, sample identifier, path, residual, or outer result. All outer action diagnostics are computed after policy calls and are never used for legal selection.
