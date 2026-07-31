# P11 OpenMind Residual-Fusion Diagnostic Evidence

## Outcome

- Decision: `VERIFIED_NO_PROMOTION`; default enabled: `False`.
- PyKrige development OOF RMSE: `0.0284497281702`.
- Legacy mean-input mixed-stage Ridge10 RMSE: `0.0288029396248` (five independent spatial folds W/L/T `2/2/1`).
- Best diagnostic route: `mean_stage5_all / fixed_ridge10`; RMSE `0.0284497281702`; material gain `0.000000%`; independent-fold W/L/T `0/0/5`.
- Highest independent spatial-fold win count across all adaptations: `2/5`; promotion requires `4/5`.
- Any positive material gain: `False`; meaningful fold-win improvement over the legacy five-fold pattern: `False`.

## Statistical units and uncertainty

- There are **five genuinely independent spatial units**: locked outer folds 0–4.
- The three random seeds are **paired pseudo-repeats within each spatial fold**, not three additional independent samples. They are never counted as inferential n=15.
- The 95% intervals resample the five whole spatial folds with replacement for 10,000 deterministic replicates. All voxel errors inside a sampled fold move together; the three seed predictions stay paired and are averaged.

## Diagnostic answers

- **Single-stage ablation:** stage0 was worse than PyKrige under both heads. Stage5 with fixed Ridge10 selected gate=0 in every independent spatial fold, so it added no measurable signal; train-only alpha selection made stage5 unstable rather than useful.
- **Three independent channel forwards:** the strongest safe variant was per-channel mixed16 with fixed Ridge10, but it still had negative material gain. The apparent maximum of 2/5 independent-fold wins came from `per_channel_mixed16_concat / fixed_ridge10` and also had 2 independent-fold losses with -0.042375% aggregate gain, so it is not a meaningful win-rate improvement.
- **Stronger Ridge regularization:** train-only alpha search strongly stabilized the ungated mixed path from RMSE `0.758759979284` to `0.0346668779648` and the mean-input stage5 path from `7.50870399407` to `0.235225085978`. That numerical stabilization did not survive gated OOF evaluation: material gain stayed negative and fold wins did not improve.
- **Required random-init control:** every feature route was repeated with the same OpenMind encoder architecture, but checkpoint weights were replaced by seed-specific random initialization before frozen feature extraction. This is distinct from both the Gaussian negative control and the hand-crafted structural control.

## Same-split comparison

| Feature route | Head | Ungated RMSE | Gated RMSE | Material gain | Independent W/L/T | Random-init RMSE | Structural RMSE | RMSE delta 95% block CI | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean_mixed16` | `fixed_ridge10` | 0.758759979284 | 0.0288029396248 | -1.241528% | 2/2/1 | 0.0301102422254 | 0.0285397611875 | [-0.000120559, +0.000890979] | False |
| `mean_mixed16` | `train_only_alpha_grid` | 0.0346668779648 | 0.0287190938275 | -0.946813% | 1/2/2 | 0.0286853618447 | 0.0291200730052 | [-1.31339e-05, +0.000604625] | False |
| `mean_stage0_all` | `fixed_ridge10` | 0.0303241182277 | 0.0289468714363 | -1.747445% | 0/2/3 | 0.0289018315407 | 0.0285397611875 | [+0, +0.00118123] | False |
| `mean_stage0_all` | `train_only_alpha_grid` | 0.0293699846484 | 0.0289278627516 | -1.680630% | 0/2/3 | 0.0289150878132 | 0.0291200730052 | [+0, +0.000956805] | False |
| `mean_stage5_all` | `fixed_ridge10` | 7.50870399407 | 0.0284497281702 | 0.000000% | 0/0/5 | 0.0313261812583 | 0.0285397611875 | [-3.46945e-18, +3.46945e-18] | False |
| `mean_stage5_all` | `train_only_alpha_grid` | 0.235225085978 | 0.0408555136848 | -43.605990% | 1/3/1 | 0.0289916670133 | 0.0291200730052 | [-1.84004e-05, +0.0298172] | False |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1.2664822916 | 0.0284617838462 | -0.042375% | 2/2/1 | 0.0288529212689 | 0.0285397611875 | [-0.000108592, +0.000157323] | False |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 0.0822995228268 | 0.0288056045259 | -1.250895% | 1/4/0 | 0.0291566901588 | 0.0291200730052 | [+2.54949e-05, +0.000773866] | False |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 0.0307521684416 | 0.0287743261219 | -1.140953% | 0/2/3 | 0.0287988229528 | 0.0285397611875 | [+0, +0.000831206] | False |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 0.0295570911265 | 0.0288442598437 | -1.386768% | 0/2/3 | 0.0288176281293 | 0.0291200730052 | [+0, +0.000880433] | False |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 38.2998692746 | 0.0284497281702 | 0.000000% | 0/0/5 | 0.028810387127 | 0.0285397611875 | [-3.46945e-18, +3.46945e-18] | False |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 0.400335412438 | 0.0334882699236 | -17.710334% | 2/2/1 | 0.0295756808171 | 0.0291200730052 | [-0.000615631, +0.0138207] | False |

## Independent spatial-fold outcomes

These five folds—not the seeds—are the independent win/loss/tie units. Each fold metric averages its three paired seed repeats.

| Feature route | Head | Fold | PyKrige RMSE | Gated mean-seed RMSE | Delta | Outcome |
|---|---:|---:|---:|---:|---:|---:|
| `mean_mixed16` | `fixed_ridge10` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 1 | 0.0286276066454 | 0.0298116913452 | +0.00118408469976 | **loss** |
| `mean_mixed16` | `fixed_ridge10` | 2 | 0.0186192911897 | 0.0185078371534 | -0.00011145403627 | **win** |
| `mean_mixed16` | `fixed_ridge10` | 3 | 0.0286050499507 | 0.0294098481166 | +0.000804798165946 | **loss** |
| `mean_mixed16` | `fixed_ridge10` | 4 | 0.0363383384067 | 0.0361577186937 | -0.000180619713028 | **win** |
| `mean_mixed16` | `train_only_alpha_grid` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 1 | 0.0286276066454 | 0.0288879222994 | +0.000260315654015 | **loss** |
| `mean_mixed16` | `train_only_alpha_grid` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 3 | 0.0286050499507 | 0.0285765785301 | -2.84714205755e-05 | **win** |
| `mean_mixed16` | `train_only_alpha_grid` | 4 | 0.0363383384067 | 0.0372026774325 | +0.000864339025755 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 1 | 0.0286276066454 | 0.0304424221144 | +0.00181481546896 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 3 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 4 | 0.0363383384067 | 0.0368231625637 | +0.00048482415696 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 1 | 0.0286276066454 | 0.0293940267514 | +0.000766420105965 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 3 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 4 | 0.0363383384067 | 0.0375922457941 | +0.00125390738742 | **loss** |
| `mean_stage5_all` | `fixed_ridge10` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 1 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 3 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 4 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `mean_stage5_all` | `train_only_alpha_grid` | 0 | 0.027239559051 | 0.0698164072517 | +0.0425768482006 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 1 | 0.0286276066454 | 0.0314443906542 | +0.0028167840088 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `train_only_alpha_grid` | 3 | 0.0286050499507 | 0.0286062160924 | +1.16614171398e-06 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 4 | 0.0363383384067 | 0.0363013906559 | -3.69477508003e-05 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 0 | 0.027239559051 | 0.0275291055087 | +0.000289546457633 | **loss** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1 | 0.0286276066454 | 0.0284679333923 | -0.00015967325315 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 2 | 0.0186192911897 | 0.0185128793186 | -0.000106411871057 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 3 | 0.0286050499507 | 0.028611076317 | +6.02636634795e-06 | **loss** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 4 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 0 | 0.027239559051 | 0.0272996661439 | +6.01070928453e-05 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 1 | 0.0286276066454 | 0.028916831102 | +0.000289224456561 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 2 | 0.0186192911897 | 0.0186647311045 | +4.54399148521e-05 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 3 | 0.0286050499507 | 0.0286019217481 | -3.12820261536e-06 | **win** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 4 | 0.0363383384067 | 0.0374202230346 | +0.00108188462788 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 1 | 0.0286276066454 | 0.0299212206082 | +0.00129361396276 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 3 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 4 | 0.0363383384067 | 0.0365733415454 | +0.000235003138666 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 1 | 0.0286276066454 | 0.0289970074672 | +0.000369400821829 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 3 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 4 | 0.0363383384067 | 0.0375793789628 | +0.00124104055612 | **loss** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 1 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 3 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 4 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 0 | 0.027239559051 | 0.0475961968292 | +0.0203566377781 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 1 | 0.0286276066454 | 0.0283592975947 | -0.000268309050695 | **win** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 2 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 3 | 0.0286050499507 | 0.0307882877892 | +0.00218323783855 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 4 | 0.0363383384067 | 0.0352572932924 | -0.00108104511431 | **win** |

## Seed-level diagnostic details (paired pseudo-repeats)

The rows below preserve the requested fold×seed audit trail, but they are correlated pseudo-repeats and are not counted as independent evidence. Exact gate-zero degenerations are ties.

| Feature route | Head | Seed | Fold | Alpha | PyKrige RMSE | Gated RMSE | Delta | Diagnostic outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean_mixed16` | `fixed_ridge10` | 1867973658 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 1867973658 | 1 | 10.0 | 0.0286276066454 | 0.0321798607447 | +0.00355225409929 | **loss** |
| `mean_mixed16` | `fixed_ridge10` | 1867973658 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 1867973658 | 3 | 10.0 | 0.0286050499507 | 0.0302065362833 | +0.00160148633264 | **loss** |
| `mean_mixed16` | `fixed_ridge10` | 1867973658 | 4 | 10.0 | 0.0363383384067 | 0.0357964792676 | -0.000541859139085 | **win** |
| `mean_mixed16` | `fixed_ridge10` | 2137841944 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 2137841944 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 2137841944 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 2137841944 | 3 | 10.0 | 0.0286050499507 | 0.0286040617243 | -9.8822641683e-07 | **win** |
| `mean_mixed16` | `fixed_ridge10` | 2137841944 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 3902865753 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 3902865753 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `mean_mixed16` | `fixed_ridge10` | 3902865753 | 2 | 10.0 | 0.0186192911897 | 0.0182849290809 | -0.000334362108811 | **win** |
| `mean_mixed16` | `fixed_ridge10` | 3902865753 | 3 | 10.0 | 0.0286050499507 | 0.0294189463423 | +0.00081389639161 | **loss** |
| `mean_mixed16` | `fixed_ridge10` | 3902865753 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 1867973658 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 1867973658 | 1 | 10000.0 | 0.0286276066454 | 0.0290072506821 | +0.000379644036696 | **loss** |
| `mean_mixed16` | `train_only_alpha_grid` | 1867973658 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 1867973658 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 1867973658 | 4 | 10000.0 | 0.0363383384067 | 0.0369584487638 | +0.000620110357086 | **loss** |
| `mean_mixed16` | `train_only_alpha_grid` | 2137841944 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 2137841944 | 1 | 10000.0 | 0.0286276066454 | 0.0290413744831 | +0.00041376783766 | **loss** |
| `mean_mixed16` | `train_only_alpha_grid` | 2137841944 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 2137841944 | 3 | 10000.0 | 0.0286050499507 | 0.0285196356889 | -8.54142617266e-05 | **win** |
| `mean_mixed16` | `train_only_alpha_grid` | 2137841944 | 4 | 10000.0 | 0.0363383384067 | 0.0371403598644 | +0.000802021457664 | **loss** |
| `mean_mixed16` | `train_only_alpha_grid` | 3902865753 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 3902865753 | 1 | 10000.0 | 0.0286276066454 | 0.0286151417331 | -1.2464912311e-05 | **win** |
| `mean_mixed16` | `train_only_alpha_grid` | 3902865753 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 3902865753 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_mixed16` | `train_only_alpha_grid` | 3902865753 | 4 | 10000.0 | 0.0363383384067 | 0.0375092236692 | +0.00117088526252 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 1867973658 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 1867973658 | 1 | 10.0 | 0.0286276066454 | 0.0304424221144 | +0.00181481546896 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 1867973658 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 1867973658 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 1867973658 | 4 | 10.0 | 0.0363383384067 | 0.0368231625637 | +0.00048482415696 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 2137841944 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 2137841944 | 1 | 10.0 | 0.0286276066454 | 0.0304424221144 | +0.00181481546896 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 2137841944 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 2137841944 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 2137841944 | 4 | 10.0 | 0.0363383384067 | 0.0368231625637 | +0.00048482415696 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 3902865753 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 3902865753 | 1 | 10.0 | 0.0286276066454 | 0.0304424221144 | +0.00181481546896 | **loss** |
| `mean_stage0_all` | `fixed_ridge10` | 3902865753 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 3902865753 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `fixed_ridge10` | 3902865753 | 4 | 10.0 | 0.0363383384067 | 0.0368231625637 | +0.00048482415696 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 1867973658 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 1867973658 | 1 | 10000.0 | 0.0286276066454 | 0.0293940267514 | +0.000766420105965 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 1867973658 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 1867973658 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 1867973658 | 4 | 10000.0 | 0.0363383384067 | 0.0375922457941 | +0.00125390738742 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 2137841944 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 2137841944 | 1 | 10000.0 | 0.0286276066454 | 0.0293940267514 | +0.000766420105965 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 2137841944 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 2137841944 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 2137841944 | 4 | 10000.0 | 0.0363383384067 | 0.0375922457941 | +0.00125390738742 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 3902865753 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 3902865753 | 1 | 10000.0 | 0.0286276066454 | 0.0293940267514 | +0.000766420105965 | **loss** |
| `mean_stage0_all` | `train_only_alpha_grid` | 3902865753 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 3902865753 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage0_all` | `train_only_alpha_grid` | 3902865753 | 4 | 10000.0 | 0.0363383384067 | 0.0375922457941 | +0.00125390738742 | **loss** |
| `mean_stage5_all` | `fixed_ridge10` | 1867973658 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 1867973658 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 1867973658 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 1867973658 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 1867973658 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 2137841944 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 2137841944 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 2137841944 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 2137841944 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 2137841944 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 3902865753 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 3902865753 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 3902865753 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 3902865753 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `mean_stage5_all` | `fixed_ridge10` | 3902865753 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `mean_stage5_all` | `train_only_alpha_grid` | 1867973658 | 0 | 1000.0 | 0.027239559051 | 0.0698164072517 | +0.0425768482006 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 1867973658 | 1 | 10000.0 | 0.0286276066454 | 0.0314443906542 | +0.0028167840088 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 1867973658 | 2 | 100.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `train_only_alpha_grid` | 1867973658 | 3 | 1000.0 | 0.0286050499507 | 0.0286062160924 | +1.16614171398e-06 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 1867973658 | 4 | 1000.0 | 0.0363383384067 | 0.0363013906559 | -3.69477508003e-05 | **win** |
| `mean_stage5_all` | `train_only_alpha_grid` | 2137841944 | 0 | 1000.0 | 0.027239559051 | 0.0698164072517 | +0.0425768482006 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 2137841944 | 1 | 10000.0 | 0.0286276066454 | 0.0314443906542 | +0.0028167840088 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 2137841944 | 2 | 100.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `train_only_alpha_grid` | 2137841944 | 3 | 1000.0 | 0.0286050499507 | 0.0286062160924 | +1.16614171398e-06 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 2137841944 | 4 | 1000.0 | 0.0363383384067 | 0.0363013906559 | -3.69477508003e-05 | **win** |
| `mean_stage5_all` | `train_only_alpha_grid` | 3902865753 | 0 | 1000.0 | 0.027239559051 | 0.0698164072517 | +0.0425768482006 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 3902865753 | 1 | 10000.0 | 0.0286276066454 | 0.0314443906542 | +0.0028167840088 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 3902865753 | 2 | 100.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `mean_stage5_all` | `train_only_alpha_grid` | 3902865753 | 3 | 1000.0 | 0.0286050499507 | 0.0286062160924 | +1.16614171398e-06 | **loss** |
| `mean_stage5_all` | `train_only_alpha_grid` | 3902865753 | 4 | 1000.0 | 0.0363383384067 | 0.0363013906559 | -3.69477508003e-05 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1867973658 | 0 | 10.0 | 0.027239559051 | 0.0281996741617 | +0.000960115110616 | **loss** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1867973658 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1867973658 | 2 | 10.0 | 0.0186192911897 | 0.0183000555765 | -0.000319235613172 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1867973658 | 3 | 10.0 | 0.0286050499507 | 0.0286231290497 | +1.80790990438e-05 | **loss** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 1867973658 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 2137841944 | 0 | 10.0 | 0.027239559051 | 0.0271480833133 | -9.14757377188e-05 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 2137841944 | 1 | 10.0 | 0.0286276066454 | 0.0287669441808 | +0.000139337535395 | **loss** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 2137841944 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 2137841944 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 2137841944 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 3902865753 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 3902865753 | 1 | 10.0 | 0.0286276066454 | 0.0280092493506 | -0.000618357294844 | **win** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 3902865753 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 3902865753 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_mixed16_concat` | `fixed_ridge10` | 3902865753 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 1867973658 | 0 | 100.0 | 0.027239559051 | 0.0274198803296 | +0.000180321278536 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 1867973658 | 1 | 10000.0 | 0.0286276066454 | 0.0290452444194 | +0.000417637773976 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 1867973658 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 1867973658 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 1867973658 | 4 | 10000.0 | 0.0363383384067 | 0.0370195729381 | +0.000681234531353 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 2137841944 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 2137841944 | 1 | 10000.0 | 0.0286276066454 | 0.0289982149107 | +0.000370608265235 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 2137841944 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 2137841944 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 2137841944 | 4 | 10000.0 | 0.0363383384067 | 0.0365521583172 | +0.000213819910459 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 3902865753 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 3902865753 | 1 | 10000.0 | 0.0286276066454 | 0.0287070339759 | +7.94273304728e-05 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 3902865753 | 2 | 10000.0 | 0.0186192911897 | 0.0187556109342 | +0.000136319744556 | **loss** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 3902865753 | 3 | 10000.0 | 0.0286050499507 | 0.0285956653428 | -9.38460784607e-06 | **win** |
| `per_channel_mixed16_concat` | `train_only_alpha_grid` | 3902865753 | 4 | 10000.0 | 0.0363383384067 | 0.0386889378485 | +0.00235059944182 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 1867973658 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 1867973658 | 1 | 10.0 | 0.0286276066454 | 0.0299212206082 | +0.00129361396276 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 1867973658 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 1867973658 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 1867973658 | 4 | 10.0 | 0.0363383384067 | 0.0365733415454 | +0.000235003138666 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 2137841944 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 2137841944 | 1 | 10.0 | 0.0286276066454 | 0.0299212206082 | +0.00129361396276 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 2137841944 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 2137841944 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 2137841944 | 4 | 10.0 | 0.0363383384067 | 0.0365733415454 | +0.000235003138666 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 3902865753 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 3902865753 | 1 | 10.0 | 0.0286276066454 | 0.0299212206082 | +0.00129361396276 | **loss** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 3902865753 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 3902865753 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `fixed_ridge10` | 3902865753 | 4 | 10.0 | 0.0363383384067 | 0.0365733415454 | +0.000235003138666 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 1867973658 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 1867973658 | 1 | 10000.0 | 0.0286276066454 | 0.0289970074672 | +0.000369400821829 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 1867973658 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 1867973658 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 1867973658 | 4 | 10000.0 | 0.0363383384067 | 0.0375793789628 | +0.00124104055612 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 2137841944 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 2137841944 | 1 | 10000.0 | 0.0286276066454 | 0.0289970074672 | +0.000369400821829 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 2137841944 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 2137841944 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 2137841944 | 4 | 10000.0 | 0.0363383384067 | 0.0375793789628 | +0.00124104055612 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 3902865753 | 0 | 10000.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 3902865753 | 1 | 10000.0 | 0.0286276066454 | 0.0289970074672 | +0.000369400821829 | **loss** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 3902865753 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 3902865753 | 3 | 10000.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage0_all_concat` | `train_only_alpha_grid` | 3902865753 | 4 | 10000.0 | 0.0363383384067 | 0.0375793789628 | +0.00124104055612 | **loss** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 1867973658 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 1867973658 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 1867973658 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 1867973658 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 1867973658 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 2137841944 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 2137841944 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 2137841944 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 2137841944 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 2137841944 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 3902865753 | 0 | 10.0 | 0.027239559051 | 0.027239559051 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 3902865753 | 1 | 10.0 | 0.0286276066454 | 0.0286276066454 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 3902865753 | 2 | 10.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 3902865753 | 3 | 10.0 | 0.0286050499507 | 0.0286050499507 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `fixed_ridge10` | 3902865753 | 4 | 10.0 | 0.0363383384067 | 0.0363383384067 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 1867973658 | 0 | 10000.0 | 0.027239559051 | 0.0475961968292 | +0.0203566377781 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 1867973658 | 1 | 10000.0 | 0.0286276066454 | 0.0283592975947 | -0.000268309050695 | **win** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 1867973658 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 1867973658 | 3 | 10000.0 | 0.0286050499507 | 0.0307882877892 | +0.00218323783855 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 1867973658 | 4 | 10000.0 | 0.0363383384067 | 0.0352572932924 | -0.00108104511431 | **win** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 2137841944 | 0 | 10000.0 | 0.027239559051 | 0.0475961968292 | +0.0203566377781 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 2137841944 | 1 | 10000.0 | 0.0286276066454 | 0.0283592975947 | -0.000268309050695 | **win** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 2137841944 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 2137841944 | 3 | 10000.0 | 0.0286050499507 | 0.0307882877892 | +0.00218323783855 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 2137841944 | 4 | 10000.0 | 0.0363383384067 | 0.0352572932924 | -0.00108104511431 | **win** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 3902865753 | 0 | 10000.0 | 0.027239559051 | 0.0475961968292 | +0.0203566377781 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 3902865753 | 1 | 10000.0 | 0.0286276066454 | 0.0283592975947 | -0.000268309050695 | **win** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 3902865753 | 2 | 10000.0 | 0.0186192911897 | 0.0186192911897 | +0 | **tie** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 3902865753 | 3 | 10000.0 | 0.0286050499507 | 0.0307882877892 | +0.00218323783855 | **loss** |
| `per_channel_stage5_all_concat` | `train_only_alpha_grid` | 3902865753 | 4 | 10000.0 | 0.0363383384067 | 0.0352572932924 | -0.00108104511431 | **win** |

## Holdout firewall and feature identity

- The CLI accepts no test or holdout path.
- The only HDF5 file opened was `train.h5`.
- OpenMind read only `seismic_patch[0:3]` and `seismic_patch[8]`; it did not read labels.
- PyKrige targets and predictions are the same hash-verified P5 Stage-3 development OOF rows used by committed P11.
- Checkpoint SHA-256: `7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933`.
- Per-channel feature cache SHA-256: `204dc89c09334551afacd0287e62a2b8cf70381b871a547a4e0e09d7921c39c0`.
- Same-architecture random-init cache SHA-256: `2d781bf8438fd4fa0539a4eda06abd25f4f8b6c0ea9a7556f5a10edcb4d15edd`.
- Per-voxel prediction-error artifact SHA-256: `60235b9248796871fbf6139391cc62157fc4fb40721a83c30b21c2eeed3f31c3`.

## Recommendation boundary

现有OpenMind checkpoint的适配空间已基本穷尽，建议更换更贴近地震/地质领域的基础模型。

更换基础模型必须升级给负责人/军伟决策；本实验没有自行更换 OpenMind checkpoint。

Full controls, alpha-search scores, gate/residual statistics, block-bootstrap intervals, and paired seed details are retained in `summary.json`; per-voxel errors are in `prediction_errors.npz`.
