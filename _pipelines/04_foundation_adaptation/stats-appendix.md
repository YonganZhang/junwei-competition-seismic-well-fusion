# Statistical appendix

## Evidence level

The two P6 comparisons use one fixed development split and three training
seeds. Reported dispersion is the population standard deviation across those
seeds. Seeds quantify optimization sensitivity; they are not independent
geological samples and are not a substitute for well-family or spatial folds.

No p-values or confidence intervals are reported because the current design
does not provide enough independent split-level observations for a defensible
inferential test.

## Property

- Development split hash:
  `48897a8dfb1a33929a09bf31f97c30a81ba1fd443239370433fb2e30f415453d`
- Ridge macro standardized RMSE: 0.864632.
- Pretrained GPT-2 + LoRA seed RMSE:
  1.219503, 1.027005, 1.020904.
- Random GPT-2 + LoRA seed RMSE:
  1.271454, 1.478952, 1.027112.
- Mean absolute pretraining gain: 0.170035 RMSE.
- Relative pretraining gain: 13.5037%.

## Lithofacies

- Development split hash:
  `278eff958d3fa0a2c7b35c65f0515df3972abbb475d6fc777a58ca3e1fd9ef3d`
- Logistic accuracy: 0.393939; fixed-nine macro-F1: 0.156817.
- Pretrained GPT-2 + LoRA seed accuracy:
  0.356061, 0.234848, 0.242424.
- Random GPT-2 + LoRA seed accuracy:
  0.060606, 0.219697, 0.060606.
- Mean accuracy gain from pretraining: 0.164141.
- Mean fixed-nine macro-F1 gain from pretraining: 0.089571.

## Stop line

These results justify retaining pretrained initialization in the research
branch. They do not justify production promotion, leaderboard claims or
frozen-test consumption.
