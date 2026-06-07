# Experiment results log

Every training run goes here. Append-only — even null results stay, they're part of the paper's experiments section.

## Note on scoring

The Kaggle public leaderboard is unavailable for PANDA — the competition closed in 2020 and late submissions return "Notebook Threw Exception (after deadline)" without a score. **LB QWK is blank for all entries.** Our primary metric is held-out fold QWK from the shared 5-fold split (`data/train_folds.csv`).

## Columns

- **Date** — YYYY-MM-DD
- **Who** — A / B / C
- **Tag** — short label, e.g. `tiles36-imsize192-fold0`
- **Hyperparams** — anything that varies from default
- **Val QWK** — out-of-fold validation score
- **5-fold mean** — if this experiment was trained on all 5 folds, mean QWK across all folds
- **Weights** — Kaggle Dataset slug
- **Notes**

---

| Date | Who | Tag | Hyperparams | Val QWK | 5-fold mean | Weights | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-08 | A | baseline-thumbnails | effnet-b0, 512 thumbnail, fold 0, 6 epochs, lr 3e-4, SmoothL1 | 0.7001 | — | `panda-effnet-b0-weights` | First end-to-end pipeline. Used xhlulu's 512 thumbnails because TIFF reading was broken on current Kaggle base image. Late LB submission rejected (after deadline). |
| 2026-05-15 | A | baseline-scaffold | effnet-b0, 512 thumbnail, fold 0, 6 epochs, lr 3e-4, SmoothL1 | 0.7117 | — | `panda-effnetb0-weights-scaffold` | Reproduced through GitHub scaffold. Confirmed eval loop works. |
| 2026-05-18 | Solo | baseline-5fold-oof | effnet-b0, 512 thumb, 5-fold mean, 6 ep | 0.7116 (OOF) | 0.7116 | `panda-effnetb0-5fold-baseline` | Global OOF QWK. Per-class accuracy 22-61%. Model under-predicts ISUP 5 (22% acc), best at ISUP 0 (61%). |
| 2026-05-26 | B | baseline-thumbnails-b0-fold0 | effnet-b0, 512 thumbnail, fold 0, 6 epochs, lr 3e-4, SmoothL1, dropout 0.3 | 0.7117 | — | — | Reproduced baseline on Kaggle with xhlulu 512x512 resized PNGs. Saved `/kaggle/working/efficientnetb0_fold0.pth`. |
| 2026-05-26 | B | baseline-thumbnails-b1-fold0 | effnet-b1, 512 thumbnail, fold 0, 6 epochs, lr 3e-4, SmoothL1, dropout 0.3 | 0.7125 | — | — | Ran EfficientNet-B1 backbone ablation on Kaggle with xhlulu 512x512 resized PNGs. Saved `/kaggle/working/efficientnetb1_fold0.pth`. |
| 2026-05-26 | B | baseline-thumbnails-b0-mse-fold0 | effnet-b0, 512 thumbnail, fold 0, 6 epochs, lr 3e-4, MSE, dropout 0.3 | 0.7123 | — | — | Ran MSE loss ablation on Kaggle with xhlulu 512x512 resized PNGs. Saved `/kaggle/working/efficientnetb0_mse_fold0.pth`. |
| 2026-05-26 | B | baseline-thumbnails-b0-ordinal-fold0 | effnet-b0, 512 thumbnail, fold 0, 6 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.7314 | — | — | Ran ordinal loss ablation on Kaggle with xhlulu 512x512 resized PNGs. Saved `/kaggle/working/efficientnetb0_ordinal_fold0.pth`. |
| 2026-05-28 | C | baseline-thumbnails-b0-5fold-rerun | effnet-b0, 512 thumbnail, 5 folds, 6 epochs, lr 3e-4, SmoothL1, dropout 0.3 | 0.7032 (OOF) | 0.7037 ± 0.0189 | — | Kaggle rerun from `02b_train_all_folds.ipynb`. Fold scores: 0.6723 / 0.7047 / 0.7294 / 0.6980 / 0.7142. Fold 0 was skipped because `efficientnetb0_fold0.pth` already existed; its OOF score (0.6723) conflicts with the earlier standalone fold-0 run at 0.7117, so this 5-fold summary likely underestimates the clean baseline . |
| 2026-06-01 | C | baseline-thumbnails-b0-5fold-clean | effnet-b0, 512 thumbnail, 5 folds, 6 epochs, lr 3e-4, SmoothL1, dropout 0.3 | 0.7116 (OOF) | 0.7116 ± 0.0106 | — | Clean full 5-fold Kaggle rerun from `02b_train_all_folds.ipynb` followed by `src.oof`. Fold scores: 0.7117 / 0.7047 / 0.7294 / 0.6980 / 0.7142. Wrote `/kaggle/working/efficientnetb0_oof_predictions.csv` with 10,616 rows. Global OOF QWK matched the fold mean at 0.7116. |
| 2026-06-02 | B | baseline-thumbnails-b0-mse-5fold | effnet-b0, 512 thumbnail, 5 folds, 6 epochs, lr 3e-4, MSE, dropout 0.3 | 0.7030 (OOF) | 0.7030 ± 0.0055 | — | Full 5-fold Kaggle MSE rerun plus `src.oof`. Fold scores: 0.7123 / 0.6969 / 0.7022 / 0.6983 / 0.7051. Wrote `/kaggle/working/efficientnetb0_mse_oof_predictions.csv` with 10,616 rows. Saved `efficientnetb0_mse_fold{0..4}.pth` (~16.34 MB each). Confusion matrix shows persistent under-calling of ISUP 5 (221/1224 correct) and strong confusion between adjacent grades. |
| 2026-06-04 | B | baseline-thumbnails-b0-ordinal-5fold | effnet-b0, 512 thumbnail, 5 folds, 6 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.7244 (OOF) | 0.7244 | — | OOF predictions recovered from `ensemble_oof_predictions.csv`. File contains `pred_b0_ordinal` only, so `pred_ensemble` is identical to the ordinal model output. |
| 2026-06-04 | A | tiles36-imsize192-cpu-smoke-fold0 | notebook-defined tiny CNN, 36x192 tile PNGs, fold 0, 1 epoch, batch size 1, CPU smoke test | 0.0000 | — | — | CPU-only sanity check from `03a_train0fold_tiles.ipynb`. Found 10,616 usable tile artifacts; split was train=8,492 and val=2,124. Saved `/kaggle/working/cpu_smoke/cpu_smoke_tinycnn_tiles36_imsize192_cpu_smoke_ordinal_fold0.pth`. This is only a pipeline check and is not comparable to the real GPU EfficientNet tile run. |
| 2026-06-06 | B | tiles16-imsize192-b0-ordinal-fold0 | effnet-b0, 16 tiles kept at loader time from 36x192 PNG tile artifacts, fold 0, 6 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.6819 | — | — | First real GPU concat-tile-pooling run family on Track A tile artifacts using `--n-tiles 16` with `/kaggle/input/datasets/franciscacarneiro/panda-tiles-36x192-png/panda_tiles_36x192_png`. Saved `/kaggle/working/efficientnetb0_tiles16_imsize192_ordinal_fold0.pth`. Faster than the 36-tile run and the best tile fold-0 result so far, but still below the thumbnail ordinal fold-0 baseline of 0.7314. |
| 2026-06-06 | B | tiles32-imsize192-b0-ordinal-fold0 | effnet-b0, 32 tiles kept at loader time from 36x192 PNG tile artifacts, fold 0, 6 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.6676 | — | — | Second real GPU concat-tile-pooling run family on Track A tile artifacts using `--n-tiles 32` with `/kaggle/input/datasets/franciscacarneiro/panda-tiles-36x192-png/panda_tiles_36x192_png`. Saved `/kaggle/working/efficientnetb0_tiles32_imsize192_ordinal_fold0.pth`. Slightly better than the 36-tile run but still worse than both the 16-tile variant and the thumbnail ordinal fold-0 baseline of 0.7314. |
| 2026-06-06 | B | tiles36-imsize192-b0-ordinal-fold0 | effnet-b0, 36 tiles, 192 tile size, fold 0, 6 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.6720 | — | — | First real GPU concat-tile-pooling run on Track A tile artifacts using `/kaggle/input/datasets/franciscacarneiro/panda-tiles-36x192-png/panda_tiles_36x192_png`. Saved `/kaggle/working/efficientnetb0_tiles36_imsize192_ordinal_fold0.pth`. Completed cleanly but underperformed both the 16-tile variant and the thumbnail ordinal fold-0 baseline of 0.7314. |
| 2026-06-06 | B | baseline-thumbnails-b0-ordinal-fold0-5ep | effnet-b0, 512 thumbnail, fold 0, 5 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.7243 | — | — | Reran the PNG thumbnail baseline on xhlulu 512x512 resized images for 5 epochs. Saved `/kaggle/working/efficientnetb0_ordinal_fold0.pth`. |
| 2026-06-07 | B | baseline-thumbnails-b0-ordinal-5fold-5ep | effnet-b0, 512 thumbnail, 5 folds, 5 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.7220 (OOF) | 0.7221 ± 0.0095 | — | Clean 5-fold rerun on xhlulu 512x512 PNG thumbnails. Fold scores: 0.7297 / 0.7067 / 0.7342 / 0.7203 / 0.7197. Wrote `/kaggle/working/efficientnetb0_ordinal_oof_predictions.csv` with 10,616 rows and saved `efficientnetb0_ordinal_fold{0..4}.pth`. |
| 2026-06-07 | B | baseline-thumbnails-b0-ordinal-5fold-5ep | effnet-b0, 512 thumbnail, 5 folds, 5 epochs, lr 3e-4, ordinal BCE, dropout 0.3 | 0.7221 (OOF) | 0.7221 ± 0.0095 | — | Clean 5-fold rerun on xhlulu 512x512 PNG thumbnails. Fold scores: 0.7297 / 0.7068 / 0.7342 / 0.7203 / 0.7197. Wrote `/kaggle/working/ensemble_oof_predictions.csv`; however, `pred_ensemble` is identical to `pred_b0_ordinal`, so this should be treated as a single-model ordinal OOF result rather than a true ensemble. Confusion matrix shows strong performance for ISUP 0 but continued adjacent-grade confusion, especially among ISUP 3–5. |
Ensemble confusion matrix (rows = true ISUP grade, cols = predicted ISUP grade):

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1966 | 768 | 61 | 58 | 29 | 10 |
| 1 | 484 | 1407 | 539 | 171 | 55 | 10 |
| 2 | 89 | 423 | 474 | 229 | 95 | 33 |
| 3 | 44 | 185 | 269 | 352 | 282 | 110 |
| 4 | 54 | 202 | 153 | 255 | 412 | 173 |
| 5 | 46 | 85 | 74 | 151 | 362 | 506 |