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
