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
