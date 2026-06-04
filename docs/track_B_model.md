# Track B — Model architecture

Last updated: 2026-06-04

You own how the model processes the input and produces a prediction.

## Files you own

- `src/model.py` — model definitions
- `src/train.py` — training loop

## Current status

- `ConcatTilePoolingModel` already exists in `src/model.py`.
- `src/train.py` already supports:
  - `efficientnet-b0` and `efficientnet-b1`
  - `smoothl1`, `mse`, and `ordinal` losses
  - tile-mode training through `--tile-dir`
- Track A already has a first tile dataset.
- `PandaTileDataset` already exists in `src/dataset.py`, so tile-mode is no longer blocked by missing repo plumbing.
- Logged thumbnail-only results so far:
  - clean 5-fold `b0 + SmoothL1`: `0.7116`
  - clean 5-fold `b0 + MSE`: `0.7030`
  - fold-0 `b1 + SmoothL1`: `0.7125`
  - fold-0 `b0 + ordinal`: `0.7314`
- clean 5-fold `b0 + ordinal`: `0.7244`
- The remaining gap is not model plumbing; it is getting the first real GPU tile run logged and then deciding whether tiles beat the ordinal thumbnail family strongly enough to become the new primary track.

## Objective from here to project success

Pick one best single model family, train it across all 5 folds, and hand those weights to Track C for fair OOF ensembling and external validation.

In practice, that means:

1. finish the best remaining thumbnail ablation quickly
2. switch to tile models as soon as Track A unblocks them
3. avoid spending compute on low-value reruns

## What is left right now

1. Treat `b0 + ordinal` as the current best thumbnail family.
2. Do not spend more compute on full-5-fold `b0 + mse` reruns; that family is already weaker than the clean SmoothL1 baseline.
3. Do not commit to full-5-fold `b1` unless a cheap fold-0 test clearly beats the best `b0` setup.
4. Run the first real tile model on fold 0 with `efficientnet-b0`.
5. Compare tile counts on fold 0 only after the first tile model works.
6. If tiles beat the thumbnail ordinal family cleanly, pick one best tile config and train all 5 folds with it.
7. Only after the base tile model is competitive should you consider mixup.

## Recommended run order

1. Now: `b0 + tiles36 + imsize192`, fold 0 on the existing tile dataset.
2. Then: `b0` tile-count sweep on fold 0, such as `16`, `32`, `36`.
3. Compare those results directly against the current thumbnail reference, `b0 + ordinal` at `0.7244`.
4. Then: best tile config, all 5 folds.
5. Only if compute remains: `b1` follow-up or mixup.

## What not to optimize right now

- Do not chase tiny `b0` vs `b1` differences before the first real tile runs are settled.
- Do not add mixup before there is a stable tile baseline.
- Do not compare runs across different fold splits; `data/train_folds.csv` remains fixed.

## Done when

- The team has a clearly chosen best single model family, with an explicit comparison between thumbnail ordinal and the best tile run.
- There are 5 weights for that family with consistent naming.
- `results.md` contains rows for every serious ablation, not just single-fold spot checks.
- Track C can run fair OOF and final ensembling without needing changes in `src/train.py` or `src/model.py`.

## Reference architecture

The tile model should keep this structure:

1. input `[B, N, 3, H, W]`
2. reshape to `[B*N, 3, H, W]`
3. run the EfficientNet backbone
4. reshape back to `[B, N, F]`
5. average over tiles
6. apply the prediction head

That is the minimum architectural change needed to make tile-based PANDA experiments meaningful.
