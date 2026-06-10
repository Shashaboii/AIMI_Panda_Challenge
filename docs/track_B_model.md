# Track B — Model architecture

Last updated: 2026-06-07

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
- Logged tile fold-0 results so far:
  - `b0 + 16 tiles + ordinal`: `0.6819`
  - `b0 + 32 tiles + ordinal`: `0.6676`
  - `b0 + 36 tiles + ordinal`: `0.6720`
- The current `panda_tiles_36x192_png` artifact set was audited and is not a fair tile benchmark:
  - all audited slides had only `0..9` real tiles out of `36`
  - mean real tiles per slide: `5.356`
  - median real tiles per slide: `5.0`
  - this matches a `512x512` thumbnail source tiled at `192`, so the rest of each artifact is white padding
- The remaining gap is no longer "get the first tile run working." The remaining gap is regenerating a valid tile artifact set from raw TIFF WSIs and then checking whether tile models can beat the ordinal thumbnail family.

## Objective from here to project success

Pick one best single model family, train it across all 5 folds, and hand those weights to Track C for fair OOF ensembling and external validation.

In practice, that means:

1. treat `b0 + ordinal` as the current production baseline
2. do not start 5-fold tile training until the tile artifact source is fixed
3. once Track A regenerates TIFF-derived tiles, rerun a cheap fold-0 tile comparison before scaling out

## What is left right now

1. Treat `b0 + ordinal` as the current best thumbnail family.
2. Do not spend more compute on full-5-fold `b0 + mse` reruns; that family is already weaker than the clean SmoothL1 baseline.
3. Do not commit to full-5-fold `b1` unless a cheap fold-0 test clearly beats the best `b0` setup.
4. Regenerate tile artifacts from raw TIFF WSIs and reject any source path whose capacity audit shows heavy white padding.
5. Run one fold-0 tile model on the fixed TIFF-derived artifact set with `efficientnet-b0`.
6. Compare tile counts on fold 0 only after the first fixed-source tile model works.
7. If tiles beat the thumbnail ordinal family cleanly, pick one best tile config and train all 5 folds with it.
8. Only after the base tile model is competitive should you consider mixup.

## Recommended run order

1. Now: treat `5fold_ordinal_oof_predictions.csv` as the current baseline reference, since it is the best clean OOF file in hand.
2. Next: regenerate tiles from raw TIFF WSIs, not the current thumbnail-derived PNG source.
3. Immediately audit the regenerated artifact directory before training:
   - use `python scripts/audit_tile_artifacts.py --tile-dir <tile_dir>`
   - if the real-tile count still looks capped near `9`, stop and fix the source path
4. Then: `b0 + tiles36 + imsize192`, fold 0 on the fixed tile dataset.
5. Then: `b0` tile-count sweep on fold 0, such as `16`, `32`, `36`, only if the first fixed-source run is competitive.
6. Compare those results directly against the current thumbnail reference, `b0 + ordinal` at `0.7244`.
7. Then: best tile config, all 5 folds.
8. Only if compute remains: `b1` follow-up or mixup.

## What not to optimize right now

- Do not chase tiny `b0` vs `b1` differences before the fixed-source tile runs are settled.
- Do not add mixup before there is a stable tile baseline.
- Do not compare runs across different fold splits; `data/train_folds.csv` remains fixed.
- Do not treat the existing `panda_tiles_36x192_png` results as evidence that tile models are inherently weaker than thumbnails.

## Done when

- The team has a clearly chosen best single model family, with an explicit comparison between thumbnail ordinal and the best valid tile run.
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
