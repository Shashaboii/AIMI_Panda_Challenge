# Track A — Data pipeline

Last updated: 2026-06-04

You own everything between the raw whole-slide images and the tensors that go into the model.

## Files you own

- `src/dataset.py` — PyTorch Dataset
- `src/tiles.py` — tile extraction
- `scripts/preprocess_tiles.py` — bulk preprocessing script
- `data/train_folds.csv` — 5-fold split CSV (generated once, then frozen)

## Current status

- Tile extraction code is in `src/tiles.py`.
- Bulk preprocessing is in `scripts/preprocess_tiles.py`.
- `PandaTileDataset` is implemented in `src/dataset.py` and can load `.npy`, `.npz`, or stacked `.png` tile artifacts as `[N, 3, H, W]`.
- `src.train` now supports `--tile-dir` and respects `--n-tiles` at loader time, so one 36-tile artifact set can be reused for `16/32/36` tile-count ablations.
- `notebooks/03_dataset_tiles.ipynb` is the Track A Kaggle notebook for preprocessing, fold-0 training, ablations, and Kaggle Dataset packaging.
- All logged scores in `results.md` are still from the thumbnail baseline until the first tile fold-0 run is recorded.

## Objective from here to project success

Deliver one reliable tile pipeline end to end:

1. extract tiles from the WSIs
2. publish one team-usable Kaggle Dataset of tile artifacts
3. expose those tiles through `PandaTileDataset`
4. unblock Track B to train concat-tile-pooling models

The first target is not "perfect tile engineering." The first target is one working tile configuration that lets Track B beat the clean thumbnail baseline and move toward `0.78+` QWK.

## What is left right now

1. Finish a clean fold-0 training run from `notebooks/03_dataset_tiles.ipynb`.
2. Record the first tile-model fold-0 QWK in `results.md`.
3. Run the planned tile-count ablation at fixed `tile_size=192`: `16`, `32`, and `36`.
4. Publish the chosen artifact directory as a Kaggle Dataset and hand the slug to Tracks B and C.
5. If the team gets access to raw TIFF WSIs, rerun the same pipeline with raw tiles and compare against the PNG-fallback result.

## Recommended order

1. Default config:
   - `n_tiles=36`
   - `tile_size=192`
   - `format=png` on Kaggle to stay within disk limits
2. Use `notebooks/03_dataset_tiles.ipynb` to:
   - preprocess one smoke-test subset
   - preprocess the full dataset
   - train fold 0
   - prepare `dataset-metadata.json`
   - run `16/32/36` ablations
3. Treat the resized PNG input path as an unblocker mode.
4. Treat the raw TIFF path as the preferred final mode for the best score.

## Done when

- There is a Kaggle Dataset containing tile artifacts for all usable slides.
- The code that produced that dataset is present in the repo and reproducible.
- `PandaTileDataset` is merged and Track B can train with `--tile-dir` without touching data code.
- At least one tile configuration has a logged QWK in `results.md`.
- The team has a short paper-ready description of the tile extraction algorithm.

## Locked conventions

- Artifact directory naming:
  - `/kaggle/working/panda_tiles_<N>x<S>_<format>`
  - example: `/kaggle/working/panda_tiles_36x192_png`
- Training feature tag:
  - `tiles<N>_imsize<S>`
  - example: `tiles36_imsize192`
- Weight filename:
  - `<backbone>_<feature_tag>_<loss>_fold<F>.pth`
  - example: `efficientnetb0_tiles36_imsize192_ordinal_fold0.pth`
- Kaggle dataset slug:
  - `panda-tiles<N>x<S>-<format>`
  - example: `panda-tiles36x192-png`

## Kaggle workflow

1. Run notebook cell 1 to set the shared config.
2. Run cell 2 for a 5-slide smoke test.
3. Run cell 3 for full preprocessing.
4. Run cell 4 for the first fold-0 tile model.
5. Run cell 5 to generate `dataset-metadata.json`.
6. Run cell 6 for the `16/32/36` tile-count ablation.

## Reference extraction recipe

Use the winners' repo for inspiration, then re-implement the idea cleanly. The intended algorithm is:

1. Read the WSI at level 1 of its pyramid with `tifffile`.
2. Pad the image so both dimensions are multiples of `tile_size`.
3. Reshape into a tile grid.
4. Score each tile by tissue content: sum of `(255 - pixel_value)` over the tile.
5. Keep the top `N` tiles by score.
6. Save either:
   - one grid/stack PNG per slide, or
   - one `[N, tile_size, tile_size, 3]` array per slide

The array form is the better long-term fit for concat-tile-pooling.
