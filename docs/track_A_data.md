# Track A — Data pipeline

Last updated: 2026-06-04

You own everything between the raw whole-slide images and the tensors that go into the model.

## Files you own

- `src/dataset.py` — PyTorch Dataset
- `src/tiles.py` — tile extraction
- `scripts/preprocess_tiles.py` — bulk preprocessing script
- `data/train_folds.csv` — 5-fold split CSV (generated once, then frozen)

## Current status

- All logged scores so far are still from the thumbnail baseline, not from tiles.
- The current repo only has `PandaDataset` in `src/dataset.py`; there is no working `PandaTileDataset` yet.
- A first Track A tile dataset already exists on Kaggle, but the repo still needs the loader-side integration to use it cleanly from `src/dataset.py`.
- This track is no longer blocked on tile artifacts existing; it is now blocked on turning those artifacts into a stable training input path for Tracks B and C.

## Objective from here to project success

Deliver one reliable tile pipeline end to end:

1. extract tiles from the WSIs
2. publish one team-usable Kaggle Dataset of tile artifacts
3. expose those tiles through `PandaTileDataset`
4. unblock Track B to train concat-tile-pooling models

The first target is not "perfect tile engineering." The first target is one working tile configuration that lets Track B beat the clean thumbnail baseline and move toward `0.78+` QWK.

## What is left right now

1. Implement `src/tiles.py` with:
   - `get_tiles(image, tile_size=192, n_tiles=36) -> np.ndarray`
   - `save_tile_stack(image_id, tiles, out_dir)`
2. If the existing tile dataset is the intended final artifact format, commit the extraction/preprocessing code that produced it so the dataset is reproducible from the repo.
3. Sanity-check the published tile artifacts visually on 5 slides and confirm shape, ordering, and naming conventions.
4. Extend `src/dataset.py` with `PandaTileDataset` returning `[N, 3, H, W]` per slide.
5. Apply augmentation per tile, not once to the whole stack.
6. Hand the final dataset slug, artifact format, and naming convention to Tracks B and C.
7. Only if needed, regenerate and republish the tile dataset after loader-format fixes.

## Recommended order

1. Start with one default configuration: `n_tiles=36`, `tile_size=192`.
2. Get the first tile dataset and loader working before trying multiple tile counts.
3. Once Track B has a first fold-0 tile model, compare `16`, `32`, and `36` tiles if compute allows.
4. Treat stain normalization as a stretch goal, not a prerequisite.

## Done when

- There is a Kaggle Dataset containing tile artifacts for all usable slides.
- The code that produced that dataset is present in the repo and reproducible.
- `PandaTileDataset` is merged and Track B can train with `--tile-dir` without touching data code.
- At least one tile configuration has a logged QWK in `results.md`.
- The team has a short paper-ready description of the tile extraction algorithm.

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
