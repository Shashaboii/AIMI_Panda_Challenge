# Track A — Data pipeline

You own everything between the raw whole-slide images and the tensors that go into the model.

## Files you own

- `src/dataset.py` — PyTorch Dataset
- `src/tiles.py` — tile extraction (you'll create this)
- `data/train_folds.csv` — 5-fold split CSV (generated once, then frozen)

## Goal for the project

Replace the baseline's 512×512 thumbnail input with a tile-based input that gives the model access to high-resolution tissue regions. This is the single biggest score-mover in PANDA — expected lift from val QWK ~0.70 to ~0.78–0.82.

## Week 1 — Tile extraction

Look at `s07_simple_tile.py` in the [winners' repo](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution). Read it, take notes on the approach, **close the tab**, then write your own version. The algorithm in plain English:

1. Read the WSI at level 1 of its pyramid (use `tifffile`, not `skimage.io.MultiImage` — the latter is broken on the current Kaggle base image).
2. Pad the image so its dimensions are multiples of your tile size (e.g. 192 or 256).
3. Reshape into a grid of tiles.
4. Score each tile by "tissue content" — sum of (255 − pixel value) across the tile. Higher score = more tissue (less white space).
5. Keep the top N tiles by score (N is typically 16, 32, or 36).
6. Either save them stacked vertically as one big PNG (simple, works with the baseline model), or save them as a numpy array of shape `[N, tile_size, tile_size, 3]` (better for Person B's concat-tile-pooling model).

Build it as a standalone module:

```python
# src/tiles.py
def get_tiles(image: np.ndarray, tile_size=192, n_tiles=36) -> np.ndarray:
    """Returns [n_tiles, tile_size, tile_size, 3] uint8 array."""
    ...

def save_tile_stack(image_id, tiles, out_dir):
    """Save tiles as one PNG (concatenated grid) for easy loading."""
    ...
```

Then a script `scripts/preprocess_tiles.py` that iterates over all training slides and produces a Kaggle Dataset.

## Week 2 — Tile-aware Dataset

Once tiles exist, extend `src/dataset.py`. Add a `PandaTileDataset` class that returns `[N, 3, H, W]` per slide (Person B will average-pool over the N dimension in their model). Augmentation should be applied per-tile (random flip/rotate each).

## Week 3 — Stain normalization (optional)

Radboud and Karolinska slides look slightly different (different scanners, different H&E protocols). Try one of:
- Reinhard normalization (`staintools` library, simple, fast)
- Macenko normalization (also in `staintools`)

Check whether it helps val QWK. Don't be surprised if it doesn't — sometimes augmentation alone is enough.

## Week 4 — Lock down the final tile set

Pick whichever config gave the best val QWK in your ablation. Hand off to Person C for the final ensemble.

## What success looks like

- A Kaggle Dataset called something like `panda-tiles-36x192` containing tile stacks for every slide.
- A `src/dataset.py` with a working `PandaTileDataset`.
- `results.md` rows showing each configuration's val QWK.
- A short paragraph in the paper describing the tile extraction algorithm.

## Where to start Monday

1. Pull the latest from main
2. Read the existing `src/dataset.py` and `src/eval.py` so you understand the existing API
3. Set up a Kaggle notebook that imports your tile code and tests it on 5 slides
4. Visualize the extracted tiles — sanity check they actually contain tissue and not just background
5. Once happy, run the preprocessing on all ~10k slides (about 45 min)
6. Publish as Kaggle Dataset, note its slug in `results.md`
