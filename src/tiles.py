"""Tile extraction helpers for PANDA whole-slide images."""

import os
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


WHITE_PIXEL = 255


def _ensure_rgb_uint8(image):
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim != 3:
        raise ValueError(f'Expected a 2-D or 3-D image array, got shape {tuple(image.shape)}')

    if image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[-1] != 3:
        raise ValueError(f'Expected 1, 3, or 4 channels, got shape {tuple(image.shape)}')

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def read_slide_image(slide_path, level=1):
    """Read a pyramid level from a PANDA TIFF slide as an RGB uint8 array."""
    if level < 0:
        raise ValueError('level must be non-negative')

    with tifffile.TiffFile(slide_path) as tif:
        series = tif.series[0]
        levels = getattr(series, 'levels', None)
        if levels is not None and len(levels) > level:
            image = levels[level].asarray()
        elif len(tif.pages) > level:
            image = tif.pages[level].asarray()
        else:
            image = series.asarray()
    return _ensure_rgb_uint8(image)


def read_raster_image(image_path):
    """Read a standard raster image such as PNG/JPEG as RGB uint8."""
    with Image.open(image_path) as img:
        return _ensure_rgb_uint8(img.convert('RGB'))


def pad_to_tile_size(image, tile_size, pad_value=WHITE_PIXEL):
    """Pad HWC image with white pixels so both spatial dims divide tile_size."""
    if tile_size <= 0:
        raise ValueError('tile_size must be positive')

    image = _ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    pad_h = (-height) % tile_size
    pad_w = (-width) % tile_size
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=pad_value,
    )


def tile_scores(tiles):
    """Score tiles by tissue content: higher score means less white background."""
    tiles = np.asarray(tiles, dtype=np.uint8)
    if tiles.ndim != 4:
        raise ValueError(f'Expected tiles shaped [N, H, W, C], got {tuple(tiles.shape)}')
    return (WHITE_PIXEL - tiles.astype(np.int32)).sum(axis=(1, 2, 3), dtype=np.int64)


def get_tiles(image, tile_size=192, n_tiles=36):
    """Return the top-N tissue tiles as a ``[N, tile_size, tile_size, 3]`` array."""
    if n_tiles <= 0:
        raise ValueError('n_tiles must be positive')

    padded = pad_to_tile_size(image, tile_size)
    grid_h = padded.shape[0] // tile_size
    grid_w = padded.shape[1] // tile_size
    tiles = padded.reshape(
        grid_h,
        tile_size,
        grid_w,
        tile_size,
        3,
    ).transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    scores = tile_scores(tiles)
    order = np.argsort(-scores, kind='stable')
    selected = tiles[order[:min(n_tiles, len(order))]]

    if selected.shape[0] < n_tiles:
        filler = np.full(
            (n_tiles - selected.shape[0], tile_size, tile_size, 3),
            WHITE_PIXEL,
            dtype=np.uint8,
        )
        selected = np.concatenate([selected, filler], axis=0)
    return np.ascontiguousarray(selected)


def extract_tiles_from_slide(slide_path, tile_size=192, n_tiles=36, level=1):
    """Read a slide or raster image from disk and return its top-N tiles.

    TIFF inputs use ``level`` to read from the WSI pyramid. PNG/JPEG inputs are
    treated as already-rendered RGB images and ignore ``level``.
    """
    suffix = Path(slide_path).suffix.lower()
    if suffix in ('.tif', '.tiff'):
        image = read_slide_image(slide_path, level=level)
    else:
        image = read_raster_image(slide_path)
    return get_tiles(image, tile_size=tile_size, n_tiles=n_tiles)


def tiles_to_stack_image(tiles):
    """Convert ``[N, H, W, 3]`` tiles into one vertical RGB image for inspection."""
    tiles = np.asarray(tiles, dtype=np.uint8)
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError(f'Expected tiles shaped [N, H, W, 3], got {tuple(tiles.shape)}')
    return np.ascontiguousarray(tiles.reshape(-1, tiles.shape[2], 3))


def save_tile_stack(image_id, tiles, out_dir):
    """Save tiles as one vertically concatenated PNG for easy visual QC."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = tiles_to_stack_image(tiles)
    out_path = out_dir / f'{image_id}.png'
    Image.fromarray(stack, mode='RGB').save(out_path)
    return str(out_path)


def save_tile_array(image_id, tiles, out_dir):
    """Save tiles as ``.npy`` so the training dataset can load them directly."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{image_id}.npy'
    tmp_path = out_dir / f'{image_id}.npy.tmp'
    try:
        with open(tmp_path, 'wb') as f:
            np.save(f, np.asarray(tiles, dtype=np.uint8))
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return str(out_path)
