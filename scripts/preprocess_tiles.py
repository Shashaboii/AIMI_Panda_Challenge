"""Precompute per-slide PANDA tile artifacts for tile-based training.

On Kaggle, prefer ``--format png`` for full-dataset runs because uncompressed
``.npy`` tiles can exceed the notebook working-disk budget.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tiles import (
    extract_tiles_from_slide,
    read_raster_image,
    read_slide_image,
    save_tile_array,
    save_tile_stack,
    tile_grid_capacity,
)

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib is in requirements, but keep the fallback simple.
    Parallel = None
    delayed = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides-dir', required=True,
                    help='Directory containing PANDA slide images (.tiff/.tif/.png/.jpg)')
    ap.add_argument('--output-dir', required=True,
                    help='Directory to write one tile artifact per slide')
    ap.add_argument('--folds-csv', default='data/train_folds.csv',
                    help='Optional CSV used to define which image_ids to process')
    ap.add_argument('--tile-size', type=int, default=192)
    ap.add_argument('--n-tiles', type=int, default=36)
    ap.add_argument('--level', type=int, default=1,
                    help='Pyramid level for TIFF inputs; ignored for PNG/JPEG inputs')
    ap.add_argument('--format', choices=['npy', 'png', 'both'], default='npy',
                    help='Artifact format to save')
    ap.add_argument('--limit', type=int,
                    help='Optional cap for quick smoke tests')
    ap.add_argument('--n-jobs', type=int, default=1,
                    help='Parallel workers; 1 keeps execution sequential')
    ap.add_argument('--overwrite', action='store_true',
                    help='Recompute artifacts even if outputs already exist')
    ap.add_argument('--preflight-sample', type=int, default=16,
                    help='How many source images to inspect before preprocessing')
    ap.add_argument('--fail-on-low-capacity', action='store_true',
                    help='Abort if the sampled sources cannot supply n_tiles without padding')
    return ap.parse_args()


def collect_image_ids(slides_dir, folds_csv=None, limit=None):
    if folds_csv:
        df = pd.read_csv(folds_csv, usecols=['image_id'])
        image_ids = df.image_id.astype(str).tolist()
    else:
        slides_dir = Path(slides_dir)
        patterns = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
        image_ids = []
        for pattern in patterns:
            image_ids.extend(path.stem for path in sorted(slides_dir.glob(pattern)))
        image_ids = sorted(set(image_ids))
    if limit is not None:
        image_ids = image_ids[:limit]
    return image_ids


def resolve_slide_path(slides_dir, image_id):
    slides_dir = Path(slides_dir)
    for suffix in ('.tiff', '.tif', '.png', '.jpg', '.jpeg'):
        candidate = slides_dir / f'{image_id}{suffix}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'Could not find {image_id} with any supported extension '
        f'(.tiff, .tif, .png, .jpg, .jpeg) in {slides_dir}'
    )


def outputs_exist(output_dir, image_id, save_format):
    output_dir = Path(output_dir)
    checks = []
    if save_format in ('npy', 'both'):
        checks.append(output_dir / f'{image_id}.npy')
    if save_format in ('png', 'both'):
        checks.append(output_dir / f'{image_id}.png')
    return bool(checks) and all(path.exists() for path in checks)


def format_gib(num_bytes):
    return f'{num_bytes / (1024 ** 3):.1f} GiB'


def estimate_npy_bytes(num_slides, tile_size, n_tiles):
    return num_slides * n_tiles * tile_size * tile_size * 3


def check_disk_budget(output_dir, num_pending, tile_size, n_tiles, save_format):
    if save_format not in ('npy', 'both') or num_pending == 0:
        return

    est_npy_bytes = estimate_npy_bytes(num_pending, tile_size, n_tiles)
    free_bytes = shutil.disk_usage(output_dir).free
    print(
        'Estimated uncompressed NPY footprint:',
        format_gib(est_npy_bytes),
        f'(free: {format_gib(free_bytes)})',
    )
    if est_npy_bytes > free_bytes:
        raise RuntimeError(
            'Requested output format includes .npy tiles, but the estimated uncompressed '
            f'footprint is {format_gib(est_npy_bytes)} and only {format_gib(free_bytes)} '
            'is free. Use --format png on Kaggle, or reduce tile count/size.'
        )


def process_one(image_id, slides_dir, output_dir, tile_size, n_tiles, level, save_format):
    slide_path = resolve_slide_path(slides_dir, image_id)
    tiles = extract_tiles_from_slide(
        slide_path,
        tile_size=tile_size,
        n_tiles=n_tiles,
        level=level,
    )
    saved_paths = []
    if save_format in ('npy', 'both'):
        saved_paths.append(save_tile_array(image_id, tiles, output_dir))
    if save_format in ('png', 'both'):
        saved_paths.append(save_tile_stack(image_id, tiles, output_dir))
    return image_id, saved_paths


def inspect_source_capacity(slide_path, tile_size, level):
    suffix = slide_path.suffix.lower()
    if suffix in ('.tif', '.tiff'):
        image = read_slide_image(slide_path, level=level)
        source_kind = 'tiff'
    else:
        image = read_raster_image(slide_path)
        source_kind = 'raster'

    height, width = image.shape[:2]
    capacity = tile_grid_capacity(image.shape, tile_size)
    return {
        'source_kind': source_kind,
        'height': int(height),
        'width': int(width),
        'capacity': int(capacity),
        'path': str(slide_path),
    }


def run_preflight_capacity_check(image_ids, slides_dir, tile_size, n_tiles, level,
                                 sample_size=16, fail_on_low_capacity=False):
    sample_size = max(0, int(sample_size))
    if sample_size == 0 or not image_ids:
        return

    sample_ids = image_ids[:min(sample_size, len(image_ids))]
    rows = [
        inspect_source_capacity(resolve_slide_path(slides_dir, image_id), tile_size, level)
        for image_id in sample_ids
    ]
    capacities = [row['capacity'] for row in rows]
    source_kinds = sorted({row['source_kind'] for row in rows})
    dims_preview = ', '.join(
        f"{row['width']}x{row['height']}"
        for row in rows[:min(4, len(rows))]
    )

    print(
        f'Preflight capacity ({len(rows)} sample slides, sources={source_kinds}): '
        f'min={min(capacities)} median={int(pd.Series(capacities).median())} max={max(capacities)} '
        f'for requested n_tiles={n_tiles}'
    )
    print(f'Example source dimensions: {dims_preview}')

    if max(capacities) < n_tiles:
        message = (
            'Sampled source images cannot provide the requested tile count before white padding. '
            f'Max sampled capacity was {max(capacities)} tiles, but n_tiles={n_tiles}. '
            'This usually means the source directory contains low-resolution thumbnails '
            'instead of whole-slide images.'
        )
        if fail_on_low_capacity:
            raise RuntimeError(message)
        print(f'WARNING: {message}')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_csv = args.folds_csv if args.folds_csv else None
    image_ids = collect_image_ids(args.slides_dir, folds_csv=folds_csv, limit=args.limit)
    if not image_ids:
        raise RuntimeError('No image_ids found to process')

    pending_ids = []
    skipped = 0
    for image_id in image_ids:
        if not args.overwrite and outputs_exist(output_dir, image_id, args.format):
            skipped += 1
            continue
        pending_ids.append(image_id)

    print(f'{len(image_ids)} slide ids requested')
    print(f'{len(pending_ids)} to process, {skipped} already present')
    if not pending_ids:
        return
    run_preflight_capacity_check(
        pending_ids,
        args.slides_dir,
        args.tile_size,
        args.n_tiles,
        args.level,
        sample_size=args.preflight_sample,
        fail_on_low_capacity=args.fail_on_low_capacity,
    )
    check_disk_budget(output_dir, len(pending_ids), args.tile_size, args.n_tiles, args.format)

    worker_args = (
        args.slides_dir,
        output_dir,
        args.tile_size,
        args.n_tiles,
        args.level,
        args.format,
    )

    if args.n_jobs == 1 or Parallel is None:
        iterator = tqdm(pending_ids, desc='tiles')
        for image_id in iterator:
            process_one(image_id, *worker_args)
    else:
        Parallel(n_jobs=args.n_jobs, prefer='threads')(
            delayed(process_one)(image_id, *worker_args)
            for image_id in tqdm(pending_ids, desc='queue')
        )

    print(f'Finished writing tile artifacts to {output_dir}')


if __name__ == '__main__':
    main()
