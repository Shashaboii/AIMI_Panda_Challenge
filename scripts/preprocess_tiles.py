"""Precompute per-slide PANDA tile artifacts for tile-based training."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tiles import extract_tiles_from_slide, save_tile_array, save_tile_stack

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib is in requirements, but keep the fallback simple.
    Parallel = None
    delayed = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides-dir', required=True,
                    help='Directory containing raw PANDA .tiff/.tif slides')
    ap.add_argument('--output-dir', required=True,
                    help='Directory to write one tile artifact per slide')
    ap.add_argument('--folds-csv', default='data/train_folds.csv',
                    help='Optional CSV used to define which image_ids to process')
    ap.add_argument('--tile-size', type=int, default=192)
    ap.add_argument('--n-tiles', type=int, default=36)
    ap.add_argument('--level', type=int, default=1,
                    help='Pyramid level to read from the TIFF slide')
    ap.add_argument('--format', choices=['npy', 'png', 'both'], default='npy',
                    help='Artifact format to save')
    ap.add_argument('--limit', type=int,
                    help='Optional cap for quick smoke tests')
    ap.add_argument('--n-jobs', type=int, default=1,
                    help='Parallel workers; 1 keeps execution sequential')
    ap.add_argument('--overwrite', action='store_true',
                    help='Recompute artifacts even if outputs already exist')
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
