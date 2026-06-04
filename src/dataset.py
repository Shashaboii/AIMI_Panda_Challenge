"""
PyTorch Dataset for PANDA.

CURRENT STATE: Reads pre-resized 512x512 PNGs from xhlulu's dataset
(the baseline approach). This is enough to train the baseline and serves
as the API contract everyone codes against.

PERSON A: extend this. You can either:
  - Keep PandaDataset, add a `tile_dir` mode that loads tile-grid PNGs
    produced by your tile extractor, OR
  - Add a new class PandaTileDataset that loads multiple tiles per slide
    and returns a tensor of shape [N_tiles, 3, H, W].

The API the training script depends on:
  __init__(df, image_dir, train: bool)
  __len__() -> int
  __getitem__(i) -> (tensor [3, H, W] or [N, 3, H, W], target float)
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE_ARTIFACT_EXTS = ('.npy', '.npz', '.png')


def _augment_hwc_image(img):
    if random.random() < 0.5:
        img = img[:, ::-1, :]
    if random.random() < 0.5:
        img = img[::-1, :, :]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k)
    return img


def _normalize_hwc_image(img):
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def _coerce_rgb_image(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim != 3:
        raise ValueError(f'Expected HWC image, got shape {tuple(img.shape)}')

    if img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] != 3:
        raise ValueError(f'Expected 1, 3, or 4 channels, got shape {tuple(img.shape)}')

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and np.max(img) <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _coerce_tile_array(tiles):
    tiles = np.asarray(tiles)
    if tiles.ndim == 3:
        tiles = tiles[None, ...]
    if tiles.ndim != 4:
        raise ValueError(
            'Expected tile array shaped [N, H, W, C] or [N, C, H, W], '
            f'got {tuple(tiles.shape)}'
        )

    if tiles.shape[-1] in (1, 3, 4):
        hwc_tiles = tiles
    elif tiles.shape[1] in (1, 3, 4):
        hwc_tiles = tiles.transpose(0, 2, 3, 1)
    else:
        raise ValueError(
            'Could not infer channel dimension from tile array '
            f'shape {tuple(tiles.shape)}'
        )

    return np.ascontiguousarray(
        np.stack([_coerce_rgb_image(tile) for tile in hwc_tiles], axis=0)
    )


def _read_rgb_image(path):
    with Image.open(path) as img:
        return np.ascontiguousarray(img.convert('RGB'))


class PandaDataset(Dataset):
    """Baseline dataset: one 512x512 thumbnail per slide.

    df must have columns: image_id, isup_grade.
    image_dir contains <image_id>.png files (xhlulu's format).
    """
    def __init__(self, df, image_dir, train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.train = train

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        return _augment_hwc_image(img)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.image_dir, f'{row.image_id}.png')
        if not os.path.exists(path):
            path = os.path.join(self.image_dir, f'{row.image_id}.tiff')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Could not read {path}')
        img = _read_rgb_image(path)
        if self.train:
            img = self._augment(img)
        img = _normalize_hwc_image(img)
        target = torch.tensor(row.isup_grade, dtype=torch.float32)
        return torch.from_numpy(img), target


class PandaTileDataset(Dataset):
    """Tile dataset: one slide -> [N_tiles, 3, H, W] tensor.

    ``image_dir`` is a directory of per-slide tile artifacts. Supported formats:
      - ``<image_id>.npy`` or ``.npz`` storing ``[N, H, W, 3]`` or ``[N, 3, H, W]``
      - ``<image_id>.png`` storing a vertical stack of tiles
    """
    def __init__(self, df, image_dir, train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.train = train
        self.artifact_paths = self._index_artifacts()

    def __len__(self):
        return len(self.df)

    def _index_artifacts(self):
        artifact_paths = {}
        ext_priority = {ext: i for i, ext in enumerate(TILE_ARTIFACT_EXTS)}
        for name in os.listdir(self.image_dir):
            path = Path(self.image_dir) / name
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in ext_priority:
                continue
            stem = path.stem
            current = artifact_paths.get(stem)
            if current is None:
                artifact_paths[stem] = str(path)
                continue
            current_ext = Path(current).suffix.lower()
            if ext_priority[ext] < ext_priority[current_ext]:
                artifact_paths[stem] = str(path)
        return artifact_paths

    def _load_npz(self, path):
        with np.load(path, allow_pickle=False) as data:
            if 'tiles' in data:
                return data['tiles']
            first_key = next(iter(data.files), None)
            if first_key is None:
                raise ValueError(f'No arrays found in {path}')
            return data[first_key]

    def _load_png_stack(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Could not read {path}')
        img = _coerce_rgb_image(_read_rgb_image(path))
        tile_size = img.shape[1]
        if tile_size == 0 or img.shape[0] % tile_size != 0:
            raise ValueError(
                'PNG tile stacks must be saved as a vertical concat of square tiles; '
                f'got shape {tuple(img.shape)} from {path}'
            )
        n_tiles = img.shape[0] // tile_size
        return img.reshape(n_tiles, tile_size, tile_size, 3)

    def _load_tiles(self, image_id):
        path = self.artifact_paths.get(str(image_id))
        if path is None:
            raise FileNotFoundError(
                f'No tile artifact found for {image_id} in {self.image_dir}; '
                f'looked for {", ".join(TILE_ARTIFACT_EXTS)}'
            )

        ext = Path(path).suffix.lower()
        if ext == '.npy':
            tiles = np.load(path, allow_pickle=False)
        elif ext == '.npz':
            tiles = self._load_npz(path)
        elif ext == '.png':
            tiles = self._load_png_stack(path)
        else:
            raise ValueError(f'Unsupported tile artifact extension {ext!r}')

        return _coerce_tile_array(tiles)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        tiles = self._load_tiles(row.image_id)
        if self.train:
            tiles = np.stack([_augment_hwc_image(tile) for tile in tiles], axis=0)
        tiles = tiles.astype(np.float32) / 255.0
        tiles = (tiles - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
        tiles = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
        target = torch.tensor(row.isup_grade, dtype=torch.float32)
        return torch.from_numpy(tiles), target
