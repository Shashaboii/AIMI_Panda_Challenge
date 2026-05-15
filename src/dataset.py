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

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        if random.random() < 0.5:
            img = img[:, ::-1, :]
        if random.random() < 0.5:
            img = img[::-1, :, :]
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
        return img

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.image_dir, f'{row.image_id}.png')
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Could not read {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train:
            img = self._augment(img)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        target = torch.tensor(row.isup_grade, dtype=torch.float32)
        return torch.from_numpy(img), target
