"""
Training script. Takes a fold ID, trains a model on the other 4 folds,
evaluates on the held-out fold from `data/train_folds.csv`, saves weights,
and prints final QWK.

Usage:
    python3 -m src.train --fold 0 --image-dir data/train_folds.csv --epochs 6
    python -m src.train --fold 0 --tile-dir /path/to/tile_artifacts --epochs 6
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PandaDataset
from src.eval import qwk
from src.model import ConcatTilePoolingModel, EfficientNetBaseline

try:
    from src.dataset import PandaTileDataset
except ImportError:
    PandaTileDataset = None


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_data_dir(args):
    return args.tile_dir or args.image_dir


def use_tiles(args):
    return args.tile_dir is not None


def filter_existing_rows(df, data_dir):
    available_ids = {
        Path(name).stem
        for name in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, name))
    }
    return df[df.image_id.astype(str).isin(available_ids)].reset_index(drop=True)


def make_datasets(train_df, val_df, args):
    data_dir = get_data_dir(args)
    if use_tiles(args):
        if PandaTileDataset is None:
            raise RuntimeError('Tile training requires PandaTileDataset in src.dataset')
        dataset_cls = PandaTileDataset
    else:
        dataset_cls = PandaDataset
    return (
        dataset_cls(train_df, data_dir, train=True),
        dataset_cls(val_df, data_dir, train=False),
    )


def make_model(args):
    out_dim = 5 if args.loss == 'ordinal' else 1
    if use_tiles(args):
        return ConcatTilePoolingModel(args.backbone, pretrained=True, out_dim=out_dim)
    return EfficientNetBaseline(args.backbone, pretrained=True, out_dim=out_dim)


def make_weight_path(args):
    parts = [args.backbone.replace('-', '')]
    if args.feature_tag:
        parts.append(args.feature_tag)
    elif use_tiles(args):
        parts.append('tiles')
    if args.loss != 'smoothl1':
        parts.append(args.loss)
    parts.append(f'fold{args.fold}')
    return '_'.join(parts) + '.pth'


def make_loss(args):
    if args.loss == 'mse':
        return nn.MSELoss()
    if args.loss == 'ordinal':
        return nn.BCEWithLogitsLoss()
    return nn.SmoothL1Loss()


def make_targets(y, args):
    if args.loss != 'ordinal':
        return y
    thresholds = torch.arange(5, device=y.device, dtype=y.dtype)
    return (y.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def outputs_to_preds(out, args):
    if args.loss == 'ordinal':
        return (out > 0).sum(dim=1).cpu().numpy()
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True,
                    help='Which fold to hold out (0..4). Trains on the other 4.')
    ap.add_argument('--folds-csv', default='data/train_folds.csv')
    ap.add_argument('--image-dir',
                    help='Directory containing baseline slide PNGs')
    ap.add_argument('--tile-dir',
                    help='Directory containing one tile artifact per slide')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone', default='efficientnet-b0')
    ap.add_argument('--loss', choices=['smoothl1', 'mse', 'ordinal'], default='smoothl1')
    ap.add_argument('--feature-tag',
                    help='Optional experiment tag, e.g. tiles36_imsize192')
    ap.add_argument('--output-dir', default='outputs',
                    help='Where to save weights')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    data_dir = get_data_dir(args)
    if data_dir is None:
        raise ValueError('Pass --image-dir for baseline training or --tile-dir for tile training.')
    print('Validation: held-out fold QWK from the fold CSV, not Kaggle scoring.')

    df = pd.read_csv(args.folds_csv)
    df = filter_existing_rows(df, data_dir)
    print(f'{len(df)} slides usable')

    train_df = df[df.fold != args.fold].reset_index(drop=True)
    val_df = df[df.fold == args.fold].reset_index(drop=True)
    print(f'fold {args.fold}: train={len(train_df)}  val={len(val_df)}')

    train_ds, val_ds = make_datasets(train_df, val_df, args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = make_model(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    loss_fn = make_loss(args)

    os.makedirs(args.output_dir, exist_ok=True)
    weight_path = os.path.join(args.output_dir, make_weight_path(args))
    best_qwk = -1.0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f'ep{epoch} train'):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            targets = make_targets(yb, args)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= n

        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                preds.append(outputs_to_preds(model(xb), args))
                targs.append(yb.numpy())
        preds = np.concatenate(preds)
        targs = np.concatenate(targs).astype(int)
        val_qwk = qwk(preds, targs)

        elapsed = (time.time() - t0) / 60.0
        print(f'ep{epoch}  train_loss={train_loss:.4f}  val_QWK={val_qwk:.4f}  '
              f'time={elapsed:.1f}m')
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), weight_path)
            print(f'  saved {weight_path}')

    print(f'\nBest val QWK fold {args.fold}: {best_qwk:.4f}')
    return best_qwk


if __name__ == '__main__':
    main()
