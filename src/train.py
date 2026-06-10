"""
Training script. Takes a fold ID, trains a model on the other 4 folds,
evaluates on the held-out fold from `data/train_folds.csv`, saves weights,
and prints final QWK.

Usage:
    python3 -m src.train --fold 0 --image-dir /path/to/train_images --epochs 6
    python3 -m src.train --fold 0 --tile-dir /path/to/tile_artifacts \
        --n-tiles 36 --tile-size 192 --epochs 6
"""
import argparse
import os
import random
import time
from contextlib import nullcontext
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


def seed_everything(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = True


def get_data_dir(args):
    return args.tile_dir or args.image_dir


def use_tiles(args):
    return args.tile_dir is not None


def get_available_ids(data_dir, args):
    allowed_suffixes = None if use_tiles(args) else {'.png'}
    available_ids = set()
    seen_suffixes = set()

    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        suffix = Path(name).suffix.lower()
        if suffix:
            seen_suffixes.add(suffix)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        available_ids.add(Path(name).stem)

    return available_ids, seen_suffixes


def filter_existing_rows(df, data_dir, args):
    available_ids, seen_suffixes = get_available_ids(data_dir, args)
    if not use_tiles(args) and not available_ids:
        suffix_list = ', '.join(sorted(seen_suffixes)) or '(no files found)'
        raise RuntimeError(
            f'No .png thumbnails found in {data_dir}. '
            f'Baseline training expects pre-resized PNGs, but saw: {suffix_list}'
        )
    return df[df.image_id.astype(str).isin(available_ids)].reset_index(drop=True)


def make_datasets(train_df, val_df, args):
    data_dir = get_data_dir(args)
    if use_tiles(args):
        if PandaTileDataset is None:
            raise RuntimeError('Tile training requires PandaTileDataset in src.dataset')
        dataset_cls = PandaTileDataset
        train_kwargs = {'n_tiles': args.n_tiles}
        val_kwargs = {'n_tiles': args.n_tiles}
    else:
        dataset_cls = PandaDataset
        train_kwargs = {}
        val_kwargs = {}
    return (
        dataset_cls(train_df, data_dir, train=True, **train_kwargs),
        dataset_cls(val_df, data_dir, train=False, **val_kwargs),
    )


def make_model(args):
    out_dim = 5 if args.loss == 'ordinal' else 1
    if use_tiles(args):
        return ConcatTilePoolingModel(
            args.backbone,
            pretrained=True,
            dropout=args.dropout,
            out_dim=out_dim,
        )
    return EfficientNetBaseline(
        args.backbone,
        pretrained=True,
        dropout=args.dropout,
        out_dim=out_dim,
    )


def auto_feature_tag(args):
    if args.feature_tag:
        return args.feature_tag
    if not use_tiles(args):
        return None
    parts = ['tiles']
    if args.n_tiles is not None:
        parts[0] = f'tiles{args.n_tiles}'
    if args.tile_size is not None:
        parts.append(f'imsize{args.tile_size}')
    return '_'.join(parts)


def make_weight_path(args):
    parts = [args.backbone.replace('-', '')]
    feature_tag = auto_feature_tag(args)
    if feature_tag is not None:
        parts.append(feature_tag)
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
    return out.detach().cpu().numpy()


def use_amp(args, device):
    return args.amp and device.type == 'cuda'


def pin_memory_enabled(args, device):
    if args.pin_memory is not None:
        return args.pin_memory
    if device.type != 'cuda':
        return False
    if use_tiles(args):
        return False
    return True


def autocast_context(args, device):
    if use_amp(args, device):
        return torch.cuda.amp.autocast()
    return nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True,
                    help='Which fold to hold out (0..4). Trains on the other 4.')
    ap.add_argument('--folds-csv', default='data/train_folds.csv')
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image-dir',
                             help='Directory containing baseline slide PNGs')
    input_group.add_argument('--tile-dir',
                             help='Directory containing one tile artifact per slide')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--amp', action='store_true',
                    help='Enable mixed-precision training on CUDA')
    ap.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=None,
                    help='Override DataLoader pin_memory. Defaults to off for tiles, on for baseline CUDA.')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone', default='efficientnet-b0')
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--loss', choices=['smoothl1', 'mse', 'ordinal'], default='smoothl1')
    ap.add_argument('--ordinal-mode', choices=['threshold', 'expected'], default='threshold',
                    help='How to decode ordinal logits for validation QWK')
    ap.add_argument('--n-tiles', type=int,
                    help='Tile count metadata for experiment naming, e.g. 36')
    ap.add_argument('--tile-size', type=int,
                    help='Tile size metadata for experiment naming, e.g. 192')
    ap.add_argument('--feature-tag',
                    help='Optional experiment tag. Overrides the auto-generated tile tag.')
    ap.add_argument('--output-dir', default='outputs',
                    help='Where to save weights')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--deterministic', action='store_true',
                    help='Enable deterministic cuDNN behavior at the cost of throughput')
    args = ap.parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    data_dir = get_data_dir(args)
    print('Validation: held-out fold QWK from the fold CSV, not Kaggle scoring.')
    print(
        'Config:',
        f'backbone={args.backbone}',
        f'loss={args.loss}',
        f'ordinal_mode={args.ordinal_mode}',
        f'dropout={args.dropout}',
        f'batch_size={args.batch_size}',
        f'num_workers={args.num_workers}',
        f'amp={use_amp(args, device)}',
        f'pin_memory={pin_memory_enabled(args, device)}',
        f'deterministic={args.deterministic}',
        f'feature_tag={auto_feature_tag(args)}',
    )

    df = pd.read_csv(args.folds_csv)
    df = filter_existing_rows(df, data_dir, args)
    if len(df) == 0:
        raise RuntimeError(f'No usable slides found in {data_dir}')
    print(f'{len(df)} slides usable')

    train_df = df[df.fold != args.fold].reset_index(drop=True)
    val_df = df[df.fold == args.fold].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            f'Fold {args.fold} has train={len(train_df)} and val={len(val_df)} after filtering'
        )
    print(f'fold {args.fold}: train={len(train_df)}  val={len(val_df)}')

    train_ds, val_ds = make_datasets(train_df, val_df, args)
    pin_memory = pin_memory_enabled(args, device)
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    model = make_model(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    loss_fn = make_loss(args)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp(args, device))

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
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                out = model(xb)
                loss = loss_fn(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= n

        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                with autocast_context(args, device):
                    out = model(xb)
                preds.append(outputs_to_preds(out, args))
                targs.append(yb.numpy())
        preds = np.concatenate(preds)
        targs = np.concatenate(targs).astype(int)
        val_qwk = qwk(preds, targs, ordinal_mode=args.ordinal_mode)

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
