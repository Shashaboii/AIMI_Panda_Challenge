"""
Training script. Takes a fold ID, trains a model on the other 4 folds,
evaluates on the held-out fold, saves weights and prints final QWK.

Usage:
    python -m src.train --fold 0 --image-dir /path/to/train_images --epochs 6

The training notebook on Kaggle is a thin wrapper around this — see
notebooks/02_train.ipynb.
"""
import argparse
import os
import random
import time

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
from src.model import EfficientNetBaseline


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True,
                    help='Which fold to hold out (0..4). Trains on the other 4.')
    ap.add_argument('--folds-csv', default='data/train_folds.csv')
    ap.add_argument('--image-dir', required=True,
                    help='Directory containing <image_id>.png files')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone', default='efficientnet-b0')
    ap.add_argument('--output-dir', default='/kaggle/working',
                    help='Where to save weights')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = pd.read_csv(args.folds_csv)
    # Filter to slides that exist on disk
    have = set(os.listdir(args.image_dir))
    df = df[df.image_id.apply(lambda x: f'{x}.png' in have)].reset_index(drop=True)
    print(f'{len(df)} slides usable')

    train_df = df[df.fold != args.fold].reset_index(drop=True)
    val_df = df[df.fold == args.fold].reset_index(drop=True)
    print(f'fold {args.fold}: train={len(train_df)}  val={len(val_df)}')

    train_ds = PandaDataset(train_df, args.image_dir, train=True)
    val_ds = PandaDataset(val_df, args.image_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = EfficientNetBaseline(args.backbone, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    loss_fn = nn.SmoothL1Loss()

    os.makedirs(args.output_dir, exist_ok=True)
    weight_path = os.path.join(args.output_dir,
                               f'{args.backbone.replace("-", "")}_fold{args.fold}.pth')
    best_qwk = -1.0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f'ep{epoch} train'):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
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
                preds.append(model(xb).cpu().numpy())
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
