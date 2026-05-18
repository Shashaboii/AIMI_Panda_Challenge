"""
Out-of-fold (OOF) evaluation. Loads all 5 fold weights, predicts on each
fold's validation set with the model that didn't see those slides during
training, and reports:

  - val QWK per fold
  - 5-fold mean ± std
  - The OOF predictions CSV (one prediction per slide)

The OOF CSV is what Person C uses later for label-noise analysis,
calibration, and ensembling.

Usage:
    python -m src.oof \
        --folds-csv data/train_folds.csv \
        --image-dir /kaggle/input/.../train_images \
        --weights-dir /kaggle/working \
        --weight-pattern "efficientnetb0_fold{fold}.pth" \
        --output-csv /kaggle/working/oof_predictions.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PandaDataset
from src.eval import qwk, round_preds
from src.model import EfficientNetBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds-csv', default='data/train_folds.csv')
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--weights-dir', default='/kaggle/working')
    ap.add_argument('--weight-pattern', default='efficientnetb0_fold{fold}.pth',
                    help='Filename pattern with {fold} placeholder')
    ap.add_argument('--output-csv', default='oof_predictions.csv')
    ap.add_argument('--backbone', default='efficientnet-b0')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--n-folds', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = pd.read_csv(args.folds_csv)
    have = set(os.listdir(args.image_dir))
    df = df[df.image_id.apply(lambda x: f'{x}.png' in have)].reset_index(drop=True)
    print(f'{len(df)} usable slides')

    fold_qwks = []
    oof_rows = []

    for fold in range(args.n_folds):
        weight_file = os.path.join(args.weights_dir,
                                   args.weight_pattern.format(fold=fold))
        if not os.path.exists(weight_file):
            print(f'WARN: skipping fold {fold} — no weights at {weight_file}')
            continue

        val_df = df[df.fold == fold].reset_index(drop=True)
        if len(val_df) == 0:
            print(f'WARN: fold {fold} val set is empty, skipping')
            continue

        # Load model with pretrained=False (we're loading saved weights, not ImageNet)
        model = EfficientNetBaseline(args.backbone, pretrained=False).to(device).eval()
        state = torch.load(weight_file, map_location=device)
        model.load_state_dict(state)

        ds = PandaDataset(val_df, args.image_dir, train=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

        preds = []
        with torch.no_grad():
            for xb, _ in tqdm(loader, desc=f'fold {fold}'):
                xb = xb.to(device, non_blocking=True)
                preds.append(model(xb).cpu().numpy())
        preds = np.concatenate(preds)
        targs = val_df.isup_grade.values.astype(int)

        fold_qwk = qwk(preds, targs)
        fold_qwks.append(fold_qwk)
        print(f'fold {fold}: val_QWK = {fold_qwk:.4f}  (n={len(val_df)})')

        # Store per-slide OOF rows
        for iid, p, y in zip(val_df.image_id.values, preds, targs):
            oof_rows.append({
                'image_id': iid,
                'fold': fold,
                'pred_raw': float(p),
                'pred_rounded': int(round_preds([p])[0]),
                'isup_grade': int(y),
            })

    if not fold_qwks:
        raise RuntimeError('No folds processed — check weights directory')

    fold_qwks = np.array(fold_qwks)
    print('\n=== 5-fold summary ===')
    for i, q in enumerate(fold_qwks):
        print(f'  fold {i}: {q:.4f}')
    print(f'  mean ± std: {fold_qwks.mean():.4f} ± {fold_qwks.std():.4f}')

    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(args.output_csv, index=False)
    print(f'\nOOF predictions written to {args.output_csv} ({len(oof_df)} rows)')

    # Also report the global OOF QWK (all slides pooled), not just per-fold mean
    global_qwk = qwk(oof_df.pred_raw.values, oof_df.isup_grade.values)
    print(f'Global OOF QWK (pooled across folds): {global_qwk:.4f}')


if __name__ == '__main__':
    main()
