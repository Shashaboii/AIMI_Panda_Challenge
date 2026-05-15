"""
Produce the shared 5-fold split CSV. Run this ONCE locally (or in any Python
environment with pandas + sklearn), commit the resulting data/train_folds.csv
to git. Don't re-run unless you really need to change the split — everyone's
val QWKs become incomparable if the folds change.

Stratifies by ISUP grade AND data provider so both centres are represented
proportionally in every fold.

Usage:
    python scripts/make_folds.py --train-csv /path/to/train.csv --output data/train_folds.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True,
                    help='Path to PANDA train.csv (with image_id, isup_grade, gleason_score, data_provider)')
    ap.add_argument('--output', default='data/train_folds.csv',
                    help='Where to write the fold CSV')
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    print(f'Loaded {len(df)} rows from {args.train_csv}')
    print(df['data_provider'].value_counts())
    print(df['isup_grade'].value_counts().sort_index())

    # Stratify by joint (isup_grade, data_provider) so both axes are balanced
    strat_key = df['isup_grade'].astype(str) + '_' + df['data_provider'].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, strat_key)):
        df.loc[val_idx, 'fold'] = fold_idx

    assert (df['fold'] >= 0).all(), 'Some rows did not get a fold assigned'

    print()
    print('Fold distribution:')
    print(df.groupby(['fold', 'isup_grade']).size().unstack(fill_value=0))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[['image_id', 'isup_grade', 'gleason_score', 'data_provider', 'fold']].to_csv(out, index=False)
    print(f'Wrote {len(df)} rows to {out}')


if __name__ == '__main__':
    main()
