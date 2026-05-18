# PANDA — AIMI 2526 project

Final project for the AI in Medical Imaging course (NWI-IMC037) at Radboud University, spring 2026.

## What this is

A pipeline for predicting ISUP grade (0–5) from whole-slide H&E-stained prostate biopsy images, scored by quadratic-weighted Cohen's kappa.

We re-implement the techniques from [Team PND's 1st-place solution](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution) that matter most for the score:
- Tile-based input
- Concat-tile-pooling architecture
- 5-fold cross-validation with averaging

…and skip the more expensive engineering pieces (label-noise cleaning, multi-architecture ensembles).

## Evaluation

The PANDA competition closed in 2020 and late submissions no longer go through scoring. Our evaluation strategy:
1. **Primary**: 5-fold stratified cross-validation on the public training set
2. **Secondary**: external validation on the [PBGG-1 dataset](https://zenodo.org/records/8102833) from Tolkach et al. 2023 (npj Precision Oncology)

This matches the methodology of Bulten et al.'s Nature Medicine paper.

## Repo layout

```
panda/
├── CLAUDE.md                       # context file for Claude conversations
├── README.md                       # this file
├── results.md                      # running log of experiments
├── requirements.txt                # Python deps
├── data/
│   └── train_folds.csv             # shared 5-fold split (generated once, frozen)
├── scripts/
│   └── make_folds.py               # generates the fold split CSV
├── src/
│   ├── dataset.py                  # PyTorch Dataset (Person A extends)
│   ├── model.py                    # model definitions (Person B extends)
│   ├── train.py                    # training script (CLI)
│   ├── inference.py                # inference helpers (Person C extends)
│   └── eval.py                     # shared QWK function
├── notebooks/
│   └── 02_train.ipynb              # thin Kaggle wrapper around src/train.py
└── docs/
    ├── track_A_data.md             # kickoff doc for Person A
    ├── track_B_model.md            # kickoff doc for Person B
    └── track_C_eval.md             # kickoff doc for Person C
```

## Workflow

We use **Kaggle for compute** and **GitHub for code**. Code lives in `src/`. Kaggle notebooks are thin wrappers that clone this repo and run `python -m src.train` (or similar).

To run training on Kaggle:
1. Open `notebooks/02_train.ipynb` on Kaggle
2. Edit the `REPO`, `BRANCH`, `FOLD` variables at the top
3. Attach the `panda-resized-train-data-512x512` dataset (by xhlulu) as input
4. Save & Run All (Commit)
5. Find the trained weights in `/kaggle/working/`, publish as a Kaggle Dataset

To make a code change:
1. Pull from main
2. Make changes on a feature branch
3. Push, run training on Kaggle from that branch
4. Log results in `results.md`
5. Open a PR, get one teammate to review, merge

## Reproducing the baseline

```bash
# Local — generate the fold split (only ever done once)
python scripts/make_folds.py --train-csv /path/to/train.csv --output data/train_folds.csv

# On Kaggle — train one fold of the baseline
python -m src.train \
    --fold 0 \
    --folds-csv data/train_folds.csv \
    --image-dir /kaggle/input/panda-resized-train-data-512x512/train_images/train_images \
    --epochs 6 \
    --batch-size 16 \
    --output-dir /kaggle/working
```

Expected output: `effnetb0_fold0.pth` in `/kaggle/working` with val QWK ≈ 0.70.

## Team

| Person | Track | Files |
|---|---|---|
| _[name]_ | Data pipeline | `src/dataset.py`, `src/tiles.py` (TBA), tile preprocessing |
| _[name]_ | Model & training | `src/model.py`, `src/train.py` |
| _[name]_ | Eval & external validation | `src/eval.py`, `src/inference.py`, external validation |
following the ideas described in their writeup, not a derivative work.
