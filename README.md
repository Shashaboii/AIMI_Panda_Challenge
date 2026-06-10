
# Prostate cANcer graDe Assessment (PANDA) Challenge

Predicting the **ISUP grade (0–5)** of prostate-biopsy whole-slide images (WSIs) from
the [PANDA challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment).
We compare two approaches on the same task and tiling pipeline:

1. **Baseline CNN** — EfficientNet-B0 trained from scratch on tile montages (the 2020
   competition recipe).
2. **Foundation model** — frozen [**UNI**](https://huggingface.co/MahmoodLab/UNI)
   (a pathology ViT-L) as a tile encoder + a small gated-attention MIL head (a 2024
   approach), including a **cross-centre generalisation** study.

The metric is **quadratic weighted kappa (QWK)**.

---

## Results

Validation is fold 0 (`StratifiedKFold`, `random_state=42`), 2,124 held-out slides.

| Model | Val QWK | Missed cancers* | False alarms** | Leaderboard |
|---|---|---|---|---|
| EfficientNet-B0 (ours, 20 epochs from cache) | **0.876** | 43 | 72 | ~0.89 |
| UNI + attention-MIL | 0.859 | 62 | 120 | ~0.86 |


\ predicted benign (ISUP 0) when cancer was present,the clinically dangerous error.
\ predicted cancer when truly benign.

**Cross-centre generalisation** (train one centre, test the other) is produced by
`06_uni_train_mil_head.ipynb` — see its `cross()` output.

**Data-quality finding:** ISUP is derived deterministically from the Gleason score, so
we checked all 10,616 slides for consistency — only **1 mismatch (0.01%)**. The
recorded labels are therefore internally consistent; the residual label noise comes
from grading subjectivity (inter-observer variability), not data-entry errors, which is
why model errors cluster on **adjacent** grades rather than on the mismatch slides.

---

## Repository structure

```
panda-isup-grading/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_tile_caching.ipynb            # build the 36×256 montage cache (run once)
│   ├── 02_eda_data_understanding.ipynb  # EDA + label-consistency + error analysis
│   ├── 03_train_efficientnet_b0.ipynb   # train EffNet-B0 from the cache
│   ├── 03b_dump_effnet_val_predictions.ipynb # val preds -> results/val_df_effnet.csv
│   ├── 04_inference_efficientnet_b0.ipynb # EffNet test inference -> submission.csv
│   ├── 05_uni_extract_embeddings.ipynb  # frozen UNI -> per-slide embeddings (run once)
│   ├── 06_uni_train_mil_head.ipynb      # attention-MIL head + cross-centre study
│   └── 07_uni_inference.ipynb           # UNI test inference -> submission.csv
└── results/
    ├── val_df_effnet.csv                # EffNet val predictions (from 03b)
    └── val_df_uni.csv                   # UNI val predictions (from 06)
```

---

## Data & artefacts

Everything runs on **Kaggle** (GPU). The pipeline produces intermediate artefacts that
are saved as Kaggle Datasets and consumed by later notebooks:

| Dataset | Produced by | Consumed by | Contents |
|---|---|---|---|
| `prostate-cancer-grade-assessment` (competition) | — | all | raw WSIs + `train.csv` |
| `panda-tiles-36x256` | 01 | 02, 03 | one 1536×1536 montage JPEG per slide + `train.csv` |
| `panda-effnetb0-fold0` | 03 | 04 | `effnetb0_36x256_fold0_best.pth` |
| `panda-uni-embeddings1` | 05 | 06 | one `36×1024` `.npy` per slide + `train.csv` |
| `uni-weights` | 05 | 07 | `uni_pytorch_model.bin` (for offline inference) |
| `panda-uni-head` | 06 | 07 | `uni_mil_fold0.pth` |

> Paths in the notebooks point at the author's Kaggle mounts
> (`/kaggle/input/datasets/<user>/...`). Re-point `CACHE_DIR`, `MODEL_DIR`, `EMB_DIR`,
> `UNI_WEIGHTS`, `HEAD_PATH` to wherever your datasets mount.

---

## How to reproduce (run order)

| # | Notebook | GPU | Internet | Attach | Notes |
|---|---|---|---|---|---|
| 1 | `01_tile_caching` | yes | on | competition | builds cache (~slow, one-time); publish `tiles_cache` as `panda-tiles-36x256` |
| 2 | `02_eda_data_understanding` | no | on | competition (+ val CSVs) | EDA, label check, error analysis |
| 3 | `03_train_efficientnet_b0` | yes | on | competition + cache | trains EffNet-B0; publish `*_best.pth` |
| 3b | `03b_dump_effnet_val_predictions` | yes | on | competition + cache + `panda-effnetb0-fold0` | dumps `results/val_df_effnet.csv` for error analysis |
| 4 | `04_inference_efficientnet_b0` | yes | **off** | competition + `panda-effnetb0-fold0` | writes `submission.csv` |
| 5 | `05_uni_extract_embeddings` | yes | on + `HF_TOKEN` | competition | frozen UNI embeddings; publish `panda-uni-embeddings1` + `uni-weights` |
| 6 | `06_uni_train_mil_head` | yes | on | `panda-uni-embeddings1` | trains MIL head + cross-centre; publish `panda-uni-head`, dumps `val_df_uni.csv` |
| 7 | `07_uni_inference` | yes | **off** | competition + `uni-weights` + `panda-uni-head` | writes `submission.csv` |

**HF token (notebook 05 only):** UNI is gated. Create a read token at huggingface.co,
add it as a Kaggle **Secret** named `HF_TOKEN`, and enable it for the notebook.

**Internet-off notebooks (04, 07):** do **not** `pip install` anything — Kaggle's
preinstalled `timm`/`openslide` are used, and weights load from attached datasets.

---

## Method notes (what makes the weights valid)

* **Tiling:** read TIFF level 1 with openslide → split into 256×256 tiles → keep the 36
  with the most tissue → stitch into a 6×6 (1536×1536) montage.
* **CNN montages are colour-inverted** (`255 − tile`); **UNI tiles are NOT inverted**
  (UNI expects normal H&E, ImageNet-normalised, resized to 224). Mixing these up
  silently destroys accuracy.
* **Ordinal target:** grade *k* → `[1]*k + [0]*(5−k)`, trained with `BCEWithLogitsLoss`;
  at inference, `round(sum(sigmoid(logits)))`. This respects grade ordering and targets QWK.
* **Cache once, train fast:** the slow part is reading WSIs; `01` does it once so every
  later run loads small JPEGs / `.npy` files.
* **TTA:** EffNet inference averages two tilings (grid offsets 0 and 2).

---

## Environment

See `requirements.txt`. Two system notes:
* **openslide** needs the system library (`apt-get install -y openslide-tools`) in
  addition to `openslide-python`; both are preinstalled on Kaggle GPU images.
* Trained and validated on Kaggle (Python 3.12, single P100/T4, CUDA, mixed precision).

---

## Attribution

* Tiling + montage + ordinal-BCE recipe follows the public PANDA baselines by
  *iafoss* (concat-tile pooling) and *haqishen* (36×256, LB ≈ 0.87).
* UNI foundation model: Chen et al., *Towards a general-purpose foundation model for
  computational pathology*, Nature Medicine 2024 (`MahmoodLab/UNI`).
* PANDA challenge / dataset: Bulten et al., *Nature Medicine* 2022.
=======
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

## Latest baseline result

Clean 5-fold baseline rerun on the xhlulu 512x512 thumbnail dataset:

- Backbone: EfficientNet-B0
- Loss: SmoothL1
- Dropout: 0.3
- Epochs: 6 per fold
- Data split: shared 5-fold split in `data/train_folds.csv`

Fold validation QWKs:

```text
fold 0: 0.7117
fold 1: 0.7047
fold 2: 0.7294
fold 3: 0.6980
fold 4: 0.7142
mean ± std: 0.7116 ± 0.0106
global OOF QWK: 0.7116
```

OOF predictions were written to `/kaggle/working/efficientnetb0_oof_predictions.csv`
with 10,616 rows. The confusion matrix shows most errors are near-misses between
adjacent grades, with the strongest performance on ISUP 0 and the weakest
performance on ISUP 5.

## Team

| Person | Track | Files |
|---|---|---|
| _[name]_ | Data pipeline | `src/dataset.py`, `src/tiles.py` (TBA), tile preprocessing |
| _[name]_ | Model & training | `src/model.py`, `src/train.py` |
| _[name]_ | Eval & external validation | `src/eval.py`, `src/inference.py`, external validation |
following the ideas described in their writeup, not a derivative work.
>>>>>>> 4c19ed0daf0a74d611e9f6d0ff6c50d43949ec11
