# Automated Gleason / ISUP Grading of Prostate Biopsies (PANDA)

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
| EfficientNet-B0 (public, 30 epochs — reference) | — | — | — | 0.915 |

\* predicted benign (ISUP 0) when cancer was present — the clinically dangerous error.
\*\* predicted cancer when truly benign.

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
