# Track C — Evaluation, ensembling, and external validation

You own how we measure model quality and how the final predictions are produced.

## Files you own

- `src/eval.py` — shared QWK function (already exists, modify carefully)
- `src/inference.py` — inference and ensembling helpers (you'll create)
- `notebooks/03_external_validation.ipynb` — PBGG-1 evaluation
- The team's "did this experiment work" answer

## Why this track matters

**Kaggle leaderboard submissions don't get scored anymore** (the competition closed in 2020 and late submissions return "after deadline" without a score). So our evaluation strategy is:

1. **Primary**: 5-fold stratified cross-validation on the PANDA training set. Mean ± std QWK across 5 folds. This is the same methodology as Bulten et al. (Nature Medicine 2022) and the 1st-place solution.
2. **Secondary**: external validation on the publicly available PBGG-1 dataset from Tolkach et al. 2023 (npj Precision Oncology). 50 slides graded by 10 pathologists. Lets us compare directly to published medical AI numbers.

This is a stronger evaluation than a single leaderboard score would have been.

## Week 1 — Get cross-validation working

The fold-split already exists in `data/train_folds.csv`. Your job week 1:

- Write `src/inference.py` with a function `predict_oof(model, df, image_dir) -> np.ndarray` that returns per-slide predictions for the validation fold
- Set up an "experiment runner" notebook that, for any given config, trains 5 folds and reports `mean ± std` val QWK across them
- Verify by running the baseline 5 times on 5 folds — you should see numbers around 0.68–0.72 with some variation

## Week 2 — Ensembling

When Person B's first multi-fold weights exist, write the ensembling logic:

```python
def ensemble_predict(models, x):
    """Average predictions across N trained models."""
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(x).cpu().numpy())
    return np.mean(preds, axis=0)
```

Verify that 5-fold ensemble val QWK is higher than any single fold's val QWK. Typical lift: +0.02 QWK.

## Week 3 — External validation on PBGG-1

Download the PBGG-1 dataset from Zenodo: https://zenodo.org/records/8102833

50 OME-TIFF slides plus a CSV of 10 pathologists' ISUP gradings. To use:

1. Add as a Kaggle Dataset for the team
2. Write a small loader (it's only 50 slides, no fancy infrastructure needed) — likely OME-TIFF format, may need `tifffile` to read
3. Compute majority-vote labels per slide
4. Run your ensemble on each slide
5. Report QWK against the majority labels

Also report inter-pathologist agreement on the same slides for context — Tolkach et al. reported pathologist range 0.62–0.80, so if your model lands in that range, you're at pathologist-level performance.

## Week 4 — Test-time augmentation and final submission

TTA is a free ~0.005 QWK lift. Predict on each test slide, its horizontal flip, its 90° rotation, etc., then average. Should add maybe 10 lines to `src/inference.py`.

Then produce the final numbers for the paper:
- Per-fold val QWK (5 numbers)
- Mean ± std val QWK
- External validation QWK on PBGG-1
- Comparison to pathologist range (with citation)

## What success looks like

- An ablation table in `results.md` with mean ± std for every experiment the team has run
- A working external validation pipeline that runs in <10 minutes on Kaggle
- A clean inference notebook (`notebooks/03_external_validation.ipynb`) that any team member can re-run
- Numbers ready for the paper

## Where to start Monday

1. Pull the latest from main
2. Read `src/eval.py` — that's your single source of truth for QWK; everyone imports from there
3. Reproduce the baseline val QWK 0.70 on fold 0 to confirm the pipeline works for you locally / in a Kaggle notebook
4. Write a script that trains all 5 folds and prints the mean — run it once with the baseline so we have a real "5-fold mean baseline QWK" number to beat
