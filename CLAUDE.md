# CLAUDE.md — PANDA project context

Paste this file (or upload it) at the start of any Claude conversation about this project. It keeps the team aligned across Claude sessions.

**Maintainers:** all three teammates. Update after each Friday sync.
**Last updated:** 2026-05-13

---

## Project

PANDA (Prostate cANcer graDe Assessment) Kaggle challenge as the AIMI 2526 final project at Radboud University. Course code NWI-IMC037, supervisor Colin Jacobs (`colin.jacobs@ru.nl`). Project runs Apr 6 – Jun 5 2026; presentations Jun 12.

**Task:** predict ISUP grade (0–5) for whole-slide images of prostate biopsies.
**Metric:** quadratic-weighted Cohen's kappa (QWK), computed by `src/eval.qwk()`.
**Reference implementation we're learning from:** [kentaroy47/Kaggle-PANDA-1st-place-solution](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution). We re-implement the high-impact parts, not copy.

## Evaluation strategy

**The Kaggle leaderboard is unavailable.** PANDA closed in 2020 and late submissions return "Notebook Threw Exception (after deadline)" without a real score. We confirmed this empirically.

So our evaluation is:
1. **Primary**: 5-fold stratified CV on the PANDA training set (stratified jointly by ISUP grade and data provider). Mean ± std QWK across 5 folds. Fold assignment in `data/train_folds.csv` is fixed — don't regenerate it.

This keeps the project evaluation simple, reproducible, and internally consistent.

## Team and ownership

Each track owns its own code module(s) in `src/`. Cross-cutting changes go to the Friday sync.

- **Person A — Data pipeline.** Owns `src/dataset.py`, `src/tiles.py` (to be created), and the `data/train_folds.csv`. See `docs/track_A_data.md`.
- **Person B — Model and training.** Owns `src/model.py`, `src/train.py`. See `docs/track_B_model.md`.
- **Person C — Eval and ensembling.** Owns `src/eval.py`, `src/inference.py`, and OOF / ensemble reporting. See `docs/track_C_eval.md`.

## Current status

- **Baseline trained**: EfficientNet-B0, 512×512 thumbnails (xhlulu's dataset), fold 0 of 5-fold split, 6 epochs, regression + SmoothL1, **val QWK 0.70**. Weights: Kaggle Dataset `panda-effnet-b0-weights`.
- **Scaffold ready**: shared repo with `src/` modules and `data/train_folds.csv`.
- **Next milestone**: tile-based input pipeline (Person A) + concat-tile-pooling model (Person B), targeting val QWK > 0.78.

## Goal

Final submission with **5-fold mean QWK ≥ 0.83 on PANDA**, plus a 4-page IEEE paper and reproducible GitHub repo.

Not trying to beat the winners' 0.94 — trying to learn the techniques and execute them cleanly in 4 weeks. A strong reproducible PANDA result is better than an overextended project plan.

## Approach (minimum viable plan)

Three techniques from the winners' solution, ranked by impact per unit effort:
1. **Tile-based input** (Person A). ~+0.05–0.08 QWK.
2. **Concat-tile-pooling architecture** (Person B). ~+0.02–0.04 QWK.
3. **5-fold cross-validation with averaging** (Person C). ~+0.02 QWK.

Optional in week 4 if time: mixup, TTA. **Out of scope**: label noise cleaning, two-architecture ensemble, imagehash deduplication.

## Shared decisions — don't break without a sync

- **K-fold split is fixed.** `data/train_folds.csv` is the source of truth. Re-running `scripts/make_folds.py` makes everyone's old val QWKs incomparable. Don't.
- **Evaluation function**: `src/eval.qwk(preds, targets)`. Always import from there, never reimplement.
- **ISUP grade as float** for training, rounded at evaluation. SmoothL1 loss is the default.
- **Image normalization**: ImageNet mean/std `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`. Defined in `src/dataset.py`.
- **Weight naming**: `<backbone>_<feature>_fold<F>.pth` — e.g. `effnetb0_tiles36_fold0.pth`, `effnetb1_mixup_fold2.pth`.
- **No notebook outputs in git.** Notebooks committed should be cleared (run `jupyter nbconvert --clear-output` before committing).
- **Results log**: every meaningful experiment goes in `results.md` with date, who, hyperparams, val QWK, Kaggle Dataset slug for weights.

## Workflow

**Compute**: Kaggle (free T4 GPUs, ~30 GPU hrs/week per account). Cluster access is being requested — TBD.
**Code**: GitHub. Code lives in `src/`. Kaggle notebooks are thin wrappers that clone the repo and run `python -m src.train`.

Per teammate, per change:
1. `git pull` on your local clone
2. Edit `src/...`. If the change affects another track, post in chat before pushing.
3. Push to a feature branch (e.g. `personA/tiles-stride-experiment`)
4. On Kaggle: in `notebooks/02_train.ipynb`, set `BRANCH = 'personA/tiles-stride-experiment'`, Save & Run All (Commit). Auto-clones + runs.
5. Save trained weights as a Kaggle Dataset. Note slug in `results.md`.
6. Open PR on GitHub. Merge to `main` after one teammate reviews.

## How to use this file with Claude

Paste this file's contents (or upload it as a file) at the start of any Claude conversation about this project. Claude then has the same context the rest of the team has.

If a Claude conversation reaches a decision that changes shared decisions or current status, update this file and push it. Don't let it go stale.

Example good prompts:
- *"Here's our project context [paste CLAUDE.md]. I'm Person A working on tiles.py. Help me write the tile-ranking function."*
- *"Here's our context [paste CLAUDE.md]. Val QWK plateaus at 0.78 with tiles. What to try before mixup?"*
- *"Here's our context [paste CLAUDE.md]. The training script errors with X. Where to look?"*

## Notes / open questions

- Cluster access: pending — to ask Colin Jacobs at next supervision meeting.
- Kaggle GPU quota across three accounts is ~90 hrs/week, plenty for the planned experiments.
