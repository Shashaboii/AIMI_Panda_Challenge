# Track C — Evaluation, ensembling, and external validation

Last updated: 2026-06-04

You own how we measure model quality and how the final predictions are produced.

## Files you own

- `src/eval.py` — shared QWK function
- `src/inference.py` — inference and ensembling helpers
- `src/oof.py` — fair out-of-fold evaluation script
- `notebooks/02b_train_all_folds.ipynb` — train-all-folds runner
- `notebooks/02c_ensemble_oof.ipynb` — fair ensemble OOF notebook
- `notebooks/03_external_validation.ipynb` — PBGG-1 evaluation

## Current status

- The shared QWK utilities are in place.
- The repo already has a working OOF pipeline through `src.oof`.
- Clean 5-fold thumbnail baseline is logged at `0.7116` global OOF QWK.
- Clean 5-fold `b0 + mse` is logged at `0.7030` global OOF QWK.
- 5-fold `b0 + ordinal` weights already exist and should be included in the next fair ensemble comparison.
- `ensemble_predict(...)` exists, and `02c_ensemble_oof.ipynb` exists for fair fold-wise ensembling.
- Track A already has a first tile dataset, so external validation and ensemble planning should assume tile-family weights are expected later, not that tile artifacts are missing entirely.
- `03_external_validation.ipynb` is still missing and PBGG-1 is not yet part of the team workflow.

## Objective from here to project success

Own the team's final answer to "did this experiment actually help?" by:

1. keeping evaluation fair
2. producing ensemble OOF numbers
3. building the external validation pipeline
4. turning all of that into paper-ready tables

## What is left right now

1. Run `src.oof` or the equivalent notebook flow on the existing 5-fold `b0 + ordinal` family and log the result if it is not already in `results.md`.
2. Run a fair OOF ensemble on the model families that already have all 5 folds.
3. Create or attach one Kaggle Dataset per weight family for the ensemble notebook.
4. Log the ensemble result in `results.md` with:
   - per-family mean ± std
   - ensemble mean ± std
   - global OOF QWK
5. Re-run the ensemble summary after any new tile-family 5-fold weights arrive.
6. Create `notebooks/03_external_validation.ipynb`.
7. Mirror PBGG-1 to a Kaggle Dataset the team can attach.
8. Build the PBGG-1 loader, compute majority-vote labels, and report QWK.
9. Add TTA only to the final selected model or ensemble, not to exploratory runs.

## Recommended order

1. Use `02c_ensemble_oof.ipynb` to validate the ensemble pipeline now.
2. Start with `b0 + SmoothL1` plus `b0 + ordinal`, since both are expected to be full 5-fold families.
3. Add or compare `b0 + MSE` only if it improves the ensemble or helps analysis.
4. Re-run the ensemble with stronger tile families once they exist.
5. Only after the final family or ensemble is chosen should you spend time on TTA and external-validation polish.

## Fairness rules that must not be broken

- For validation fold `f`, only use the checkpoint trained with fold `f` held out.
- Do not average all five fold checkpoints onto the same validation fold.
- Do not compare experiments across different fold splits.
- Always use `src.eval.qwk(...)` as the source of truth.

## What a "weight family" means

For ensembling, one family means one training recipe across all 5 folds.

Examples:

- `b0 + SmoothL1`, folds `0..4`
- `b0 + MSE`, folds `0..4`
- `b0 + ordinal`, folds `0..4`

Each family should live in its own Kaggle Dataset so the ensemble notebook can attach it cleanly.

## Done when

- `results.md` has a paper-ready ablation table for all serious experiments.
- There is at least one fair ensemble OOF result logged.
- `03_external_validation.ipynb` runs end to end on Kaggle.
- The team has final PANDA CV numbers and PBGG-1 numbers ready for the paper.
