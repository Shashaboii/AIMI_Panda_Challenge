# data/

This directory holds **only** the fold-split CSV.

Large data (images, weights) lives in Kaggle Datasets, never in git.

## Files expected here

- `train_folds.csv` — produced by `scripts/make_folds.py`, frozen once produced. Schema:
  ```
  image_id, isup_grade, gleason_score, data_provider, fold
  ```

Do **not** commit `train_images/`, `test_images/`, or any `.tiff`/`.png`. The `.gitignore` already excludes them.
