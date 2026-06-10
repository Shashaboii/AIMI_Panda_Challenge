# One-time setup

For every team member, before starting their track.

## 1. GitHub

1. The repo lives at `https://github.com/<team>/panda`. Owner adds the other two as collaborators (Settings → Collaborators).
2. Each member clones it locally:
   ```bash
   git clone https://github.com/<team>/panda.git
   cd panda
   ```
3. Configure git with your name + Radboud email so commits are attributable:
   ```bash
   git config user.name "Your Name"
   git config user.email "your.email@ru.nl"
   ```

## 2. Local Python environment (optional)

You don't strictly need a local Python environment — all training happens on Kaggle. But it's useful for editing and running small scripts (e.g. `scripts/make_folds.py`).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Kaggle

1. Create a Kaggle account if you don't have one. Verify your phone number — required for GPU access.
2. Enable GPU/TPU access in your account settings.
3. Accept the PANDA competition rules at https://www.kaggle.com/competitions/prostate-cancer-grade-assessment — required to access the data. (Yes, late submissions don't get scored, but you still need to accept rules to mount the data.)

## 4. Generate the fold split (one person, once)

The fold split is a frozen artifact — don't regenerate without a team sync. Person A typically does this on their local machine and commits the resulting CSV.

```bash
# Download train.csv from Kaggle to your local machine first
python scripts/make_folds.py --train-csv ~/Downloads/train.csv --output data/train_folds.csv
git add data/train_folds.csv
git commit -m "Add 5-fold split"
git push
```

## 5. Test the baseline notebook on Kaggle

To verify everything is wired up:

1. Open https://www.kaggle.com → Create → New Notebook
2. File → Import Notebook → upload `notebooks/02_train.ipynb` from your local clone
3. Edit the `REPO` variable to point at your team's GitHub URL
4. Add Data → attach `panda-resized-train-data-512x512` (by xhlulu)
5. Settings → Accelerator: GPU T4 × 2, Internet: On
6. Run all cells interactively for a smoke test (each cell should pass quickly)
7. If smoke test passes, Save & Run All (Commit) for a full ~30-minute training run
8. Verify the resulting val QWK is around 0.70

If you can run the baseline and reproduce ~0.70, your environment is set up correctly.

## 6. Read your track doc

Open `docs/track_A_data.md`, `docs/track_B_model.md`, or `docs/track_C_eval.md` depending on which track you own. That's your week-by-week plan.

## 7. Read CLAUDE.md

For any Claude conversation about this project, paste CLAUDE.md at the start. Your teammates do the same — keeps everyone's AI conversations on the same page.
