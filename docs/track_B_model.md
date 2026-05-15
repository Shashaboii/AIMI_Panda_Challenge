# Track B — Model architecture

You own how the model processes the input and produces a prediction.

## Files you own

- `src/model.py` — model definitions
- `src/train.py` — training loop (modify with caution — Person C also depends on this)

## Goal for the project

Replace the baseline's "single image in, single scalar out" EfficientNet with the **concat-tile-pooling** architecture that pools features across multiple tiles per slide. This is the architectural piece that made tile-based PANDA approaches work — without it, tile-based models often underperform thumbnail-based ones.

Expected lift on top of Person A's tiles: +0.02–0.04 val QWK.

## Week 1 — Implement concat-tile-pooling

The architecture in plain English:

1. Input shape changes from `[B, 3, H, W]` to `[B, N_tiles, 3, H, W]`.
2. Reshape to `[B*N_tiles, 3, H, W]` and run the EfficientNet backbone — get `[B*N_tiles, F]` features.
3. Reshape back to `[B, N_tiles, F]`.
4. Average-pool over the tile dimension to get `[B, F]`.
5. Final linear head → `[B, 1]` regression output.

In `src/model.py`:

```python
class ConcatTilePoolingModel(nn.Module):
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(backbone_name)
        in_f = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 1),
        )

    def forward(self, x):
        # x: [B, N, 3, H, W]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)            # [B*N, F]
        feats = feats.view(B, N, -1)         # [B, N, F]
        feats = feats.mean(dim=1)            # [B, F]
        return self.head(feats).squeeze(-1)
```

Read [Iafoss's PANDA notebook](https://www.kaggle.com/code/iafoss/panda-concat-tile-pooling-starter-0-79-lb) for context on how the original authors did this — then close it and write your own.

## Week 2 — Baseline experiments

Compare on the same fold (use fold 0 always for quick comparisons):

- EffNet-B0 vs EffNet-B1 — bigger backbone usually helps but is slower
- SmoothL1 vs MSE vs ordinal regression (BCE on cumulative thresholds)
- Different tile counts (work with Person A)

Log each result in `results.md`. Pick a "best single model" config by end of week 2.

## Week 3 — Mixup and augmentation

Mixup is a regularization technique that combines two training examples by interpolating both inputs and labels. The PANDA winners reported a small but consistent gain (~0.005 QWK). Implement it in `src/train.py`:

```python
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y + (1 - lam) * y[perm]
    return x_mix, y_mix
```

Apply with probability 0.5 per batch.

## Week 4 — Finalize for ensemble

Lock in your best config. Train all 5 folds with it. Hand 5 .pth files to Person C for ensembling.

## What success looks like

- A `ConcatTilePoolingModel` class in `src/model.py`
- Mixup integrated as a flag in `src/train.py` (`--mixup` argument)
- 5 sets of weights, one per fold, with consistent naming (`<backbone>_tiles<N>_imsize<S>_fold<F>.pth`)
- Ablation rows in `results.md` showing impact of each architectural change

## Where to start Monday

1. Pull the latest from main
2. Re-read `src/model.py`, `src/train.py`, `src/eval.py`
3. Wait for Person A to publish their first tile Dataset (Monday or Tuesday)
4. Implement `ConcatTilePoolingModel`, run on fold 0 with their tiles
5. Compare val QWK against the baseline (~0.70). Target for week 1: > 0.75
