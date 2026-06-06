"""
Model definitions for PANDA.

CURRENT STATE: simple EfficientNet-B0 with a regression head on a single
512x512 image input. This matches the baseline that reached val QWK 0.70.

PERSON B: extend this. The most important addition is concat-tile-pooling:
  - Input shape changes from [B, 3, H, W] to [B, N_tiles, 3, H, W]
  - Reshape to [B*N_tiles, 3, H, W], run backbone, get features [B*N_tiles, F]
  - Reshape back to [B, N_tiles, F], average-pool over N_tiles to [B, F]
  - Final linear head -> [B, 1]

You can either modify EfficientNetBaseline in place or add a new class
ConcatTilePoolingModel. Whichever you pick, keep the constructor signature
similar so train.py doesn't need to change much.
"""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def _build_backbone(backbone_name='efficientnet-b0', pretrained=True):
    if pretrained:
        backbone = EfficientNet.from_pretrained(backbone_name)
    else:
        backbone = EfficientNet.from_name(backbone_name)
    in_features = backbone._fc.in_features
    backbone._fc = nn.Identity()
    return backbone, in_features


class EfficientNetBaseline(nn.Module):
    """Baseline: single-image input, regression output."""
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True,
                 dropout=0.3, out_dim=1):
        super().__init__()
        self.backbone, in_features = _build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class ConcatTilePoolingModel(nn.Module):
    """Tile model: pool backbone features across N tiles per slide.

    The original mean-pool version was very sensitive to noisy tail tiles:
    once the first tissue-rich tiles were averaged together with near-empty
    ones, validation QWK dropped sharply as ``N`` increased. This version keeps
    the same public API but uses a more robust parameter-free pooler:

      - detect padded / near-blank tiles from the input variance
      - compute a soft attention score from tile feature energy
      - combine attention pooling with max pooling so one strong tile is not
        washed out by many weak ones

    The output dimensionality stays the same, so ``train.py`` and the baseline
    head contract do not need to change.
    """
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True,
                 dropout=0.3, out_dim=1):
        super().__init__()
        self.backbone, in_features = _build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_dim),
        )

    def _tile_mask(self, x):
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        return flat.var(dim=-1, unbiased=False) > 1e-6

    def _pool_tiles(self, features, tile_mask):
        if tile_mask is not None:
            # If a row somehow contains only padded tiles, fall back to keeping
            # every tile valid so softmax / max remain well-defined.
            all_invalid = ~tile_mask.any(dim=1)
            if all_invalid.any():
                tile_mask = tile_mask.clone()
                tile_mask[all_invalid] = True

        tile_scores = features.pow(2).mean(dim=-1)
        if tile_mask is not None:
            tile_scores = tile_scores.masked_fill(~tile_mask, -1e9)

        attn = torch.softmax(tile_scores, dim=1)
        attn_pooled = (features * attn.unsqueeze(-1)).sum(dim=1)

        if tile_mask is not None:
            max_features = features.masked_fill(~tile_mask.unsqueeze(-1), -1e9)
            max_pooled = max_features.max(dim=1).values
            max_pooled = torch.where(
                torch.isfinite(max_pooled),
                max_pooled,
                attn_pooled,
            )
        else:
            max_pooled = features.max(dim=1).values

        return 0.5 * (attn_pooled + max_pooled)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f'Expected tile input [B, N, C, H, W], got shape {tuple(x.shape)}')
        batch_size, n_tiles, channels, height, width = x.shape
        tile_mask = self._tile_mask(x)
        x = x.reshape(batch_size * n_tiles, channels, height, width)
        features = self.backbone(x).reshape(batch_size, n_tiles, -1)
        pooled = self._pool_tiles(features, tile_mask)
        return self.head(pooled).squeeze(-1)
