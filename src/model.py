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
    """Tile model: pool backbone features across N tiles per slide."""
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True,
                 dropout=0.3, out_dim=1):
        super().__init__()
        self.backbone, in_features = _build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_dim),
        )

    def forward(self, x):
        batch_size, n_tiles, channels, height, width = x.shape
        x = x.reshape(batch_size * n_tiles, channels, height, width)
        features = self.backbone(x).reshape(batch_size, n_tiles, -1)
        return self.head(features.mean(dim=1)).squeeze(-1)
