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


class EfficientNetBaseline(nn.Module):
    """Baseline: single-image input, regression output."""
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True, dropout=0.3):
        super().__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(backbone_name)
        else:
            self.backbone = EfficientNet.from_name(backbone_name)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)
