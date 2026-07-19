"""One shared visual backbone used by every ReliPose-HOI branch."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8 if c_out >= 8 else 1, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(8 if c_out >= 8 else 1, c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SharedResNetFPN(nn.Module):
    """Compact ResNet-FPN-style backbone.

    Input: images [B, 3, H, W].
    Output levels "0".."3" with stride 4/8/16/32 and `out_channels`.

    This module is the only image backbone in the project. The name is
    kept for continuity with the paper and configs; the compact
    implementation is intentionally lightweight for Colab research runs.
    """

    feature_names = ("0", "1", "2", "3")

    def __init__(self, out_channels: int = 64, trainable_stages: int = 3) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.forward_calls = 0
        widths = [out_channels // 2, out_channels, out_channels * 2, out_channels * 4]
        self.stem = ConvBlock(3, widths[0], stride=2)
        self.c2 = ConvBlock(widths[0], widths[0], stride=2)
        self.c3 = ConvBlock(widths[0], widths[1], stride=2)
        self.c4 = ConvBlock(widths[1], widths[2], stride=2)
        self.c5 = ConvBlock(widths[2], widths[3], stride=2)
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in widths)
        self.output = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in widths)
        self.set_trainable_stages(trainable_stages)

    def set_trainable_stages(self, trainable_stages: int) -> None:
        stages = [self.stem, self.c2, self.c3, self.c4, self.c5]
        for stage in stages:
            for p in stage.parameters():
                p.requires_grad = False
        for stage in stages[-int(trainable_stages):]:
            for p in stage.parameters():
                p.requires_grad = True
        self._frozen = [s for s in stages if not any(p.requires_grad for p in s.parameters())]

    def train(self, mode: bool = True):
        super().train(mode)
        for stage in self._frozen:
            stage.eval()
        return self

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if not torch.isfinite(images).all():
            raise ValueError("images must be finite")
        self.forward_calls += 1
        x = self.stem(images)
        feats = [self.c2(x)]
        feats.append(self.c3(feats[-1]))
        feats.append(self.c4(feats[-1]))
        feats.append(self.c5(feats[-1]))
        inner = [lat(feat) for lat, feat in zip(self.lateral, feats)]
        out = [inner[-1]]
        for i in range(len(inner) - 2, -1, -1):
            out.insert(0, inner[i] + F.interpolate(out[0], size=inner[i].shape[-2:], mode="nearest"))
        return OrderedDict((str(i), layer(feat)) for i, (layer, feat) in enumerate(zip(self.output, out)))


def build_backbone(config) -> SharedResNetFPN:
    return SharedResNetFPN(config.fpn_dim, config.trainable_backbone_stages)

