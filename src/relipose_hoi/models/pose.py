"""Internal sparse pose tokenizer and pose-quality-guided refinement."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from relipose_hoi.structures import IntegratedPoseOutput


class PoseTokenizer(nn.Module):
    def __init__(self, in_channels: int, num_joints: int = 17, dim: int = 64, heads: int = 4, layers: int = 1, roi_size: int = 7) -> None:
        super().__init__()
        self.num_joints, self.dim, self.roi_size = num_joints, dim, roi_size
        self.proj = nn.Conv2d(in_channels, dim, 1)
        self.pos = nn.Parameter(torch.zeros(roi_size * roi_size, dim))
        self.query = nn.Parameter(torch.randn(num_joints, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.coord = nn.Linear(dim, 2)
        self.conf = nn.Linear(dim, 1)
        self.unc = nn.Linear(dim, 1)
        self.quality = nn.Linear(dim + 2, 1)
        nn.init.zeros_(self.coord.weight)
        nn.init.constant_(self.coord.bias, 0.0)
        nn.init.constant_(self.conf.bias, -1.0)

    def forward(self, roi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        n = roi.shape[0]
        if n == 0:
            e = roi.new_empty
            return e((0, self.num_joints, self.dim)), e((0, self.num_joints, 2)), e((0, self.num_joints)), e((0, self.num_joints)), e((0, self.num_joints)), e((0, self.num_joints))
        mem = self.proj(roi).flatten(2).transpose(1, 2) + self.pos[None]
        q = self.query[None].expand(n, -1, -1)
        tokens = self.norm(self.decoder(q, mem))
        coords = torch.sigmoid(self.coord(tokens))
        conf_logits = self.conf(tokens).squeeze(-1)
        conf = torch.sigmoid(conf_logits)
        unc = F.softplus(self.unc(tokens)).squeeze(-1) + 1e-4
        quality = torch.sigmoid(self.quality(torch.cat([tokens, conf[..., None], unc[..., None]], -1))).squeeze(-1)
        return tokens, coords, conf_logits, conf, unc, quality


class SparseRefiner(nn.Module):
    def __init__(self, in_channels: int, dim: int = 64, patch_size: int = 5, radius: float = 0.18, threshold: float = 0.5, temperature: float = 0.1, max_offset: float = 0.15) -> None:
        super().__init__()
        self.patch_size, self.radius, self.threshold, self.temperature, self.max_offset = patch_size, radius, threshold, temperature, max_offset
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1),
            nn.GroupNorm(8 if dim >= 8 else 1, dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fuse = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU())
        self.coord_delta = nn.Linear(dim, 2)
        self.conf_delta = nn.Linear(dim, 1)
        self.unc_delta = nn.Linear(dim, 1)
        for head in (self.coord_delta, self.conf_delta, self.unc_delta):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def patches(self, roi: Tensor, coords: Tensor) -> Tensor:
        n, m = coords.shape[:2]
        if n == 0 or m == 0:
            return roi.new_empty((n, m, roi.shape[1], self.patch_size, self.patch_size))
        axis = torch.linspace(-self.radius, self.radius, self.patch_size, device=roi.device, dtype=roi.dtype)
        oy, ox = torch.meshgrid(axis, axis, indexing="ij")
        x = coords[..., 0, None, None] + ox
        y = coords[..., 1, None, None] + oy
        grid = torch.stack([2 * x - 1, 2 * y - 1], -1).reshape(n * m, self.patch_size, self.patch_size, 2)
        source = roi[:, None].expand(-1, m, -1, -1, -1).reshape(n * m, roi.shape[1], roi.shape[2], roi.shape[3])
        patches = F.grid_sample(source, grid, padding_mode="border", align_corners=False)
        return patches.reshape(n, m, roi.shape[1], self.patch_size, self.patch_size)

    def forward(self, roi: Tensor, tokens: Tensor, coords: Tensor, conf_logits: Tensor, unc: Tensor, quality: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        n, k = quality.shape
        if n == 0:
            z = quality.new_empty((0, k))
            return tokens, coords, torch.sigmoid(conf_logits), unc, tokens, z, z.bool()
        gate = torch.sigmoid((self.threshold - quality) / self.temperature)
        patch = self.patches(roi, coords)
        feat = self.patch_encoder(patch.reshape(n * k, roi.shape[1], self.patch_size, self.patch_size)).flatten(1).reshape(n, k, -1)
        fused = self.fuse(torch.cat([tokens, feat], -1))
        dxy = torch.tanh(self.coord_delta(fused)) * self.max_offset * gate[..., None]
        tokens_out = tokens + fused * gate[..., None]
        coords_out = (coords + dxy).clamp(0, 1)
        conf_logits_out = conf_logits + self.conf_delta(fused).squeeze(-1) * gate
        unc_out = (unc * torch.exp(torch.tanh(self.unc_delta(fused).squeeze(-1)) * gate)).clamp_min(1e-4)
        mask = gate > 0.5
        return tokens_out, coords_out, torch.sigmoid(conf_logits_out), unc_out, feat, gate, mask


class IntegratedPoseModule(nn.Module):
    """Human RoI features -> internal sparse pose output."""

    def __init__(self, in_channels: int, num_joints: int = 17, dim: int = 64, roi_size: int = 7, heads: int = 4, layers: int = 1) -> None:
        super().__init__()
        self.tokenizer = PoseTokenizer(in_channels, num_joints, dim, heads, layers, roi_size)
        self.refiner = SparseRefiner(in_channels, dim)

    def forward(self, human_roi_features: Tensor) -> IntegratedPoseOutput:
        tok, xy, conf_logits, conf, unc, rho = self.tokenizer(human_roi_features)
        rtok, rxy, rconf, runc, local, gate, mask = self.refiner(human_roi_features, tok, xy, conf_logits, unc, rho)
        rconf_logits = torch.logit(rconf.clamp(1e-6, 1 - 1e-6))
        return IntegratedPoseOutput(tok, xy, conf_logits, conf, unc, rho, rtok, rxy, rconf_logits, rconf, runc, local, gate, mask)

