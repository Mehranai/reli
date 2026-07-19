"""Object-independent controlled corruption of internal pose outputs."""

from __future__ import annotations

import torch

from relipose_hoi.config import CorruptionConfig
from relipose_hoi.data.transforms import COCO_FLIP
from relipose_hoi.structures import IntegratedPoseOutput, PoseCorruptionOutput


class PoseCorruptor:
    """Training-only pose corruptor. It accepts no object or pair inputs."""

    def __init__(self, cfg: CorruptionConfig) -> None:
        self.cfg = cfg

    def __call__(
        self,
        pose: IntegratedPoseOutput,
        valid_mask: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> PoseCorruptionOutput:
        coords = pose.joint_coordinates_roi
        n, k, _ = coords.shape
        if n == 0:
            z = coords.new_empty((0, k))
            return PoseCorruptionOutput(pose, z.bool(), z, z, z.bool())
        sample = torch.rand((n, k), generator=generator, device=coords.device)
        mask = sample < self.cfg.joint_probability
        if valid_mask is not None:
            mask = mask & valid_mask.to(coords.device)
        severity = torch.zeros((n, k), device=coords.device, dtype=coords.dtype)
        jitter = torch.randn(coords.shape, generator=generator, device=coords.device, dtype=coords.dtype) * self.cfg.coordinate_jitter_std
        new_coords = coords + jitter * mask[..., None]
        severity = torch.maximum(severity, mask.float() * jitter.norm(dim=-1).clamp_max(1))

        drop = torch.rand((n, k), generator=generator, device=coords.device) < self.cfg.dropout_probability
        drop = drop & mask
        new_tokens = pose.joint_tokens + torch.randn(pose.joint_tokens.shape, generator=generator, device=coords.device) * self.cfg.token_noise_std * mask[..., None]
        new_local = pose.local_joint_features + torch.randn(pose.local_joint_features.shape, generator=generator, device=coords.device) * self.cfg.token_noise_std * mask[..., None]
        new_conf = torch.where(drop, pose.joint_confidence * 0.05, pose.joint_confidence)
        new_unc = torch.where(drop, pose.joint_uncertainty * 4.0 + 1.0, pose.joint_uncertainty)
        severity = torch.maximum(severity, drop.float())

        if k == COCO_FLIP.numel() and torch.rand((), generator=generator).item() < self.cfg.swap_probability:
            swap = COCO_FLIP.to(coords.device)
            new_coords = new_coords[:, swap]
            new_tokens = new_tokens[:, swap]
            new_local = new_local[:, swap]
            new_conf = new_conf[:, swap]
            new_unc = new_unc[:, swap]
            severity = torch.maximum(severity[:, swap], torch.full_like(severity, 0.75))

        margin = self.cfg.maximum_margin
        new_coords = new_coords.clamp(-margin, 1 + margin)
        logits = torch.logit(new_conf.clamp(1e-6, 1 - 1e-6))
        target = torch.exp(-self.cfg.reliability_alpha * severity).detach()
        target = torch.where(drop, torch.zeros_like(target), target)
        corrupted = IntegratedPoseOutput(
            pose.joint_tokens_initial,
            pose.joint_coordinates_roi_coarse,
            pose.joint_confidence_logits_coarse,
            pose.joint_confidence_coarse,
            pose.joint_uncertainty_coarse,
            pose.coarse_pose_quality,
            new_tokens,
            new_coords,
            logits,
            new_conf,
            new_unc.clamp_min(1e-4),
            new_local,
            pose.refinement_gate,
            pose.refinement_mask,
        )
        return PoseCorruptionOutput(corrupted, mask.detach(), severity.detach().clamp(0, 1), target, (mask | (valid_mask.to(coords.device) if valid_mask is not None else torch.ones_like(mask))).detach())

