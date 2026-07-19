"""Compact anatomy, final joint representations and object-independent r_hi."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from relipose_hoi.data.transforms import COCO_FLIP, COCO_JOINT_NAMES
from relipose_hoi.structures import HumanJointReasoningOutput, IntegratedPoseOutput


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


class Anatomy(nn.Module):
    """Differentiable low-dimensional human-only anatomical descriptor."""

    feature_dim = 22

    def __init__(self, num_joints: int = 17, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.eps = eps
        degree = torch.zeros(num_joints)
        for a, b in COCO_EDGES:
            degree[a] += 1
            degree[b] += 1
        self.register_buffer("degree", degree / degree.max().clamp_min(1))
        self.register_buffer("flip", COCO_FLIP.clone())

    def forward(self, coords: Tensor, valid: Tensor | None = None) -> Tensor:
        n, k, _ = coords.shape
        if n == 0:
            return coords.new_empty((0, k, self.feature_dim))
        valid_f = torch.ones((n, k), device=coords.device, dtype=coords.dtype) if valid is None else valid.float()
        weights = valid_f[..., None]
        body = (coords * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        hips = coords[:, [11, 12]].mean(1) if k > 12 else body
        torso = (coords[:, 5] - coords[:, 11]).norm(dim=-1, keepdim=True) if k > 11 else coords.new_ones((n, 1))
        torso = torso.clamp_min(self.eps)
        rel_body = (coords - body[:, None]) / torso[:, None]
        rel_hip = (coords - hips[:, None]) / torso[:, None]
        margins = torch.cat([coords, 1 - coords], -1)
        radial = (coords - body[:, None]).norm(dim=-1, keepdim=True) / torso[:, None]
        sym = (coords - coords[:, self.flip.to(coords.device)]).norm(dim=-1, keepdim=True) / torso[:, None]
        degree = self.degree.to(coords.device, coords.dtype).view(1, k, 1).expand(n, -1, -1)
        # Local neighbor mean direction without letting invalid joints dominate.
        neighbor = torch.zeros_like(coords)
        counts = torch.zeros((n, k, 1), device=coords.device, dtype=coords.dtype)
        for a, b in COCO_EDGES:
            if a < k and b < k:
                wa = valid_f[:, a:a + 1]
                wb = valid_f[:, b:b + 1]
                neighbor[:, a] += (coords[:, b] - coords[:, a]) * wb
                neighbor[:, b] += (coords[:, a] - coords[:, b]) * wa
                counts[:, a] += wb
                counts[:, b] += wa
        neighbor = neighbor / counts.clamp_min(1.0)
        neighbor = neighbor / torso[:, None]
        angle = torch.cat([torch.sin(neighbor), torch.cos(neighbor)], -1)
        min_margin = margins.amin(dim=-1, keepdim=True)
        features = torch.cat(
            [
                rel_body,
                rel_hip,
                neighbor,
                angle,
                margins,
                coords,
                radial,
                sym,
                min_margin,
                valid_f[..., None],
                torso[:, None].expand(-1, k, -1),
                degree,
            ],
            -1,
        )
        features = torch.nan_to_num(features)
        return features[..., : self.feature_dim]


class HumanJointReasoningModule(nn.Module):
    """Compute p_hi and learned object-independent Reliability r_hi.

    The public forward signature accepts only `IntegratedPoseOutput`;
    object and pair information cannot enter by construction.
    """

    def __init__(self, pose_dim: int = 64, joint_dim: int = 64, num_joints: int = 17) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.anatomy = Anatomy(num_joints)
        self.type_embedding = nn.Embedding(num_joints, 16)
        in_dim = pose_dim * 2 + 2 + self.anatomy.feature_dim + 7 + 16
        self.encoder = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, joint_dim), nn.GELU(), nn.Linear(joint_dim, joint_dim))
        self.residual = nn.Linear(pose_dim, joint_dim)
        self.norm = nn.LayerNorm(joint_dim)
        self.reliability = nn.Sequential(nn.LayerNorm(joint_dim + self.anatomy.feature_dim + 5), nn.Linear(joint_dim + self.anatomy.feature_dim + 5, joint_dim), nn.GELU(), nn.Linear(joint_dim, 1))
        nn.init.normal_(self.reliability[-1].weight, std=1e-3)
        nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, pose: IntegratedPoseOutput) -> HumanJointReasoningOutput:
        coords = pose.joint_coordinates_roi
        n, k, _ = coords.shape
        conf = pose.joint_confidence
        unc = pose.joint_uncertainty
        finite = torch.isfinite(coords).all(-1)
        in_bounds = ((coords >= 0) & (coords <= 1)).all(-1)
        validity_prior = (conf * torch.exp(-unc.clamp_max(20))).clamp(0, 1)
        valid = finite & in_bounds & (conf > 0.05) & (unc < 10.0)
        anatomy = self.anatomy(coords, valid)
        disp = (pose.joint_coordinates_roi - pose.joint_coordinates_roi_coarse).norm(dim=-1)
        if n == 0:
            return HumanJointReasoningOutput(
                anatomy,
                coords.new_empty((0, k, self.joint_dim)),
                disp,
                validity_prior,
                valid,
                coords.new_empty((0, k)),
                coords.new_empty((0, k)),
            )
        ids = torch.arange(k, device=coords.device)
        jt = self.type_embedding(ids)[None].expand(n, -1, -1)
        scalar = torch.stack([conf, unc.log().clamp(-8, 8), pose.coarse_pose_quality, pose.refinement_gate, disp, coords[..., 0], coords[..., 1]], -1)
        enc_in = torch.cat([pose.joint_tokens, pose.local_joint_features, coords, anatomy, scalar, jt], -1)
        reps = self.norm(self.encoder(enc_in) + self.residual(pose.joint_tokens))
        rel_in = torch.cat([reps, anatomy, scalar[..., :5]], -1)
        logits = self.reliability(rel_in).squeeze(-1)
        r = torch.sigmoid(logits)
        return HumanJointReasoningOutput(anatomy, reps, disp, validity_prior, valid, logits, r)
