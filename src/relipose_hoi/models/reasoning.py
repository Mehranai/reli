"""Interaction Relevance, gated attention and adaptive fusion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from relipose_hoi.structures import HumanJointReasoningOutput, HumanObjectPairOutput, InteractionReasoningOutput


def joint_roi_to_image(joint_roi: Tensor, human_boxes: Tensor, human_to_image: Tensor, image_sizes: Sequence[tuple[int, int]]) -> Tensor:
    if joint_roi.shape[0] == 0:
        return joint_roi.new_empty(joint_roi.shape)
    wh = (human_boxes[:, 2:] - human_boxes[:, :2]).clamp_min(1e-6)
    xy = human_boxes[:, None, :2] + joint_roi * wh[:, None, :]
    sizes = torch.tensor([[image_sizes[int(i)][1], image_sizes[int(i)][0]] for i in human_to_image.tolist()], dtype=xy.dtype, device=xy.device)
    return xy / sizes[:, None, :].clamp_min(1)


class InteractionReasoningModule(nn.Module):
    """Connect human-level r_hi and pair q_ho to fused pair features."""

    def __init__(self, joint_dim: int = 64, pair_dim: int = 64, object_dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.joint_dim, self.pair_dim, self.heads = joint_dim, pair_dim, heads
        self.geom = nn.Sequential(nn.Linear(17, pair_dim), nn.GELU(), nn.Linear(pair_dim, pair_dim))
        rel_in = joint_dim + pair_dim + object_dim + pair_dim + 1
        self.relevance = nn.Sequential(nn.LayerNorm(rel_in), nn.Linear(rel_in, pair_dim), nn.GELU(), nn.Linear(pair_dim, 1))
        self.q = nn.Linear(pair_dim, heads * pair_dim)
        self.k = nn.Linear(joint_dim, heads * pair_dim)
        self.v = nn.Linear(joint_dim, heads * pair_dim)
        self.attn_out = nn.Linear(heads * pair_dim, pair_dim)
        self.pose_proj = nn.Linear(pair_dim, pair_dim)
        self.beta = nn.Sequential(nn.LayerNorm(pair_dim + 1), nn.Linear(pair_dim + 1, pair_dim), nn.GELU(), nn.Linear(pair_dim, 1))
        self.norm = nn.LayerNorm(pair_dim)

    def joint_object_geometry(self, joints: Tensor, human_boxes: Tensor, object_boxes: Tensor, image_sizes: Tensor) -> Tensor:
        n, k, _ = joints.shape
        if n == 0:
            return joints.new_empty((0, k, 16))
        wh = image_sizes[:, [1, 0]].clamp_min(1)
        hn = human_boxes / image_sizes[:, [1, 0, 1, 0]].clamp_min(1)
        on = object_boxes / image_sizes[:, [1, 0, 1, 0]].clamp_min(1)
        oc = ((on[:, :2] + on[:, 2:]) / 2)[:, None]
        ow = (on[:, 2] - on[:, 0]).clamp_min(1e-6)[:, None]
        oh = (on[:, 3] - on[:, 1]).clamp_min(1e-6)[:, None]
        disp = joints - oc
        rel_obj = torch.stack([(joints[..., 0] - on[:, None, 0]) / ow[:, :, None].squeeze(-1), (joints[..., 1] - on[:, None, 1]) / oh[:, :, None].squeeze(-1)], -1)
        inside = ((joints[..., 0] >= on[:, None, 0]) & (joints[..., 0] <= on[:, None, 2]) & (joints[..., 1] >= on[:, None, 1]) & (joints[..., 1] <= on[:, None, 3])).float()[..., None]
        dist = disp.norm(dim=-1, keepdim=True)
        boundary = torch.stack([joints[..., 0] - on[:, None, 0], on[:, None, 2] - joints[..., 0], joints[..., 1] - on[:, None, 1], on[:, None, 3] - joints[..., 1]], -1)
        hcenter = ((hn[:, :2] + hn[:, 2:]) / 2)[:, None]
        rel_h = joints - hcenter
        return torch.cat([joints, oc.expand(-1, k, -1), disp, rel_obj, dist, (dist + 1e-6).log(), inside, boundary, rel_h], -1)

    def forward(
        self,
        human: HumanJointReasoningOutput,
        pairs: HumanObjectPairOutput,
        joint_coordinates_image: Tensor,
        human_boxes: Tensor,
        object_boxes: Tensor,
        image_sizes: Sequence[tuple[int, int]],
    ) -> InteractionReasoningOutput:
        ph, po = pairs.pair_human_indices_global, pairs.pair_object_indices_global
        n = ph.numel()
        k = human.joint_representations.shape[1] if human.joint_representations.ndim == 3 else 17
        if n == 0:
            e = pairs.pair_features.new_empty
            b = torch.empty(0, dtype=torch.bool, device=pairs.pair_features.device)
            return InteractionReasoningOutput(e((0, k, self.joint_dim)), e((0, k)), b.new_empty((0, k)), e((0, k, 17)), e((0, k, self.pair_dim)), e((0, k)), e((0, k)), e((0, k)), e((0, self.heads, k)), e((0, self.pair_dim)), e((0,)), e((0,)), e((0, self.pair_dim)), b)
        jr = human.joint_representations.index_select(0, ph)
        rr = human.joint_reliability.index_select(0, ph)
        valid = human.joint_valid_mask.index_select(0, ph)
        ji = joint_coordinates_image.index_select(0, ph)
        sizes = torch.tensor([image_sizes[int(i)] for i in pairs.pair_to_image.tolist()], dtype=human_boxes.dtype, device=human_boxes.device)
        geom_raw = self.joint_object_geometry(ji, human_boxes[ph], object_boxes[po], sizes)
        geom_feat = self.geom(geom_raw)
        q = pairs.pair_features[:, None].expand(-1, k, -1)
        obj = pairs.object_semantic_features[:, None].expand(-1, k, -1)
        rel_logits = self.relevance(torch.cat([jr, q, obj, geom_feat, human.joint_validity_prior.index_select(0, ph)[..., None]], -1)).squeeze(-1)
        masked = rel_logits.masked_fill(~valid, -1e9)
        has = valid.any(1)
        a = torch.zeros_like(rel_logits)
        if has.any():
            a[has] = torch.softmax(masked[has], dim=1)
        g = rr * a
        qh = self.q(pairs.pair_features).reshape(n, self.heads, self.pair_dim)
        kh = self.k(jr).reshape(n, k, self.heads, self.pair_dim).transpose(1, 2)
        vh = self.v(jr).reshape(n, k, self.heads, self.pair_dim).transpose(1, 2)
        logits = (qh[:, :, None] * kh).sum(-1) / (self.pair_dim ** 0.5)
        logits = logits + torch.log(rr.clamp_min(1e-6))[:, None] + torch.log(a.clamp_min(1e-6))[:, None]
        logits = logits.masked_fill(~valid[:, None], -1e9)
        weights = torch.zeros_like(logits)
        if has.any():
            weights[has] = torch.softmax(logits[has], -1)
        pose_heads = (weights[..., None] * vh).sum(2).reshape(n, self.heads * self.pair_dim)
        pose_aware = self.attn_out(pose_heads)
        pose_aware = torch.where(has[:, None], pose_aware, torch.zeros_like(pose_aware))
        denom = a.sum(1).clamp_min(1e-6)
        summary = torch.where(has, (a * rr).sum(1) / denom, torch.zeros_like(denom))
        beta = torch.sigmoid(self.beta(torch.cat([pairs.pair_features, summary[:, None]], 1))).squeeze(1)
        beta = torch.where(has, beta, torch.zeros_like(beta))
        fused = self.norm(pairs.pair_features + beta[:, None] * self.pose_proj(pose_aware))
        fallback = self.norm(pairs.pair_features)
        fused = torch.where(has[:, None], fused, fallback)
        return InteractionReasoningOutput(jr, rr, valid, geom_raw, geom_feat, rel_logits, a, g, weights, pose_aware, summary, beta, fused, has)
