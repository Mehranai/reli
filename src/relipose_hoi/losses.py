"""Training losses for the compact ReliPose-HOI implementation."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from relipose_hoi.config import LossConfig
from relipose_hoi.structures import IntegratedPoseOutput, PoseCorruptionOutput, PoseTarget, ReliPoseHOIOutput


def _zero(device: torch.device) -> Tensor:
    return torch.zeros((), device=device)


def pose_loss(pose: IntegratedPoseOutput, target: PoseTarget, cfg: LossConfig) -> dict[str, Tensor]:
    mask = target.joint_label_mask.to(pose.joint_coordinates_roi.device)
    if mask.numel() == 0 or not mask.any():
        z = _zero(pose.joint_coordinates_roi.device)
        return {"pose": z, "pose_coarse": z, "pose_refined": z, "confidence": z}
    tgt = target.joint_coordinates_roi_target.to(pose.joint_coordinates_roi.device)
    vis = target.joint_visible_mask.to(pose.joint_coordinates_roi.device).float()
    def coord(pred: Tensor, unc: Tensor) -> Tensor:
        err = F.smooth_l1_loss(pred, tgt, reduction="none").sum(-1)
        loss = err / unc.clamp_min(cfg.uncertainty_eps) + unc.clamp_min(cfg.uncertainty_eps).log()
        return loss[mask].mean()
    coarse = coord(pose.joint_coordinates_roi_coarse, pose.joint_uncertainty_coarse)
    refined = coord(pose.joint_coordinates_roi, pose.joint_uncertainty)
    conf = F.binary_cross_entropy_with_logits(pose.joint_confidence_logits, vis, reduction="none")[mask].mean()
    total = cfg.pose_weight * (coarse + refined + cfg.confidence_weight * conf)
    return {"pose": total, "pose_coarse": coarse, "pose_refined": refined, "confidence": conf}


def verb_loss(output: ReliPoseHOIOutput, targets: Tensor, cfg: LossConfig) -> dict[str, Tensor]:
    logits = output.verb_logits
    if logits.numel() == 0:
        z = _zero(logits.device)
        return {"verb": z, "verb_positive_labels": z, "verb_valid_labels": z}
    targets = targets.to(logits.device).float()
    valid = output.valid_verb_mask.to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if cfg.focal_gamma > 0:
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0, p, 1 - p)
        bce = bce * (1 - pt).pow(cfg.focal_gamma)
    loss = bce[valid].mean() if valid.any() else _zero(logits.device)
    return {
        "verb": cfg.verb_weight * loss,
        "verb_positive_labels": targets[valid].sum().detach(),
        "verb_valid_labels": valid.sum().float().detach(),
    }


def reliability_loss(logits: Tensor, synthetic_target: Tensor, mask: Tensor, cfg: LossConfig) -> dict[str, Tensor]:
    if logits.numel() == 0 or not mask.any():
        z = _zero(logits.device)
        return {"reliability": z}
    target = synthetic_target.to(logits.device).detach()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return {"reliability": cfg.reliability_weight * loss[mask.to(logits.device)].mean()}


def consistency_loss(clean_logits: Tensor, corrupt_logits: Tensor, valid: Tensor, cfg: LossConfig) -> dict[str, Tensor]:
    if clean_logits.numel() == 0 or not valid.any():
        z = _zero(clean_logits.device)
        return {"consistency": z}
    clean = clean_logits.detach().sigmoid()
    corrupt = corrupt_logits.sigmoid()
    loss = (corrupt - clean).square()
    return {"consistency": cfg.consistency_weight * loss[valid.to(loss.device)].mean()}


def sparse_loss(pose: IntegratedPoseOutput, cfg: LossConfig) -> dict[str, Tensor]:
    gate = pose.refinement_gate
    if gate.numel() == 0:
        return {"sparse": _zero(gate.device), "refinement_ratio": _zero(gate.device)}
    ratio = gate.mean()
    return {"sparse": cfg.sparse_weight * (ratio - cfg.sparse_target).abs(), "refinement_ratio": ratio.detach()}


def total_hoi_loss(
    clean: ReliPoseHOIOutput,
    *,
    verb_targets: Tensor | None,
    cfg: LossConfig,
    corrupted: tuple[PoseCorruptionOutput, ReliPoseHOIOutput] | None = None,
) -> dict[str, Tensor]:
    pieces: dict[str, Tensor] = {}
    if verb_targets is not None:
        pieces.update(verb_loss(clean, verb_targets, cfg))
    pieces.update(sparse_loss(clean.pose, cfg))
    if corrupted is not None:
        corr_pose, corr_out = corrupted
        pieces.update(reliability_loss(corr_out.reliability.joint_reliability_logits, corr_pose.synthetic_reliability_target, corr_pose.reliability_supervision_mask, cfg))
        pieces.update(consistency_loss(clean.verb_logits, corr_out.verb_logits, clean.valid_verb_mask, cfg))
    total = sum(v for k, v in pieces.items() if not k.endswith("labels") and k != "refinement_ratio")
    pieces["total"] = total
    return pieces

