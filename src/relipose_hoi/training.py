"""Simple training utilities: AdamW, cosine warmup, epochs and robust path."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from relipose_hoi.config import Config
from relipose_hoi.corruption import PoseCorruptor
from relipose_hoi.data.coco import keypoints_image_to_roi_normalized
from relipose_hoi.losses import pose_loss, total_hoi_loss
from relipose_hoi.structures import Batch, ReliPoseHOIOutput, RobustOutput


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)


class CosineWarmup:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, total: int) -> None:
        self.optimizer, self.warmup, self.total, self.step_num = optimizer, max(1, warmup), max(1, total), 0
        self.base = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1
        for lr, group in zip(self.base, self.optimizer.param_groups):
            if self.step_num <= self.warmup:
                scale = self.step_num / self.warmup
            else:
                t = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
                scale = 0.5 * (1 + math.cos(math.pi * min(1, t)))
            group["lr"] = lr * scale

    def state_dict(self) -> dict:
        return {"step_num": self.step_num, "base": self.base, "warmup": self.warmup, "total": self.total}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = state["step_num"]
        self.base = state["base"]
        self.warmup = state["warmup"]
        self.total = state["total"]


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> CosineWarmup:
    return CosineWarmup(optimizer, cfg.optim.warmup_steps, cfg.optim.total_steps)


def _step(loss: Tensor, model, optimizer, scheduler, cfg: Config) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
    optimizer.step()
    scheduler.step()
    return float(norm)


def train_pose_epoch(model, dataloader: Iterable[Batch], optimizer, scheduler, cfg: Config, device: torch.device, max_steps: int | None = None) -> dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    for batch in dataloader:
        if max_steps is not None and count >= max_steps:
            break
        images = batch.images.to(device)
        targets = [t.as_model_target() if hasattr(t, "as_model_target") else {"human_boxes": t.person_boxes.to(device), "human_scores": torch.ones(t.person_boxes.shape[0], device=device), "object_boxes": torch.empty((0, 4), device=device), "object_scores": torch.empty(0, device=device), "object_labels": torch.empty(0, dtype=torch.long, device=device), "pair_human_indices": torch.empty(0, dtype=torch.long, device=device), "pair_object_indices": torch.empty(0, dtype=torch.long, device=device), "verb_targets": torch.empty((0, cfg.model.num_verbs), device=device)} for t in batch.targets]
        out = model(images, batch.image_sizes, targets, pair_mode="training")
        pose_targets = [
            keypoints_image_to_roi_normalized(
                t.keypoints_image.to(device),
                t.person_boxes.to(device),
                t.keypoint_label_mask.to(device),
                t.keypoint_visible_mask.to(device),
            )
            for t in batch.targets
        ]
        if pose_targets:
            pt = pose_targets[0].__class__(
                torch.cat([p.joint_coordinates_roi_target for p in pose_targets], 0),
                torch.cat([p.joint_label_mask for p in pose_targets], 0),
                torch.cat([p.joint_visible_mask for p in pose_targets], 0),
            )
            loss = pose_loss(out.pose, pt, cfg.losses)["pose"]
        else:
            loss = out.verb_logits.sum() * 0
        _step(loss, model, optimizer, scheduler, cfg)
        total += float(loss.detach())
        count += 1
    return {"pose_loss": total / max(1, count)}


def train_hoi_epoch(model, dataloader: Iterable[Batch], optimizer, scheduler, cfg: Config, device: torch.device, *, robust: bool = False, max_steps: int | None = None) -> dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    for batch in dataloader:
        if max_steps is not None and count >= max_steps:
            break
        images = batch.images.to(device)
        targets = [t.as_model_target() for t in batch.targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gen = torch.Generator(device=device).manual_seed(cfg.train.seed)
        clean = model(images, batch.image_sizes, targets, pair_mode="training", generator=gen)
        verb_targets = clean.training_pairs.verb_targets if clean.training_pairs is not None else clean.verb_logits.new_empty(clean.verb_logits.shape)
        corrupted = None
        if robust:
            corr = robust_forward(model, clean, batch.image_sizes, cfg, gen)
            corrupted = (corr.corrupted_pose, corr.clean.__class__(
                corr.clean.detections, corr.clean.roi, corr.corrupted_pose.corrupted_pose, corr.corrupted_reliability, corr.clean.pairs, corr.corrupted_reasoning,
                corr.corrupted_verb_logits, corr.corrupted_verb_logits.sigmoid(), corr.clean.valid_verb_mask, corr.corrupted_hoi_scores, corr.clean.pair_object_labels, corr.clean.training_pairs
            ))
        pieces = total_hoi_loss(clean, verb_targets=verb_targets, cfg=cfg.losses, corrupted=corrupted)
        _step(pieces["total"], model, optimizer, scheduler, cfg)
        total += float(pieces["total"].detach())
        count += 1
    return {"loss": total / max(1, count)}


def robust_forward(model, clean: ReliPoseHOIOutput, image_sizes, cfg: Config, generator: torch.Generator | None = None) -> RobustOutput:
    corruptor = PoseCorruptor(cfg.corruption)
    corr_pose = corruptor(clean.pose, clean.reliability.joint_valid_mask, generator=generator)
    rel = model.reliability(corr_pose.corrupted_pose)
    from relipose_hoi.models.reasoning import joint_roi_to_image
    from relipose_hoi.models.pair import flatten_detections

    hb, ob, _, hti, _, _, _, _, _ = flatten_detections(clean.detections)
    ji = joint_roi_to_image(corr_pose.corrupted_pose.joint_coordinates_roi.clamp(0, 1), hb, hti, image_sizes)
    reasoning = model.reasoning(rel, clean.pairs, ji, hb, ob, image_sizes)
    logits = model.verb(reasoning.fused_pair_features)
    _, _, hoi = model.projector(logits, clean.pair_object_labels)
    return RobustOutput(clean, corr_pose, rel, reasoning, logits, hoi)


def log_json(path: str | Path, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
