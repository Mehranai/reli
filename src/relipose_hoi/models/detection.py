"""Detection branches that consume shared FPN features, never raw images."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from relipose_hoi.structures import DetectionOutput


class DetectionBranch(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        features: Mapping[str, Tensor],
        image_sizes: Sequence[tuple[int, int]],
        targets: Sequence[Mapping[str, Tensor]] | None = None,
    ) -> DetectionOutput:
        raise NotImplementedError


class OracleDetectionBranch(DetectionBranch):
    """Development detector that returns supplied GT/matched boxes."""

    def forward(self, features, image_sizes, targets=None) -> DetectionOutput:
        if targets is None:
            raise ValueError("OracleDetectionBranch requires targets")
        human_boxes, human_scores, object_boxes, object_scores, object_labels = [], [], [], [], []
        for target in targets:
            hb = target.get("human_boxes", torch.empty((0, 4))).float()
            ob = target.get("object_boxes", torch.empty((0, 4))).float()
            human_boxes.append(hb)
            object_boxes.append(ob)
            human_scores.append(target.get("human_scores", torch.ones(hb.shape[0])).float().to(hb.device))
            object_scores.append(target.get("object_scores", torch.ones(ob.shape[0])).float().to(ob.device))
            object_labels.append(target.get("object_labels", torch.zeros(ob.shape[0], dtype=torch.long)).long().to(ob.device))
        return DetectionOutput(human_boxes, human_scores, object_boxes, object_scores, object_labels).validate()


class SharedFPNDetector(DetectionBranch):
    """Minimal detected-box branch using only precomputed shared FPN maps."""

    def __init__(self, in_channels: int, num_classes: int = 80, human_class_index: int = 0, topk: int = 20) -> None:
        super().__init__()
        self.human_class_index = int(human_class_index)
        self.topk = int(topk)
        self.score = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.box_delta = nn.Conv2d(in_channels, 4, 3, padding=1)

    def forward(self, features, image_sizes, targets=None) -> DetectionOutput:
        feature = features["0"]
        b, _, h, w = feature.shape
        objness = self.score(feature).sigmoid().flatten(2)
        cls_logits = self.cls(feature).flatten(2)
        deltas = self.box_delta(feature).tanh().permute(0, 2, 3, 1).reshape(b, h * w, 4)
        ys, xs = torch.meshgrid(torch.arange(h, device=feature.device), torch.arange(w, device=feature.device), indexing="ij")
        centers = torch.stack([xs.flatten(), ys.flatten()], dim=1).float()
        outputs = ([], [], [], [], [])
        for i, (im_h, im_w) in enumerate(image_sizes):
            k = min(self.topk, h * w)
            vals, idx = torch.topk(objness[i, 0], k)
            labels = cls_logits[i, :, idx].argmax(dim=0)
            stride_x, stride_y = im_w / float(w), im_h / float(h)
            c = centers[idx]
            cx = (c[:, 0] + 0.5 + deltas[i, idx, 0] * 0.5) * stride_x
            cy = (c[:, 1] + 0.5 + deltas[i, idx, 1] * 0.5) * stride_y
            bw = (1.5 + deltas[i, idx, 2].abs()) * stride_x
            bh = (1.5 + deltas[i, idx, 3].abs()) * stride_y
            boxes = torch.stack([cx - bw, cy - bh, cx + bw, cy + bh], dim=1)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, im_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, im_h)
            boxes[:, 2:] = torch.maximum(boxes[:, 2:], boxes[:, :2] + 1.0)
            human = labels == self.human_class_index
            outputs[0].append(boxes[human])
            outputs[1].append(vals[human])
            outputs[2].append(boxes)
            outputs[3].append(vals)
            outputs[4].append(labels.long())
        return DetectionOutput(*outputs).validate()

