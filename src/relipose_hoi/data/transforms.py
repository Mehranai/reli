"""Image, box and keypoint transforms used by COCO and HICO readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor


COCO_JOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_FLIP = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


@dataclass
class TransformMeta:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    input_size: tuple[int, int]
    scale: float
    padding: tuple[int, int, int, int]
    flipped: bool


class ImageTransform:
    """RGB -> aspect-preserving resize -> top-left pad -> normalize."""

    def __init__(
        self,
        size: tuple[int, int] = (128, 128),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        flip_prob: float = 0.0,
    ) -> None:
        self.size = tuple(int(x) for x in size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.flip_prob = float(flip_prob)

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor | None = None,
        keypoints: Tensor | None = None,
        visibility: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, TransformMeta]:
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        out_h, out_w = self.size
        scale = min(out_w / orig_w, out_h / orig_h)
        new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        flip = False
        if self.flip_prob > 0:
            r = torch.rand((), generator=generator).item() if generator is not None else torch.rand(()).item()
            flip = r < self.flip_prob
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        canvas = Image.new("RGB", (out_w, out_h))
        canvas.paste(image, (0, 0))
        byte = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = byte.view(out_h, out_w, 3).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std

        boxes_out = None
        if boxes is not None:
            boxes_out = boxes.clone().float() * scale
            if boxes_out.numel() and flip:
                x1 = boxes_out[:, 0].clone()
                x2 = boxes_out[:, 2].clone()
                boxes_out[:, 0] = new_w - x2
                boxes_out[:, 2] = new_w - x1
            if boxes_out.numel():
                boxes_out[:, [0, 2]] = boxes_out[:, [0, 2]].clamp(0, out_w)
                boxes_out[:, [1, 3]] = boxes_out[:, [1, 3]].clamp(0, out_h)

        kpts_out = None
        vis_out = visibility.clone() if visibility is not None else None
        if keypoints is not None:
            kpts_out = keypoints.clone().float() * scale
            if kpts_out.numel() and flip:
                kpts_out[..., 0] = new_w - kpts_out[..., 0]
                kpts_out = kpts_out[:, COCO_FLIP.to(kpts_out.device)]
                if vis_out is not None:
                    vis_out = vis_out[:, COCO_FLIP.to(vis_out.device)]
            if kpts_out.numel():
                kpts_out[..., 0] = kpts_out[..., 0].clamp(0, out_w)
                kpts_out[..., 1] = kpts_out[..., 1].clamp(0, out_h)

        meta = TransformMeta(
            original_size=(orig_h, orig_w),
            resized_size=(new_h, new_w),
            input_size=(out_h, out_w),
            scale=float(scale),
            padding=(0, 0, out_w - new_w, out_h - new_h),
            flipped=flip,
        )
        return tensor, boxes_out, kpts_out, vis_out, meta
