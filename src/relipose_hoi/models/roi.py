"""Human, object and union RoI extraction from shared FPN maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from relipose_hoi.structures import RoIOutput, check_boxes


def offsets(counts: list[int], device: torch.device) -> Tensor:
    return torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.tensor(counts, dtype=torch.long, device=device).cumsum(0)])


def sample_boxes(feature: Tensor, boxes: list[Tensor], image_sizes: Sequence[tuple[int, int]], size: int) -> Tensor:
    c = feature.shape[1]
    if sum(x.shape[0] for x in boxes) == 0:
        return feature.new_empty((0, c, size, size))
    axis = torch.linspace(0, 1, size, device=feature.device, dtype=feature.dtype)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    outs = []
    for b, image_boxes in enumerate(boxes):
        if image_boxes.numel() == 0:
            continue
        h, w = image_sizes[b]
        x1, y1, x2, y2 = image_boxes.to(feature.device).unbind(1)
        sx = x1[:, None, None] + (x2 - x1)[:, None, None] * gx
        sy = y1[:, None, None] + (y2 - y1)[:, None, None] * gy
        grid = torch.stack([2 * sx / max(w, 1) - 1, 2 * sy / max(h, 1) - 1], -1)
        fmap = feature[b : b + 1].expand(image_boxes.shape[0], -1, -1, -1)
        outs.append(F.grid_sample(fmap, grid, padding_mode="border", align_corners=False))
    return torch.cat(outs, 0)


class MultiRoIExtractor(nn.Module):
    def __init__(self, out_channels: int = 64, output_size: int = 7) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.output_size = output_size

    def forward(
        self,
        *,
        features: Mapping[str, Tensor],
        image_sizes: Sequence[tuple[int, int]],
        human_boxes: Sequence[Tensor],
        object_boxes: Sequence[Tensor],
        pair_human_indices: Sequence[Tensor],
        pair_object_indices: Sequence[Tensor],
    ) -> RoIOutput:
        feature = features["0"]
        device = feature.device
        h_boxes = [b.to(device=device, dtype=torch.float32) for b in human_boxes]
        o_boxes = [b.to(device=device, dtype=torch.float32) for b in object_boxes]
        for i, b in enumerate(h_boxes):
            check_boxes(b, f"human_boxes[{i}]")
        for i, b in enumerate(o_boxes):
            check_boxes(b, f"object_boxes[{i}]")
        h_counts = [x.shape[0] for x in h_boxes]
        o_counts = [x.shape[0] for x in o_boxes]
        p_counts = [x.numel() for x in pair_human_indices]
        h_off, o_off, p_off = offsets(h_counts, device), offsets(o_counts, device), offsets(p_counts, device)
        unions, gh, go, pi = [], [], [], []
        for i, (hb, ob, hi, oi) in enumerate(zip(h_boxes, o_boxes, pair_human_indices, pair_object_indices)):
            hi = hi.to(device=device, dtype=torch.long)
            oi = oi.to(device=device, dtype=torch.long)
            if hi.numel() == 0:
                unions.append(hb.new_empty((0, 4)))
                continue
            u = torch.cat([torch.minimum(hb[hi, :2], ob[oi, :2]), torch.maximum(hb[hi, 2:], ob[oi, 2:])], 1)
            unions.append(u)
            gh.append(hi + h_off[i])
            go.append(oi + o_off[i])
            pi.append(torch.full_like(hi, i))
        pair_h = torch.cat(gh) if gh else torch.empty(0, dtype=torch.long, device=device)
        pair_o = torch.cat(go) if go else torch.empty(0, dtype=torch.long, device=device)
        pair_i = torch.cat(pi) if pi else torch.empty(0, dtype=torch.long, device=device)
        return RoIOutput(
            human_roi_features=sample_boxes(feature, h_boxes, image_sizes, self.output_size),
            object_roi_features=sample_boxes(feature, o_boxes, image_sizes, self.output_size),
            union_roi_features=sample_boxes(feature, unions, image_sizes, self.output_size),
            human_offsets=h_off,
            object_offsets=o_off,
            pair_offsets=p_off,
            pair_human_indices_global=pair_h,
            pair_object_indices_global=pair_o,
            pair_to_image=pair_i,
            union_boxes=unions,
        )

