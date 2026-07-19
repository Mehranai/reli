"""Minimal COCO Keypoints reader and pose target conversion."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from relipose_hoi.structures import Batch, COCOKeypointsTarget, PoseTarget
from relipose_hoi.data.transforms import ImageTransform


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


class COCOKeypointsDataset(Dataset):
    """Reads standard local COCO person-keypoint JSON. No downloads."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        transform: ImageTransform | None = None,
        *,
        min_visible: int = 1,
        min_area: float = 1.0,
        include_crowd: bool = False,
        include_empty: bool = False,
        num_joints: int = 17,
    ) -> None:
        self.root = Path(image_root)
        self.transform = transform or ImageTransform()
        data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        images = {int(x["id"]): x for x in data.get("images", [])}
        anns_by_image: dict[int, list[dict]] = {i: [] for i in images}
        person_ids = {int(c["id"]) for c in data.get("categories", []) if c.get("name") == "person"}
        for ann in data.get("annotations", []):
            if person_ids and int(ann.get("category_id", -1)) not in person_ids:
                continue
            if (not include_crowd) and int(ann.get("iscrowd", 0)):
                continue
            kpts = ann.get("keypoints", [])
            if len(kpts) != num_joints * 3:
                raise ValueError("malformed COCO keypoint length")
            visible = sum(1 for j in range(num_joints) if kpts[3 * j + 2] > 0)
            if visible < min_visible:
                continue
            if float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])) < min_area:
                continue
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        self.items = []
        for image_id in sorted(images):
            anns = anns_by_image.get(image_id, [])
            if anns or include_empty:
                self.items.append((images[image_id], anns))
        self.num_joints = num_joints

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, COCOKeypointsTarget]:
        info, anns = self.items[index]
        image = Image.open(self.root / info["file_name"])
        boxes = torch.tensor([xywh_to_xyxy(a["bbox"]) for a in anns], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        kpts = torch.zeros((len(anns), self.num_joints, 2), dtype=torch.float32)
        vis = torch.zeros((len(anns), self.num_joints), dtype=torch.long)
        for i, ann in enumerate(anns):
            flat = ann["keypoints"]
            for j in range(self.num_joints):
                kpts[i, j] = torch.tensor(flat[3 * j : 3 * j + 2])
                vis[i, j] = int(flat[3 * j + 2])
        image_t, boxes_t, kpts_t, vis_t, meta = self.transform(image, boxes, kpts, vis)
        target = COCOKeypointsTarget(
            image_id=int(info["id"]),
            person_boxes=boxes_t if boxes_t is not None else boxes,
            keypoints_image=kpts_t if kpts_t is not None else kpts,
            keypoint_visibility=vis_t if vis_t is not None else vis,
            original_size=meta.original_size,
        )
        return image_t, target


def keypoints_image_to_roi_normalized(
    keypoints_image: Tensor,
    person_boxes: Tensor,
    label_mask: Tensor,
    visible_mask: Tensor | None = None,
) -> PoseTarget:
    visible = label_mask.bool() if visible_mask is None else visible_mask.bool()
    if keypoints_image.numel() == 0:
        return PoseTarget(
            keypoints_image.new_empty((*keypoints_image.shape[:-1], 2)),
            label_mask.bool(),
            visible,
        )
    wh = (person_boxes[:, 2:] - person_boxes[:, :2]).clamp_min(1e-6)
    coords = (keypoints_image - person_boxes[:, None, :2]) / wh[:, None, :]
    coords = coords.clamp(0.0, 1.0)
    return PoseTarget(coords, label_mask.bool(), visible)


def coco_collate(batch: list[tuple[Tensor, COCOKeypointsTarget]]) -> Batch:
    if not batch:
        raise ValueError("empty batch")
    images, targets = zip(*batch)
    return Batch(torch.stack(list(images)), list(targets), [tuple(img.shape[-2:]) for img in images])
