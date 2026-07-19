"""Prepared HICO-DET JSON reader and correspondence loading."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from relipose_hoi.structures import Batch, HICOCorrespondenceTable, HICOTarget
from relipose_hoi.data.transforms import ImageTransform


def _to_zero(x: int, base: int) -> int:
    return int(x) - int(base)


def convert_hico_annotation(
    ann: dict,
    *,
    num_verbs: int = 117,
    num_hoi: int = 600,
    index_base: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    boxes_h = ann.get("boxes_h", [])
    boxes_o = ann.get("boxes_o", [])
    verbs = ann.get("verb", [])
    objects = ann.get("object", [])
    hois = ann.get("hoi", [])
    lengths = {len(boxes_h), len(boxes_o), len(verbs), len(objects), len(hois)}
    if len(lengths) != 1:
        raise ValueError("HICO annotation fields must have equal lengths")
    human_map: dict[tuple[float, ...], int] = {}
    object_map: dict[tuple[float, ...], int] = {}
    humans: list[list[float]] = []
    objects_out: list[list[float]] = []
    labels: list[int] = []
    pair_map: dict[tuple[int, int], int] = {}
    pair_h: list[int] = []
    pair_o: list[int] = []
    verb_targets: list[Tensor] = []
    hoi_targets: list[Tensor] = []
    for bh, bo, verb, obj, hoi in zip(boxes_h, boxes_o, verbs, objects, hois):
        obj_i = _to_zero(obj, index_base)
        verb_i = _to_zero(verb, index_base)
        hoi_i = _to_zero(hoi, index_base)
        if not 0 <= verb_i < num_verbs or not 0 <= hoi_i < num_hoi or obj_i < 0:
            raise ValueError("HICO label out of range")
        hk = tuple(float(x) for x in bh)
        ok = tuple(float(x) for x in bo) + (float(obj_i),)
        if hk not in human_map:
            human_map[hk] = len(humans)
            humans.append(list(hk))
        if ok not in object_map:
            object_map[ok] = len(objects_out)
            objects_out.append(list(ok[:4]))
            labels.append(obj_i)
        key = (human_map[hk], object_map[ok])
        if key not in pair_map:
            pair_map[key] = len(pair_h)
            pair_h.append(key[0])
            pair_o.append(key[1])
            verb_targets.append(torch.zeros(num_verbs))
            hoi_targets.append(torch.zeros(num_hoi))
        p = pair_map[key]
        verb_targets[p][verb_i] = 1.0
        hoi_targets[p][hoi_i] = 1.0
    return (
        torch.tensor(humans, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(objects_out, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(pair_h, dtype=torch.long),
        torch.tensor(pair_o, dtype=torch.long),
        torch.stack(verb_targets) if verb_targets else torch.empty((0, num_verbs)),
        torch.stack(hoi_targets) if hoi_targets else torch.empty((0, num_hoi)),
    )


def load_hico_correspondence(
    annotation_file: str | Path,
    *,
    num_objects: int = 80,
    num_verbs: int = 117,
    num_hoi: int = 600,
    index_base: int = 0,
) -> HICOCorrespondenceTable:
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    records = data.get("correspondence")
    if records is None:
        raise ValueError("HICO JSON is missing correspondence")
    return HICOCorrespondenceTable.from_records(
        records,
        num_objects=num_objects,
        num_verbs=num_verbs,
        num_hoi=num_hoi,
        index_base=index_base,
    )


class HICODataset(Dataset):
    """Prepared HICO JSON reader. Targets never contain pose predictions."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        transform: ImageTransform | None = None,
        *,
        index_base: int = 0,
        include_empty: bool = True,
        num_verbs: int = 117,
        num_hoi: int = 600,
    ) -> None:
        self.root = Path(image_root)
        self.transform = transform or ImageTransform()
        self.data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        self.index_base = index_base
        self.num_verbs = num_verbs
        self.num_hoi = num_hoi
        self.filenames = self.data.get("filenames", [])
        annotations = self.data.get("annotation", [{} for _ in self.filenames])
        self.items = [
            (i, f, annotations[i] if i < len(annotations) else {})
            for i, f in enumerate(self.filenames)
            if include_empty or len((annotations[i] if i < len(annotations) else {}).get("boxes_h", []))
        ]
        self.correspondence = load_hico_correspondence(
            annotation_file,
            num_verbs=num_verbs,
            num_hoi=num_hoi,
            index_base=index_base,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, HICOTarget]:
        image_id, filename, ann = self.items[index]
        image = Image.open(self.root / filename)
        h, o, labels, ph, po, vt, ht = convert_hico_annotation(
            ann,
            num_verbs=self.num_verbs,
            num_hoi=self.num_hoi,
            index_base=self.index_base,
        )
        all_boxes = torch.cat([h, o], dim=0) if h.numel() or o.numel() else torch.empty((0, 4))
        image_t, boxes_t, _, _, meta = self.transform(image, all_boxes)
        if boxes_t is None:
            boxes_t = all_boxes
        h_t = boxes_t[: h.shape[0]]
        o_t = boxes_t[h.shape[0] :]
        target = HICOTarget(
            image_id=int(image_id),
            human_boxes=h_t,
            object_boxes=o_t,
            object_labels=labels,
            pair_human_indices=ph,
            pair_object_indices=po,
            verb_targets=vt,
            hoi_targets=ht,
            original_size=meta.original_size,
        )
        return image_t, target


def hico_collate(batch: list[tuple[Tensor, HICOTarget]]) -> Batch:
    if not batch:
        raise ValueError("empty batch")
    images, targets = zip(*batch)
    return Batch(torch.stack(list(images)), list(targets), [tuple(img.shape[-2:]) for img in images])


def create_train_validation_indices(n: int, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1)")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    k = max(1, int(round(n * validation_ratio))) if n else 0
    return sorted(perm[k:]), sorted(perm[:k])

