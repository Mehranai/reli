"""Pair construction, pair geometry, object semantics and q_ho encoding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from relipose_hoi.structures import DetectionOutput, HumanObjectPairOutput, PairBuildOutput, RoIOutput


def _offsets(counts: list[int], device: torch.device) -> Tensor:
    return torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.tensor(counts, dtype=torch.long, device=device).cumsum(0)])


def flatten_detections(d: DetectionOutput) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    device = d.human_boxes[0].device if d.human_boxes else torch.device("cpu")
    hc, oc = [x.shape[0] for x in d.human_boxes], [x.shape[0] for x in d.object_boxes]
    hb = torch.cat(d.human_boxes) if sum(hc) else torch.empty((0, 4), device=device)
    ob = torch.cat(d.object_boxes) if sum(oc) else torch.empty((0, 4), device=device)
    ol = torch.cat(d.object_labels) if sum(oc) else torch.empty((0,), dtype=torch.long, device=device)
    hti = torch.cat([torch.full((c,), i, dtype=torch.long, device=device) for i, c in enumerate(hc)]) if sum(hc) else torch.empty((0,), dtype=torch.long, device=device)
    oti = torch.cat([torch.full((c,), i, dtype=torch.long, device=device) for i, c in enumerate(oc)]) if sum(oc) else torch.empty((0,), dtype=torch.long, device=device)
    hs = torch.cat(d.human_scores) if sum(hc) else hb.new_empty((0,))
    os = torch.cat(d.object_scores) if sum(oc) else ob.new_empty((0,))
    return hb, ob, ol, hti, oti, _offsets(hc, device), _offsets(oc, device), hs, os


class TrainingPairBuilder:
    def __init__(self, num_verbs: int = 117, negative_ratio: float = 2.0, max_negatives: int = 64) -> None:
        self.num_verbs, self.negative_ratio, self.max_negatives = num_verbs, negative_ratio, max_negatives

    def __call__(self, human_boxes: Sequence[Tensor], object_boxes: Sequence[Tensor], targets: Sequence[dict[str, Tensor]], *, generator: torch.Generator | None = None) -> PairBuildOutput:
        device = human_boxes[0].device if human_boxes else torch.device("cpu")
        pair_h, pair_o, vt, pos, counts = [], [], [], [], []
        for img, (hb, ob, tgt) in enumerate(zip(human_boxes, object_boxes, targets)):
            ph = tgt.get("pair_human_indices", torch.empty(0, dtype=torch.long)).to(device)
            po = tgt.get("pair_object_indices", torch.empty(0, dtype=torch.long)).to(device)
            verbs = tgt.get("verb_targets", torch.empty((ph.numel(), self.num_verbs))).to(device)
            seen = {(int(h), int(o)) for h, o in zip(ph.tolist(), po.tolist())}
            local_h, local_o, local_v, local_pos = list(ph), list(po), list(verbs), [True] * ph.numel()
            candidates = [(h, o) for h in range(hb.shape[0]) for o in range(ob.shape[0]) if (h, o) not in seen]
            need = min(len(candidates), self.max_negatives, int(max(1, ph.numel()) * self.negative_ratio))
            if need and candidates:
                order = torch.randperm(len(candidates), generator=generator)[:need].tolist()
                for idx in order:
                    h, o = candidates[idx]
                    local_h.append(torch.tensor(h, device=device))
                    local_o.append(torch.tensor(o, device=device))
                    local_v.append(torch.zeros(self.num_verbs, device=device))
                    local_pos.append(False)
            counts.append(len(local_h))
            pair_h.extend(local_h)
            pair_o.extend(local_o)
            vt.extend(local_v)
            pos.extend(local_pos)
        ph = torch.stack(pair_h).long() if pair_h else torch.empty(0, dtype=torch.long, device=device)
        po = torch.stack(pair_o).long() if pair_o else torch.empty(0, dtype=torch.long, device=device)
        verb = torch.stack(vt).float() if vt else torch.empty((0, self.num_verbs), device=device)
        pm = torch.tensor(pos, dtype=torch.bool, device=device)
        return PairBuildOutput(_split(ph, counts), _split(po, counts), _offsets(counts, device), verb, pm, ~pm)


class InferencePairBuilder:
    def __init__(self, human_class_index: int = 0, max_pairs: int = 1000) -> None:
        self.human_class_index = human_class_index
        self.max_pairs = max_pairs

    def __call__(self, human_boxes: Sequence[Tensor], object_boxes: Sequence[Tensor]) -> PairBuildOutput:
        device = human_boxes[0].device if human_boxes else torch.device("cpu")
        hs, os, counts = [], [], []
        for hb, ob in zip(human_boxes, object_boxes):
            pairs = [(h, o) for h in range(hb.shape[0]) for o in range(ob.shape[0])]
            pairs = pairs[: self.max_pairs]
            counts.append(len(pairs))
            hs.extend(torch.tensor(h, device=device) for h, _ in pairs)
            os.extend(torch.tensor(o, device=device) for _, o in pairs)
        ph = torch.stack(hs).long() if hs else torch.empty(0, dtype=torch.long, device=device)
        po = torch.stack(os).long() if os else torch.empty(0, dtype=torch.long, device=device)
        return PairBuildOutput(_split(ph, counts), _split(po, counts), _offsets(counts, device))


def _split(values: Tensor, counts: list[int]) -> list[Tensor]:
    out, start = [], 0
    for c in counts:
        out.append(values[start : start + c])
        start += c
    return out


class PairEncoder(nn.Module):
    def __init__(self, in_channels: int, pair_dim: int = 64, num_objects: int = 80) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.visual = nn.Sequential(nn.Conv2d(in_channels, pair_dim, 3, padding=1), nn.GroupNorm(8, pair_dim), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.geom = nn.Sequential(nn.Linear(20, pair_dim), nn.GELU(), nn.Linear(pair_dim, pair_dim))
        self.obj = nn.Embedding(num_objects, pair_dim // 2)
        self.fuse = nn.Sequential(nn.LayerNorm(pair_dim * 4 + pair_dim // 2 + 2), nn.Linear(pair_dim * 4 + pair_dim // 2 + 2, pair_dim), nn.GELU(), nn.Linear(pair_dim, pair_dim))

    def encode_roi(self, x: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.pair_dim))
        return self.visual(x).flatten(1)

    def geometry(self, h: Tensor, o: Tensor, image_sizes: Tensor) -> Tensor:
        if h.shape[0] == 0:
            return h.new_empty((0, 18))
        wh = image_sizes[:, [1, 0, 1, 0]].clamp_min(1)
        hn, on = h / wh, o / wh
        hc, oc = (hn[:, :2] + hn[:, 2:]) / 2, (on[:, :2] + on[:, 2:]) / 2
        hw, hh = (h[:, 2] - h[:, 0]).clamp_min(1), (h[:, 3] - h[:, 1]).clamp_min(1)
        ow, oh = (o[:, 2] - o[:, 0]).clamp_min(1), (o[:, 3] - o[:, 1]).clamp_min(1)
        inter = torch.cat([torch.maximum(h[:, :2], o[:, :2]), torch.minimum(h[:, 2:], o[:, 2:])], 1)
        ia = (inter[:, 2] - inter[:, 0]).clamp_min(0) * (inter[:, 3] - inter[:, 1]).clamp_min(0)
        union = hw * hh + ow * oh - ia
        extra = torch.stack([(oc[:, 0] - hc[:, 0]) / hw, (oc[:, 1] - hc[:, 1]) / hh, (ow / hw).log(), (oh / hh).log(), ia / union.clamp_min(1), ia / (hw * hh).clamp_min(1), ia / (ow * oh).clamp_min(1), (oc - hc).norm(dim=-1)], 1)
        return torch.cat([hn, on, hc, oc, extra], 1)


class HumanObjectPairModule(nn.Module):
    """RoI features + boxes + labels -> q_ho."""

    def __init__(self, in_channels: int, pair_dim: int = 64, num_objects: int = 80) -> None:
        super().__init__()
        self.encoder = PairEncoder(in_channels, pair_dim, num_objects)
        self.forward_calls = 0

    def forward(self, roi: RoIOutput, human_boxes: Tensor, object_boxes: Tensor, object_labels: Tensor, image_sizes: Sequence[tuple[int, int]], human_scores: Tensor | None = None, object_scores: Tensor | None = None) -> HumanObjectPairOutput:
        self.forward_calls += 1
        ph, po = roi.pair_human_indices_global, roi.pair_object_indices_global
        n = ph.numel()
        if n == 0:
            e = roi.union_roi_features.new_empty
            return HumanObjectPairOutput(e((0, self.encoder.pair_dim)), e((0, self.encoder.pair_dim)), e((0, self.encoder.pair_dim)), e((0, 20)), e((0, self.encoder.pair_dim)), e((0, self.encoder.pair_dim // 2)), e((0, self.encoder.pair_dim)), ph, po, roi.pair_to_image)
        hf = self.encoder.encode_roi(roi.human_roi_features.index_select(0, ph))
        of = self.encoder.encode_roi(roi.object_roi_features.index_select(0, po))
        uf = self.encoder.encode_roi(roi.union_roi_features)
        sizes = torch.tensor([image_sizes[int(i)] for i in roi.pair_to_image.tolist()], dtype=human_boxes.dtype, device=human_boxes.device)
        geom = self.encoder.geometry(human_boxes[ph], object_boxes[po], sizes)
        gf = self.encoder.geom(geom)
        ef = self.encoder.obj(object_labels[po].clamp(0, self.encoder.obj.num_embeddings - 1))
        hs = torch.ones(n, device=hf.device) if human_scores is None else human_scores[ph]
        os = torch.ones(n, device=hf.device) if object_scores is None else object_scores[po]
        pair = self.encoder.fuse(torch.cat([hf, of, uf, gf, ef, hs[:, None], os[:, None]], 1))
        return HumanObjectPairOutput(hf, of, uf, geom, gf, ef, pair, ph, po, roi.pair_to_image)
