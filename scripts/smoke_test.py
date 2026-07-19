"""End-to-end CPU smoke test for the compact ReliPose-HOI repo."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image
from torch.utils.data import DataLoader

from relipose_hoi.checkpoint import load_checkpoint, save_checkpoint
from relipose_hoi.config import load_config
from relipose_hoi.corruption import PoseCorruptor
from relipose_hoi.data import COCOKeypointsDataset, HICODataset, ImageTransform, coco_collate, hico_collate, keypoints_image_to_roi_normalized
from relipose_hoi.losses import pose_loss, total_hoi_loss
from relipose_hoi.models import build_model
from relipose_hoi.training import make_optimizer, make_scheduler, robust_forward, seed_everything, train_hoi_epoch


def _tiny_files(tmp: Path) -> tuple[Path, Path, Path]:
    root = tmp / "images"
    root.mkdir()
    Image.new("RGB", (64, 48), (140, 120, 90)).save(root / "a.jpg")
    keypoints = []
    for j in range(17):
        keypoints += [20 + j % 4, 10 + j, 2]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 6, 32, 36], "area": 1152, "iscrowd": 0, "keypoints": keypoints}],
    }
    hico = {
        "filenames": ["a.jpg"],
        "annotation": [{"boxes_h": [[8, 6, 40, 42]], "boxes_o": [[30, 15, 55, 40]], "verb": [0], "object": [1], "hoi": [1]}],
        "correspondence": [{"hoi_index": i, "object_index": i % 80, "verb_index": i % 117} for i in range(600)],
        "objects": [str(i) for i in range(80)],
        "verbs": [str(i) for i in range(117)],
    }
    coco_path = tmp / "coco.json"
    hico_path = tmp / "hico.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    hico_path.write_text(json.dumps(hico), encoding="utf-8")
    return root, coco_path, hico_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as td:
        root, coco_json, hico_json = _tiny_files(Path(td))
        transform = ImageTransform(cfg.data.image_size)
        coco = COCOKeypointsDataset(root, coco_json, transform)
        hico = HICODataset(root, hico_json, transform)
        coco_batch = coco_collate([coco[0]])
        hico_batch = hico_collate([hico[0]])
        assert not any("pose" in k.lower() for k in hico_batch.targets[0].__dict__)

        model = build_model(cfg, hico.correspondence).to(device)
        model.train()
        targets = [hico_batch.targets[0].as_model_target()]
        out = model(hico_batch.images, hico_batch.image_sizes, targets, pair_mode="training", generator=torch.Generator().manual_seed(0))
        assert out.verb_logits.shape[1] == 117
        assert out.hoi_scores.shape[1] == 600
        assert torch.allclose(out.g_hoi, out.reasoning.paired_joint_reliability * out.a_hoi)
        loss = out.verb_logits.square().mean()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())

        pose_target = keypoints_image_to_roi_normalized(coco_batch.targets[0].keypoints_image, coco_batch.targets[0].person_boxes, coco_batch.targets[0].keypoint_label_mask)
        pl = pose_loss(out.pose, pose_target, cfg.losses)["pose"]
        assert torch.isfinite(pl)

        opt = make_optimizer(model, cfg)
        sch = make_scheduler(opt, cfg)
        metrics = train_hoi_epoch(model, DataLoader([hico[0]], batch_size=1, collate_fn=hico_collate), opt, sch, cfg, device)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

        clean = model(hico_batch.images, hico_batch.image_sizes, targets, pair_mode="training", generator=torch.Generator().manual_seed(1))
        robust = robust_forward(model, clean, hico_batch.image_sizes, cfg, torch.Generator().manual_seed(1))
        assert model.backbone.forward_calls >= 1
        assert model.pair.forward_calls >= 1
        loss_dict = total_hoi_loss(clean, verb_targets=clean.training_pairs.verb_targets, cfg=cfg.losses, corrupted=(robust.corrupted_pose, clean.__class__(
            clean.detections, clean.roi, robust.corrupted_pose.corrupted_pose, robust.corrupted_reliability, clean.pairs, robust.corrupted_reasoning,
            robust.corrupted_verb_logits, robust.corrupted_verb_logits.sigmoid(), clean.valid_verb_mask, robust.corrupted_hoi_scores, clean.pair_object_labels, clean.training_pairs
        )))
        assert torch.isfinite(loss_dict["total"])

        ckpt = Path(td) / "ckpt.pt"
        save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch, epoch=1, global_step=2, config=cfg)
        state = load_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch)
        assert state["epoch"] == 1 and state["global_step"] == 2
    print("smoke_test passed")


if __name__ == "__main__":
    main()
