"""Single training entry point for compact ReliPose-HOI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from relipose_hoi.checkpoint import load_checkpoint, save_checkpoint
from relipose_hoi.config import load_config
from relipose_hoi.data import COCOKeypointsDataset, HICODataset, ImageTransform, coco_collate, hico_collate
from relipose_hoi.models import build_model
from relipose_hoi.training import make_optimizer, make_scheduler, seed_everything, train_hoi_epoch, train_pose_epoch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["pose", "hoi_oracle", "hoi_detected", "robust", "finetune"], required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--overfit", action="store_true", help="Run a tiny deterministic overfit/debug pass.")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    device = torch.device(args.device)
    transform = ImageTransform(cfg.data.image_size, cfg.data.image_mean, cfg.data.image_std, cfg.data.horizontal_flip_prob)
    if args.stage == "pose":
        if not cfg.data.coco_image_root or not cfg.data.coco_annotation_file:
            raise SystemExit("COCO paths are required for pose stage")
        dataset = COCOKeypointsDataset(cfg.data.coco_image_root, cfg.data.coco_annotation_file, transform)
        if args.max_samples is not None:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, collate_fn=coco_collate, num_workers=cfg.data.num_workers)
        correspondence = None
    else:
        if not cfg.data.hico_train_image_root or not cfg.data.hico_train_annotation_file:
            raise SystemExit("HICO train paths are required for HOI stages")
        dataset = HICODataset(cfg.data.hico_train_image_root, cfg.data.hico_train_annotation_file, transform, index_base=cfg.data.hico_index_base)
        correspondence = dataset.correspondence
        if args.max_samples is not None:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, collate_fn=hico_collate, num_workers=cfg.data.num_workers)
    model = build_model(cfg, correspondence).to(device)
    opt = make_optimizer(model, cfg)
    sch = make_scheduler(opt, cfg)
    if args.overfit:
        print("Tiny overfit/debug mode enabled. Oracle-box stages are diagnostic and non-standard.")
    if args.resume:
        load_checkpoint(args.resume, model=model, optimizer=opt, scheduler=sch, map_location=device)
    for epoch in range(cfg.train.epochs):
        if args.stage == "pose":
            metrics = train_pose_epoch(model, loader, opt, sch, cfg, device, max_steps=args.max_steps)
        else:
            metrics = train_hoi_epoch(model, loader, opt, sch, cfg, device, robust=args.stage == "robust", max_steps=args.max_steps)
        print({"epoch": epoch, **metrics})
        out = Path(cfg.train.output_dir) / "latest.pt"
        save_checkpoint(out, model=model, optimizer=opt, scheduler=sch, epoch=epoch + 1, global_step=sch.step_num, config=cfg)


if __name__ == "__main__":
    main()
