"""Validate local COCO/HICO paths without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from relipose_hoi.config import load_config
from relipose_hoi.data import COCOKeypointsDataset, HICODataset, ImageTransform, keypoints_image_to_roi_normalized


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=["coco", "hico"], required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    transform = ImageTransform(cfg.data.image_size)
    if args.dataset == "coco":
        if not (cfg.data.coco_image_root and cfg.data.coco_annotation_file):
            raise SystemExit("COCO image_root and annotation_file are required")
        coco = COCOKeypointsDataset(cfg.data.coco_image_root, cfg.data.coco_annotation_file, transform)
        image, target = coco[0]
        pose = keypoints_image_to_roi_normalized(target.keypoints_image, target.person_boxes, target.keypoint_label_mask)
        if target.keypoints_image.shape[1] != cfg.model.num_joints:
            raise SystemExit("COCO keypoint count does not match model.num_joints")
        if target.person_boxes.numel() and not torch.isfinite(target.person_boxes).all():
            raise SystemExit("COCO boxes contain non-finite values")
        print(f"COCO samples: {len(coco)}")
        print(f"image: {tuple(image.shape)}")
        print(f"person_boxes: {tuple(target.person_boxes.shape)}")
        print(f"keypoints_image: {tuple(target.keypoints_image.shape)}")
        print(f"keypoint_visibility: {tuple(target.keypoint_visibility.shape)}")
        print(f"pose_target_roi: {tuple(pose.joint_coordinates_roi_target.shape)}")
        return

    image_root = cfg.data.hico_train_image_root or cfg.data.hico_test_image_root
    annotation_file = cfg.data.hico_train_annotation_file or cfg.data.hico_test_annotation_file
    if not (image_root and annotation_file):
        raise SystemExit("HICO image_root and annotation_file are required")
    hico = HICODataset(image_root, annotation_file, transform, index_base=cfg.data.hico_index_base)
    if len(hico) == 0:
        raise SystemExit("HICO dataset is empty")
    _, target = hico[0]
    if any("pose" in k.lower() for k in target.__dict__):
        raise SystemExit("HICO target unexpectedly contains pose fields")
    if target.object_labels.numel() and (
        target.object_labels.min() < 0 or target.object_labels.max() >= cfg.model.num_objects
    ):
        raise SystemExit("HICO object labels out of range")
    if target.pair_human_indices.numel() and (
        target.pair_human_indices.max() >= target.human_boxes.shape[0]
        or target.pair_object_indices.max() >= target.object_boxes.shape[0]
    ):
        raise SystemExit("HICO pair indices out of range")
    if hico.correspondence.num_hoi != cfg.model.num_hoi:
        raise SystemExit("HICO correspondence count does not match model.num_hoi")
    print(f"HICO samples: {len(hico)}")
    print(f"image root: {image_root}")
    print(f"human_boxes: {tuple(target.human_boxes.shape)}")
    print(f"object_boxes: {tuple(target.object_boxes.shape)}")
    print(f"object_labels: {tuple(target.object_labels.shape)}")
    print(f"pair_human_indices: {tuple(target.pair_human_indices.shape)}")
    print(f"pair_object_indices: {tuple(target.pair_object_indices.shape)}")
    print(f"verb_targets: {tuple(target.verb_targets.shape)}")
    print(f"hoi_targets: {tuple(target.hoi_targets.shape)}")
    print(f"correspondence: {hico.correspondence.num_hoi}")
    print("HICO target contains no pose inputs")


if __name__ == "__main__":
    main()
