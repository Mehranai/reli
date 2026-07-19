import json
from pathlib import Path

import torch
from PIL import Image

from relipose_hoi.data import COCO_FLIP, COCOKeypointsDataset, HICODataset, ImageTransform, coco_collate, hico_collate, keypoints_image_to_roi_normalized


def tiny_files(tmp_path: Path):
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(root / "a.jpg")
    kpts = []
    for j in range(17):
        kpts += [10 + j % 3, 5 + j % 5, 2]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 4, 20, 18], "area": 360, "iscrowd": 0, "keypoints": kpts}],
    }
    hico = {
        "filenames": ["a.jpg"],
        "annotation": [{"boxes_h": [[5, 4, 25, 22], [5, 4, 25, 22]], "boxes_o": [[20, 10, 35, 25], [20, 10, 35, 25]], "verb": [0, 1], "object": [2, 2], "hoi": [0, 1]}],
        "correspondence": [{"hoi_index": i, "object_index": i % 80, "verb_index": i % 117} for i in range(600)],
    }
    coco_p = tmp_path / "coco.json"
    hico_p = tmp_path / "hico.json"
    coco_p.write_text(json.dumps(coco))
    hico_p.write_text(json.dumps(hico))
    return root, coco_p, hico_p


def test_coco_transform_flip_and_pose_target(tmp_path):
    root, coco_p, _ = tiny_files(tmp_path)
    gen = torch.Generator().manual_seed(0)
    t = ImageTransform((64, 64), flip_prob=1.0)
    ds = COCOKeypointsDataset(root, coco_p, t)
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert target.keypoints_image.shape == (1, 17, 2)
    assert torch.equal(COCO_FLIP[COCO_FLIP], torch.arange(17))
    pose_target = keypoints_image_to_roi_normalized(target.keypoints_image, target.person_boxes, target.keypoint_label_mask)
    assert pose_target.joint_coordinates_roi_target.min() >= 0
    assert pose_target.joint_coordinates_roi_target.max() <= 1
    assert coco_collate([(image, target)]).images.shape[0] == 1


def test_hico_reader_aggregates_pairs_and_has_no_pose(tmp_path):
    root, _, hico_p = tiny_files(tmp_path)
    ds = HICODataset(root, hico_p, ImageTransform((64, 64)))
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert target.human_boxes.shape[0] == 1
    assert target.object_boxes.shape[0] == 1
    assert target.verb_targets.shape == (1, 117)
    assert target.verb_targets[0, :2].sum() == 2
    assert not any("pose" in key.lower() for key in target.__dict__)
    assert hico_collate([(image, target)]).images.shape[0] == 1

