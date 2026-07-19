import torch

from relipose_hoi.config import Config
from relipose_hoi.corruption import PoseCorruptor
from relipose_hoi.losses import reliability_loss, sparse_loss, verb_loss
from relipose_hoi.models import build_model


def model_output():
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    cfg.model.roi_size = 5
    model = build_model(cfg)
    target = [{
        "human_boxes": torch.tensor([[5.,5.,35.,50.]]),
        "human_scores": torch.ones(1),
        "object_boxes": torch.tensor([[25.,15.,55.,55.]]),
        "object_scores": torch.ones(1),
        "object_labels": torch.ones(1, dtype=torch.long),
        "pair_human_indices": torch.tensor([0]),
        "pair_object_indices": torch.tensor([0]),
        "verb_targets": torch.zeros((1, 117)),
    }]
    out = model(torch.randn(1,3,96,96), [(96,96)], target, pair_mode="training")
    return cfg, out


def test_verb_loss_and_projection_mask():
    cfg, out = model_output()
    target = torch.zeros_like(out.verb_logits)
    loss = verb_loss(out, target, cfg.losses)["verb"]
    assert torch.isfinite(loss)
    loss.backward()
    assert (out.hoi_scores >= 0).all() and (out.hoi_scores <= 1).all()


def test_corruption_and_reliability_loss_object_independent():
    cfg, out = model_output()
    corrupt = PoseCorruptor(cfg.corruption)(out.pose, out.reliability.joint_valid_mask, generator=torch.Generator().manual_seed(2))
    assert corrupt.synthetic_reliability_target.min() >= 0
    assert corrupt.synthetic_reliability_target.max() <= 1
    loss = reliability_loss(out.reliability.joint_reliability_logits, corrupt.synthetic_reliability_target, corrupt.reliability_supervision_mask, cfg.losses)["reliability"]
    assert torch.isfinite(loss)


def test_sparse_loss_empty_safe():
    _, out = model_output()
    assert torch.isfinite(sparse_loss(out.pose, Config().losses)["sparse"])
