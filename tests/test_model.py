import torch

from relipose_hoi.config import Config
from relipose_hoi.models import build_model


def targets(empty_h=False, empty_o=False):
    hb = torch.empty((0,4)) if empty_h else torch.tensor([[5.,5.,35.,50.]])
    ob = torch.empty((0,4)) if empty_o else torch.tensor([[25.,15.,55.,55.]])
    return [{
        "human_boxes": hb,
        "human_scores": torch.ones(hb.shape[0]),
        "object_boxes": ob,
        "object_scores": torch.ones(ob.shape[0]),
        "object_labels": torch.ones(ob.shape[0], dtype=torch.long),
        "pair_human_indices": torch.tensor([0]) if hb.numel() and ob.numel() else torch.empty(0, dtype=torch.long),
        "pair_object_indices": torch.tensor([0]) if hb.numel() and ob.numel() else torch.empty(0, dtype=torch.long),
        "verb_targets": torch.zeros((1 if hb.numel() and ob.numel() else 0, 117)),
    }]


def test_full_model_forward_backward_and_single_backbone():
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    cfg.model.roi_size = 5
    model = build_model(cfg)
    images = torch.randn(1, 3, 96, 96)
    out = model(images, [(96, 96)], targets(), pair_mode="training", generator=torch.Generator().manual_seed(1))
    assert out.verb_logits.shape[1] == 117
    assert out.hoi_scores.shape[1] == 600
    assert torch.allclose(out.g_hoi, out.reasoning.paired_joint_reliability * out.a_hoi)
    out.verb_logits.square().mean().backward()
    assert model.backbone.forward_calls == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())


def test_empty_humans_objects_pairs():
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    cfg.model.roi_size = 5
    model = build_model(cfg)
    for t in (targets(empty_h=True), targets(empty_o=True)):
        out = model(torch.randn(1,3,96,96), [(96,96)], t, pair_mode="training")
        assert out.verb_logits.shape[0] == 0
        assert out.hoi_scores.shape[0] == 0

def test_no_direct_600_classifier():
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    model = build_model(cfg)
    assert model.verb.num_verbs == 117
    assert model.verb.net[-1].out_features == 117

