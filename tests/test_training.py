import torch
from torch.utils.data import DataLoader

from relipose_hoi.checkpoint import load_checkpoint, save_checkpoint
from relipose_hoi.config import Config
from relipose_hoi.models import build_model
from relipose_hoi.structures import Batch, COCOKeypointsTarget, HICOTarget
from relipose_hoi.training import make_optimizer, make_scheduler, robust_forward, train_hoi_epoch, train_pose_epoch


def batch():
    t = HICOTarget(
        image_id=0,
        human_boxes=torch.tensor([[5.,5.,35.,50.]]),
        object_boxes=torch.tensor([[25.,15.,55.,55.]]),
        object_labels=torch.ones(1, dtype=torch.long),
        pair_human_indices=torch.tensor([0]),
        pair_object_indices=torch.tensor([0]),
        verb_targets=torch.zeros((1,117)),
        hoi_targets=torch.zeros((1,600)),
        original_size=(96,96),
    )
    return Batch(torch.randn(1,3,96,96), [t], [(96,96)])


def pose_batch():
    vis = torch.full((1, 17), 2, dtype=torch.long)
    keypoints = torch.stack([torch.linspace(8, 32, 17), torch.linspace(10, 50, 17)], -1).unsqueeze(0)
    t = COCOKeypointsTarget(
        image_id=0,
        person_boxes=torch.tensor([[5., 5., 40., 60.]]),
        keypoints_image=keypoints,
        keypoint_visibility=vis,
        original_size=(96, 96),
    )
    return Batch(torch.randn(1, 3, 96, 96), [t], [(96, 96)])


def test_training_update_checkpoint_and_robust_counts(tmp_path):
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    cfg.model.roi_size = 5
    model = build_model(cfg)
    opt = make_optimizer(model, cfg)
    sch = make_scheduler(opt, cfg)
    before = next(model.verb.parameters()).detach().clone()
    metrics = train_hoi_epoch(model, [batch()], opt, sch, cfg, torch.device("cpu"), robust=False)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.allclose(before, next(model.verb.parameters()).detach())
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch, epoch=1, global_step=sch.step_num, config=cfg)
    state = load_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch)
    assert state["epoch"] == 1
    clean = model(batch().images, batch().image_sizes, [batch().targets[0].as_model_target()], pair_mode="training")
    b0, p0 = model.backbone.forward_calls, model.pair.forward_calls
    robust_forward(model, clean, batch().image_sizes, cfg, torch.Generator().manual_seed(0))
    assert model.backbone.forward_calls == b0
    assert model.pair.forward_calls == p0


def test_pose_pretraining_update():
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    cfg.model.roi_size = 5
    model = build_model(cfg)
    opt = make_optimizer(model, cfg)
    sch = make_scheduler(opt, cfg)
    before = model.pose.tokenizer.coord.weight.detach().clone()
    metrics = train_pose_epoch(model, [pose_batch()], opt, sch, cfg, torch.device("cpu"), max_steps=1)
    assert torch.isfinite(torch.tensor(metrics["pose_loss"]))
    assert not torch.allclose(before, model.pose.tokenizer.coord.weight.detach())


def test_robust_hico_training_update_keeps_cached_visual_counts():
    cfg = Config()
    cfg.model.fpn_dim = cfg.model.pose_dim = cfg.model.joint_dim = cfg.model.pair_dim = 32
    cfg.model.roi_size = 5
    model = build_model(cfg)
    opt = make_optimizer(model, cfg)
    sch = make_scheduler(opt, cfg)
    metrics = train_hoi_epoch(model, [batch()], opt, sch, cfg, torch.device("cpu"), robust=True, max_steps=1)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert model.backbone.forward_calls == 1
    assert model.pair.forward_calls == 1
