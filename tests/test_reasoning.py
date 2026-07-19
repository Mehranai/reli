import torch

from relipose_hoi.structures import HumanJointReasoningOutput, HumanObjectPairOutput
from relipose_hoi.models.reasoning import InteractionReasoningModule


def sample(valid=True):
    jr = torch.randn(1, 17, 32, requires_grad=True)
    r = torch.sigmoid(torch.randn(1, 17, requires_grad=True))
    mask = torch.ones(1, 17, dtype=torch.bool) if valid else torch.zeros(1, 17, dtype=torch.bool)
    human = HumanJointReasoningOutput(torch.randn(1,17,22), jr, torch.zeros(1,17), torch.ones(1,17), mask, torch.randn(1,17), r)
    pair = HumanObjectPairOutput(
        torch.randn(1,32), torch.randn(1,32), torch.randn(1,32), torch.randn(1,18), torch.randn(1,32), torch.randn(1,16),
        torch.randn(1,32, requires_grad=True), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
    )
    return human, pair


def test_relevance_gate_and_fusion_valid():
    human, pair = sample(True)
    mod = InteractionReasoningModule(32, 32, 16, 4)
    out = mod(human, pair, torch.rand(1, 17, 2), torch.tensor([[0.,0.,20.,20.]]), torch.tensor([[10.,10.,30.,30.]]), [(40,40)])
    assert torch.allclose(out.g_hoi, out.r_hi * out.a_hoi)
    assert torch.allclose(out.a_hoi.sum(1), torch.ones(1), atol=1e-5)
    out.fused_pair_features.square().mean().backward()
    assert pair.pair_features.grad is not None and pair.pair_features.grad.abs().sum() > 0


def test_all_invalid_fallback():
    human, pair = sample(False)
    mod = InteractionReasoningModule(32, 32, 16, 4)
    out = mod(human, pair, torch.rand(1, 17, 2), torch.tensor([[0.,0.,20.,20.]]), torch.tensor([[10.,10.,30.,30.]]), [(40,40)])
    assert out.a_hoi.sum() == 0
    assert out.pose_aware_pair_features.abs().sum() == 0
    assert out.fusion_gate.item() == 0
    assert torch.isfinite(out.fused_pair_features).all()
