import torch

from relipose_hoi.models.pose import IntegratedPoseModule
from relipose_hoi.models.reliability import HumanJointReasoningModule


def test_pose_shapes_ranges_and_gradients():
    roi = torch.randn(2, 32, 5, 5, requires_grad=True)
    pose = IntegratedPoseModule(32, 17, 32, 5)(roi)
    assert pose.joint_coordinates_roi.shape == (2, 17, 2)
    assert pose.joint_coordinates_roi.min() >= 0 and pose.joint_coordinates_roi.max() <= 1
    assert pose.joint_uncertainty.min() > 0
    loss = pose.joint_tokens.square().mean() + pose.joint_coordinates_roi.mean()
    loss.backward()
    assert roi.grad is not None and roi.grad.abs().sum() > 0


def test_reliability_object_independent_signature_and_gradients():
    roi = torch.randn(1, 32, 5, 5, requires_grad=True)
    pose = IntegratedPoseModule(32, 17, 32, 5)(roi)
    mod = HumanJointReasoningModule(32, 32, 17)
    out = mod(pose)
    assert out.joint_reliability.shape == (1, 17)
    assert out.joint_reliability.min() >= 0 and out.joint_reliability.max() <= 1
    assert "object" not in HumanJointReasoningModule.forward.__annotations__
    out.joint_reliability_logits.mean().backward()
    assert roi.grad is not None and roi.grad.abs().sum() > 0

def test_empty_pose_and_reliability():
    pose = IntegratedPoseModule(32, 17, 32, 5)(torch.empty(0, 32, 5, 5))
    out = HumanJointReasoningModule(32, 32, 17)(pose)
    assert pose.joint_tokens.shape[0] == 0
    assert out.joint_representations.shape[0] == 0

