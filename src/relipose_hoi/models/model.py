"""Complete ReliPose-HOI model and simple builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from relipose_hoi.config import Config
from relipose_hoi.models.backbone import SharedResNetFPN
from relipose_hoi.models.detection import DetectionBranch, OracleDetectionBranch, SharedFPNDetector
from relipose_hoi.models.pair import HumanObjectPairModule, InferencePairBuilder, TrainingPairBuilder, flatten_detections
from relipose_hoi.models.pose import IntegratedPoseModule
from relipose_hoi.models.reasoning import InteractionReasoningModule, joint_roi_to_image
from relipose_hoi.models.reliability import HumanJointReasoningModule
from relipose_hoi.models.roi import MultiRoIExtractor
from relipose_hoi.structures import HICOCorrespondenceTable, ReliPoseHOIOutput


class VerbHead(nn.Module):
    def __init__(self, pair_dim: int = 64, num_verbs: int = 117, positive_prior: float = 0.01) -> None:
        super().__init__()
        self.num_verbs = num_verbs
        self.net = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, pair_dim), nn.GELU(), nn.Linear(pair_dim, num_verbs))
        nn.init.constant_(self.net[-1].bias, torch.logit(torch.tensor(positive_prior)).item())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) if x.shape[0] else x.new_empty((0, self.num_verbs))


class HOIProjector(nn.Module):
    def __init__(self, table: HICOCorrespondenceTable) -> None:
        super().__init__()
        self.table = table
        self.register_buffer("hoi_to_object", table.hoi_to_object.clone())
        self.register_buffer("hoi_to_verb", table.hoi_to_verb.clone())
        self.register_buffer("valid_by_object", table.valid_verb_by_object.clone())

    def forward(self, verb_logits: Tensor, object_labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probs = verb_logits.sigmoid()
        if object_labels.numel() == 0:
            return probs, probs.new_empty(probs.shape, dtype=torch.bool), probs.new_empty((0, self.hoi_to_object.numel()))
        labels = object_labels.long().clamp(0, self.valid_by_object.shape[0] - 1)
        valid = self.valid_by_object[labels]
        hoi_probs = probs[:, self.hoi_to_verb]
        obj_ok = labels[:, None] == self.hoi_to_object[None]
        return probs, valid, torch.where(obj_ok, hoi_probs, torch.zeros_like(hoi_probs))


class ReliPoseHOIModel(nn.Module):
    """End-to-end paper architecture with exactly one shared backbone."""

    def __init__(
        self,
        backbone: SharedResNetFPN,
        detector: DetectionBranch,
        roi: MultiRoIExtractor,
        pose: IntegratedPoseModule,
        reliability: HumanJointReasoningModule,
        pair: HumanObjectPairModule,
        reasoning: InteractionReasoningModule,
        verb: VerbHead,
        projector: HOIProjector,
        training_pairs: TrainingPairBuilder,
        inference_pairs: InferencePairBuilder,
    ) -> None:
        super().__init__()
        self.backbone, self.detector, self.roi = backbone, detector, roi
        self.pose, self.reliability, self.pair, self.reasoning = pose, reliability, pair, reasoning
        self.verb, self.projector = verb, projector
        self.training_pair_builder, self.inference_pair_builder = training_pairs, inference_pairs

    def forward(self, images: Tensor, image_sizes: Sequence[tuple[int, int]], targets: Sequence[Mapping[str, Tensor]] | None = None, *, pair_mode: str = "auto", generator: torch.Generator | None = None) -> ReliPoseHOIOutput:
        mode = ("training" if self.training else "inference") if pair_mode == "auto" else pair_mode
        features = self.backbone(images)
        det = self.detector(features, image_sizes, targets).validate()
        if mode == "training":
            if targets is None:
                raise ValueError("training pair mode requires targets")
            pair_build = self.training_pair_builder(det.human_boxes, det.object_boxes, targets, generator=generator)
        else:
            pair_build = self.inference_pair_builder(det.human_boxes, det.object_boxes)
        roi = self.roi(features=features, image_sizes=image_sizes, human_boxes=det.human_boxes, object_boxes=det.object_boxes, pair_human_indices=pair_build.pair_human_indices, pair_object_indices=pair_build.pair_object_indices)
        hb, ob, ol, hti, _, _, _, hs, os = flatten_detections(det)
        pose = self.pose(roi.human_roi_features)
        rel = self.reliability(pose)
        pairs = self.pair(roi, hb, ob, ol, image_sizes, hs, os)
        joint_img = joint_roi_to_image(pose.joint_coordinates_roi, hb, hti, image_sizes)
        reasoning = self.reasoning(rel, pairs, joint_img, hb, ob, image_sizes)
        logits = self.verb(reasoning.fused_pair_features)
        pair_labels = ol.index_select(0, pairs.pair_object_indices_global) if pairs.pair_object_indices_global.numel() else ol.new_empty((0,))
        probs, valid, hoi = self.projector(logits, pair_labels)
        return ReliPoseHOIOutput(det, roi, pose, rel, pairs, reasoning, logits, probs, valid, hoi, pair_labels, pair_build if mode == "training" else None)


def build_model(config: Config, correspondence: HICOCorrespondenceTable | None = None, detector: DetectionBranch | None = None) -> ReliPoseHOIModel:
    m = config.model
    table = correspondence or HICOCorrespondenceTable.synthetic(num_objects=m.num_objects, num_verbs=m.num_verbs, num_hoi=m.num_hoi)
    backbone = SharedResNetFPN(m.fpn_dim, m.trainable_backbone_stages)
    det = detector or (OracleDetectionBranch() if m.detector_mode == "oracle" else SharedFPNDetector(m.fpn_dim, m.num_objects, m.human_class_index))
    roi = MultiRoIExtractor(m.fpn_dim, m.roi_size)
    pose = IntegratedPoseModule(m.fpn_dim, m.num_joints, m.pose_dim, m.roi_size, m.pose_heads, m.pose_layers)
    reliability = HumanJointReasoningModule(m.pose_dim, m.joint_dim, m.num_joints)
    pair = HumanObjectPairModule(m.fpn_dim, m.pair_dim, m.num_objects)
    reasoning = InteractionReasoningModule(m.joint_dim, m.pair_dim, m.pair_dim // 2, m.attention_heads)
    verb = VerbHead(m.pair_dim, m.num_verbs)
    projector = HOIProjector(table)
    return ReliPoseHOIModel(backbone, det, roi, pose, reliability, pair, reasoning, verb, projector, TrainingPairBuilder(m.num_verbs), InferencePairBuilder(m.human_class_index))

