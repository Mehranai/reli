"""Small public data structures for the compact ReliPose-HOI codebase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


def check_boxes(boxes: Tensor, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [N, 4]")
    if not torch.is_floating_point(boxes):
        raise TypeError(f"{name} must be floating point")
    if boxes.numel() and (
        torch.any(boxes[:, 2] <= boxes[:, 0])
        or torch.any(boxes[:, 3] <= boxes[:, 1])
        or not torch.isfinite(boxes).all()
    ):
        raise ValueError(f"{name} must contain finite positive xyxy boxes")


def empty_like_2d(reference: Tensor, cols: int) -> Tensor:
    return reference.new_empty((0, cols))


@dataclass
class DetectionOutput:
    """Detector output. Lists have length B; boxes are image xyxy pixels."""

    human_boxes: list[Tensor]
    human_scores: list[Tensor]
    object_boxes: list[Tensor]
    object_scores: list[Tensor]
    object_labels: list[Tensor]
    instance_ids: list[Tensor] | None = None
    losses: dict[str, Tensor] | None = None

    def validate(self) -> "DetectionOutput":
        sizes = {
            len(self.human_boxes),
            len(self.human_scores),
            len(self.object_boxes),
            len(self.object_scores),
            len(self.object_labels),
        }
        if self.instance_ids is not None:
            sizes.add(len(self.instance_ids))
        if len(sizes) != 1:
            raise ValueError("detection lists must have the same batch length")
        for i, boxes in enumerate(self.human_boxes):
            check_boxes(boxes, f"human_boxes[{i}]")
            if self.human_scores[i].shape != (boxes.shape[0],):
                raise ValueError("human score length mismatch")
        for i, boxes in enumerate(self.object_boxes):
            check_boxes(boxes, f"object_boxes[{i}]")
            if self.object_scores[i].shape != (boxes.shape[0],):
                raise ValueError("object score length mismatch")
            if self.object_labels[i].shape != (boxes.shape[0],):
                raise ValueError("object label length mismatch")
            if self.object_labels[i].dtype != torch.long:
                raise TypeError("object labels must be long")
        return self


@dataclass
class RoIOutput:
    """Flattened RoI features and pair index metadata.

    human/object/union features are [N_H/N_O/N_P, D, S, S].
    Pair indices are flattened global indices in image-major order.
    """

    human_roi_features: Tensor
    object_roi_features: Tensor
    union_roi_features: Tensor
    human_offsets: Tensor
    object_offsets: Tensor
    pair_offsets: Tensor
    pair_human_indices_global: Tensor
    pair_object_indices_global: Tensor
    pair_to_image: Tensor
    union_boxes: list[Tensor]


@dataclass
class IntegratedPoseOutput:
    """Internal pose output. Coordinates are RoI-normalized unless named image."""

    joint_tokens_initial: Tensor
    joint_coordinates_roi_coarse: Tensor
    joint_confidence_logits_coarse: Tensor
    joint_confidence_coarse: Tensor
    joint_uncertainty_coarse: Tensor
    coarse_pose_quality: Tensor  # rho_hi
    joint_tokens: Tensor
    joint_coordinates_roi: Tensor
    joint_confidence_logits: Tensor
    joint_confidence: Tensor
    joint_uncertainty: Tensor
    local_joint_features: Tensor
    refinement_gate: Tensor
    refinement_mask: Tensor

    @property
    def rho_hi(self) -> Tensor:
        return self.coarse_pose_quality


@dataclass
class HumanJointReasoningOutput:
    """Human-level outputs computed before pair expansion."""

    anatomical_features: Tensor
    joint_representations: Tensor
    refinement_displacement: Tensor
    joint_validity_prior: Tensor
    joint_valid_mask: Tensor
    joint_reliability_logits: Tensor
    joint_reliability: Tensor  # r_hi

    @property
    def r_hi(self) -> Tensor:
        return self.joint_reliability


@dataclass
class PairBuildOutput:
    """Local pair indices and optional training targets."""

    pair_human_indices: list[Tensor]
    pair_object_indices: list[Tensor]
    pair_offsets: Tensor
    verb_targets: Tensor | None = None
    positive_pair_mask: Tensor | None = None
    negative_pair_mask: Tensor | None = None


@dataclass
class HumanObjectPairOutput:
    """Visual/geometric pair representation q_ho and inspectable pieces."""

    human_visual_features: Tensor
    object_visual_features: Tensor
    union_visual_features: Tensor
    pair_geometry: Tensor
    pair_geometry_features: Tensor
    object_semantic_features: Tensor
    pair_features: Tensor  # q_ho
    pair_human_indices_global: Tensor
    pair_object_indices_global: Tensor
    pair_to_image: Tensor

    @property
    def q_ho(self) -> Tensor:
        return self.pair_features


@dataclass
class InteractionReasoningOutput:
    """Pair-dependent pose reasoning outputs."""

    paired_joint_representations: Tensor
    paired_joint_reliability: Tensor
    paired_joint_valid_mask: Tensor
    joint_object_geometry: Tensor
    joint_object_geometry_features: Tensor
    interaction_relevance_logits: Tensor
    interaction_relevance: Tensor  # a_hoi
    reliability_relevance_gate: Tensor  # g_hoi = r_hi * a_hoi
    attention_weights: Tensor
    pose_aware_pair_features: Tensor
    pair_reliability_summary: Tensor
    fusion_gate: Tensor
    fused_pair_features: Tensor
    pair_has_valid_joint: Tensor

    @property
    def r_hi(self) -> Tensor:
        return self.paired_joint_reliability

    @property
    def a_hoi(self) -> Tensor:
        return self.interaction_relevance

    @property
    def g_hoi(self) -> Tensor:
        return self.reliability_relevance_gate


@dataclass
class HICOCorrespondenceTable:
    """Mapping used to project 117 verb probabilities to 600 HOI scores."""

    hoi_to_object: Tensor
    hoi_to_verb: Tensor
    num_objects: int = 80
    num_verbs: int = 117

    @classmethod
    def from_records(
        cls,
        records: list[Mapping[str, int] | tuple[int, int, int]],
        *,
        num_objects: int = 80,
        num_verbs: int = 117,
        num_hoi: int = 600,
        index_base: int = 0,
    ) -> "HICOCorrespondenceTable":
        hoi_to_object = torch.full((num_hoi,), -1, dtype=torch.long)
        hoi_to_verb = torch.full((num_hoi,), -1, dtype=torch.long)
        seen: set[int] = set()
        for rec in records:
            if isinstance(rec, Mapping):
                h = int(rec.get("hoi_index", rec.get("hoi"))) - index_base
                o = int(rec.get("object_index", rec.get("object"))) - index_base
                v = int(rec.get("verb_index", rec.get("verb"))) - index_base
            else:
                h, o, v = (int(x) - index_base for x in rec)
            if h in seen:
                raise ValueError(f"duplicate HOI index {h}")
            if not 0 <= h < num_hoi or not 0 <= o < num_objects or not 0 <= v < num_verbs:
                raise ValueError("correspondence index out of range")
            seen.add(h)
            hoi_to_object[h] = o
            hoi_to_verb[h] = v
        if len(seen) != num_hoi:
            raise ValueError("correspondence must define every HOI index")
        return cls(hoi_to_object, hoi_to_verb, num_objects, num_verbs)

    @classmethod
    def synthetic(cls, *, num_objects: int = 80, num_verbs: int = 117, num_hoi: int = 600) -> "HICOCorrespondenceTable":
        records = [
            {"hoi_index": i, "object_index": i % num_objects, "verb_index": i % num_verbs}
            for i in range(num_hoi)
        ]
        return cls.from_records(records, num_objects=num_objects, num_verbs=num_verbs, num_hoi=num_hoi)

    @property
    def num_hoi(self) -> int:
        return int(self.hoi_to_object.numel())

    @property
    def valid_verb_by_object(self) -> Tensor:
        mask = torch.zeros((self.num_objects, self.num_verbs), dtype=torch.bool)
        mask[self.hoi_to_object, self.hoi_to_verb] = True
        return mask

    @property
    def object_verb_to_hoi(self) -> Tensor:
        table = torch.full((self.num_objects, self.num_verbs), -1, dtype=torch.long)
        table[self.hoi_to_object, self.hoi_to_verb] = torch.arange(self.num_hoi)
        return table


@dataclass
class ReliPoseHOIOutput:
    """Complete model output with all scientific variables inspectable."""

    detections: DetectionOutput
    roi: RoIOutput
    pose: IntegratedPoseOutput
    reliability: HumanJointReasoningOutput
    pairs: HumanObjectPairOutput
    reasoning: InteractionReasoningOutput
    verb_logits: Tensor
    verb_probabilities: Tensor
    valid_verb_mask: Tensor
    hoi_scores: Tensor
    pair_object_labels: Tensor
    training_pairs: PairBuildOutput | None = None

    @property
    def rho_hi(self) -> Tensor:
        return self.pose.rho_hi

    @property
    def r_hi(self) -> Tensor:
        return self.reliability.r_hi

    @property
    def a_hoi(self) -> Tensor:
        return self.reasoning.a_hoi

    @property
    def g_hoi(self) -> Tensor:
        return self.reasoning.g_hoi


@dataclass
class COCOKeypointsTarget:
    image_id: int
    person_boxes: Tensor
    keypoints_image: Tensor
    keypoint_visibility: Tensor
    original_size: tuple[int, int]

    @property
    def keypoint_label_mask(self) -> Tensor:
        return self.keypoint_visibility > 0

    @property
    def keypoint_visible_mask(self) -> Tensor:
        return self.keypoint_visibility == 2


@dataclass
class PoseTarget:
    joint_coordinates_roi_target: Tensor
    joint_label_mask: Tensor
    joint_visible_mask: Tensor


@dataclass
class HICOTarget:
    image_id: int
    human_boxes: Tensor
    object_boxes: Tensor
    object_labels: Tensor
    pair_human_indices: Tensor
    pair_object_indices: Tensor
    verb_targets: Tensor
    hoi_targets: Tensor
    original_size: tuple[int, int]

    def as_model_target(self) -> dict[str, Tensor]:
        ones_h = torch.ones((self.human_boxes.shape[0],), dtype=torch.float32)
        ones_o = torch.ones((self.object_boxes.shape[0],), dtype=torch.float32)
        return {
            "human_boxes": self.human_boxes,
            "human_scores": ones_h,
            "object_boxes": self.object_boxes,
            "object_scores": ones_o,
            "object_labels": self.object_labels,
            "pair_human_indices": self.pair_human_indices,
            "pair_object_indices": self.pair_object_indices,
            "verb_targets": self.verb_targets,
        }


@dataclass
class Batch:
    images: Tensor
    targets: list[Any]
    image_sizes: list[tuple[int, int]]


@dataclass
class PoseCorruptionOutput:
    corrupted_pose: IntegratedPoseOutput
    corruption_mask: Tensor
    corruption_severity: Tensor
    synthetic_reliability_target: Tensor
    reliability_supervision_mask: Tensor


@dataclass
class RobustOutput:
    clean: ReliPoseHOIOutput
    corrupted_pose: PoseCorruptionOutput
    corrupted_reliability: HumanJointReasoningOutput
    corrupted_reasoning: InteractionReasoningOutput
    corrupted_verb_logits: Tensor
    corrupted_hoi_scores: Tensor


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
