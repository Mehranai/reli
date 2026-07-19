"""Compact dataclass configuration for ReliPose-HOI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin


def _coerce(cls: type, data: Mapping[str, Any]):
    kwargs = {}
    for f in fields(cls):
        value = data.get(f.name, f.default)
        if value is f.default and getattr(f, "default_factory", None):  # type: ignore[attr-defined]
            try:
                value = f.default_factory()  # type: ignore[misc,attr-defined]
            except TypeError:
                pass
        target = f.type
        if isinstance(value, Mapping) and isinstance(target, type) and is_dataclass(target):
            value = _coerce(target, value)
        elif get_origin(target) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class ModelConfig:
    backbone_name: str = "tiny_resnet_fpn"
    fpn_dim: int = 64
    roi_size: int = 7
    num_joints: int = 17
    pose_dim: int = 64
    joint_dim: int = 64
    pair_dim: int = 64
    num_verbs: int = 117
    num_objects: int = 80
    num_hoi: int = 600
    pose_heads: int = 4
    pose_layers: int = 1
    attention_heads: int = 4
    pretrained_backbone: bool = False
    trainable_backbone_stages: int = 3
    detector_mode: str = "oracle"
    human_class_index: int = 0
    allow_test_fallback_ops: bool = False


@dataclass
class DataConfig:
    image_size: tuple[int, int] = (128, 128)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_prob: float = 0.0
    coco_image_root: str | None = None
    coco_annotation_file: str | None = None
    hico_train_image_root: str | None = None
    hico_test_image_root: str | None = None
    hico_train_annotation_file: str | None = None
    hico_test_annotation_file: str | None = None
    hico_index_base: int = 0
    batch_size: int = 2
    num_workers: int = 0


@dataclass
class LossConfig:
    pose_weight: float = 1.0
    verb_weight: float = 1.0
    reliability_weight: float = 0.5
    consistency_weight: float = 0.2
    sparse_weight: float = 0.05
    sparse_target: float = 0.35
    confidence_weight: float = 0.25
    uncertainty_eps: float = 1e-4
    focal_gamma: float = 0.0


@dataclass
class CorruptionConfig:
    enabled: bool = True
    joint_probability: float = 0.35
    coordinate_jitter_std: float = 0.08
    dropout_probability: float = 0.15
    swap_probability: float = 0.10
    token_noise_std: float = 0.05
    maximum_margin: float = 0.25
    reliability_alpha: float = 3.0


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 5
    total_steps: int = 100
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 1
    seed: int = 7
    output_dir: str = "outputs"
    checkpoint: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Config":
        return cls(
            model=_coerce(ModelConfig, data.get("model", {})),
            data=_coerce(DataConfig, data.get("data", {})),
            losses=_coerce(LossConfig, data.get("losses", {})),
            corruption=_coerce(CorruptionConfig, data.get("corruption", {})),
            optim=_coerce(OptimConfig, data.get("optim", {})),
            train=_coerce(TrainConfig, data.get("train", {})),
        )


def load_config(path: str | Path) -> Config:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install PyYAML or use JSON config files") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise TypeError("config root must be a mapping")
    return Config.from_mapping(data)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj

