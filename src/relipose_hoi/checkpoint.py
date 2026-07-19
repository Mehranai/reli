"""Single checkpoint format for research training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from relipose_hoi.config import to_dict


def save_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None, epoch: int = 0, global_step: int = 0, config: Any = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": to_dict(config) if config is not None else None,
        "python_random": random.getstate(),
        "torch_random": torch.get_rng_state(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    return state

