"""Unified checkpoint loading for both models."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load a training checkpoint into the model and set to eval mode."""
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    logger.info("Checkpoint loaded (epoch %s)", ckpt.get("epoch", "?"))
    return model
