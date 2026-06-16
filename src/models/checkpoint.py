"""Unified checkpoint loading for both models."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load a training checkpoint into the model and set to eval mode.

    Epoch checkpoints store live weights under ``"model"`` and EMA shadow
    weights under ``"ema.shadow"``.  ``best.pth`` stores EMA weights directly
    under ``"model"`` (no separate ``"ema"`` key).  EMA weights are always
    preferred because training validation metrics and dashboard previews are
    computed with EMA — using live weights at inference would reproduce a
    quality gap relative to what was shown during training.
    """
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=True)

    ema_block = ckpt.get("ema")
    if isinstance(ema_block, dict) and "shadow" in ema_block:
        weights = {k: v.to(device) for k, v in ema_block["shadow"].items()}
        logger.info("Checkpoint loaded (epoch %s) — using EMA weights", ckpt.get("epoch", "?"))
    else:
        weights = ckpt["model"]
        logger.info("Checkpoint loaded (epoch %s)", ckpt.get("epoch", "?"))

    model.load_state_dict(weights, strict=False)
    model.to(device).eval()
    return model
