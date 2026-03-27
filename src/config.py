"""
Configuration management — YAML file + environment variable overrides.

Checkpoint auto-detection
-------------------------
Set seg_checkpoint / removal_checkpoint to "auto" (the default) and the
loader will scan ``models_dir`` for the newest file matching the prefix:

    model_seg_*.pth   →  segmentation checkpoint
    model_rem_*.pth   →  removal checkpoint

This lets you version checkpoints freely (model_seg_1.0.pth,
model_rem_2.1_finetune.pth, …) and the service always picks the latest
by filesystem modification time.

Real-ESRGAN upscaling (optional)
---------------------------------
Set upscale.enabled = true and place the desired weight file in models_dir.
Supported model names and their expected weight files:

    RealESRGAN_x4plus          → RealESRGAN_x4plus.pth          (4x, general)
    RealESRGAN_x2plus          → RealESRGAN_x2plus.pth          (2x, general)
    RealESRGAN_x4plus_anime_6B → RealESRGAN_x4plus_anime_6B.pth (4x, anime)
    realesr-animevideov3       → realesr-animevideov3.pth        (4x, anime video)
    realesr-general-x4v3       → realesr-general-x4v3.pth       (4x, general fast)

Set upscale.model_path to "auto" to detect ``{model_name}.pth`` in models_dir
automatically, or provide an explicit path.
"""

import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── auto-detection helper ─────────────────────────────────────────────────────

def _find_latest(directory: Path, prefix: str) -> str | None:
    """Return the path of the newest ``<prefix>*.pth`` file in *directory*."""
    matches = sorted(directory.glob(f"{prefix}*.pth"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return str(matches[-1])


def resolve_checkpoints(cfg: "ServiceConfig") -> "ServiceConfig":
    """Replace ``"auto"`` checkpoint paths with auto-detected files."""
    models_dir = Path(cfg.model.models_dir)

    if cfg.model.seg_checkpoint == "auto":
        found = _find_latest(models_dir, "model_seg_")
        if found is None:
            raise FileNotFoundError(
                f"No model_seg_*.pth found in {models_dir}. "
                "Place a checkpoint there or set model.seg_checkpoint explicitly."
            )
        logger.info("Auto-detected seg checkpoint: %s", found)
        cfg.model.seg_checkpoint = found

    if cfg.model.removal_checkpoint == "auto":
        found = _find_latest(models_dir, "model_rem_")
        if found is None:
            raise FileNotFoundError(
                f"No model_rem_*.pth found in {models_dir}. "
                "Place a checkpoint there or set model.removal_checkpoint explicitly."
            )
        logger.info("Auto-detected removal checkpoint: %s", found)
        cfg.model.removal_checkpoint = found

    if cfg.upscale.enabled and cfg.upscale.model_path == "auto":
        candidate = models_dir / f"{cfg.upscale.model_name}.pth"
        if candidate.exists():
            cfg.upscale.model_path = str(candidate)
            logger.info("Auto-detected ESRGAN weights: %s", cfg.upscale.model_path)
        else:
            logger.warning(
                "upscale.enabled=true but %s not found in %s — "
                "upscaling will be skipped unless model_path is set explicitly.",
                f"{cfg.upscale.model_name}.pth", models_dir,
            )
            cfg.upscale.model_path = ""   # sentinel: model unavailable

    return cfg


# ── config models ─────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    models_dir: str = "./models"
    seg_checkpoint: str = "auto"
    removal_checkpoint: str = "auto"
    seg_image_size: int = 256
    removal_image_size: int = 256
    seg_encoder: str = "efficientnet-b0"
    removal_base_channels: int = 32
    removal_depth: int = 4


class UpscaleConfig(BaseModel):
    enabled: bool = False
    model_name: str = "RealESRGAN_x4plus"   # selects architecture + weight file name
    model_path: str = "auto"                # "auto" → models_dir/{model_name}.pth
    tile: int = 512                         # tile size (0 = no tiling)
    tile_pad: int = 10
    half: bool = True                       # fp16 inference on CUDA
    resolution_threshold: int = 720        # upscale images whose longest side is ≤ this (0 = disabled)


class InferenceConfig(BaseModel):
    device: str = "auto"
    mask_threshold: float = 0.5
    mask_dilate_ksize: int = 5
    mask_clamp_ksize: int = 3
    feather_radius: int = 9
    mask_expand: int = 0
    amp: bool = True


class BatchConfig(BaseModel):
    max_batch_size: int = 8
    io_workers: int = 4
    max_concurrent_jobs: int = 4


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class ServiceConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    upscale: UpscaleConfig = UpscaleConfig()
    batch: BatchConfig = BatchConfig()
    server: ServerConfig = ServerConfig()


def load_config(path: str | None = None) -> ServiceConfig:
    """Load config from YAML file, with env var override for path."""
    if path is None:
        path = os.environ.get("WM_CONFIG_PATH", "config/default.yaml")

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = ServiceConfig(**raw)
    else:
        cfg = ServiceConfig()

    return resolve_checkpoints(cfg)
