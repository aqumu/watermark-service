"""UpscaleStep — Real-ESRGAN pre-upscaling for low-resolution inputs.

Automatically upscales images whose longest side is below *resolution_threshold*
before they enter the segmentation and removal models. High-resolution images are
passed through unchanged. No external packages required — only PyTorch.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.context import ImageContext

logger = logging.getLogger(__name__)


# ── RRDBNet architecture ──────────────────────────────────────────────────────

class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class _RRDBNet(nn.Module):
    """Generator for RealESRGAN_x4plus and RealESRGAN_x2plus.

    For scale=2 the input is pixel-unshuffled (spatial /2, channels ×4) before
    conv_first, then upsampled twice (×4 total) for a net ×2 output — matching
    the original basicsr architecture exactly.
    """

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_block: int = 23,
                 num_grow_ch: int = 32, scale: int = 4) -> None:
        super().__init__()
        self.scale = scale
        first_in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
        self.conv_first = nn.Conv2d(first_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ── architecture registry ─────────────────────────────────────────────────────

_ARCH_SPECS: dict[str, dict] = {
    "RealESRGAN_x4plus": dict(
        cls=_RRDBNet,
        kwargs=dict(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4),
        scale=4,
    ),
    "RealESRGAN_x2plus": dict(
        cls=_RRDBNet,
        kwargs=dict(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2),
        scale=2,
    ),
    "RealESRGAN_x4plus_anime_6B": dict(
        cls=_RRDBNet,
        kwargs=dict(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4),
        scale=4,
    ),
}


# ── step ──────────────────────────────────────────────────────────────────────

class UpscaleStep:
    """Pre-upscales low-resolution images before seg/removal.

    Images whose longest side is <= *resolution_threshold* are upscaled in-place
    (ctx.original_bgr is replaced). Higher-resolution images are skipped.
    When no model is loaded the step is always a no-op.
    """

    def __init__(
        self,
        *,
        model_name: str = "RealESRGAN_x2plus",
        model_path: str = "",
        device: torch.device | None = None,
        tile: int = 512,
        tile_pad: int = 10,
        half: bool = True,
        resolution_threshold: int = 720,
    ) -> None:
        self.model: _RRDBNet | None = None
        self.device = device or torch.device("cpu")
        self.scale = 2
        self.tile = tile
        self.tile_pad = tile_pad
        self.half = half and self.device.type == "cuda"
        self.resolution_threshold = resolution_threshold

        if not model_path:
            logger.info("UpscaleStep: no model path provided — disabled")
            return

        spec = _ARCH_SPECS.get(model_name)
        if spec is None:
            logger.warning(
                "UpscaleStep: unknown model_name '%s'. Supported: %s — disabled",
                model_name, list(_ARCH_SPECS),
            )
            return

        try:
            model = spec["cls"](**spec["kwargs"])
            self.scale = spec["scale"]

            loadnet = torch.load(model_path, map_location="cpu", weights_only=True)
            if "params_ema" in loadnet:
                model.load_state_dict(loadnet["params_ema"])
            elif "params" in loadnet:
                model.load_state_dict(loadnet["params"])
            else:
                model.load_state_dict(loadnet)

            model.eval()
            if self.half:
                model = model.half()
            self.model = model.to(self.device)

            logger.info(
                "UpscaleStep: loaded %s (scale=%dx, half=%s, threshold=%dpx) from %s",
                model_name, self.scale, self.half, resolution_threshold, model_path,
            )
        except Exception as exc:
            logger.warning(
                "UpscaleStep: failed to load %s — disabled. Error: %s",
                model_name, exc,
            )
            self.model = None

    # ── public interface ──────────────────────────────────────────────────────

    def process_batch(self, contexts: list[ImageContext]) -> None:
        """Upscale original_bgr in-place for images below the resolution threshold."""
        if self.model is None:
            return
        for ctx in contexts:
            if ctx.error is not None:
                continue
            h, w = ctx.original_bgr.shape[:2]
            if max(h, w) <= self.resolution_threshold:
                try:
                    ctx.original_bgr = self._enhance(ctx.original_bgr)
                    ctx.original_size = (ctx.original_bgr.shape[1], ctx.original_bgr.shape[0])
                    logger.debug("Upscaled %s: %dx%d → %dx%d",
                                 ctx.image_id, w, h,
                                 ctx.original_size[0], ctx.original_size[1])
                except Exception as exc:
                    logger.warning("Upscaling failed for %s: %s — using original",
                                   ctx.image_id, exc)

    # ── internals ─────────────────────────────────────────────────────────────

    def _enhance(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        if self.half:
            img_t = img_t.half()
        img_t = img_t.to(self.device)

        with torch.inference_mode():
            out_t = self._tile_inference(img_t) if self.tile > 0 else self.model(img_t)

        out = out_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        return cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _tile_inference(self, img_t: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img_t.shape
        pad = self.tile_pad
        tile = self.tile
        scale = self.scale

        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)
        output = torch.zeros(b, c, h * scale, w * scale,
                             dtype=img_t.dtype, device=img_t.device)

        for iy in range(tiles_y):
            for ix in range(tiles_x):
                x0 = max(ix * tile - pad, 0)
                y0 = max(iy * tile - pad, 0)
                x1 = min((ix + 1) * tile + pad, w)
                y1 = min((iy + 1) * tile + pad, h)

                tile_out = self.model(img_t[:, :, y0:y1, x0:x1])

                ox0 = (ix * tile - x0) * scale
                oy0 = (iy * tile - y0) * scale
                out_x0 = ix * tile * scale
                out_y0 = iy * tile * scale
                out_x1 = min((ix + 1) * tile * scale, w * scale)
                out_y1 = min((iy + 1) * tile * scale, h * scale)

                output[:, :, out_y0:out_y1, out_x0:out_x1] = \
                    tile_out[:, :, oy0:oy0 + (out_y1 - out_y0), ox0:ox0 + (out_x1 - out_x0)]

        return output
