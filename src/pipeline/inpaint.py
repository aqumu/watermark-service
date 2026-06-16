"""
InpaintStep — fixed-rectangle inpainting for the "new" processing mode.

Supports two methods:
  - "telea": OpenCV Telea inpainting (CPU, fast, no model)
  - "lama":  LaMa FFCResNetGenerator (GPU, better quality)
"""

import logging

import cv2
import numpy as np
import torch

from src.models.lama import build_lama_model
from src.pipeline.context import ImageContext

logger = logging.getLogger(__name__)


class InpaintStep:
    """CPU/GPU step: inpaints a fixed rectangle at the bottom-right corner."""

    def __init__(
        self,
        method: str = "telea",
        inpaint_width: int = 114,
        inpaint_height: int = 48,
        inpaint_radius: int = 3,
        inpaint_size: int = 256,
        checkpoint_path: str = "",
        device: torch.device | None = None,
        amp: bool = False,
    ):
        self.method = method
        self.inpaint_width = inpaint_width
        self.inpaint_height = inpaint_height
        self.inpaint_radius = inpaint_radius
        self.inpaint_size = inpaint_size
        self.device = device or torch.device("cpu")
        self.amp = amp and self.device.type == "cuda"

        self.lama_model: torch.nn.Module | None = None
        if method == "lama":
            if not checkpoint_path:
                raise RuntimeError("InpaintStep: method='lama' requires checkpoint_path")
            self.lama_model = build_lama_model(checkpoint_path, self.device)
            self.lama_model = self.lama_model.to(self.device)
            logger.info(
                "InpaintStep (LaMa): rect=%dx%d, device=%s, amp=%s",
                inpaint_width, inpaint_height, self.device, self.amp,
            )
        else:
            logger.info(
                "InpaintStep (Telea): rect=%dx%d, radius=%d",
                inpaint_width, inpaint_height, inpaint_radius,
            )

    def process_batch(self, contexts: list[ImageContext]) -> None:
        for ctx in contexts:
            if ctx.error is not None:
                continue

            h, w = ctx.original_bgr.shape[:2]
            if h < self.inpaint_height or w < self.inpaint_width:
                ctx.error = (
                    f"Image too small ({w}x{h}) for "
                    f"{self.inpaint_width}x{self.inpaint_height} inpaint region"
                )
                continue

            ctx.result_bgr = self._inpaint(ctx.original_bgr)

    def _inpaint(self, bgr: np.ndarray) -> np.ndarray:
        if self.method == "lama":
            return self._lama_inpaint(bgr)
        return self._telea_inpaint(bgr)

    # ── Telea ─────────────────────────────────────────────────────────────────

    def _telea_inpaint(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h - self.inpaint_height :, w - self.inpaint_width :] = 255
        return cv2.inpaint(bgr, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

    # ── LaMa ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _lama_inpaint(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        crop_size = self.inpaint_size

        # Mask is always at the bottom-right corner
        mw, mh = self.inpaint_width, self.inpaint_height

        y0 = max(0, h - crop_size)
        x0 = max(0, w - crop_size)
        crop = bgr[y0:y0 + crop_size, x0:x0 + crop_size].copy()

        # Pad to crop_size if image is smaller than the crop
        ch, cw = crop.shape[:2]
        if ch < crop_size or cw < crop_size:
            crop = cv2.copyMakeBorder(crop, 0, crop_size - ch, 0, crop_size - cw, cv2.BORDER_REPLICATE)

        # Build mask in crop coordinates
        mask = np.zeros((crop_size, crop_size), dtype=np.float32)
        my0 = h - mh - y0
        mx0 = w - mw - x0
        if 0 <= my0 < crop_size and 0 <= mx0 < crop_size:
            mask[my0:my0 + mh, mx0:mx0 + mw] = 1.0

        # Normalize crop to [0, 1] — HWC → CHW
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_chw = rgb.transpose(2, 0, 1)

        # Build 4-channel input
        masked_rgb = rgb_chw * (1.0 - mask[None, :, :])
        input_tensor = np.concatenate([masked_rgb, mask[None, :, :]], axis=0)

        inp = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)

        with torch.autocast(self.device.type, enabled=self.amp):
            output_t = self.lama_model(inp)

        out_np = output_t.squeeze(0).cpu().numpy().astype(np.float32)

        # Composite
        mask_3ch = mask[None, :, :]
        blended = out_np * mask_3ch + rgb_chw * (1.0 - mask_3ch)
        blended_u8 = (blended * 255.0).clip(0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(blended_u8.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        # Paste back
        out = bgr.copy()
        out[y0:y0 + crop_size, x0:x0 + crop_size] = result_bgr[:ch, :cw]
        return out
