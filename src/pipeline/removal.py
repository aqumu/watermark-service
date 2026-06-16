"""RemovalStep: run the removal model on a fixed-aspect ROI."""

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.pipeline.context import ImageContext
from src.processing.image_utils import crop_removal_roi


class RemovalStep:
    """Batch-capable GPU step: runs blind removal on ROI crops."""

    def __init__(
        self,
        model: nn.Module,
        image_width: int,
        image_height: int,
        crop_aspect_ratio: float,
        crop_margin_ratio: float,
        crop_min_width_ratio: float,
        device: torch.device,
        amp: bool,
    ):
        self.model = model
        self.image_width = image_width
        self.image_height = image_height
        self.crop_aspect_ratio = crop_aspect_ratio
        self.crop_margin_ratio = crop_margin_ratio
        self.crop_min_width_ratio = crop_min_width_ratio
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def process_batch(self, contexts: list[ImageContext]) -> None:
        """Build ROI inputs, run the model, and store ROI predictions."""
        valid = [ctx for ctx in contexts if ctx.error is None and ctx.mask is not None]
        if not valid:
            return

        tensors = []
        valid_with_roi: list[ImageContext] = []

        for ctx in valid:
            mask_f = ctx.mask.astype(np.float32) / 255.0
            wm_r, _, _, roi, _ = crop_removal_roi(
                ctx.original_bgr,
                ctx.original_bgr,
                mask_f,
                self.image_width,
                self.image_height,
                crop_aspect_ratio=self.crop_aspect_ratio,
                margin_ratio=self.crop_margin_ratio,
                min_width_ratio=self.crop_min_width_ratio,
            )

            ctx.blend_mask = ctx.mask
            ctx.roi = None
            ctx.model_pred_bgr = None

            if not np.any(mask_f > 0):
                continue

            rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)))

            ctx.roi = roi
            valid_with_roi.append(ctx)

        if not tensors:
            return

        batch = torch.stack(tensors).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            delta = self.model(batch)

        pred = (batch[:, :3] - delta).clamp(-1, 1).cpu().numpy()
        for ctx, pred_roi in zip(valid_with_roi, pred):
            pred_u8 = ((pred_roi + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            ctx.model_pred_bgr = cv2.cvtColor(pred_u8.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
