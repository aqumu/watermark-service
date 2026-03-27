"""RemovalStep — run the MaskedUNet to remove watermarks."""

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.pipeline.context import ImageContext
from src.processing.image_utils import compute_gradient, dilate_mask

logger = logging.getLogger(__name__)


class RemovalStep:
    """Batch-capable GPU step: removes watermarks using predicted masks."""

    def __init__(self, model: nn.Module, image_size: int,
                 dilate_ksize: int, clamp_dilate_ksize: int,
                 device: torch.device, amp: bool):
        self.model = model
        self.image_size = image_size
        self.dilate_ksize = dilate_ksize
        self.clamp_dilate_ksize = clamp_dilate_ksize
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def process_batch(self, contexts: list[ImageContext]) -> None:
        """Build 5-channel inputs, run model, store predictions."""
        valid = [ctx for ctx in contexts if ctx.error is None and ctx.mask is not None]
        if not valid:
            return

        tensors = []
        clamp_masks = []
        for ctx in valid:
            sz = self.image_size

            # Resize watermarked image and mask to model size
            wm_r = cv2.resize(ctx.original_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
            mask_r = cv2.resize(ctx.mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

            # Binarize + dilate mask
            mask_binary = (mask_r > 127).astype(np.float32)
            mask_dilated = dilate_mask(mask_binary, ksize=self.dilate_ksize)

            # Smaller clamping mask — covers full watermark but less boundary bleed
            mask_clamp = dilate_mask(mask_binary, ksize=self.clamp_dilate_ksize)
            clamp_masks.append(torch.from_numpy(mask_clamp).unsqueeze(0))  # 1xHxW

            # Normalize RGB to [-1, 1]
            rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))        # 3xHxW

            # Mask channel
            mask_t = torch.from_numpy(mask_dilated).unsqueeze(0)     # 1xHxW

            # Gradient channel
            grad_t = compute_gradient(wm_r)                          # 1xHxW

            # Concatenate 5-channel input
            inp = torch.cat([rgb_t, mask_t, grad_t], dim=0)          # 5xHxW
            tensors.append(inp)

        batch = torch.stack(tensors).to(self.device)                  # Bx5xHxW
        clamp_mask = torch.stack(clamp_masks).to(self.device)         # Bx1xHxW

        with torch.autocast(self.device.type, enabled=self.amp):
            delta = self.model(batch)                                 # Bx3xHxW

        # Apply delta only within the (smaller) clamping mask
        pred = (batch[:, :3] - delta * clamp_mask).clamp(-1, 1)

        # Convert back to uint8 BGR
        pred_np = pred.cpu().numpy()                                  # Bx3xHxW
        for ctx, p in zip(valid, pred_np):
            p_uint8 = ((p + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            ctx.model_pred_bgr = cv2.cvtColor(p_uint8.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
