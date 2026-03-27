"""SegmentationStep — predict binary watermark masks via the seg model."""

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.pipeline.context import ImageContext

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegmentationStep:
    """Batch-capable GPU step: predicts binary masks for each image."""

    def __init__(self, model: nn.Module, image_size: int,
                 threshold: float, device: torch.device, amp: bool):
        self.model = model
        self.image_size = image_size
        self.threshold = threshold
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def process_batch(self, contexts: list[ImageContext]) -> None:
        """Run segmentation on all non-errored contexts in one forward pass."""
        valid = [ctx for ctx in contexts if ctx.error is None]
        if not valid:
            return

        # Preprocess: resize + ImageNet normalize
        tensors = []
        for ctx in valid:
            resized = cv2.resize(ctx.original_bgr, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _MEAN) / _STD
            tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)))

        batch = torch.stack(tensors).to(self.device)

        with torch.autocast(self.device.type, enabled=self.amp):
            logits = self.model(batch)                         # Bx1xHxW
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy() # BxHxW

        # Post-process: threshold + resize to original resolution
        for ctx, prob in zip(valid, probs):
            binary = (prob > self.threshold).astype(np.uint8) * 255
            w, h = ctx.original_size
            ctx.mask = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
