"""BlendingStep — feathered blend of model prediction back to original resolution."""

import cv2
import numpy as np

from src.pipeline.context import ImageContext


class BlendingStep:
    """CPU step (per-image, different resolutions): blend pred into original."""

    def __init__(self, feather_radius: int = 9, mask_expand: int = 0):
        self.feather_radius = feather_radius
        self.mask_expand = mask_expand

    def process_batch(self, contexts: list[ImageContext]) -> None:
        for ctx in contexts:
            if ctx.error is not None or ctx.mask is None:
                continue

            pred_bgr = ctx.model_pred_bgr
            if pred_bgr is None:
                continue

            ctx.result_bgr = self._blend_back(
                pred_bgr, ctx.original_bgr, ctx.mask, ctx.original_size,
                self.feather_radius, self.mask_expand,
            )

    @staticmethod
    def _blend_back(pred_bgr: np.ndarray, orig_wm: np.ndarray,
                    orig_mask: np.ndarray, orig_size: tuple[int, int],
                    feather: int, mask_expand: int) -> np.ndarray:
        """
        Paste the model prediction back at original resolution.

        Matches training/infer.py blend_back() logic exactly.
        """
        pred_up = cv2.resize(pred_bgr, orig_size, interpolation=cv2.INTER_CUBIC)

        if feather > 0 or mask_expand > 0:
            scale = orig_size[0] / pred_bgr.shape[1]
            feather_scaled = max(1, round(feather * scale))
            mask_expand_scaled = round(mask_expand * scale)

            working_mask = orig_mask

            # First pass: expand coverage to fully capture approximate masks
            if mask_expand_scaled > 0:
                exp_k = mask_expand_scaled * 2 + 1
                working_mask = cv2.dilate(
                    working_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp_k, exp_k)),
                )

            # Second pass: feather blend zone into clean background
            if feather_scaled > 0:
                dil_k = feather_scaled * 2 + 1
                working_mask = cv2.dilate(
                    working_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k)),
                )
                working_mask = cv2.GaussianBlur(
                    working_mask.astype(np.float32),
                    (dil_k, dil_k),
                    feather_scaled / 2,
                )

            m = (working_mask / 255.0)[:, :, None]
        else:
            m = (orig_mask > 127)[:, :, None].astype(np.float32)

        out = (pred_up * m + orig_wm * (1 - m)).clip(0, 255).astype(np.uint8)
        return out
