"""BlendingStep: feathered ROI blend back to the original image."""

import cv2
import numpy as np

from src.pipeline.context import ImageContext


class BlendingStep:
    """CPU step: paste the ROI prediction back into the original image."""

    def __init__(self, feather_radius: int = 9, mask_expand: int = 0):
        self.feather_radius = feather_radius
        self.mask_expand = mask_expand

    def process_batch(self, contexts: list[ImageContext]) -> None:
        for ctx in contexts:
            if ctx.error is not None:
                continue

            if ctx.model_pred_bgr is None or ctx.roi is None or ctx.blend_mask is None:
                ctx.result_bgr = ctx.original_bgr.copy()
                continue

            ctx.result_bgr = self._blend_back(
                ctx.model_pred_bgr,
                ctx.original_bgr,
                ctx.blend_mask,
                ctx.roi,
                self.feather_radius,
                self.mask_expand,
            )

    @staticmethod
    def _blend_back(
        pred_bgr: np.ndarray,
        orig_wm: np.ndarray,
        orig_mask: np.ndarray,
        roi: dict,
        feather: int,
        mask_expand: int,
    ) -> np.ndarray:
        """Paste a ROI prediction back at original resolution."""
        out = orig_wm.copy()
        crop_w = int(roi["width"])
        crop_h = int(roi["height"])
        x0 = int(roi["x0"])
        y0 = int(roi["y0"])
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        orig_h, orig_w = orig_wm.shape[:2]
        ox0 = max(0, x0)
        oy0 = max(0, y0)
        ox1 = min(orig_w, x1)
        oy1 = min(orig_h, y1)
        if ox0 >= ox1 or oy0 >= oy1:
            return out

        pred_up = cv2.resize(pred_bgr, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        px0 = ox0 - x0
        py0 = oy0 - y0
        px1 = px0 + (ox1 - ox0)
        py1 = py0 + (oy1 - oy0)

        pred_crop = pred_up[py0:py1, px0:px1]
        orig_crop = orig_wm[oy0:oy1, ox0:ox1]

        scale = crop_w / pred_bgr.shape[1]
        if feather > 0 or mask_expand > 0:
            feather_scaled = max(1, round(feather * scale))
            mask_expand_scaled = round(mask_expand * scale)

            working_mask = orig_mask
            if mask_expand_scaled > 0:
                exp_k = mask_expand_scaled * 2 + 1
                working_mask = cv2.dilate(
                    working_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp_k, exp_k)),
                )
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
            mask_crop = working_mask[oy0:oy1, ox0:ox1]
            mask_alpha = (mask_crop / 255.0)[:, :, None]
        else:
            mask_crop = orig_mask[oy0:oy1, ox0:ox1]
            mask_alpha = (mask_crop > 127)[:, :, None].astype(np.float32)

        out[oy0:oy1, ox0:ox1] = (
            pred_crop * mask_alpha + orig_crop * (1.0 - mask_alpha)
        ).clip(0, 255).astype(np.uint8)
        return out
