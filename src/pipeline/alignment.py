"""AlignmentStep — align watermark template to the segmentation probability map."""

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import fftconvolve

from src.pipeline.context import ImageContext

logger = logging.getLogger(__name__)


def _render_template(template: np.ndarray, scale: float,
                     img_w: int, img_h: int, seg_w: int, seg_h: int) -> np.ndarray:
    tw_orig = max(1, int(round(img_w * scale)))
    th_orig = max(1, int(round(tw_orig * template.shape[0] / template.shape[1])))
    tw_seg = max(1, int(round(tw_orig * seg_w / img_w)))
    th_seg = max(1, int(round(th_orig * seg_h / img_h)))
    return cv2.resize(template, (tw_seg, th_seg), interpolation=cv2.INTER_AREA)


def _xcorr_peak(signal: np.ndarray, kernel: np.ndarray,
                subpixel: bool = True):
    """Cross-correlate signal with kernel, return (tx, ty) peak in signal coords."""
    corr = fftconvolve(signal, kernel[::-1, ::-1], mode="full")
    peak_idx = np.unravel_index(corr.argmax(), corr.shape)
    th, tw = kernel.shape[:2]
    ipy, ipx = int(peak_idx[0]), int(peak_idx[1])

    px, py = float(ipx), float(ipy)
    if subpixel and 0 < ipy < corr.shape[0] - 1 and 0 < ipx < corr.shape[1] - 1:
        denom_x = corr[ipy, ipx - 1] - 2 * corr[ipy, ipx] + corr[ipy, ipx + 1]
        denom_y = corr[ipy - 1, ipx] - 2 * corr[ipy, ipx] + corr[ipy + 1, ipx]
        px = ipx + float(np.clip(0.5 * (corr[ipy, ipx - 1] - corr[ipy, ipx + 1]) / (denom_x + 1e-12), -1, 1))
        py = ipy + float(np.clip(0.5 * (corr[ipy - 1, ipx] - corr[ipy + 1, ipx]) / (denom_y + 1e-12), -1, 1))

    return px - (tw - 1), py - (th - 1), float(corr[peak_idx])


def _subpixel_xcorr(prob_map: np.ndarray, template: np.ndarray,
                    img_w: int, img_h: int,
                    base_scale: float = 0.79,
                    scale_range: float = 0.05,
                    n_scales: int = 11):
    """Align template to prob_map^2 via scale search + subpixel xcorr peak."""
    confidence = prob_map ** 2
    seg_h, seg_w = prob_map.shape[:2]
    scales = np.linspace(base_scale * (1 - scale_range),
                         base_scale * (1 + scale_range), n_scales)

    best_score = -np.inf
    best_tx_seg = best_ty_seg = 0.0
    best_scale = base_scale

    for s in scales:
        t_scaled = _render_template(template, s, img_w, img_h, seg_w, seg_h)
        tx_seg, ty_seg, score = _xcorr_peak(confidence, t_scaled, subpixel=True)
        if score > best_score:
            best_score = score
            best_tx_seg, best_ty_seg, best_scale = tx_seg, ty_seg, s

    tx = best_tx_seg * img_w / seg_w
    ty = best_ty_seg * img_h / seg_h
    return tx, ty, best_scale


def _place_template(template: np.ndarray, tx: float, ty: float,
                    scale: float, img_w: int, img_h: int) -> np.ndarray:
    """Render the template as a binary uint8 mask at original image resolution."""
    tw = max(1, int(round(img_w * scale)))
    th = max(1, int(round(tw * template.shape[0] / template.shape[1])))
    t_bin = (cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    x0, y0 = int(round(tx)), int(round(ty))
    x1, y1 = min(x0 + tw, img_w), min(y0 + th, img_h)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    cx0, cy0 = max(0, x0), max(0, y0)
    if x1 > cx0 and y1 > cy0:
        canvas[cy0:y1, cx0:x1] = t_bin[sy0:sy0 + (y1 - cy0), sx0:sx0 + (x1 - cx0)]
    return canvas * 255


class AlignmentStep:
    """CPU step: aligns the watermark template to the seg prob map via subpixel xcorr.

    Replaces the raw seg mask with a precisely-placed binary template mask.
    Dilation is left to the removal step, matching training behaviour exactly.
    Falls back to the seg mask if no template is loaded or prob_map is missing.
    """

    def __init__(self, watermark_path: str,
                 base_scale: float = 0.79,
                 scale_range: float = 0.05,
                 n_scales: int = 11):
        self.template: np.ndarray | None = None
        self.base_scale = base_scale
        self.scale_range = scale_range
        self.n_scales = n_scales

        path = Path(watermark_path)
        if not path.exists():
            logger.warning("AlignmentStep: watermark not found at %s — will use raw seg mask", path)
            return

        wm_rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if wm_rgba is None or wm_rgba.shape[2] < 4:
            logger.warning("AlignmentStep: cannot read RGBA watermark from %s — will use raw seg mask", path)
            return

        self.template = wm_rgba[:, :, 3].astype(np.float32) / 255.0
        logger.info("AlignmentStep: loaded watermark template %dx%d from %s",
                    self.template.shape[1], self.template.shape[0], path)

    def process_batch(self, contexts: list[ImageContext]) -> None:
        if self.template is None:
            return  # no template — seg mask used as-is

        for ctx in contexts:
            if ctx.error is not None or ctx.prob_map is None:
                continue

            img_h, img_w = ctx.original_bgr.shape[:2]
            try:
                tx, ty, scale = _subpixel_xcorr(
                    ctx.prob_map, self.template, img_w, img_h,
                    base_scale=self.base_scale,
                    scale_range=self.scale_range,
                    n_scales=self.n_scales,
                )
                ctx.mask = _place_template(self.template, tx, ty, scale, img_w, img_h)
            except Exception as exc:
                logger.warning("AlignmentStep failed for %s: %s — keeping seg mask", ctx.image_id, exc)
