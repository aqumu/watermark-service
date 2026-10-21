# Vendored from watermark-removal/wm_shared/marks/localize.py (commit 5710cb5). Keep in sync.
"""
Locate a known mark inside an inference window.

Correlates the window's detail layer with the detail layer of the mark rendered
for the listing (normalised cross-correlation, either sign: the mark brightens
dark backgrounds and darkens near-white ones), then refines the peak to sub-pixel
precision with a parabola. The text style is not known in advance, so every
style is tried and the best match wins. Training runs this same function on
synthetic windows, so the model learns this localiser's actual error distribution.
"""
from __future__ import annotations

import cv2
import numpy as np

from .profile import MarkProfile, shift_image

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
HP_SIGMA = 6.0


def detail(gray: np.ndarray, sigma: float = HP_SIGMA) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def _parabola(m1: float, c: float, p1: float) -> float:
    den = m1 - 2 * c + p1
    return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))


def _match(window_detail: np.ndarray, mark: np.ndarray,
           box: tuple[int, int, int, int] | None) -> tuple[float, float, float]:
    """NCC peak of the mark's detail layer, searched only inside box = (x0, y0, x1, y1) of origins."""
    if box is not None:
        x0, y0, x1, y1 = box
        mh, mw = mark.shape
        window_detail = window_detail[y0:y1 + mh, x0:x1 + mw]
    else:
        x0 = y0 = 0
    score = cv2.matchTemplate(window_detail, detail(mark), cv2.TM_CCOEFF_NORMED)
    a = np.abs(score)
    y, x = np.unravel_index(int(a.argmax()), a.shape)
    fx = _parabola(a[y, x - 1], a[y, x], a[y, x + 1]) if 0 < x < a.shape[1] - 1 else 0.0
    fy = _parabola(a[y - 1, x], a[y, x], a[y + 1, x]) if 0 < y < a.shape[0] - 1 else 0.0
    return x0 + x + fx, y0 + y + fy, float(score[y, x])


def search_box(profile: MarkProfile, prior: tuple[float, float], window_shape: tuple[int, int],
               margin: float = 3.0) -> tuple[int, int, int, int]:
    """Integer origin range around the expected origin covering the profile's jitter."""
    h, w = window_shape
    fw, fh = profile.frame_size
    jx, jy = profile.jitter["x"], profile.jitter["y"]
    x0 = max(0, int(np.floor(prior[0] + jx[0] - margin)))
    y0 = max(0, int(np.floor(prior[1] + jy[0] - margin)))
    x1 = min(w - fw, int(np.ceil(prior[0] + jx[1] + margin)))
    y1 = min(h - fh, int(np.ceil(prior[1] + jy[1] + margin)))
    return x0, y0, max(x0, x1), max(y0, y1)


TEXT_SEARCH = (6, 4)        # +- px around a style's nominal text placement (x, y)
MIN_TEXT_SCORE = 0.12       # below this (with a clearly visible logo) the mark has no text line
SCALES = (0.98, 0.99, 1.0, 1.01, 1.02)   # stamped size varies by a few % between photos


def locate_text(window_detail: np.ndarray, profile: MarkProfile, listing_id: str, x: float, y: float,
                styles: list[str] | None = None, scale: float = 1.0) -> tuple[str, tuple[float, float], float]:
    """Best text style and placement (offset from the style's nominal position) for a mark at (x, y).

    The text line's placement relative to the logo varies between photos by a few
    pixels, so it is searched separately once the logo has fixed the frame.
    """
    sx, sy = TEXT_SEARCH
    h, w = window_detail.shape
    best = (profile.default_style, (0.0, 0.0), 0.0)
    for style in styles or list(profile.styles):
        t = profile.render_text_only(listing_id, style)
        if scale != 1.0:
            t = profile.place(t, scale=scale)
        ys, xs = np.nonzero(t > 1e-3)
        if len(xs) == 0:
            continue
        # text template (with margin) and the window region it may occupy
        ty0, ty1, tx0, tx1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
        tmpl = shift_image(t, x - np.floor(x), y - np.floor(y))[ty0:ty1, tx0:tx1]
        ox, oy = int(np.floor(x)) + tx0, int(np.floor(y)) + ty0
        rx0, ry0 = max(0, ox - sx), max(0, oy - sy)
        rx1, ry1 = min(w, ox + tmpl.shape[1] + sx), min(h, oy + tmpl.shape[0] + sy)
        region = window_detail[ry0:ry1, rx0:rx1]
        if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
            continue
        score = cv2.matchTemplate(region, detail(tmpl), cv2.TM_CCOEFF_NORMED)
        a = np.abs(score)
        j, i = np.unravel_index(int(a.argmax()), a.shape)
        fx = _parabola(a[j, i - 1], a[j, i], a[j, i + 1]) if 0 < i < a.shape[1] - 1 else 0.0
        fy = _parabola(a[j - 1, i], a[j, i], a[j + 1, i]) if 0 < j < a.shape[0] - 1 else 0.0
        off = (rx0 + i + fx - ox, ry0 + j + fy - oy)
        if a[j, i] > abs(best[2]):
            best = (style, (float(off[0]), float(off[1])), float(score[j, i]))
    return best


def localize(window_rgb: np.ndarray, profile: MarkProfile, listing_id: str,
             prior: tuple[float, float] | None = None,
             styles: list[str] | None = None) -> tuple[float, float, float, str, tuple[float, float], float]:
    """Return (x, y, score, style, text_offset, scale).

    1. coarse: the full mark (each style at its nominal text placement) finds the
       neighbourhood; it carries more signal than the logo alone;
    2. logo: the logo alone (identical in every variant) fixes the origin within
       +-3 px of it, so an off-nominal text line cannot pull the frame;
    3. text: the text line is searched separately around its nominal placement
       (it varies between photos by a few px); "none" if the logo is clear but no
       text line is;
    4. final: the full mark with the text where it was found re-fixes the origin
       within +-2 px (most signal, and the text no longer pulls it off).
    prior: the expected frame origin inside the window; the coarse search is
    limited to the profile's jitter range around it (plus a small margin).
    """
    wd = detail(window_rgb.astype(np.float32) @ _LUMA)
    box = search_box(profile, prior, wd.shape) if prior is not None else None
    coarse = None
    for st in styles or list(profile.styles):
        x, y, s = _match(wd, profile.render(listing_id, style=st), box)
        if coarse is None or abs(s) > abs(coarse[2]):
            coarse = (x, y, s)
    # logo refine, also over a few scales (the stamped size varies by a few % between photos)
    best = None
    for sc in SCALES:
        logo = profile.logo if sc == 1.0 else profile.place(profile.logo, scale=sc)
        x, y, s = _match(wd, logo, _local_box(coarse[0], coarse[1], profile, wd.shape))
        if best is None or abs(s) > abs(best[2]):
            best = (x, y, s, sc)
    x, y, s, scale = best
    style, offset, ts = locate_text(wd, profile, listing_id, x, y, styles, scale)
    if abs(ts) < MIN_TEXT_SCORE:
        style, offset = ("none", (0.0, 0.0)) if abs(s) >= 0.3 else (style, (0.0, 0.0))
    # 4. final: the full mark with its text where it was found carries the most signal
    full = profile.render_frame(listing_id, style, offset)
    if scale != 1.0:
        full = profile.place(full, scale=scale)
    x, y, s = _match(wd, full, _local_box(x, y, profile, wd.shape, r=2))
    return x, y, s, style, offset, scale


def _local_box(x: float, y: float, profile: MarkProfile, shape, r: int = 3) -> tuple[int, int, int, int]:
    h, w = shape
    fw, fh = profile.frame_size
    x0, y0 = max(0, int(round(x)) - r), max(0, int(round(y)) - r)
    return x0, y0, max(x0, min(w - fw, int(round(x)) + r)), max(y0, min(h - fh, int(round(y)) + r))
