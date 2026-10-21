# Vendored from watermark-removal/wm_shared/marks/profile.py (commit 5710cb5). Keep in sync.
"""
Anchored overlay marks: a semi-transparent, fixed-pixel-size mark whose content
is known per image (a constant logo raster plus a text line such as
"ID <listing id>"), stamped near a fixed anchor with some jitter, then
JPEG-compressed.

Everything watermark-specific lives in a profile directory (assets/marks/<name>/):
    mark.json  geometry, text styles, overlay colour, anchor model, window, codec
    mark.npz   "logo": opacity of the constant part in the mark frame
    fonts/     font files referenced by the text styles

Opacity model (8-bit sRGB, per channel):  out = bg + m * (overlay - bg)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path("./models")   # the service keeps mark assets next to the checkpoints


def shift_image(img: np.ndarray, dx: float, dy: float, size: tuple[int, int] | None = None) -> np.ndarray:
    """Translate by a (possibly fractional) offset with Lanczos resampling; zero outside."""
    w, h = size if size is not None else (img.shape[1], img.shape[0])
    mat = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@lru_cache(maxsize=16)
def _font(path: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, px)


@dataclass
class TextStyle:
    font: str          # absolute path
    size: float        # pixel size
    opacity: float
    center_x: float    # frame x of the ink box centre
    top: float         # frame y of the ink box top
    prob: float = 1.0


def _layout(text: str, style: TextStyle, ss: int = 8):
    """Canvas size and draw parameters for `text` supersampled ss x (2-px margin, room for a sub-pixel shift)."""
    font = _font(style.font, int(round(style.size * ss)))
    l, t, r, b = font.getbbox(text, anchor="ls")
    pad = 2 * ss
    w = math.ceil((r - l + 2 * pad) / ss) * ss + ss
    h = math.ceil((b - t + 2 * pad) / ss) * ss + ss
    return w, h, l, t, pad, font


def text_coverage(text: str, style: TextStyle, x_left: float, y_top: float, ss: int = 8):
    """Coverage of `text` whose ink box starts at (x_left, y_top) in frame coordinates.

    Returns (coverage, ix, iy): paste coverage at integer offset (ix, iy).
    """
    w, h, l, t, pad, font = _layout(text, style, ss)
    ix, iy = math.floor(x_left) - 2, math.floor(y_top) - 2
    ox = int(round((x_left - ix) * ss)) - l   # draw offset so the ink starts at x_left
    oy = int(round((y_top - iy) * ss)) - t
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((ox, oy), text, fill=255, font=font, anchor="ls")
    a = np.asarray(img, np.float32) / 255.0
    return a.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3)), ix, iy


def text_ink_width(text: str, style: TextStyle, ss: int = 8) -> float:
    font = _font(style.font, int(round(style.size * ss)))
    l, _, r, _ = font.getbbox(text, anchor="ls")
    return (r - l) / ss


@dataclass
class MarkProfile:
    name: str
    logo: np.ndarray            # frame_h x frame_w, float32
    styles: dict                # style name -> TextStyle
    text_prefix: str            # e.g. "ID "
    id_lengths: dict            # {length: probability}
    overlay: float              # overlay grey level
    anchor: dict                # frame origin = (W - (ax*W + bx), H - (ay*H + by))
    jitter: dict                # {"x": [lo, hi], "y": [lo, hi]} around the anchor, pixels
    window: tuple[int, int]     # inference window (w, h), multiples of 32
    grid: int                   # window origin snapping (JPEG MCU size)
    codec: dict                 # {"jpeg_quality": {q: prob}}
    sizes: list = field(default_factory=list)   # observed (W, H) image sizes to sample from

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.logo.shape[1], self.logo.shape[0]

    @property
    def default_style(self) -> str:
        return max(self.styles, key=lambda s: self.styles[s].prob)

    # ── rendering ────────────────────────────────────────────────────────────

    def render_frame(self, listing_id: str, style: str | None = None,
                     text_offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Opacity map of the full mark for an ID in its own frame (cached).

        style "none" renders the logo without a text line. text_offset moves the
        text line relative to the logo (the placement varies between photos).
        """
        style = style or self.default_style
        key = (str(listing_id), style, round(text_offset[0], 2), round(text_offset[1], 2))
        cache = self.__dict__.setdefault("_frame_cache", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            m = self.logo.copy()
            if style != "none":
                st = self.styles[style]
                text = self.text_prefix + str(listing_id)
                x_left = st.center_x + text_offset[0] - text_ink_width(text, st) / 2
                cov, tx, ty = text_coverage(text, st, x_left, st.top + text_offset[1])
                _paste_add(m, cov * st.opacity, tx, ty)
            cache[key] = m
        return cache[key]

    def render_text_only(self, listing_id: str, style: str,
                         text_offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Frame-sized opacity of the text line alone."""
        return self.render_frame(listing_id, style, text_offset) - self.logo

    def render(self, listing_id: str, dx: float = 0.0, dy: float = 0.0,
               size: tuple[int, int] | None = None, style: str | None = None,
               text_offset: tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> np.ndarray:
        """Opacity map of the full mark for an ID, with the frame origin at (dx, dy).

        With size=(w, h) the canvas is w x h; otherwise it is frame-sized. Fractional
        offsets and scale (about the frame centre; the stamped size varies by a few %
        between photos) are applied with Lanczos resampling of the frame rendering.
        """
        return self.place(self.render_frame(listing_id, style, text_offset), dx, dy, size, scale)

    def place(self, frame_map: np.ndarray, dx: float = 0.0, dy: float = 0.0,
              size: tuple[int, int] | None = None, scale: float = 1.0) -> np.ndarray:
        """Put a frame-sized map on a canvas with its origin at (dx, dy), scaled about the frame centre."""
        fw, fh = self.frame_size
        w, h = size if size is not None else (fw, fh)
        if scale == 1.0:
            ix, iy = math.floor(dx), math.floor(dy)
            fx, fy = dx - ix, dy - iy
            shifted = frame_map if (fx == 0 and fy == 0) else shift_image(frame_map, fx, fy, size=(fw + 1, fh + 1))
            m = np.zeros((h, w), np.float32)
            _paste_add(m, shifted, ix, iy)
            return m
        cx, cy = fw / 2.0, fh / 2.0
        mat = np.array([[scale, 0, cx * (1 - scale) + dx], [0, scale, cy * (1 - scale) + dy]], np.float32)
        return cv2.warpAffine(frame_map, mat, (w, h), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @staticmethod
    def composite(bg: np.ndarray, m: np.ndarray, overlay: float) -> np.ndarray:
        """bg: H x W x 3 float (0..255), m: H x W. Returns the stamped image as float."""
        return bg + m[..., None] * (overlay - bg)

    @staticmethod
    def invert(img: np.ndarray, m: np.ndarray, overlay: float) -> np.ndarray:
        """Exact inverse of composite() for a known opacity map (before compression)."""
        mm = m[..., None]
        return (img - mm * overlay) / np.maximum(1.0 - mm, 1e-3)

    # ── geometry ─────────────────────────────────────────────────────────────

    def expected_origin(self, width: int, height: int) -> tuple[float, float]:
        a = self.anchor
        return width - (a["ax"] * width + a["bx"]), height - (a["ay"] * height + a["by"])

    def window_origin(self, width: int, height: int) -> tuple[int, int]:
        """Top-left of the inference window, snapped to the JPEG grid and kept inside the image."""
        ww, wh = self.window
        fw, fh = self.frame_size
        ex, ey = self.expected_origin(width, height)
        g = self.grid
        x0 = int(np.floor((ex - (ww - fw) / 2) / g)) * g
        y0 = int(np.floor((ey - (wh - fh) / 2) / g)) * g
        x0 = max(0, min(x0, (width - ww) // g * g))
        y0 = max(0, min(y0, (height - wh) // g * g))
        return x0, y0


def _paste_add(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = dst.shape
    sh, sw = src.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, w), min(y + sh, h)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] += src[y0 - y:y1 - y, x0 - x:x1 - x]


def load_mark(name: str = "cian", assets_dir: Path = ASSETS_DIR) -> MarkProfile:
    root = Path(assets_dir) / name
    meta = json.loads((root / "mark.json").read_text(encoding="utf-8"))
    arrays = np.load(root / "mark.npz")
    styles = {k: TextStyle(font=str(root / v["font"]), size=float(v["size"]), opacity=float(v["opacity"]),
                           center_x=float(v["center_x"]), top=float(v["top"]), prob=float(v.get("prob", 1.0)))
              for k, v in meta["text"]["styles"].items()}
    return MarkProfile(
        name=name,
        logo=arrays["logo"].astype(np.float32),
        styles=styles,
        text_prefix=meta["text"].get("prefix", ""),
        id_lengths={int(k): float(v) for k, v in meta["text"].get("id_lengths", {"9": 1.0}).items()},
        overlay=float(meta["overlay"]),
        anchor=meta["anchor"],
        jitter=meta["jitter"],
        window=tuple(meta["window"]),
        grid=int(meta.get("grid", 16)),
        codec=meta["codec"],
        sizes=[tuple(s) for s in meta.get("sizes", [])],
    )
