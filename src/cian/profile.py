# Vendored from watermark-removal/wm_shared/marks/profile.py (commits 6c1b6f9, f5bfe5c, febb794). Keep in sync.
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

Measured on real marks against synthetic ones (tools/cian_watermark/fit_halo.py):
  shadow  a darkening around (and under) the logo that does not depend on the
          background (out -= shadow, in grey levels), by distance to the logo's
          glyph edge, with per-channel factors. mark.json "shadow".
  text    the real text line is bolder than the font rendering: an opacity gain
          and an outer glow of the white layer, glow * max(blur_sigma(t) - t, 0).
          mark.json "edge": {"text": {"gain": g, "glow": [glow, sigma], "blur": sigma}}.
          Per character the strength differs too (the "ID" prefix is fainter than
          the digits): "char_gain": {style: {"I": g, "D": g, "digit": g}}.
  logo    opacity gain of the stacked logo raster: "edge": {"logo": {"gain": g}}.
  rim     optional inner rim, gain * max(m - blur_sigma(m), 0) ("edge": {part: {"rim": [g, s]}}).
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
EDGE_LEVEL = 0.06      # logo opacity taken as the glyph edge (shadow distances)


def shift_image(img: np.ndarray, dx: float, dy: float, size: tuple[int, int] | None = None) -> np.ndarray:
    """Translate by a (possibly fractional) offset with Lanczos resampling; zero outside."""
    w, h = size if size is not None else (img.shape[1], img.shape[0])
    mat = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def inner_rim(m: np.ndarray, sigma: float) -> np.ndarray:
    """Brightening just inside the glyph edges: the positive part of m - blur(m) (no outer undershoot)."""
    return np.maximum(m - cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT), 0)


def with_rim(m: np.ndarray, rim: tuple[float, float] | None) -> np.ndarray:
    return m if not rim or not rim[0] else m + rim[0] * inner_rim(m, rim[1])


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


def _char_gains(text: str, font, ox: int, oy: int, size: tuple[int, int], gains: dict) -> np.ndarray:
    """Supersampled per-pixel gain: each glyph's ink gets the gain of its character
    ("digit" for 0-9); the string itself is still drawn in one go."""
    g = np.ones(size[::-1], np.float32)
    for i, ch in enumerate(text):
        gain = gains.get("digit" if ch.isdigit() else ch)
        if gain is None or ch == " ":
            continue
        x = ox + font.getlength(text[:i + 1]) - font.getlength(ch)   # pen position, kerning included
        img = Image.new("L", size, 0)
        ImageDraw.Draw(img).text((x, oy), ch, fill=255, font=font, anchor="ls")
        g[np.asarray(img) > 0] = gain
    return g


def text_coverage(text: str, style: TextStyle, x_left: float, y_top: float, ss: int = 8,
                  gains: dict | None = None):
    """Coverage of `text` whose ink box starts at (x_left, y_top) in frame coordinates.

    gains: optional opacity gain per character ({"I": g, "D": g, "digit": g}).
    Returns (coverage, ix, iy): paste coverage at integer offset (ix, iy).
    """
    w, h, l, t, pad, font = _layout(text, style, ss)
    ix, iy = math.floor(x_left) - 2, math.floor(y_top) - 2
    ox = int(round((x_left - ix) * ss)) - l   # draw offset so the ink starts at x_left
    oy = int(round((y_top - iy) * ss)) - t
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((ox, oy), text, fill=255, font=font, anchor="ls")
    a = np.asarray(img, np.float32) / 255.0
    if gains:
        a = a * _char_gains(text, font, ox, oy, (w, h), gains)
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
    text_rim: tuple[float, float] | None = None  # (gain, sigma) inner rim of the text line
    text_blur: float = 0.0                       # sigma of the text coverage blur
    text_gain: float = 1.0                       # text opacity multiplier
    text_glow: tuple[float, float] | None = None  # (gain, sigma) outer glow of the text line
    char_gain: dict = field(default_factory=dict)  # style -> {"I": g, "D": g, "digit": g}
    shadow: dict | None = None                   # {"distance": [...], "levels": [...], "rgb": [r, g, b]}

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
                cov, tx, ty = text_coverage(text, st, x_left, st.top + text_offset[1],
                                            gains=self.char_gain.get(style))
                cov = cov * st.opacity * self.text_gain
                if self.text_glow and self.text_glow[0]:
                    g, sg = self.text_glow
                    cov = cov + g * np.maximum(
                        cv2.GaussianBlur(cov, (0, 0), sg, borderType=cv2.BORDER_CONSTANT) - cov, 0)
                if self.text_blur:
                    cov = cv2.GaussianBlur(cov, (0, 0), self.text_blur, borderType=cv2.BORDER_CONSTANT)
                _paste_add(m, with_rim(cov, self.text_rim), tx, ty)
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

    def shadow_map(self, logo_placed: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """H x W x 3 darkening (grey levels) for a placed logo opacity map (render(style="none"))."""
        h, w = logo_placed.shape
        if not self.shadow or gain == 0:
            return np.zeros((h, w, 3), np.float32)
        mask = (logo_placed > EDGE_LEVEL).astype(np.uint8)
        d = np.where(mask > 0, -cv2.distanceTransform(mask, cv2.DIST_L2, 5),
                     cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5))
        lv = np.interp(d, self.shadow["distance"], self.shadow["levels"], right=0.0).astype(np.float32)
        return gain * lv[..., None] * np.asarray(self.shadow["rgb"], np.float32)

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
    edge = meta.get("edge", {})
    rim = {part: tuple(v["rim"]) for part, v in edge.items() if "rim" in v}
    styles = {k: TextStyle(font=str(root / v["font"]), size=float(v["size"]), opacity=float(v["opacity"]),
                           center_x=float(v["center_x"]), top=float(v["top"]), prob=float(v.get("prob", 1.0)))
              for k, v in meta["text"]["styles"].items()}
    return MarkProfile(
        name=name,
        logo=with_rim(arrays["logo"].astype(np.float32) * float(edge.get("logo", {}).get("gain", 1.0)),
                      rim.get("logo")),
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
        text_rim=rim.get("text"),
        text_blur=float(edge.get("text", {}).get("blur", 0.0)),
        text_gain=float(edge.get("text", {}).get("gain", 1.0)),
        text_glow=tuple(edge["text"]["glow"]) if "glow" in edge.get("text", {}) else None,
        char_gain=edge.get("text", {}).get("char_gain", {}),
        shadow=meta.get("shadow"),
    )
