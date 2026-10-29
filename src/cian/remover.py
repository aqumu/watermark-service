# Vendored from watermark-removal/training/src/tasks/overlay/onnx_remover.py (BGR in/out here). Keep in sync.
"""
Cian mark removal on ONNX Runtime + numpy.

The mark's position follows the image size, so only a fixed window around it
is processed: the mark is located there (logo, size, ID-line style and
placement for the listing's ID), the exact mark is rendered as a hint, and the
model predicts a correction on top of the exact inverse. Pixels outside the
window are returned untouched.
"""
from __future__ import annotations

import numpy as np

from src.cian.localize import localize
from src.cian.profile import MarkProfile


def build_input(win_rgb: np.ndarray, hint: np.ndarray, overlay: float, hint_scale: float) -> np.ndarray:
    """win_rgb: HxWx3 uint8, hint: HxW opacity -> 7xHxW float32 (stamped, exact inverse, hint)."""
    wm = win_rgb.astype(np.float32).transpose(2, 0, 1)
    m = hint.astype(np.float32)[None]
    inv = np.clip((wm - m * overlay) / np.maximum(1.0 - m, 1e-3), -32.0, 287.0)
    return np.concatenate([wm / 127.5 - 1.0, inv / 127.5 - 1.0, m * hint_scale], axis=0)


class CianRemover:
    def __init__(self, onnx_path: str, profile: MarkProfile, threads: int = 0, hint_scale: float = 8.0):
        import onnxruntime as ort
        self.profile = profile
        self.hint_scale = hint_scale
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    def prepare(self, bgr: np.ndarray, listing_id: str) -> tuple[tuple[int, int], np.ndarray, dict]:
        """Window origin, model input (7xhxw) and localisation diagnostics for one BGR image."""
        p = self.profile
        rgb = bgr[..., ::-1]
        h, w = rgb.shape[:2]
        ww, wh = p.window
        x0, y0 = p.window_origin(w, h)
        win = np.ascontiguousarray(rgb[y0:y0 + wh, x0:x0 + ww])
        if win.shape[0] != wh or win.shape[1] != ww:
            win = np.pad(win, ((0, wh - win.shape[0]), (0, ww - win.shape[1]), (0, 0)), mode="edge")
        ex, ey = p.expected_origin(w, h)
        lx, ly, score, style, offset, scale = localize(win, p, str(listing_id), prior=(ex - x0, ey - y0))
        hint = p.render(str(listing_id), lx, ly, size=(ww, wh), style=style, text_offset=offset, scale=scale)
        info = {"window": (x0, y0), "score": score, "style": style, "scale": scale}
        return (x0, y0), build_input(win, hint, p.overlay, self.hint_scale), info

    @staticmethod
    def paste(bgr: np.ndarray, x: np.ndarray, delta: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        pred = np.clip(x[:3] - delta, -1.0, 1.0).transpose(1, 2, 0)
        win = np.rint((pred + 1.0) * 127.5).clip(0, 255).astype(np.uint8)[..., ::-1]
        out = bgr.copy()
        x0, y0 = origin
        h, w = bgr.shape[:2]
        out[y0:y0 + win.shape[0], x0:x0 + win.shape[1]] = win[:h - y0, :w - x0]
        return out

    def run(self, xs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"x": xs.astype(np.float32)})[0]
