# Mirrors watermark-removal: wm_shared/marks/fixed.py (geometry, maps) and
# training/src/tasks/overlay/features.py build_input_layered (model input). Keep in sync.
"""
Small Avito mark removal on ONNX Runtime + numpy.

The mark is stamped at the same pixels relative to the bottom-right corner of
every image (after resizing), so there is nothing to locate: the window is the
image's last window_h rows x window_w columns, and the hint is the nominal mark,
the same for every image. Per pixel and channel the mark is
    out = b * (1 - Alpha) + A
and the model predicts a correction on top of the exact inverse (out - A) / (1 - Alpha).
Pixels outside the window are returned untouched.

Profile directory (models/avito/, exported with the model by watermark-removal):
    mark.json  "window": [w, h], "frame": [w, h] (the mark maps' size), ...
    mark.npz   "alpha", "a": frame_h x frame_w x 3 nominal maps, bottom-right aligned
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AvitoMark:
    window: tuple[int, int]      # (w, h)
    alpha: np.ndarray            # window_h x window_w x 3
    a: np.ndarray

    def window_origin(self, width: int, height: int) -> tuple[int, int]:
        ww, wh = self.window
        return width - ww, height - wh


def load_mark(mark_dir: str | Path) -> AvitoMark:
    root = Path(mark_dir)
    meta = json.loads((root / "mark.json").read_text(encoding="utf-8"))
    if meta.get("type") != "fixed_layered":
        raise ValueError(f"{root}/mark.json is not a fixed layered mark (type {meta.get('type')!r})")
    z = np.load(root / "mark.npz")
    ww, wh = meta["window"]
    maps = []
    for key in ("alpha", "a"):
        m = z[key].astype(np.float32)
        out = np.zeros((wh, ww, 3), np.float32)
        out[wh - m.shape[0]:, ww - m.shape[1]:] = m
        maps.append(out)
    return AvitoMark(window=(int(ww), int(wh)), alpha=maps[0], a=maps[1])


def build_input(win_rgb: np.ndarray, alpha: np.ndarray, a: np.ndarray, hint_scale: float) -> np.ndarray:
    """win_rgb: HxWx3 uint8, alpha / a: HxWx3 -> 7xHxW float32 (stamped, exact inverse, mean Alpha)."""
    wm = win_rgb.astype(np.float32)
    inv = np.clip((wm - a) / np.maximum(1.0 - alpha, 1e-3), -32.0, 287.0)
    hint = alpha.mean(axis=2, keepdims=True) * hint_scale
    return np.concatenate([wm / 127.5 - 1.0, inv / 127.5 - 1.0, hint], axis=2).transpose(2, 0, 1)


class AvitoRemover:
    def __init__(self, onnx_path: str, mark: AvitoMark, threads: int = 0, hint_scale: float = 8.0):
        import onnxruntime as ort
        self.mark = mark
        self.hint_scale = hint_scale
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    def prepare(self, bgr: np.ndarray) -> tuple[tuple[int, int], np.ndarray] | None:
        """Window origin and model input (7xhxw) for one BGR image; None if the image is smaller than the window."""
        h, w = bgr.shape[:2]
        ww, wh = self.mark.window
        if h < wh or w < ww:
            return None
        x0, y0 = self.mark.window_origin(w, h)
        win = np.ascontiguousarray(bgr[y0:, x0:, ::-1])
        return (x0, y0), build_input(win, self.mark.alpha, self.mark.a, self.hint_scale)

    @staticmethod
    def paste(bgr: np.ndarray, x: np.ndarray, delta: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        pred = np.clip(x[:3] - delta, -1.0, 1.0).transpose(1, 2, 0)
        win = np.rint((pred + 1.0) * 127.5).clip(0, 255).astype(np.uint8)[..., ::-1]
        out = bgr.copy()
        x0, y0 = origin
        out[y0:y0 + win.shape[0], x0:x0 + win.shape[1]] = win
        return out

    def run(self, xs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"x": xs.astype(np.float32)})[0]
