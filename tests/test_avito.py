"""Tests for mode "avito" (small Avito logo) and the renamed legacy mode.

The removal test needs the exported model and mark assets in models/
(model_avito_*.onnx, models/avito/) and is skipped without them.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.schemas import ProcessingMode
from src.avito import build_input, load_mark
from src.pipeline.avito import AvitoStep
from src.pipeline.context import ImageContext

MODELS = Path(__file__).resolve().parents[1] / "models"


def _mark_dir(tmp_path, frame=(20, 10), window=(32, 16)):
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0, 0.6, (frame[1], frame[0], 3)).astype(np.float32)
    a = (alpha * 250).astype(np.float32)
    np.savez(tmp_path / "mark.npz", alpha=alpha, a=a)
    (tmp_path / "mark.json").write_text(json.dumps({"type": "fixed_layered", "frame": list(frame),
                                                    "window": list(window)}))
    return tmp_path, alpha, a


class TestAvitoMark:
    def test_maps_are_bottom_right_aligned(self, tmp_path):
        d, alpha, a = _mark_dir(tmp_path)
        mark = load_mark(d)
        assert mark.alpha.shape == (16, 32, 3)
        assert np.array_equal(mark.alpha[-10:, -20:], alpha) and not mark.alpha[:6].any()
        assert np.array_equal(mark.a[-10:, -20:], a)
        assert mark.window_origin(100, 50) == (68, 34)

    def test_rejects_other_mark_types(self, tmp_path):
        np.savez(tmp_path / "mark.npz", logo=np.zeros((4, 4), np.float32))
        (tmp_path / "mark.json").write_text(json.dumps({"window": [32, 16]}))
        with pytest.raises(ValueError, match="fixed layered"):
            load_mark(tmp_path)

    def test_input_inverts_the_mark(self, tmp_path):
        d, _, _ = _mark_dir(tmp_path)
        mark = load_mark(d)
        bg = np.full((16, 32, 3), 90.0, np.float32)
        stamped = np.clip(np.rint(bg * (1 - mark.alpha) + mark.a), 0, 255).astype(np.uint8)
        x = build_input(stamped, mark.alpha, mark.a, 8.0)
        assert x.shape == (7, 16, 32) and x.dtype == np.float32
        inv = (x[3:6] + 1) * 127.5
        assert np.abs(inv - 90.0).max() < 1.5        # rounding of the stamp only
        assert np.allclose(x[6], mark.alpha.mean(axis=2) * 8.0)


class TestAvitoStep:
    def _step(self, tmp_path):
        from src.avito import AvitoRemover
        d, _, _ = _mark_dir(tmp_path)
        remover = AvitoRemover.__new__(AvitoRemover)
        remover.mark, remover.hint_scale = load_mark(d), 8.0
        remover.run = MagicMock(side_effect=lambda xs: np.zeros((len(xs), 3) + xs.shape[2:], np.float32))
        return AvitoStep(remover, 2, ThreadPoolExecutor(2)), remover

    def test_zero_correction_returns_the_input(self, tmp_path):
        step, remover = self._step(tmp_path)
        rng = np.random.default_rng(1)
        ctxs = [ImageContext(image_id=str(i), original_bgr=rng.integers(0, 256, (40 + i, 60, 3), dtype=np.uint8))
                for i in range(3)]
        step.process_batch(ctxs)
        assert remover.run.call_count == 2          # sub-batches of 2
        for c in ctxs:
            assert c.error is None and np.array_equal(c.result_bgr, c.original_bgr)

    def test_correction_only_inside_the_window(self, tmp_path):
        step, remover = self._step(tmp_path)
        remover.run = MagicMock(side_effect=lambda xs: np.full((len(xs), 3) + xs.shape[2:], 0.1, np.float32))
        img = np.full((40, 60, 3), 128, np.uint8)
        ctx = ImageContext(image_id="a", original_bgr=img)
        step.process_batch([ctx])
        changed = (ctx.result_bgr != img).any(axis=2)
        assert changed[-16:, -32:].all() and not changed[:-16].any() and not changed[:, :-32].any()

    def test_small_image_unchanged(self, tmp_path):
        step, remover = self._step(tmp_path)
        ctx = ImageContext(image_id="s", original_bgr=np.zeros((10, 10, 3), np.uint8))
        step.process_batch([ctx])
        assert ctx.error is None and np.array_equal(ctx.result_bgr, ctx.original_bgr)
        remover.run.assert_not_called()


class TestModes:
    def test_mode_names(self):
        assert {m.value for m in ProcessingMode} >= {"avito", "legacy_avito", "cian", "old", "new"}

    def test_avito_requires_enabled_config(self):
        from src.pipeline.pipeline import WatermarkRemovalPipeline
        pipeline = WatermarkRemovalPipeline.__new__(WatermarkRemovalPipeline)
        pipeline.avito_step = None
        pipeline._decode_to_context = lambda data, image_id=None: ImageContext(
            image_id="x", original_bgr=np.zeros((8, 8, 3), np.uint8))
        with pytest.raises(RuntimeError, match="Avito mode not enabled"):
            pipeline.process_single(b"", mode="avito")


def _models_present() -> bool:
    return any(MODELS.glob("model_avito_*.onnx")) and (MODELS / "avito" / "mark.json").exists()


@pytest.mark.skipif(not _models_present(), reason="needs models/model_avito_*.onnx and models/avito/")
def test_real_model_runs(tmp_path):
    from src.avito import AvitoRemover
    onnx = sorted(MODELS.glob("model_avito_*.onnx"))[-1]
    remover = AvitoRemover(str(onnx), load_mark(MODELS / "avito"))
    step = AvitoStep(remover, 4, ThreadPoolExecutor(2))
    img = np.full((960, 1280, 3), 120, np.uint8)
    ctx = ImageContext(image_id="r", original_bgr=img)
    step.process_batch([ctx])
    assert ctx.error is None and ctx.result_bgr.shape == img.shape
    assert np.array_equal(ctx.result_bgr[:-64], img[:-64])
