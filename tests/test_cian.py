"""Tests for mode "cian".

API tests mock the pipeline. The removal test needs the exported model and mark
assets in models/ (model_cian_*.onnx, models/cian/) and is skipped without them.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.pipeline.context import ImageContext
from src.processing.io import encode_image

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    png = encode_image(np.full((64, 64, 3), 255, dtype=np.uint8), fmt="png")
    pipeline.process_single.return_value = png
    pipeline.process_batch.return_value = [png, png]
    pipeline.device = MagicMock()
    pipeline.device.type = "cpu"
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    from src.api.dependencies import get_pipeline
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    with TestClient(app) as c:
        yield c


def _png():
    return encode_image(np.zeros((32, 32, 3), dtype=np.uint8), fmt="png")


class TestCianApi:
    def test_single_passes_listing_id(self, client, mock_pipeline):
        resp = client.post("/api/v1/process?mode=cian&listing_id=333768830",
                           files={"image": ("a.png", _png(), "image/png")})
        assert resp.status_code == 200
        kwargs = mock_pipeline.process_single.call_args.kwargs
        assert kwargs["mode"] == "cian" and kwargs["listing_id"] == "333768830"

    def test_batch_passes_listing_ids_in_order(self, client, mock_pipeline):
        resp = client.post("/api/v1/process/batch?mode=cian",
                           files=[("images", ("a.png", _png(), "image/png")), ("images", ("b.png", _png(), "image/png"))],
                           data={"listing_ids": "111111111, 222222222"})
        assert resp.status_code == 200
        assert mock_pipeline.process_batch.call_args.kwargs["listing_ids"] == ["111111111", "222222222"]

    def test_async_batch_rejects_missing_ids(self, client):
        resp = client.post("/api/v1/process/batch/async?mode=cian",
                           files=[("images", ("a.png", _png(), "image/png")), ("images", ("b.png", _png(), "image/png"))],
                           data={"listing_ids": "111111111"})
        assert resp.status_code == 400

    def test_cian_requires_enabled_config(self):
        from src.pipeline.pipeline import WatermarkRemovalPipeline
        pipeline = WatermarkRemovalPipeline.__new__(WatermarkRemovalPipeline)
        pipeline.cian_step = None
        pipeline._decode_to_context = lambda data, image_id=None: ImageContext(
            image_id="x", original_bgr=np.zeros((8, 8, 3), np.uint8))
        with pytest.raises(RuntimeError, match="Cian mode not enabled"):
            pipeline.process_single(b"", mode="cian", listing_id="1")


def _models_present() -> bool:
    return (MODELS / "cian" / "mark.json").exists() and any(MODELS.glob("model_cian_*.onnx"))


@pytest.mark.skipif(not _models_present(), reason="needs models/model_cian_*.onnx and models/cian/")
class TestCianRemoval:
    def _remover(self):
        from src.cian import CianRemover, load_mark
        profile = load_mark("cian", MODELS)
        onnx = sorted(MODELS.glob("model_cian_*.onnx"))[-1]
        return CianRemover(str(onnx), profile, threads=2)

    def test_removes_mark_and_leaves_rest_untouched(self):
        from src.pipeline.cian import CianStep
        remover = self._remover()
        p = remover.profile
        h, w = 960, 1280
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:h, 0:w]
        clean = np.stack([90 + 40 * np.sin(xx / 97.0), 110 + 30 * np.cos(yy / 71.0),
                          100 + 20 * np.sin((xx + yy) / 53.0)], -1) + rng.normal(0, 2, (h, w, 3))
        clean = np.clip(clean, 0, 255).astype(np.float32)
        ex, ey = p.expected_origin(w, h)
        m = p.render("333768830", ex + 3.4, ey - 2.2, size=(w, h), style="bold")
        stamped = np.clip(np.rint(p.composite(clean, m, p.overlay)), 0, 255).astype(np.uint8)

        def jpeg(img):
            ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
            return cv2.imdecode(enc, cv2.IMREAD_COLOR)   # BGR

        wm_bgr, clean_bgr = jpeg(stamped), jpeg(clean.astype(np.uint8))
        ctx = ImageContext(image_id="t", original_bgr=wm_bgr)
        CianStep(remover, 4, ThreadPoolExecutor(2)).process_batch([ctx], ["333768830"])
        assert ctx.error is None and ctx.result_bgr is not None

        region = m > 1e-3
        before = np.abs(wm_bgr.astype(int) - clean_bgr.astype(int))[region].mean()
        after = np.abs(ctx.result_bgr.astype(int) - clean_bgr.astype(int))[region].mean()
        assert after < before * 0.35, (before, after)

        x0, y0 = p.window_origin(w, h)
        ww, wh = p.window
        outside = np.ones((h, w), bool)
        outside[y0:y0 + wh, x0:x0 + ww] = False
        assert np.array_equal(ctx.result_bgr[outside], wm_bgr[outside])

    def test_bad_listing_id_is_reported_per_image(self):
        from src.pipeline.cian import CianStep
        ctx = ImageContext(image_id="t", original_bgr=np.zeros((960, 1280, 3), np.uint8))
        CianStep(self._remover(), 4, ThreadPoolExecutor(1)).process_batch([ctx], ["abc"])
        assert ctx.error and "listing ID" in ctx.error
