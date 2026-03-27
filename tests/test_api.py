"""API integration tests using httpx (no GPU / model weights required).

These tests mock the pipeline to avoid needing actual model checkpoints.
"""

import io
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.processing.io import encode_image


def _make_test_app(mock_pipeline):
    """Create a minimal FastAPI app with mocked pipeline dependency."""
    from src.api.dependencies import get_pipeline

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    return app


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline that returns a white image."""
    pipeline = MagicMock()
    result_bgr = np.full((64, 64, 3), 255, dtype=np.uint8)
    result_bytes = encode_image(result_bgr, fmt="png")
    pipeline.process_single.return_value = result_bytes
    pipeline.process_batch.return_value = [result_bytes, result_bytes]
    pipeline.device = MagicMock()
    pipeline.device.type = "cpu"
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """TestClient with mocked pipeline (no lifespan, no real model loading)."""
    app = _make_test_app(mock_pipeline)
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


class TestProcessEndpoint:
    def test_single_image(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/process",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_single_image_error(self, client, mock_pipeline, sample_image_bytes):
        mock_pipeline.process_single.side_effect = ValueError("bad image")
        resp = client.post(
            "/api/v1/process",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 400


class TestBatchEndpoint:
    def test_batch_returns_zip(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/process/batch",
            files=[
                ("images", ("a.png", sample_image_bytes, "image/png")),
                ("images", ("b.png", sample_image_bytes, "image/png")),
            ],
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        assert len(zf.namelist()) == 2
