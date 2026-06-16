"""Unit tests for pipeline components that do not require real model weights."""

import numpy as np
import pytest

from src.pipeline.blending import BlendingStep
from src.pipeline.context import ImageContext
from src.pipeline.upscale import UpscaleStep
from src.processing.image_utils import (
    compute_gradient,
    dilate_mask,
    dilate_mask_input,
    make_fixed_aspect_crop,
)
from src.processing.io import decode_image, encode_image


class TestImageIO:
    def test_roundtrip_png(self, sample_bgr):
        encoded = encode_image(sample_bgr, fmt="png")
        decoded = decode_image(encoded)
        np.testing.assert_array_equal(sample_bgr, decoded)

    def test_roundtrip_jpeg(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = np.arange(64).reshape(1, 64).repeat(64, axis=0)
        bgr[:, :, 1] = 128
        bgr[:, :, 2] = 200
        encoded = encode_image(bgr, fmt="jpeg", quality=100)
        decoded = decode_image(encoded)
        assert np.abs(bgr.astype(int) - decoded.astype(int)).mean() < 5

    def test_decode_invalid(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_image(b"not an image")


class TestImageUtils:
    def test_compute_gradient_shape(self, sample_bgr):
        grad = compute_gradient(sample_bgr)
        assert grad.shape == (1, 64, 64)
        assert grad.min() >= 0
        assert grad.max() <= 1

    def test_dilate_mask(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        dilated = dilate_mask(mask, ksize=5)
        assert dilated.sum() > mask.sum()
        assert dilated.dtype == np.float32

    def test_dilate_mask_input(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        dilated = dilate_mask_input(mask)
        assert dilated.sum() > mask.sum()
        assert dilated.dtype == np.float32

    def test_fixed_aspect_crop_respects_ratio(self):
        mask = np.zeros((120, 240), dtype=np.float32)
        mask[30:60, 120:180] = 1.0
        roi = make_fixed_aspect_crop(mask, crop_aspect_ratio=4.0, margin_ratio=0.0, min_width_ratio=0.25)

        assert roi["width"] >= 60
        assert roi["height"] >= 15
        assert abs((roi["width"] / roi["height"]) - 4.0) < 0.15


class TestImageContext:
    def test_context_creation(self, sample_bgr):
        ctx = ImageContext(image_id="test", original_bgr=sample_bgr)
        assert ctx.original_size == (64, 64)
        assert ctx.mask is None
        assert ctx.roi is None
        assert ctx.error is None


class TestUpscaleStep:
    def test_passthrough_without_model(self, sample_bgr):
        ctx = ImageContext(image_id="test", original_bgr=sample_bgr.copy())
        UpscaleStep(model_path="").process_batch([ctx])
        np.testing.assert_array_equal(ctx.original_bgr, sample_bgr)

    def test_skips_errored(self, sample_bgr):
        ctx = ImageContext(image_id="test", original_bgr=sample_bgr.copy())
        ctx.error = "previous step failed"
        UpscaleStep(model_path="").process_batch([ctx])
        np.testing.assert_array_equal(ctx.original_bgr, sample_bgr)


class TestBlendingStep:
    def test_blend_identity(self):
        bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        ctx = ImageContext(image_id="test", original_bgr=bgr)
        ctx.blend_mask = mask
        ctx.roi = {"x0": 0, "y0": 0, "width": 100, "height": 100}
        ctx.model_pred_bgr = bgr.copy()

        BlendingStep(feather_radius=0, mask_expand=0).process_batch([ctx])
        np.testing.assert_array_equal(ctx.result_bgr, bgr)

    def test_blend_back_only_updates_roi(self):
        orig = np.zeros((80, 120, 3), dtype=np.uint8)
        pred = np.full((20, 40, 3), 255, dtype=np.uint8)
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:40, 30:70] = 255
        roi = {"x0": 30, "y0": 20, "width": 40, "height": 20}

        out = BlendingStep._blend_back(pred, orig, mask, roi, feather=0, mask_expand=0)

        assert np.all(out[:20] == 0)
        assert np.all(out[:, :30] == 0)
        assert np.all(out[20:40, 30:70] == 255)
