"""Unit tests for pipeline components (no GPU / model weights required)."""

import numpy as np
import pytest

from src.pipeline.blending import BlendingStep
from src.pipeline.context import ImageContext
from src.pipeline.upscale import UpscaleStep
from src.processing.image_utils import compute_gradient, dilate_mask
from src.processing.io import decode_image, encode_image


class TestImageIO:
    def test_roundtrip_png(self, sample_bgr):
        encoded = encode_image(sample_bgr, fmt="png")
        decoded = decode_image(encoded)
        np.testing.assert_array_equal(sample_bgr, decoded)

    def test_roundtrip_jpeg(self):
        # Use a smooth gradient (not random noise) so JPEG compression is reasonable
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
        # Dilated mask should be larger
        assert dilated.sum() > mask.sum()
        assert dilated.dtype == np.float32

    def test_dilate_mask_empty(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        dilated = dilate_mask(mask, ksize=5)
        assert dilated.sum() == 0


class TestImageContext:
    def test_context_creation(self, sample_bgr):
        ctx = ImageContext(image_id="test", original_bgr=sample_bgr)
        assert ctx.original_size == (64, 64)
        assert ctx.mask is None
        assert ctx.error is None


class TestUpscaleStep:
    def test_passthrough(self, sample_bgr):
        ctx = ImageContext(image_id="test", original_bgr=sample_bgr)
        ctx.model_pred_bgr = sample_bgr
        UpscaleStep().process_batch([ctx])
        np.testing.assert_array_equal(ctx.upscaled_bgr, sample_bgr)

    def test_skips_errored(self, sample_bgr):
        ctx = ImageContext(image_id="test", original_bgr=sample_bgr)
        ctx.error = "previous step failed"
        UpscaleStep().process_batch([ctx])
        assert ctx.upscaled_bgr is None


class TestBlendingStep:
    def test_blend_identity(self):
        """If pred == original, blend should return the original."""
        bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        ctx = ImageContext(image_id="test", original_bgr=bgr)
        ctx.mask = mask
        ctx.model_pred_bgr = bgr.copy()
        ctx.upscaled_bgr = bgr.copy()

        BlendingStep(feather_radius=0, mask_expand=0).process_batch([ctx])
        np.testing.assert_array_equal(ctx.result_bgr, bgr)
