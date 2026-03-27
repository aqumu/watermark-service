"""Shared test fixtures."""

import numpy as np
import pytest

from src.processing.io import encode_image


@pytest.fixture
def sample_image_bytes():
    """A small synthetic BGR image encoded as PNG bytes."""
    bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return encode_image(bgr, fmt="png")


@pytest.fixture
def sample_bgr():
    """A small synthetic BGR numpy array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
