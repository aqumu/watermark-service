"""
Image processing utilities adapted from training/src/image_utils.py.

Simplified for deterministic inference (no augmentation, fixed dilation).
"""

import cv2
import numpy as np
import torch


def compute_gradient(bgr: np.ndarray) -> torch.Tensor:
    """
    Compute normalised grayscale Sobel gradient magnitude.
    uint8 BGR HxWx3 → float32 1xHxW in [0, 1]
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val
    return torch.from_numpy(mag).unsqueeze(0)  # 1xHxW


def dilate_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Dilate a binary mask with a fixed elliptical kernel.

    Ensures the true watermark edge is inside the mask boundary,
    preventing partially watermarked pixels from reaching the model
    as "clean" context.

    mask : HxW float32 in [0, 1]
    returns HxW float32 in [0, 1]
    """
    binary = (mask >= 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated.astype(np.float32)
