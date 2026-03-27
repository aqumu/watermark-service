"""Image I/O: decode bytes ↔ BGR numpy arrays."""

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (JPEG, PNG, WebP, etc.) to BGR uint8 ndarray."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image — unsupported format or corrupt data")
    return img


def encode_image(bgr: np.ndarray, fmt: str = "png", quality: int = 95) -> bytes:
    """Encode BGR uint8 ndarray to image bytes."""
    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif fmt == "webp":
        ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        ok, buf = cv2.imencode(".png", bgr)

    if not ok:
        raise RuntimeError(f"Failed to encode image as {fmt}")
    return buf.tobytes()
