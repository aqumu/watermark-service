"""Image processing utilities shared across service inference steps."""

import cv2
import numpy as np
import torch


def _resize_to_shape(image: np.ndarray, width: int, height: int, is_mask: bool = False) -> np.ndarray:
    """Resize an image or mask with interpolation chosen by direction."""
    h, w = image.shape[:2]
    if h == height and w == width:
        return image.copy()

    shrinking = h > height or w > width
    if is_mask:
        interp = cv2.INTER_LINEAR if image.dtype != np.uint8 else cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interp)


def _crop_with_reflect_padding(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Crop a possibly out-of-bounds rectangle by reflect-padding the source first."""
    h, w = image.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    x0 += pad_left
    x1 += pad_left
    y0 += pad_top
    y1 += pad_top
    return image[y0:y1, x0:x1]


def make_fixed_aspect_crop(
    mask: np.ndarray,
    crop_aspect_ratio: float,
    margin_ratio: float = 0.10,
    min_width_ratio: float = 0.50,
) -> dict:
    """Build a fixed-aspect crop centered on the watermark mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    if crop_aspect_ratio <= 0:
        raise ValueError("crop_aspect_ratio must be > 0")

    h, w = mask.shape
    binary = (mask > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        crop_w = max(1, int(round(w * min_width_ratio)))
        crop_h = max(1, int(round(crop_w / crop_aspect_ratio)))
        cx = w / 2.0
        cy = h / 2.0
    else:
        x_min, x_max = xs.min(), xs.max() + 1
        y_min, y_max = ys.min(), ys.max() + 1
        box_w = x_max - x_min
        box_h = y_max - y_min

        margin_x = int(round(box_w * margin_ratio))
        margin_y = int(round(box_h * margin_ratio))
        box_w += 2 * margin_x
        box_h += 2 * margin_y

        crop_w = max(box_w, int(round(box_h * crop_aspect_ratio)))
        crop_h = max(box_h, int(round(crop_w / crop_aspect_ratio)))
        crop_w = max(crop_w, int(round(w * min_width_ratio)))
        crop_h = max(crop_h, int(round(crop_w / crop_aspect_ratio)))
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

    crop_w = int(crop_w)
    crop_h = int(crop_h)
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    return {
        "x0": x0,
        "y0": y0,
        "width": crop_w,
        "height": crop_h,
        "x1": x0 + crop_w,
        "y1": y0 + crop_h,
    }


def crop_removal_roi(
    wm: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
    crop_aspect_ratio: float = 3.54,
    margin_ratio: float = 0.10,
    min_width_ratio: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """Exact ROI extraction contract used by removal training/inference.

    Critical — must match ``wm_shared/preprocess.crop_removal_roi`` exactly:
      - crop_mask = mask (no dilation, raw mask used for bounding box)
      - mask resized with is_mask=True (mask-aware interpolation)
      - normalization guard uses > 1.01 (not 1.0)
    """
    crop_mask = mask  # no dilation — matches training repo
    roi = make_fixed_aspect_crop(
        crop_mask,
        crop_aspect_ratio=crop_aspect_ratio,
        margin_ratio=margin_ratio,
        min_width_ratio=min_width_ratio,
    )
    x0 = int(roi["x0"])
    y0 = int(roi["y0"])
    x1 = int(roi["x1"])
    y1 = int(roi["y1"])

    wm_crop = _crop_with_reflect_padding(wm, x0, y0, x1, y1)
    clean_crop = _crop_with_reflect_padding(clean, x0, y0, x1, y1)
    mask_crop = _crop_with_reflect_padding(mask, x0, y0, x1, y1)
    crop_mask_crop = _crop_with_reflect_padding(crop_mask, x0, y0, x1, y1)

    wm_sq = _resize_to_shape(wm_crop, width, height, is_mask=False)
    clean_sq = _resize_to_shape(clean_crop, width, height, is_mask=False)
    mask_sq = _resize_to_shape(mask_crop, width, height, is_mask=True).astype(np.float32)
    crop_mask_sq = _resize_to_shape(crop_mask_crop, width, height, is_mask=True).astype(np.float32)

    if mask_sq.max() > 1.01:
        mask_sq /= 255.0
    if crop_mask_sq.max() > 1.01:
        crop_mask_sq /= 255.0

    return wm_sq, clean_sq, mask_sq, roi, crop_mask_sq


def crop_by_roi(
    image: np.ndarray,
    roi: dict,
    width: int,
    height: int,
    is_mask: bool = False,
) -> np.ndarray:
    """Extract an explicit ROI, pad if needed, then resize to the target shape."""
    x0 = int(roi["x0"])
    y0 = int(roi["y0"])
    crop_w = int(roi["width"])
    crop_h = int(roi["height"])
    crop = _crop_with_reflect_padding(image, x0, y0, x0 + crop_w, y0 + crop_h)
    return _resize_to_shape(crop, width, height, is_mask=is_mask)


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


def dilate_mask_input(mask: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Create the deterministic binary mask hint fed into removal inference."""
    del image_size
    binary = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary.astype(np.float32)


def dilate_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Dilate a binary mask with a fixed elliptical kernel.

    Ensures the true watermark edge is inside the mask boundary,
    preventing partially watermarked pixels from reaching the model
    as "clean" context.

    mask : HxW float32 in [0, 1]
    returns HxW float32 in [0, 1]
    """
    binary = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated.astype(np.float32)
