"""Run one image through the service pipeline and save intermediate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.pipeline.pipeline import WatermarkRemovalPipeline
from src.processing.image_utils import crop_by_roi


def _save_image(path: Path, image: np.ndarray | None) -> None:
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image)


def _make_mask_overlay(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.35
    return (overlay.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def _draw_roi(image: np.ndarray, roi: dict | None) -> np.ndarray:
    canvas = image.copy()
    if roi is None:
        return canvas
    x0 = int(roi["x0"])
    y0 = int(roi["y0"])
    x1 = int(roi["x1"])
    y1 = int(roi["y1"])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return canvas


def _shape(image: np.ndarray | None) -> list[int] | None:
    return list(image.shape) if image is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Service config path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_out"),
        help="Directory to save debug artifacts",
    )
    args = parser.parse_args()

    cfg = load_config(str(args.config))
    pipeline = WatermarkRemovalPipeline(cfg)

    image_path = args.image.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = image_path.read_bytes()
    ctx = pipeline._decode_to_context(raw, image_id=image_path.stem)
    if ctx.error:
        raise RuntimeError(ctx.error)

    original_bgr = ctx.original_bgr.copy()
    _save_image(out_dir / "00_original.png", original_bgr)

    original_bgr_pre_upscale = ctx.original_bgr.copy()
    pipeline.upscale_step.process_batch([ctx])
    upscaled_bgr = ctx.original_bgr.copy()
    _save_image(out_dir / "01_upscaled_or_passthrough.png", upscaled_bgr)

    pipeline.seg_step.process_batch([ctx])
    _save_image(out_dir / "02_segmentation_mask.png", ctx.mask)
    _save_image(out_dir / "03_segmentation_overlay.png", _make_mask_overlay(upscaled_bgr, ctx.mask))

    pipeline.alignment_step.process_batch([ctx])
    _save_image(out_dir / "03b_aligned_mask.png", ctx.mask)
    _save_image(out_dir / "03c_aligned_mask_overlay.png", _make_mask_overlay(upscaled_bgr, ctx.mask))

    pipeline.removal_step.process_batch([ctx])
    _save_image(out_dir / "04_blend_mask.png", ctx.blend_mask)
    _save_image(out_dir / "05_blend_mask_overlay.png", _make_mask_overlay(upscaled_bgr, ctx.blend_mask))
    _save_image(out_dir / "06_roi_box.png", _draw_roi(upscaled_bgr, ctx.roi))
    _save_image(out_dir / "09_removal_pred_roi.png", ctx.model_pred_bgr)

    pipeline.blend_step.process_batch([ctx])
    _save_image(out_dir / "10_final_output.png", ctx.result_bgr)

    metadata = {
        "image_path": str(image_path),
        "config_path": str(args.config.resolve()),
        "device": str(pipeline.device),
        "segmentation_model_input_size": [cfg.model.seg_image_size, cfg.model.seg_image_size],
        "removal_model_input_size": [cfg.model.removal_image_width, cfg.model.removal_image_height],
        "removal_crop_aspect_ratio": cfg.model.removal_crop_aspect_ratio,
        "removal_crop_margin_ratio": cfg.model.removal_crop_margin_ratio,
        "removal_crop_min_width_ratio": cfg.model.removal_crop_min_width_ratio,
        "original_shape_hwc": _shape(original_bgr),
        "upscaled_shape_hwc": _shape(upscaled_bgr),
        "segmentation_mask_shape_hw": list(ctx.mask.shape) if ctx.mask is not None else None,
        "blend_mask_shape_hw": list(ctx.blend_mask.shape) if ctx.blend_mask is not None else None,
        "roi": ctx.roi,

        "removal_prediction_shape_hwc": _shape(ctx.model_pred_bgr),
        "final_output_shape_hwc": _shape(ctx.result_bgr),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved debug artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
