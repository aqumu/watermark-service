"""Run one synthetic dataset sample through the service removal path using the ground-truth mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.pipeline.context import ImageContext
from src.pipeline.pipeline import WatermarkRemovalPipeline


def _save(path: Path, image: np.ndarray | None) -> None:
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return image


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return mask


def _overlay(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.35
    return (image.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def _draw_roi(image: np.ndarray, roi: dict | None) -> np.ndarray:
    out = image.copy()
    if roi is None:
        return out
    cv2.rectangle(out, (int(roi["x0"]), int(roi["y0"])), (int(roi["x1"]), int(roi["y1"])), (0, 255, 255), 2)
    return out


def _shape(image: np.ndarray | None) -> list[int] | None:
    return None if image is None else list(image.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path, help="Synthetic dataset sample dir with watermarked.jpg/clean.png/mask.png")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("debug_dataset_pair"))
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    wm_path = sample_dir / "watermarked.jpg"
    clean_path = sample_dir / "clean.png"
    mask_path = sample_dir / "mask.png"

    watermarked = _read_bgr(wm_path)
    clean = _read_bgr(clean_path)
    mask = _read_mask(mask_path)

    cfg = load_config(str(args.config))
    pipeline = WatermarkRemovalPipeline(cfg)

    ctx = ImageContext(image_id=sample_dir.name, original_bgr=watermarked.copy())
    ctx.mask = mask.copy()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _save(out_dir / "00_watermarked.png", watermarked)
    _save(out_dir / "01_clean_gt.png", clean)
    _save(out_dir / "02_gt_mask.png", mask)
    _save(out_dir / "03_gt_mask_overlay.png", _overlay(watermarked, mask))

    pipeline.removal_step.process_batch([ctx])
    _save(out_dir / "04_blend_mask.png", ctx.blend_mask)
    _save(out_dir / "05_blend_mask_overlay.png", _overlay(watermarked, ctx.blend_mask))
    _save(out_dir / "06_roi_box.png", _draw_roi(watermarked, ctx.roi))
    _save(out_dir / "09_removal_pred_roi.png", ctx.model_pred_bgr)

    pipeline.blend_step.process_batch([ctx])
    _save(out_dir / "10_final_service_output.png", ctx.result_bgr)

    if ctx.result_bgr is not None:
        abs_diff = np.abs(ctx.result_bgr.astype(np.int16) - clean.astype(np.int16)).astype(np.uint8)
        mae_map = abs_diff.mean(axis=2).astype(np.uint8)
        _save(out_dir / "11_error_map.png", cv2.applyColorMap(mae_map, cv2.COLORMAP_JET))

    metadata = {
        "sample_dir": str(sample_dir),
        "config_path": str(args.config.resolve()),
        "device": str(pipeline.device),
        "removal_model_input_size": [cfg.model.removal_image_width, cfg.model.removal_image_height],
        "original_shape_hwc": _shape(watermarked),
        "clean_shape_hwc": _shape(clean),
        "mask_shape_hw": list(mask.shape),
        "blend_mask_shape_hw": None if ctx.blend_mask is None else list(ctx.blend_mask.shape),
        "roi": ctx.roi,
        "removal_prediction_shape_hwc": _shape(ctx.model_pred_bgr),
        "final_output_shape_hwc": _shape(ctx.result_bgr),
        "final_vs_clean_mae": None if ctx.result_bgr is None else float(np.abs(ctx.result_bgr.astype(np.float32) - clean.astype(np.float32)).mean()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved debug artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
