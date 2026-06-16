"""
WatermarkRemovalPipeline — orchestrates all steps with GPU batching.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch

from src.config import ServiceConfig
from src.models.checkpoint import load_checkpoint
from src.models.masked_unet import MaskedUNet, build_model
from src.models.seg_model import build_seg_model
from src.pipeline.alignment import AlignmentStep
from src.pipeline.blending import BlendingStep
from src.pipeline.context import ImageContext
from src.pipeline.inpaint import InpaintStep
from src.pipeline.removal import RemovalStep
from src.pipeline.segmentation import SegmentationStep
from src.pipeline.upscale import UpscaleStep
from src.processing.io import decode_image, encode_image

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class WatermarkRemovalPipeline:
    """Chains segmentation → alignment → removal → blending → upscale.

    Upscaling is applied to the final result (not the input) so the seg and
    removal models always see naturally-captured image texture.
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.device = _resolve_device(config.inference.device)
        logger.info("Using device: %s", self.device)

        # Build and load segmentation model
        seg_model = build_seg_model(encoder=config.model.seg_encoder)
        seg_model = load_checkpoint(seg_model, config.model.seg_checkpoint, self.device)

        # Build and load removal model
        removal_cfg = {
            "model": {
                "type": "scratch",
                "base_channels": config.model.removal_base_channels,
                "depth": config.model.removal_depth,
            }
        }
        removal_model = build_model(removal_cfg)
        removal_model = load_checkpoint(removal_model, config.model.removal_checkpoint, self.device)

        # Initialize pipeline steps
        amp = config.inference.amp and self.device.type == "cuda"
        self.seg_step = SegmentationStep(
            model=seg_model,
            image_size=config.model.seg_image_size,
            threshold=config.inference.mask_threshold,
            device=self.device,
            amp=amp,
        )
        self.alignment_step = AlignmentStep(
            watermark_path=config.model.watermark_path,
        )
        self.removal_step = RemovalStep(
            model=removal_model,
            image_width=config.model.removal_image_width,
            image_height=config.model.removal_image_height,
            crop_aspect_ratio=config.model.removal_crop_aspect_ratio,
            crop_margin_ratio=config.model.removal_crop_margin_ratio,
            crop_min_width_ratio=config.model.removal_crop_min_width_ratio,
            device=self.device,
            amp=amp,
        )
        self.blend_step = BlendingStep(
            feather_radius=config.inference.feather_radius,
            mask_expand=config.inference.mask_expand,
        )
        self.upscale_step = UpscaleStep(
            model_name=config.upscale.model_name,
            model_path=config.upscale.model_path if config.upscale.enabled else "",
            device=self.device,
            tile=config.upscale.tile,
            tile_pad=config.upscale.tile_pad,
            half=config.upscale.half,
            resolution_threshold=config.upscale.resolution_threshold,
        )

        # GPU steps process in sub-batches; CPU steps use thread pool
        self.max_batch_size = config.batch.max_batch_size
        self.io_pool = ThreadPoolExecutor(max_workers=config.batch.io_workers)

        self.inpaint_step: InpaintStep | None = None
        if config.new.enabled:
            self.inpaint_step = InpaintStep(
                method=config.new.method,
                inpaint_width=config.new.inpaint_width,
                inpaint_height=config.new.inpaint_height,
                inpaint_radius=config.new.inpaint_radius,
                inpaint_size=config.new.inpaint_size,
                checkpoint_path=config.new.checkpoint,
                device=self.device,
                amp=config.inference.amp,
            )

        logger.info("Pipeline ready (seg=%dx%d, removal=%dx%d, batch=%d, upscale_threshold=%dpx)",
                     config.model.seg_image_size, config.model.seg_image_size,
                     config.model.removal_image_width, config.model.removal_image_height,
                     self.max_batch_size, config.upscale.resolution_threshold)

    def _decode_to_context(self, image_bytes: bytes, image_id: str | None = None) -> ImageContext:
        """Decode raw bytes into an ImageContext."""
        if image_id is None:
            image_id = uuid.uuid4().hex[:12]
        try:
            bgr = decode_image(image_bytes)
            return ImageContext(image_id=image_id, original_bgr=bgr)
        except Exception as e:
            import numpy as np
            ctx = ImageContext(image_id=image_id, original_bgr=np.zeros((1, 1, 3), dtype=np.uint8))
            ctx.error = str(e)
            return ctx

    def _run_gpu_step(self, step, contexts: list[ImageContext]) -> None:
        """Run a GPU step in sub-batches of max_batch_size."""
        bs = self.max_batch_size
        for i in range(0, len(contexts), bs):
            chunk = contexts[i : i + bs]
            try:
                step.process_batch(chunk)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    logger.warning("OOM at batch size %d, retrying with size %d", len(chunk), len(chunk) // 2)
                    torch.cuda.empty_cache()
                    mid = len(chunk) // 2
                    step.process_batch(chunk[:mid])
                    step.process_batch(chunk[mid:])
                else:
                    for ctx in chunk:
                        ctx.error = str(e)

    def process_single(self, image_bytes: bytes, fmt: str = "png",
                       quality: int = 95, mode: str = "old") -> bytes:
        """Process one image synchronously. Returns encoded result bytes."""
        ctx = self._decode_to_context(image_bytes)
        if ctx.error:
            raise ValueError(ctx.error)

        if mode == "old":
            self.seg_step.process_batch([ctx])
            self.alignment_step.process_batch([ctx])
            self.removal_step.process_batch([ctx])
            self.blend_step.process_batch([ctx])
            self.upscale_step.process_batch([ctx])
        else:
            if self.inpaint_step is None:
                raise RuntimeError("New processing mode not enabled — set new.enabled=true in config")
            ctx.result_bgr = ctx.original_bgr.copy()
            self.upscale_step.process_batch([ctx])
            ctx.original_bgr = ctx.result_bgr.copy()
            self.inpaint_step.process_batch([ctx])

        if ctx.error:
            raise RuntimeError(ctx.error)
        return encode_image(ctx.result_bgr, fmt=fmt, quality=quality)

    def process_batch(
        self,
        images: list[bytes],
        progress_callback: Callable[[int, int], None] | None = None,
        fmt: str = "png",
        quality: int = 95,
        mode: str = "old",
    ) -> list[bytes | str]:
        """
        Process multiple images. Returns list of encoded bytes (success) or
        error strings (failure) in the same order as input.
        """
        total = len(images)

        # Decode in parallel (I/O bound)
        contexts = list(self.io_pool.map(
            lambda args: self._decode_to_context(args[1], image_id=f"img_{args[0]:04d}"),
            enumerate(images),
        ))

        if mode == "old":
            self._run_gpu_step(self.seg_step, contexts)

            list(self.io_pool.map(lambda ctx: self.alignment_step.process_batch([ctx]), contexts))
            if progress_callback:
                progress_callback(total // 4, total)

            self._run_gpu_step(self.removal_step, contexts)
            if progress_callback:
                progress_callback(2 * total // 4, total)

            list(self.io_pool.map(lambda ctx: self.blend_step.process_batch([ctx]), contexts))
            if progress_callback:
                progress_callback(3 * total // 4, total)

            self.upscale_step.process_batch(contexts)
            if progress_callback:
                progress_callback(total, total)
        else:
            if self.inpaint_step is None:
                raise RuntimeError("New processing mode not enabled — set new.enabled=true in config")
            for ctx in contexts:
                if ctx.error is None:
                    ctx.result_bgr = ctx.original_bgr.copy()
            self.upscale_step.process_batch(contexts)
            for ctx in contexts:
                if ctx.error is None and ctx.result_bgr is not None:
                    ctx.original_bgr = ctx.result_bgr.copy()
            self.inpaint_step.process_batch(contexts)
            if progress_callback:
                progress_callback(total, total)

        # Encode results in parallel
        def _encode(ctx: ImageContext) -> bytes | str:
            if ctx.error:
                return f"error: {ctx.error}"
            return encode_image(ctx.result_bgr, fmt=fmt, quality=quality)

        return list(self.io_pool.map(_encode, contexts))
