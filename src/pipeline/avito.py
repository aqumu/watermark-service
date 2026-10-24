"""AvitoStep: remove the small Avito mark with the ONNX model.

The mark sits at a fixed position relative to the bottom-right corner, so there is
no segmentation or localisation: each image's corner window goes straight to the
model (sub-batches). Images smaller than the window are returned unchanged.
"""

from concurrent.futures import Executor

import numpy as np

from src.avito import AvitoRemover
from src.pipeline.context import ImageContext


class AvitoStep:
    def __init__(self, remover: AvitoRemover, batch_size: int, pool: Executor):
        self.remover = remover
        self.batch_size = max(1, batch_size)
        self.pool = pool

    def _prepare(self, ctx: ImageContext):
        try:
            prepared = self.remover.prepare(ctx.original_bgr)
        except Exception as e:  # per-image error isolation, like the other steps
            ctx.error = f"avito preprocessing failed: {e}"
            return None
        if prepared is None:
            ctx.result_bgr = ctx.original_bgr.copy()
            return None
        return ctx, *prepared

    def process_batch(self, contexts: list[ImageContext]) -> None:
        jobs = [ctx for ctx in contexts if ctx.error is None]
        prepared = [p for p in self.pool.map(self._prepare, jobs) if p is not None]
        for i in range(0, len(prepared), self.batch_size):
            chunk = prepared[i:i + self.batch_size]
            try:
                deltas = self.remover.run(np.stack([x for _, _, x in chunk]))
            except Exception as e:
                for ctx, _, _ in chunk:
                    ctx.error = f"avito model failed: {e}"
                continue
            for (ctx, origin, x), delta in zip(chunk, deltas):
                ctx.result_bgr = self.remover.paste(ctx.original_bgr, x, delta, origin)
