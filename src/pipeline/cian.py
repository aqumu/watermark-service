"""CianStep: remove the Cian watermark (logo + "ID <listing id>") with the ONNX model.

Unlike the Avito path there is no segmentation model: the mark's position follows
the image size and its content follows the listing ID, so each image needs its
listing ID. Preprocessing (mark localisation, hint rendering) runs per image on a
thread pool; the model runs on sub-batches.
"""

from concurrent.futures import Executor

import numpy as np

from src.cian.remover import CianRemover
from src.pipeline.context import ImageContext


class CianStep:
    def __init__(self, remover: CianRemover, batch_size: int, pool: Executor):
        self.remover = remover
        self.batch_size = max(1, batch_size)
        self.pool = pool

    def _prepare(self, ctx: ImageContext, listing_id: str):
        try:
            origin, x, _ = self.remover.prepare(ctx.original_bgr, listing_id)
            return ctx, origin, x
        except Exception as e:  # per-image error isolation, like the other steps
            ctx.error = f"cian preprocessing failed: {e}"
            return None

    def process_batch(self, contexts: list[ImageContext], listing_ids: list[str]) -> None:
        if len(contexts) != len(listing_ids):
            raise ValueError(f"{len(contexts)} images but {len(listing_ids)} listing IDs")
        jobs = []
        for ctx, lid in zip(contexts, listing_ids):
            if ctx.error is not None:
                continue
            lid = str(lid or "").strip()
            if not lid.isdigit():
                ctx.error = f"invalid listing ID {lid!r} (digits expected)"
                continue
            jobs.append((ctx, lid))
        prepared = [p for p in self.pool.map(lambda j: self._prepare(*j), jobs) if p is not None]
        for i in range(0, len(prepared), self.batch_size):
            chunk = prepared[i:i + self.batch_size]
            try:
                deltas = self.remover.run(np.stack([x for _, _, x in chunk]))
            except Exception as e:
                for ctx, _, _ in chunk:
                    ctx.error = f"cian model failed: {e}"
                continue
            for (ctx, origin, x), delta in zip(chunk, deltas):
                ctx.result_bgr = self.remover.paste(ctx.original_bgr, x, delta, origin)
