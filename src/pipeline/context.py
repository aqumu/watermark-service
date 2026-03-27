"""ImageContext — carries per-image state through the pipeline."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImageContext:
    """Mutable container that travels through every pipeline step."""

    image_id: str
    original_bgr: np.ndarray                          # original resolution
    original_size: tuple[int, int] = field(init=False) # (W, H)

    # filled by SegmentationStep
    mask: np.ndarray | None = None                     # uint8 HxW at original res

    # filled by RemovalStep
    model_pred_bgr: np.ndarray | None = None           # uint8 at model resolution

    # filled by BlendingStep
    result_bgr: np.ndarray | None = None               # uint8 at original res

    # per-image error isolation
    error: str | None = None

    def __post_init__(self):
        self.original_size = (self.original_bgr.shape[1], self.original_bgr.shape[0])
