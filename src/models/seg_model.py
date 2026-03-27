"""
Segmentation model for watermark mask prediction
-------------------------------------------------
Input  : 3-ch RGB tensor in [-1, 1]
Output : 1-ch soft mask logits (apply sigmoid for probabilities)

Uses a pretrained EfficientNet-B0 encoder (ImageNet weights) via
segmentation-models-pytorch.  The pretrained encoder already learns
rich semantic features that distinguish logos/text from natural textures,
which is the primary failure mode of the custom U-Net.

The explicit Sobel gradient channel is dropped — pretrained conv layers
detect edges implicitly and more accurately.
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


def build_seg_model(encoder: str = "efficientnet-b0") -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
