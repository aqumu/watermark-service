"""
Masked U-Net for semi-transparent watermark removal
----------------------------------------------------
Input  : 5 channels  (RGB watermarked in [-1,1]  +
                      binary mask in {0,1} +
                      grayscale gradient magnitude in [0,1])
Output : 3 channels  (RGB residual delta in [0,1])
         pred_clean = watermarked_rgb − model_output   (clamped to [-1,1])

The network predicts how much white to subtract from each pixel rather than
reconstructing the clean image directly.  This is more robust:
  - Interior : learns a consistent small positive offset to remove
  - Edges    : learns to predict near-zero where watermark fades to nothing
  - Outside  : naturally outputs ≈0, leaving clean pixels untouched

Architecture
  Encoder : N blocks of  Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
  Bridge  : same double-conv without pooling
  Decoder : Upsample(nearest) → 3×3 Conv → concat(skip) → double-conv
  Head    : 1×1 Conv(C→3)

The mask is concatenated as the 4th input channel so the network can
distinguish masked (watermarked) pixels from clean context at every level.
The gradient magnitude is concatenated as the 5th channel to provide an
explicit structural signal of where brightness jumps (the watermark edge) occur.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# building blocks
# ──────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Residual Conv-BN-ReLU block to preserve high-frequency details."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Identity shortcut — helps gradients flow through sharp edges
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out += self.shortcut(x)
        return self.relu(out)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip   # (pooled, skip-connection)


class DecoderBlock(nn.Module):
    """Upsample via nearest-neighbor + 3×3 conv + DoubleConv"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        mid_ch = in_ch // 2
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(mid_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# U-Net
# ──────────────────────────────────────────────────────────────────────────────

class MaskedUNet(nn.Module):
    """
    Parameters
    ----------
    base_channels : feature width at the first encoder stage.
                    Subsequent stages double: base, 2×, 4×, 8×
    depth         : number of encoder/decoder stages (≥ 2, ≤ 5)
    in_channels   : 5  (RGB + mask + gradient)
    out_channels  : 3  (RGB)
    """

    def __init__(self, base_channels: int = 32, depth: int = 4,
                 in_channels: int = 5, out_channels: int = 3):
        super().__init__()
        assert 2 <= depth <= 5, "depth must be between 2 and 5"

        chs = [base_channels * (2 ** i) for i in range(depth)]

        # encoder
        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.encoders.append(EncoderBlock(prev, c))
            prev = c

        # bridge
        bridge_ch = chs[-1] * 2
        self.bridge = DoubleConv(prev, bridge_ch)
        prev = bridge_ch

        # decoder (reversed channel list)
        self.decoders = nn.ModuleList()
        for c in reversed(chs):
            self.decoders.append(DecoderBlock(prev, c, c))
            prev = c

        # output head — predicts residual delta
        self.head = nn.Conv2d(prev, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Identity initialization for the head:
        # By setting the last layer to zero, the model starts by outputting
        # exactement 0 residue (Tanh(0) = 0), which is a perfect identity
        # mapping. This prevents the "cyan/black" saturation at Epoch 1.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx5xHxW (RGB + mask + gradient) → out: Bx3xHxW"""
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bridge(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # Map raw conv output to [-2, 2] range.
        # This allows for a perfect 0 residue (identity) at tanh(0)
        # and covers the full possible dynamic range of RGB error.
        return 2.0 * torch.tanh(self.head(x))


# ──────────────────────────────────────────────────────────────────────────────
# convenience
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m.get("type", "scratch") == "pretrained":
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=m.get("encoder", "efficientnet-b0"),
            encoder_weights=m.get("encoder_weights", "imagenet"),
            in_channels=5,   # RGB + mask + gradient
            classes=3,       # residual delta
            activation="sigmoid",
        )
    return MaskedUNet(
        base_channels=m["base_channels"],
        depth=m["depth"],
    )
