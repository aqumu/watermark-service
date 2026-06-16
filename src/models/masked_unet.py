"""
U-Net for blind semi-transparent watermark removal
---------------------------------------------------
Input  : 3 channels  (RGB watermarked in [-1,1])
Output : 3 channels  (RGB residual delta in [-2, 2])
         pred_clean = watermarked_rgb − model_output   (clamped to [-1,1])

Uses GroupNorm instead of BatchNorm so that inference is stable regardless
of batch size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, num_channels), num_channels=num_channels)


# ──────────────────────────────────────────────────────────────────────────────
# building blocks
# ──────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Residual Conv-GN-ReLU block to preserve high-frequency details."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                _gn(out_ch)
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
            _gn(mid_ch),
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
    in_channels   : 3  (RGB)
    out_channels  : 3  (RGB)
    """

    def __init__(self, base_channels: int = 32, depth: int = 4,
                 in_channels: int = 3, out_channels: int = 3):
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
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Identity initialization for the head:
        # By setting the last layer to zero, the model starts by outputting
        # exactement 0 residue (Tanh(0) = 0), which is a perfect identity
        # mapping. This prevents the "cyan/black" saturation at Epoch 1.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx3xHxW (RGB) → out: Bx3xHxW"""
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
            in_channels=3,
            classes=3,
            activation="sigmoid",
        )
    return MaskedUNet(
        base_channels=m["base_channels"],
        depth=m["depth"],
    )
