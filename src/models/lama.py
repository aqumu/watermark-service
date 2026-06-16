"""LaMa (FFCResNetGenerator) model for inference."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_activation(kind: str | bool = "tanh") -> nn.Module:
    if kind == "tanh":
        return nn.Tanh()
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f"Unknown activation kind {kind}")


class SELayer(nn.Module):
    """Minimal SE layer — only needed if use_se=True (not used by default)."""

    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__()
        r = 16
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // r, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x


class LearnableSpatialTransformWrapper(nn.Module):
    """Stub — only used when spatial_transform_layers is set (not by default)."""

    def __init__(self, module: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.module(x)


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        spatial_scale_factor: float | None = None,
        spatial_scale_mode: str = "bilinear",
        spectral_pos_encoding: bool = False,
        use_se: bool = False,
        se_kwargs: dict | None = None,
        ffc3d: bool = False,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        self.groups = groups

        conv_in = in_channels * 2 + (2 if spectral_pos_encoding else 0)
        self.conv_layer = nn.Conv2d(
            in_channels=conv_in,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se = SELayer(self.conv_layer.in_channels, **(se_kwargs or {}))

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        enable_lfu: bool = True,
        **fu_kwargs: Any,
    ) -> None:
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()
        self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, : c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        enable_lfu: bool = True,
        padding_type: str = "reflect",
        gated: bool = False,
        **spectral_kwargs: Any,
    ) -> None:
        super().__init__()
        assert stride in (1, 2), "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        def _conv(in_ch: int, out_ch: int) -> nn.Module:
            if in_ch == 0 or out_ch == 0:
                return nn.Identity()
            return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)

        self.convl2l = _conv(in_cl, out_cl)
        self.convl2g = _conv(in_cl, out_cg)
        self.convg2l = _conv(in_cg, out_cl)

        self.convg2g = nn.Identity()
        if in_cg != 0 and out_cg != 0:
            self.convg2g = SpectralTransform(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        self.gate = nn.Identity()
        if gated and in_cg != 0 and out_cl != 0:
            self.gate = nn.Conv2d(in_channels, 2, 1)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl: torch.Tensor = 0
        out_xg: torch.Tensor = 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.Identity,
        padding_type: str = "reflect",
        enable_lfu: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.ffc = FFC(
            in_channels, out_channels, kernel_size,
            ratio_gin, ratio_gout, stride, padding, dilation,
            groups, bias, enable_lfu, padding_type=padding_type, **kwargs,
        )

        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.Identity() if ratio_gout == 1 else norm_layer(out_channels - global_channels)
        self.bn_g = nn.Identity() if ratio_gout == 0 else norm_layer(global_channels)
        self.act_l = nn.Identity() if ratio_gout == 1 else activation_layer(inplace=True)
        self.act_g = nn.Identity() if ratio_gout == 0 else activation_layer(inplace=True)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.ReLU,
        dilation: int = 1,
        spatial_transform_kwargs: dict | None = None,
        inline: bool = False,
        **conv_kwargs: Any,
    ) -> None:
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.inline:
            x_l, x_g = x[:, : -self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        elif isinstance(x, tuple):
            x_l, x_g = x
        else:
            x_l, x_g = x, 0

        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g

        if self.inline:
            return torch.cat((x_l, x_g), dim=1)
        return x_l, x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_l, x_g = x
        if torch.is_tensor(x_l) and torch.is_tensor(x_g):
            return torch.cat((x_l, x_g), dim=1)
        return x_l


class FFCResNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 4,
        output_nc: int = 3,
        ngf: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 9,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        padding_type: str = "reflect",
        activation_layer: type[nn.Module] = nn.ReLU,
        init_conv_kwargs: dict | None = None,
        downsample_conv_kwargs: dict | None = None,
        resnet_conv_kwargs: dict | None = None,
        add_out_act: str | bool = True,
        max_features: int = 1024,
    ) -> None:
        super().__init__()
        init_conv_kwargs = init_conv_kwargs or {}
        downsample_conv_kwargs = downsample_conv_kwargs or {}
        resnet_conv_kwargs = resnet_conv_kwargs or {}

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0,
                       norm_layer=norm_layer, activation_layer=activation_layer, **init_conv_kwargs),
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model.append(
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, ngf * mult * 2),
                    kernel_size=3, stride=2, padding=1,
                    norm_layer=norm_layer, activation_layer=activation_layer,
                    **cur_conv_kwargs,
                )
            )

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        for _ in range(n_blocks):
            model.append(
                FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                               activation_layer=activation_layer, norm_layer=norm_layer,
                               **resnet_conv_kwargs)
            )

        model.append(ConcatTupleLayer())

        for i in range(n_downsampling):
            mult_val = 2 ** (n_downsampling - i)
            model.append(
                nn.ConvTranspose2d(
                    min(max_features, ngf * mult_val),
                    min(max_features, int(ngf * mult_val / 2)),
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                )
            )
            model.append(norm_layer(min(max_features, int(ngf * mult_val / 2))))
            model.append(activation_layer(inplace=True))

        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))
        if add_out_act:
            model.append(get_activation("tanh" if add_out_act is True else add_out_act))

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_lama_model(checkpoint_path: str, device: torch.device) -> FFCResNetGenerator:
    model = FFCResNetGenerator(
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        downsample_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        resnet_conv_kwargs={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
        add_out_act="sigmoid",
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    gen_sd = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            gen_sd[k[len("generator."):]] = v

    missing, unexpected = model.load_state_dict(gen_sd, strict=False)
    if missing:
        logger.warning("LaMa missing keys: %s", missing)
    if unexpected:
        logger.warning("LaMa unexpected keys: %s", unexpected)

    model.eval()
    model = model.to(device)
    logger.info("LaMa model loaded from %s onto %s", checkpoint_path, device)
    return model
