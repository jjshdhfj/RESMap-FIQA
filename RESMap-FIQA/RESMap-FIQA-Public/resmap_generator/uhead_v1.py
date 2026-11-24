from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    """Choose a GroupNorm configuration that works for small batch sizes."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UncertaintyHeadV1(nn.Module):
    """Post-hoc V1 uncertainty head.

    Input (8 channels):
        [low-quality RGB, enhanced RGB, luminance response, fundus mask]

    Output:
        one-channel bounded log-scale s = log(b).

    All RGB tensors are expected in [0, 1]. With the defaults, the predicted
    Laplace scale b=exp(s) is bounded to approximately [9.1e-4, 1.0].
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        log_scale_min: float = -7.0,
        log_scale_max: float = 0.0,
    ) -> None:
        super().__init__()
        if log_scale_min >= log_scale_max:
            raise ValueError("log_scale_min must be smaller than log_scale_max")

        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.enc4 = DownBlock(c * 4, c * 8)
        self.bottleneck = DownBlock(c * 8, c * 16)

        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)
        self.out_conv = nn.Conv2d(c, 1, 1)

        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.bottleneck(e4)

        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        raw = self.out_conv(d1)
        return self.log_scale_min + (
            self.log_scale_max - self.log_scale_min
        ) * torch.sigmoid(raw)


def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [B,3,H,W] in [0,1] to one-channel luminance."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor [B,3,H,W], got {tuple(rgb.shape)}")
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def build_uhead_input_v1(
    low_quality_01: torch.Tensor,
    enhanced_01: torch.Tensor,
    mask: torch.Tensor,
    response_luma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the exact 8-channel V1 input.

    When ``response_luma`` is supplied, it must be the precomputed fixed-scale
    map ``abs(Y(enhanced)-Y(low_quality))`` in [0,1]. Otherwise it is computed
    online from the two RGB tensors.
    """
    if low_quality_01.ndim != 4 or low_quality_01.shape[1] != 3:
        raise ValueError(
            f"Expected low-quality RGB [B,3,H,W], got {tuple(low_quality_01.shape)}"
        )
    if enhanced_01.shape != low_quality_01.shape:
        raise ValueError(
            "low_quality_01 and enhanced_01 must have identical shapes; "
            f"got {low_quality_01.shape} and {enhanced_01.shape}"
        )
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask [B,1,H,W], got {tuple(mask.shape)}")
    if mask.shape[0] != low_quality_01.shape[0] or mask.shape[-2:] != low_quality_01.shape[-2:]:
        raise ValueError("Mask batch/spatial shape does not match RGB tensors")

    mask = (mask > 0.5).to(low_quality_01.dtype)
    low_quality_01 = low_quality_01 * mask
    enhanced_01 = enhanced_01 * mask

    if response_luma is None:
        response_luma = torch.abs(
            rgb_to_luminance(enhanced_01) - rgb_to_luminance(low_quality_01)
        )
    else:
        if response_luma.ndim != 4 or response_luma.shape[1] != 1:
            raise ValueError(
                f"Expected response_luma [B,1,H,W], got {tuple(response_luma.shape)}"
            )
        if response_luma.shape[0] != mask.shape[0] or response_luma.shape[-2:] != mask.shape[-2:]:
            raise ValueError("response_luma batch/spatial shape does not match mask")
        response_luma = response_luma.to(low_quality_01.dtype)

    response_luma = response_luma.clamp(0.0, 1.0) * mask
    return torch.cat([low_quality_01, enhanced_01, response_luma, mask], dim=1)


def erode_binary_mask(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Binary erosion implemented with max-pooling on the inverse mask."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask [B,1,H,W], got {tuple(mask.shape)}")
    if kernel_size <= 1:
        return (mask > 0.5).to(mask.dtype)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    binary = (mask > 0.5).to(mask.dtype)
    pad = kernel_size // 2
    return 1.0 - F.max_pool2d(1.0 - binary, kernel_size, stride=1, padding=pad)


def restoration_error_map(
    enhanced_01: torch.Tensor,
    high_quality_01: torch.Tensor,
) -> torch.Tensor:
    """Luminance restoration error ``abs(Y(enhanced)-Y(HQ))``.

    This function is kept for synthetic diagnostics. Offline training should
    load the precomputed uint16 luminance-error target instead of recomputing it.
    """
    if enhanced_01.shape != high_quality_01.shape:
        raise ValueError(
            "enhanced_01 and high_quality_01 must have identical shapes; "
            f"got {enhanced_01.shape} and {high_quality_01.shape}"
        )
    return torch.abs(
        rgb_to_luminance(enhanced_01) - rgb_to_luminance(high_quality_01)
    )


def masked_laplace_nll(
    log_scale: torch.Tensor,
    error_map: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked heteroscedastic Laplace NLL.

    L = sum M * [e * exp(-s) + s] / (sum M + eps)

    The additive constant log(2) is omitted because it does not affect
    optimization. The resulting scalar may therefore be negative.
    """
    if not (log_scale.shape == error_map.shape == valid_mask.shape):
        raise ValueError(
            "log_scale, error_map and valid_mask must have identical shapes; "
            f"got {log_scale.shape}, {error_map.shape}, {valid_mask.shape}"
        )
    loss_map = error_map * torch.exp(-log_scale) + log_scale
    return (loss_map * valid_mask).sum() / (valid_mask.sum() + eps)
