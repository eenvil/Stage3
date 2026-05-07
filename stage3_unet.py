from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g1 = min(groups, out_ch)
        while out_ch % g1 != 0 and g1 > 1:
            g1 -= 1

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResidualUNet(nn.Module):
    """
    Small residual U-Net.

    Input:
      x: [B, input_channels, H, W]
      composite: [B, 3, H, W]

    Output:
      pred = clamp(composite + residual, 0, 1)
    """

    def __init__(self, input_channels: int = 9, base_channels: int = 32, residual_scale: float = 0.25) -> None:
        super().__init__()
        c = base_channels
        self.residual_scale = residual_scale

        self.inc = ConvBlock(input_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.mid = ConvBlock(c * 8, c * 8)

        self.up3 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)

        self.out = nn.Conv2d(c, 3, kernel_size=1)

        # Start close to identity: composite + tiny correction.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, composite: torch.Tensor) -> torch.Tensor:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        z = self.down3(s3)

        z = self.mid(z)

        z = self.up3(z, s3)
        z = self.up2(z, s2)
        z = self.up1(z, s1)

        residual = torch.tanh(self.out(z)) * self.residual_scale
        pred = torch.clamp(composite + residual, 0.0, 1.0)
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
