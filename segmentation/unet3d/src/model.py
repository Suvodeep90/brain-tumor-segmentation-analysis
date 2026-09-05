"""
3D U-Net architecture for brain tumor segmentation.

Input:  (B, 4, D, H, W)  — FLAIR, T1, T1CE, T2
Output: (B, 3, D, H, W)  — WT, TC, ET binary masks (raw logits)

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015 — adapted to 3-D with GroupNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3D → GroupNorm → ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """MaxPool3d(2) → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DecoderBlock(nn.Module):
    """Trilinear upsample → pad → cat skip → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = True):
        super().__init__()
        self.up = (
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            if trilinear
            else nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad to match skip-connection spatial dims
        dZ = skip.size(2) - x.size(2)
        dY = skip.size(3) - x.size(3)
        dX = skip.size(4) - x.size(4)
        x = F.pad(x, [dX // 2, dX - dX // 2,
                       dY // 2, dY - dY // 2,
                       dZ // 2, dZ - dZ // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3d(nn.Module):
    """
    Symmetric 3-D U-Net with 4 encoding and 4 decoding stages.

    Args:
        in_channels:  number of input modalities (4 for BraTS)
        n_classes:    number of output segmentation classes (3 for WT/TC/ET)
        n_channels:   base channel width (doubled at each encoder stage)
    """

    def __init__(self, in_channels: int = 4, n_classes: int = 3, n_channels: int = 24):
        super().__init__()
        C = n_channels

        # Encoder
        self.input_conv = DoubleConv(in_channels, C)
        self.enc1 = EncoderBlock(C,     2 * C)
        self.enc2 = EncoderBlock(2 * C, 4 * C)
        self.enc3 = EncoderBlock(4 * C, 8 * C)
        self.enc4 = EncoderBlock(8 * C, 8 * C)   # bottleneck

        # Decoder
        self.dec1 = DecoderBlock(16 * C, 4 * C)
        self.dec2 = DecoderBlock(8  * C, 2 * C)
        self.dec3 = DecoderBlock(4  * C, C)
        self.dec4 = DecoderBlock(2  * C, C)

        # Output projection
        self.out_conv = nn.Conv3d(C, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        s1 = self.input_conv(x)
        s2 = self.enc1(s1)
        s3 = self.enc2(s2)
        s4 = self.enc3(s3)
        bottleneck = self.enc4(s4)

        # Decoder path with skip connections
        out = self.dec1(bottleneck, s4)
        out = self.dec2(out, s3)
        out = self.dec3(out, s2)
        out = self.dec4(out, s1)
        return self.out_conv(out)


def build_model(in_channels: int = 4, n_classes: int = 3,
                n_channels: int = 24) -> UNet3d:
    model = UNet3d(in_channels, n_classes, n_channels)
    total = sum(p.numel() for p in model.parameters())
    print(f"UNet3d | params: {total:,} | in={in_channels} → out={n_classes} | base_ch={n_channels}")
    return model
