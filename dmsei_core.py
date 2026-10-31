"""Partial public preview of the DMS-EI fusion module.

This file exposes only the quality-guided bidirectional interaction used by
DMS-EI. It intentionally omits the controlled-degradation training strategy,
quality-supervision loss, heterogeneous experts, detector integration, and the
complete multiscale training/evaluation pipeline. Consequently, this preview
cannot reproduce the results reported in the paper.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["PartialDMSEICore"]


class ConvBNAct(nn.Sequential):
    """A compact convolution, normalization, and activation block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class PartialDMSEICore(nn.Module):
    """Partial DMS-EI preview containing quality-guided bidirectional fusion.

    Args:
        channels: Channel count of each input modality.
        hidden_channels: Internal channel count. By default, one quarter of
            ``channels`` is used, with a minimum value of 16.

    Inputs:
        ir: Infrared feature tensor of shape ``[B, C, H, W]``.
        rgb: RGB feature tensor of shape ``[B, C, H, W]``.

    Returns:
        A tuple ``(ir_out, rgb_out, quality)``. The two output features retain
        the input shape, and ``quality`` contains the predicted infrared and
        RGB quality scores with shape ``[B, 2]``.
    """

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or max(channels // 4, 16)

        # The two modalities use independent projections.
        self.reduce_ir = ConvBNAct(channels, hidden)
        self.reduce_rgb = ConvBNAct(channels, hidden)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # This public preview uses task features and their absolute difference.
        descriptor_dim = 3 * hidden
        self.quality_head = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

        self.rgb_to_ir_gate = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ir_to_rgb_gate = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out_ir = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.out_rgb = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, ir: Tensor, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if ir.shape != rgb.shape:
            raise ValueError(
                "IR and RGB features must have identical shapes, "
                f"but received {tuple(ir.shape)} and {tuple(rgb.shape)}."
            )

        z_ir = self.reduce_ir(ir)
        z_rgb = self.reduce_rgb(rgb)
        descriptor = torch.cat(
            (
                self.pool(z_ir).flatten(1),
                self.pool(z_rgb).flatten(1),
                self.pool((z_ir - z_rgb).abs()).flatten(1),
            ),
            dim=1,
        )

        quality = self.quality_head(descriptor)
        q_ir = quality[:, 0:1, None, None]
        q_rgb = quality[:, 1:2, None, None]

        ir_update = z_ir + q_rgb * self.rgb_to_ir_gate(z_rgb) * z_rgb
        rgb_update = z_rgb + q_ir * self.ir_to_rgb_gate(z_ir) * z_ir

        ir_out = ir + self.out_ir(ir_update)
        rgb_out = rgb + self.out_rgb(rgb_update)
        return ir_out, rgb_out, quality
