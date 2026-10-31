"""Minimal shape-checking demo for the partial DMS-EI preview."""

from __future__ import annotations

import argparse

import torch

from dmsei_core import PartialDMSEICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    module = PartialDMSEICore(args.channels).to(device).eval()

    shape = (args.batch_size, args.channels, args.height, args.width)
    ir = torch.randn(shape, device=device)
    rgb = torch.randn(shape, device=device)

    with torch.inference_mode():
        ir_out, rgb_out, quality = module(ir, rgb)

    print(f"device:      {device}")
    print(f"IR input:    {tuple(ir.shape)}")
    print(f"RGB input:   {tuple(rgb.shape)}")
    print(f"IR output:   {tuple(ir_out.shape)}")
    print(f"RGB output:  {tuple(rgb_out.shape)}")
    print(f"quality:     {tuple(quality.shape)}")

    assert ir_out.shape == ir.shape
    assert rgb_out.shape == rgb.shape
    assert quality.shape == (args.batch_size, 2)


if __name__ == "__main__":
    main()

