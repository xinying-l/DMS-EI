# DMS-EI

This repository provides an **initial research release** of DMS-EI,
a degradation-supervised multimodal state-guided expert interaction network for
RGB--infrared object detection.

## Current Release Scope

The current release presents selected components of the quality-guided
bidirectional interaction design, including:

- modality-specific feature projection;
- feature-based multimodal state description;
- modality-quality estimation;
- quality-guided RGB-to-infrared and infrared-to-RGB feature injection; and
- shape-preserving residual outputs.

This initial release provides selected components of the proposed method for
early research access. The complete implementation, including the full model,
training and evaluation pipelines, experimental configurations, and supporting
resources, will be released after the paper is accepted and the required
institutional review is completed.

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later

## Repository Contents

```text
.
|-- dmsei_core.py       # Interaction-component implementation
|-- demo_forward.py     # Random-tensor shape check
|-- example_config.yaml # Representative feature-shape example
|-- requirements.txt
`-- README.md
```

The preview accepts paired infrared and RGB feature tensors with identical
dimensions. The two modalities are projected independently, summarized into a
compact feature descriptor, and updated through quality-guided directional
interaction. The two outputs retain the original feature dimensions.

## Minimal Usage

```python
import torch

from dmsei_core import PartialDMSEICore

module = PartialDMSEICore(channels=64)
ir = torch.randn(1, 64, 80, 80)
rgb = torch.randn(1, 64, 80, 80)
ir_out, rgb_out, quality = module(ir, rgb)

print(ir_out.shape, rgb_out.shape, quality.shape)
```

Expected output shapes are:

```text
torch.Size([1, 64, 80, 80])
torch.Size([1, 64, 80, 80])
torch.Size([1, 2])
```

The same check can be run from the command line:

```bash
python demo_forward.py --channels 64 --height 80 --width 80
```

The example configuration documents representative P3, P4, and P5 feature
shapes for convenient inspection of the multiscale interface.

## Release Plan

The complete model, training and evaluation implementation, ablation
configurations, and checkpoints will be released after the paper is accepted
and the required institutional review is completed.

## Citation

Citation information will be added after acceptance.

## License

No reuse license is granted for this preview release. A license compatible with
the upstream dependencies will be provided with the complete release.

