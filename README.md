# DMS-EI

This repository currently provides a **partial reference preview** of DMS-EI,
a degradation-supervised multimodal state-guided expert interaction network for
RGB--infrared object detection.

## Current Release Scope

The current preview contains only a simplified implementation of the
quality-guided bidirectional interaction component. It demonstrates:

- modality-specific feature projection;
- feature-based multimodal state description;
- modality-quality estimation;
- quality-guided RGB-to-infrared and infrared-to-RGB feature injection; and
- shape-preserving residual outputs.

The following components are **not included** in this initial release:

- controlled single-modality degradation and explicit quality supervision;
- original-image statistical descriptors;
- joint and modality-specific expert implementations;
- adaptive expert weighting;
- the complete dual-stream YOLO11 integration and multiscale feedback path;
- training, evaluation, ablation, visualization, datasets, and checkpoints.

Therefore, the current code cannot reproduce the quantitative results reported
in the paper.

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later

## Repository Contents

```text
.
|-- dmsei_core.py       # Partial interaction-component preview
|-- demo_forward.py     # Random-tensor shape check
|-- example_config.yaml # Public feature-shape example (not for training)
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
shapes only. It does not expose the detector-training configuration used in the
paper.

## Release Plan

The complete training and evaluation implementation, ablation configurations,
and checkpoints are planned for release after the paper is accepted and the
required institutional review is completed.

## Citation

Citation information will be added after acceptance.

## License

No reuse license is granted for this preview release. A license compatible with
the upstream dependencies will be provided with the complete release.

