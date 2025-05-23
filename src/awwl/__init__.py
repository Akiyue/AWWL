"""AWWL: Adaptive Wavelet-Weighted Loss for diffusion-model fine-tuning.

Two methods are supported:

* :mod:`awwl.methods.dreambooth` — Stable-Diffusion DreamBooth (full UNet) and
  DreamBooth+LoRA fine-tuning, the original AWWL recipe.
* :mod:`awwl.methods.finetune` — From-scratch DDPM fine-tuning on small image
  datasets (e.g. CIFAR-10), the AWWL-Diff variant.

Both share :mod:`awwl.losses`, :mod:`awwl.data`, and the training/utility
helpers. Method-specific code lives under :mod:`awwl.methods`.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
