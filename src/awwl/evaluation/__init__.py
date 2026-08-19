"""Evaluation: CLIP scores, FID/IS, KID/PR/spectral, timestep error analysis.

The heavy metric backends (transformers for CLIP, clean-fid / torch-fidelity for
FID/IS/KID) are imported lazily on attribute access, so importing this package
does not pull them in until a metric is actually requested.
"""

from __future__ import annotations

import importlib

_SUBMODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "advanced_metrics": ("compute_advanced_metrics",),
    "clip_scores": (
        "evaluate_clip_over_models",
        "image_image_similarity",
        "text_image_similarity",
    ),
    "cost": ("format_cost_table", "measure_costs"),
    "fid_is": ("compute_fid_is",),
    "restoration": ("evaluate_restoration", "psnr", "ssim"),
    "sensitivity": ("measure_sensitivity",),
    "spectrum": ("radial_profile",),
    "timestep_analysis": ("TimestepProfile", "evaluate_timestep_errors"),
}

__all__ = [name for names in _SUBMODULE_ATTRS.values() for name in names]


def __getattr__(name: str):
    for module_name, names in _SUBMODULE_ATTRS.items():
        if name in names:
            module = importlib.import_module(f"{__name__}.{module_name}")
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
