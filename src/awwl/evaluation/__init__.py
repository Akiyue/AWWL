"""Evaluation: CLIP scores, FID/IS, KID/PR/spectral, timestep error analysis."""

from __future__ import annotations

from awwl.evaluation.advanced_metrics import compute_advanced_metrics
from awwl.evaluation.clip_scores import (
    evaluate_clip_over_models,
    image_image_similarity,
    text_image_similarity,
)
from awwl.evaluation.cost import format_cost_table, measure_costs
from awwl.evaluation.fid_is import compute_fid_is
from awwl.evaluation.spectrum import radial_profile
from awwl.evaluation.timestep_analysis import TimestepProfile, evaluate_timestep_errors

__all__ = [
    "TimestepProfile",
    "compute_advanced_metrics",
    "format_cost_table",
    "measure_costs",
    "compute_fid_is",
    "evaluate_clip_over_models",
    "evaluate_timestep_errors",
    "image_image_similarity",
    "radial_profile",
    "text_image_similarity",
]
