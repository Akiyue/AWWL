"""Finetune method: from-scratch unconditional DDPM (CIFAR-10)."""

from __future__ import annotations

from awwl.methods.finetune.inference import generate_samples
from awwl.methods.finetune.trainer import train_finetune

__all__ = ["generate_samples", "train_finetune"]
