"""Method-specific trainers and inference routines."""

from __future__ import annotations

from awwl.core.exceptions import UnknownMethodError

KNOWN_METHODS = ("dreambooth", "dreambooth_lora", "finetune", "restoration")


def get_trainer(method: str):
    """Return the ``train_*`` callable for ``method``.

    The trainer takes one positional argument: a merged config dict.
    """
    if method == "dreambooth":
        from awwl.methods.dreambooth import train_dreambooth

        return train_dreambooth
    if method == "dreambooth_lora":
        from awwl.methods.dreambooth import train_dreambooth_lora

        return train_dreambooth_lora
    if method == "finetune":
        from awwl.methods.finetune import train_finetune

        return train_finetune
    if method == "restoration":
        from awwl.methods.restoration.trainer import train_restoration

        return train_restoration
    raise UnknownMethodError(f"unknown method {method!r}; known: {', '.join(KNOWN_METHODS)}")


__all__ = ["KNOWN_METHODS", "get_trainer"]
