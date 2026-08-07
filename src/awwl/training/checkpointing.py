"""Crash-safe training state: save, prune, and auto-resume.

A "checkpoint" in diffusers usually means an exported pipeline folder — good
for sampling, useless for resuming, because it carries no optimiser moments,
no LR-scheduler position and no RNG state. Restarting from one silently
restarts training rather than continuing it.

This module saves the *whole* training state so a killed run resumes where it
stopped:

    <run_dir>/state/
        latest.json          pointer, replaced atomically
        step-000012000/
            unet/            model weights (diffusers format)
            optimizer.pt
            lr_scheduler.pt
            ema.pt           (only when EMA is on)
            rng.pt           python / numpy / torch / cuda RNG states
            meta.json        epoch, global_step, config fingerprint

Writes go to ``<name>.tmp/`` and are renamed into place, then ``latest.json``
is replaced atomically. A crash mid-write therefore leaves either the old
state or the new one, never a half-written directory that resumes into
garbage.

Old state directories are pruned to ``keep_last`` so a 200-epoch run does not
fill the disk with 400 MB snapshots.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_LATEST = "latest.json"


@dataclass
class ResumeState:
    """What :meth:`CheckpointManager.load` recovered from disk."""

    epoch: int
    global_step: int
    state_dir: Path
    meta: dict[str, Any]


class CheckpointManager:
    """Save and restore full training state under ``run_dir/state``.

    Args:
        run_dir: The run's output directory.
        keep_last: How many state snapshots to retain on disk.
        save_every_epochs: Snapshot cadence. With CIFAR-10 epochs at roughly a
            minute, the default of 5 bounds the work lost to a crash at a few
            minutes while keeping disk traffic modest.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        keep_last: int = 2,
        save_every_epochs: int = 5,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.state_root = self.run_dir / "state"
        self.keep_last = max(1, int(keep_last))
        self.save_every_epochs = max(1, int(save_every_epochs))

    # ------------------------------------------------------------------ save

    def should_save(self, epoch: int, num_epochs: int) -> bool:
        """True on the cadence, and always on the final epoch."""
        return (epoch + 1) % self.save_every_epochs == 0 or epoch == num_epochs - 1

    def save(
        self,
        *,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any | None = None,
        ema: Any | None = None,
        extras: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """Write one state snapshot and repoint ``latest.json`` at it.

        Args:
            extras: Additional ``state_dict``-bearing objects to persist, keyed
                by filename stem. Used for losses that carry learned
                parameters — without this a resumed run would restart the
                learned weighting from its initialisation while the network
                continued, silently invalidating the run.
        """
        self.state_root.mkdir(parents=True, exist_ok=True)
        final = self.state_root / f"step-{global_step:09d}"
        staging = self.state_root / f"step-{global_step:09d}.tmp"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        model.save_pretrained(staging / "unet")
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), staging / "lr_scheduler.pt")
        if ema is not None:
            torch.save(ema.state_dict(), staging / "ema.pt")
        for label, obj in (extras or {}).items():
            torch.save(obj.state_dict(), staging / f"{label}.pt")
        torch.save(_rng_state(), staging / "rng.pt")

        payload = {"epoch": int(epoch), "global_step": int(global_step), **(meta or {})}
        (staging / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        self._write_latest(final, payload)
        self._prune()
        logger.info("saved training state -> %s (epoch %d, step %d)", final, epoch, global_step)
        return final

    def _write_latest(self, state_dir: Path, payload: dict[str, Any]) -> None:
        tmp = self.state_root / (_LATEST + ".tmp")
        body = {"dir": state_dir.name, **payload}
        tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_root / _LATEST)

    def _prune(self) -> None:
        snaps = sorted(
            (p for p in self.state_root.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp")),
            key=lambda p: p.name,
        )
        for stale in snaps[: -self.keep_last]:
            shutil.rmtree(stale, ignore_errors=True)
            logger.debug("pruned old state %s", stale)

    # ------------------------------------------------------------------ load

    def latest_dir(self) -> Path | None:
        """Path of the newest complete snapshot, or ``None`` if there is none."""
        pointer = self.state_root / _LATEST
        if not pointer.exists():
            return None
        try:
            body = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("unreadable %s (%s); starting fresh", pointer, exc)
            return None
        candidate = self.state_root / str(body.get("dir", ""))
        if not (candidate / "meta.json").exists():
            logger.warning("state pointer references missing %s; starting fresh", candidate)
            return None
        return candidate

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: Any | None = None,
        ema: Any | None = None,
        extras: dict[str, Any] | None = None,
        map_location: str | torch.device = "cpu",
    ) -> ResumeState | None:
        """Restore everything found in the latest snapshot.

        Model weights are loaded in place. Missing sub-files are tolerated and
        logged — a snapshot taken without EMA can still resume a run that has
        EMA switched on, it just starts the shadow copy from the restored
        weights.

        Returns ``None`` when there is nothing to resume from.
        """
        state_dir = self.latest_dir()
        if state_dir is None:
            return None

        meta = json.loads((state_dir / "meta.json").read_text(encoding="utf-8"))

        weights = state_dir / "unet"
        loaded = type(model).from_pretrained(weights)
        model.load_state_dict(loaded.state_dict())
        del loaded

        if optimizer is not None:
            self._load_into(optimizer, state_dir / "optimizer.pt", map_location, "optimizer")
        if lr_scheduler is not None:
            self._load_into(lr_scheduler, state_dir / "lr_scheduler.pt", map_location, "lr_scheduler")
        if ema is not None:
            self._load_into(ema, state_dir / "ema.pt", map_location, "ema")
        for label, obj in (extras or {}).items():
            self._load_into(obj, state_dir / f"{label}.pt", map_location, label)

        rng_path = state_dir / "rng.pt"
        if rng_path.exists():
            _restore_rng(torch.load(rng_path, map_location="cpu", weights_only=False))

        resume = ResumeState(
            epoch=int(meta.get("epoch", -1)),
            global_step=int(meta.get("global_step", 0)),
            state_dir=state_dir,
            meta=meta,
        )
        logger.info(
            "resumed from %s — continuing at epoch %d (step %d)",
            state_dir,
            resume.epoch + 1,
            resume.global_step,
        )
        return resume

    @staticmethod
    def _load_into(obj: Any, path: Path, map_location: str | torch.device, label: str) -> None:
        if not path.exists():
            logger.warning("no %s state in snapshot; leaving it at its initial value", label)
            return
        obj.load_state_dict(torch.load(path, map_location=map_location, weights_only=False))


# ---------------------------------------------------------------------- RNG


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                # Resuming on a different GPU count — seed what we can rather
                # than aborting the run.
                torch.cuda.set_rng_state(cuda_state[0])
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("could not fully restore RNG state (%s); continuing", exc)
