"""Optimiser state must live where its parameters live.

The bug: a learned objective's parameters were loaded from a checkpoint while
the loss module was still on the CPU, so `Optimizer.load_state_dict` cast their
Adam moments to the CPU. The module was moved to the GPU afterwards, which
moved the parameters and left the moments behind. Training resumed at epoch 20
and died one optimiser step later inside `_multi_tensor_adam`.

Two devices are needed to reproduce it literally, so the traversal is tested
against stubs that report a device without owning memory on one.
"""

from __future__ import annotations

import torch

from awwl.training.optimizer_state import align_optimizer_state


class _Tensor(torch.Tensor):
    """A tensor that claims to be somewhere it is not, and records the move."""

    @staticmethod
    def __new__(cls, data, device_name):
        obj = torch.Tensor._make_subclass(cls, data)
        obj._device_name = device_name
        return obj

    @property
    def device(self):  # type: ignore[override]
        return self._device_name

    def to(self, target, *args, **kwargs):  # type: ignore[override]
        return _Tensor(torch.as_tensor(self.detach()), target)


class _Optimizer:
    """The three attributes `align_optimizer_state` touches."""

    def __init__(self, param_groups, state):
        self.param_groups = param_groups
        self.state = state


def _setup(param_device: str, state_device: str, *, step_device: str = "cpu"):
    param = _Tensor(torch.zeros(2), param_device)
    entry = {
        "exp_avg": _Tensor(torch.zeros(2), state_device),
        "exp_avg_sq": _Tensor(torch.zeros(2), state_device),
        "step": _Tensor(torch.zeros(()), step_device),
    }
    return param, entry, _Optimizer([{"params": [param]}], {param: entry})


def test_state_on_the_wrong_device_is_moved():
    param, entry, optimizer = _setup("cuda:0", "cpu")

    moved = align_optimizer_state(optimizer)

    assert moved == 2, "both moments belong on the parameter's device"
    assert entry["exp_avg"].device == "cuda:0"
    assert entry["exp_avg_sq"].device == "cuda:0"


def test_step_is_left_on_the_cpu():
    """torch keeps the step counter on the CPU for the fused paths."""
    _, entry, optimizer = _setup("cuda:0", "cuda:0", step_device="cpu")

    align_optimizer_state(optimizer)

    assert entry["step"].device == "cpu", "moving the step counter would be the bug"


def test_a_healthy_optimizer_is_untouched():
    _, entry, optimizer = _setup("cuda:0", "cuda:0")

    assert align_optimizer_state(optimizer) == 0
    assert entry["exp_avg"].device == "cuda:0"


def test_an_optimizer_with_no_state_yet_is_fine():
    """Before the first step there is nothing to align."""
    param = _Tensor(torch.zeros(2), "cuda:0")

    assert align_optimizer_state(_Optimizer([{"params": [param]}], {})) == 0


def test_it_runs_on_a_real_optimizer_after_a_step():
    """The traversal must match what torch actually stores."""
    param = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([param])
    param.grad = torch.ones(3)
    optimizer.step()

    assert align_optimizer_state(optimizer) == 0
    assert "exp_avg" in optimizer.state[param], "the traversal assumes this key exists"


def test_non_tensor_state_entries_are_skipped():
    param = _Tensor(torch.zeros(2), "cuda:0")
    entry = {"exp_avg": _Tensor(torch.zeros(2), "cpu"), "amsgrad": False, "name": "x"}

    assert align_optimizer_state(_Optimizer([{"params": [param]}], {param: entry})) == 1
    assert entry["amsgrad"] is False
