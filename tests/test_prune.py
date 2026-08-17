"""Freeing disk without losing a run.

The rule under test is the one a hurried `rm -rf` breaks: weights may go only
when the ledger already holds the numbers they produced.
"""

from __future__ import annotations

import json

import pytest
import yaml

from awwl.pipeline.prune import format_survey, prune, survey

MANIFEST = {
    "name": "toy",
    "method": "dreambooth",
    "base_config": "configs/dreambooth.yaml",
    "defaults": {"seeds": [1, 2]},
    "experiments": [{"group": "mse", "tier": 1, "overrides": {"loss.name": "mse"}}],
}


@pytest.fixture
def sweep(tmp_path):
    """Two runs with weights on disk; only mse_s1 has been evaluated."""
    root = tmp_path / "runs" / "toy"
    spec = {**MANIFEST, "output_root": str(root), "ledger": str(root / "results.jsonl")}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    for exp in ("mse_s1", "mse_s2"):
        unet = root / exp / "unet"
        unet.mkdir(parents=True)
        (unet / "weights.bin").write_bytes(b"x" * 2048)
        (root / exp / "config.json").write_text("{}", encoding="utf-8")

    with (root / "results.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"exp": "mse_s1", "group": "mse", "kind": "eval",
                             "clip_score": 0.3, "similarity": 0.8}) + "\n")
    return path, root


def test_only_evaluated_runs_are_reclaimable(sweep):
    path, _ = sweep

    _, items = survey(path)

    by_exp = {i.exp: i for i in items}
    assert by_exp["mse_s1"].evaluated is True
    assert by_exp["mse_s2"].evaluated is False


def test_prune_deletes_evaluated_weights_and_keeps_the_rest(sweep):
    path, root = sweep
    _, items = survey(path)

    runs, freed = prune(items)

    assert runs == 1
    assert freed > 0
    assert not (root / "mse_s1" / "unet").exists()
    assert (root / "mse_s2" / "unet").exists(), "a run with no results must survive"


def test_prune_never_touches_the_config_or_the_ledger(sweep):
    path, root = sweep
    _, items = survey(path)

    prune(items)

    assert (root / "mse_s1" / "config.json").exists(), "config identifies the run"
    assert (root / "results.jsonl").exists(), "the ledger is the result"


def test_survey_reports_what_is_held_back_and_why(sweep):
    path, _ = sweep

    name, items = survey(path)
    report = format_survey(name, items)

    assert "evaluated -- can be freed" in report
    assert "NO RESULTS -- keeping" in report
    assert "deleting these would lose the run" in report


def test_a_train_only_ledger_row_does_not_count_as_evaluated(tmp_path):
    """Training wrote a row; the metrics did not. The weights are still needed."""
    root = tmp_path / "runs" / "toy"
    spec = {**MANIFEST, "output_root": str(root), "ledger": str(root / "results.jsonl")}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    unet = root / "mse_s1" / "unet"
    unet.mkdir(parents=True)
    (unet / "w.bin").write_bytes(b"x" * 512)
    (root / "results.jsonl").write_text(
        json.dumps({"exp": "mse_s1", "group": "mse", "kind": "train"}) + "\n",
        encoding="utf-8",
    )

    _, items = survey(path)

    assert all(not i.evaluated for i in items)


def test_a_sweep_with_no_weights_on_disk_is_not_an_error(tmp_path):
    root = tmp_path / "runs" / "toy"
    root.mkdir(parents=True)
    spec = {**MANIFEST, "output_root": str(root), "ledger": str(root / "results.jsonl")}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    name, items = survey(path)

    assert items == []
    assert "no model weights on disk" in format_survey(name, items)
