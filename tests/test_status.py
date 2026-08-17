"""Coverage reporting: what the ledger says against what the queue says.

The case worth testing is the disagreement. A queue that reports ``done`` for
a job whose ledger row never landed will otherwise be believed, and the paper
will quietly average over fewer seeds than it claims.
"""

from __future__ import annotations

import json

import pytest
import yaml

from awwl.pipeline.status import collect_status, format_status

MANIFEST = {
    "name": "toy",
    "method": "finetune",
    "base_config": "configs/base.yaml",
    "output_root": None,  # filled in per test
    "defaults": {"seeds": [1, 2, 3], "eval_epochs": [99, 199]},
    "experiments": [
        {"group": "mse", "tier": 1, "overrides": {"loss.name": "mse"}},
        {"group": "awwl", "tier": 2, "overrides": {"loss.name": "adaptive_wavelet"}},
    ],
}


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "runs" / "toy"
    spec = {**MANIFEST, "output_root": str(root), "ledger": str(root / "results.jsonl")}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path, root


def write_ledger(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def eval_row(group, seed, **metrics):
    return {"group": group, "seed": seed, "kind": "eval", **metrics}


def test_seed_totals_count_seeds_not_checkpoints(manifest):
    """Eval jobs carry an epoch suffix; counting them would inflate the total."""
    path, _ = manifest

    _, _, statuses, _ = collect_status(path)

    # Three seeds and two eval epochs: the plan is three, not six.
    assert {s.group: len(s.planned_seeds) for s in statuses} == {"mse": 3, "awwl": 3}


def test_complete_group_is_reported_complete(manifest):
    path, root = manifest
    write_ledger(root, [
        eval_row("mse", s, fid=18.0, is_mean=7.9, kid_tf=0.01) for s in (1, 2, 3)
    ])

    _, method, statuses, counts = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    mse = next(s for s in statuses if s.group == "mse")
    assert mse.complete
    assert "complete" in report
    assert "usable for the paper: 1/2" in report


def test_a_missing_metric_is_named(manifest):
    """All seeds present but one metric absent is invisible in a seed count."""
    path, root = manifest
    write_ledger(root, [eval_row("mse", s, fid=18.0, is_mean=7.9) for s in (1, 2, 3)])

    _, method, statuses, counts = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    assert "missing kid_tf" in report


def test_partial_seeds_are_not_complete(manifest):
    path, root = manifest
    write_ledger(root, [
        eval_row("mse", s, fid=18.0, is_mean=7.9, kid_tf=0.01) for s in (1, 2)
    ])

    _, method, statuses, counts = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    mse = next(s for s in statuses if s.group == "mse")
    assert not mse.complete
    assert "2/3" in report


def test_queue_done_without_ledger_rows_is_flagged(manifest):
    """The dangerous disagreement: the queue is satisfied, the paper has nothing."""
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import JobStore, store_path

    path, root = manifest
    write_ledger(root, [])

    store = JobStore(store_path(root))
    jobs = build_jobs(load_manifest(path))
    store.add_jobs(jobs)
    for _ in jobs:
        claimed = store.claim("w")
        if claimed:
            store.finish(claimed.job_id)

    _, method, statuses, counts = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    assert counts.get("done")
    assert "WARNING: queue reports done but no ledger rows exist" in report
    assert "usable for the paper: 0/2" in report


def test_failed_jobs_surface_in_the_status_column(manifest):
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import JobStore, store_path

    path, root = manifest
    store = JobStore(store_path(root), max_attempts=1)
    store.add_jobs(build_jobs(load_manifest(path)))
    claimed = store.claim("w")
    store.finish(claimed.job_id, error="boom")

    _, method, statuses, counts = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    assert counts.get("failed") == 1
    assert "FAILED" in report


def test_the_queue_reader_looks_where_the_runner_writes(tmp_path):
    """Regression: the reader guessed ``jobs.db`` and the runner writes ``state.db``.

    Nothing failed. ``collect_status`` found no database, reported no queue
    state, and 25 failed jobs appeared as an unexplained gap in the ledger --
    the worst kind of bug in a tool whose job is to tell you what went wrong.
    Both sides now call ``store_path``; this pins the name so a rename has to
    be deliberate.
    """
    from awwl.pipeline.store import store_path

    assert store_path(tmp_path) == tmp_path / "pipeline" / "state.db"


def test_a_broken_manifest_does_not_hide_the_others(tmp_path, manifest):
    from awwl.pipeline.status import status_report

    path, _ = manifest
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: nope\n", encoding="utf-8")

    report = status_report([broken, path])

    assert "could not read" in report
    assert "## toy" in report
