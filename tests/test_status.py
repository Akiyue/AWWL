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

    _, _, statuses, _, _ = collect_status(path)

    # Three seeds and two eval epochs: the plan is three, not six.
    assert {s.group: len(s.planned_seeds) for s in statuses} == {"mse": 3, "awwl": 3}


def test_complete_group_is_reported_complete(manifest):
    path, root = manifest
    write_ledger(root, [
        eval_row("mse", s, fid=18.0, is_mean=7.9, kid_tf=0.01) for s in (1, 2, 3)
    ])

    _, method, statuses, counts, _ = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    mse = next(s for s in statuses if s.group == "mse")
    assert mse.complete
    assert "complete" in report
    assert "usable for the paper: 1/2" in report


def test_a_missing_metric_is_named(manifest):
    """All seeds present but one metric absent is invisible in a seed count."""
    path, root = manifest
    write_ledger(root, [eval_row("mse", s, fid=18.0, is_mean=7.9) for s in (1, 2, 3)])

    _, method, statuses, counts, _ = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    assert "missing kid_tf" in report


def test_partial_seeds_are_not_complete(manifest):
    path, root = manifest
    write_ledger(root, [
        eval_row("mse", s, fid=18.0, is_mean=7.9, kid_tf=0.01) for s in (1, 2)
    ])

    _, method, statuses, counts, _ = collect_status(path)
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

    _, method, statuses, counts, _ = collect_status(path)
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

    _, method, statuses, counts, _ = collect_status(path)
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


def test_failures_are_reported_with_their_reason(manifest):
    """Knowing that jobs failed is not the point; knowing why is.

    Sweeps fail the same way many times over, so the report groups identical
    reasons rather than printing one line per job.
    """
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import JobStore, store_path

    path, root = manifest
    store = JobStore(store_path(root), max_attempts=1)
    store.add_jobs(build_jobs(load_manifest(path)))
    for _ in range(3):
        claimed = store.claim("w")
        store.finish(
            claimed.job_id,
            error='Traceback (most recent call last):\n  File "x.py", line 1\n'
                  "torch.OutOfMemoryError: CUDA out of memory.",
        )

    _, method, statuses, counts, _ = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    assert "failures, by reason:" in report
    assert "torch.OutOfMemoryError: CUDA out of memory." in report
    assert "3x" in report, "identical reasons are counted, not repeated"
    assert 'File "x.py"' not in report, "frame headers identify nothing"


def test_last_meaningful_line_skips_traceback_furniture():
    from awwl.pipeline.status import _last_meaningful_line

    error = (
        "Traceback (most recent call last):\n"
        '  File "/a/b.py", line 42, in main\n'
        "    loss = fn(x)\n"
        "           ^^^^^\n"
        "ValueError: perceptual loss needs 1 or 3 channels, got 4\n"
    )

    assert _last_meaningful_line(error) == (
        "ValueError: perceptual loss needs 1 or 3 channels, got 4"
    )


def test_a_job_with_no_recorded_error_does_not_crash_the_report():
    from awwl.pipeline.status import _last_meaningful_line

    assert "killed, or output truncated" in _last_meaningful_line("   \n\n  ")


def test_interpreter_shutdown_noise_is_not_read_as_the_failure():
    """The reason 19 DreamBooth failures all reported the same wrong cause.

    `multiprocess` raises this on every exit under Python 3.12, successful runs
    included. It is printed after the real traceback, so taking the last line
    of stderr reports teardown as the cause and hides what actually happened.
    """
    from awwl.pipeline.status import _last_meaningful_line

    error = (
        "Traceback (most recent call last):\n"
        '  File "/a/train.py", line 10, in main\n'
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        "Exception ignored in: <function ResourceTracker.__del__ at 0x7f>\n"
        "Traceback (most recent call last):\n"
        '  File "/x/resource_tracker.py", line 80, in __del__\n'
        "AttributeError: '_thread.RLock' object has no attribute '_recursion_count'\n"
    )

    assert _last_meaningful_line(error) == (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    )


def test_shutdown_noise_alone_is_reported_as_no_traceback():
    """A killed process leaves only teardown output; say so rather than guess."""
    from awwl.pipeline.status import _last_meaningful_line

    error = (
        "Exception ignored in: <function ResourceTracker.__del__ at 0x7f>\n"
        "AttributeError: '_thread.RLock' object has no attribute '_recursion_count'\n"
    )

    assert "killed, or output truncated" in _last_meaningful_line(error)


def test_show_errors_dumps_the_full_stderr(manifest):
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import JobStore, store_path

    path, root = manifest
    store = JobStore(store_path(root), max_attempts=1)
    store.add_jobs(build_jobs(load_manifest(path)))
    claimed = store.claim("w")
    store.finish(claimed.job_id, error="line one\nRuntimeError: the real cause")

    _, method, statuses, counts, _ = collect_status(path)

    assert "line one" not in format_status("toy", method, statuses, counts)
    assert "line one" in format_status("toy", method, statuses, counts, show_errors=True)


def test_the_exception_is_found_inside_a_rich_traceback_box():
    """rich draws tracebacks in a box and tqdm keeps writing after the failure.

    Taking the last line then yields a progress bar carrying a per-run loss
    value, which is both wrong and unique per job, so identical failures never
    group. This is the shape the DreamBooth saves actually failed in.
    """
    from awwl.pipeline.status import _last_meaningful_line

    error = (
        "│ /a/trainer.py:168 in train_dreambooth                    │\n"
        "│ ❱ 168 │   unet_unwrapped.save_pretrained(save_path) │\n"
        "│ OSError: [Errno 28] No space left on device               │\n"
        "╰────────────╯\n"
        "100%|█████| 400/400 [02:19<00:00,  2.87it/s, loss=0.0419]\n"
    )

    assert _last_meaningful_line(error) == "OSError: [Errno 28] No space left on device"


def test_identical_failures_group_despite_differing_progress_bars(manifest):
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import JobStore, store_path

    path, root = manifest
    store = JobStore(store_path(root), max_attempts=1)
    store.add_jobs(build_jobs(load_manifest(path)))
    for loss in (0.0419, 0.133, 0.0878):
        claimed = store.claim("w")
        store.finish(
            claimed.job_id,
            error=(
                "OSError: [Errno 28] No space left on device\n"
                f"100%|██| 400/400 [02:19<00:00, 2.87it/s, loss={loss}]\n"
            ),
        )

    _, method, statuses, counts, _ = collect_status(path)
    report = format_status("toy", method, statuses, counts)

    assert "3x  OSError: [Errno 28] No space left on device" in report
    assert "loss=" not in report, "a per-run loss value must not become the reason"


def test_jobs_of_a_deleted_experiment_are_visible_and_retirable(manifest, tmp_path):
    """Deleting an experiment must stop it running, not just stop listing it.

    `perceptual` was removed from the manifest. build_jobs stopped producing
    its jobs, so add_jobs could neither update nor remove the ones already
    queued; the runner kept claiming them, while the status report -- built
    from the manifest -- showed a clean sweep. It ran on both GPUs for hours.
    """
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import RETIRED, Job, JobStore, store_path

    path, root = manifest
    store = JobStore(store_path(root))
    jobs = build_jobs(load_manifest(path))
    store.add_jobs(jobs)
    store.add_jobs([
        Job(job_id="toy:train:gone_s1", pipeline="toy", kind="train", group_id="gone",
            tier=3, depends_on=None, payload={"argv": ["echo", "gone"]},
            status="pending", attempts=0)
    ])

    _, method, statuses, counts, orphans = collect_status(path)
    report = format_status("toy", method, statuses, counts, orphans=orphans)

    assert orphans == {"gone": 1}
    assert "QUEUED BUT NOT IN THE MANIFEST" in report
    assert "gone" in report

    retired = store.retire_missing("toy", {j.job_id for j in jobs})

    assert retired == 1
    assert store.claim("w", max_tier=99).group_id != "gone"
    assert {j.status for j in store.list_jobs() if j.group_id == "gone"} == {RETIRED}


def test_retiring_never_discards_a_finished_result(manifest):
    """A done job is the record of a result on disk; dropping the plan keeps it."""
    from awwl.pipeline.manifest import build_jobs, load_manifest
    from awwl.pipeline.store import DONE, JobStore, store_path

    path, root = manifest
    store = JobStore(store_path(root))
    jobs = build_jobs(load_manifest(path))
    store.add_jobs(jobs)
    claimed = store.claim("w")
    store.finish(claimed.job_id)

    store.retire_missing("toy", set())

    finished = [j for j in store.list_jobs() if j.job_id == claimed.job_id]
    assert [j.status for j in finished] == [DONE]
