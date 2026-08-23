"""Job-store and manifest tests — the crash-recovery guarantees, checked."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from awwl.core.exceptions import ConfigError
from awwl.pipeline.manifest import build_jobs, load_manifest
from awwl.pipeline.store import DONE, FAILED, PENDING, RETIRED, RUNNING, Job, JobStore


def _job(job_id: str, *, tier: int = 1, depends_on: str | None = None) -> Job:
    return Job(
        job_id=job_id,
        pipeline="test",
        kind="train",
        group_id="g",
        tier=tier,
        depends_on=depends_on,
        payload={"argv": ["echo", job_id]},
        status=PENDING,
        attempts=0,
    )


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "state.db", stale_after=0.5, max_attempts=2)


def test_add_jobs_is_idempotent(store):
    """Re-running an unchanged manifest must not duplicate or reset work."""
    jobs = [_job("a"), _job("b")]
    assert store.add_jobs(jobs) == 2
    assert store.add_jobs(jobs) == 0
    assert len(store.list_jobs()) == 2


def test_claim_is_exclusive(store):
    """Two workers must never receive the same job."""
    store.add_jobs([_job("a")])
    first = store.claim("w1")
    second = store.claim("w2")
    assert first is not None and first.job_id == "a"
    assert second is None


def test_claim_respects_dependencies(store):
    """A job whose dependency is unfinished is not runnable."""
    store.add_jobs([_job("train"), _job("sample", depends_on="train")])
    claimed = store.claim("w1")
    assert claimed.job_id == "train"
    assert store.claim("w2") is None, "sample became runnable before train finished"

    store.finish("train")
    assert store.claim("w2").job_id == "sample"


def test_claim_respects_tier_order(store):
    """Lower tiers drain first, so a budget cut stops at a tier boundary."""
    store.add_jobs([_job("t2", tier=2), _job("t1", tier=1)])
    assert store.claim("w").job_id == "t1"


def test_max_tier_filters(store):
    store.add_jobs([_job("t2", tier=2)])
    assert store.claim("w", max_tier=1) is None
    assert store.claim("w", max_tier=2).job_id == "t2"


def test_claim_is_scoped_to_pipeline(store):
    """Two manifests sharing an output_root share one store; neither may
    execute the other's jobs — concurrent runners would otherwise put two
    heavy jobs on whichever GPU polls first."""
    store.add_jobs([_job("b_first", tier=1), _job("a_later", tier=2)])
    with store._connect() as conn:
        conn.execute("UPDATE jobs SET pipeline = 'other' WHERE job_id = 'b_first'")
        conn.execute("UPDATE jobs SET pipeline = 'mine' WHERE job_id = 'a_later'")

    claimed = store.claim("w", pipeline="mine")
    assert (
        claimed is not None and claimed.job_id == "a_later"
    ), "claim crossed the pipeline boundary"
    assert store.claim("w", pipeline="mine") is None


def test_reclaim_stale_requeues_dead_jobs(store):
    """The core crash guarantee: a job whose worker died comes back."""
    store.add_jobs([_job("a")])
    store.claim("worker-that-dies")
    assert store.counts().get(RUNNING) == 1

    time.sleep(0.6)  # exceed stale_after
    assert store.reclaim_stale() == 1
    assert store.counts().get(PENDING) == 1
    assert store.claim("fresh-worker") is not None


def test_reclaim_leaves_live_jobs_alone(store):
    """A heartbeating job must not be stolen from its worker."""
    store.add_jobs([_job("a")])
    store.claim("w1")
    time.sleep(0.6)
    store.heartbeat("a")
    assert store.reclaim_stale() == 0
    assert store.counts().get(RUNNING) == 1


def test_failure_retries_then_parks(store):
    """Transient faults retry; a persistent one is parked without stalling."""
    store.add_jobs([_job("a")])
    store.claim("w")
    store.finish("a", error="boom")
    assert store.counts().get(PENDING) == 1, "first failure should be retryable"

    store.claim("w")
    store.finish("a", error="boom again")
    assert store.counts().get(FAILED) == 1

    assert store.claim("w") is None, "a parked job must not be handed out again"
    assert store.reset() == 1
    assert store.claim("w") is not None


def test_pending_work_reaches_zero(store):
    """The workers' exit condition."""
    store.add_jobs([_job("a")])
    assert store.pending_work() == 1
    store.claim("w")
    assert store.pending_work() == 1, "a running job is still outstanding work"
    store.finish("a")
    assert store.pending_work() == 0
    assert store.counts().get(DONE) == 1


# ------------------------------------------------------------------ manifest

_MANIFEST = """
name: t
base_config: configs/finetune.yaml
output_root: ./runs/t
real_images: ./data/ref
defaults:
  seeds: [1, 2]
  eval_epochs: [9]
experiments:
  - group: mse
    overrides: {loss.name: mse}
  - group: awwl
    tier: 2
    seeds: [1]
    overrides: {loss.name: adaptive_wavelet, loss.alpha: 0.2}
"""


def test_manifest_expands_seed_cross_product(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(_MANIFEST, encoding="utf-8")
    jobs = build_jobs(load_manifest(path))

    kinds = [j.kind for j in jobs]
    assert kinds.count("train") == 3  # mse x2 seeds + awwl x1
    assert kinds.count("sample") == 3
    assert kinds.count("eval") == 3

    ids = {j.job_id for j in jobs}
    assert "t:train:mse_s1" in ids
    assert "t:train:awwl_s2" not in ids, "per-experiment seeds should override defaults"


def test_manifest_chains_dependencies(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(_MANIFEST, encoding="utf-8")
    by_id = {j.job_id: j for j in build_jobs(load_manifest(path))}

    assert by_id["t:train:mse_s1"].depends_on is None
    assert by_id["t:sample:mse_s1:9"].depends_on == "t:train:mse_s1"
    assert by_id["t:eval:mse_s1:9"].depends_on == "t:sample:mse_s1:9"


def test_manifest_job_ids_are_stable(tmp_path):
    """Ids must not depend on iteration order, or restarts would re-queue work."""
    path = tmp_path / "m.yaml"
    path.write_text(_MANIFEST, encoding="utf-8")
    first = [j.job_id for j in build_jobs(load_manifest(path))]
    second = [j.job_id for j in build_jobs(load_manifest(path))]
    assert first == second


def test_manifest_carries_overrides_into_argv(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(_MANIFEST, encoding="utf-8")
    by_id = {j.job_id: j for j in build_jobs(load_manifest(path))}
    argv = " ".join(by_id["t:train:awwl_s1"].argv)
    assert "loss.name=adaptive_wavelet" in argv
    assert "loss.alpha=0.2" in argv
    assert "seed=1" in argv
    assert "output.group=awwl" in argv
    assert by_id["t:train:awwl_s1"].tier == 2


def test_manifest_rejects_incomplete_spec(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: t\nexperiments: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_manifest(path)


_REUSE_MANIFEST = """
name: r
base_config: configs/finetune.yaml
output_root: ./runs/phase0
reuse_runs: true
real_images: ./data/ref
defaults:
  seeds: [1]
  eval_epochs: [199]
  sample: {sampler: ddpm, steps: 1000, out_name: samples_ddpm1000}
experiments:
  - group: mse
    overrides: {loss.name: mse}
"""


def test_reuse_runs_skips_train_and_never_chains_to_it(tmp_path):
    """A re-scoring sweep must not create train jobs for runs that exist.

    A no-op ``train`` looks harmless but rewrites the run's config.json with
    the new sweep's ledger path before it notices the run is finished, so the
    only safe reuse is to never enter the trainer at all.
    """
    path = tmp_path / "r.yaml"
    path.write_text(_REUSE_MANIFEST, encoding="utf-8")
    by_id = {j.job_id: j for j in build_jobs(load_manifest(path))}

    assert not any(jid.startswith("r:train:") for jid in by_id)
    assert "r:sample:mse_s1:199" in by_id
    assert by_id["r:sample:mse_s1:199"].depends_on is None
    assert by_id["r:eval:mse_s1:199"].depends_on == "r:sample:mse_s1:199"


def test_reuse_runs_samples_land_beside_the_originals(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(_REUSE_MANIFEST, encoding="utf-8")
    by_id = {j.job_id: j for j in build_jobs(load_manifest(path))}

    argv = " ".join(by_id["r:sample:mse_s1:199"].argv).replace("\\", "/")
    assert "--sampler ddpm" in argv and "--steps 1000" in argv
    # The DDIM samples behind the paper's figures live in samples/ep199;
    # a different sampler must write somewhere else.
    assert "samples_ddpm1000/ep199" in argv


_REUSE_RESTORE_MANIFEST = """
name: rr
base_config: configs/finetune.yaml
method: restoration
reuse_runs: true
output_root: ./runs/restored
real_images: ./data/ref
defaults:
  seeds: [1]
  eval_epochs: [99]
experiments:
  - group: mse
    overrides: {loss.name: mse}
"""


def test_reuse_runs_never_depends_on_a_train_job_that_was_not_created(tmp_path):
    """restoration/dreambooth evals chained to a skipped train would deadlock."""
    path = tmp_path / "rr.yaml"
    path.write_text(_REUSE_RESTORE_MANIFEST, encoding="utf-8")
    by_id = {j.job_id: j for j in build_jobs(load_manifest(path))}

    assert "rr:train:mse_s1" not in by_id
    assert by_id["rr:eval:mse_s1:99"].depends_on is None


def test_format_status_reports_retired_separately_from_remaining(tmp_path):
    """Retired jobs are never going to run; listing them as 'remaining'
    made a finished sweep look like it still had work."""
    from awwl.pipeline.runner import format_status

    store = JobStore(tmp_path / "state.db")
    store.add_jobs([_job("d"), _job("gone")])
    store.finish("d")
    # add_jobs inserts as pending; retirement happens later, on a re-expand
    # that no longer lists the job.
    with store._connect() as conn:
        conn.execute("UPDATE jobs SET status = ? WHERE job_id = 'gone'", (RETIRED,))

    text = format_status(store)
    assert "remaining:" not in text, "retired jobs were counted as outstanding work"
    assert "retired" in text


def test_manifest_edits_reach_unfinished_jobs(store):
    """A job's command is otherwise frozen at the moment it was first queued.

    25 DreamBooth evaluations failed against a dataset path that did not
    exist; correcting the manifest and re-running changed nothing, because the
    wrong argv was already in the database.
    """
    store.add_jobs([_job("a")])
    store.claim("w")
    store.finish("a", error="wrong path")

    fixed = _job("a")
    fixed.payload = {"argv": ["echo", "corrected"]}
    store.add_jobs([fixed])

    store.reset()
    assert store.claim("w").argv == ["echo", "corrected"]


def test_refresh_never_touches_a_finished_job(store):
    """Rewriting a completed job's command would invalidate its result."""
    store.add_jobs([_job("a")])
    store.claim("w")
    store.finish("a")

    changed = _job("a")
    changed.payload = {"argv": ["echo", "different"]}
    store.add_jobs([changed])

    (job,) = [j for j in store.list_jobs() if j.job_id == "a"]
    assert job.status == DONE
    assert job.argv == ["echo", "a"], "a done job must keep the command that produced it"


def test_refresh_can_be_disabled(store):
    store.add_jobs([_job("a")])
    changed = _job("a")
    changed.payload = {"argv": ["echo", "no"]}
    store.add_jobs([changed], refresh=False)
    assert store.claim("w").argv == ["echo", "a"]


_DREAMBOOTH_MANIFEST = """
name: db
method: dreambooth
base_config: configs/dreambooth.yaml
output_root: ./runs/db
defaults:
  seeds: [1]
  sample: {num_samples: 4, steps: 10, prompts: assets/prompts/robot_toy.txt}
experiments:
  - group: mse
    overrides: {loss.name: mse}
"""


def test_dreambooth_eval_is_scheduled_without_real_images(tmp_path):
    """The eval must exist even when the manifest names no reference folder.

    It previously did not: omitting `real_images` silently dropped every eval
    job, and naming a path that had stopped existing failed all of them at
    argument validation while training carried its own copy and succeeded.
    The reference set is now taken from each run's own config.
    """
    path = tmp_path / "db.yaml"
    path.write_text(_DREAMBOOTH_MANIFEST, encoding="utf-8")

    jobs = build_jobs(load_manifest(path))

    evals = [j for j in jobs if j.kind == "eval"]
    assert len(evals) == 1
    assert "--real" not in evals[0].argv
    assert "--prompts" in evals[0].argv


def test_dreambooth_real_images_still_overrides_when_given(tmp_path):
    path = tmp_path / "db.yaml"
    path.write_text(
        _DREAMBOOTH_MANIFEST.replace(
            "output_root: ./runs/db", "output_root: ./runs/db\nreal_images: ./data/ref"
        ),
        encoding="utf-8",
    )

    argv = [j for j in build_jobs(load_manifest(path)) if j.kind == "eval"][0].argv

    assert argv[argv.index("--real") + 1] == "./data/ref"


def test_refresh_propagates_a_tier_change_with_an_unchanged_command(tmp_path):
    """Demoting an experiment must actually demote it.

    The refresh guard compared payloads only, so a manifest edit that changed
    a job's tier and nothing else matched no rows. `perceptual` was moved from
    tier 3 to tier 5 to stop it blocking tier 4; the stored tier stayed at 3,
    `--max-tier 4` kept selecting it, and it ran for hours on two GPUs.
    """
    from awwl.pipeline.store import PENDING, JobStore

    store = JobStore(tmp_path / "state.db")
    job = Job(
        job_id="p:train:x",
        pipeline="p",
        kind="train",
        group_id="x",
        tier=3,
        depends_on=None,
        payload={"argv": ["awwl", "train"]},
        status=PENDING,
        attempts=0,
    )
    store.add_jobs([job])

    demoted = replace(job, tier=5)  # same command, later tier
    store.add_jobs([demoted])

    stored = {j.job_id: j for j in store.list_jobs()}["p:train:x"]
    assert stored.tier == 5
    assert store.claim("w", max_tier=4) is None, "a tier-5 job must not be claimed at max_tier 4"
    assert store.claim("w", max_tier=5) is not None
