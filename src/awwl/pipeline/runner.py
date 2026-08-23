"""Execute pipeline jobs across GPUs, one subprocess at a time per device.

Each GPU gets a worker thread that claims a job from the :class:`JobStore` and
runs it as a **subprocess** pinned to that device via ``CUDA_VISIBLE_DEVICES``.
Running jobs out-of-process rather than in-process is what keeps the sweep
alive: a CUDA OOM or a segfault in one experiment kills only that subprocess,
the worker records the failure, and the queue moves on. It also lets the
worker emit a heartbeat while a job runs for hours, which is what tells a
restarted pipeline that a job died with the machine rather than finishing.

Independent single-GPU jobs are used in preference to one DDP job spanning
both cards. For models this small the sweep is throughput-bound, not
latency-bound — two independent runs finish a 40-run matrix in half the time
that data-parallel runs would, and these cards have no NVLink, so DDP would
pay full PCIe cost on every gradient sync.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from awwl.pipeline.store import DONE, FAILED, RETIRED, Job, JobStore
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class RunReport:
    """Outcome of one :func:`run_pipeline` invocation."""

    completed: int = 0
    failed: int = 0
    interrupted: bool = False
    failures: list[tuple[str, str]] = field(default_factory=list)


class _Stop:
    """Cooperative stop flag shared by the worker threads."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> None:
        self._event.wait(seconds)


def run_pipeline(
    store: JobStore,
    *,
    gpus: list[str],
    log_dir: str | Path,
    max_tier: int | None = None,
    poll_interval: float = 20.0,
    heartbeat_interval: float = 60.0,
    cwd: str | Path | None = None,
    pipeline: str | None = None,
) -> RunReport:
    """Drain the queue using one worker per entry in ``gpus``.

    Args:
        gpus: Device ids as strings, e.g. ``["0", "1"]``. Each becomes the
            ``CUDA_VISIBLE_DEVICES`` value for one worker, so a job always
            sees exactly one GPU and addresses it as ``cuda:0``.
        log_dir: Per-job stdout/stderr lands in ``<log_dir>/<job_id>.log``.
        max_tier: Only run jobs at or below this tier.
        poll_interval: How long a worker waits before re-checking for work
            when the queue has nothing runnable (typically because it is
            waiting on another worker's training job to finish).
        heartbeat_interval: How often a running job's liveness is recorded.
            Must be well under the store's ``stale_after``.
        pipeline: Claim only this pipeline's jobs. The store is shared by
            every manifest writing to the same ``output_root``, so an
            unscoped runner would execute whichever other sweep's pending
            jobs it found first — on its GPUs.

    Returns:
        A :class:`RunReport`. Failed jobs do not abort the sweep; they are
        listed so they can be inspected and retried with ``pipeline reset``.
    """
    reclaimed = store.reclaim_stale()
    if reclaimed:
        logger.info("requeued %d job(s) left running by a previous process", reclaimed)

    logs = ensure_dir(log_dir)
    stop = _Stop()
    report = RunReport()
    lock = threading.Lock()

    previous_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):  # pragma: no cover - interactive path
        del signum, frame
        if not stop.requested:
            logger.warning("interrupt received — finishing current jobs, then stopping")
            report.interrupted = True
            stop.set()
        else:
            logger.warning("second interrupt — exiting now")
            raise KeyboardInterrupt

    # Only the main thread may install handlers; embedded callers get none.
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGINT, _on_sigint)

    def _worker(gpu: str) -> None:
        worker_id = f"gpu{gpu}@{os.getpid()}"
        while not stop.requested:
            job = store.claim(worker_id, max_tier=max_tier, pipeline=pipeline)
            if job is None:
                if store.pending_work(max_tier=max_tier, pipeline=pipeline) == 0:
                    return
                stop.wait(poll_interval)
                continue

            logger.info("[%s] %s %s", worker_id, job.kind, job.job_id)
            error = _run_job(
                job,
                store=store,
                gpu=gpu,
                log_path=logs / f"{_safe_name(job.job_id)}.log",
                heartbeat_interval=heartbeat_interval,
                stop=stop,
                cwd=cwd,
            )
            store.finish(job.job_id, error=error)
            with lock:
                if error is None:
                    report.completed += 1
                else:
                    report.failed += 1
                    report.failures.append((job.job_id, error))

    threads = [
        threading.Thread(target=_worker, args=(gpu,), name=f"awwl-worker-{gpu}", daemon=True)
        for gpu in gpus
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    finally:
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, previous_sigint)

    return report


def _run_job(
    job: Job,
    *,
    store: JobStore,
    gpu: str,
    log_path: Path,
    heartbeat_interval: float,
    stop: _Stop,
    cwd: str | Path | None,
) -> str | None:
    """Run one job's subprocess. Returns ``None`` on success, else the error."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTHONUNBUFFERED", "1")

    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} attempt={job.attempts} ===\n"
        )
        log.write(" ".join(job.argv) + "\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                job.argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(cwd) if cwd else None,
            )
        except OSError as exc:
            return f"could not start job: {exc}"

        while True:
            try:
                code = proc.wait(timeout=heartbeat_interval)
                break
            except subprocess.TimeoutExpired:
                store.heartbeat(job.job_id)
                if stop.requested:
                    logger.info("stopping %s", job.job_id)
                    _terminate(proc)
                    return "interrupted before completion"

    elapsed = time.time() - started
    if code == 0:
        logger.info("[gpu%s] done %s in %.1f min", gpu, job.job_id, elapsed / 60)
        return None
    tail = _tail(log_path)
    return f"exit code {code} after {elapsed / 60:.1f} min\n{tail}"


def _terminate(proc: subprocess.Popen) -> None:
    """Ask a child to exit, then insist."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()
        proc.wait(timeout=10)


def _tail(path: Path, *, lines: int = 25) -> str:
    """Last few log lines, for the stored error message."""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:  # pragma: no cover
        return "(log unavailable)"


def _safe_name(job_id: str) -> str:
    """Job ids contain ``:``, which is not a legal filename character on Windows."""
    return job_id.replace(":", "__").replace("/", "_")


def format_status(store: JobStore) -> str:
    """Human-readable summary of the queue, including failures."""
    counts = store.counts()
    total = sum(counts.values())
    order = ("done", "running", "pending", "failed")
    parts = [f"{k}={counts.get(k, 0)}" for k in order if counts.get(k)]
    lines = [f"{total} job(s): " + (", ".join(parts) if parts else "none")]

    failed = store.list_jobs(status=FAILED)
    if failed:
        lines.append("")
        lines.append("failed:")
        for job in failed:
            lines.append(f"  {job.job_id}  (after {job.attempts} attempt(s))")
        lines.append("")
        lines.append("retry them with:  awwl pipeline reset --manifest <manifest>")

    remaining = [j for j in store.list_jobs() if j.status not in (DONE, FAILED, RETIRED)]
    retired = counts.get(RETIRED, 0)
    if retired:
        lines.append("")
        lines.append(f"retired: {retired} (no longer in the manifest; they are never claimed)")
    if remaining:
        by_kind: dict[str, int] = {}
        for job in remaining:
            by_kind[job.kind] = by_kind.get(job.kind, 0) + 1
        lines.append("")
        lines.append("remaining: " + ", ".join(f"{k}×{v}" for k, v in sorted(by_kind.items())))
    return "\n".join(lines)


def default_gpus() -> list[str]:
    """Every visible CUDA device, or ``["0"]`` when detection fails."""
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:  # pragma: no cover - torch always present in practice
        count = 0
    if count <= 0:
        print("no CUDA devices detected; running a single worker", file=sys.stderr)
        return ["0"]
    return [str(i) for i in range(count)]
