"""SQLite-backed job store: the piece that makes the pipeline crash-proof.

The store is the single source of truth for what has run. Workers do not talk
to each other; they each claim jobs out of one SQLite file, which gives three
properties that matter when a long sweep shares a machine with other people:

* **Atomic claim.** ``BEGIN IMMEDIATE`` serialises the select-then-update, so
  two workers can never take the same job even when they poll simultaneously.
* **Crash recovery.** A running job records a heartbeat. If the process (or
  the whole machine) dies, the heartbeat goes stale and
  :meth:`JobStore.reclaim_stale` puts the job back in the queue on the next
  start. Nothing is lost except the work since the job's own last checkpoint.
* **Idempotent restart.** Finished jobs stay ``done``. Re-running the pipeline
  after a crash resumes the sweep rather than recomputing it.

SQLite in WAL mode is used deliberately over a job queue daemon: it is in the
standard library, survives a hard power cut, and the state is one file you can
copy off the server and inspect with ``sqlite3``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def store_path(output_root: str | Path) -> Path:
    """Where a pipeline's job database lives, given its ``output_root``.

    One definition, because the name was previously written out at each call
    site and a reader that guessed it wrong found no database and silently
    reported no queue state at all -- the failure mode of a path that is a
    convention rather than a function.
    """
    return Path(output_root) / "pipeline" / "state.db"

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    pipeline    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    group_id    TEXT NOT NULL,
    tier        INTEGER NOT NULL DEFAULT 1,
    depends_on  TEXT,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    worker      TEXT,
    heartbeat   REAL,
    started_at  REAL,
    finished_at REAL,
    error       TEXT,
    ordering    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, tier, ordering);
"""


@dataclass
class Job:
    """One unit of work: a subprocess invocation plus its bookkeeping."""

    job_id: str
    pipeline: str
    kind: str
    group_id: str
    tier: int
    depends_on: str | None
    payload: dict[str, Any]
    status: str
    attempts: int
    # Why the job failed, as recorded by `finish`. Carried on the Job so a
    # reader does not have to open the database to find out.
    error: str | None = None

    @property
    def argv(self) -> list[str]:
        """The command this job runs, as an argv list."""
        return list(self.payload["argv"])


class JobStore:
    """Persistent queue of pipeline jobs.

    Args:
        db_path: Location of the SQLite file. Created with its parents.
        stale_after: Seconds without a heartbeat before a ``running`` job is
            considered dead. Must comfortably exceed the heartbeat interval;
            the default tolerates a machine that freezes for a few minutes.
        max_attempts: How many times a job may be retried before it is parked
            as ``failed``. Covers transient faults (a CUDA OOM caused by
            another tenant) without looping forever on a real bug.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        stale_after: float = 900.0,
        max_attempts: int = 3,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.stale_after = float(stale_after)
        self.max_attempts = int(max_attempts)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=60.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    # ----------------------------------------------------------- population

    def add_jobs(self, jobs: list[Job], *, refresh: bool = True) -> int:
        """Insert jobs, ignoring any whose ``job_id`` is already present.

        Re-running the same manifest is therefore safe and additive: new
        experiments appear, existing progress is untouched.

        Args:
            refresh: Also update the stored command of jobs that have **not**
                succeeded. Without this, a job's argv is frozen at the moment
                it was first queued, so correcting a wrong path in the
                manifest and re-running changes nothing and the job fails
                again with the identical error — which is exactly what
                happened to 25 DreamBooth evaluations pointed at a dataset
                directory that did not exist. Finished jobs are never touched,
                so refreshing cannot silently invalidate a completed result.
        """
        inserted = 0
        refreshed = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for order, job in enumerate(jobs):
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                        (job_id, pipeline, kind, group_id, tier, depends_on, payload,
                         status, attempts, ordering)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        job.job_id,
                        job.pipeline,
                        job.kind,
                        job.group_id,
                        job.tier,
                        job.depends_on,
                        json.dumps(job.payload),
                        PENDING,
                        order,
                    ),
                )
                inserted += cur.rowcount
                if refresh and cur.rowcount == 0:
                    # The guard must cover every column being written, not just
                    # the payload. Gating on `payload != ?` alone meant a
                    # manifest edit that changed only a job's tier never
                    # propagated: moving `perceptual` from tier 3 to tier 5 left
                    # the stored tier at 3, so `--max-tier 4` kept selecting it
                    # and the demotion silently did nothing.
                    payload = json.dumps(job.payload)
                    updated = conn.execute(
                        """
                        UPDATE jobs SET payload = ?, tier = ?, depends_on = ?
                         WHERE job_id = ? AND status != ?
                           AND (payload != ? OR tier != ? OR depends_on IS NOT ?)
                        """,
                        (
                            payload,
                            job.tier,
                            job.depends_on,
                            job.job_id,
                            DONE,
                            payload,
                            job.tier,
                            job.depends_on,
                        ),
                    )
                    refreshed += updated.rowcount
            conn.execute("COMMIT")
        if refreshed:
            logger.info("refreshed %d unfinished job(s) from the manifest", refreshed)
        return inserted

    # ---------------------------------------------------------------- claim

    def claim(self, worker: str, *, max_tier: int | None = None) -> Job | None:
        """Atomically take the next runnable job, or return ``None``.

        A job is runnable when it is ``pending``, within ``max_tier``, and its
        dependency (if any) is ``done``. Ordering is by tier first so the
        decisive low-tier experiments finish before the exploratory ones start.
        """
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = ?
                      AND (? IS NULL OR tier <= ?)
                      AND (depends_on IS NULL
                           OR depends_on IN (SELECT job_id FROM jobs WHERE status = ?))
                    ORDER BY tier, ordering
                    LIMIT 1
                    """,
                    (PENDING, max_tier, max_tier, DONE),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    """
                    UPDATE jobs
                       SET status = ?, worker = ?, heartbeat = ?, started_at = ?,
                           attempts = attempts + 1, error = NULL
                     WHERE job_id = ?
                    """,
                    (RUNNING, worker, now, now, row["job_id"]),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return _to_job(row, status=RUNNING, attempts=row["attempts"] + 1)

    def heartbeat(self, job_id: str) -> None:
        """Record that ``job_id`` is still alive."""
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET heartbeat = ? WHERE job_id = ?", (time.time(), job_id))

    def finish(self, job_id: str, *, error: str | None = None) -> None:
        """Mark a job ``done``, or ``failed``/``pending`` when it errored.

        A job that has attempts left goes back to ``pending`` for another
        worker to retry; one that has exhausted them is parked as ``failed``
        so the sweep continues instead of blocking on it.
        """
        now = time.time()
        with self._connect() as conn:
            if error is None:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error = NULL WHERE job_id = ?",
                    (DONE, now, job_id),
                )
                return
            row = conn.execute("SELECT attempts FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            attempts = int(row["attempts"]) if row else self.max_attempts
            status = FAILED if attempts >= self.max_attempts else PENDING
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE job_id = ?",
                (status, now, error[-4000:], job_id),
            )
            logger.warning("job %s -> %s (attempt %d): %s", job_id, status, attempts, error[:200])

    # ------------------------------------------------------------- recovery

    def reclaim_stale(self) -> int:
        """Return dead ``running`` jobs to the queue. Call this at worker start.

        This is what turns "the server rebooted" into "the sweep continues":
        without it, every job that was in flight during the crash would sit in
        ``running`` forever and the pipeline would stall.
        """
        cutoff = time.time() - self.stale_after
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT job_id, attempts FROM jobs WHERE status = ? AND (heartbeat IS NULL OR heartbeat < ?)",
                (RUNNING, cutoff),
            ).fetchall()
            for row in rows:
                status = FAILED if int(row["attempts"]) >= self.max_attempts else PENDING
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                    (status, "reclaimed after stale heartbeat", row["job_id"]),
                )
            conn.execute("COMMIT")
        if rows:
            logger.info("reclaimed %d stale job(s) from a previous run", len(rows))
        return len(rows)

    def reset(self, *, statuses: tuple[str, ...] = (FAILED,)) -> int:
        """Put jobs in ``statuses`` back to ``pending`` with attempts cleared."""
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET status = ?, attempts = 0, error = NULL WHERE status IN ({placeholders})",
                (PENDING, *statuses),
            )
            return cur.rowcount

    # -------------------------------------------------------------- reading

    def counts(self) -> dict[str, int]:
        """Job count per status."""
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def list_jobs(self, *, status: str | None = None) -> list[Job]:
        """All jobs, optionally filtered by status, in execution order."""
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY tier, ordering"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_job(r) for r in rows]

    def pending_work(self, *, max_tier: int | None = None) -> int:
        """How many jobs are still unfinished (pending or running) within tier."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM jobs
                 WHERE status IN (?, ?) AND (? IS NULL OR tier <= ?)
                """,
                (PENDING, RUNNING, max_tier, max_tier),
            ).fetchone()
        return int(row["n"])


def _to_job(row: sqlite3.Row, *, status: str | None = None, attempts: int | None = None) -> Job:
    return Job(
        job_id=row["job_id"],
        pipeline=row["pipeline"],
        kind=row["kind"],
        group_id=row["group_id"],
        tier=int(row["tier"]),
        depends_on=row["depends_on"],
        payload=json.loads(row["payload"]),
        status=status if status is not None else row["status"],
        attempts=attempts if attempts is not None else int(row["attempts"]),
        error=row["error"],
    )
