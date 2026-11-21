"""SQLite storage for UTC-day-partitioned HyperLogLog summaries.

Design notes
------------
* ``submissions`` is an append-only audit log of accepted upload batches.
* ``partition_revisions`` keeps every revision of a UTC day, including
  superseded ones, so late data remains traceable and immutable.
* ``active_partitions`` is the current view: exactly one revision per day.
  One upload batch is one transaction, so a batch covering several days either
  becomes active for all of them or for none.  A late revision therefore
  replaces the *whole batch* of partitions it carries, atomically.
* A query copies the exact register bytes, precision, revision and revision
  id of the active partitions into ``snapshot_partitions`` and computes the
  answer in the same ``BEGIN IMMEDIATE`` transaction.  That transaction is the
  only writer, so no revision can land between pinning and snapshot storage
  and an answer never mixes pre- and post-revision state.
* A late revision marks every snapshot that adopted one of its days as
  invalidated.  Cached answers for invalidated snapshots are no longer
  reused, but remain inspectable through ``GET /v1/snapshots/{id}``.
* Writers use ``BEGIN IMMEDIATE`` under a process lock; SQLite's single
  writer plus a busy timeout serializes uploads across processes, while WAL
  lets readers proceed concurrently.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import sqlite3
import struct
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .hll import (
    HEADER_SIZE,
    MAGIC,
    Summary,
    error_bound,
    estimate_registers,
    max_rank,
    merge_register_bytes,
    register_count,
)

SCHEMA = """
PRAGMA busy_timeout = 10000;
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS submissions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key       TEXT    NOT NULL UNIQUE,
    revision              INTEGER NOT NULL,
    precision             INTEGER NOT NULL,
    first_partition_date  TEXT    NOT NULL,
    partition_count       INTEGER NOT NULL,
    payload_sha256        TEXT    NOT NULL,
    created_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS partition_revisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    partition_date    TEXT    NOT NULL,
    revision          INTEGER NOT NULL,
    precision         INTEGER NOT NULL,
    encoding_version  INTEGER NOT NULL,
    registers         BLOB    NOT NULL,
    checksum          INTEGER NOT NULL,
    submission_id     INTEGER NOT NULL REFERENCES submissions(id),
    created_at        TEXT    NOT NULL,
    UNIQUE (partition_date, revision)
);

CREATE INDEX IF NOT EXISTS idx_partition_revisions_lookup
    ON partition_revisions (partition_date, revision DESC);

CREATE TABLE IF NOT EXISTS active_partitions (
    partition_date    TEXT    NOT NULL PRIMARY KEY,
    revision          INTEGER NOT NULL,
    revision_id       INTEGER NOT NULL REFERENCES partition_revisions(id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date          TEXT    NOT NULL,
    end_date            TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    invalidated_at      TEXT,
    invalidation_reason TEXT
);

CREATE TABLE IF NOT EXISTS snapshot_partitions (
    snapshot_id       INTEGER NOT NULL REFERENCES snapshots(id),
    partition_date    TEXT    NOT NULL,
    revision          INTEGER NOT NULL,
    revision_id       INTEGER NOT NULL REFERENCES partition_revisions(id),
    precision         INTEGER NOT NULL,
    encoding_version  INTEGER NOT NULL,
    registers         BLOB    NOT NULL,
    PRIMARY KEY (snapshot_id, partition_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_partitions_day
    ON snapshot_partitions (partition_date);

CREATE TABLE IF NOT EXISTS query_results (
    snapshot_id       INTEGER PRIMARY KEY REFERENCES snapshots(id),
    start_date        TEXT    NOT NULL,
    end_date          TEXT    NOT NULL,
    response_json     TEXT    NOT NULL,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_results_range
    ON query_results (start_date, end_date, snapshot_id DESC);

-- The deployment-wide precision and encoding version.  The first accepted
-- revision pins them; every later upload must match so all partitions in the
-- store are always mergeable.
CREATE TABLE IF NOT EXISTS service_config (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    precision         INTEGER NOT NULL,
    encoding_version  INTEGER NOT NULL,
    pinned_at         TEXT    NOT NULL
);
"""

UTC = timezone.utc
MAX_RANGE_DAYS = 366 * 5


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class StorageError(Exception):
    """Base class for errors that map to specific HTTP responses."""

    code = "storage_error"
    status_code = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidRange(StorageError):
    code = "invalid_range"
    status_code = 400


class MissingPartitions(StorageError):
    code = "missing_partitions"
    status_code = 404

    def __init__(self, missing: Sequence[date]) -> None:
        self.missing = [day.isoformat() for day in missing]
        super().__init__(
            "no active partition for one or more UTC days in the range",
            missing_dates=self.missing,
        )


class PrecisionMismatch(StorageError):
    code = "precision_mismatch"
    status_code = 422

    def __init__(
        self,
        precisions: Sequence[int],
        versions: Sequence[int],
        *,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message
            or (
                "active partitions use incompatible precision or encoding "
                "version; refusing to merge"
            ),
            precisions=sorted(set(precisions)),
            encoding_versions=sorted(set(versions)),
        )


class PrecisionConflict(PrecisionMismatch):
    """An upload disagrees with the deployment-wide precision/version."""

    code = "precision_conflict"


class EncodingVersionConflict(PrecisionMismatch):
    code = "encoding_version_conflict"


class StaleRevision(StorageError):
    code = "stale_revision"
    status_code = 409

    def __init__(self, partition_date: date, active_revision: int, revision: int) -> None:
        super().__init__(
            f"revision {revision} for {partition_date.isoformat()} is not newer "
            f"than the active revision {active_revision}",
            partition_date=partition_date.isoformat(),
            active_revision=active_revision,
            revision=revision,
        )


class RevisionConflict(StorageError):
    code = "revision_conflict"
    status_code = 409

    def __init__(self, partition_date: date, revision: int) -> None:
        super().__init__(
            f"revision {revision} for {partition_date.isoformat()} already "
            "exists with different content; revisions are immutable",
            partition_date=partition_date.isoformat(),
            revision=revision,
        )


class StoredSummaryCorrupt(StorageError):
    code = "stored_summary_corrupt"
    status_code = 500


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise InvalidRange("end_date must be on or after start_date")
    span = (end - start).days + 1
    if span > MAX_RANGE_DAYS:
        raise InvalidRange(f"date range may not exceed {MAX_RANGE_DAYS} UTC days")
    return [start + timedelta(days=offset) for offset in range(span)]


def payload_digest(
    revision: int, precision: int, summaries: Sequence[tuple[date, Summary]]
) -> str:
    digest = hashlib.sha256()
    digest.update(f"v1\n{revision}\n{precision}\n".encode())
    for day, summary in sorted(summaries, key=lambda item: item[0]):
        digest.update(day.isoformat().encode())
        digest.update(b"\x00")
        digest.update(str(summary.encoding_version).encode())
        digest.update(b"\x00")
        digest.update(summary.registers)
        digest.update(b"\x01")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------


class Database:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        self._write_lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # -- connections -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        connection.execute("BEGIN")
        try:
            yield connection
        finally:
            connection.rollback()
            connection.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
            finally:
                connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(SCHEMA)
        finally:
            connection.close()
    # -- ingest ----------------------------------------------------------

    def submit_revision(
        self,
        *,
        revision: int,
        precision: int,
        summaries: Sequence[tuple[date, Summary]],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Atomically apply one revision batch.

        ``summaries`` holds ``(day, summary)`` pairs; every summary must share
        the batch ``precision``.  Validation happens before this call; this
        method enforces revision ordering and immutability.
        """

        if not summaries:
            raise ValueError("a revision must include at least one partition")

        digest = payload_digest(revision, precision, summaries)
        now = utc_now_iso()

        with self.write() as connection:
            existing = connection.execute(
                "SELECT * FROM submissions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != digest:
                    return {
                        "status": "idempotency_conflict",
                        "submission_id": existing["id"],
                        "message": (
                            "idempotency key was already used with a different payload"
                        ),
                    }
                return {
                    "status": "duplicate",
                    "submission_id": existing["id"],
                    "revision": existing["revision"],
                    "precision": existing["precision"],
                    "payload_sha256": digest,
                    "accepted_partitions": [],
                    "superseded_revisions": [],
                    "invalidated_snapshots": [],
                }

            # Validate ordering for every day before writing anything so a
            # partially applicable batch cannot leave the DB changed.
            pin = connection.execute(
                "SELECT precision, encoding_version FROM service_config WHERE id = 1"
            ).fetchone()
            pinned_versions = (pin["precision"], pin["encoding_version"]) if pin else None
            for day, summary in summaries:
                if (
                    pinned_versions is not None
                    and summary.precision != pinned_versions[0]
                ):
                    raise PrecisionConflict(
                        [pinned_versions[0], summary.precision],
                        [pinned_versions[1], summary.encoding_version],
                        message=(
                            f"this store only accepts precision {pinned_versions[0]}; "
                            f"partition {day.isoformat()} declares precision "
                            f"{summary.precision}"
                        ),
                    )
                if (
                    pinned_versions is not None
                    and summary.encoding_version != pinned_versions[1]
                ):
                    raise EncodingVersionConflict(
                        [pinned_versions[0], summary.precision],
                        [pinned_versions[1], summary.encoding_version],
                        message=(
                            f"this store only accepts encoding version "
                            f"{pinned_versions[1]}; partition {day.isoformat()} "
                            f"declares {summary.encoding_version}"
                        ),
                    )

            prior_by_day: dict[date, sqlite3.Row] = {}
            for day, _summary in summaries:
                prior = connection.execute(
                    "SELECT revision, revision_id FROM active_partitions "
                    "WHERE partition_date = ?",
                    (day.isoformat(),),
                ).fetchone()
                if prior is not None:
                    if revision < prior["revision"]:
                        raise StaleRevision(day, prior["revision"], revision)
                    if revision == prior["revision"]:
                        same = connection.execute(
                            "SELECT registers, precision, encoding_version "
                            "FROM partition_revisions "
                            "WHERE partition_date = ? AND revision = ?",
                            (day.isoformat(), revision),
                        ).fetchone()
                        target = dict(summaries).get(day)
                        if (
                            same is None
                            or target is None
                            or bytes(same["registers"]) != target.registers
                            or same["precision"] != target.precision
                            or same["encoding_version"] != target.encoding_version
                        ):
                            raise RevisionConflict(day, revision)
                prior_by_day[day] = prior

            submission_id = int(
                connection.execute(
                    "INSERT INTO submissions "
                    "(idempotency_key, revision, precision, first_partition_date, "
                    " partition_count, payload_sha256, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        idempotency_key,
                        revision,
                        precision,
                        min(day for day, _ in summaries).isoformat(),
                        len(summaries),
                        digest,
                        now,
                    ),
                ).lastrowid
            )

            if pinned_versions is None:
                connection.execute(
                    "INSERT INTO service_config (id, precision, encoding_version, "
                    "pinned_at) VALUES (1, ?, ?, ?)",
                    (precision, summaries[0][1].encoding_version, now),
                )

            accepted: list[dict[str, Any]] = []
            superseded: list[dict[str, Any]] = []
            changed_days: list[date] = []

            for day, summary in sorted(summaries, key=lambda item: item[0]):
                revision_id = self._insert_partition_revision(
                    connection,
                    day=day,
                    revision=revision,
                    summary=summary,
                    submission_id=submission_id,
                    created_at=now,
                )
                connection.execute(
                    "INSERT INTO active_partitions "
                    "(partition_date, revision, revision_id) VALUES (?, ?, ?) "
                    "ON CONFLICT (partition_date) DO UPDATE SET "
                    "revision = excluded.revision, revision_id = excluded.revision_id",
                    (day.isoformat(), revision, revision_id),
                )

                prior = prior_by_day.get(day)
                if prior is not None and prior["revision"] != revision:
                    superseded.append(
                        {
                            "partition_date": day.isoformat(),
                            "replaced_revision": prior["revision"],
                            "applied_revision": revision,
                        }
                    )
                if prior is None or prior["revision_id"] != revision_id:
                    changed_days.append(day)

                accepted.append(
                    {
                        "partition_date": day.isoformat(),
                        "revision": revision,
                        "revision_id": revision_id,
                        "precision": summary.precision,
                    }
                )

            invalidated = self._invalidate_snapshots(
                connection,
                days=changed_days,
                reason=(
                    f"revision {revision} replaced {len(changed_days)} partition(s)"
                ),
                now=now,
            )

        return {
            "status": "unchanged" if not changed_days else "accepted",
            "submission_id": submission_id,
            "revision": revision,
            "precision": precision,
            "payload_sha256": digest,
            "accepted_partitions": accepted,
            "superseded_revisions": superseded,
            "invalidated_snapshots": invalidated,
        }

    def _insert_partition_revision(
        self,
        connection: sqlite3.Connection,
        *,
        day: date,
        revision: int,
        summary: Summary,
        submission_id: int,
        created_at: str,
    ) -> int:
        existing = connection.execute(
            "SELECT id, precision, encoding_version, registers "
            "FROM partition_revisions WHERE partition_date = ? AND revision = ?",
            (day.isoformat(), revision),
        ).fetchone()
        if existing is not None:
            if (
                existing["precision"] != summary.precision
                or existing["encoding_version"] != summary.encoding_version
                or bytes(existing["registers"]) != summary.registers
            ):
                raise RevisionConflict(day, revision)
            return int(existing["id"])

        cursor = connection.execute(
            "INSERT INTO partition_revisions "
            "(partition_date, revision, precision, encoding_version, registers, "
            " checksum, submission_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                day.isoformat(),
                revision,
                summary.precision,
                summary.encoding_version,
                summary.registers,
                summary.checksum,
                submission_id,
                created_at,
            ),
        )
        return int(cursor.lastrowid)

    def _invalidate_snapshots(
        self,
        connection: sqlite3.Connection,
        *,
        days: Sequence[date],
        reason: str,
        now: str,
    ) -> list[int]:
        if not days:
            return []
        placeholders = ",".join("?" for _ in days)
        rows = connection.execute(
            "SELECT DISTINCT snapshot_id FROM snapshot_partitions "
            f"WHERE partition_date IN ({placeholders}) "
            "ORDER BY snapshot_id",
            [day.isoformat() for day in days],
        ).fetchall()
        snapshot_ids = [int(row["snapshot_id"]) for row in rows]
        if not snapshot_ids:
            return []
        id_placeholders = ",".join("?" for _ in snapshot_ids)
        connection.execute(
            "UPDATE snapshots SET invalidated_at = ?, invalidation_reason = ? "
            f"WHERE id IN ({id_placeholders}) AND invalidated_at IS NULL",
            [now, reason, *snapshot_ids],
        )
        return snapshot_ids

    # -- query -----------------------------------------------------------

    def estimate(self, *, start: date, end: date) -> dict[str, Any]:
        """Answer a range query from one immutable committed snapshot.

        A still-valid cached answer is served straight from a read
        transaction.  Otherwise the query runs entirely inside one
        ``BEGIN IMMEDIATE`` transaction: it pins the active partitions,
        checks the cache again, merges, and stores the snapshot.  Because
        that transaction is the only writer, no revision can land between
        pinning and snapshot storage, so an answer can never mix revisions.
        """

        days = date_range(start, end)

        with self.read() as connection:
            cached = self._cached_response(connection, start, end)
        if cached is not None:
            return cached

        with self.write() as connection:
            pinned = self._pin_active(connection, days)

            # A concurrent query may have committed an equivalent snapshot
            # while this one waited for the writer lock.
            cached = self._cached_response(connection, start, end)
            if cached is not None:
                return cached

            return self._write_snapshot(
                connection, start=start, end=end, pinned=pinned
            )

    @staticmethod
    def _cached_response(
        connection: sqlite3.Connection, start: date, end: date
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT qr.response_json FROM query_results qr "
            "JOIN snapshots s ON s.id = qr.snapshot_id "
            "WHERE qr.start_date = ? AND qr.end_date = ? "
            "  AND s.invalidated_at IS NULL "
            "ORDER BY qr.snapshot_id DESC LIMIT 1",
            (start.isoformat(), end.isoformat()),
        ).fetchone()
        if row is None:
            return None
        response = json.loads(row["response_json"])
        response["reused_snapshot"] = True
        return response

    def _pin_active(
        self, connection: sqlite3.Connection, days: Sequence[date]
    ) -> tuple[tuple[Any, ...], ...]:
        placeholders = ",".join("?" for _ in days)
        rows = connection.execute(
            "SELECT a.partition_date, a.revision, a.revision_id, r.precision, "
            "       r.encoding_version, r.registers, r.checksum "
            "FROM active_partitions a "
            "JOIN partition_revisions r ON r.id = a.revision_id "
            f"WHERE a.partition_date IN ({placeholders}) "
            "ORDER BY a.partition_date",
            [day.isoformat() for day in days],
        ).fetchall()

        present = {row["partition_date"] for row in rows}
        missing = [day for day in days if day.isoformat() not in present]
        if missing:
            raise MissingPartitions(missing)

        precisions = {row["precision"] for row in rows}
        versions = {row["encoding_version"] for row in rows}
        if len(precisions) != 1 or len(versions) != 1:
            raise PrecisionMismatch(precisions, versions)

        return tuple(
            (
                row["partition_date"],
                row["revision"],
                row["revision_id"],
                row["precision"],
                row["encoding_version"],
                bytes(row["registers"]),
                row["checksum"],
            )
            for row in rows
        )

    def _write_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        start: date,
        end: date,
        pinned: Sequence[tuple[Any, ...]],
    ) -> dict[str, Any]:
        # Validate and merge before touching the snapshot tables.  A
        # revision's registers are immutable, so validation against the
        # pinned bytes is enough; no concurrent writer can invalidate them.
        precision = pinned[0][3]
        version = pinned[0][4]
        for (
            _day,
            _revision,
            revision_id,
            part_precision,
            part_version,
            registers,
            checksum,
        ) in pinned:
            self._validate_persisted(
                revision_id, part_precision, part_version, registers, checksum
            )
        merged = merge_register_bytes((item[5] for item in pinned), precision)
        value = estimate_registers(merged)
        bounds = error_bound(value, precision)

        now = utc_now_iso()
        snapshot_id = int(
            connection.execute(
                "INSERT INTO snapshots (start_date, end_date, created_at) "
                "VALUES (?, ?, ?)",
                (start.isoformat(), end.isoformat(), now),
            ).lastrowid
        )

        partitions: list[dict[str, Any]] = []
        for (
            day,
            revision,
            revision_id,
            part_precision,
            part_version,
            registers,
            _checksum,
        ) in pinned:
            connection.execute(
                "INSERT INTO snapshot_partitions "
                "(snapshot_id, partition_date, revision, revision_id, "
                " precision, encoding_version, registers) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    day,
                    revision,
                    revision_id,
                    part_precision,
                    part_version,
                    registers,
                ),
            )
            partitions.append(
                {
                    "partition_date": day,
                    "revision": revision,
                    "revision_id": revision_id,
                    "precision": part_precision,
                }
            )

        response = {
            "snapshot_id": snapshot_id,
            "reused_snapshot": False,
            "query": {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
            "estimated_unique_users": int(round(value)),
            "estimate": value,
            "precision": precision,
            "encoding_version": version,
            "error_bound": bounds,
            "partitions_adopted": partitions,
            "partition_count": len(partitions),
            "created_at": now,
        }
        connection.execute(
            "INSERT INTO query_results "
            "(snapshot_id, start_date, end_date, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                snapshot_id,
                start.isoformat(),
                end.isoformat(),
                json.dumps(response),
                now,
            ),
        )
        return response

    @staticmethod
    def _validate_persisted(
        revision_id: int,
        precision: int,
        version: int,
        registers: bytes,
        checksum: int,
    ) -> None:
        """Re-check persisted registers so DB bit rot cannot skew an answer.

        The CRC over header + registers detects any byte change.  The rank
        check catches rows written by code that bypassed the encoder.  Both
        use C-level primitives so a year of p=14 partitions validates in well
        under a second.
        """

        expected = register_count(precision)
        if len(registers) != expected:
            raise StoredSummaryCorrupt(
                f"stored summary {revision_id} has {len(registers)} registers, "
                f"expected {expected}"
            )
        body = (
            MAGIC
            + struct.pack(">BBHI", version, precision, 0, expected)
            + bytes(registers)
        )
        if len(body) != HEADER_SIZE + expected:
            raise StoredSummaryCorrupt(
                f"stored summary {revision_id} has an inconsistent header"
            )
        actual = binascii.crc32(body) & 0xFFFFFFFF
        if actual != checksum:
            raise StoredSummaryCorrupt(
                f"stored summary {revision_id} checksum mismatch"
            )
        if registers and max(registers) > max_rank(precision):
            raise StoredSummaryCorrupt(
                f"stored summary {revision_id} has a register above the maximum rank"
            )

    # -- inspection ------------------------------------------------------

    def snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
            if row is None:
                return None
            partitions = connection.execute(
                "SELECT partition_date, revision, revision_id, precision, "
                "encoding_version FROM snapshot_partitions "
                "WHERE snapshot_id = ? ORDER BY partition_date",
                (snapshot_id,),
            ).fetchall()
            cached = connection.execute(
                "SELECT response_json FROM query_results WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            return {
                "snapshot_id": snapshot_id,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "created_at": row["created_at"],
                "still_valid": row["invalidated_at"] is None,
                "invalidated_at": row["invalidated_at"],
                "invalidation_reason": row["invalidation_reason"],
                "partitions_adopted": [dict(partition) for partition in partitions],
                "response": json.loads(cached["response_json"]) if cached else None,
            }

    def active_partitions(self) -> list[dict[str, Any]]:
        with self.read() as connection:
            rows = connection.execute(
                "SELECT a.partition_date, a.revision, a.revision_id, r.precision, "
                "r.encoding_version FROM active_partitions a "
                "JOIN partition_revisions r ON r.id = a.revision_id "
                "ORDER BY a.partition_date"
            ).fetchall()
            return [dict(row) for row in rows]

    def health(self) -> dict[str, Any]:
        start_time = time.perf_counter()
        with self.read() as connection:
            connection.execute("SELECT 1").fetchone()
            active = connection.execute(
                "SELECT COUNT(*) AS n FROM active_partitions"
            ).fetchone()["n"]
            snapshots = connection.execute(
                "SELECT COUNT(*) AS n FROM snapshots"
            ).fetchone()["n"]
        return {
            "database": "ok",
            "active_partitions": active,
            "snapshots": snapshots,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }
