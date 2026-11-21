"""Storage-level tests: snapshot reproducibility, rollback, corruption, races."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest

import app.db as db_module
from app.db import (
    Database,
    InvalidRange,
    MissingPartitions,
    RevisionConflict,
    StaleRevision,
    StoredSummaryCorrupt,
)
from app.hll import (
    Summary,
    build_registers,
    decode_summary,
    encode_summary,
)

from .conftest import upload_body


def summary(values, precision=14) -> Summary:
    return decode_summary(encode_summary(build_registers(values, precision), precision))


def test_snapshot_is_replayable_after_late_revision(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    db.submit_revision(
        revision=1,
        precision=14,
        summaries=[
            (date(2026, 11, 1), summary([f"a{i}" for i in range(500)])),
            (date(2026, 11, 2), summary([f"b{i}" for i in range(500)])),
        ],
        idempotency_key="r1",
    )
    first = db.estimate(start=date(2026, 11, 1), end=date(2026, 11, 2))
    first_estimate = first["estimate"]

    db.submit_revision(
        revision=2,
        precision=14,
        summaries=[(date(2026, 11, 2), summary([f"c{i}" for i in range(10)]))],
        idempotency_key="r2",
    )

    record = db.snapshot(first["snapshot_id"])
    assert record is not None
    assert record["still_valid"] is False
    # The original answer and registers are preserved verbatim.
    assert record["response"]["estimate"] == first_estimate
    assert record["response"]["partitions_adopted"] == first["partitions_adopted"]
    assert record["invalidated_at"]

    # A fresh query recomputes from the new active revision.
    second = db.estimate(start=date(2026, 11, 1), end=date(2026, 11, 2))
    assert second["snapshot_id"] != first["snapshot_id"]
    by_day = {p["partition_date"]: p["revision"] for p in second["partitions_adopted"]}
    assert by_day == {"2026-11-01": 1, "2026-11-02": 2}


def test_revision_rollback_leaves_no_partial_batch(tmp_path):
    path = tmp_path / "db.sqlite3"
    db = Database(path)
    db.submit_revision(
        revision=5,
        precision=14,
        summaries=[(date(2026, 11, 1), summary(["a"]))],
        idempotency_key="r5",
    )
    with pytest.raises(StaleRevision):
        db.submit_revision(
            revision=4,
            precision=14,
            summaries=[
                (date(2026, 11, 1), summary(["changed"])),
                (date(2026, 11, 2), summary(["new"])),
            ],
            idempotency_key="stale",
        )
    active = {row["partition_date"]: row for row in db.active_partitions()}
    assert list(active) == ["2026-11-01"]
    assert active["2026-11-01"]["revision"] == 5

    with sqlite3.connect(path) as connection:
        submissions = connection.execute(
            "SELECT COUNT(*) FROM submissions WHERE idempotency_key = 'stale'"
        ).fetchone()[0]
        revisions = connection.execute(
            "SELECT COUNT(*) FROM partition_revisions"
        ).fetchone()[0]
    assert submissions == 0
    assert revisions == 1


def test_revision_reuse_with_different_content_raises(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    db.submit_revision(
        revision=1,
        precision=14,
        summaries=[(date(2026, 11, 1), summary(["a"]))],
        idempotency_key="r1",
    )
    with pytest.raises(RevisionConflict):
        db.submit_revision(
            revision=1,
            precision=14,
            summaries=[(date(2026, 11, 1), summary(["b"]))],
            idempotency_key="r1-again",
        )


def test_stored_register_corruption_is_detected(tmp_path):
    path = tmp_path / "db.sqlite3"
    db = Database(path)
    db.submit_revision(
        revision=1,
        precision=14,
        summaries=[(date(2026, 11, 1), summary(["a"]))],
        idempotency_key="r1",
    )
    with sqlite3.connect(path) as connection:
        original = connection.execute(
            "SELECT registers FROM partition_revisions"
        ).fetchone()[0]
        tampered = bytearray(original)
        tampered[0] = 255
        connection.execute(
            "UPDATE partition_revisions SET registers = ?", (bytes(tampered),)
        )
        connection.commit()

    with pytest.raises(StoredSummaryCorrupt) as excinfo:
        db.estimate(start=date(2026, 11, 1), end=date(2026, 11, 1))
    assert "checksum mismatch" in str(excinfo.value)


def test_missing_partition_and_unpinned_precision_checks(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    db.submit_revision(
        revision=1,
        precision=12,
        summaries=[(date(2026, 11, 1), summary(["a"], 12))],
        idempotency_key="r1",
    )
    with pytest.raises(MissingPartitions):
        db.estimate(start=date(2026, 11, 1), end=date(2026, 11, 2))
    with pytest.raises(InvalidRange):
        db.estimate(start=date(2026, 11, 2), end=date(2026, 11, 1))


def test_query_pins_and_commits_inside_one_write_transaction(tmp_path, monkeypatch):
    """A revision cannot land between pinning and snapshot storage."""

    db = Database(tmp_path / "db.sqlite3")
    db.submit_revision(
        revision=1,
        precision=14,
        summaries=[(date(2026, 11, 1), summary(["a"]))],
        idempotency_key="r1",
    )

    events: list[str] = []
    original_pin = db._pin_active

    def pin_then_try_to_write(connection, days):
        events.append("pin")
        result = original_pin(connection, days)
        # We are inside the query's BEGIN IMMEDIATE transaction; simulate a
        # racing uploader by opening a second connection.  It must block (or
        # fail on the busy timeout) instead of committing under us.
        other = sqlite3.connect(tmp_path / "db.sqlite3", timeout=0.2)
        try:
            with pytest.raises(sqlite3.OperationalError):
                other.execute("BEGIN IMMEDIATE")
            events.append("writer_blocked")
        finally:
            other.close()
        return result

    monkeypatch.setattr(db, "_pin_active", pin_then_try_to_write)
    result = db.estimate(start=date(2026, 11, 1), end=date(2026, 11, 1))
    assert events == ["pin", "writer_blocked"]
    assert result["partition_count"] == 1
    assert db.snapshot(result["snapshot_id"])["still_valid"] is True


def test_concurrent_queries_and_uploads_stay_consistent(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    precision = 12
    day_one = [f"d1-{i}" for i in range(2000)]
    db.submit_revision(
        revision=1,
        precision=precision,
        summaries=[
            (date(2026, 11, 1), summary(day_one, precision)),
            (date(2026, 11, 2), summary([f"d2-{i}" for i in range(2000)], precision)),
        ],
        idempotency_key="seed",
    )

    stop = threading.Event()
    barrier = threading.Barrier(5)
    errors: list[BaseException] = []

    def writer():
        try:
            barrier.wait(timeout=5)
            for revision in range(2, 12):
                db.submit_revision(
                    revision=revision,
                    precision=precision,
                    summaries=[
                        (
                            date(2026, 11, 2),
                            summary([f"late-{revision}-{i}" for i in range(50)], precision),
                        )
                    ],
                    idempotency_key=f"late-{revision}",
                )
                stop.wait(0.01)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            stop.set()

    def reader(_):
        seen: list[int] = []
        barrier.wait(timeout=5)
        for _ in range(400):
            result = db.estimate(start=date(2026, 11, 1), end=date(2026, 11, 2))
            revisions = {
                p["partition_date"]: p["revision"] for p in result["partitions_adopted"]
            }
            assert revisions["2026-11-01"] == 1
            # Every snapshot must match exactly one committed state.
            record = db.snapshot(result["snapshot_id"])
            assert record is not None
            assert [
                (p["partition_date"], p["revision"])
                for p in record["partitions_adopted"]
            ] == [
                (p["partition_date"], p["revision"])
                for p in result["partitions_adopted"]
            ]
            seen.append(revisions["2026-11-02"])
            if stop.is_set():
                break
        return seen

    with ThreadPoolExecutor(max_workers=5) as pool:
        writer_future = pool.submit(writer)
        results = list(pool.map(reader, range(4)))
        writer_future.result(timeout=30)
    assert not errors
    assert all(len(seen) > 0 for seen in results)
    # Queries raced the uploads and observed at least two committed revisions;
    # each individual answer still came from exactly one snapshot.
    assert len({rev for seen in results for rev in seen}) > 1


def test_upload_endpoint_rejects_precision_mixing_within_store(client):
    upload = client.post(
        "/v1/revisions", json=upload_body(1, 14, {"2026-11-01": ["a"]})
    )
    assert upload.status_code == 200
    second = client.post(
        "/v1/revisions", json=upload_body(2, 13, {"2026-11-02": ["b"]})
    )
    assert second.status_code == 422
    assert second.json()["error"]["code"] == "precision_conflict"


# -- re-upload invalidation receipts ----------------------------------------


def test_consecutive_reuploads_list_only_newly_invalidated_snapshots(tmp_path):
    """Each receipt names only the snapshots this batch actually killed."""

    db = Database(tmp_path / "db.sqlite3")
    day = date(2026, 11, 1)
    db.submit_revision(
        revision=1,
        precision=12,
        summaries=[(day, summary(["v1"], 12))],
        idempotency_key="r1",
    )
    first = db.estimate(start=day, end=day)["snapshot_id"]

    r2 = db.submit_revision(
        revision=2,
        precision=12,
        summaries=[(day, summary(["v2"], 12))],
        idempotency_key="r2",
    )
    assert r2["invalidated_snapshots"] == [first]
    first_record = db.snapshot(first)
    first_stamp = first_record["invalidated_at"]
    assert first_record["invalidation_reason"] == "revision 2 replaced 1 partition(s)"

    second = db.estimate(start=day, end=day)["snapshot_id"]
    r3 = db.submit_revision(
        revision=3,
        precision=12,
        summaries=[(day, summary(["v3"], 12))],
        idempotency_key="r3",
    )
    # The snapshot killed by r2 must not be listed again.
    assert r3["invalidated_snapshots"] == [second]

    third = db.estimate(start=day, end=day)["snapshot_id"]
    r4 = db.submit_revision(
        revision=4,
        precision=12,
        summaries=[(day, summary(["v4"], 12))],
        idempotency_key="r4",
    )
    assert r4["invalidated_snapshots"] == [third]

    # Historical snapshots keep their original invalidation time and reason.
    assert db.snapshot(first)["invalidated_at"] == first_stamp
    assert db.snapshot(first)["invalidation_reason"] == (
        "revision 2 replaced 1 partition(s)"
    )
    assert db.snapshot(second)["invalidation_reason"] == (
        "revision 3 replaced 1 partition(s)"
    )


def test_snapshot_covering_several_changed_days_is_listed_once(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    days = [date(2026, 11, 1) + timedelta(days=i) for i in range(3)]
    db.submit_revision(
        revision=1,
        precision=12,
        summaries=[(day, summary([f"d{day.day}"], 12)) for day in days],
        idempotency_key="r1",
    )
    snapshot_id = db.estimate(start=days[0], end=days[2])["snapshot_id"]

    result = db.submit_revision(
        revision=2,
        precision=12,
        summaries=[
            (days[0], summary(["new-a"], 12)),
            (days[1], summary(["new-b"], 12)),
        ],
        idempotency_key="r2",
    )
    # The snapshot adopted two of the replaced days but appears exactly once.
    assert result["invalidated_snapshots"] == [snapshot_id]
    assert db.snapshot(snapshot_id)["invalidation_reason"] == (
        "revision 2 replaced 2 partition(s)"
    )


def test_invalidation_scales_past_bind_parameter_ceiling(tmp_path):
    """More still-valid snapshots than SQLite's classic 999-variable limit.

    The invalidation must still commit in the same transaction as the
    partition replacement, and the receipt must name exactly the snapshots
    that transitioned from valid to invalid.
    """

    db = Database(tmp_path / "db.sqlite3")
    base = date(2026, 1, 1)
    days = [base + timedelta(days=i) for i in range(64)]
    mid = days[31]
    db.submit_revision(
        revision=1,
        precision=4,
        summaries=[(day, summary([f"u{day.day}"], 4)) for day in days],
        idempotency_key="seed",
    )

    # Every range that includes `mid` becomes a distinct still-valid snapshot:
    # 32 starts * 33 ends = 1056 > 999 (the classic bind-parameter ceiling).
    expected: set[int] = set()
    for start in days[:32]:
        for end in days[31:]:
            expected.add(db.estimate(start=start, end=end)["snapshot_id"])
    assert len(expected) == 32 * 33

    # A snapshot that does not cover `mid` must survive with its cache.
    unaffected = db.estimate(start=days[0], end=days[30])["snapshot_id"]

    result = db.submit_revision(
        revision=2,
        precision=4,
        summaries=[(mid, summary(["late"], 4))],
        idempotency_key="late",
    )
    receipt = result["invalidated_snapshots"]
    assert result["status"] == "accepted"
    assert len(receipt) == len(expected)
    assert set(receipt) == expected
    assert len(set(receipt)) == len(receipt)

    # Partition replacement and invalidation committed together.
    active = {row["partition_date"]: row for row in db.active_partitions()}
    assert active[mid.isoformat()]["revision"] == 2
    assert db.snapshot(receipt[0])["still_valid"] is False

    # The unaffected snapshot is still valid and its cached answer is reused.
    assert db.snapshot(unaffected)["still_valid"] is True
    reused = db.estimate(start=days[0], end=days[30])
    assert reused["snapshot_id"] == unaffected
    assert reused["reused_snapshot"] is True

    # A further re-upload with no intervening query re-lists nobody.
    again = db.submit_revision(
        revision=3,
        precision=4,
        summaries=[(mid, summary(["later"], 4))],
        idempotency_key="later",
    )
    assert again["invalidated_snapshots"] == []


def test_invalidation_chunks_days_below_bind_ceiling(tmp_path, monkeypatch):
    """Force tiny day chunks: snapshots spanning chunk boundaries are
    invalidated and listed exactly once."""

    monkeypatch.setattr(db_module, "_MAX_BIND_PARAMS", 3)
    db = Database(tmp_path / "db.sqlite3")
    days = [date(2026, 11, 1) + timedelta(days=i) for i in range(5)]
    db.submit_revision(
        revision=1,
        precision=12,
        summaries=[(day, summary([f"d{day.day}"], 12)) for day in days],
        idempotency_key="r1",
    )
    spanning = db.estimate(start=days[0], end=days[4])["snapshot_id"]
    head = db.estimate(start=days[0], end=days[0])["snapshot_id"]
    tail = db.estimate(start=days[4], end=days[4])["snapshot_id"]

    result = db.submit_revision(
        revision=2,
        precision=12,
        summaries=[(day, summary([f"n{day.day}"], 12)) for day in days],
        idempotency_key="r2",
    )
    assert result["invalidated_snapshots"] == sorted([spanning, head, tail])
    for snapshot_id in (spanning, head, tail):
        assert db.snapshot(snapshot_id)["still_valid"] is False


def test_invalidation_failure_rolls_back_partition_replacement(tmp_path, monkeypatch):
    """If the commit fails, neither the partition replacement nor the
    snapshot invalidation may take effect."""

    db = Database(tmp_path / "db.sqlite3")
    day = date(2026, 11, 1)
    db.submit_revision(
        revision=1,
        precision=12,
        summaries=[(day, summary(["old"], 12))],
        idempotency_key="r1",
    )
    snapshot_id = db.estimate(start=day, end=day)["snapshot_id"]

    def boom(connection, *, days, reason, now):
        raise RuntimeError("injected invalidation failure")

    monkeypatch.setattr(db, "_invalidate_snapshots", boom)
    with pytest.raises(RuntimeError, match="injected"):
        db.submit_revision(
            revision=2,
            precision=12,
            summaries=[(day, summary(["new"], 12))],
            idempotency_key="r2",
        )

    active = {row["partition_date"]: row for row in db.active_partitions()}
    assert active[day.isoformat()]["revision"] == 1
    assert db.snapshot(snapshot_id)["still_valid"] is True
    with sqlite3.connect(tmp_path / "db.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM partition_revisions WHERE revision = 2"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM submissions WHERE idempotency_key = 'r2'"
        ).fetchone()[0] == 0

    # The same query is still served from the still-valid cached snapshot.
    cached = db.estimate(start=day, end=day)
    assert cached["snapshot_id"] == snapshot_id
    assert cached["reused_snapshot"] is True


def test_long_range_query_chunks_pinning_below_bind_ceiling(tmp_path, monkeypatch):
    """A range wider than the (patched) bind-parameter ceiling still pins
    every day, in order, and still detects gaps."""

    monkeypatch.setattr(db_module, "_MAX_BIND_PARAMS", 2)
    db = Database(tmp_path / "db.sqlite3")
    days = [date(2026, 11, 1) + timedelta(days=i) for i in range(5)]
    db.submit_revision(
        revision=1,
        precision=12,
        summaries=[(day, summary([f"d{day.day}"], 12)) for day in days],
        idempotency_key="r1",
    )
    result = db.estimate(start=days[0], end=days[4])
    assert result["partition_count"] == 5
    assert [p["partition_date"] for p in result["partitions_adopted"]] == [
        day.isoformat() for day in days
    ]
    with pytest.raises(MissingPartitions):
        db.estimate(start=days[0], end=days[4] + timedelta(days=1))
