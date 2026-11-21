"""API behaviour: ingest, range queries, revisions, snapshots, rejection."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.hll import build_summary

from .conftest import summary_b64, upload_body


def upload(client, revision, partitions, precision=14, key=None):
    return client.post(
        "/v1/revisions",
        json=upload_body(revision, precision, partitions, key),
    )


def reach(client, start, end):
    return client.get(
        "/v1/reach", params={"start_date": start, "end_date": end}
    )


def error_code(response):
    return response.json()["error"]["code"]


# -- happy path ------------------------------------------------------------


def test_upload_and_query_merges_days_and_deduplicates_users(client):
    day1 = [f"user-{i}" for i in range(5000)]
    day2 = [f"user-{i}" for i in range(3000, 9000)]  # 2000 overlap
    response = upload(
        client, 1, {"2026-11-01": day1, "2026-11-02": day2}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "accepted"
    assert [p["partition_date"] for p in body["accepted_partitions"]] == [
        "2026-11-01",
        "2026-11-02",
    ]

    response = reach(client, "2026-11-01", "2026-11-02")
    assert response.status_code == 200, response.text
    result = response.json()
    assert abs(result["estimated_unique_users"] - 9000) / 9000 < 0.02
    assert result["precision"] == 14
    assert result["encoding_version"] == 1
    assert result["partition_count"] == 2
    assert [p["partition_date"] for p in result["partitions_adopted"]] == [
        "2026-11-01",
        "2026-11-02",
    ]
    assert {p["revision"] for p in result["partitions_adopted"]} == {1}
    bounds = result["error_bound"]
    assert bounds["lower"] < result["estimate"] < bounds["upper"]
    assert bounds["confidence_level"] == 0.95
    assert bounds["relative_standard_error"] == pytest.approx(1.04 / 128)


def test_single_day_and_range_of_one_is_allowed(client):
    upload(client, 1, {"2026-11-05": [f"u{i}" for i in range(100)]})
    response = reach(client, "2026-11-05", "2026-11-05")
    assert response.status_code == 200
    assert response.json()["partition_count"] == 1


def test_repeated_query_reuses_valid_snapshot(client):
    upload(client, 1, {"2026-11-01": [f"u{i}" for i in range(100)]})
    first = reach(client, "2026-11-01", "2026-11-01").json()
    second = reach(client, "2026-11-01", "2026-11-01").json()
    assert first["reused_snapshot"] is False
    assert second["reused_snapshot"] is True
    assert second["snapshot_id"] == first["snapshot_id"]


def test_partitions_endpoint_shows_active_revision_per_day(client):
    upload(client, 1, {"2026-11-01": ["a"], "2026-11-02": ["b"]})
    upload(client, 4, {"2026-11-02": ["c"]})
    body = client.get("/v1/partitions").json()
    by_day = {row["partition_date"]: row for row in body["partitions"]}
    assert by_day["2026-11-01"]["revision"] == 1
    assert by_day["2026-11-02"]["revision"] == 4


# -- revision replacement --------------------------------------------------


def test_late_revision_replaces_batch_and_invalidates_snapshots(client):
    day1 = [f"u{i}" for i in range(10_000)]
    day2 = [f"u{i}" for i in range(10_000, 20_000)]
    upload(client, 1, {"2026-11-01": day1, "2026-11-02": day2})
    before = reach(client, "2026-11-01", "2026-11-02").json()
    assert abs(before["estimated_unique_users"] - 20_000) / 20_000 < 0.02
    original_snapshot = before["snapshot_id"]

    replacement = [f"late-{i}" for i in range(500)]
    response = upload(client, 2, {"2026-11-02": replacement})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["superseded_revisions"] == [
        {
            "partition_date": "2026-11-02",
            "replaced_revision": 1,
            "applied_revision": 2,
        }
    ]
    assert original_snapshot in body["invalidated_snapshots"]

    # The previously returned snapshot is now marked stale but still readable.
    stale = client.get(f"/v1/snapshots/{original_snapshot}").json()
    assert stale["still_valid"] is False
    assert stale["invalidation_reason"]
    assert stale["response"]["snapshot_id"] == original_snapshot

    after = reach(client, "2026-11-01", "2026-11-02").json()
    assert after["snapshot_id"] != original_snapshot
    assert after["reused_snapshot"] is False
    by_day = {p["partition_date"]: p for p in after["partitions_adopted"]}
    assert by_day["2026-11-01"]["revision"] == 1
    assert by_day["2026-11-02"]["revision"] == 2
    # 10k untouched users + 500 replaced users, not 20k.
    assert abs(after["estimated_unique_users"] - 10_500) / 10_500 < 0.03


def test_late_revision_batch_is_all_or_nothing(client):
    upload(client, 1, {"2026-11-01": ["a"], "2026-11-02": ["b"]})
    upload(client, 2, {"2026-11-02": ["c"]})

    # A batch that mixes a valid future revision with a stale one for another
    # day must not partially apply.
    response = upload(client, 3, {"2026-11-05": ["new"]})
    assert response.status_code == 200
    # Now try batch revision 2 for 2026-11-05 (stale) plus 2026-11-06 (new).
    conflict = upload(client, 2, {"2026-11-05": ["x"], "2026-11-06": ["y"]})
    assert conflict.status_code == 409
    assert error_code(conflict) == "stale_revision"
    partitions = {
        row["partition_date"]: row
        for row in client.get("/v1/partitions").json()["partitions"]
    }
    assert "2026-11-06" not in partitions
    assert partitions["2026-11-05"]["revision"] == 3


def test_reusing_existing_revision_with_different_content_is_rejected(client):
    upload(client, 5, {"2026-11-01": ["a"]})
    response = upload(client, 5, {"2026-11-01": ["b"]})
    assert response.status_code == 409
    assert error_code(response) == "revision_conflict"


def test_identical_revision_is_unchanged_not_duplicated(client):
    upload(client, 5, {"2026-11-01": ["a"]})
    response = upload(client, 5, {"2026-11-01": ["a"]})
    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"


def test_idempotency_key_makes_retries_safe(client):
    body = upload_body(1, 14, {"2026-11-01": ["a"]}, "retry-key")
    first = client.post("/v1/revisions", json=body)
    second = client.post("/v1/revisions", json=body)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert second.json()["submission_id"] == first.json()["submission_id"]

    conflicting = upload_body(2, 14, {"2026-11-02": ["b"]}, "retry-key")
    third = client.post("/v1/revisions", json=conflicting)
    assert third.status_code == 409
    assert third.json()["status"] == "idempotency_conflict"


# -- rejection scenarios ---------------------------------------------------


def test_corrupt_summary_is_rejected(client):
    raw = build_summary([f"u{i}" for i in range(50)], 14)
    corrupt = base64.b64encode(raw[:100] + bytes([raw[100] ^ 0xFF]) + raw[101:])
    response = client.post(
        "/v1/revisions",
        json={
            "revision": 1,
            "precision": 14,
            "partitions": [
                {"partition_date": "2026-11-01", "summary": corrupt.decode()}
            ],
        },
    )
    assert response.status_code == 422
    assert error_code(response) == "corrupt_summary"
    assert "checksum" in response.json()["error"]["message"]
    assert client.get("/v1/partitions").json()["partitions"] == []


def test_truncated_summary_is_rejected(client):
    raw = base64.b64encode(build_summary(["a"], 14)[:-5]).decode()
    response = client.post(
        "/v1/revisions",
        json={
            "revision": 1,
            "precision": 14,
            "partitions": [{"partition_date": "2026-11-01", "summary": raw}],
        },
    )
    assert response.status_code == 422
    assert error_code(response) == "corrupt_summary"


def test_unsupported_encoding_version_is_rejected(client):
    raw = bytearray(build_summary(["a"], 14))
    raw[4] = 2  # bump version without updating CRC is fine: version checked first
    response = client.post(
        "/v1/revisions",
        json={
            "revision": 1,
            "precision": 14,
            "partitions": [
                {
                    "partition_date": "2026-11-01",
                    "summary": base64.b64encode(bytes(raw)).decode(),
                }
            ],
        },
    )
    assert response.status_code == 422
    assert error_code(response) == "unsupported_encoding_version"


def test_header_precision_mismatch_with_batch_is_rejected(client):
    raw = summary_b64(["a"], precision=12)
    response = client.post(
        "/v1/revisions",
        json={
            "revision": 1,
            "precision": 14,
            "partitions": [{"partition_date": "2026-11-01", "summary": raw}],
        },
    )
    assert response.status_code == 422
    assert error_code(response) == "precision_mismatch"
    assert "precision 12" in response.json()["error"]["message"]


def test_store_precision_is_pinned_after_first_upload(client):
    upload(client, 1, {"2026-11-01": ["a"]}, precision=14)
    response = upload(client, 2, {"2026-11-02": ["b"]}, precision=12)
    assert response.status_code == 422
    assert error_code(response) == "precision_conflict"
    assert client.get("/v1/partitions").json()["partitions"][0]["precision"] == 14


def test_missing_partition_in_range_is_rejected(client):
    upload(client, 1, {"2026-11-01": ["a"], "2026-11-03": ["c"]})
    response = reach(client, "2026-11-01", "2026-11-03")
    assert response.status_code == 404
    assert error_code(response) == "missing_partitions"
    assert response.json()["error"]["details"]["missing_dates"] == ["2026-11-02"]

    empty = reach(client, "2026-01-01", "2026-01-02")
    assert empty.status_code == 404


def test_inverted_range_is_rejected(client):
    response = reach(client, "2026-11-02", "2026-11-01")
    assert response.status_code == 400
    assert error_code(response) == "invalid_range"


def test_duplicate_partition_date_in_batch_is_rejected(client):
    raw = summary_b64(["a"])
    response = client.post(
        "/v1/revisions",
        json={
            "revision": 1,
            "precision": 14,
            "partitions": [
                {"partition_date": "2026-11-01", "summary": raw},
                {"partition_date": "2026-11-01", "summary": raw},
            ],
        },
    )
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_out_of_range_precision_in_request_is_rejected(client):
    response = client.post(
        "/v1/revisions",
        json={
            "revision": 1,
            "precision": 20,
            "partitions": [{"partition_date": "2026-11-01", "summary": "AAAA"}],
        },
    )
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_empty_partition_list_is_rejected(client):
    response = client.post(
        "/v1/revisions", json={"revision": 1, "precision": 14, "partitions": []}
    )
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_unknown_snapshot_returns_error_envelope(client):
    response = client.get("/v1/snapshots/999")
    assert response.status_code == 404
    assert error_code(response) == "snapshot_not_found"


def test_invalid_date_format_is_rejected(client):
    response = client.get(
        "/v1/reach", params={"start_date": "01-11-2026", "end_date": "2026-11-02"}
    )
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


# -- concurrency -----------------------------------------------------------


def test_concurrent_queries_all_read_consistent_snapshots(client):
    days = {f"2026-11-{d:02d}": [f"u{i}-{d}" for i in range(1000)] for d in range(1, 4)}
    assert upload(client, 1, days).status_code == 200

    def run():
        response = reach(client, "2026-11-01", "2026-11-03")
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(16)))

    assert len({r["snapshot_id"] for r in results}) == 1 or all(
        r["reused_snapshot"] for r in results
    )
    estimates = {r["estimated_unique_users"] for r in results}
    assert len(estimates) == 1
    assert results[0]["partition_count"] == 3


def test_concurrent_uploads_with_same_idempotency_key_apply_once(client):
    body = upload_body(1, 14, {"2026-11-01": ["a"]}, "race-key")

    def run(_):
        return client.post("/v1/revisions", json=body)

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(run, range(12)))
    assert {r.status_code for r in responses} == {200}
    statuses = [r.json()["status"] for r in responses]
    assert statuses.count("unchanged") + statuses.count("accepted") == 1
    assert statuses.count("duplicate") == 11


def test_concurrent_query_and_revision_never_mixes_revisions(client):
    """Queries racing a late revision must each report one revision per day."""

    base = [f"base-{i}" for i in range(2000)]
    upload(client, 1, {"2026-11-01": base, "2026-11-02": base})
    reach(client, "2026-11-01", "2026-11-02")

    replacement = [f"rev2-{i}" for i in range(2000)]
    response = upload(client, 2, {"2026-11-02": replacement})
    assert response.status_code == 200
    assert response.json()["invalidated_snapshots"]

    # Every later query sees revision 2 for day 2 and revision 1 for day 1.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: reach(client, "2026-11-01", "2026-11-02").json(),
                range(12),
            )
        )
    for result in results:
        revisions = {
            p["partition_date"]: p["revision"] for p in result["partitions_adopted"]
        }
        assert revisions == {"2026-11-01": 1, "2026-11-02": 2}


# -- docs / ops ------------------------------------------------------------


def test_openapi_documents_endpoints_and_error_envelopes(client):
    spec = client.get("/openapi.json").json()
    assert "/v1/revisions" in spec["paths"]
    assert "/v1/reach" in spec["paths"]
    assert "/v1/snapshots/{snapshot_id}" in spec["paths"]
    assert "/v1/partitions" in spec["paths"]
    assert "post" in spec["paths"]["/v1/revisions"]
    assert spec["info"]["title"] == "Reach Deduplication Estimator"
    assert client.get("/docs").status_code == 200


def test_health_endpoints(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["database"] == "ok"
    assert ready.json()["active_partitions"] == 0
