"""End-to-end walkthrough used by the README and `examples/demo.py`.

It builds real HLL summaries with the same codec the API accepts, uploads two
UTC days, queries them, then shows a late revision replacing one partition,
invalidating the earlier snapshot, and a few rejections.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.hll import build_summary  # noqa: E402

BASE = os.environ.get("REACH_URL", "http://localhost:8000")


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def partition(day: str, values: list[str], precision: int = 14) -> dict:
    summary = build_summary(values, precision)
    return {
        "partition_date": day,
        "summary": base64.b64encode(summary).decode("ascii"),
    }


def main() -> None:
    day_one = [f"user-{i}" for i in range(20_000)]
    day_two = [f"user-{i}" for i in range(15_000, 40_000)]  # 5k overlap

    status, body = call(
        "POST",
        "/v1/revisions",
        {
            "revision": 1,
            "precision": 14,
            "idempotency_key": "demo-r1",
            "partitions": [
                partition("2026-11-01", day_one),
                partition("2026-11-02", day_two),
            ],
        },
    )
    print(f"upload r1 -> {status} {body.get('status')}")

    status, before = call("GET", "/v1/reach?start_date=2026-11-01&end_date=2026-11-02")
    print(
        f"query -> {status} unique≈{before['estimated_unique_users']} "
        f"95% CI [{before['error_bound']['lower']:.0f}, "
        f"{before['error_bound']['upper']:.0f}] snapshot={before['snapshot_id']}"
    )

    late = [f"corrected-{i}" for i in range(1_000)]
    status, body = call(
        "POST",
        "/v1/revisions",
        {
            "revision": 2,
            "precision": 14,
            "idempotency_key": "demo-r2",
            "partitions": [partition("2026-11-02", late)],
        },
    )
    print(
        f"late r2 -> {status} superseded={body.get('superseded_revisions')} "
        f"invalidated_snapshots={body.get('invalidated_snapshots')}"
    )

    status, snapshot = call("GET", f"/v1/snapshots/{before['snapshot_id']}")
    print(f"old snapshot still_valid -> {snapshot['still_valid']}")

    status, after = call("GET", "/v1/reach?start_date=2026-11-01&end_date=2026-11-02")
    print(
        f"query after r2 -> {status} unique≈{after['estimated_unique_users']} "
        f"partitions={[(p['partition_date'], p['revision']) for p in after['partitions_adopted']]}"
    )

    broken = base64.b64encode(b"HLL1 not a real sketch").decode()
    status, error = call(
        "POST",
        "/v1/revisions",
        {
            "revision": 3,
            "precision": 14,
            "partitions": [{"partition_date": "2026-11-03", "summary": broken}],
        },
    )
    print(f"corrupt summary -> {status} {error['error']['code']}")

    status, error = call("GET", "/v1/reach?start_date=2026-11-01&end_date=2026-11-09")
    print(f"missing partition -> {status} {error['error']['details']}")


if __name__ == "__main__":
    main()
