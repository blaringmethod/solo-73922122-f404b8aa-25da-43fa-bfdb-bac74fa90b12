"""Generate a ready-to-paste upload body for /docs or curl.

Usage:
    python examples/generate_upload.py 2026-11-01 2026-11-02 > body.json

Each argument is a UTC day. The generated summaries are deterministic so
re-running the command produces byte-identical payloads (handy with
``idempotency_key``). Pass ``--precision`` and ``--revision`` to match a store
that is already pinned to a different precision.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.hll import MIN_PRECISION, MAX_PRECISION, build_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("days", nargs="+", help="UTC days (YYYY-MM-DD)")
    parser.add_argument("--precision", type=int, default=14)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--users-per-day", type=int, default=5000)
    parser.add_argument("--idempotency-key", default=None)
    args = parser.parse_args()

    if not MIN_PRECISION <= args.precision <= MAX_PRECISION:
        parser.error(f"precision must be {MIN_PRECISION}..{MAX_PRECISION}")

    body = {
        "revision": args.revision,
        "precision": args.precision,
        "partitions": [
            {
                "partition_date": day,
                "summary": base64.b64encode(
                    build_summary(
                        [f"user-{i}-{day}" for i in range(args.users_per_day)],
                        args.precision,
                    )
                ).decode("ascii"),
            }
            for day in args.days
        ],
    }
    if args.idempotency_key:
        body["idempotency_key"] = args.idempotency_key
    json.dump(body, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
