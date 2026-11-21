# Reach Deduplication Estimator

FastAPI + SQLite service that estimates **de-duplicated cross-device unique
users** over UTC day ranges from HyperLogLog (HLL) summaries uploaded by data
engineers.

The service merges only summaries that share one precision **and** one
encoding version, replaces a whole late-revision batch atomically, invalidates
the snapshots that touched the replaced days, and answers every query from a
single committed snapshot.

## Quick start

### Docker Compose (recommended)

```bash
docker compose up --build
# API:      http://localhost:8000
# Docs:     http://localhost:8000/docs
# OpenAPI:  http://localhost:8000/openapi.json
```

SQLite lives on the `reach-data` volume (`REACH_DB_PATH=/data/reach.sqlite3`).
The container runs as a non-root user and defines a `HEALTHCHECK` on
`/healthz`.

### Local Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
REACH_DB_PATH=./data/reach.sqlite3 uvicorn app.main:app --reload
pytest -q            # 58 tests
python examples/demo.py     # end-to-end walkthrough against localhost:8000
```

`examples/generate_upload.py` prints a ready-to-paste upload body for
`/docs`:

```bash
python examples/generate_upload.py 2026-11-01 2026-11-02 --revision 1 \
  | curl -sS -X POST localhost:8000/v1/revisions \
      -H 'content-type: application/json' --data-binary @-
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/revisions` | Upload one revision batch of daily HLL summaries |
| `GET` | `/v1/reach?start_date=&end_date=` | Estimate unique users, error bound, adopted partitions |
| `GET` | `/v1/snapshots/{id}` | Inspect a snapshot and whether it was invalidated |
| `GET` | `/v1/partitions` | Active revision per UTC day |
| `GET` | `/healthz`, `/readyz` | Liveness / readiness (SQLite round trip) |

All errors use one envelope:

```json
{"error": {"code": "missing_partitions", "message": "...", "details": {"missing_dates": ["2026-11-02"]}}}
```

### Upload

```json
{
  "revision": 2,
  "precision": 14,
  "idempotency_key": "reach-2026-11-02-r2",
  "partitions": [
    {"partition_date": "2026-11-02", "summary": "<base64 of HLL1 summary>"}
  ]
}
```

* All `partitions` are applied in **one transaction**; a late revision
  replaces exactly the days it carries and leaves other days untouched.
* The first accepted upload pins the store precision and encoding version;
  later uploads at a different precision get `422 precision_conflict`.
* A revision number lower than the active one → `409 stale_revision`. Reusing
  an existing revision number with different bytes → `409 revision_conflict`.
  Re-sending the identical revision is `200 {"status": "unchanged"}`.
* Reusing an `idempotency_key` with an identical payload → `200
  {"status": "duplicate"}`; with a different payload → `409`.
* Status codes: `200` accepted/unchanged/duplicate, `409` revision or
  idempotency conflict, `422` corrupt/colliding summary or bad request.

### Query

```bash
curl 'localhost:8000/v1/reach?start_date=2026-11-01&end_date=2026-11-02'
```

```json
{
  "snapshot_id": 2,
  "reused_snapshot": false,
  "estimated_unique_users": 20944,
  "estimate": 20944.075930203453,
  "precision": 14,
  "encoding_version": 1,
  "error_bound": {
    "confidence_level": 0.95,
    "relative_standard_error": 0.008125,
    "absolute_margin": 333.5,
    "lower": 20610.0,
    "upper": 21277.0
  },
  "partitions_adopted": [
    {"partition_date": "2026-11-01", "revision": 1, "revision_id": 1, "precision": 14},
    {"partition_date": "2026-11-02", "revision": 2, "revision_id": 3, "precision": 14}
  ],
  "partition_count": 2,
  "created_at": "2026-11-21T07:00:00.000000Z"
}
```

Rejected ranges: missing day → `404 missing_partitions`, incompatible
precision/version → `422 precision_mismatch`, inverted range → `400
invalid_range`, more than 1826 days → `400 invalid_range`.

## HLL summary format (v1)

```
offset  size  field
0       4     magic b"HLL1"
4       1     encoding version (1)
5       1     precision p (4..16)
6       2     reserved, must be zero
8       4     register count m = 1 << p (u32)
12      m     dense registers, one u8 each
12+m    4     CRC32 (u32) over all preceding bytes
```

* Hash: SHA-256 of the identity bytes; the high `p` bits pick the register,
  the rank is the position of the first 1 bit in the remaining bits.
* A summary is rejected when the magic, version, precision range, reserved
  bytes, declared register count, exact byte length, CRC32, or register values
  are wrong.
* Encoding version is part of the merge contract: v1 sketches never merge
  with another version, and `/v1/reach` refuses a range that mixes them.

## Design notes

* **Tables**: `submissions` (audit log), `partition_revisions` (immutable
  history, every revision), `active_partitions` (one revision per day),
  `snapshots` + `snapshot_partitions` (copies of the exact register bytes an
  answer used), `query_results` (answer cache), `service_config` (pinned
  precision/version).
* **Snapshot isolation**: a cold query runs `BEGIN IMMEDIATE`, pins the
  active partitions, merges, writes the snapshot and result, and commits.
  Since that transaction is the only writer, no revision can land between
  pinning and snapshot storage — an answer never mixes revisions. A
  still-valid cached answer is served from a read-only transaction.* **Invalidation**: a revision marks every snapshot that adopted a replaced
  day with `invalidated_at` + reason; the old answer stays readable via
  `GET /v1/snapshots/{id}` but is no longer reused.
* **Bit rot**: register blobs are re-validated (length, header CRC32, max
  rank) when copied into a snapshot; a mismatch returns `500
  stored_summary_corrupt` instead of a wrong estimate.
* **Concurrency**: WAL mode, `busy_timeout=10s`, a process-level writer lock,
  and `BEGIN IMMEDIATE` serialize writers. Reads proceed concurrently.
* **Error bound**: nominal 95% normal approximation,
  `1.96 * (1.04 / sqrt(m)) * estimate`, clamped at 0. It is a model-based
  bound, not a measured confidence interval.

## Verified

```
$ pytest -q
58 passed
```

Covered: codec round trip and every corruption mode, register-wise merge,
estimator accuracy, error-bound math, upload/query happy paths, late-revision
batch replacement + snapshot invalidation, stale/conflicting revisions,
idempotency retries, precision pinning, missing-partition/inverted-range
rejections, snapshot inspection, OpenAPI route presence, and concurrent
queries+uploads (each snapshot matches one committed state, and a racing
writer is blocked while a query transaction is open). `examples/demo.py` was
also run against a live uvicorn server.

## Not done / limitations

* Docker and Compose files are provided but **not built or run here** (no
  Docker daemon in this environment); `docker-compose.yml` was only
  YAML-validated.
* Single SQLite database; there is no cross-region replication or read
  replica.
* No authentication, tenant scoping, or rate limiting — put it behind a
  gateway before exposing it.
* `query_results` is an append-only cache; invalidated rows are retained (by
  design, for auditability) and there is no retention/compaction job.
* A 365-day p=14 cold query takes ~1.4 s here (merge is pure Python). p=16
  is slower and memory-heavier; p=12 is the recommended ceiling for very
  long ranges.
* The error bound is the standard theoretical HLL approximation; it is not
  calibrated against ground truth per deployment.
