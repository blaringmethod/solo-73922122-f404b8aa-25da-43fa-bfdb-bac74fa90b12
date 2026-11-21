"""FastAPI application: ingest HLL summaries, estimate de-duplicated reach."""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import Body, Depends, FastAPI, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .db import (
    Database,
    InvalidRange,
    StorageError,
)
from .hll import Summary, SummaryError, decode_summary_base64
from .schemas import (
    ActivePartitionsResponse,
    ErrorResponse,
    HealthResponse,
    ReachResponse,
    SnapshotResponse,
    UploadRevisionRequest,
    UploadRevisionResponse,
)

DEFAULT_DB_PATH = os.environ.get("REACH_DB_PATH", "./data/reach.sqlite3")

DESCRIPTION = """
Deduplicated cross-device reach estimation over UTC-day-partitioned
HyperLogLog summaries.

**Upload** one revision batch per UTC day. All summaries in a batch share one
precision and one encoding version. A later revision replaces every partition
it carries *atomically*; partitions it does not mention stay untouched.
Superseded snapshots are marked invalid and later queries recompute.

**Query** a UTC date range. The service merges only active summaries with the
same precision and encoding version, reads the result from one committed
snapshot, and returns the estimate, its theoretical 95% error bound, and the
exact partitions (day, revision, revision id) it adopted.

**Rejected** with a structured error: corrupt or truncated summaries, a
header precision that differs from the batch precision, incompatible
precisions already active in the range, a missing day in the range, and
revision numbers that are stale or reuse existing content.
"""

TAGS_METADATA = [
    {"name": "ingest", "description": "Upload HLL summaries as revision batches."},
    {"name": "query", "description": "Estimate deduplicated reach over UTC day ranges."},
    {"name": "inspect", "description": "Active partitions and snapshot history."},
    {"name": "ops", "description": "Liveness and readiness."},
]


def _error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message, "details": details or {}}
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = Database(os.environ.get("REACH_DB_PATH", DEFAULT_DB_PATH))
    yield


app = FastAPI(
    title="Reach Deduplication Estimator",
    version=__version__,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid request or incompatible/corrupt summaries.",
        },
        500: {"model": ErrorResponse, "description": "Stored summary corruption."},
    },
)


def get_db(request: Request) -> Database:
    return request.app.state.db


# -- exception handlers ----------------------------------------------------


@app.exception_handler(StorageError)
async def storage_error_handler(_request: Request, exc: StorageError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(SummaryError)
async def summary_error_handler(_request: Request, exc: SummaryError) -> JSONResponse:
    return _error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        exc.code,
        f"summary rejected: {exc.message}",
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Pydantic error contexts may hold exception objects (e.g. a ValueError
    # raised by a model validator); reduce them to JSON-safe primitives.
    safe_errors = []
    for error in exc.errors():
        safe = {
            "type": error.get("type"),
            "loc": list(error.get("loc", ())),
            "msg": str(error.get("msg", "")),
        }
        if "input" in error:
            value = error["input"]
            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                safe["input"] = value
        safe_errors.append(safe)
    return _error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "invalid_request",
        "request body or query parameters failed validation",
        {"errors": safe_errors},
    )


# -- routes ----------------------------------------------------------------


@app.get("/healthz", tags=["ops"], summary="Liveness")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/readyz",
    tags=["ops"],
    response_model=HealthResponse,
    summary="Readiness including a SQLite round trip",
)
def readyz(db: Database = Depends(get_db)) -> dict[str, Any]:
    result = db.health()
    return {"status": "ok", **result}


@app.post(
    "/v1/revisions",
    tags=["ingest"],
    status_code=status.HTTP_200_OK,
    response_model=UploadRevisionResponse,
    responses={
        200: {"description": "Revision accepted, unchanged, or idempotent duplicate."},
        409: {
            "model": ErrorResponse,
            "description": "Stale revision, immutable revision conflict, or idempotency reuse.",
        },
        422: {
            "model": ErrorResponse,
            "description": (
                "Corrupt summary, unsupported encoding version, header precision "
                "not equal to batch precision, or malformed request."
            ),
        },
    },
    summary="Upload a revision batch of daily HLL summaries",
)
def upload_revision(
    response: Response,
    payload: UploadRevisionRequest = Body(
        ...,
        openapi_examples={
            "two_day_batch": {
                "summary": "Two UTC days at p=14",
                "value": {
                    "revision": 1,
                    "precision": 14,
                    "idempotency_key": "reach-2026-11-01-r1",
                    "partitions": [
                        {
                            "partition_date": "2026-11-01",
                            "summary": "<base64 HLL1 summary, see examples/demo.py>",
                        },
                        {
                            "partition_date": "2026-11-02",
                            "summary": "<base64 HLL1 summary, see examples/demo.py>",
                        },
                    ],
                },
            },
            "late_correction": {
                "summary": "A late revision replacing one day's partition",
                "value": {
                    "revision": 2,
                    "precision": 14,
                    "idempotency_key": "reach-2026-11-02-r2",
                    "partitions": [
                        {
                            "partition_date": "2026-11-02",
                            "summary": "<corrected base64 HLL1 summary>",
                        }
                    ],
                },
            },
        },
    ),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    seen: set[date] = set()
    decoded: list[tuple[date, Summary]] = []
    for partition in payload.partitions:
        summary, _raw = decode_summary_base64(partition.summary)
        if summary.precision != payload.precision:
            raise SummaryError(
                f"partition {partition.partition_date.isoformat()} has precision "
                f"{summary.precision} but the batch declares {payload.precision}",
                code="precision_mismatch",
            )
        if partition.partition_date in seen:
            raise SummaryError(
                f"duplicate partition_date {partition.partition_date.isoformat()}"
            )
        seen.add(partition.partition_date)
        decoded.append((partition.partition_date, summary))

    idempotency_key = payload.idempotency_key or f"auto-{secrets.token_hex(16)}"
    result = db.submit_revision(
        revision=payload.revision,
        precision=payload.precision,
        summaries=decoded,
        idempotency_key=idempotency_key,
    )
    if result["status"] == "idempotency_conflict":
        response.status_code = status.HTTP_409_CONFLICT
    return result


@app.get(
    "/v1/reach",
    tags=["query"],
    response_model=ReachResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid date range."},
        404: {
            "model": ErrorResponse,
            "description": "A UTC day in the range has no active partition.",
        },
        422: {
            "model": ErrorResponse,
            "description": "Active summaries in range disagree on precision/version.",
        },
    },
    summary="Estimate deduplicated unique users over a UTC date range",
)
def estimate_reach(
    start_date: date = Query(..., description="Inclusive UTC start day."),
    end_date: date = Query(..., description="Inclusive UTC end day."),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    if end_date < start_date:
        raise InvalidRange("end_date must be on or after start_date")
    return db.estimate(start=start_date, end=end_date)


@app.get(
    "/v1/snapshots/{snapshot_id}",
    tags=["inspect"],
    response_model=SnapshotResponse,
    responses={404: {"model": ErrorResponse, "description": "Unknown snapshot."}},
    summary="Inspect a committed snapshot and whether a later revision invalidated it",
)
def get_snapshot(snapshot_id: int, db: Database = Depends(get_db)) -> dict[str, Any]:
    record = db.snapshot(snapshot_id)
    if record is None:
        return _error_response(404, "snapshot_not_found", "unknown snapshot id")
    return record


@app.get(
    "/v1/partitions",
    tags=["inspect"],
    response_model=ActivePartitionsResponse,
    summary="List the currently active partition and revision per UTC day",
)
def list_partitions(db: Database = Depends(get_db)) -> dict[str, Any]:
    return {"partitions": db.active_partitions()}
