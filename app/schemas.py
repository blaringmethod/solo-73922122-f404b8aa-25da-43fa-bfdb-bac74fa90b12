"""Request and response models for the reach API (also feeds /docs)."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .hll import MAX_PRECISION, MAX_SUMMARY_B64_LENGTH, MIN_PRECISION


class PartitionUpload(BaseModel):
    partition_date: date = Field(
        ...,
        description="UTC day this HyperLogLog summary covers (YYYY-MM-DD).",
        examples=["2026-11-01"],
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SUMMARY_B64_LENGTH,
        description=(
            "Base64 of the binary HLL summary. Layout: magic HLL1, version u8, "
            "precision u8, two reserved zero bytes, register count u32, dense "
            "registers (one u8 each), CRC32 u32, all big endian."
        ),
    )

    @field_validator("summary")
    @classmethod
    def summary_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
        return value


class UploadRevisionRequest(BaseModel):
    revision: int = Field(
        ...,
        ge=1,
        le=2_147_483_647,
        description=(
            "Partition revision number. A late correction must use a strictly "
            "higher revision than the active one; the batch replaces those days "
            "atomically."
        ),
        examples=[2],
    )
    precision: int = Field(
        ...,
        ge=MIN_PRECISION,
        le=MAX_PRECISION,
        description=(
            "Common HLL precision p for every summary in the batch "
            f"({MIN_PRECISION}..{MAX_PRECISION}). Must match each summary header."
        ),
        examples=[14],
    )
    partitions: list[PartitionUpload] = Field(
        ...,
        min_length=1,
        max_length=400,
        description="One summary per UTC day. Duplicate days are rejected.",
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional client-generated key. Retrying with the same key and "
            "identical payload returns the original submission; reusing the key "
            "with a different payload is rejected with 409."
        ),
        examples=["reach-2026-11-01-r2"],
    )

    @field_validator("idempotency_key")
    @classmethod
    def key_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("idempotency_key must not be blank when provided")
        return value

    @model_validator(mode="after")
    def unique_days(self) -> "UploadRevisionRequest":
        seen: set[date] = set()
        for partition in self.partitions:
            if partition.partition_date in seen:
                raise ValueError(
                    f"duplicate partition_date {partition.partition_date.isoformat()} "
                    "in one revision batch"
                )
            seen.add(partition.partition_date)
        return self


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class AcceptedPartition(BaseModel):
    partition_date: date
    revision: int
    revision_id: int
    precision: int


class SupersededPartition(BaseModel):
    partition_date: date
    replaced_revision: int
    applied_revision: int


class UploadRevisionResponse(BaseModel):
    status: str = Field(
        description="accepted, unchanged (identical revision already active), "
        "duplicate (idempotent retry) or idempotency_conflict."
    )
    submission_id: int | None = None
    revision: int | None = None
    precision: int | None = None
    payload_sha256: str | None = None
    accepted_partitions: list[AcceptedPartition] = Field(default_factory=list)
    superseded_revisions: list[SupersededPartition] = Field(default_factory=list)
    invalidated_snapshots: list[int] = Field(default_factory=list)
    message: str | None = None


class ErrorBound(BaseModel):
    confidence_level: float = Field(description="Nominal confidence level, 0.95.")
    relative_standard_error: float = Field(
        description="Theoretical HLL standard error 1.04 / sqrt(m)."
    )
    absolute_margin: float = Field(
        description="1.96 * relative_standard_error * estimate."
    )
    lower: float
    upper: float


class AdoptedPartition(BaseModel):
    partition_date: date
    revision: int
    revision_id: int
    precision: int


class QueryInfo(BaseModel):
    start_date: date
    end_date: date


class ReachResponse(BaseModel):
    snapshot_id: int = Field(
        description="Immutable committed snapshot the answer was read from."
    )
    reused_snapshot: bool = Field(
        description="True when an equivalent still-valid snapshot answer was reused."
    )
    query: QueryInfo
    estimated_unique_users: int
    estimate: float
    precision: int
    encoding_version: int
    error_bound: ErrorBound
    partitions_adopted: list[AdoptedPartition]
    partition_count: int
    created_at: str


class SnapshotResponse(BaseModel):
    snapshot_id: int
    start_date: date
    end_date: date
    created_at: str
    still_valid: bool = Field(
        description="False once a later revision replaced a day this snapshot used."
    )
    invalidated_at: str | None = None
    invalidation_reason: str | None = None
    partitions_adopted: list[AdoptedPartition]
    response: ReachResponse | None = None


class ActivePartition(BaseModel):
    partition_date: date
    revision: int
    revision_id: int
    precision: int
    encoding_version: int


class ActivePartitionsResponse(BaseModel):
    partitions: list[ActivePartition]


class HealthResponse(BaseModel):
    status: str
    database: str
    active_partitions: int
    snapshots: int
    latency_ms: float
