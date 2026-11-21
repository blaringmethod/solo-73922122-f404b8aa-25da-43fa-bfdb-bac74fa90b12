"""Shared fixtures and client-side helpers for the reach API tests."""

from __future__ import annotations

import base64
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.hll import build_summary  # noqa: E402
from app.main import app  # noqa: E402


def summary_b64(values, precision: int = 14) -> str:
    return base64.b64encode(build_summary(values, precision)).decode("ascii")


def upload_body(
    revision: int,
    precision: int,
    partitions: dict[str, list[str]],
    idempotency_key: str | None = None,
) -> dict:
    body = {
        "revision": revision,
        "precision": precision,
        "partitions": [
            {"partition_date": day, "summary": summary_b64(values, precision)}
            for day, values in partitions.items()
        ],
    }
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return body


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("REACH_DB_PATH", str(tmp_path / "reach.sqlite3"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "reach.sqlite3"
    monkeypatch.setenv("REACH_DB_PATH", str(path))
    return path
