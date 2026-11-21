"""Unit tests for the HyperLogLog codec, merge, and error bounds."""

from __future__ import annotations

import base64
import binascii
import math
import struct

import pytest

from app.hll import (
    SummaryError,
    build_registers,
    build_summary,
    decode_summary,
    decode_summary_base64,
    encode_summary,
    error_bound,
    estimate_registers,
    hash_value,
    max_rank,
    merge_register_bytes,
    merge_summaries,
)


def test_estimate_tracks_truth_for_mid_range_cardinality():
    values = [f"user-{i}" for i in range(20_000)]
    summary = decode_summary(build_summary(values, 14))
    estimate = estimate_registers(summary.registers)
    assert abs(estimate - 20_000) / 20_000 < 0.02


def test_estimate_zero_for_empty_sketch():
    summary = decode_summary(build_summary([], 14))
    assert estimate_registers(summary.registers) == pytest.approx(0.0, abs=1e-9)


def test_small_cardinality_uses_linear_counting():
    # HLL switches to linear counting below 2.5*m registers; check exactness-ish
    summary = decode_summary(build_summary([f"u{i}" for i in range(10)], 14))
    assert estimate_registers(summary.registers) == pytest.approx(10, abs=1)


def test_hash_is_stable_and_rank_is_bounded():
    assert hash_value("abc") == hash_value(b"abc")
    assert hash_value("abc") != hash_value("abd")
    assert 1 <= hash_value("abc") <= (1 << 64) - 1


def test_known_vector_locks_wire_format():
    # A fixed, tiny sketch whose bytes must never change without a version bump.
    registers = build_registers(["a", "b", "c"], 4)
    encoded = encode_summary(registers, 4)
    assert encoded[:4] == b"HLL1"
    assert encoded[4] == 1
    assert encoded[5] == 4
    assert struct.unpack(">H", encoded[6:8])[0] == 0
    assert struct.unpack(">I", encoded[8:12])[0] == 16
    assert len(encoded) == 12 + 16 + 4
    decoded = decode_summary(encoded)
    assert decoded.registers == registers
    assert decoded.precision == 4
    assert decoded.encoding_version == 1


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda raw: raw[:4] + bytes([9]) + raw[5:], "unsupported_encoding_version"),
        (lambda raw: b"NOPE" + raw[4:], "corrupt_summary"),
        (lambda raw: raw[:-1], "corrupt_summary"),
        (lambda raw: raw + b"\x00", "corrupt_summary"),
        (lambda raw: raw[:20] + bytes([raw[20] ^ 0x01]) + raw[21:], "corrupt_summary"),
        (lambda raw: raw[:6] + b"\x00\x01" + raw[8:], "corrupt_summary"),
    ],
)
def test_corrupt_summaries_are_rejected(mutate, code):
    raw = build_summary([f"u{i}" for i in range(100)], 14)
    with pytest.raises(SummaryError) as excinfo:
        decode_summary(mutate(raw))
    assert excinfo.value.code == code


def test_register_value_above_max_rank_is_rejected():
    registers = bytearray(build_registers(["a"], 4))
    registers[0] = max_rank(4) + 1
    # Encoder refuses to produce it ...
    with pytest.raises(ValueError):
        encode_summary(bytes(registers), 4)
    # ... and the decoder rejects hand-crafted bytes with a valid CRC.
    body = b"HLL1" + struct.pack(">BBHI", 1, 4, 0, 16) + bytes(registers)
    raw = body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    with pytest.raises(SummaryError):
        decode_summary(raw)


def test_encode_rejects_wrong_register_count_and_precision_range():
    with pytest.raises(ValueError):
        encode_summary(b"\x00" * 10, 4)
    with pytest.raises(ValueError):
        encode_summary(b"\x00" * 16, 3)


def test_base64_round_trip_accepts_wrapped_whitespace():
    encoded = base64.b64encode(build_summary(["x"], 12)).decode()
    wrapped = "\n".join(encoded[i : i + 40] for i in range(0, len(encoded), 40))
    summary, raw = decode_summary_base64(wrapped)
    assert summary.precision == 12
    assert raw == base64.b64decode(encoded)


@pytest.mark.parametrize("text", ["", "   ", "not base64!!", "====", "HLL1"])
def test_base64_rejects_garbage(text):
    with pytest.raises(SummaryError):
        decode_summary_base64(text)


def test_merge_is_register_wise_max_and_matches_single_sketch():
    left = decode_summary(build_summary([f"u{i}" for i in range(1000)], 12))
    right = decode_summary(build_summary([f"u{i}" for i in range(500, 1500)], 12))
    union = decode_summary(
        build_summary([f"u{i}" for i in range(1500)], 12)
    )
    merged = merge_summaries([left, right])
    assert merged.registers == union.registers
    assert estimate_registers(merged.registers) == pytest.approx(
        estimate_registers(union.registers), rel=1e-12
    )


def test_merge_requires_same_precision_and_version():
    a = decode_summary(build_summary(["a"], 12))
    b = decode_summary(build_summary(["b"], 13))
    with pytest.raises(ValueError):
        merge_summaries([a, b])


def test_merge_register_bytes_rejects_length_mismatch():
    with pytest.raises(ValueError):
        merge_register_bytes([b"\x00" * 16, b"\x00" * 32], 4)


def test_error_bound_contains_estimate_and_matches_theory():
    bounds = error_bound(10_000.0, 14)
    assert bounds["confidence_level"] == 0.95
    assert bounds["relative_standard_error"] == pytest.approx(1.04 / math.sqrt(16384))
    assert bounds["lower"] < 10_000.0 < bounds["upper"]
    assert bounds["absolute_margin"] == pytest.approx(
        1.959963984540054 * (1.04 / math.sqrt(16384)) * 10_000.0
    )
    assert bounds["lower"] >= 0.0


def test_error_bound_is_non_negative_at_low_precision():
    # The margin is proportional to the estimate, so lower stays positive for
    # every supported precision; the clamp is a defensive invariant.
    bounds = error_bound(1.0, 4)
    assert bounds["lower"] >= 0.0
    assert bounds["upper"] > bounds["lower"]
