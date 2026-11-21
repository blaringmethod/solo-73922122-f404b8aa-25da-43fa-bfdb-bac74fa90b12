"""HyperLogLog summaries used by the reach deduplication API.

Only dense, fixed-precision HyperLogLog sketches are accepted.  Every sketch is
self-describing and carries a CRC32 so truncated or bit-flipped uploads are
rejected before they can enter the database.

Version 1 wire format (all integers are big endian)::

    offset  size  field
    0       4     magic, b"HLL1"
    4       1     encoding version (1)
    5       1     precision p (MIN_PRECISION..MAX_PRECISION)
    6       2     reserved, must be zero
    8       4     register count m == 1 << p (uint32)
    12      m     dense registers, one uint8 per register
    12+m    4     CRC32 of every preceding byte (uint32)

The hash is SHA-256 of the identity bytes; the most significant ``p`` bits
select a register and the rank is the position of the first one bit in the
remaining bits.  Version 1 fixes both the hash and the register encoding, so
only sketches with the same encoding version may be merged.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Iterable

MAGIC = b"HLL1"
ENCODING_VERSION = 1
MAX_ENCODING_VERSION = 255
MIN_PRECISION = 4
MAX_PRECISION = 16
HEADER_SIZE = 12
TRAILER_SIZE = 4
MAX_SUMMARY_B64_LENGTH = 200_000

_U64_MASK = (1 << 64) - 1
_CONFIDENCE_Z = 1.959963984540054


class SummaryError(ValueError):
    """A summary is malformed, corrupt, or uses an unsupported encoding."""

    def __init__(self, message: str, *, code: str = "corrupt_summary") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class Summary:
    precision: int
    encoding_version: int
    registers: bytes
    checksum: int = 0

    @property
    def register_count(self) -> int:
        return 1 << self.precision


def register_count(precision: int) -> int:
    return 1 << precision


def max_rank(precision: int) -> int:
    return 64 - precision + 1


def hash_value(value: str | bytes) -> int:
    """Return the 64-bit HLL hash of one identity."""

    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("identity must be str or bytes")
    digest = hashlib.sha256(bytes(value)).digest()
    return int.from_bytes(digest[:8], "big")


def _index_and_rank(value_hash: int, precision: int) -> tuple[int, int]:
    remaining_bits = 64 - precision
    index = value_hash >> remaining_bits
    remainder = value_hash & ((1 << remaining_bits) - 1)
    if remainder == 0:
        rank = remaining_bits + 1
    else:
        rank = remaining_bits - remainder.bit_length() + 1
    return index, rank


def build_registers(values: Iterable[str | bytes], precision: int) -> bytes:
    """Build dense registers from an iterable of identities (client side)."""

    if not MIN_PRECISION <= precision <= MAX_PRECISION:
        raise ValueError(f"precision must be between {MIN_PRECISION} and {MAX_PRECISION}")
    registers = bytearray(register_count(precision))
    for value in values:
        index, rank = _index_and_rank(hash_value(value), precision)
        if rank > registers[index]:
            registers[index] = rank
    return bytes(registers)


def encode_summary(
    registers: bytes | bytearray,
    precision: int,
    version: int = ENCODING_VERSION,
) -> bytes:
    """Serialize registers into the versioned binary summary format."""

    if not MIN_PRECISION <= precision <= MAX_PRECISION:
        raise ValueError(f"precision must be between {MIN_PRECISION} and {MAX_PRECISION}")
    if not 1 <= version <= MAX_ENCODING_VERSION:
        raise ValueError("encoding version must fit in one byte")
    expected = register_count(precision)
    if len(registers) != expected:
        raise ValueError(f"expected {expected} registers for precision {precision}")
    limit = max_rank(precision)
    if any(value > limit for value in registers):
        raise ValueError(f"register value exceeds maximum rank {limit}")

    body = (
        MAGIC
        + struct.pack(">BBHI", version, precision, 0, expected)
        + bytes(registers)
    )
    checksum = binascii.crc32(body) & 0xFFFFFFFF
    return body + struct.pack(">I", checksum)


def decode_summary(data: bytes) -> Summary:
    """Parse and fully validate a binary summary."""

    if not data:
        raise SummaryError("summary is empty")
    if len(data) < HEADER_SIZE + TRAILER_SIZE + register_count(MIN_PRECISION):
        raise SummaryError("summary is truncated")
    if data[:4] != MAGIC:
        raise SummaryError("summary magic is not HLL1")

    version, precision, reserved, declared_count = struct.unpack(">BBHI", data[4:12])
    if version != ENCODING_VERSION:
        raise SummaryError(
            f"unsupported encoding version {version}; supported version is {ENCODING_VERSION}",
            code="unsupported_encoding_version",
        )
    if not MIN_PRECISION <= precision <= MAX_PRECISION:
        raise SummaryError(
            f"precision {precision} is outside {MIN_PRECISION}..{MAX_PRECISION}"
        )
    if reserved != 0:
        raise SummaryError("reserved header bytes must be zero")

    expected_count = register_count(precision)
    if declared_count != expected_count:
        raise SummaryError(
            f"register count {declared_count} does not match precision {precision}"
        )
    expected_length = HEADER_SIZE + expected_count + TRAILER_SIZE
    if len(data) != expected_length:
        raise SummaryError(
            f"summary length {len(data)} does not match expected {expected_length}"
        )

    stored_checksum = struct.unpack(">I", data[-TRAILER_SIZE:])[0]
    actual_checksum = binascii.crc32(data[:-TRAILER_SIZE]) & 0xFFFFFFFF
    if stored_checksum != actual_checksum:
        raise SummaryError("summary checksum mismatch")

    registers = data[HEADER_SIZE : HEADER_SIZE + expected_count]
    limit = max_rank(precision)
    if any(value > limit for value in registers):
        raise SummaryError(f"register value exceeds maximum rank {limit}")

    return Summary(
        precision=precision,
        encoding_version=version,
        registers=bytes(registers),
        checksum=stored_checksum,
    )


def decode_summary_base64(text: str) -> tuple[Summary, bytes]:
    """Decode a base64 summary and return the parsed summary plus raw bytes.

    ASCII whitespace is ignored (line-wrapped base64 is common), but the
    remaining characters must be strict base64.
    """

    compact = "".join(text.split())
    if not compact:
        raise SummaryError("summary is empty")
    try:
        raw = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SummaryError("summary is not valid base64") from exc
    return decode_summary(raw), raw


def merge_register_bytes(parts: Iterable[bytes | bytearray], precision: int) -> bytes:
    """Merge compatible sketches by taking the register-wise maximum.

    ``map(max, *parts)`` is used because the C-level dispatch is roughly 3x
    faster than a Python loop, which matters for a year of p=16 partitions.
    """

    blocks = [bytes(part) for part in parts]
    if not blocks:
        raise ValueError("at least one register block is required")
    expected = register_count(precision)
    for block in blocks:
        if len(block) != expected:
            raise ValueError(
                f"cannot merge register block of length {len(block)} with "
                f"precision {precision}"
            )
    if len(blocks) == 1:
        return blocks[0]
    return bytes(map(max, *blocks))


def merge_summaries(summaries: Iterable[Summary]) -> Summary:
    items = list(summaries)
    if not items:
        raise ValueError("at least one summary is required")
    precision = items[0].precision
    version = items[0].encoding_version
    for summary in items:
        if summary.precision != precision or summary.encoding_version != version:
            raise ValueError("summaries must share precision and encoding version")
    return Summary(
        precision=precision,
        encoding_version=version,
        registers=merge_register_bytes(
            (summary.registers for summary in items), precision
        ),
        checksum=0,
    )


def estimate_registers(registers: bytes | bytearray) -> float:
    """Estimate distinct cardinality from dense registers."""

    m = len(registers)
    if m < register_count(MIN_PRECISION) or m & (m - 1):
        raise ValueError("register count must be a power of two and at least 16")

    if m == 16:
        alpha = 0.673
    elif m == 32:
        alpha = 0.697
    elif m == 64:
        alpha = 0.709
    else:
        alpha = 0.7213 / (1.0 + 1.079 / m)

    harmonic_sum = math.fsum(2.0 ** (-value) for value in registers)
    zeros = registers.count(0)
    if harmonic_sum == 0.0:
        raw_estimate = 0.0
    else:
        raw_estimate = alpha * m * m / harmonic_sum

    if raw_estimate <= 2.5 * m and zeros > 0:
        estimate = m * math.log(m / zeros)
    elif raw_estimate > (2.0**64) / 30.0:
        ratio = raw_estimate / (2.0**64)
        if ratio >= 1.0:
            estimate = float(2.0**64)
        else:
            estimate = -(2.0**64) * math.log1p(-ratio)
    else:
        estimate = raw_estimate
    return estimate


def error_bound(estimate_value: float, precision: int) -> dict[str, float]:
    """Return the theoretical 95% HLL error bound around an estimate.

    This is the conventional ``1.96 * 1.04 / sqrt(m)`` normal approximation to
    the HyperLogLog standard error, not an exact confidence interval.
    """

    m = register_count(precision)
    relative_standard_error = 1.04 / math.sqrt(m)
    absolute_margin = _CONFIDENCE_Z * relative_standard_error * estimate_value
    return {
        "confidence_level": 0.95,
        "relative_standard_error": relative_standard_error,
        "absolute_margin": absolute_margin,
        "lower": max(0.0, estimate_value - absolute_margin),
        "upper": estimate_value + absolute_margin,
    }


def build_summary(values: Iterable[str | bytes], precision: int) -> bytes:
    """Convenience helper for producing a valid upload payload (client side)."""

    return encode_summary(build_registers(values, precision), precision)
