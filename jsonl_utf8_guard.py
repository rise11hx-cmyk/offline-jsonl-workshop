"""Dependency-free JSONL verifier for an offline Claude Code workshop.

Measures raw UTF-8 bytes rather than the number of Unicode characters.
The command never prints or uploads input records.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

# Keep in sync with [project].version in pyproject.toml (enforced by a unit test).
# A constant, not importlib.metadata, so the script also works from a clone or ZIP.
__version__ = "0.2.0"

DEFAULT_MAX_BYTES = 65_536
UTF8_BOM = b"\xef\xbb\xbf"


class JSONLValidationError(ValueError):
    """An invalid input record, identified by its physical line number."""


class _DuplicateKeyError(ValueError):
    """Raised by the object hook; deliberately carries no key or value."""


class _NonFiniteNumberError(ValueError):
    """Raised by parse_constant for NaN, Infinity and -Infinity."""


def _reject_json_constant(value: str) -> None:
    raise _NonFiniteNumberError("Non-finite JSON numbers are not supported")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON object members without echoing sensitive names."""
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise _DuplicateKeyError("Duplicate JSON object key")
        data[key] = value
    return data


def _json_type_name(value: object) -> str:
    """What was found instead of an object: a JSON type name, never the value (#47).

    A string or number may hold a token or private text, so only the type is reported;
    `bool` is a subclass of `int` and must be named before the number branch.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, str):
        return "a string"
    if value is None:
        return "null"
    return "a number"


def iter_jsonl_records(
    source: BinaryIO, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects while imposing a strict per-record byte limit.

    Bounded readline(max_bytes + 3) allows a trailing CRLF; longer records
    are rejected before their potentially sensitive content is parsed.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    line_number = 0
    while True:
        raw = source.readline(max_bytes + 3)
        if not raw:
            return
        line_number += 1
        payload = raw[:-1] if raw.endswith(b"\n") else raw
        if payload.endswith(b"\r"):
            payload = payload[:-1]
        if len(payload) > max_bytes:
            raise JSONLValidationError(
                f"Line {line_number}: exceeds {max_bytes} UTF-8 bytes"
            )
        if not payload.strip():
            continue
        if line_number == 1 and payload.startswith(UTF8_BOM):
            raise JSONLValidationError(
                "Line 1: starts with a UTF-8 byte order mark (BOM)"
            )
        if b"\r" in payload:
            raise JSONLValidationError(
                f"Line {line_number}: contains a bare carriage return; "
                "use LF or CRLF line endings"
            )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise JSONLValidationError(
                f"Line {line_number}: invalid UTF-8 encoding"
            ) from exc
        try:
            data = json.loads(
                text,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except RecursionError as exc:
            raise JSONLValidationError(
                f"Line {line_number}: JSON nesting is too deep"
            ) from exc
        except _DuplicateKeyError as exc:
            raise JSONLValidationError(
                f"Line {line_number}: duplicate JSON object key"
            ) from exc
        except _NonFiniteNumberError as exc:
            raise JSONLValidationError(
                f"Line {line_number}: non-finite number (NaN or Infinity) "
                "is not allowed"
            ) from exc
        except ValueError as exc:
            raise JSONLValidationError(
                f"Line {line_number}: invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise JSONLValidationError(
                f"Line {line_number}: expected a JSON object, got {_json_type_name(data)}"
            )
        yield data


def count_jsonl(source: BinaryIO, *, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
    return sum(1 for _ in iter_jsonl_records(source, max_bytes=max_bytes))


def _positive_int(value: str) -> int:
    """argparse type for --max-bytes: a base-10 integer at least 1."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {number}")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input file; omit for stdin")
    parser.add_argument("--max-bytes", type=_positive_int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--version", action="version", version=f"jsonl_utf8_guard {__version__}"
    )
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            count = count_jsonl(sys.stdin.buffer, max_bytes=args.max_bytes)
        else:
            with args.input.open("rb") as source:
                count = count_jsonl(source, max_bytes=args.max_bytes)
    except (OSError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "records": count, "max_bytes": args.max_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
