#!/usr/bin/env python3
"""
Download Coinbase BTC-USD hourly OHLCV data.

This script performs data acquisition only. It does not clean, interpolate,
fill gaps, create features, or train models.

API source:
    Coinbase Exchange REST API
    GET https://api.exchange.coinbase.com/products/BTC-USD/candles

Important timestamp convention:
    --start is inclusive.
    --end is exclusive.
Both must be UTC timestamps aligned to an exact hour.

The Coinbase Exchange candles endpoint accepts Unix-second start/end values,
uses the candle "time" as the bucket start, supports 3600-second candles,
and limits a request to 300 candles. The downloader deliberately requests
250-hour chunks and filters returned candles to the exact requested interval.
This leaves room for Coinbase's documented behavior that a response may
include candles preceding the declared start.

No missing hours are filled and no observations are removed because they
look suspicious. Duplicate timestamps are detected and cause the acquisition
to fail rather than being silently deduplicated.

Restart safety:
    Successful chunks are stored in data/raw/.btc_usd_hourly_chunks/.
    A deterministic request manifest is stored alongside them. Re-running
    the same command resumes missing chunks. A changed request range starts
    a new manifest namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests


API_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
PRODUCT = "BTC-USD"
GRANULARITY_SECONDS = 3600
CHUNK_HOURS = 250
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0

RAW_PATH = Path("data/raw/btc_usd_hourly_raw.csv")
METADATA_PATH = Path("data/raw/btc_usd_hourly_metadata.json")
CHUNK_ROOT = Path("data/raw/.btc_usd_hourly_chunks")


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot be completed safely."""


def parse_utc_hour(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and require UTC + exact hourly alignment."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ISO-8601 timestamp: {value!r}"
        ) from exc

    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"Timestamp must include UTC timezone information: {value!r}"
        )

    dt = dt.astimezone(timezone.utc)

    if any((dt.minute, dt.second, dt.microsecond)):
        raise argparse.ArgumentTypeError(
            f"Timestamp must be aligned to an exact UTC hour: {value!r}"
        )

    return dt


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unix_seconds(dt: datetime) -> int:
    return int(dt.timestamp())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def request_json(
    session: requests.Session,
    start: datetime,
    end_exclusive: datetime,
) -> list:
    """
    Fetch one bounded API chunk.

    We send an end timestamp one second before the final requested candle
    bucket. This makes the intended interval [start, end_exclusive)
    unambiguous at the bucket level.
    """
    request_end = end_exclusive - timedelta(seconds=1)

    params = {
        "granularity": str(GRANULARITY_SECONDS),
        "start": str(unix_seconds(start)),
        "end": str(unix_seconds(request_end)),
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise requests.HTTPError(
                    f"retryable HTTP status {response.status_code}: "
                    f"{response.text[:300]}",
                    response=response,
                )

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, list):
                raise AcquisitionError(
                    f"Unexpected Coinbase response type for "
                    f"{iso_z(start)}–{iso_z(end_exclusive)}: "
                    f"{type(payload).__name__}"
                )

            return payload

        except (requests.RequestException, ValueError, AcquisitionError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break

            sleep_seconds = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"Request failed (attempt {attempt}/{MAX_RETRIES}): {exc}. "
                f"Retrying in {sleep_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise AcquisitionError(
        f"Coinbase request failed after {MAX_RETRIES} attempts for "
        f"{iso_z(start)}–{iso_z(end_exclusive)}: {last_error}"
    )


def normalize_candles(
    candles: list,
    chunk_start: datetime,
    chunk_end: datetime,
) -> tuple[list[dict[str, str]], list[int]]:
    """
    Normalize Coinbase rows to:
        timestamp_utc, open, high, low, close, volume

    Return in-range rows plus timestamps returned outside the requested
    interval. Out-of-range rows are reported by the caller rather than
    silently becoming part of the dataset.
    """
    in_range: list[dict[str, str]] = []
    out_of_range: list[int] = []
    seen: set[int] = set()

    start_ts = unix_seconds(chunk_start)
    end_ts = unix_seconds(chunk_end)

    for row in candles:
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise AcquisitionError(
                f"Unexpected candle schema in chunk {iso_z(chunk_start)}–"
                f"{iso_z(chunk_end)}: {row!r}"
            )

        # Coinbase Exchange candle schema:
        # [time, low, high, open, close, volume]
        try:
            ts = int(row[0])
        except (TypeError, ValueError) as exc:
            raise AcquisitionError(f"Invalid candle timestamp: {row!r}") from exc

        if ts in seen:
            raise AcquisitionError(
                f"Duplicate timestamp returned by Coinbase inside one chunk: "
                f"{iso_z(datetime.fromtimestamp(ts, tz=timezone.utc))}"
            )
        seen.add(ts)

        if ts < start_ts or ts >= end_ts:
            out_of_range.append(ts)
            continue

        if ts % GRANULARITY_SECONDS != 0:
            raise AcquisitionError(
                f"Non-hour-aligned Coinbase timestamp returned: {ts}"
            )

        in_range.append(
            {
                "timestamp_utc": iso_z(
                    datetime.fromtimestamp(ts, tz=timezone.utc)
                ),
                "open": str(row[3]),
                "high": str(row[2]),
                "low": str(row[1]),
                "close": str(row[4]),
                "volume": str(row[5]),
            }
        )

    in_range.sort(key=lambda row: row["timestamp_utc"])
    return in_range, sorted(out_of_range)


def expected_timestamps(start: datetime, end: datetime) -> list[int]:
    current = start
    result: list[int] = []
    while current < end:
        result.append(unix_seconds(current))
        current += timedelta(seconds=GRANULARITY_SECONDS)
    return result


def chunk_ranges(start: datetime, end: datetime) -> Iterable[tuple[int, datetime, datetime]]:
    current = start
    index = 0

    while current < end:
        chunk_end = min(
            current + timedelta(hours=CHUNK_HOURS),
            end,
        )
        yield index, current, chunk_end
        current = chunk_end
        index += 1


def chunk_paths(manifest_id: str, index: int) -> Path:
    return CHUNK_ROOT / manifest_id / f"chunk_{index:06d}.csv"


def manifest_path(manifest_id: str) -> Path:
    return CHUNK_ROOT / manifest_id / "manifest.json"


def make_manifest_id(start: datetime, end: datetime) -> str:
    material = f"{PRODUCT}|{GRANULARITY_SECONDS}|{iso_z(start)}|{iso_z(end)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def write_chunk(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def read_chunk(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        if set(reader.fieldnames or []) != required:
            raise AcquisitionError(f"Unexpected chunk schema: {path}")

        return list(reader)


def build_raw_csv(
    start: datetime,
    end: datetime,
    manifest_id: str,
) -> tuple[int, str, str, int]:
    """
    Combine completed chunks into the final raw CSV.

    Returns:
        row_count, actual_first, actual_last, duplicate_count
    """
    all_rows: list[dict[str, str]] = []

    ranges = list(chunk_ranges(start, end))
    for index, chunk_start, chunk_end in ranges:
        path = chunk_paths(manifest_id, index)
        if not path.exists():
            raise AcquisitionError(f"Missing completed chunk: {path}")
        all_rows.extend(read_chunk(path))

    all_rows.sort(key=lambda row: row["timestamp_utc"])

    timestamps = [row["timestamp_utc"] for row in all_rows]
    duplicates = {
        ts for ts in timestamps if timestamps.count(ts) > 1
    }

    # This is deliberately a hard failure. No duplicate is silently removed.
    if duplicates:
        raise AcquisitionError(
            "Duplicate timestamps detected while assembling the raw dataset: "
            + ", ".join(sorted(duplicates)[:10])
            + (" ..." if len(duplicates) > 10 else "")
        )

    expected = expected_timestamps(start, end)
    actual = [
        int(
            datetime.fromisoformat(
                row["timestamp_utc"].replace("Z", "+00:00")
            ).timestamp()
        )
        for row in all_rows
    ]

    expected_set = set(expected)
    actual_set = set(actual)

    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    if unexpected:
        raise AcquisitionError(
            f"Unexpected timestamps found in final dataset: {unexpected[:10]}"
        )

    # Missing hours are NOT filled. They are reported and the acquisition
    # remains a valid raw snapshot because Coinbase documents that historical
    # candles can be absent when there were no ticks.
    if missing:
        print(
            f"WARNING: {len(missing)} hourly buckets are missing. "
            "They were not filled or interpolated.",
            file=sys.stderr,
        )

    tmp = RAW_PATH.with_name(RAW_PATH.name + ".tmp")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, RAW_PATH)

    actual_first = all_rows[0]["timestamp_utc"] if all_rows else None
    actual_last = all_rows[-1]["timestamp_utc"] if all_rows else None

    return len(all_rows), actual_first, actual_last, len(duplicates)


def validate_existing_manifest(
    manifest: dict,
    start: datetime,
    end: datetime,
) -> None:
    expected = {
        "source": "Coinbase Exchange REST API",
        "product": PRODUCT,
        "granularity_seconds": GRANULARITY_SECONDS,
        "requested_start": iso_z(start),
        "requested_end_exclusive": iso_z(end),
        "chunk_hours": CHUNK_HOURS,
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise AcquisitionError(
                f"Existing checkpoint manifest does not match current "
                f"request for {key!r}: {manifest.get(key)!r} != {value!r}. "
                "Use the matching start/end range or remove the old "
                "checkpoint directory intentionally."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Coinbase BTC-USD hourly OHLCV data."
    )
    parser.add_argument(
        "--start",
        required=True,
        type=parse_utc_hour,
        help="Inclusive UTC start, e.g. 2021-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=parse_utc_hour,
        help=(
            "Exclusive UTC end, e.g. 2026-01-01T00:00:00Z. "
            "Must be later than --start."
        ),
    )
    args = parser.parse_args()

    start: datetime = args.start
    end: datetime = args.end

    if end <= start:
        parser.error("--end must be later than --start.")

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_id = make_manifest_id(start, end)
    mpath = manifest_path(manifest_id)

    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        validate_existing_manifest(manifest, start, end)
    else:
        manifest = {
            "source": "Coinbase Exchange REST API",
            "api_url": API_URL,
            "product": PRODUCT,
            "granularity_seconds": GRANULARITY_SECONDS,
            "requested_start": iso_z(start),
            "requested_end_exclusive": iso_z(end),
            "chunk_hours": CHUNK_HOURS,
            "manifest_id": manifest_id,
            "created_at_utc": iso_z(datetime.now(timezone.utc)),
            "completed_chunks": [],
            "chunk_diagnostics": {},
        }
        atomic_write_json(mpath, manifest)

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "BTC-Hackathon-Data-Acquisition/1.0",
        }
    )

    ranges = list(chunk_ranges(start, end))
    total_chunks = len(ranges)

    print(
        f"Acquiring {PRODUCT} hourly candles from {iso_z(start)} "
        f"to {iso_z(end)} (end exclusive)."
    )
    print(f"Chunks: {total_chunks} × up to {CHUNK_HOURS} hours.")

    for index, chunk_start, chunk_end in ranges:
        path = chunk_paths(manifest_id, index)

        if path.exists():
            # Validate an existing checkpoint chunk before trusting it.
            rows = read_chunk(path)
            expected_count = int(
                (chunk_end - chunk_start).total_seconds()
                // GRANULARITY_SECONDS
            )
            timestamps = [
                int(
                    datetime.fromisoformat(
                        row["timestamp_utc"].replace("Z", "+00:00")
                    ).timestamp()
                )
                for row in rows
            ]

            if len(timestamps) != len(set(timestamps)):
                raise AcquisitionError(
                    f"Duplicate timestamps in existing checkpoint chunk: {path}"
                )

            if timestamps != sorted(timestamps):
                raise AcquisitionError(
                    f"Unsorted timestamps in existing checkpoint chunk: {path}"
                )

            if any(
                ts < unix_seconds(chunk_start)
                or ts >= unix_seconds(chunk_end)
                for ts in timestamps
            ):
                raise AcquisitionError(
                    f"Out-of-range timestamp in existing checkpoint chunk: {path}"
                )

            # A chunk may legitimately contain fewer rows than expected due
            # to Coinbase's documented missing-candle behavior.
            manifest.setdefault("completed_chunks", [])
            if index not in manifest["completed_chunks"]:
                manifest["completed_chunks"].append(index)
                manifest["completed_chunks"].sort()
                atomic_write_json(mpath, manifest)

            print(
                f"[{index + 1}/{total_chunks}] checkpoint exists: "
                f"{len(rows)} rows"
            )
            continue

        print(
            f"[{index + 1}/{total_chunks}] requesting "
            f"{iso_z(chunk_start)} -> {iso_z(chunk_end)}"
        )

        candles = request_json(session, chunk_start, chunk_end)
        rows, out_of_range = normalize_candles(
            candles,
            chunk_start,
            chunk_end,
        )

        expected_count = int(
            (chunk_end - chunk_start).total_seconds()
            // GRANULARITY_SECONDS
        )

        timestamps = [
            int(
                datetime.fromisoformat(
                    row["timestamp_utc"].replace("Z", "+00:00")
                ).timestamp()
            )
            for row in rows
        ]

        if len(timestamps) != len(set(timestamps)):
            raise AcquisitionError(
                f"Duplicate timestamps detected after normalization in "
                f"chunk {index}."
            )

        if timestamps != sorted(timestamps):
            raise AcquisitionError(
                f"Chunk {index} is not chronological after normalization."
            )

        if out_of_range:
            print(
                f"  note: Coinbase returned {len(out_of_range)} "
                "out-of-range candle(s); they were excluded from this "
                "chunk because the requested interval is authoritative."
            )

        missing_count = expected_count - len(rows)
        if missing_count < 0:
            raise AcquisitionError(
                f"Chunk {index} returned more in-range candles than the "
                f"{expected_count} requested hourly buckets."
            )

        write_chunk(path, rows)

        manifest.setdefault("completed_chunks", [])
        manifest["completed_chunks"].append(index)
        manifest["completed_chunks"] = sorted(set(manifest["completed_chunks"]))
        manifest.setdefault("chunk_diagnostics", {})[str(index)] = {
            "requested_start": iso_z(chunk_start),
            "requested_end_exclusive": iso_z(chunk_end),
            "api_response_rows": len(candles),
            "in_range_rows": len(rows),
            "missing_hours_in_chunk": missing_count,
            "out_of_range_rows": len(out_of_range),
            "saved_at_utc": iso_z(datetime.now(timezone.utc)),
        }
        atomic_write_json(mpath, manifest)

    if len(manifest.get("completed_chunks", [])) != total_chunks:
        raise AcquisitionError(
            "Not all chunks are complete; final raw CSV was not written."
        )

    row_count, actual_first, actual_last, duplicate_count = build_raw_csv(
        start,
        end,
        manifest_id,
    )

    # Compute missing-hour count for metadata without filling anything.
    expected_count = len(expected_timestamps(start, end))
    missing_hours = expected_count - row_count

    downloaded_at = datetime.now(timezone.utc)

    metadata = {
        "source": "Coinbase Exchange REST API",
        "api_url": API_URL,
        "product": PRODUCT,
        "granularity": "1 hour",
        "granularity_seconds": GRANULARITY_SECONDS,
        "timezone": "UTC",
        "download_timestamp_utc": iso_z(downloaded_at),
        "requested_start": iso_z(start),
        "requested_end_exclusive": iso_z(end),
        "actual_first_timestamp": actual_first,
        "actual_last_timestamp": actual_last,
        "row_count": row_count,
        "expected_hourly_buckets": expected_count,
        "missing_hour_count": missing_hours,
        "duplicate_timestamp_count": duplicate_count,
        "raw_csv_sha256": sha256_file(RAW_PATH),
        "raw_csv_path": str(RAW_PATH),
        "restart_manifest_id": manifest_id,
        "chunk_hours": CHUNK_HOURS,
        "notes": [
            "Raw OHLCV only; no indicators or forecasting features were created.",
            "No missing hours were filled.",
            "No prices were interpolated.",
            "No suspicious observations were removed.",
            "Coinbase candle rows are interpreted as [time, low, high, open, close, volume].",
            "Missing hourly buckets are retained as missing because Coinbase documents that historical candles may be incomplete when there are no ticks.",
        ],
    }

    atomic_write_json(METADATA_PATH, metadata)

    print("\nAcquisition complete.")
    print(f"Raw CSV:  {RAW_PATH}")
    print(f"Metadata:  {METADATA_PATH}")
    print(f"Rows:      {row_count:,}")
    print(f"Missing:   {missing_hours:,} hourly buckets (not filled)")
    print(f"SHA-256:   {metadata['raw_csv_sha256']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
