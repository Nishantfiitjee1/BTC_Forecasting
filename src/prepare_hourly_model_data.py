"""
Prepare a leakage-safe hourly BTC/USD modeling dataset.

Purpose
-------
Transform the frozen raw Coinbase BTC-USD hourly OHLCV snapshot into a
modeling dataset with an explicitly defined one-hour-ahead target.

Target definition
-----------------
For an observation at timestamp t, the one-hour-ahead target is the CLOSE
of the candle whose timestamp is exactly t + 1 hour.

A target is valid only when that exact next-hour candle exists in the raw
dataset. We never use pandas shift(-1) alone because that would incorrectly
bridge gaps.

This script:
- never modifies the raw input
- validates schema, timestamps, numeric finiteness, OHLC consistency
- detects duplicate timestamps
- identifies gaps without filling/interpolating them
- creates explicit target availability/status
- creates next-hour close and log-return targets only for contiguous hours
- records a reproducibility/audit JSON with input/output SHA-256 hashes

It does NOT create forecasting features or split train/test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


RAW_DEFAULT: Final = Path("data/raw/btc_usd_hourly_raw.csv")
OUTPUT_DEFAULT: Final = Path("data/processed/btc_usd_hourly_model.csv")
AUDIT_DEFAULT: Final = Path("data/processed/btc_usd_hourly_model_audit.json")

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
)


class DataPreparationError(RuntimeError):
    """Raised when the raw dataset violates the preparation contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe BTC/USD hourly modeling data."
    )
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--audit", type=Path, default=AUDIT_DEFAULT)
    return parser.parse_args()


def fail(message: str) -> None:
    raise DataPreparationError(message)


def validate_schema(df: pd.DataFrame) -> None:
    if tuple(df.columns) != REQUIRED_COLUMNS:
        fail(
            "Unexpected input schema.\n"
            f"Expected: {list(REQUIRED_COLUMNS)}\n"
            f"Received: {list(df.columns)}"
        )


def validate_timestamps(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        fail("Input dataset is empty.")

    if df["timestamp_utc"].isna().any():
        fail("Input contains missing timestamps.")

    if not isinstance(df["timestamp_utc"].dtype, pd.DatetimeTZDtype):
        fail("timestamp_utc must be timezone-aware UTC timestamps.")

    if str(df["timestamp_utc"].dt.tz) != "UTC":
        fail("timestamp_utc must use UTC.")

    if df["timestamp_utc"].duplicated().any():
        duplicates = (
            df.loc[df["timestamp_utc"].duplicated(keep=False), "timestamp_utc"]
            .astype(str)
            .tolist()
        )
        fail(f"Duplicate timestamps detected: {duplicates[:10]}")

    diffs = df["timestamp_utc"].diff().dropna()

    if not (diffs > pd.Timedelta(0)).all():
        fail("Timestamps are not strictly increasing.")

    off_grid = df["timestamp_utc"].dt.minute.ne(0) | (
        df["timestamp_utc"].dt.second.ne(0)
    ) | (df["timestamp_utc"].dt.microsecond.ne(0))

    if off_grid.any():
        bad = df.loc[off_grid, "timestamp_utc"].head(10).astype(str).tolist()
        fail(f"Non-hour-aligned timestamps detected: {bad}")

    return diffs


def validate_numeric_data(df: pd.DataFrame) -> None:
    for column in PRICE_COLUMNS + ("volume",):
        values = pd.to_numeric(df[column], errors="coerce")

        if values.isna().any():
            fail(f"Column {column!r} contains non-numeric or missing values.")

        finite = np.isfinite(values.to_numpy(dtype=float))
        if not finite.all():
            fail(f"Column {column!r} contains non-finite values.")

    if (df[list(PRICE_COLUMNS)] <= 0).any().any():
        fail("One or more OHLC prices are non-positive.")

    if (df["volume"] < 0).any():
        fail("Negative volume detected.")


def validate_ohlc(df: pd.DataFrame) -> None:
    high_violation = df["high"] < df[list(("open", "close"))].max(axis=1)
    low_violation = df["low"] > df[list(("open", "close"))].min(axis=1)

    if high_violation.any():
        idx = df.index[high_violation][:10].tolist()
        fail(f"OHLC high violation detected at rows: {idx}")

    if low_violation.any():
        idx = df.index[low_violation][:10].tolist()
        fail(f"OHLC low violation detected at rows: {idx}")

    if (df["low"] > df["high"]).any():
        idx = df.index[df["low"] > df["high"]][:10].tolist()
        fail(f"OHLC low > high violation detected at rows: {idx}")


def find_missing_hours(
    timestamps: pd.Series,
) -> pd.DatetimeIndex:
    expected = pd.date_range(
        start=timestamps.iloc[0],
        end=timestamps.iloc[-1],
        freq="1h",
        tz="UTC",
    )
    return expected.difference(pd.DatetimeIndex(timestamps))


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    next_timestamp = result["timestamp_utc"].shift(-1)
    next_close = result["close"].shift(-1)

    contiguous_next_hour = (
        next_timestamp - result["timestamp_utc"]
        == pd.Timedelta(hours=1)
    )

    result["target_timestamp_utc"] = (
        result["timestamp_utc"] + pd.Timedelta(hours=1)
    ).where(contiguous_next_hour)

    result["target_close_1h"] = next_close.where(contiguous_next_hour)

    result["target_return_1h"] = np.where(
        contiguous_next_hour,
        np.log(result["target_close_1h"] / result["close"]),
        np.nan,
    )

    result["target_available_1h"] = contiguous_next_hour.astype("int8")

    result["target_status"] = np.select(
        [
            contiguous_next_hour,
            result["timestamp_utc"].eq(result["timestamp_utc"].iloc[-1]),
        ],
        [
            "valid",
            "dataset_end",
        ],
        default="missing_next_hour",
    )

    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        fail(f"Input file does not exist: {args.input}")

    raw_sha256 = sha256_file(args.input)

    df = pd.read_csv(
        args.input,
        parse_dates=["timestamp_utc"],
    )

    validate_schema(df)

    # Normalize timestamp representation explicitly to UTC.
    try:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="raise",
        )
    except Exception as exc:
        fail(f"Unable to parse timestamp_utc as UTC: {exc}")

    # The raw file is expected to already be chronological. Do not silently
    # reorder it because ordering is part of the raw-data contract.
    validate_timestamps(df)
    validate_numeric_data(df)
    validate_ohlc(df)

    missing_hours = find_missing_hours(df["timestamp_utc"])

    model_df = build_targets(df)

    valid_targets = int(model_df["target_available_1h"].sum())
    missing_target_rows = int(
        (model_df["target_status"] == "missing_next_hour").sum()
    )
    dataset_end_rows = int(
        (model_df["target_status"] == "dataset_end").sum()
    )

    if dataset_end_rows != 1:
        fail(
            "Expected exactly one dataset_end row, "
            f"found {dataset_end_rows}."
        )

    # Final target sanity checks.
    valid = model_df["target_available_1h"].eq(1)

    if model_df.loc[valid, "target_timestamp_utc"].isna().any():
        fail("Valid target rows contain missing target timestamps.")

    if model_df.loc[valid, "target_close_1h"].isna().any():
        fail("Valid target rows contain missing target closes.")

    if model_df.loc[valid, "target_return_1h"].isna().any():
        fail("Valid target rows contain missing target returns.")

    if not (
        model_df.loc[valid, "target_timestamp_utc"]
        == model_df.loc[valid, "timestamp_utc"] + pd.Timedelta(hours=1)
    ).all():
        fail("Target timestamp alignment check failed.")

    if not np.isfinite(
        model_df.loc[valid, "target_return_1h"].to_numpy(dtype=float)
    ).all():
        fail("Valid target returns contain non-finite values.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Explicit column order is part of the modeling-data contract.
    output_columns = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target_timestamp_utc",
        "target_close_1h",
        "target_return_1h",
        "target_available_1h",
        "target_status",
    ]

    model_df = model_df[output_columns]

    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    # Preserve enough decimal precision for float64 round-tripping. The target return is recomputed from prices by the AR integrity check, so CSV serialization must not truncate it to ~8 decimal places.
    model_df.to_csv(temp_output, index=False, float_format="%.17g")
    temp_output.replace(args.output)

    output_sha256 = sha256_file(args.output)

    audit = {
        "schema_version": "1.0",
        "input": {
            "path": str(args.input),
            "sha256": raw_sha256,
            "row_count": int(len(df)),
            "first_timestamp_utc": df["timestamp_utc"].iloc[0].isoformat(),
            "last_timestamp_utc": df["timestamp_utc"].iloc[-1].isoformat(),
        },
        "target_definition": {
            "horizon": "1 hour",
            "target_price": "next hourly candle close",
            "target_return": "log(close[t+1] / close[t])",
            "eligibility_rule": (
                "The next observed timestamp must equal timestamp[t] + 1 hour."
            ),
            "gap_policy": (
                "Missing hourly buckets are not filled or interpolated; "
                "rows immediately preceding a gap have unavailable targets."
            ),
            "dataset_end_policy": (
                "The final observation has no future target and is marked dataset_end."
            ),
        },
        "data_quality": {
            "expected_hourly_buckets_between_first_last": int(
                len(df) + len(missing_hours)
            ),
            "actual_rows": int(len(df)),
            "missing_hour_count_between_first_last": int(len(missing_hours)),
            "missing_hour_timestamps_utc": [
                ts.isoformat() for ts in missing_hours
            ],
            "duplicate_timestamp_count": int(
                df["timestamp_utc"].duplicated().sum()
            ),
            "nan_count": int(df.isna().sum().sum()),
            "non_positive_price_count": int(
                (df[list(PRICE_COLUMNS)] <= 0).any(axis=1).sum()
            ),
            "negative_volume_count": int((df["volume"] < 0).sum()),
        },
        "target_quality": {
            "valid_target_rows": valid_targets,
            "missing_next_hour_rows": missing_target_rows,
            "dataset_end_rows": dataset_end_rows,
        },
        "output": {
            "path": str(args.output),
            "sha256": output_sha256,
            "row_count": int(len(model_df)),
        },
        "feature_policy": {
            "forecasting_features_created": False,
            "future_information_used": False,
        },
    }

    write_json(args.audit, audit)

    print("Model dataset preparation complete.")
    print(f"Input rows:                  {len(df):,}")
    print(f"Output rows:                 {len(model_df):,}")
    print(f"Missing source hours:        {len(missing_hours):,}")
    print(f"Valid 1h targets:            {valid_targets:,}")
    print(f"Missing-next-hour targets:   {missing_target_rows:,}")
    print(f"Dataset-end rows:            {dataset_end_rows:,}")
    print(f"Raw SHA-256:                 {raw_sha256}")
    print(f"Model CSV SHA-256:           {output_sha256}")
    print(f"Model CSV:                   {args.output}")
    print(f"Audit JSON:                  {args.audit}")

    if missing_hours.size:
        print("\nMissing source hourly buckets:")
        for ts in missing_hours:
            print(f"  {ts.isoformat()}")

    print("\nTarget-status counts:")
    print(model_df["target_status"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataPreparationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
