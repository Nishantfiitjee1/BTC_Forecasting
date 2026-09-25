"""
BTC/USD Round-1 persistence baseline backtest.

Purpose
-------
Establish the first leakage-safe benchmark for the one-hour-ahead
BTC/USD forecasting problem.

Forecast definition
-------------------
At forecast origin t, use only the completed candle at t and predict:

    predicted_close_(t+1) = close_t
    predicted_return_(t+1) = 0

The processed modeling dataset is expected to have been created by
prepare_hourly_model_data.py and must contain an explicit target_status.

Evaluation windows
------------------
Development: 2021-01-01 through 2023-12-31
Validation:  2024-01-01 through 2024-12-31
Test:        2025-01-01 through 2025-12-31

The test period is never used to select or tune this baseline.

This script intentionally does NOT:
- fill missing hours
- interpolate prices
- create technical indicators
- train an ML model
- use future observations as features
- calculate trading P&L

Trading/P&L evaluation will be added only after the forecasting
benchmark is established.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/btc_usd_hourly_model.csv")
DEFAULT_OUTPUT_DIR = Path("results/backtests/persistence")


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


PERIODS = (
    Period(
        "development",
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
    Period(
        "validation",
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    Period(
        "test",
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
)


REQUIRED_COLUMNS = {
    "timestamp_utc",
    "close",
    "target_close_1h",
    "target_return_1h",
    "target_status",
}


class BacktestError(RuntimeError):
    """Raised when the backtest input violates the dataset contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BTC/USD one-hour persistence baseline backtest."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Model dataset path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise BacktestError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise BacktestError(
            f"Input dataset is missing required columns: {sorted(missing)}"
        )

    # Parse explicitly as UTC. A naive timestamp is not acceptable.
    try:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], utc=True, errors="raise"
        )
    except Exception as exc:
        raise BacktestError("timestamp_utc contains an invalid timestamp.") from exc

    if df["timestamp_utc"].duplicated().any():
        duplicates = (
            df.loc[df["timestamp_utc"].duplicated(keep=False), "timestamp_utc"]
            .astype(str)
            .tolist()
        )
        raise BacktestError(
            f"Duplicate timestamps detected; refusing to backtest. Examples: {duplicates[:5]}"
        )

    if not df["timestamp_utc"].is_monotonic_increasing:
        raise BacktestError(
            "timestamp_utc is not strictly chronological. "
            "The backtester refuses to sort silently."
        )

    if not df["timestamp_utc"].is_unique:
        raise BacktestError("timestamp_utc must be unique.")

    # Numeric validation for fields used by the baseline.
    for column in ("close", "target_close_1h", "target_return_1h"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            # target columns may be NaN for explicitly unavailable targets.
            if column in {"target_close_1h", "target_return_1h"}:
                continue
            raise BacktestError(f"{column} contains non-numeric values.")
        if not np.isfinite(values.dropna()).all():
            raise BacktestError(f"{column} contains non-finite values.")

    if (df["close"] <= 0).any():
        raise BacktestError("close contains non-positive values.")

    allowed_statuses = {"valid", "missing_next_hour", "dataset_end"}
    statuses = set(df["target_status"].dropna().unique())
    unexpected = statuses - allowed_statuses
    if unexpected:
        raise BacktestError(
            f"Unexpected target_status values: {sorted(unexpected)}"
        )

    # The baseline may only evaluate rows explicitly marked valid.
    invalid_status_target = df["target_status"].eq("valid") & (
        df["target_close_1h"].isna() | df["target_return_1h"].isna()
    )
    if invalid_status_target.any():
        examples = df.loc[
            invalid_status_target, ["timestamp_utc", "target_status"]
        ].head(5)
        raise BacktestError(
            "Rows marked 'valid' contain missing target values. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # Conversely, unavailable targets must not contain target values.
    unavailable_with_target = ~df["target_status"].eq("valid") & (
        df["target_close_1h"].notna() | df["target_return_1h"].notna()
    )
    if unavailable_with_target.any():
        raise BacktestError(
            "Rows with unavailable target_status contain target values."
        )

    return df


def select_period(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    mask = (
        (df["timestamp_utc"] >= period.start)
        & (df["timestamp_utc"] < period.end_exclusive)
    )
    period_df = df.loc[mask].copy()

    # Only rows with a real next-hour target are eligible for evaluation.
    period_df = period_df.loc[period_df["target_status"].eq("valid")].copy()

    return period_df


def evaluate_predictions(period_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if period_df.empty:
        raise BacktestError("No valid target rows available for evaluation.")

    out = period_df[
        [
            "timestamp_utc",
            "close",
            "target_close_1h",
            "target_return_1h",
        ]
    ].copy()

    out.rename(
        columns={
            "close": "forecast_origin_close",
            "target_close_1h": "actual_close_1h",
            "target_return_1h": "actual_return_1h",
        },
        inplace=True,
    )

    # Persistence forecast:
    # predict the next-hour close as the current completed-hour close.
    out["predicted_close_1h"] = out["forecast_origin_close"]
    out["predicted_return_1h"] = 0.0

    close_error = (
        out["predicted_close_1h"] - out["actual_close_1h"]
    )
    abs_error = close_error.abs()
    squared_error = close_error.pow(2)

    actual_return = out["actual_return_1h"]
    predicted_return = out["predicted_return_1h"]

    # A persistence forecast is deliberately direction-neutral:
    # predicted_return_(t+1) = 0 for every observation.
    #
    # Therefore conventional directional accuracy is NOT a meaningful
    # metric for this baseline. Reporting it as 0% would incorrectly
    # imply that the baseline made directional calls and got all of them
    # wrong. We record it explicitly as not applicable.
    metrics = {
        "n_predictions": int(len(out)),
        "mae_close": float(abs_error.mean()),
        "rmse_close": float(np.sqrt(squared_error.mean())),
        "median_absolute_error_close": float(abs_error.median()),
        "mean_error_close": float(close_error.mean()),
        "mean_absolute_actual_return": float(actual_return.abs().mean()),
        "rmse_return": float(
            np.sqrt((predicted_return - actual_return).pow(2).mean())
        ),
        "directional_accuracy": None,
        "directional_accuracy_status": "not_applicable_zero_return_forecast",
        "zero_actual_return_count": int(actual_return.eq(0).sum()),
        "forecast_origin_first": out["timestamp_utc"].min().isoformat(),
        "forecast_origin_last": out["timestamp_utc"].max().isoformat(),
    }

    return out, metrics


def run(input_path: Path, output_dir: Path) -> None:
    df = load_and_validate(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {
        "method": "persistence",
        "forecast_definition": (
            "predicted_close_(t+1) = close_t; predicted_return_(t+1) = 0"
        ),
        "input_file": str(input_path),
        "periods": {},
    }

    for period in PERIODS:
        period_df = select_period(df, period)

        predictions, metrics = evaluate_predictions(period_df)

        prediction_path = output_dir / f"{period.name}_predictions.csv"
        metrics_path = output_dir / f"{period.name}_metrics.json"

        predictions.to_csv(prediction_path, index=False)

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)

        all_metrics["periods"][period.name] = metrics

        print(
            f"{period.name:>12}: "
            f"{metrics['n_predictions']:>6,} predictions | "
            f"MAE={metrics['mae_close']:.4f} | "
            f"RMSE={metrics['rmse_close']:.4f} | "
            f"ReturnRMSE={metrics['rmse_return']:.8f} | "
            f"DirAcc=N/A"
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, allow_nan=False)

    print("\nPersistence baseline complete.")
    print(f"Results: {output_dir}")


def main() -> int:
    args = parse_args()

    try:
        run(args.input, args.output_dir)
    except BacktestError as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
