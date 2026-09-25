"""Production one-hour-ahead BTC/USD forecast engine.

Protocol:
- Point forecast: refit the locked AR(3)+intercept specification on all
  currently available historical model data, then forecast the next hour.
- Volatility: refit the locked Normal GARCH(1,1) specification on all
  currently available valid hourly returns, then forecast next-hour variance.
- Intervals: use the previously selected validation-only calibration rule:
  empirical standardized-residual quantiles for 50/80/90%; Gaussian 95%.
- No test-set selection, no automatic model switching, no interpolation.
- The input must be the already validated hourly model dataset.

This script is for the final production forecast after the research/backtest
protocol has already selected the model specifications. It does not alter the
locked backtest artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_baseline import Period
from fit_ar_baseline import (
    ARModelError,
    build_ar_design_matrix,
    fit_ar_model,
    load_model_frame,
)
from garch_volatility import (
    GARCH11Model,
    GARCHModelError,
    fit_garch11,
    load_garch_frame,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "btc_usd_hourly_model.csv"
DEFAULT_CALIBRATION = (
    ROOT
    / "results"
    / "backtests"
    / "gaussian_calibration"
    / "validation_calibration_predictions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "forecasts"


class ForecastEngineError(RuntimeError):
    pass


def _history_period(df: pd.DataFrame, *, end_exclusive: pd.Timestamp | None = None) -> Period:
    if df.empty:
        raise ForecastEngineError("Input dataset is empty.")
    first = pd.Timestamp(df["timestamp_utc"].min())
    last = pd.Timestamp(df["timestamp_utc"].max())
    if end_exclusive is None:
        end_exclusive = last + pd.Timedelta(hours=1)
    end_exclusive = pd.Timestamp(end_exclusive)
    if end_exclusive <= first:
        raise ForecastEngineError("Production history period is empty.")
    return Period(
        name="production_history",
        start=first,
        end_exclusive=end_exclusive,
    )


def _validate_latest_row(df: pd.DataFrame) -> int:
    if df["timestamp_utc"].duplicated().any():
        raise ForecastEngineError("Input contains duplicate timestamps.")
    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ForecastEngineError("Input timestamps are not sorted.")
    if len(df) < 4:
        raise ForecastEngineError("At least four hourly rows are required.")

    latest_idx = len(df) - 1
    latest_ts = pd.Timestamp(df.at[latest_idx, "timestamp_utc"])

    if latest_idx == 0 or latest_ts - pd.Timestamp(df.at[latest_idx - 1, "timestamp_utc"]) != pd.Timedelta(hours=1):
        raise ForecastEngineError(
            "Latest observation does not have a contiguous preceding hour; "
            "refusing to forecast across a terminal data gap."
        )

    if not np.isfinite(float(df.at[latest_idx, "close"])) or float(df.at[latest_idx, "close"]) <= 0:
        raise ForecastEngineError("Latest close must be finite and positive.")

    if int(df.at[latest_idx, "log_return_available_1h"]) != 1:
        raise ForecastEngineError("Latest hourly return is unavailable.")

    # The production origin must be the last observed candle. Its own target
    # is necessarily unknown at forecast time. Requiring the canonical
    # dataset-end marker prevents a refreshed file with an accidentally
    # attached future target from leaking that target into model fitting.
    if str(df.at[latest_idx, "target_status"]) != "dataset_end":
        raise ForecastEngineError(
            "Latest row must have target_status='dataset_end'; "
            "refusing to forecast from a row with a known/attached future target."
        )
    if pd.notna(df.at[latest_idx, "target_return_1h"]) or pd.notna(
        df.at[latest_idx, "target_close_1h"]
    ):
        raise ForecastEngineError(
            "Latest row contains a target value; refusing to risk target leakage."
        )

    return latest_idx


def _latest_ar_forecast(df: pd.DataFrame) -> tuple[float, object]:
    latest_idx = _validate_latest_row(df)
    latest_ts = pd.Timestamp(df.at[latest_idx, "timestamp_utc"])

    # AR fitting must stop strictly before the forecast origin. The origin row
    # has no future target available at forecast time. This remains safe even
    # if a caller supplies a malformed dataset whose latest row is mislabeled.
    period = _history_period(df, end_exclusive=latest_ts)

    model, fit_metrics = fit_ar_model(
        df,
        lag=3,
        include_intercept=True,
        period=period,
    )

    # For production, the final forecast origin has no known target, so build
    # its feature vector directly from the gap-safe return series.
    returns = df["log_return_1h"].to_numpy(dtype=float)
    latest_positions = np.arange(latest_idx - model.lag + 1, latest_idx + 1)
    feature_timestamps = pd.to_datetime(
        df.loc[latest_positions, "timestamp_utc"].to_numpy(), utc=True
    )
    if not np.all(np.diff(feature_timestamps.view("int64")) == pd.Timedelta(hours=1).value):
        raise ForecastEngineError(
            "Latest AR lag history contains a timestamp gap; refusing to bridge it."
        )

    X = returns[latest_positions[::-1]].reshape(1, model.lag)

    # Verify the same eligibility logic used by the canonical AR implementation
    # at the latest origin. This catches terminal-gap and missing-return errors.
    if not np.isfinite(X).all():
        raise ForecastEngineError("Latest AR feature vector is non-finite.")

    predicted_return = float(model.predict(X)[0])
    origin_close = float(df.at[latest_idx, "close"])
    predicted_close = float(origin_close * np.exp(predicted_return))

    return predicted_return, {
        "model": model,
        "fit_metrics": fit_metrics,
        "origin_close": origin_close,
        "predicted_close": predicted_close,
    }


def _garch_state_at_latest(df: pd.DataFrame, model: GARCH11Model) -> float:
    """Filter the fitted GARCH state through the latest observed return."""
    returns = df["log_return_1h"].to_numpy(dtype=float)
    available = df["log_return_available_1h"].to_numpy(dtype=bool)

    variance = float(model.unconditional_variance)
    last_timestamp: pd.Timestamp | None = None

    for i in range(len(df)):
        ts = pd.Timestamp(df.at[i, "timestamp_utc"])
        if not available[i]:
            variance = float(model.unconditional_variance)
            last_timestamp = None
            continue

        if last_timestamp is not None and ts - last_timestamp != pd.Timedelta(hours=1):
            variance = float(model.unconditional_variance)

        variance = model.forecast_variance(float(returns[i]), variance)
        last_timestamp = ts

    if not np.isfinite(variance) or variance <= 0:
        raise ForecastEngineError("Filtered GARCH variance is invalid.")
    return float(variance)


def _latest_garch_forecast(df: pd.DataFrame) -> tuple[float, dict]:
    latest_idx = _validate_latest_row(df)
    period = _history_period(df)
    model, fit_metrics = fit_garch11(df, period)

    latest_return = float(df.at[latest_idx, "log_return_1h"])

    state_variance = _garch_state_at_latest(df, model)
    # state_variance is already the one-step-ahead variance after the latest
    # observed return, so it is the forecast variance for latest+1 hour.
    predicted_variance = state_variance
    predicted_volatility = float(np.sqrt(predicted_variance))

    return predicted_variance, {
        "model": model,
        "fit_metrics": fit_metrics,
        "latest_return": latest_return,
        "predicted_volatility": predicted_volatility,
    }


def _empirical_calibration_quantiles(path: Path) -> dict[str, tuple[float, float]]:
    """Reconstruct the previously selected 70/30 validation calibration factors.

    The original calibration experiment estimated separate lower/upper
    standardized-residual quantiles. The production engine preserves that
    protocol exactly:
      50%, 80%, 90% -> empirical
      95% -> Gaussian

    Only the 2024 validation artifact is read here.
    """
    if not path.exists():
        raise ForecastEngineError(f"Calibration artifact not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    required = {"timestamp_utc", "standardized_residual"}
    missing = required.difference(df.columns)
    if missing:
        raise ForecastEngineError(
            f"Calibration artifact missing columns: {sorted(missing)}"
        )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    val_start = pd.Timestamp("2024-01-01", tz="UTC")
    val_end = pd.Timestamp("2025-01-01", tz="UTC")
    df = df.loc[
        (df["timestamp_utc"] >= val_start)
        & (df["timestamp_utc"] < val_end)
    ].copy()

    if len(df) != 8784:
        raise ForecastEngineError(
            f"Expected 8784 validation calibration rows, found {len(df)}."
        )
    if df["timestamp_utc"].duplicated().any() or not df["timestamp_utc"].is_monotonic_increasing:
        raise ForecastEngineError("Calibration timestamps are invalid.")

    z = df["standardized_residual"].to_numpy(dtype=float)
    if not np.isfinite(z).all():
        raise ForecastEngineError("Calibration standardized residuals are non-finite.")

    n_cal = int(np.floor(0.70 * len(z)))
    z_cal = z[:n_cal]
    if len(z_cal) < 100:
        raise ForecastEngineError("Too few calibration observations.")

    result: dict[str, tuple[float, float]] = {}
    for level in (0.50, 0.80, 0.90):
        lo = float(np.quantile(z_cal, (1.0 - level) / 2.0, method="linear"))
        hi = float(np.quantile(z_cal, (1.0 + level) / 2.0, method="linear"))
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ForecastEngineError(
                f"Invalid empirical quantiles for {level:.0%}: {lo}, {hi}"
            )
        result[str(int(level * 100))] = (lo, hi)

    result["95"] = (
        float(norm.ppf(0.025)),
        float(norm.ppf(0.975)),
    )
    return result


def _build_intervals(
    predicted_return: float,
    predicted_volatility: float,
    origin_close: float,
    calibration: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}

    for level in ("50", "80", "90", "95"):
        lo_factor, hi_factor = calibration[level]
        method = "empirical" if level != "95" else "gaussian"

        lower_return = predicted_return + lo_factor * predicted_volatility
        upper_return = predicted_return + hi_factor * predicted_volatility

        lower_close = origin_close * np.exp(lower_return)
        upper_close = origin_close * np.exp(upper_return)

        if not (
            np.isfinite(lower_return)
            and np.isfinite(upper_return)
            and np.isfinite(lower_close)
            and np.isfinite(upper_close)
            and lower_close <= upper_close
        ):
            raise ForecastEngineError(f"Invalid {level}% interval.")

        result[level] = {
            "method": method,
            "lower_return": float(lower_return),
            "upper_return": float(upper_return),
            "lower_close": float(lower_close),
            "upper_close": float(upper_close),
            "lower_factor": float(lo_factor),
            "upper_factor": float(hi_factor),
        }

    return result


def run(input_path: Path, calibration_path: Path, output_dir: Path) -> Path:
    ar_df = load_model_frame(input_path)
    latest_idx = _validate_latest_row(ar_df)

    garch_df = load_garch_frame(input_path)
    if len(ar_df) != len(garch_df) or not (
        ar_df["timestamp_utc"].to_numpy() == garch_df["timestamp_utc"].to_numpy()
    ).all():
        raise ForecastEngineError("AR and GARCH input frames do not align.")

    predicted_return, ar_info = _latest_ar_forecast(ar_df)
    predicted_variance, garch_info = _latest_garch_forecast(garch_df)

    predicted_volatility = float(np.sqrt(predicted_variance))
    origin_ts = pd.Timestamp(ar_df.at[latest_idx, "timestamp_utc"])
    target_ts = origin_ts + pd.Timedelta(hours=1)
    origin_close = float(ar_df.at[latest_idx, "close"])

    calibration = _empirical_calibration_quantiles(calibration_path)
    intervals = _build_intervals(
        predicted_return=predicted_return,
        predicted_volatility=predicted_volatility,
        origin_close=origin_close,
        calibration=calibration,
    )

    output = {
        "forecast_timestamp_utc": origin_ts.isoformat(),
        "target_timestamp_utc": target_ts.isoformat(),
        "input_latest_timestamp_utc": origin_ts.isoformat(),
        "forecast_origin_close": origin_close,
        "predicted_return_1h": predicted_return,
        "predicted_close_1h": float(origin_close * np.exp(predicted_return)),
        "predicted_variance_1h": predicted_variance,
        "predicted_volatility_1h": predicted_volatility,
        "production_refit_scope": {
            "point_model": "all labeled origins strictly before forecast origin",
            "volatility_model": "all valid hourly returns through forecast origin",
        },
        "point_model": {
            "specification": "AR(3)_intercept_true",
            "fit_rows": int(ar_info["model"].n_train),
            "train_origin_first": ar_info["model"].train_origin_first,
            "train_origin_last": ar_info["model"].train_origin_last,
            "fit_slice_sha256": ar_info["model"].fit_slice_sha256,
            "coefficients": list(ar_info["model"].coefficients),
            "intercept": float(ar_info["model"].intercept),
        },
        "volatility_model": {
            "specification": "GARCH(1,1)_normal",
            "fit_rows": int(garch_info["model"].n_train),
            "train_origin_first": garch_info["model"].train_origin_first,
            "train_origin_last": garch_info["model"].train_origin_last,
            "fit_slice_sha256": garch_info["model"].fit_slice_sha256,
            "mu": float(garch_info["model"].mu),
            "omega": float(garch_info["model"].omega),
            "alpha": float(garch_info["model"].alpha),
            "beta": float(garch_info["model"].beta),
            "alpha_plus_beta": float(
                garch_info["model"].alpha + garch_info["model"].beta
            ),
        },
        "interval_calibration": {
            "source": str(calibration_path),
            "selection_protocol": "2024 validation 70/30; 2025 test excluded",
            "factors": calibration,
            "intervals": intervals,
        },
        "protocol": {
            "target": "BTC/USD next 1 hour",
            "point_forecast": "AR(3) with intercept",
            "variance_forecast": "Normal GARCH(1,1)",
            "model_selection_used_test_data": False,
            "historical_test_period_reused_for_production_refit": True,
            "automatic_model_selection": False,
            "forecast_origin_is_last_observed_candle": True,
            "latest_origin_target_used_for_fit": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_forecast.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Origin:             {origin_ts.isoformat()}")
    print(f"Target:             {target_ts.isoformat()}")
    print(f"Origin close:       {origin_close:.8f}")
    print(f"Predicted return:   {predicted_return:.10g}")
    print(f"Predicted close:    {output['predicted_close_1h']:.8f}")
    print(f"Predicted volatility:{predicted_volatility:.10g}")
    for level, item in intervals.items():
        print(
            f"{level}% {item['method']:>8}: "
            f"{float(item['lower_close']):.8f} -> "
            f"{float(item['upper_close']):.8f}"
        )
    print(f"Output:             {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        run(args.input, args.calibration, args.output_dir)
    except (ARModelError, GARCHModelError, ForecastEngineError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
