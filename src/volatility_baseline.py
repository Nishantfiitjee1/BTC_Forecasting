"""Simple gap-aware volatility baselines for one-hour BTC forecasting.

The forecast made at origin t uses only returns through t and predicts
the variance of return t+1. No future information is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


WINDOWS = (24, 168, 720)
EPSILON = 1e-16


@dataclass(frozen=True)
class VolatilityForecast:
    window: int
    origin_timestamp: pd.Timestamp
    variance_forecast: float
    target_return: float


def compute_gap_aware_returns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_utc", "close", "target_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    if out["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    close = pd.to_numeric(out["close"], errors="raise")
    if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError("Close prices must be finite and strictly positive.")

    ts = out["timestamp_utc"]
    gap_ok = ts.diff().eq(pd.Timedelta(hours=1))
    returns = np.full(len(out), np.nan, dtype=float)
    returns[gap_ok.to_numpy()] = (
        np.log(close[gap_ok].to_numpy())
        - np.log(close.shift(1)[gap_ok].to_numpy())
    )
    out["log_return_1h"] = returns
    return out


def rolling_variance_forecast(
    returns: pd.Series,
    timestamps: pd.Series,
    window: int,
) -> pd.Series:
    """Forecast variance at t+1 using the last `window` returns through t.

    A forecast exists only when the window consists of consecutive hourly
    observations. This prevents a long calendar gap from being treated as
    a single-step return history.
    """
    r = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    ts = pd.to_datetime(timestamps, utc=True)
    result = np.full(len(r), np.nan, dtype=float)

    valid_run = 0
    for i in range(len(r)):
        if i == 0 or not np.isfinite(r[i]) or ts[i] - ts[i - 1] != pd.Timedelta(hours=1):
            valid_run = 0
        else:
            valid_run += 1

        # valid_run counts consecutive valid returns ending at i.
        if valid_run >= window:
            sample = r[i - window + 1 : i + 1]
            var = float(np.var(sample, ddof=1))
            result[i] = max(var, EPSILON)

    return pd.Series(result, index=returns.index, name=f"rolling_var_{window}")


def qlike(realized_squared_return: np.ndarray, variance_forecast: np.ndarray) -> float:
    y = np.asarray(realized_squared_return, dtype=float)
    v = np.asarray(variance_forecast, dtype=float)

    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    if not mask.any():
        raise ValueError("No valid observations for QLIKE.")

    ratio = y[mask] / v[mask]

    # Same normalized QLIKE definition used by the GARCH evaluation:
    # ratio - log(ratio) - 1
    return float(np.mean(
        ratio - np.log(ratio + 1e-300) - 1.0
    ))


def mse_squared_return(realized_squared_return: np.ndarray, variance_forecast: np.ndarray) -> float:
    y = np.asarray(realized_squared_return, dtype=float)
    v = np.asarray(variance_forecast, dtype=float)
    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    if not mask.any():
        raise ValueError("No valid observations for variance MSE.")
    return float(np.mean((y[mask] - v[mask]) ** 2))
