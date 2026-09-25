import numpy as np
import pandas as pd
import pytest

from src.volatility_baseline import (
    compute_gap_aware_returns,
    rolling_variance_forecast,
    qlike,
    mse_squared_return,
)


def make_df():
    ts = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    close = [100, 101, 102, 101, 103, 104, 103, 105]
    return pd.DataFrame({
        "timestamp_utc": ts,
        "close": close,
        "target_status": ["valid"] * 7 + ["dataset_end"],
    })


def test_returns_break_at_gap():
    df = make_df()
    df = df.drop(index=[3]).reset_index(drop=True)
    out = compute_gap_aware_returns(df)
    assert pd.isna(out.loc[3, "log_return_1h"])


def test_rolling_variance_requires_consecutive_window():
    df = make_df()
    out = compute_gap_aware_returns(df)
    var = rolling_variance_forecast(out["log_return_1h"], out["timestamp_utc"], 3)
    assert var.iloc[:3].isna().all()
    assert np.isfinite(var.iloc[3])


def test_rolling_variance_resets_after_gap():
    df = make_df().drop(index=[3]).reset_index(drop=True)
    out = compute_gap_aware_returns(df)
    var = rolling_variance_forecast(out["log_return_1h"], out["timestamp_utc"], 3)
    # After the gap, three new consecutive returns are required.
    assert var.iloc[3:5].isna().all()


def test_qlike_rejects_empty():
    with pytest.raises(ValueError):
        qlike(np.array([np.nan]), np.array([1.0]))


def test_mse_is_nonnegative():
    assert mse_squared_return(np.array([1e-4, 4e-4]), np.array([2e-4, 3e-4])) >= 0


def test_positive_variance_floor():
    r = pd.Series([0.0, 0.0, 0.0])
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    out = rolling_variance_forecast(r, ts, 2)
    assert out.iloc[-1] > 0
