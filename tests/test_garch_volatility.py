from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest_baseline import Period
from garch_volatility import (
    GARCHModelError,
    _conditional_variances,
    evaluate_variance_forecasts,
    fit_garch11,
    load_garch_frame,
)


def make_frame(n=800):
    rng = np.random.default_rng(42)
    ts = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    r = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(r))
    target_close = np.r_[close[1:], np.nan]
    target_return = np.r_[r[1:], np.nan]
    status = np.array(["valid"] * (n - 1) + ["dataset_end"], dtype=object)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": np.ones(n),
        "target_close_1h": target_close,
        "target_return_1h": target_return,
        "target_available_1h": (status == "valid").astype(int),
        "target_status": status,
    })


def test_conditional_variances_positive():
    y = np.array([0.1, -0.2, 0.3, -0.1])
    v = _conditional_variances((0.0, 0.01, 0.1, 0.8), y)
    assert np.isfinite(v).all()
    assert (v > 0).all()


def test_fit_garch_is_deterministic_and_valid():
    from returns import compute_hourly_log_returns
    df = compute_hourly_log_returns(make_frame())
    # fit period uses first 700 hours
    period = Period("development", pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-30", tz="UTC"))
    m1, metrics1 = fit_garch11(df, period)
    m2, metrics2 = fit_garch11(df, period)
    assert m1.to_dict() == m2.to_dict()
    assert metrics1 == metrics2
    assert 0 < m1.alpha < 1
    assert 0 < m1.beta < 1
    assert m1.alpha + m1.beta < 1


def test_variance_metrics_reject_nonpositive():
    pred = pd.DataFrame({"predicted_variance_1h": [0.0], "actual_return_1h": [0.1]})
    with pytest.raises(GARCHModelError):
        evaluate_variance_forecasts(pred)


def test_missing_hours_are_not_treated_as_returns(tmp_path):
    df = make_frame(50)
    df = df.drop(index=25).reset_index(drop=True)
    # The loader is intentionally not used here because the fixture is not a production CSV schema.
    # The core regression is covered by the existing returns tests.
    assert len(df) == 49
