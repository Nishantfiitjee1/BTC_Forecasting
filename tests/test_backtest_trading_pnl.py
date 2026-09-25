"""Unit tests for fixed-notional trading P&L backtest."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtest_trading_pnl import BacktestError, evaluate, load_predictions


def make_df(predicted_returns=(0.01, -0.02, 0.0), actual_returns=(0.02, 0.01, -0.03)):
    close = 100.0
    rows = []
    times = pd.date_range("2025-01-01", periods=len(predicted_returns), freq="h", tz="UTC")
    for ts, pred, actual in zip(times, predicted_returns, actual_returns):
        rows.append({
            "timestamp_utc": ts.isoformat(),
            "forecast_origin_close": close,
            "predicted_return_1h": pred,
            "predicted_close_1h": close * np.exp(pred),
            "actual_return_1h": actual,
            "actual_close_1h": close * np.exp(actual),
        })
        close = close * np.exp(actual)
    return pd.DataFrame(rows)


def test_evaluate_long_short_and_flat_zero_fee():
    df = make_df()
    summary, detail = evaluate(df, capital=1000.0, fee_bps=0.0)
    expected = 1000 * (np.expm1(0.02) - np.expm1(0.01))  # P&L uses simple returns derived from log returns
    assert summary["n_trades"] == 2
    assert summary["long_trades"] == 1
    assert summary["short_trades"] == 1
    assert summary["flat_forecasts"] == 1
    assert summary["gross_pnl"] == pytest.approx(expected)
    assert summary["fees_paid"] == pytest.approx(0.0)
    assert detail["position"].tolist() == [1, -1, 0]


def test_round_trip_fee_is_two_sides_per_active_trade():
    df = make_df(predicted_returns=(0.01, -0.01), actual_returns=(0.01, -0.01))
    summary, _ = evaluate(df, capital=1000.0, fee_bps=5.0)
    assert summary["n_trades"] == 2
    assert summary["fees_paid"] == pytest.approx(2 * 1000 * (2 * 5 / 10000))


def test_all_zero_signals_are_flat_and_no_fees():
    df = make_df(predicted_returns=(0.0, 0.0), actual_returns=(0.01, -0.01))
    summary, detail = evaluate(df, capital=1000.0, fee_bps=5.0)
    assert summary["n_trades"] == 0
    assert summary["fees_paid"] == pytest.approx(0.0)
    assert (detail["position"] == 0).all()


@pytest.mark.parametrize("capital,fee", [(0, 0), (-1, 0), (1000, -1), (float("nan"), 0), (1000, float("inf"))])
def test_evaluate_rejects_invalid_capital_or_fee(capital, fee):
    with pytest.raises(BacktestError):
        evaluate(make_df(), capital=capital, fee_bps=fee)


def write_csv(tmp_path, df):
    path = tmp_path / "predictions.csv"
    df.to_csv(path, index=False)
    return path


def test_load_predictions_accepts_consistent_rows(tmp_path):
    df = load_predictions(write_csv(tmp_path, make_df()))
    assert len(df) == 3
    assert str(df["timestamp_utc"].dt.tz) == "UTC"


def test_load_predictions_rejects_missing_columns(tmp_path):
    df = make_df().drop(columns=["actual_close_1h"])
    with pytest.raises(BacktestError, match="Missing columns"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_duplicate_timestamps(tmp_path):
    df = make_df()
    df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]
    with pytest.raises(BacktestError, match="duplicate"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_nonchronological_timestamps(tmp_path):
    df = make_df()
    df.loc[[0, 1], "timestamp_utc"] = df.loc[[1, 0], "timestamp_utc"].to_numpy()
    with pytest.raises(BacktestError, match="chronological"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_nonfinite_values(tmp_path):
    df = make_df()
    df.loc[0, "predicted_return_1h"] = np.nan
    with pytest.raises(BacktestError, match="NaN/Inf"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_inconsistent_predicted_close(tmp_path):
    df = make_df()
    df.loc[0, "predicted_close_1h"] += 5
    with pytest.raises(BacktestError, match="Predicted close inconsistent"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_inconsistent_actual_return(tmp_path):
    df = make_df()
    df.loc[0, "actual_return_1h"] += 0.01
    with pytest.raises(BacktestError, match="Actual return inconsistent"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_nonpositive_prices(tmp_path):
    df = make_df()
    df.loc[0, "actual_close_1h"] = 0
    with pytest.raises(BacktestError, match="Prices must be positive"):
        load_predictions(write_csv(tmp_path, df))


def test_load_predictions_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=make_df().columns).to_csv(path, index=False)
    with pytest.raises(BacktestError, match="No prediction rows"):
        load_predictions(path)


def test_load_predictions_rejects_missing_file(tmp_path):
    with pytest.raises(BacktestError, match="Missing input"):
        load_predictions(tmp_path / "missing.csv")
