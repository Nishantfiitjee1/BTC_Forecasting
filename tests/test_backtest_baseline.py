"""
Tests for the BTC/USD persistence baseline.

Run:
    python -m pytest tests/test_backtest_baseline.py -q
"""

from pathlib import Path

import math

import numpy as np
import pandas as pd
import pytest

# Allow the test to import src/backtest_baseline.py when executed from
# the repository root.
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest_baseline import BacktestError, evaluate_predictions, load_and_validate


def make_valid_df() -> pd.DataFrame:
    ts = pd.date_range(
        "2025-01-01T00:00:00Z",
        periods=4,
        freq="h",
    )

    close = [100.0, 101.0, 102.0, 103.0]
    target_close = [101.0, 102.0, 103.0, None]

    # target_return_1h must equal the production definition exactly:
    #     log(close[t+1] / close[t])
    # computed here from the exact close/target_close values above,
    # not approximated, so the fixture cannot silently drift from the
    # real formula again (see test_fixture_target_return_matches_...
    # below, which re-derives and checks this on every run).
    target_return = [
        math.log(101.0 / 100.0),
        math.log(102.0 / 101.0),
        math.log(103.0 / 102.0),
        None,
    ]

    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "close": close,
            "target_close_1h": target_close,
            "target_return_1h": target_return,
            "target_status": [
                "valid",
                "valid",
                "valid",
                "dataset_end",
            ],
        }
    )


def test_fixture_target_return_matches_log_return_definition():
    """
    Guards against the fixture drifting away from the real target
    definition (log(close[t+1] / close[t])) the way it did previously,
    when target_return_1h was hardcoded to an approximate 0.01 for
    every row instead of being derived from close/target_close.
    """
    df = make_valid_df()
    valid = df["target_status"].eq("valid")

    expected = np.log(
        df.loc[valid, "target_close_1h"].to_numpy(dtype=float)
        / df.loc[valid, "close"].to_numpy(dtype=float)
    )
    actual = df.loc[valid, "target_return_1h"].to_numpy(dtype=float)

    assert np.allclose(actual, expected)


def test_persistence_prediction_equals_current_close():
    df = make_valid_df().iloc[:3].copy()
    predictions, metrics = evaluate_predictions(df)

    assert predictions["predicted_close_1h"].tolist() == [100.0, 101.0, 102.0]
    assert predictions["predicted_return_1h"].tolist() == [0.0, 0.0, 0.0]
    assert metrics["n_predictions"] == 3


def test_duplicate_timestamps_are_rejected(tmp_path: Path):
    df = make_valid_df()
    df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="Duplicate timestamps"):
        load_and_validate(path)


def test_non_chronological_input_is_rejected(tmp_path: Path):
    df = make_valid_df().iloc[[0, 2, 1, 3]].copy()

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="not strictly chronological"):
        load_and_validate(path)


def test_valid_row_cannot_have_missing_target(tmp_path: Path):
    df = make_valid_df()
    df.loc[0, "target_close_1h"] = None

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="marked 'valid'"):
        load_and_validate(path)


def test_unavailable_target_cannot_contain_target_value(tmp_path: Path):
    df = make_valid_df()
    df.loc[3, "target_status"] = "dataset_end"
    df.loc[3, "target_close_1h"] = 104.0
    df.loc[3, "target_return_1h"] = 0.01

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="unavailable target_status"):
        load_and_validate(path)


def test_nonpositive_close_is_rejected(tmp_path: Path):
    df = make_valid_df()
    df.loc[0, "close"] = 0.0

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="non-positive"):
        load_and_validate(path)


def test_persistence_directional_accuracy_is_not_reported_as_zero():
    df = make_valid_df().iloc[:3].copy()
    _, metrics = evaluate_predictions(df)

    assert metrics["directional_accuracy"] is None
    assert metrics["directional_accuracy_status"] == (
        "not_applicable_zero_return_forecast"
    )


# ---------------------------------------------------------------------------
# New: coverage for previously-untested validated paths
# ---------------------------------------------------------------------------


def test_missing_required_column_is_rejected(tmp_path: Path):
    df = make_valid_df().drop(columns=["target_status"])

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="missing required columns"):
        load_and_validate(path)


def test_non_numeric_close_is_rejected(tmp_path: Path):
    df = make_valid_df()
    df["close"] = df["close"].astype(object)
    df.loc[0, "close"] = "not-a-number"

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="non-numeric values"):
        load_and_validate(path)


def test_non_finite_close_is_rejected(tmp_path: Path):
    df = make_valid_df()
    df.loc[0, "close"] = float("inf")

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="non-finite values"):
        load_and_validate(path)


def test_unrecognized_target_status_is_rejected(tmp_path: Path):
    df = make_valid_df()
    df.loc[0, "target_status"] = "bogus_status"

    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(BacktestError, match="Unexpected target_status values"):
        load_and_validate(path)


def test_empty_period_evaluation_is_rejected():
    empty_df = make_valid_df().iloc[0:0]

    with pytest.raises(BacktestError, match="No valid target rows available"):
        evaluate_predictions(empty_df)
