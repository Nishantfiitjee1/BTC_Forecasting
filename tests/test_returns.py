"""
Tests for src/returns.py -- the gap-aware hourly log-return and
AR-lag-validity engine.

Expected values are hand-derived and written down explicitly in each
test (per project standard: a future refactor must not be able to
silently change the semantics without a test failing and forcing the
author to re-justify the new numbers).

Run:
    python -m pytest tests/test_returns.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from returns import (
    ReturnsError,
    compute_consecutive_valid_return_run_length,
    compute_hourly_log_returns,
    usable_mask_for_lag,
)


def _ts(hours: list[str]) -> pd.Series:
    return pd.to_datetime(pd.Series(hours), utc=True)


def make_df(
    hours: list[str],
    closes: list[float],
    target_status: str | list[str] = "valid",
) -> pd.DataFrame:
    if isinstance(target_status, str):
        target_status = [target_status] * len(hours)
    return pd.DataFrame(
        {
            "timestamp_utc": _ts(hours),
            "close": closes,
            "target_status": target_status,
        }
    )


def make_consecutive_hours(n: int, start: str = "2024-01-01T00:00:00Z") -> list[str]:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in idx]


# ---------------------------------------------------------------------------
# Helper: run_length in isolation
# ---------------------------------------------------------------------------


def test_run_length_helper_directly():
    available = pd.Series([0, 1, 1, 1, 0, 1, 1], dtype="int8")
    result = compute_consecutive_valid_return_run_length(available)
    assert result.tolist() == [0, 1, 2, 3, 0, 1, 2]


# ---------------------------------------------------------------------------
# A-E: gap placement edge cases
# ---------------------------------------------------------------------------


def test_a_no_gaps_all_returns_valid_except_first_row():
    hours = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T02:00:00Z",
        "2024-01-01T03:00:00Z",
    ]
    closes = [100.0, 101.0, 99.0, 102.0]
    df = make_df(hours, closes)

    result = compute_hourly_log_returns(df)

    assert result["log_return_available_1h"].tolist() == [0, 1, 1, 1]
    assert result["consecutive_valid_return_run_length"].tolist() == [0, 1, 2, 3]

    assert np.isnan(result["log_return_1h"].iloc[0])
    assert result["log_return_1h"].iloc[1] == pytest.approx(np.log(101.0 / 100.0))
    assert result["log_return_1h"].iloc[2] == pytest.approx(np.log(99.0 / 101.0))
    assert result["log_return_1h"].iloc[3] == pytest.approx(np.log(102.0 / 99.0))


def test_b_one_hour_gap_breaks_run_length_once():
    # 00,01,02 present, 03 missing, 04,05 present.
    hours = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T02:00:00Z",
        "2024-01-01T04:00:00Z",
        "2024-01-01T05:00:00Z",
    ]
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    df = make_df(hours, closes)

    result = compute_hourly_log_returns(df)

    # Row 04:00 is 2 hours after 02:00 -> its return is undefined.
    assert result["log_return_available_1h"].tolist() == [0, 1, 1, 0, 1]
    assert result["consecutive_valid_return_run_length"].tolist() == [0, 1, 2, 0, 1]
    assert np.isnan(result["log_return_1h"].iloc[3])


def test_c_multi_hour_gap_breaks_run_length_once():
    # 10,11 present, 12,13 missing, 14,15 present.
    hours = [
        "2024-01-01T10:00:00Z",
        "2024-01-01T11:00:00Z",
        "2024-01-01T14:00:00Z",
        "2024-01-01T15:00:00Z",
    ]
    closes = [200.0, 201.0, 205.0, 206.0]
    df = make_df(hours, closes)

    result = compute_hourly_log_returns(df)

    assert result["log_return_available_1h"].tolist() == [0, 1, 0, 1]
    assert result["consecutive_valid_return_run_length"].tolist() == [0, 1, 0, 1]


def test_d_first_row_always_has_zero_run_length_no_error():
    hours = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
    closes = [100.0, 101.0]
    df = make_df(hours, closes)

    result = compute_hourly_log_returns(df)

    assert result["log_return_available_1h"].iloc[0] == 0
    assert result["consecutive_valid_return_run_length"].iloc[0] == 0
    assert np.isnan(result["log_return_1h"].iloc[0])


def test_e_gap_immediately_before_last_row_resets_run_length():
    hours = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T03:00:00Z"]
    closes = [100.0, 101.0, 103.0]
    df = make_df(hours, closes)

    result = compute_hourly_log_returns(df)

    assert result["log_return_available_1h"].tolist() == [0, 1, 0]
    assert result["consecutive_valid_return_run_length"].tolist() == [0, 1, 0]


# ---------------------------------------------------------------------------
# F-J: AR lag boundaries for k = 1, 3, 6, 24, 168 on gap-free series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 6, 24])
def test_fghi_boundary_for_small_k_on_gap_free_series(k):
    n = k + 5  # a little headroom past the boundary
    hours = make_consecutive_hours(n)
    closes = [100.0 + i for i in range(n)]
    df = compute_hourly_log_returns(make_df(hours, closes))

    mask = usable_mask_for_lag(df, k)

    # Row index k is the first row with k consecutive valid returns
    # ending at it (see module docstring worked example).
    expected = [False] * k + [True] * (n - k)
    assert mask.tolist() == expected


def test_j_k168_boundary_on_gap_free_series():
    n = 200
    hours = make_consecutive_hours(n)
    closes = [100.0 + 0.1 * i for i in range(n)]
    df = compute_hourly_log_returns(make_df(hours, closes))

    mask = usable_mask_for_lag(df, 168)

    expected = [False] * 168 + [True] * (n - 168)
    assert mask.tolist() == expected


def test_k_lag_larger_than_available_history_is_all_false_not_an_error():
    n = 50
    hours = make_consecutive_hours(n)
    closes = [100.0 + i for i in range(n)]
    df = compute_hourly_log_returns(make_df(hours, closes))

    mask = usable_mask_for_lag(df, 1000)

    assert mask.tolist() == [False] * n


def test_k_argument_must_be_positive_integer():
    hours = make_consecutive_hours(3)
    df = compute_hourly_log_returns(make_df(hours, [100.0, 101.0, 102.0]))

    with pytest.raises(ValueError, match="k must be >= 1"):
        usable_mask_for_lag(df, 0)

    with pytest.raises(TypeError, match="k must be a positive integer"):
        usable_mask_for_lag(df, 1.5)

    with pytest.raises(TypeError, match="k must be a positive integer"):
        usable_mask_for_lag(df, True)  # bool is an int subclass; reject explicitly


# ---------------------------------------------------------------------------
# L-Q: malformed-input handling (fail loudly, never repair)
# ---------------------------------------------------------------------------


def test_l_duplicate_timestamps_are_rejected():
    hours = ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
    closes = [100.0, 100.0, 101.0]
    df = make_df(hours, closes)

    with pytest.raises(ReturnsError, match="Duplicate timestamps"):
        compute_hourly_log_returns(df)


def test_m_non_monotonic_timestamps_are_rejected():
    hours = ["2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z"]
    closes = [101.0, 100.0, 102.0]
    df = make_df(hours, closes)

    with pytest.raises(ReturnsError, match="not strictly chronological"):
        compute_hourly_log_returns(df)


def test_n_timezone_naive_timestamps_are_rejected():
    naive = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"])
    df = pd.DataFrame(
        {
            "timestamp_utc": naive,
            "close": [100.0, 101.0],
            "target_status": ["valid", "valid"],
        }
    )

    with pytest.raises(ReturnsError, match="timezone-aware"):
        compute_hourly_log_returns(df)


def test_o_non_positive_close_is_rejected():
    hours = make_consecutive_hours(3)
    closes = [100.0, 0.0, 102.0]
    df = make_df(hours, closes)

    with pytest.raises(ReturnsError, match="non-positive"):
        compute_hourly_log_returns(df)


def test_o_non_finite_close_is_rejected():
    hours = make_consecutive_hours(3)
    closes = [100.0, float("inf"), 102.0]
    df = make_df(hours, closes)

    with pytest.raises(ReturnsError, match="non-finite"):
        compute_hourly_log_returns(df)


def test_p_missing_close_column_is_rejected():
    hours = make_consecutive_hours(3)
    df = pd.DataFrame({"timestamp_utc": _ts(hours), "target_status": ["valid"] * 3})

    with pytest.raises(ReturnsError, match="missing required column"):
        compute_hourly_log_returns(df)


def test_p_usable_mask_requires_returns_computed_first():
    hours = make_consecutive_hours(3)
    df = make_df(hours, [100.0, 101.0, 102.0])
    # df has NOT been passed through compute_hourly_log_returns yet.

    with pytest.raises(ReturnsError, match="missing required column"):
        usable_mask_for_lag(df, 1)


def test_q_missing_target_status_is_rejected():
    hours = make_consecutive_hours(3)
    df = compute_hourly_log_returns(make_df(hours, [100.0, 101.0, 102.0]))
    df.loc[df.index[1], "target_status"] = None

    with pytest.raises(ReturnsError, match="missing values"):
        usable_mask_for_lag(df, 1)


def test_q_unexpected_target_status_is_rejected():
    hours = make_consecutive_hours(3)
    df = compute_hourly_log_returns(make_df(hours, [100.0, 101.0, 102.0]))
    df.loc[df.index[1], "target_status"] = "bogus_status"

    with pytest.raises(ReturnsError, match="Unexpected target_status"):
        usable_mask_for_lag(df, 1)


# ---------------------------------------------------------------------------
# Purity / no in-place mutation
# ---------------------------------------------------------------------------


def test_compute_hourly_log_returns_does_not_mutate_input():
    hours = make_consecutive_hours(3)
    df = make_df(hours, [100.0, 101.0, 102.0])
    original_columns = list(df.columns)

    _ = compute_hourly_log_returns(df)

    assert list(df.columns) == original_columns


# ---------------------------------------------------------------------------
# Section 9: critical hand-derived indexing test
# ---------------------------------------------------------------------------


def test_critical_indexing_hand_derived_gap_at_04():
    """
    Dataset: 00,01,02,03,[04 missing],05,06,07

    Hand-derived log_return_available_1h:
        00: no predecessor            -> False
        01: 00->01 is 1h              -> True
        02: 01->02 is 1h              -> True
        03: 02->03 is 1h              -> True
        05: 03->05 is 2h (04 missing) -> False
        06: 05->06 is 1h              -> True
        07: 06->07 is 1h              -> True

    Hand-derived consecutive_valid_return_run_length:
        00: 0
        01: 1
        02: 2
        03: 3
        05: 0   (reset -- its own return is invalid)
        06: 1
        07: 2

    Hand-derived AR usability:
        AR(1) usable (run_length >= 1): 01,02,03,06,07
        AR(3) usable (run_length >= 3): 03 only
        AR(6) usable (run_length >= 6): none
            (max run_length anywhere in this window is 3, reached right
            before the gap; it is never rebuilt to 6 in only 7 rows)
    """
    hours = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T02:00:00Z",
        "2024-01-01T03:00:00Z",
        "2024-01-01T05:00:00Z",
        "2024-01-01T06:00:00Z",
        "2024-01-01T07:00:00Z",
    ]
    closes = [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0]
    df = compute_hourly_log_returns(make_df(hours, closes))

    assert df["log_return_available_1h"].tolist() == [0, 1, 1, 1, 0, 1, 1]
    assert df["consecutive_valid_return_run_length"].tolist() == [0, 1, 2, 3, 0, 1, 2]

    ar1 = usable_mask_for_lag(df, 1)
    ar3 = usable_mask_for_lag(df, 3)
    ar6 = usable_mask_for_lag(df, 6)

    assert ar1.tolist() == [False, True, True, True, False, True, True]
    assert ar3.tolist() == [False, False, False, True, False, False, False]
    assert ar6.tolist() == [False] * 7


def test_critical_indexing_multi_hour_gap():
    """
    Dataset: 10:00,11:00,[12,13 missing],14:00,15:00

    Hand-derived:
        log_return_available_1h: [False, True, False, True]
        run_length:               [0,     1,    0,     1]
        AR(2) usable: none (never reaches run_length >= 2)
    """
    hours = [
        "2024-01-01T10:00:00Z",
        "2024-01-01T11:00:00Z",
        "2024-01-01T14:00:00Z",
        "2024-01-01T15:00:00Z",
    ]
    closes = [200.0, 201.0, 205.0, 206.0]
    df = compute_hourly_log_returns(make_df(hours, closes))

    assert df["log_return_available_1h"].tolist() == [0, 1, 0, 1]
    assert df["consecutive_valid_return_run_length"].tolist() == [0, 1, 0, 1]

    ar2 = usable_mask_for_lag(df, 2)
    assert ar2.tolist() == [False, False, False, False]


# ---------------------------------------------------------------------------
# Section 10: regression tests around the real data's two known gaps
# ---------------------------------------------------------------------------


def test_real_gap_2023_03_04_does_not_create_fake_one_hour_return():
    """
    Regression test for the documented raw-data gap:
        2023-03-04T18:00Z, 19:00Z, 20:00Z missing.

    Only a small synthetic window straddling the real gap boundaries is
    used here (not the full ~43,800-row dataset) -- this proves the
    invariant at the exact real-world location of the gap without
    hardcoding a large expected dataframe.
    """
    hours = [
        "2023-03-04T16:00:00Z",
        "2023-03-04T17:00:00Z",
        # 18:00, 19:00, 20:00 are the real missing hours -- omitted here
        # exactly as they are omitted from the real raw CSV.
        "2023-03-04T21:00:00Z",
        "2023-03-04T22:00:00Z",
    ]
    closes = [22000.0, 22010.0, 21950.0, 21960.0]
    result = compute_hourly_log_returns(make_df(hours, closes))

    row_21 = result.loc[
        result["timestamp_utc"] == pd.Timestamp("2023-03-04T21:00:00Z")
    ].iloc[0]
    assert row_21["log_return_available_1h"] == 0
    assert row_21["consecutive_valid_return_run_length"] == 0
    assert np.isnan(row_21["log_return_1h"])

    row_22 = result.loc[
        result["timestamp_utc"] == pd.Timestamp("2023-03-04T22:00:00Z")
    ].iloc[0]
    assert row_22["log_return_available_1h"] == 1
    assert row_22["consecutive_valid_return_run_length"] == 1


def test_real_gap_2025_10_25_does_not_create_fake_one_hour_return():
    """
    Regression test for the documented raw-data gap:
        2025-10-25T16:00Z through 20:00Z missing (5 hours).
    """
    hours = [
        "2025-10-25T14:00:00Z",
        "2025-10-25T15:00:00Z",
        # 16:00-20:00 are the real missing hours -- omitted here exactly
        # as they are omitted from the real raw CSV.
        "2025-10-25T21:00:00Z",
        "2025-10-25T22:00:00Z",
    ]
    closes = [68000.0, 68050.0, 67900.0, 67950.0]
    result = compute_hourly_log_returns(make_df(hours, closes))

    row_21 = result.loc[
        result["timestamp_utc"] == pd.Timestamp("2025-10-25T21:00:00Z")
    ].iloc[0]
    assert row_21["log_return_available_1h"] == 0
    assert row_21["consecutive_valid_return_run_length"] == 0
    assert np.isnan(row_21["log_return_1h"])

    row_22 = result.loc[
        result["timestamp_utc"] == pd.Timestamp("2025-10-25T22:00:00Z")
    ].iloc[0]
    assert row_22["log_return_available_1h"] == 1
    assert row_22["consecutive_valid_return_run_length"] == 1
