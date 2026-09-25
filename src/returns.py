"""
Gap-aware hourly log-return and AR-lag-validity engine.

Purpose
-------
This module computes backward-looking hourly log returns and determines,
for a given AR lag order k, exactly which rows have a full, gap-free
trailing window of k consecutive hourly returns available.

This module intentionally does NOT fit any model. It is a pure data
transformation / validity-masking layer that a future AR fitting script
will consume.

Pipeline order
--------------
    raw OHLCV
        -> prepare_hourly_model_data.py  (adds target_* columns)
        -> compute_hourly_log_returns()  (adds log_return_* columns, here)
        -> usable_mask_for_lag(df, k)    (per-k boolean mask, here)
        -> [not implemented yet] AR fitting on df[usable_mask_for_lag(df, k)]

usable_mask_for_lag() requires its input to already carry the columns
produced by compute_hourly_log_returns(), plus the target_status column
produced by prepare_hourly_model_data.py. Call compute_hourly_log_returns()
first.

Return definition
------------------
For row t:

    log_return_1h[t] = log(close[t] / close[t-1])

but ONLY if row t-1 exists in the dataframe (by position) AND
    timestamp_utc[t] - timestamp_utc[t-1] == exactly 1 hour.

Otherwise log_return_1h[t] = NaN. This is a deliberate mathematical
statement that the 1-hour return at t is UNDEFINED -- not zero, not
interpolated, not forward/backward filled. The row itself is always kept.

The very first row in any input can never have a valid return (there is
no preceding row at all); this is expected, not an error.

Lag-window (AR) indexing semantics
-----------------------------------
For AR(k), an origin t is usable only if ALL of
    log_return_1h[t], log_return_1h[t-1], ..., log_return_1h[t-k+1]
are valid (non-NaN) -- i.e. k consecutive rows, all with valid 1-hour
returns, ending at t.

We track this with a single O(n) forward pass producing, for every row,
the length of the run of consecutive valid returns ending at that row:

    run_length[t] = 0                    if log_return_1h[t] is invalid
    run_length[t] = run_length[t-1] + 1  if log_return_1h[t] is valid

Then, for lag order k:

    usable_mask_for_lag(df, k)[t] = (run_length[t] >= k) AND (target_status[t] == "valid")

Worked example -- why "run_length >= k" is exactly right, not off by one:
    AR(k) at origin t needs k returns: r_t, r_{t-1}, ..., r_{t-k+1}.
    That is k consecutive valid-return rows ending at t, which is
    precisely what run_length[t] counts. No further +1/-1 adjustment
    is needed or correct.

Worked example -- boundary on a gap-free run starting at the very first
row of the dataframe (row index 0):
    row 0: no predecessor at all       -> run_length = 0
    row 1: 0 -> 1 is 1h                -> run_length = 1
    row i (i >= 1, still gap-free)     -> run_length = i
    => the first row usable for AR(k) is row index k (needs k+1 total
       rows: itself plus k gap-free predecessors).

Worked example -- a gap (see module tests for the full hand-derivation):
    00:00 01:00 02:00 03:00 [04:00 missing] 05:00 06:00 07:00
    run_length:  0    1     2     3           0     1     2
    AR(1) usable: 01,02,03,06,07
    AR(3) usable: 03 only
    AR(6) usable: none (max run_length in this window is 3)

None of this fills, interpolates, or silently drops rows. The original
dataframe row is always present; only the *validity flags* differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_RETURN_INPUT_COLUMNS: tuple[str, ...] = ("timestamp_utc", "close")
REQUIRED_MASK_COLUMNS: tuple[str, ...] = (
    "consecutive_valid_return_run_length",
    "target_status",
)
ALLOWED_TARGET_STATUSES: frozenset[str] = frozenset(
    {"valid", "missing_next_hour", "dataset_end"}
)


class ReturnsError(RuntimeError):
    """Raised when input data violates the return/lag-validity contract."""


def _validate_return_input(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_RETURN_INPUT_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ReturnsError(
            "Input is missing required column(s) for return computation: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ReturnsError("Input dataframe is empty.")

    timestamps = df["timestamp_utc"]

    if timestamps.isna().any():
        raise ReturnsError("timestamp_utc contains missing values.")

    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ReturnsError(
            "timestamp_utc must be timezone-aware. Received a "
            f"timezone-naive or non-datetime dtype: {timestamps.dtype}"
        )

    if str(timestamps.dtype.tz) != "UTC":
        raise ReturnsError(
            f"timestamp_utc must use UTC, found tz={timestamps.dtype.tz}"
        )

    if timestamps.duplicated().any():
        duplicates = (
            timestamps.loc[timestamps.duplicated(keep=False)].astype(str).tolist()
        )
        raise ReturnsError(
            "Duplicate timestamps detected; refusing to compute returns. "
            f"Examples: {duplicates[:5]}"
        )

    if not timestamps.is_monotonic_increasing:
        raise ReturnsError(
            "timestamp_utc is not strictly chronological. Refusing to "
            "compute returns on an unsorted or non-monotonic series."
        )

    close = pd.to_numeric(df["close"], errors="coerce")
    if close.isna().any():
        raise ReturnsError("close contains non-numeric or missing values.")

    if not np.isfinite(close.to_numpy(dtype=float)).all():
        raise ReturnsError("close contains non-finite values.")

    if (close <= 0).any():
        raise ReturnsError("close contains non-positive values.")


def compute_consecutive_valid_return_run_length(available: pd.Series) -> pd.Series:
    """
    Given a boolean/0-1 Series indicating whether a valid 1-hour log
    return exists ending at each row (in dataframe order), return, for
    every row, the number of consecutive rows immediately preceding and
    including it that all have valid 1-hour returns.

    This is a single explicit O(n) forward pass. pandas has no built-in
    cumulative operation that "resets the running count to zero whenever
    the current value is False" without an internal groupby trick that
    would obscure the logic; an explicit loop is used here for clarity
    and auditability, matching this project's stated priority of
    correctness/auditability over micro-optimized cleverness.
    """
    values = available.astype(bool).to_numpy()
    run_length = np.zeros(len(values), dtype=np.int64)

    running = 0
    for i, is_available in enumerate(values):
        running = running + 1 if is_available else 0
        run_length[i] = running

    return pd.Series(
        run_length,
        index=available.index,
        name="consecutive_valid_return_run_length",
    )


def compute_hourly_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gap-aware backward-looking hourly log returns to a copy of df.

    Adds three columns:
        log_return_1h                      float, NaN where undefined
        log_return_available_1h            int8, 1 if log_return_1h is
                                            a genuine 1-hour return, else 0
        consecutive_valid_return_run_length int64, see module docstring

    Requires df to contain "timestamp_utc" (timezone-aware UTC, strictly
    increasing, no duplicates) and "close" (finite, positive numeric).
    Does not require or touch any target_* columns, so it can be run on
    either the raw OHLCV frame or the full modeling frame produced by
    prepare_hourly_model_data.py.

    Never mutates the input dataframe.
    """
    _validate_return_input(df)

    result = df.copy()

    prev_timestamp = result["timestamp_utc"].shift(1)
    prev_close = result["close"].shift(1)

    # For the first row, prev_timestamp is NaT; NaT-involving comparisons
    # evaluate to False (never True), which is exactly the behavior we
    # want: the first row can never have a valid predecessor.
    delta = result["timestamp_utc"] - prev_timestamp
    contiguous_prev_hour = delta == pd.Timedelta(hours=1)

    close = result["close"].to_numpy(dtype=float)
    prev_close_values = prev_close.to_numpy(dtype=float)

    # Dividing by NaN (the first row's prev_close) is intentional and
    # produces the NaN we want; suppress the resulting numpy runtime
    # warning rather than letting it clutter test/CI output.
    with np.errstate(invalid="ignore", divide="ignore"):
        log_return = np.log(close / prev_close_values)

    result["log_return_1h"] = np.where(contiguous_prev_hour, log_return, np.nan)
    result["log_return_available_1h"] = contiguous_prev_hour.astype("int8")
    result["consecutive_valid_return_run_length"] = (
        compute_consecutive_valid_return_run_length(
            result["log_return_available_1h"]
        )
    )

    # Defensive invariant check: every row flagged as having a valid
    # consecutive-hour predecessor must have produced a finite return,
    # given that close was already validated as positive and finite for
    # every row. A violation here indicates a bug, not a data problem.
    flagged_valid = result["log_return_available_1h"].eq(1)
    if not np.isfinite(
        result.loc[flagged_valid, "log_return_1h"].to_numpy(dtype=float)
    ).all():
        raise ReturnsError(
            "Computed a non-finite log return for a row flagged as having "
            "a valid consecutive-hour predecessor. This indicates an "
            "internal inconsistency, not an expected data gap."
        )

    return result


def usable_mask_for_lag(df: pd.DataFrame, k: int) -> pd.Series:
    """
    Return a boolean Series: True at row t iff AR(k) can be evaluated at
    origin t, i.e. r_t, r_{t-1}, ..., r_{t-k+1} are all valid 1-hour
    returns AND the row's forward target is itself valid
    (target_status == "valid").

    Requires df to already carry the columns produced by
    compute_hourly_log_returns() (specifically
    "consecutive_valid_return_run_length") and the "target_status" column
    produced by prepare_hourly_model_data.py.

    k larger than the longest gap-free run in df is not an error: the
    result is simply all-False for that k (there is no usable origin),
    which is the correct, informative answer for a model-selection sweep.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(
            f"k must be a positive integer, got {k!r} ({type(k).__name__})."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    missing_columns = [
        column for column in REQUIRED_MASK_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ReturnsError(
            "Input is missing required column(s) for usable_mask_for_lag: "
            f"{missing_columns}. Call compute_hourly_log_returns(df) first "
            "to obtain 'consecutive_valid_return_run_length', and ensure "
            "df includes 'target_status' from prepare_hourly_model_data.py."
        )

    target_status = df["target_status"]

    if target_status.isna().any():
        raise ReturnsError("target_status contains missing values.")

    unexpected = set(target_status.unique()) - ALLOWED_TARGET_STATUSES
    if unexpected:
        raise ReturnsError(f"Unexpected target_status values: {sorted(unexpected)}")

    run_length = df["consecutive_valid_return_run_length"]

    mask = (run_length >= k) & target_status.eq("valid")
    mask.name = f"usable_for_ar_{k}"
    return mask
