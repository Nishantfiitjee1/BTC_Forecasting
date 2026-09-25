"""
Static AR(k) model for one-hour-ahead BTC/USD log-return forecasting.

This module contains no model selection and no test-set evaluation.
It provides:
- gap-aware AR design-matrix construction
- OLS fitting with optional intercept
- deterministic prediction
- auditable model metadata

Forecast origin t:
    features = [r_t, r_(t-1), ..., r_(t-k+1)]
    target   = r_(t+1)

where r is the gap-aware hourly log return computed by returns.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_baseline import (
    PERIODS,
    Period,
    load_and_validate,
)
from returns import ReturnsError, compute_hourly_log_returns, usable_mask_for_lag


CANDIDATE_LAGS: tuple[int, ...] = (1, 3, 6, 12, 24, 48, 168)
VALID_INTERCEPT_OPTIONS: tuple[bool, ...] = (False, True)


class ARModelError(RuntimeError):
    """Raised when AR model inputs or fitted model invariants are invalid."""


@dataclass(frozen=True)
class ARModel:
    lag: int
    include_intercept: bool
    coefficients: tuple[float, ...]
    intercept: float
    n_train: int
    train_origin_first: str
    train_origin_last: str
    fit_slice_sha256: str
    condition_number: float
    rank: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.lag:
            raise ARModelError(
                f"Expected X with shape (n, {self.lag}), got {X.shape}."
            )
        beta = np.asarray(self.coefficients, dtype=float)
        if beta.shape != (self.lag,):
            raise ARModelError("Stored coefficient count does not match lag.")
        prediction = X @ beta
        if self.include_intercept:
            prediction = prediction + self.intercept
        if not np.isfinite(prediction).all():
            raise ARModelError("Model produced non-finite predictions.")
        return prediction

    def to_dict(self) -> dict:
        return asdict(self)


def load_model_frame(path: Path) -> pd.DataFrame:
    """Load, re-check target invariants, then add gap-aware returns."""
    try:
        df = load_and_validate(path)
        valid = df["target_status"].eq("valid")
        if valid.any():
            expected = np.log(
                df.loc[valid, "target_close_1h"].to_numpy(dtype=float)
                / df.loc[valid, "close"].to_numpy(dtype=float)
            )
            actual = df.loc[valid, "target_return_1h"].to_numpy(dtype=float)
            if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                raise ARModelError(
                    "target_return_1h does not match log(target_close_1h / close) "
                    "on valid target rows."
                )

        unavailable = ~valid
        if (
            df.loc[unavailable, "target_close_1h"].notna().any()
            or df.loc[unavailable, "target_return_1h"].notna().any()
        ):
            raise ARModelError(
                "Unavailable target_status rows contain target values; refusing "
                "to build AR features from an inconsistent target frame."
            )

        return compute_hourly_log_returns(df)
    except ReturnsError as exc:
        raise ARModelError(f"Unable to prepare AR model frame: {exc}") from exc


def _origin_mask(df: pd.DataFrame, period: Period) -> pd.Series:
    return (
        (df["timestamp_utc"] >= period.start)
        & (df["timestamp_utc"] < period.end_exclusive)
    )


def build_ar_design_matrix(
    df: pd.DataFrame,
    lag: int,
    *,
    period: Period | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build X and y using the full frame before filtering to origins.

    This ordering is important: historical rows needed for lag features must
    remain available at period boundaries. Filtering first would incorrectly
    destroy the history needed by the first eligible validation/test origins.
    """
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
        raise ARModelError(f"lag must be a positive integer, got {lag!r}.")
    if lag < 1:
        raise ARModelError(f"lag must be >= 1, got {lag}.")

    try:
        eligible = usable_mask_for_lag(df, int(lag))
    except Exception as exc:
        raise ARModelError(f"Unable to determine AR({lag}) eligibility: {exc}") from exc

    returns = df["log_return_1h"].to_numpy(dtype=float)
    targets = df["target_return_1h"].to_numpy(dtype=float)

    positions = np.flatnonzero(eligible.to_numpy())
    if period is not None:
        origin_period_mask = _origin_mask(df, period).to_numpy()
        positions = positions[origin_period_mask[positions]]

    if len(positions) == 0:
        return (
            np.empty((0, lag), dtype=float),
            np.empty((0,), dtype=float),
            df.iloc[[]].copy(),
        )

    # Eligibility guarantees that every positional predecessor required here
    # exists and is a genuine consecutive-hour return. Positional indexing is
    # therefore safe only AFTER the gap-aware eligibility mask is applied.
    X = np.column_stack(
        [returns[positions - j] for j in range(lag)]
    )
    y = targets[positions]

    origins = df.iloc[positions].copy()

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ARModelError(
            "Gap-aware eligibility produced non-finite AR features/targets."
        )

    return X, y, origins


def _hash_fit_slice(origins: pd.DataFrame) -> str:
    columns = ["timestamp_utc", "close", "log_return_1h", "target_return_1h"]
    payload = origins[columns].copy()
    payload["timestamp_utc"] = payload["timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    data = payload.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fit_ar_model(
    df: pd.DataFrame,
    lag: int,
    *,
    include_intercept: bool,
    period: Period,
) -> tuple[ARModel, dict]:
    """Fit one static AR(k) specification on exactly one origin period."""
    X, y, origins = build_ar_design_matrix(df, lag, period=period)

    if len(y) == 0:
        raise ARModelError(
            f"AR({lag}, intercept={include_intercept}) has no usable "
            f"training observations in {period.name}."
        )

    design = (
        np.column_stack([np.ones(len(X)), X])
        if include_intercept
        else X
    )

    rank = int(np.linalg.matrix_rank(design))
    n_parameters = design.shape[1]
    if rank < n_parameters:
        raise ARModelError(
            f"AR({lag}, intercept={include_intercept}) design matrix is "
            f"rank deficient: rank={rank}, parameters={n_parameters}."
        )

    condition_number = float(np.linalg.cond(design))
    if not np.isfinite(condition_number):
        raise ARModelError(
            f"AR({lag}, intercept={include_intercept}) design matrix has "
            "a non-finite condition number."
        )

    coefficients_all, residuals, _, _ = np.linalg.lstsq(
        design, y, rcond=None
    )

    if include_intercept:
        intercept = float(coefficients_all[0])
        coefficients = tuple(float(v) for v in coefficients_all[1:])
    else:
        intercept = 0.0
        coefficients = tuple(float(v) for v in coefficients_all)

    model = ARModel(
        lag=int(lag),
        include_intercept=bool(include_intercept),
        coefficients=coefficients,
        intercept=intercept,
        n_train=int(len(y)),
        train_origin_first=origins["timestamp_utc"].min().isoformat(),
        train_origin_last=origins["timestamp_utc"].max().isoformat(),
        fit_slice_sha256=_hash_fit_slice(origins),
        condition_number=condition_number,
        rank=rank,
    )

    fitted = model.predict(X)
    residual = y - fitted
    fit_metrics = {
        "n_train": int(len(y)),
        "mae_return": float(np.mean(np.abs(residual))),
        "rmse_return": float(np.sqrt(np.mean(residual**2))),
        "condition_number": condition_number,
        "rank": rank,
        "n_parameters": n_parameters,
    }

    return model, fit_metrics



def _validate_frozen_model_provenance(
    df: pd.DataFrame,
    model: ARModel,
) -> None:
    """Validate that a frozen AR model matches the current development fit slice.

    This is a read-only provenance check. It does not use validation/test
    observations as training data and does not mutate the model.
    """
    if model.lag < 1:
        raise ARModelError(f"Frozen model has invalid lag={model.lag}.")
    if len(model.coefficients) != model.lag:
        raise ARModelError("Frozen model coefficient count does not match lag.")
    numeric = np.asarray(
        [*model.coefficients, model.intercept, model.condition_number],
        dtype=float,
    )
    if not np.isfinite(numeric).all():
        raise ARModelError("Frozen model contains non-finite parameters/metadata.")
    if model.condition_number <= 0.0:
        raise ARModelError("Frozen model condition_number must be positive.")
    if model.rank < 1 or model.rank > (model.lag + int(model.include_intercept)):
        raise ARModelError("Frozen model rank is outside valid design dimensions.")

    try:
        first = pd.Timestamp(model.train_origin_first)
        last = pd.Timestamp(model.train_origin_last)
    except Exception as exc:
        raise ARModelError("Frozen model training timestamps are invalid.") from exc

    if first.tzinfo is None or last.tzinfo is None:
        raise ARModelError("Frozen model training timestamps must be timezone-aware.")
    if first >= last:
        raise ARModelError("Frozen model training origin range is invalid.")

    eligible = usable_mask_for_lag(df, model.lag)
    ts = df["timestamp_utc"]
    mask = (
        eligible
        & (ts >= first)
        & (ts <= last)
    )
    origins = df.loc[mask].copy()

    if len(origins) != model.n_train:
        raise ARModelError(
            "Frozen AR provenance mismatch: training row count differs "
            f"(lock={model.n_train}, current={len(origins)})."
        )
    if origins.empty:
        raise ARModelError("Frozen AR provenance mismatch: training slice is empty.")

    current_first = origins["timestamp_utc"].min()
    current_last = origins["timestamp_utc"].max()
    if current_first != first or current_last != last:
        raise ARModelError(
            "Frozen AR provenance mismatch: training origin timestamps differ."
        )

    current_hash = _hash_fit_slice(origins)
    if current_hash != model.fit_slice_sha256:
        raise ARModelError(
            "Frozen AR provenance mismatch: fit_slice_sha256 differs from current data."
        )

    X, y, _ = build_ar_design_matrix(
        df,
        model.lag,
        period=Period(
            name="frozen_training_slice",
            start=first,
            end_exclusive=last + pd.Timedelta(hours=1),
        ),
    )
    design = (
        np.column_stack([np.ones(len(X)), X])
        if model.include_intercept
        else X
    )
    rank = int(np.linalg.matrix_rank(design))
    condition_number = float(np.linalg.cond(design))
    if rank != model.rank or not np.isclose(
        condition_number, model.condition_number, rtol=1e-10, atol=1e-10
    ):
        raise ARModelError(
            "Frozen AR provenance mismatch: design rank/condition number differs."
        )

    coefficients_all, *_ = np.linalg.lstsq(design, y, rcond=None)
    if model.include_intercept:
        current_intercept = float(coefficients_all[0])
        current_coefficients = np.asarray(coefficients_all[1:], dtype=float)
    else:
        current_intercept = 0.0
        current_coefficients = np.asarray(coefficients_all, dtype=float)

    if not np.allclose(
        current_coefficients,
        np.asarray(model.coefficients, dtype=float),
        rtol=1e-12,
        atol=1e-14,
    ) or not np.isclose(
        current_intercept,
        model.intercept,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ARModelError(
            "Frozen AR provenance mismatch: stored coefficients do not match "
            "a deterministic refit of the current training slice."
        )

def predict_period(
    df: pd.DataFrame,
    model: ARModel,
    period: Period,
) -> pd.DataFrame:
    """Generate one-hour-ahead forecasts for a period using a frozen model."""
    _validate_frozen_model_provenance(df, model)
    X, y, origins = build_ar_design_matrix(df, model.lag, period=period)
    if len(y) == 0:
        raise ARModelError(
            f"No usable AR({model.lag}) origins in {period.name}."
        )

    predicted_return = model.predict(X)
    forecast_origin_close = origins["close"].to_numpy(dtype=float)
    actual_close = origins["target_close_1h"].to_numpy(dtype=float)

    predicted_close = forecast_origin_close * np.exp(predicted_return)

    out = pd.DataFrame(
        {
            "timestamp_utc": origins["timestamp_utc"].to_numpy(),
            "forecast_origin_close": forecast_origin_close,
            "predicted_return_1h": predicted_return,
            "predicted_close_1h": predicted_close,
            "actual_return_1h": y,
            "actual_close_1h": actual_close,
        }
    )

    if not np.isfinite(out.drop(columns=["timestamp_utc"]).to_numpy()).all():
        raise ARModelError("AR prediction output contains non-finite values.")

    return out


def evaluate_forecasts(predictions: pd.DataFrame) -> dict:
    """Return close/return error metrics for an AR forecast dataframe."""
    required = {
        "predicted_close_1h",
        "actual_close_1h",
        "predicted_return_1h",
        "actual_return_1h",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ARModelError(f"Forecast dataframe missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ARModelError("Cannot evaluate an empty forecast dataframe.")

    close_error = (
        predictions["predicted_close_1h"] - predictions["actual_close_1h"]
    )
    return_error = (
        predictions["predicted_return_1h"] - predictions["actual_return_1h"]
    )

    actual_direction = np.sign(predictions["actual_return_1h"].to_numpy())
    predicted_direction = np.sign(predictions["predicted_return_1h"].to_numpy())
    nonzero_actual = actual_direction != 0

    metrics = {
        "n_predictions": int(len(predictions)),
        "mae_close": float(np.mean(np.abs(close_error))),
        "rmse_close": float(np.sqrt(np.mean(close_error**2))),
        "median_absolute_error_close": float(np.median(np.abs(close_error))),
        "mean_error_close": float(np.mean(close_error)),
        "rmse_return": float(np.sqrt(np.mean(return_error**2))),
        "mae_return": float(np.mean(np.abs(return_error))),
        "mean_absolute_actual_return": float(
            np.mean(np.abs(predictions["actual_return_1h"]))
        ),
        "zero_actual_return_count": int((~nonzero_actual).sum()),
        "directional_accuracy": (
            float(np.mean(
                predicted_direction[nonzero_actual]
                == actual_direction[nonzero_actual]
            ))
            if nonzero_actual.any()
            else None
        ),
    }
    return metrics
