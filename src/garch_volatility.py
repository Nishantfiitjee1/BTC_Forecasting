"""Leakage-controlled GARCH(1,1) volatility model for hourly BTC/USD returns.

This is Experiment 2a. It forecasts the *next-hour conditional variance* from
returns available at the forecast origin. It does not replace the point-forecast
mean model. Its first purpose is to support uncertainty/range forecasting.

Model:
    r_t = mu + eps_t
    eps_t = sigma_t z_t
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

Returns are scaled by 100 during optimization for numerical stability.
The fitted parameters are transformed back to return units for reporting.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest_baseline import Period, load_and_validate
from returns import ReturnsError, compute_hourly_log_returns


class GARCHModelError(RuntimeError):
    """Raised when GARCH inputs, fitting, or forecasting invariants fail."""


@dataclass(frozen=True)
class GARCH11Model:
    mu: float
    omega: float
    alpha: float
    beta: float
    unconditional_variance: float
    n_train: int
    train_origin_first: str
    train_origin_last: str
    fit_slice_sha256: str
    loglikelihood: float
    converged: bool
    optimizer_iterations: int

    def forecast_variance(self, last_return: float, last_variance: float) -> float:
        if not np.isfinite(last_return) or not np.isfinite(last_variance):
            raise GARCHModelError("GARCH forecast inputs must be finite.")
        if last_variance <= 0:
            raise GARCHModelError("Last conditional variance must be positive.")
        eps = last_return - self.mu
        variance = self.omega + self.alpha * eps * eps + self.beta * last_variance
        if not np.isfinite(variance) or variance <= 0:
            raise GARCHModelError("GARCH produced a non-positive/non-finite variance.")
        return float(variance)

    def to_dict(self) -> dict:
        return asdict(self)


def load_garch_frame(path: Path) -> pd.DataFrame:
    """Load production model data and add the gap-aware hourly return."""
    try:
        df = load_and_validate(path)
        return compute_hourly_log_returns(df)
    except ReturnsError as exc:
        raise GARCHModelError(f"Unable to prepare GARCH frame: {exc}") from exc


def _period_mask(df: pd.DataFrame, period: Period) -> pd.Series:
    return (
        (df["timestamp_utc"] >= period.start)
        & (df["timestamp_utc"] < period.end_exclusive)
        & df["log_return_available_1h"].eq(1)
    )


def _fit_slice(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    mask = _period_mask(df, period)
    out = df.loc[mask, ["timestamp_utc", "log_return_1h"]].copy()
    if out.empty:
        raise GARCHModelError(f"No valid hourly returns in {period.name}.")
    values = out["log_return_1h"].to_numpy(float)
    if not np.isfinite(values).all():
        raise GARCHModelError("Training returns contain non-finite values.")
    return out


def _hash_slice(frame: pd.DataFrame) -> str:
    x = frame.copy()
    x["timestamp_utc"] = x["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return hashlib.sha256(x.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _unpack(theta: np.ndarray, sample_var: float) -> tuple[float, float, float, float]:
    """Map unconstrained parameters to valid GARCH parameters."""
    mu = float(theta[0])
    # Positive omega; alpha and beta are positive and constrained to alpha+beta<1.
    omega = float(np.exp(np.clip(theta[1], -40.0, 20.0)))
    ea = float(np.exp(np.clip(theta[2], -40.0, 40.0)))
    eb = float(np.exp(np.clip(theta[3], -40.0, 40.0)))
    denom = 1.0 + ea + eb
    alpha = 0.999 * ea / denom
    beta = 0.999 * eb / denom
    # omega scale is initialized/optimized in squared percent-return units.
    if not np.isfinite(sample_var):
        raise GARCHModelError("Sample variance is not finite.")
    return mu, omega, alpha, beta


def _conditional_variances(params: tuple[float, float, float, float], returns_scaled: np.ndarray) -> np.ndarray:
    mu, omega, alpha, beta = params
    n = len(returns_scaled)
    variance = np.empty(n, dtype=float)
    unconditional = omega / max(1e-12, 1.0 - alpha - beta)
    if not np.isfinite(unconditional) or unconditional <= 0:
        raise GARCHModelError("Invalid unconditional variance during fitting.")
    variance[0] = unconditional
    eps_prev = returns_scaled[0] - mu
    for i in range(1, n):
        variance[i] = omega + alpha * eps_prev * eps_prev + beta * variance[i - 1]
        if not np.isfinite(variance[i]) or variance[i] <= 0:
            raise GARCHModelError("Non-finite conditional variance during fitting.")
        eps_prev = returns_scaled[i] - mu
    return variance


def _negative_loglik(theta: np.ndarray, returns_scaled: np.ndarray) -> float:
    try:
        sample_var = float(np.var(returns_scaled))
        params = _unpack(theta, sample_var)
        mu, _, _, _ = params
        variance = _conditional_variances(params, returns_scaled)
        eps = returns_scaled - mu
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + (eps * eps) / variance)
        value = float(-ll)
        return value if np.isfinite(value) else 1e100
    except (FloatingPointError, OverflowError, GARCHModelError, ValueError):
        return 1e100


def fit_garch11(df: pd.DataFrame, period: Period) -> tuple[GARCH11Model, dict]:
    """Fit one static GARCH(1,1) model on exactly one period."""
    train = _fit_slice(df, period)
    raw = train["log_return_1h"].to_numpy(float)
    # Percentage-return scaling makes optimizer magnitudes reasonable.
    y = raw * 100.0
    sample_var = max(float(np.var(y)), 1e-10)
    sample_mean = float(np.mean(y))

    # Deterministic starting points. Two starts reduce sensitivity to the
    # initial persistence guess while keeping the experiment reproducible.
    starts = []
    for alpha0, beta0 in ((0.05, 0.90), (0.10, 0.85)):
        omega0 = max(sample_var * (1.0 - alpha0 - beta0), 1e-8)
        starts.append(np.array([
            sample_mean,
            np.log(omega0),
            np.log(alpha0 / max(1e-9, 1.0 - alpha0 - beta0)),
            np.log(beta0 / max(1e-9, 1.0 - alpha0 - beta0)),
        ], dtype=float))

    best = None
    for start in starts:
        result = minimize(
            _negative_loglik,
            start,
            args=(y,),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
        if not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise GARCHModelError("GARCH optimizer failed to produce a finite solution.")

    params_scaled = _unpack(best.x, sample_var)
    mu_s, omega_s, alpha, beta = params_scaled
    if not best.success:
        # Preserve the existing numerical protocol: a finite, valid solution
        # is accepted even when L-BFGS-B reports a line-search warning.
        # This is intentionally not changed here because altering the
        # optimizer acceptance rule would change the locked model itself.
        if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
            raise GARCHModelError(f"GARCH optimizer did not converge: {best.message}")

    if not (
        np.isfinite(mu_s)
        and np.isfinite(omega_s)
        and np.isfinite(alpha)
        and np.isfinite(beta)
        and omega_s > 0
        and 0 < alpha
        and 0 < beta
        and alpha + beta < 1
    ):
        raise GARCHModelError("Fitted GARCH parameters violate positivity/stationarity invariants.")

    # Convert variance parameters from percent-return^2 back to return^2.
    scale = 100.0
    mu = mu_s / scale
    omega = omega_s / (scale * scale)
    unconditional_variance = omega / (1.0 - alpha - beta)

    model = GARCH11Model(
        mu=float(mu),
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        unconditional_variance=float(unconditional_variance),
        n_train=int(len(train)),
        train_origin_first=train["timestamp_utc"].min().isoformat(),
        train_origin_last=train["timestamp_utc"].max().isoformat(),
        fit_slice_sha256=_hash_slice(train),
        loglikelihood=float(-best.fun),
        converged=bool(best.success),
        optimizer_iterations=int(getattr(best, "nit", 0)),
    )

    variance_scaled = _conditional_variances(params_scaled, y)
    fitted_sigma = np.sqrt(variance_scaled) / scale
    standardized = (raw - model.mu) / fitted_sigma
    metrics = {
        "n_train": int(len(train)),
        "mu": model.mu,
        "omega": model.omega,
        "alpha": model.alpha,
        "beta": model.beta,
        "alpha_plus_beta": model.alpha + model.beta,
        "unconditional_volatility": float(np.sqrt(model.unconditional_variance)),
        "mean_abs_standardized_residual": float(np.mean(np.abs(standardized))),
        "loglikelihood_scaled_returns": model.loglikelihood,
        "converged": model.converged,
        "optimizer_iterations": model.optimizer_iterations,
    }
    return model, metrics


def filter_contiguous_origins(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Return origins in a period whose current return is valid and whose next target is valid."""
    mask = _period_mask(df, period) & df["target_status"].eq("valid")
    out = df.loc[mask].copy()
    if out.empty:
        raise GARCHModelError(f"No valid forecast origins in {period.name}.")
    return out


def _validate_model_provenance(df: pd.DataFrame, model: GARCH11Model) -> int:
    """Validate that the fitted model belongs to this exact training slice.

    The forecast state is only meaningful if the model metadata and the
    currently loaded frame agree exactly. This prevents accidentally applying
    a model fitted on a different dataset/version to the current data.
    """
    first_ts = pd.Timestamp(model.train_origin_first)
    last_ts = pd.Timestamp(model.train_origin_last)

    if first_ts.tzinfo is None or last_ts.tzinfo is None:
        raise GARCHModelError("GARCH training-origin metadata must be timezone-aware.")

    matches = df["timestamp_utc"].eq(first_ts)
    if not matches.any():
        raise GARCHModelError(
            "GARCH training-origin provenance does not match input frame."
        )

    first_idx = int(np.flatnonzero(matches.to_numpy())[0])

    train_period = Period(
        name="model_provenance",
        start=first_ts,
        end_exclusive=last_ts + pd.Timedelta(hours=1),
    )
    fitted_slice = _fit_slice(df, train_period)

    if len(fitted_slice) != model.n_train:
        raise GARCHModelError(
            "GARCH training row count does not match model provenance."
        )

    actual_last = fitted_slice["timestamp_utc"].iloc[-1]
    if actual_last != last_ts:
        raise GARCHModelError(
            "GARCH training end timestamp does not match model provenance."
        )

    actual_hash = _hash_slice(fitted_slice)
    if actual_hash != model.fit_slice_sha256:
        raise GARCHModelError(
            "GARCH fit-slice SHA-256 does not match model provenance."
        )

    return first_idx


def forecast_one_step_variance(
    df: pd.DataFrame,
    model: GARCH11Model,
    period: Period,
) -> pd.DataFrame:
    """Generate one-step conditional variance forecasts using only information through each origin.

    The recursion is explicitly bounded by ``period.end_exclusive``. Therefore
    a validation forecast cannot process later test observations, even
    internally. This is a structural anti-leakage guarantee rather than a
    reliance on filtering the output afterward.
    """
    filter_contiguous_origins(df, period)  # fail fast if the requested period has no origins
    first_train_idx = _validate_model_provenance(df, model)

    returns = df["log_return_1h"].to_numpy(dtype=float)
    available = df["log_return_available_1h"].to_numpy(dtype=bool)
    timestamps = df["timestamp_utc"].to_numpy()
    target_valid = df["target_status"].eq("valid").to_numpy(dtype=bool)
    closes = df["close"].to_numpy(dtype=float)
    target_returns = df["target_return_1h"].to_numpy(dtype=float)
    target_closes = df["target_close_1h"].to_numpy(dtype=float)

    # The first row processed is the first observed training return. The
    # variance before that return is the model's unconditional variance.
    # Updating with r_i then produces the one-step variance for i+1.
    variance = model.unconditional_variance
    rows: list[tuple[int, float]] = []

    # Find the first row at or after the requested period end. No row at or
    # beyond that boundary is processed, eliminating future-data traversal.
    end_positions = np.flatnonzero(
        df["timestamp_utc"].ge(period.end_exclusive).to_numpy()
    )
    stop_idx = (
        int(end_positions[0]) if len(end_positions) else len(df)
    )

    for i in range(first_train_idx, stop_idx):
        if not available[i]:
            # A missing hourly return breaks the GARCH state. Do not bridge
            # across the gap with a multi-hour price change.
            variance = model.unconditional_variance
            continue

        r = returns[i]
        next_variance = model.forecast_variance(r, variance)

        ts = timestamps[i]
        if target_valid[i] and period.start <= ts < period.end_exclusive:
            rows.append((i, next_variance))

        variance = next_variance

    if not rows:
        raise GARCHModelError(f"No GARCH forecasts produced for {period.name}.")

    idx = np.fromiter((x[0] for x in rows), dtype=np.int64)
    var = np.fromiter((x[1] for x in rows), dtype=float)

    out = pd.DataFrame(
        {
            "timestamp_utc": timestamps[idx],
            "forecast_origin_close": closes[idx],
            "predicted_variance_1h": var,
            "predicted_volatility_1h": np.sqrt(var),
            "actual_return_1h": target_returns[idx],
            "actual_close_1h": target_closes[idx],
        }
    )

    if out["timestamp_utc"].duplicated().any():
        raise GARCHModelError("GARCH forecast output contains duplicate timestamps.")
    if not out["timestamp_utc"].is_monotonic_increasing:
        raise GARCHModelError("GARCH forecast timestamps are not increasing.")

    numeric = out.drop(columns=["timestamp_utc"]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise GARCHModelError("GARCH forecast output contains non-finite values.")
    if (out["predicted_variance_1h"] <= 0).any():
        raise GARCHModelError("GARCH forecast output contains non-positive variance.")

    return out


def evaluate_variance_forecasts(predictions: pd.DataFrame) -> dict:
    required = {"predicted_variance_1h", "actual_return_1h"}
    missing = required.difference(predictions.columns)
    if missing:
        raise GARCHModelError(f"Variance forecast missing columns: {sorted(missing)}")
    if predictions.empty:
        raise GARCHModelError("Cannot evaluate empty variance forecasts.")
    v = predictions["predicted_variance_1h"].to_numpy(float)
    r2 = predictions["actual_return_1h"].to_numpy(float) ** 2
    if (v <= 0).any() or not np.isfinite(v).all():
        raise GARCHModelError("Predicted variance must be positive and finite.")
    if not np.isfinite(r2).all():
        raise GARCHModelError("Actual returns contain non-finite values.")
    # QLIKE is a standard loss for variance forecasts and remains meaningful
    # when realized squared returns are extremely small.
    qlike = float(np.mean(r2 / v - np.log(r2 / v + 1e-300) - 1.0))
    mse = float(np.mean((r2 - v) ** 2))
    return {"n_predictions": int(len(predictions)), "qlike": qlike, "mse_squared_return": mse}
