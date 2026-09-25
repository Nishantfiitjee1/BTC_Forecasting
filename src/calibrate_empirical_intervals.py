from __future__ import annotations

"""
Validation-only empirical innovation calibration for the locked AR(3)+GARCH(1,1)
forecast distribution.

Protocol
--------
1. Load the existing 2024 validation prediction artifact.
2. Keep the AR/GARCH forecasts completely frozen.
3. Split 2024 validation chronologically:
       calibration = first 70%
       evaluation  = final 30%
4. Estimate empirical standardized-return quantiles ONLY on the calibration
   segment.
5. Apply those fixed quantiles to the untouched validation evaluation segment.
6. Compare empirical intervals with the original Gaussian intervals using
   interval score and empirical coverage.
7. Do NOT read any 2025 test observations.

This is a calibration experiment, not a model refit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "backtests" / "gaussian_calibration" / (
    "validation_calibration_predictions.csv"
)
OUT = ROOT / "results" / "backtests" / "empirical_calibration"
OUT.mkdir(parents=True, exist_ok=True)

CALIBRATION_FRACTION = 0.70
CONFIDENCE_LEVELS = (0.50, 0.80, 0.90, 0.95)


class CalibrationError(RuntimeError):
    pass


def load_validation_predictions() -> pd.DataFrame:
    if not INPUT.exists():
        raise CalibrationError(f"Missing validation artifact: {INPUT}")

    df = pd.read_csv(INPUT, parse_dates=["timestamp_utc"])
    if "timestamp_utc" not in df.columns:
        raise CalibrationError("Missing timestamp_utc column.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    required = {
        "timestamp_utc",
        "actual_return_1h",
        "forecast_origin_close",
        "predicted_return_1h",
        "predicted_volatility_1h",
        "standardized_residual",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise CalibrationError(f"Missing required columns: {missing}")

    if df.empty:
        raise CalibrationError("Validation artifact is empty.")
    if df["timestamp_utc"].duplicated().any():
        raise CalibrationError("Validation artifact contains duplicate timestamps.")
    if not df["timestamp_utc"].is_monotonic_increasing:
        raise CalibrationError("Validation timestamps are not strictly increasing.")

    numeric = [
        "actual_return_1h",
        "forecast_origin_close",
        "predicted_return_1h",
        "predicted_volatility_1h",
        "standardized_residual",
    ]
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise CalibrationError(f"{col} contains non-finite values.")

    sigma = df["predicted_volatility_1h"].to_numpy(float)
    if (sigma <= 0).any():
        raise CalibrationError("predicted_volatility_1h must be strictly positive.")

    # Reconstruct the standardized residual independently rather than trusting
    # the stored diagnostic column.
    reconstructed = (
        df["actual_return_1h"].to_numpy(float)
        - df["predicted_return_1h"].to_numpy(float)
    ) / sigma
    stored = df["standardized_residual"].to_numpy(float)

    if not np.allclose(reconstructed, stored, rtol=1e-10, atol=1e-12):
        max_diff = float(np.max(np.abs(reconstructed - stored)))
        raise CalibrationError(
            "Stored standardized residuals do not match "
            f"(actual - predicted) / sigma; max diff={max_diff:.6g}"
        )

    return df.reset_index(drop=True)


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    confidence: float,
) -> float:
    alpha = 1.0 - confidence
    penalty_scale = 2.0 / alpha

    score = (
        upper
        - lower
        + penalty_scale * np.maximum(lower - y, 0.0)
        + penalty_scale * np.maximum(y - upper, 0.0)
    )
    return float(np.mean(score))


def evaluate_intervals(
    df: pd.DataFrame,
    interval_prefix: str,
    quantiles: dict[float, tuple[float, float]],
) -> tuple[pd.DataFrame, dict]:
    y = df["actual_return_1h"].to_numpy(float)
    mu = df["predicted_return_1h"].to_numpy(float)
    sigma = df["predicted_volatility_1h"].to_numpy(float)

    output = df[["timestamp_utc"]].copy()
    metrics = {}

    for confidence in CONFIDENCE_LEVELS:
        lo_q, hi_q = quantiles[confidence]
        lower = mu + lo_q * sigma
        upper = mu + hi_q * sigma

        label = int(round(confidence * 100))
        output[f"{interval_prefix}_{label}_lower_return"] = lower
        output[f"{interval_prefix}_{label}_upper_return"] = upper

        covered = (y >= lower) & (y <= upper)
        coverage = float(covered.mean())

        metrics[str(label)] = {
            "confidence": confidence,
            "nominal_coverage": confidence,
            "actual_coverage": coverage,
            "covered_count": int(covered.sum()),
            "n": int(len(y)),
            "coverage_error": coverage - confidence,
            "mean_interval_score": interval_score(
                y, lower, upper, confidence
            ),
            "mean_return_width": float(np.mean(upper - lower)),
        }

    return output, metrics


def gaussian_quantiles() -> dict[float, tuple[float, float]]:
    from scipy.stats import norm

    return {
        confidence: (
            float(norm.ppf((1.0 - confidence) / 2.0)),
            float(norm.ppf((1.0 + confidence) / 2.0)),
        )
        for confidence in CONFIDENCE_LEVELS
    }


def empirical_quantiles(z_cal: np.ndarray) -> dict[float, tuple[float, float]]:
    if len(z_cal) < 100:
        raise CalibrationError(
            f"Too few calibration observations ({len(z_cal)}); need at least 100."
        )

    if not np.isfinite(z_cal).all():
        raise CalibrationError("Calibration residuals contain non-finite values.")

    result = {}
    for confidence in CONFIDENCE_LEVELS:
        lower_p = (1.0 - confidence) / 2.0
        upper_p = (1.0 + confidence) / 2.0

        lo = float(np.quantile(z_cal, lower_p, method="linear"))
        hi = float(np.quantile(z_cal, upper_p, method="linear"))

        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise CalibrationError(
                f"Invalid empirical quantiles for {confidence:.0%}: {lo}, {hi}"
            )

        result[confidence] = (lo, hi)

    return result


def main() -> None:
    df = load_validation_predictions()

    split = int(np.floor(len(df) * CALIBRATION_FRACTION))
    if split <= 0 or split >= len(df):
        raise CalibrationError("Invalid chronological calibration split.")

    calibration = df.iloc[:split].copy()
    evaluation = df.iloc[split:].copy()

    # Hard chronology checks.
    if calibration["timestamp_utc"].max() >= evaluation["timestamp_utc"].min():
        raise CalibrationError("Calibration/evaluation windows overlap.")

    # Ensure this experiment is confined to the 2024 validation artifact.
    val_start = pd.Timestamp("2024-01-01", tz="UTC")
    val_end = pd.Timestamp("2025-01-01", tz="UTC")
    if df["timestamp_utc"].min() < val_start or df["timestamp_utc"].max() >= val_end:
        raise CalibrationError(
            "Input contains timestamps outside the locked 2024 validation period."
        )

    z_cal = calibration["standardized_residual"].to_numpy(float)

    empirical_q = empirical_quantiles(z_cal)
    gaussian_q = gaussian_quantiles()

    empirical_output, empirical_metrics = evaluate_intervals(
        evaluation, "empirical", empirical_q
    )
    gaussian_output, gaussian_metrics = evaluate_intervals(
        evaluation, "gaussian", gaussian_q
    )

    output = evaluation[
        [
            "timestamp_utc",
            "forecast_origin_close",
            "predicted_return_1h",
            "predicted_volatility_1h",
            "actual_return_1h",
        ]
    ].copy()

    for col in empirical_output.columns:
        if col != "timestamp_utc":
            output[col] = empirical_output[col].to_numpy()

    for col in gaussian_output.columns:
        if col != "timestamp_utc":
            output[col] = gaussian_output[col].to_numpy()

    output_path = OUT / "validation_holdout_predictions.csv"
    output.to_csv(output_path, index=False, float_format="%.17g")

    comparison = {}
    for level in CONFIDENCE_LEVELS:
        key = str(int(round(level * 100)))
        e = empirical_metrics[key]
        g = gaussian_metrics[key]
        comparison[key] = {
            "nominal_coverage": level,
            "empirical": e,
            "gaussian": g,
            "empirical_minus_gaussian_interval_score": (
                e["mean_interval_score"] - g["mean_interval_score"]
            ),
            "empirical_minus_gaussian_abs_coverage_error": (
                abs(e["coverage_error"]) - abs(g["coverage_error"])
            ),
        }

    summary = {
        "protocol": {
            "input": str(INPUT),
            "validation_period_start": str(val_start),
            "validation_period_end_exclusive": str(val_end),
            "calibration_fraction": CALIBRATION_FRACTION,
            "calibration_n": len(calibration),
            "evaluation_n": len(evaluation),
            "calibration_start": str(calibration["timestamp_utc"].iloc[0]),
            "calibration_end": str(calibration["timestamp_utc"].iloc[-1]),
            "evaluation_start": str(evaluation["timestamp_utc"].iloc[0]),
            "evaluation_end": str(evaluation["timestamp_utc"].iloc[-1]),
            "test_period_accessed": False,
            "ar_or_garch_refit": False,
        },
        "empirical_standardized_residual_quantiles": {
            str(int(round(c * 100))): {
                "lower": empirical_q[c][0],
                "upper": empirical_q[c][1],
            }
            for c in CONFIDENCE_LEVELS
        },
        "comparison": comparison,
        "selection_rule": (
            "Empirical calibration is retained only if it improves mean "
            "interval score on the chronological validation holdout at the "
            "target confidence levels without materially worsening coverage. "
            "The 2025 test period is never used for this decision."
        ),
    }

    summary_path = OUT / "validation_calibration_comparison.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Input rows:          {len(df)}")
    print(f"Calibration rows:    {len(calibration)}")
    print(f"Holdout rows:        {len(evaluation)}")
    print(f"Calibration end:     {calibration['timestamp_utc'].iloc[-1]}")
    print(f"Holdout start:       {evaluation['timestamp_utc'].iloc[0]}")
    print("2025 test accessed:  NO")
    print()

    for level in CONFIDENCE_LEVELS:
        key = str(int(round(level * 100)))
        e = empirical_metrics[key]
        g = gaussian_metrics[key]
        print(
            f"{key}% | "
            f"Empirical coverage={e['actual_coverage']:.6f}, "
            f"score={e['mean_interval_score']:.8f} | "
            f"Gaussian coverage={g['actual_coverage']:.6f}, "
            f"score={g['mean_interval_score']:.8f}"
        )

    print()
    print(f"Predictions: {output_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
