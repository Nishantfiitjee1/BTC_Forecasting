"""Validation-only calibration analysis for Gaussian BTC intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
AR_FILE = (
    ROOT
    / "results"
    / "backtests"
    / "ar_static"
    / "AR(3)_intercept_true_validation_predictions.csv"
)
GARCH_FILE = (
    ROOT
    / "results"
    / "backtests"
    / "garch_static"
    / "validation_variance_predictions.csv"
)
OUT = ROOT / "results" / "backtests" / "gaussian_calibration"
REPORT_FILE = OUT / "validation_calibration.json"
PREDICTIONS_FILE = OUT / "validation_calibration_predictions.csv"

VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
VALIDATION_END = pd.Timestamp("2025-01-01T00:00:00Z")
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
CALIBRATION_BIN_EDGES = np.linspace(0.0, 1.0, 11)


def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{label} missing required columns: {sorted(missing)}"
        )


def _load_csv(path: Path, label: str, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} file is empty: {path}")

    _require_columns(df, required, label)

    timestamps = pd.to_datetime(
        df["timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    if timestamps.isna().any():
        raise ValueError(f"{label} contains invalid timestamp_utc values.")
    if timestamps.duplicated().any():
        count = int(timestamps.duplicated().sum())
        raise ValueError(
            f"{label} contains {count} duplicate timestamps."
        )
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            f"{label} timestamps are not monotonically increasing."
        )

    df["timestamp_utc"] = timestamps

    if (
        (df["timestamp_utc"] < VALIDATION_START)
        | (df["timestamp_utc"] >= VALIDATION_END)
    ).any():
        raise ValueError(
            f"{label} contains timestamps outside the 2024 validation period."
        )

    numeric_columns = sorted(required - {"timestamp_utc"})
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{label} contains non-finite values in {column}."
            )
        df[column] = values.to_numpy(dtype=float)

    return df


def _align_predictions(
    ar: pd.DataFrame,
    garch: pd.DataFrame,
) -> pd.DataFrame:
    ar_columns = [
        "timestamp_utc",
        "forecast_origin_close",
        "predicted_return_1h",
        "actual_return_1h",
        "actual_close_1h",
    ]
    garch_columns = [
        "timestamp_utc",
        "forecast_origin_close",
        "predicted_variance_1h",
        "predicted_volatility_1h",
        "actual_return_1h",
        "actual_close_1h",
    ]

    paired = ar[ar_columns].rename(
        columns={
            "forecast_origin_close": "ar_forecast_origin_close",
            "actual_return_1h": "ar_actual_return_1h",
            "actual_close_1h": "ar_actual_close_1h",
        }
    ).merge(
        garch[garch_columns].rename(
            columns={
                "forecast_origin_close": "garch_forecast_origin_close",
                "actual_return_1h": "garch_actual_return_1h",
                "actual_close_1h": "garch_actual_close_1h",
            }
        ),
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No common timestamps between AR and GARCH validation files.")

    paired = paired.sort_values("timestamp_utc").reset_index(drop=True)

    agreement_columns = (
        (
            "ar_forecast_origin_close",
            "garch_forecast_origin_close",
            "forecast origin close",
        ),
        (
            "ar_actual_return_1h",
            "garch_actual_return_1h",
            "actual return",
        ),
        (
            "ar_actual_close_1h",
            "garch_actual_close_1h",
            "actual close",
        ),
    )
    for left, right, label in agreement_columns:
        if not np.allclose(
            paired[left].to_numpy(dtype=float),
            paired[right].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            difference = np.abs(
                paired[left].to_numpy(dtype=float)
                - paired[right].to_numpy(dtype=float)
            )
            raise ValueError(
                f"AR/GARCH {label} mismatch; "
                f"max_abs_diff={float(difference.max()):.12g}"
            )

    paired["forecast_origin_close"] = paired[
        "ar_forecast_origin_close"
    ]
    paired["actual_return_1h"] = paired[
        "ar_actual_return_1h"
    ]
    paired["actual_close_1h"] = paired[
        "ar_actual_close_1h"
    ]

    required_output = [
        "timestamp_utc",
        "forecast_origin_close",
        "predicted_return_1h",
        "predicted_variance_1h",
        "predicted_volatility_1h",
        "actual_return_1h",
        "actual_close_1h",
    ]
    return paired[
        required_output
        + [
            "garch_forecast_origin_close",
            "garch_actual_return_1h",
            "garch_actual_close_1h",
        ]
    ].copy()


def _validate_forecast_inputs(paired: pd.DataFrame) -> None:
    variance = paired["predicted_variance_1h"].to_numpy(dtype=float)
    volatility = paired["predicted_volatility_1h"].to_numpy(dtype=float)

    if (variance <= 0).any():
        raise ValueError("Predicted variance must be strictly positive.")
    if (volatility <= 0).any():
        raise ValueError("Predicted volatility must be strictly positive.")
    if not np.allclose(
        volatility,
        np.sqrt(variance),
        rtol=1e-10,
        atol=1e-14,
    ):
        difference = np.abs(volatility - np.sqrt(variance))
        raise ValueError(
            "Predicted volatility is inconsistent with predicted variance; "
            f"max_abs_diff={float(difference.max()):.12g}"
        )


def _calibration_bins(
    probability: np.ndarray,
    observed_positive: np.ndarray,
) -> list[dict]:
    bin_indices = np.minimum(
        np.digitize(probability, CALIBRATION_BIN_EDGES[1:-1], right=False),
        len(CALIBRATION_BIN_EDGES) - 2,
    )
    results = []

    for index in range(len(CALIBRATION_BIN_EDGES) - 1):
        mask = bin_indices == index
        count = int(mask.sum())
        results.append(
            {
                "bin_lower": float(CALIBRATION_BIN_EDGES[index]),
                "bin_upper": float(CALIBRATION_BIN_EDGES[index + 1]),
                "count": count,
                "mean_predicted_probability": (
                    float(np.mean(probability[mask])) if count else None
                ),
                "empirical_positive_rate": (
                    float(np.mean(observed_positive[mask])) if count else None
                ),
            }
        )

    return results


def run() -> dict:
    ar = _load_csv(
        AR_FILE,
        "AR(3) validation predictions",
        {
            "timestamp_utc",
            "forecast_origin_close",
            "predicted_return_1h",
            "actual_return_1h",
            "actual_close_1h",
        },
    )
    garch = _load_csv(
        GARCH_FILE,
        "Normal-GARCH validation predictions",
        {
            "timestamp_utc",
            "forecast_origin_close",
            "predicted_variance_1h",
            "predicted_volatility_1h",
            "actual_return_1h",
            "actual_close_1h",
        },
    )

    paired = _align_predictions(ar, garch)
    _validate_forecast_inputs(paired)

    predicted_return = paired["predicted_return_1h"].to_numpy(dtype=float)
    predicted_variance = paired["predicted_variance_1h"].to_numpy(dtype=float)
    predicted_volatility = paired[
        "predicted_volatility_1h"
    ].to_numpy(dtype=float)
    actual_return = paired["actual_return_1h"].to_numpy(dtype=float)

    standardized_residual = (
        actual_return - predicted_return
    ) / predicted_volatility
    if not np.isfinite(standardized_residual).all():
        raise ValueError("Standardized residual contains non-finite values.")

    paired["standardized_residual"] = standardized_residual

    interval_results = {}
    for coverage in COVERAGE_LEVELS:
        critical_value = float(norm.ppf((1.0 + coverage) / 2.0))
        lower_return = predicted_return - critical_value * predicted_volatility
        upper_return = predicted_return + critical_value * predicted_volatility
        lower_close = paired["forecast_origin_close"].to_numpy(dtype=float) * np.exp(
            lower_return
        )
        upper_close = paired["forecast_origin_close"].to_numpy(dtype=float) * np.exp(
            upper_return
        )
        covered = (
            (actual_return >= lower_return)
            & (actual_return <= upper_return)
        )
        return_width = upper_return - lower_return
        close_width = upper_close - lower_close
        label = f"{int(coverage * 100)}"

        paired[f"gaussian_{label}_lower_return"] = lower_return
        paired[f"gaussian_{label}_upper_return"] = upper_return
        paired[f"gaussian_{label}_lower_close"] = lower_close
        paired[f"gaussian_{label}_upper_close"] = upper_close

        interval_results[label] = {
            "nominal_coverage": coverage,
            "critical_value": critical_value,
            "empirical_coverage": float(np.mean(covered)),
            "covered_count": int(np.sum(covered)),
            "average_return_interval_width": float(np.mean(return_width)),
            "average_close_interval_width": float(np.mean(close_width)),
        }

    directional_probability = norm.cdf(
        predicted_return / predicted_volatility
    )
    observed_positive = (actual_return > 0.0).astype(float)
    brier_score = float(
        np.mean((directional_probability - observed_positive) ** 2)
    )
    paired["directional_probability_positive"] = directional_probability
    paired["actual_positive_return"] = observed_positive.astype(int)

    report = {
        "validation_period": {
            "start": VALIDATION_START.isoformat(),
            "end_exclusive": VALIDATION_END.isoformat(),
        },
        "input_files": {
            "ar3": str(AR_FILE),
            "normal_garch": str(GARCH_FILE),
        },
        "population": {
            "ar3_rows": int(len(ar)),
            "normal_garch_rows": int(len(garch)),
            "common_rows": int(len(paired)),
            "ar3_only_rows": int(len(ar) - len(paired)),
            "normal_garch_only_rows": int(len(garch) - len(paired)),
            "first_timestamp": paired["timestamp_utc"].min().isoformat(),
            "last_timestamp": paired["timestamp_utc"].max().isoformat(),
        },
        "integrity": {
            "timestamps_unique_and_sorted": True,
            "timestamps_utc": True,
            "ar_garch_fields_agree": True,
            "variance_positive_and_finite": True,
            "volatility_positive_and_finite": True,
            "volatility_equals_sqrt_variance": True,
            "actuals_used_only_for_evaluation": True,
        },
        "formulas": {
            "standardized_residual": (
                "(actual_return_1h - predicted_return_1h) "
                "/ predicted_volatility_1h"
            ),
            "interval": "predicted_return_1h +/- z * predicted_volatility_1h",
            "directional_probability": (
                "NormalCDF(predicted_return_1h / predicted_volatility_1h)"
            ),
            "brier_score": "mean((probability - I(actual_return_1h > 0))^2)",
        },
        "standardized_residual": {
            "mean": float(np.mean(standardized_residual)),
            "standard_deviation": float(np.std(standardized_residual)),
            "quantiles": {
                "0.01": float(np.quantile(standardized_residual, 0.01)),
                "0.05": float(np.quantile(standardized_residual, 0.05)),
                "0.10": float(np.quantile(standardized_residual, 0.10)),
                "0.25": float(np.quantile(standardized_residual, 0.25)),
                "0.50": float(np.quantile(standardized_residual, 0.50)),
                "0.75": float(np.quantile(standardized_residual, 0.75)),
                "0.90": float(np.quantile(standardized_residual, 0.90)),
                "0.95": float(np.quantile(standardized_residual, 0.95)),
                "0.99": float(np.quantile(standardized_residual, 0.99)),
            },
        },
        "gaussian_intervals": interval_results,
        "directional_probability": {
            "event": "actual_return_1h > 0",
            "brier_score": brier_score,
            "positive_return_count": int(np.sum(observed_positive)),
            "positive_return_rate": float(np.mean(observed_positive)),
            "calibration_bins": _calibration_bins(
                directional_probability,
                observed_positive,
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paired.to_csv(
        PREDICTIONS_FILE,
        index=False,
        float_format="%.17g",
    )

    print(f"Validation observations: {len(paired)}")
    for label, values in interval_results.items():
        print(
            f"{label}% coverage: "
            f"{values['empirical_coverage']:.8f}; "
            f"average return width: "
            f"{values['average_return_interval_width']:.12g}"
        )
    print(
        "Standardized residual mean/std: "
        f"{np.mean(standardized_residual):.12g} / "
        f"{np.std(standardized_residual):.12g}"
    )
    print(f"Directional Brier score: {brier_score:.12g}")
    print("Directional calibration bins:")
    for calibration_bin in report["directional_probability"]["calibration_bins"]:
        print(json.dumps(calibration_bin, allow_nan=False))
    print(f"Report: {REPORT_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")

    return report


if __name__ == "__main__":
    run()
