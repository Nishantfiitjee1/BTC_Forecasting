"""Strict paired statistical comparison of locked GARCH vs locked rolling variance.

This comparison uses only the already-generated 2025 test prediction files.
It does not refit either model or perform model selection.

The comparison population is the exact intersection of timestamps present in
both prediction files. It validates:
- unique, UTC timestamps
- chronological ordering
- finite positive variance forecasts
- finite realized squared returns
- agreement of the realized target across both files
- exact one-to-one timestamp pairing

Losses:
- QLIKE: y / variance + log(variance)
- squared-error loss on realized squared return: (y - variance)^2

For both loss families, positive (baseline - GARCH) means GARCH has lower loss.
Statistical uncertainty is assessed with:
- Bartlett/Newey-West HAC DM-style statistic, bandwidth=24 hours
- moving-block bootstrap 95% CI, block length=24 hours, 2000 resamples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
GARCH_FILE = ROOT / "results" / "backtests" / "garch_static" / "selected_test_variance_predictions.csv"
ROLLING_FILE = ROOT / "results" / "backtests" / "volatility_baseline" / "selected_test_variance_predictions.csv"
OUT = ROOT / "results" / "backtests" / "volatility_statistical_comparison"

BANDWIDTH = 24
BLOCK_LENGTH = 24
N_BOOT = 2000
SEED = 20260924


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _load_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} prediction file: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    if df.empty:
        raise ValueError(f"{label} prediction file is empty: {path}")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        raise ValueError(f"{label} timestamp_utc could not be parsed as datetime.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    if df["timestamp_utc"].isna().any():
        raise ValueError(f"{label} contains invalid timestamp_utc values.")
    if df["timestamp_utc"].duplicated().any():
        dupes = int(df["timestamp_utc"].duplicated().sum())
        raise ValueError(f"{label} contains {dupes} duplicate timestamps.")
    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are not monotonically increasing.")

    return df


def _validate_common_pair(garch: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        garch,
        {"timestamp_utc", "predicted_variance_1h", "actual_return_1h"},
        "GARCH",
    )
    _require_columns(
        rolling,
        {"timestamp_utc", "variance_forecast", "realized_squared_return"},
        "rolling",
    )

    # Build the realized squared-return target independently from the GARCH file.
    garch = garch.copy()
    garch["realized_squared_return_from_garch"] = (
        pd.to_numeric(garch["actual_return_1h"], errors="coerce") ** 2
    )

    merged = garch[
        ["timestamp_utc", "predicted_variance_1h", "realized_squared_return_from_garch"]
    ].merge(
        rolling[
            ["timestamp_utc", "variance_forecast", "realized_squared_return"]
        ],
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No common timestamps between GARCH and rolling predictions.")

    # Both files should describe the same realized target. A tiny tolerance is
    # allowed because CSV serialization can round floating-point values.
    y_garch = merged["realized_squared_return_from_garch"].to_numpy(float)
    y_rolling = pd.to_numeric(
        merged["realized_squared_return"], errors="coerce"
    ).to_numpy(float)

    if not np.isfinite(y_garch).all() or not np.isfinite(y_rolling).all():
        raise ValueError("Non-finite realized squared returns found on common timestamps.")

    if not np.allclose(y_garch, y_rolling, rtol=1e-10, atol=1e-14):
        diff = np.abs(y_garch - y_rolling)
        raise ValueError(
            "Realized target mismatch between GARCH and rolling files. "
            f"max_abs_diff={float(diff.max()):.12g}"
        )

    merged["realized_squared_return"] = y_garch
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)

    garch_v = pd.to_numeric(merged["predicted_variance_1h"], errors="coerce").to_numpy(float)
    rolling_v = pd.to_numeric(merged["variance_forecast"], errors="coerce").to_numpy(float)

    if not np.isfinite(garch_v).all() or not np.isfinite(rolling_v).all():
        raise ValueError("Non-finite variance forecasts found.")
    if (garch_v <= 0).any():
        raise ValueError("GARCH contains non-positive variance forecasts.")
    if (rolling_v <= 0).any():
        raise ValueError("Rolling baseline contains non-positive variance forecasts.")

    return merged[
        ["timestamp_utc", "realized_squared_return",
         "predicted_variance_1h", "variance_forecast"]
    ].copy()


def _dm_statistic(loss_diff: np.ndarray, bandwidth: int = BANDWIDTH) -> tuple[float, float]:
    d = np.asarray(loss_diff, dtype=float)
    if d.ndim != 1 or len(d) < 10:
        raise ValueError("Too few observations for DM test.")
    if not np.isfinite(d).all():
        raise ValueError("DM loss difference contains non-finite values.")

    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.mean(centered * centered))
    long_run = gamma0

    bw = min(int(bandwidth), len(d) - 1)
    for k in range(1, bw + 1):
        cov = float(np.mean(centered[k:] * centered[:-k]))
        weight = 1.0 - k / (bw + 1.0)
        long_run += 2.0 * weight * cov

    if not np.isfinite(long_run) or long_run <= 0:
        raise ValueError(f"Non-positive/non-finite HAC long-run variance: {long_run}")

    stat = mean_d / np.sqrt(long_run / len(d))
    p_value = float(2.0 * norm.sf(abs(stat)))
    return float(stat), p_value


def _moving_block_bootstrap_ci(
    loss_diff: np.ndarray,
    block_length: int = BLOCK_LENGTH,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    d = np.asarray(loss_diff, dtype=float)
    n = len(d)

    if n < block_length:
        raise ValueError("Series shorter than bootstrap block.")
    if not np.isfinite(d).all():
        raise ValueError("Bootstrap loss difference contains non-finite values.")

    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_length + 1)
    blocks_needed = int(np.ceil(n / block_length))
    means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        pieces = []
        for _ in range(blocks_needed):
            start = int(rng.choice(starts))
            pieces.append(d[start : start + block_length])
        means[b] = float(np.mean(np.concatenate(pieces)[:n]))

    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _qlike_loss(realized_variance: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    # Constant terms that depend only on realized_variance are omitted because
    # we compare paired forecast losses; the model ranking and loss difference
    # are unchanged.
    return realized_variance / forecast_variance + np.log(forecast_variance)


def _mse_loss(realized_variance: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    return (realized_variance - forecast_variance) ** 2


def run() -> dict:
    garch = _load_predictions(GARCH_FILE, "GARCH")
    rolling = _load_predictions(ROLLING_FILE, "rolling")

    paired = _validate_common_pair(garch, rolling)

    y = paired["realized_squared_return"].to_numpy(float)
    vg = paired["predicted_variance_1h"].to_numpy(float)
    vr = paired["variance_forecast"].to_numpy(float)

    q_garch = _qlike_loss(y, vg)
    q_roll = _qlike_loss(y, vr)
    q_diff = q_roll - q_garch  # positive => GARCH lower QLIKE

    m_garch = _mse_loss(y, vg)
    m_roll = _mse_loss(y, vr)
    m_diff = m_roll - m_garch  # positive => GARCH lower MSE

    q_dm, q_p = _dm_statistic(q_diff)
    q_ci = _moving_block_bootstrap_ci(q_diff)

    m_dm, m_p = _dm_statistic(m_diff)
    m_ci = _moving_block_bootstrap_ci(m_diff, seed=SEED + 1)

    result = {
        "test_period": {
            "start": "2025-01-01T00:00:00+00:00",
            "end_exclusive": "2026-01-01T00:00:00+00:00",
        },
        "population": {
            "garch_rows": int(len(garch)),
            "rolling_rows": int(len(rolling)),
            "common_rows": int(len(paired)),
            "garch_only_rows": int(len(garch) - len(paired)),
            "rolling_only_rows": int(len(rolling) - len(paired)),
            "first_common_timestamp": paired["timestamp_utc"].min().isoformat(),
            "last_common_timestamp": paired["timestamp_utc"].max().isoformat(),
        },
        "method": {
            "comparison": "paired common timestamps only",
            "dm_bandwidth_hours": BANDWIDTH,
            "bootstrap": "moving_block",
            "bootstrap_block_length_hours": BLOCK_LENGTH,
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": SEED,
        },
        "qlike": {
            "garch": float(np.mean(q_garch)),
            "rolling": float(np.mean(q_roll)),
            "garch_minus_rolling": float(np.mean(q_garch) - np.mean(q_roll)),
            "rolling_minus_garch_loss_difference": float(np.mean(q_diff)),
            "dm_statistic": q_dm,
            "p_value": q_p,
            "moving_block_bootstrap_95ci": list(q_ci),
            "positive_loss_difference_means": "GARCH has lower QLIKE",
        },
        "squared_return_mse": {
            "garch": float(np.mean(m_garch)),
            "rolling": float(np.mean(m_roll)),
            "garch_minus_rolling": float(np.mean(m_garch) - np.mean(m_roll)),
            "rolling_minus_garch_loss_difference": float(np.mean(m_diff)),
            "dm_statistic": m_dm,
            "p_value": m_p,
            "moving_block_bootstrap_95ci": list(m_ci),
            "positive_loss_difference_means": "GARCH has lower squared-return MSE",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_statistical_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    paired.to_csv(
        OUT / "paired_test_predictions.csv",
        index=False,
        float_format="%.17g",
    )

    print(f"GARCH predictions: {len(garch)}")
    print(f"Rolling predictions: {len(rolling)}")
    print(f"Common predictions: {len(paired)}")
    print(f"GARCH-only: {len(garch) - len(paired)}")
    print(f"Rolling-only: {len(rolling) - len(paired)}")
    print()
    print(f"GARCH QLIKE: {np.mean(q_garch):.8f}")
    print(f"Rolling QLIKE: {np.mean(q_roll):.8f}")
    print(f"QLIKE DM statistic: {q_dm:.6f}")
    print(f"QLIKE DM p-value: {q_p:.6g}")
    print(f"QLIKE loss-difference 95% block-bootstrap CI: {q_ci}")
    print()
    print(f"GARCH MSE(squared return): {np.mean(m_garch):.12g}")
    print(f"Rolling MSE(squared return): {np.mean(m_roll):.12g}")
    print(f"MSE DM statistic: {m_dm:.6f}")
    print(f"MSE DM p-value: {m_p:.6g}")
    print(f"MSE loss-difference 95% block-bootstrap CI: {m_ci}")
    print(f"Results: {OUT}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
