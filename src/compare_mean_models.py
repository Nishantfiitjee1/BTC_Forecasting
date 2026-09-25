"""Locked 2025 statistical comparison: frozen AR(k) vs persistence.

This script performs NO fitting and NO model selection. It loads the
already-locked AR model produced by backtest_ar.py's `select` stage and
generates predictions through the canonical, tested prediction path
(fit_ar_baseline.predict_period). Persistence predictions are derived
directly from the AR prediction frame so both series are guaranteed to
cover exactly the same forecast origins/targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_ar import _model_from_dict
from backtest_baseline import PERIODS
from fit_ar_baseline import ARModelError, load_model_frame, predict_period

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "btc_usd_hourly_model.csv"
LOCK_PATH = ROOT / "results" / "backtests" / "ar_static" / "selected_model.json"
OUT = ROOT / "results" / "backtests" / "mean_statistical_comparison"


class ComparisonError(RuntimeError):
    """Raised when the locked model or its predictions fail an integrity check."""


def _test_period():
    return next(p for p in PERIODS if p.name == "test")


def load_locked_model(lock_path: Path):
    """Load and validate the frozen AR model. Fails loudly on any problem."""
    if not lock_path.exists():
        raise ComparisonError(f"No locked model found at {lock_path}.")

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Locked model file is unreadable/corrupted: {exc}") from exc

    if lock.get("status") != "LOCKED_FOR_TEST":
        raise ComparisonError(
            f"Refusing to use model with status={lock.get('status')!r}; "
            "expected 'LOCKED_FOR_TEST'."
        )
    if lock.get("test_used_for_selection") is not False:
        raise ComparisonError("Lock does not prove test isolation (test_used_for_selection).")
    if lock.get("test_refit_performed") is not False:
        raise ComparisonError("Lock does not prove no refit occurred (test_refit_performed).")
    if "selected_model" not in lock:
        raise ComparisonError("Lock file missing 'selected_model'.")

    try:
        model = _model_from_dict(lock["selected_model"])
    except ARModelError as exc:
        raise ComparisonError(f"Locked model payload is invalid: {exc}") from exc

    return model, lock


def build_predictions(input_path: Path, lock_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model, lock = load_locked_model(lock_path)

    df = load_model_frame(input_path)
    test = _test_period()

    ar_pred = predict_period(df, model, test)  # canonical, no refit
    if ar_pred.empty:
        raise ComparisonError("AR produced zero test predictions.")

    ts = pd.to_datetime(ar_pred["timestamp_utc"])
    if ts.duplicated().any():
        raise ComparisonError("AR prediction timestamps contain duplicates.")
    if not ts.is_monotonic_increasing:
        raise ComparisonError("AR prediction timestamps are not chronological.")

    # Persistence, built FROM the AR frame so timestamps/targets match exactly.
    persistence_pred = pd.DataFrame({
        "timestamp_utc": ar_pred["timestamp_utc"],
        "forecast_origin_close": ar_pred["forecast_origin_close"],
        "predicted_return_1h": 0.0,
        "predicted_close_1h": ar_pred["forecast_origin_close"],
        "actual_return_1h": ar_pred["actual_return_1h"],
        "actual_close_1h": ar_pred["actual_close_1h"],
    })

    if len(ar_pred) != len(persistence_pred):
        raise ComparisonError("AR/persistence prediction counts differ.")
    if not (ar_pred["timestamp_utc"].to_numpy() == persistence_pred["timestamp_utc"].to_numpy()).all():
        raise ComparisonError("AR/persistence timestamps do not match exactly.")

    return ar_pred, persistence_pred, lock


def dm_statistic(loss_diff: np.ndarray, bandwidth: int = 24) -> tuple[float, float]:
    """Two-sided DM-style z statistic with Bartlett/Newey-West HAC variance."""
    d = np.asarray(loss_diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        raise ComparisonError("Too few observations for DM test.")

    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.mean(centered * centered))
    long_run = gamma0

    bw = min(int(bandwidth), n - 1)
    for k in range(1, bw + 1):
        cov = float(np.mean(centered[k:] * centered[:-k]))
        weight = 1.0 - k / (bw + 1.0)
        long_run += 2.0 * weight * cov

    if long_run <= 0:
        raise ComparisonError("Non-positive HAC long-run variance.")

    stat = mean_d / np.sqrt(long_run / n)

    from scipy.stats import norm
    p = float(2.0 * norm.sf(abs(stat)))
    return float(stat), p


def moving_block_bootstrap_ci(
    loss_diff: np.ndarray,
    block_length: int = 24,
    n_boot: int = 2000,
    seed: int = 20260923,
) -> tuple[float, float]:
    d = np.asarray(loss_diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < block_length:
        raise ComparisonError("Series shorter than bootstrap block.")

    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_length + 1)
    means = np.empty(n_boot)

    blocks_needed = int(np.ceil(n / block_length))
    for b in range(n_boot):
        pieces = []
        for _ in range(blocks_needed):
            s = int(rng.choice(starts))
            pieces.append(d[s:s + block_length])
        sample = np.concatenate(pieces)[:n]
        means[b] = np.mean(sample)

    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def run(input_path: Path = DEFAULT_INPUT, lock_path: Path = LOCK_PATH) -> None:
    ar_pred, persistence_pred, lock = build_predictions(input_path, lock_path)
    n = len(ar_pred)

    ar_sq = (ar_pred["actual_return_1h"] - ar_pred["predicted_return_1h"]) ** 2
    p_sq = (persistence_pred["actual_return_1h"] - persistence_pred["predicted_return_1h"]) ** 2
    diff_return = p_sq.to_numpy() - ar_sq.to_numpy()  # positive => AR better
    if not np.isfinite(diff_return).all():
        raise ComparisonError("Non-finite values in return loss differential.")

    dm_r, dm_p_r = dm_statistic(diff_return, bandwidth=24)
    ci_r = moving_block_bootstrap_ci(diff_return, block_length=24)

    ar_abs_close = np.abs(ar_pred["actual_close_1h"] - ar_pred["predicted_close_1h"])
    p_abs_close = np.abs(persistence_pred["actual_close_1h"] - persistence_pred["predicted_close_1h"])
    diff_close = p_abs_close.to_numpy() - ar_abs_close.to_numpy()  # positive => AR better
    if not np.isfinite(diff_close).all():
        raise ComparisonError("Non-finite values in close loss differential.")

    dm_c, dm_p_c = dm_statistic(diff_close, bandwidth=24)
    ci_c = moving_block_bootstrap_ci(diff_close, block_length=24)

    model_meta = lock["selected_model"]
    result = {
        "test_period": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        "n_predictions": int(n),
        "model": (
            f"locked AR({model_meta['lag']}), intercept={model_meta['include_intercept']} "
            f"(loaded from {LOCK_PATH.relative_to(ROOT)}, no refit)"
        ),
        "coefficients": {
            "intercept": float(model_meta["intercept"]),
            **{f"lag{i+1}": float(c) for i, c in enumerate(model_meta["coefficients"])},
        },
        "fit_slice_sha256": model_meta["fit_slice_sha256"],
        "return_squared_error": {
            "ar": float(np.mean(ar_sq)),
            "persistence": float(np.mean(p_sq)),
            "loss_difference_persistence_minus_ar": float(np.mean(diff_return)),
            "dm_stat": dm_r,
            "p_value": dm_p_r,
            "moving_block_bootstrap_95ci": list(ci_r),
            "interpretation_of_positive_difference": "AR has lower squared-return loss",
        },
        "close_absolute_error": {
            "ar": float(np.mean(ar_abs_close)),
            "persistence": float(np.mean(p_abs_close)),
            "loss_difference_persistence_minus_ar": float(np.mean(diff_close)),
            "dm_stat": dm_c,
            "p_value": dm_p_c,
            "moving_block_bootstrap_95ci": list(ci_c),
            "interpretation_of_positive_difference": "AR has lower absolute close error",
        },
        "integrity": {
            "test_used_for_selection": False,
            "test_refit_performed": False,
            "ar_persistence_timestamps_identical": True,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_statistical_comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    ar_pred.to_csv(OUT / "test_predictions_ar.csv", index=False)
    persistence_pred.to_csv(OUT / "test_predictions_persistence.csv", index=False)

    print(f"Loaded locked model: AR({model_meta['lag']}), intercept={model_meta['include_intercept']}")
    print(f"TEST predictions: {n}")
    print(f"AR return MSE: {result['return_squared_error']['ar']:.12g}")
    print(f"Persistence return MSE: {result['return_squared_error']['persistence']:.12g}")
    print(f"Return DM statistic: {dm_r:.6f}")
    print(f"Return DM p-value: {dm_p_r:.6g}")
    print(f"Return loss-difference 95% block-bootstrap CI: {ci_r}")
    print(f"AR close MAE: {result['close_absolute_error']['ar']:.6f}")
    print(f"Persistence close MAE: {result['close_absolute_error']['persistence']:.6f}")
    print(f"Close DM statistic: {dm_c:.6f}")
    print(f"Close DM p-value: {dm_p_c:.6g}")
    print(f"Close loss-difference 95% block-bootstrap CI: {ci_c}")
    print(f"Results: {OUT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lock-path", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    try:
        run(args.input, args.lock_path)
    except (ComparisonError, ARModelError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())