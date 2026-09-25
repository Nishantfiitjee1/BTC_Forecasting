from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from volatility_baseline import (
    WINDOWS,
    compute_gap_aware_returns,
    rolling_variance_forecast,
    qlike,
    mse_squared_return,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "btc_usd_hourly_model.csv"
OUT = ROOT / "results" / "backtests" / "volatility_baseline"
DEV_START = pd.Timestamp("2021-01-01", tz="UTC")
VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing model dataset: {DATA}")
    df = pd.read_csv(DATA, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return compute_gap_aware_returns(df)


def prepare_predictions(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["variance_forecast"] = rolling_variance_forecast(
        out["log_return_1h"], out["timestamp_utc"], window
    )
    # At origin t, target_return_1h is already aligned to t+1 by the
    # model-data preparation stage.
    out["realized_squared_return"] = pd.to_numeric(
        out["target_return_1h"], errors="coerce"
    ) ** 2
    valid = (
        out["timestamp_utc"].ge(DEV_START)
        & out["timestamp_utc"].lt(END)
        & out["target_status"].eq("valid")
        & out["variance_forecast"].notna()
        & np.isfinite(out["realized_squared_return"])
    )
    return out.loc[valid, ["timestamp_utc", "variance_forecast", "realized_squared_return"]].copy()


def evaluate(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    x = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < end)]
    if x.empty:
        raise ValueError(f"No predictions in {start} to {end}.")
    y = x["realized_squared_return"].to_numpy(float)
    v = x["variance_forecast"].to_numpy(float)
    return {
        "n": int(len(x)),
        "qlike": qlike(y, v),
        "mse_squared_return": mse_squared_return(y, v),
    }


def run_selection() -> dict:
    df = load_data()
    rows = []
    for window in WINDOWS:
        pred = prepare_predictions(df, window)
        metrics = evaluate(pred, VAL_START, TEST_START)
        metrics["window"] = window
        rows.append(metrics)

    best = min(rows, key=lambda x: x["qlike"])
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_rule": "minimum validation QLIKE",
        "candidate_windows_hours": list(WINDOWS),
        "locked_window": int(best["window"]),
        "validation": best,
        "all_candidates": rows,
    }
    (OUT / "selected_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"LOCKED: rolling_variance_{best['window']}h")
    print(f"Validation QLIKE: {best['qlike']:.8f}")
    print(f"Validation MSE(squared return): {best['mse_squared_return']:.12g}")
    print(f"Lock file: {OUT / 'selected_model.json'}")
    return payload


def run_test() -> dict:
    lock_path = OUT / "selected_model.json"
    if not lock_path.exists():
        raise FileNotFoundError("Run --stage select first; no locked baseline exists.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    window = int(lock["locked_window"])

    df = load_data()
    pred = prepare_predictions(df, window)
    metrics = evaluate(pred, TEST_START, END)

    test_pred = pred[
        (pred["timestamp_utc"] >= TEST_START)
        & (pred["timestamp_utc"] < END)
    ].copy()

    test_pred.to_csv(
        OUT / "selected_test_variance_predictions.csv",
        index=False,
        float_format="%.17g",
    )

    payload = {
        "model": f"rolling_variance_{window}h",
        "locked_window": window,
        "test": metrics,
    }
    (OUT / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"TEST: rolling_variance_{window}h")
    print(f"Test QLIKE: {metrics['qlike']:.8f}")
    print(f"Test MSE(squared return): {metrics['mse_squared_return']:.12g}")
    print(f"Results: {OUT}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["select", "test"], required=True)
    args = p.parse_args()
    if args.stage == "select":
        run_selection()
    else:
        run_test()


if __name__ == "__main__":
    main()
