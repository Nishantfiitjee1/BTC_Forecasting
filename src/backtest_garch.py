"""Experiment 2a: static GARCH(1,1) volatility forecast for BTC/USD.

Stages:
  select: fit on 2021-2023, evaluate variance forecast on 2024, lock model.
  test: evaluate the frozen model on 2025 exactly once.

This experiment forecasts conditional variance for uncertainty/range construction;
it does not claim that volatility alone improves the point-price forecast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_baseline import PERIODS, Period
from garch_volatility import (
    GARCH11Model,
    GARCHModelError,
    evaluate_variance_forecasts,
    fit_garch11,
    forecast_one_step_variance,
    load_garch_frame,
)

DEFAULT_INPUT = Path("data/processed/btc_usd_hourly_model.csv")
DEFAULT_OUTPUT_DIR = Path("results/backtests/garch_static")
LOCK_FILE = "selected_model.json"


def _period(name: str) -> Period:
    return next(p for p in PERIODS if p.name == name)


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def _model_from_dict(d: dict) -> GARCH11Model:
    fields = {
        "mu", "omega", "alpha", "beta", "unconditional_variance", "n_train",
        "train_origin_first", "train_origin_last", "fit_slice_sha256",
        "loglikelihood", "converged", "optimizer_iterations",
    }
    missing = fields.difference(d)
    if missing:
        raise GARCHModelError(f"Locked GARCH model missing fields: {sorted(missing)}")
    return GARCH11Model(**{k: d[k] for k in fields})


def select(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_garch_frame(input_path)
    dev = _period("development")
    val = _period("validation")
    test = _period("test")
    if df["timestamp_utc"].max() < test.start:
        raise GARCHModelError("Input frame has no final-test history; refusing to lock model.")

    model, fit_metrics = fit_garch11(df, dev)
    val_pred = forecast_one_step_variance(df, model, val)
    val_metrics = evaluate_variance_forecasts(val_pred)
    val_pred.to_csv(output_dir / "validation_variance_predictions.csv", index=False)

    lock = {
        "lock_version": 1,
        "status": "LOCKED_FOR_TEST",
        "model_family": "GARCH(1,1)_normal",
        "development_period": dev.name,
        "validation_period": val.name,
        "test_period": test.name,
        "selected_model": model.to_dict(),
        "development_fit": fit_metrics,
        "validation_metrics": val_metrics,
        "test_used_for_selection": False,
        "test_refit_performed": False,
        "selection_rule": "single GARCH(1,1) specification; validation used only for documented pre-test evaluation",
    }
    _write_json(output_dir / LOCK_FILE, lock)
    print("LOCKED: GARCH(1,1)_normal")
    print(f"Validation QLIKE: {val_metrics['qlike']:.8f}")
    print(f"Validation MSE(squared return): {val_metrics['mse_squared_return']:.10g}")
    print(f"alpha+beta: {model.alpha + model.beta:.6f}")
    print(f"Lock file: {output_dir / LOCK_FILE}")


def test(input_path: Path, output_dir: Path) -> None:
    lock_path = output_dir / LOCK_FILE
    if not lock_path.exists():
        raise GARCHModelError(f"No locked model found at {lock_path}. Run --stage select first.")
    with lock_path.open("r", encoding="utf-8") as f:
        lock = json.load(f)
    if lock.get("status") != "LOCKED_FOR_TEST" or lock.get("test_used_for_selection") is not False:
        raise GARCHModelError("GARCH lock does not prove test isolation.")
    df = load_garch_frame(input_path)
    model = _model_from_dict(lock["selected_model"])
    test_period = _period("test")
    pred = forecast_one_step_variance(df, model, test_period)
    metrics = evaluate_variance_forecasts(pred)
    pred.to_csv(output_dir / "selected_test_variance_predictions.csv", index=False)
    summary = {
        "model_family": "GARCH(1,1)_normal",
        "test_period": test_period.name,
        "test_metrics": metrics,
        "test_used_for_selection": False,
        "test_refit_performed": False,
    }
    _write_json(output_dir / "test_summary.json", summary)
    print("TEST: GARCH(1,1)_normal")
    print(f"Test QLIKE: {metrics['qlike']:.8f}")
    print(f"Test MSE(squared return): {metrics['mse_squared_return']:.10g}")
    print(f"Results: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "test"), required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        if args.stage == "select":
            select(args.input, args.output_dir)
        else:
            test(args.input, args.output_dir)
    except (GARCHModelError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
