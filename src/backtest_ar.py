"""Leakage-controlled static AR(k) experiment for BTC/USD one-hour forecasts.

Two explicit stages:
    select  : fit candidates on development, score on validation, lock config.
    test    : load the locked model/config and evaluate exactly once on test.

No test observations are used by the selection stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_baseline import PERIODS, Period
from fit_ar_baseline import (
    ARModel,
    ARModelError,
    CANDIDATE_LAGS,
    VALID_INTERCEPT_OPTIONS,
    evaluate_forecasts,
    fit_ar_model,
    load_model_frame,
    predict_period,
)

DEFAULT_INPUT = Path("data/processed/btc_usd_hourly_model.csv")
DEFAULT_OUTPUT_DIR = Path("results/backtests/ar_static")
LOCK_FILE = "selected_model.json"
SELECTION_FILE = "selection.json"


def _period(name: str) -> Period:
    return next(p for p in PERIODS if p.name == name)


def _assert_period_contract(df: pd.DataFrame, period: Period, label: str) -> None:
    if df.empty:
        raise ARModelError(f"{label}: dataframe is empty.")
    if df["timestamp_utc"].min() >= period.end_exclusive:
        raise ARModelError(f"{label}: no rows before {period.end_exclusive}.")


def persistence_forecast(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    mask = (
        (df["timestamp_utc"] >= period.start)
        & (df["timestamp_utc"] < period.end_exclusive)
        & df["target_status"].eq("valid")
    )
    eligible = df.loc[mask].copy()
    if eligible.empty:
        raise ARModelError(f"No persistence observations in {period.name}.")
    return pd.DataFrame({
        "timestamp_utc": eligible["timestamp_utc"].to_numpy(),
        "forecast_origin_close": eligible["close"].to_numpy(float),
        "predicted_return_1h": 0.0,
        "predicted_close_1h": eligible["close"].to_numpy(float),
        "actual_return_1h": eligible["target_return_1h"].to_numpy(float),
        "actual_close_1h": eligible["target_close_1h"].to_numpy(float),
    })


def select_best_candidate(rows: list[dict]) -> dict:
    successful = [r for r in rows if r.get("status") == "ok"]
    if not successful:
        raise ARModelError("No candidate models were successfully evaluated.")
    return min(successful, key=lambda r: (
        r["validation"]["rmse_return"],
        r["validation"]["mae_close"],
        r["lag"],
        r["include_intercept"],
    ))


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def select(input_path: Path, output_dir: Path) -> None:
    """Fit on development, select on validation, and create the lock file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_model_frame(input_path)
    dev, val, test = _period("development"), _period("validation"), _period("test")

    # Hard process guard: selection is only allowed to consume origins strictly
    # before TEST starts. The source frame may contain test rows because labels
    # for the final validation origin legitimately live one hour later.
    selection_origins = df["timestamp_utc"] < test.start
    if not selection_origins.any():
        raise ARModelError("Selection frame contains no pre-test origins.")
    if df.loc[selection_origins, "timestamp_utc"].max() >= test.start:
        raise ARModelError("Selection guard failed: test-origin timestamp visible.")

    rows: list[dict] = []
    models: dict[tuple[int, bool], ARModel] = {}
    for lag in CANDIDATE_LAGS:
        for intercept in VALID_INTERCEPT_OPTIONS:
            label = f"AR({lag})_intercept_{str(intercept).lower()}"
            try:
                model, fit_metrics = fit_ar_model(
                    df, lag, include_intercept=intercept, period=dev
                )
                pred = predict_period(df, model, val)
                metrics = evaluate_forecasts(pred)
            except ARModelError as exc:
                # One malformed/degenerate candidate must not abort the
                # entire model-selection sweep. Record the failure loudly
                # and exclude that specification from selection.
                rows.append({
                    "label": label,
                    "lag": lag,
                    "include_intercept": intercept,
                    "status": "failed",
                    "error": str(exc),
                })
                print(f"SKIP {label}: {exc}")
                continue

            key = (lag, intercept)
            models[key] = model
            pred.to_csv(output_dir / f"{label}_validation_predictions.csv", index=False)
            rows.append({
                "label": label,
                "lag": lag,
                "include_intercept": intercept,
                "status": "ok",
                "fit": fit_metrics,
                "validation": metrics,
                "model": model.to_dict(),
            })

    selected = select_best_candidate(rows)
    model = models[(selected["lag"], selected["include_intercept"])]

    persistence = evaluate_forecasts(persistence_forecast(df, val))
    lock = {
        "lock_version": 1,
        "status": "LOCKED_FOR_TEST",
        "selection_rule": "validation return RMSE, then close MAE, then lag, then intercept=False",
        "development_period": dev.name,
        "validation_period": val.name,
        "test_period": test.name,
        "test_start": test.start.isoformat(),
        "selected_model": model.to_dict(),
        "selected_validation": selected["validation"],
        "validation_persistence": persistence,
        "test_used_for_selection": False,
        "test_refit_performed": False,
    }
    _write_json(output_dir / LOCK_FILE, lock)
    _write_json(output_dir / SELECTION_FILE, {
        "candidate_count": len(rows),
        "candidates": rows,
        "selected_label": selected["label"],
        "test_used_for_selection": False,
        "selection_max_origin_timestamp": df.loc[selection_origins, "timestamp_utc"].max().isoformat(),
    })

    print(f"LOCKED: {selected['label']}")
    print(f"Validation return RMSE: {selected['validation']['rmse_return']:.8f}")
    print(f"Validation close MAE:   {selected['validation']['mae_close']:.4f}")
    print(f"Lock file: {output_dir / LOCK_FILE}")


def _model_from_dict(d: dict) -> ARModel:
    required = {
        "lag", "include_intercept", "coefficients", "intercept", "n_train",
        "train_origin_first", "train_origin_last", "fit_slice_sha256",
        "condition_number", "rank",
    }
    missing = required.difference(d)
    if missing:
        raise ARModelError(f"Selected model is missing fields: {sorted(missing)}")
    model = ARModel(
        lag=int(d["lag"]),
        include_intercept=bool(d["include_intercept"]),
        coefficients=tuple(float(x) for x in d["coefficients"]),
        intercept=float(d["intercept"]),
        n_train=int(d["n_train"]),
        train_origin_first=str(d["train_origin_first"]),
        train_origin_last=str(d["train_origin_last"]),
        fit_slice_sha256=str(d["fit_slice_sha256"]),
        condition_number=float(d["condition_number"]),
        rank=int(d["rank"]),
    )
    if len(model.coefficients) != model.lag:
        raise ARModelError("Locked model coefficient count does not match lag.")
    return model


def evaluate_test(input_path: Path, output_dir: Path) -> None:
    """Evaluate only the already-locked development-fitted model on test."""
    lock_path = output_dir / LOCK_FILE
    if not lock_path.exists():
        raise ARModelError(
            f"No locked model found at {lock_path}. Run --stage select first."
        )
    with lock_path.open("r", encoding="utf-8") as f:
        lock = json.load(f)
    if lock.get("status") != "LOCKED_FOR_TEST":
        raise ARModelError("Selected model lock is not in LOCKED_FOR_TEST state.")
    if lock.get("test_used_for_selection") is not False:
        raise ARModelError("Lock metadata does not prove test isolation.")

    df = load_model_frame(input_path)
    test = _period("test")
    model = _model_from_dict(lock["selected_model"])

    pred = predict_period(df, model, test)
    metrics = evaluate_forecasts(pred)
    persistence = persistence_forecast(df, test)
    persistence_metrics = evaluate_forecasts(persistence)

    pred.to_csv(output_dir / "selected_test_predictions.csv", index=False)
    summary = {
        "method": "static_ar_on_hourly_log_returns",
        "selected_model": model.to_dict(),
        "test_period": test.name,
        "test_metrics": metrics,
        "test_persistence": persistence_metrics,
        "test_used_for_selection": False,
        "test_refit_performed": False,
        "ar_minus_persistence_rmse_return": metrics["rmse_return"] - persistence_metrics["rmse_return"],
        "ar_minus_persistence_mae_close": metrics["mae_close"] - persistence_metrics["mae_close"],
    }
    _write_json(output_dir / "test_summary.json", summary)

    print(f"TEST: AR({model.lag}), intercept={model.include_intercept}")
    print(f"AR test return RMSE: {metrics['rmse_return']:.8f}")
    print(f"AR test close MAE:   {metrics['mae_close']:.4f}")
    print(f"Persistence return RMSE: {persistence_metrics['rmse_return']:.8f}")
    print(f"Persistence close MAE:   {persistence_metrics['mae_close']:.4f}")
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
            evaluate_test(args.input, args.output_dir)
    except (ARModelError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
